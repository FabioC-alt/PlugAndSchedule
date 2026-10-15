"""B4 training loop (arXiv:2405.05019 Sec. V): TD3 agent processing
dynamically-arriving TT flows one at a time, with an ILP-fallback reset
whenever any link's GCL length exceeds omega -- one "epoch" (the paper's
x-axis in Fig. 4-6) is the stretch of flow decisions between two such
resets, not one dataset instance. Metrics logged per epoch match what the
paper tracks: total/requested flow counts, admission rate, mean latency,
jitter stdev.

Needs a GPU. Run via scripts/cluster/submit_b4_train.sbatch on disi.

Usage:
    python -m baselines_future.b4_gcn_td3.train \
        --data data --split train --out-checkpoint baselines_future/b4_gcn_td3/checkpoints/b4.pt \
        --log results/b4_train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from baselines_future.b4_gcn_td3.env import GCL_THRESHOLD_OMEGA, B4Environment
from baselines_future.b4_gcn_td3.td3 import ReplayBuffer, TD3Agent
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger, read_jsonl
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology


def _save_checkpoint(agent: TD3Agent, out_checkpoint: str, instance_idx: int, keep_versioned: int) -> None:
    """Overwrites the 'latest' pointer at `out_checkpoint`, and -- unless
    `keep_versioned` is 0 -- also writes a numbered snapshot alongside it,
    pruning older snapshots beyond the keep window. See --keep-versioned-checkpoints."""
    agent.save(out_checkpoint)
    if keep_versioned <= 0:
        return
    out_path = Path(out_checkpoint)
    versioned_dir = out_path.parent
    versioned_path = versioned_dir / f"{out_path.stem}_inst{instance_idx:04d}{out_path.suffix}"
    agent.save(str(versioned_path))
    existing = sorted(versioned_dir.glob(f"{out_path.stem}_inst*{out_path.suffix}"))
    for stale in existing[:-keep_versioned]:
        stale.unlink()


def _to_action(raw: torch.Tensor, valid_range: tuple[int, int]) -> tuple[bool, int]:
    """[-1,1]^2 tanh output -> (accept, offset_ns). accept_logit > 0 means
    accept (standard zero-threshold on a tanh output used for a discrete
    decision in continuous-control RL). offset_frac in [-1,1] rescaled
    linearly into [low, high)."""
    accept_logit, offset_frac = raw.tolist()
    accept = accept_logit > 0.0
    low, high = valid_range
    offset = int(low + (offset_frac + 1.0) / 2.0 * (high - low))
    return accept, offset


def run_epoch_over_instance(
    instance: dict,
    agent: TD3Agent,
    buffer: ReplayBuffer,
    batch_size: int,
    gcl_threshold: int,
    train: bool = True,
) -> list[dict]:
    """Processes one dataset instance's new_flows fully, resetting the
    environment on every GCL-threshold fallback along the way. Returns one
    dict of epoch-summary metrics per fallback reset that occurred (plus a
    final partial one for whatever's left when the instance's flows run out
    -- not a real omega-triggered epoch, but logged so no training signal
    from the tail is silently dropped)."""
    topology = Topology.from_dict(instance["topology_after"])
    topology_before = topology.pre_addition_subview(max_step=0)
    base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
    admitted_ids = set(instance["base_admitted_flow_ids"])
    admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
    new_flows = sorted([Flow.from_dict(d) for d in instance["new_flows"]], key=lambda f: f.arrival_order)

    previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)
    env = B4Environment(topology, admitted_base_flows, previous_result, gcl_threshold=gcl_threshold)

    epoch_logs: list[dict] = []
    epoch_requested = epoch_scheduled = 0
    epoch_latencies: list[float] = []

    for flow in new_flows:
        state_graph = env.state_graph()
        raw_action = agent.select_action(state_graph, explore=train)
        low, high = env.valid_offset_range(flow)

        if flow.flow_class.value == "avb":
            reward, done, info = env.step(flow, accept=False)  # ignored inside step() for AVB
        else:
            accept, offset_ns = _to_action(raw_action, (low, high))
            reward, done, info = env.step(flow, accept=accept, offset_ns=offset_ns)
            epoch_requested += 1
            if info["admitted"]:
                epoch_scheduled += 1
                epoch_latencies.append(env.latencies_us[-1])

            next_graph = env.state_graph()
            if train:
                buffer.push(state_graph, raw_action, reward, next_graph, float(done))
                agent.train_step(buffer, batch_size=batch_size)

        if done:
            epoch_logs.append({
                "requested": epoch_requested, "scheduled": epoch_scheduled,
                "admission_rate": epoch_scheduled / epoch_requested if epoch_requested else 0.0,
                "mean_latency_us": statistics.fmean(epoch_latencies) if epoch_latencies else 0.0,
                "jitter_us": statistics.pstdev(epoch_latencies) if len(epoch_latencies) > 1 else 0.0,
                "gcl_max_entries": env.gcl_max_entries(), "fallback": True,
            })
            epoch_requested = epoch_scheduled = 0
            epoch_latencies = []
            # ILP fallback: recompute the whole schedule from scratch over
            # every flow seen so far, then keep processing this instance's
            # remaining new_flows against the fresh schedule.
            fallback_result, _fp, _hp2 = solve_b1(topology, env.all_flows_seen, time_limit_s=30.0)
            admitted_flows_for_seed = [f for f in env.all_flows_seen if f.flow_id in fallback_result.admitted]
            env = B4Environment(topology, admitted_flows_for_seed, fallback_result, gcl_threshold=gcl_threshold)

    if epoch_requested:
        epoch_logs.append({
            "requested": epoch_requested, "scheduled": epoch_scheduled,
            "admission_rate": epoch_scheduled / epoch_requested if epoch_requested else 0.0,
            "mean_latency_us": statistics.fmean(epoch_latencies) if epoch_latencies else 0.0,
            "jitter_us": statistics.pstdev(epoch_latencies) if len(epoch_latencies) > 1 else 0.0,
            "gcl_max_entries": env.gcl_max_entries(), "fallback": False,
        })
    return epoch_logs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the B4 GCN-TD3 agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--limit", type=int, default=None, help="train on at most N instances")
    ap.add_argument("--gcl-threshold", type=int, default=GCL_THRESHOLD_OMEGA)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--graph-embedding-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-checkpoint", default="baselines_future/b4_gcn_td3/checkpoints/b4.pt")
    ap.add_argument("--log", default="results/b4_train_log.jsonl")
    ap.add_argument("--checkpoint-every-instances", type=int, default=10,
                     help="save --out-checkpoint after every N dataset instances, not just at the "
                          "end -- a run that only saves once at the very end loses everything to a "
                          "SLURM TIMEOUT if it doesn't finish inside the job's time budget, since "
                          "nothing survives to disk until that final save")
    ap.add_argument("--keep-versioned-checkpoints", type=int, default=8,
                     help="in addition to overwriting --out-checkpoint (the 'latest' pointer) on "
                          "every periodic save, also keep this many numbered snapshots "
                          "(<out-checkpoint stem>_instXXX.pt), oldest deleted first -- a policy that "
                          "was healthy earlier in a run can collapse later (see PROGRESS.md "
                          "2026-09-24: the actor saturated to a constant action late in a run, and "
                          "the only-ever-overwritten single checkpoint meant the earlier good policy "
                          "was unrecoverable). 0 disables versioning and keeps only the overwritten "
                          "'latest' file, matching the old behavior.")
    ap.add_argument("--resume-from", default=None,
                     help="continue an interrupted run: loads actor/critic weights from this "
                          "checkpoint, skips every instance already present in --log (read from "
                          "the existing file), and appends new epochs to it instead of overwriting. "
                          "Note: the replay buffer and optimizer state are NOT restored (TD3Agent.save "
                          "only persists network weights) -- a resumed run's early updates start from "
                          "an empty buffer again, same cold-start dynamic as a fresh run, just with an "
                          "already-partially-trained policy instead of a random one.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    instance_files = sorted((Path(args.data) / args.split).glob("*/instance.json"))
    if args.limit:
        instance_files = instance_files[: args.limit]

    agent = TD3Agent(graph_embedding_dim=args.graph_embedding_dim, device=args.device)
    buffer = ReplayBuffer()

    epoch_idx = 0
    log_mode = "w"
    if args.resume_from:
        agent.load(args.resume_from)
        already_done = set()
        for row in read_jsonl(args.log):
            already_done.add(row["instance_id"])
            epoch_idx = max(epoch_idx, row["epoch"])
        before = len(instance_files)
        instance_files = [f for f in instance_files if json.loads(f.read_text(encoding="utf-8"))["instance_id"]
                           not in already_done]
        log_mode = "a"
        print(f"Resuming from {args.resume_from}: {before - len(instance_files)} instances already "
              f"logged in {args.log}, {len(instance_files)} remaining, continuing epoch count from {epoch_idx}")

    print(f"Training on {len(instance_files)} instances from {args.split}, device={args.device}")

    Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.log, mode=log_mode)
    t0 = time.perf_counter()
    for instance_idx, instance_file in enumerate(instance_files, start=1):
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)
        epoch_logs = run_epoch_over_instance(
            instance, agent, buffer, batch_size=args.batch_size, gcl_threshold=args.gcl_threshold, train=True,
        )
        for log in epoch_logs:
            epoch_idx += 1
            log.update({"epoch": epoch_idx, "instance_id": instance["instance_id"],
                        "elapsed_s": time.perf_counter() - t0})
            logger.log(log)
            print(f"epoch {epoch_idx}: requested={log['requested']} scheduled={log['scheduled']} "
                  f"admission_rate={log['admission_rate']:.3f} jitter_us={log['jitter_us']:.1f} "
                  f"fallback={log['fallback']}")

        if instance_idx % args.checkpoint_every_instances == 0:
            _save_checkpoint(agent, args.out_checkpoint, instance_idx, args.keep_versioned_checkpoints)
            print(f"checkpoint saved at instance {instance_idx}/{len(instance_files)}, epoch {epoch_idx}")

    _save_checkpoint(agent, args.out_checkpoint, len(instance_files), args.keep_versioned_checkpoints)
    logger.close()
    print(f"Done. {epoch_idx} epochs. Checkpoint saved to {args.out_checkpoint}")


if __name__ == "__main__":
    main()
