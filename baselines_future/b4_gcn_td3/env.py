"""B4's RL environment: one MDP step = one dynamically-arriving flow request
(arXiv:2405.05019 Sec. IV-C). Wraps `tsn_bench/scheduling/simulator.py`
directly -- the same no-wait, hop-by-hop GCL model every other baseline
(B1-B3) and every SLM method (S1-S3) already share, rather than a new
OMNeT++/INET-driven environment (this project deliberately doesn't use
OMNeT++, see MILESTONES.md).

Documented deviation from the paper's action space (the single biggest
divergence in this reimplementation, and why): the paper's action vector
includes a *separate* continuous gate-opening-time action per bridge on the
flow's path (literally indexed "first bridge," "second bridge" -- it
doesn't generalize past 2 hops in the paper text). That only makes sense
under a per-hop-independent timing model. This project's shared simulator
is explicitly a "no-wait" model: `resolve_flow_path`'s docstring states a
TT flow's entire path timing is a deterministic function of ONE decision
variable, its release offset -- every hop's window is then fixed by
transmission+propagation time, with no independent per-hop slack to choose.
Every other baseline (B1/B2/B3) and every SLM method (S1-S3) already treats
"schedule this flow" as "pick one offset." Giving B4 a different,
richer action space than every other method would both (a) not be
implementable against the shared simulator without forking it just for B4,
breaking "every method sees the same environment" (MILESTONES.md), and
(b) make B4 not a fair comparison point -- it would be solving an easier
problem with more control than B1-B3/S1-S3 have. So B4's action is
`(accept: bool, offset_ns: float in [0, period_ns))`, same decision
variable as every other method, with the RL agent's TD3 policy choosing it
instead of an ILP solver.

Two other paper mechanics are inherited unchanged where our simulator
already implements them, and simplified elsewhere with the same
"already-documented-elsewhere" reasoning MILESTONES.md's "Known
simplifications" uses for B1-B3:
  - AVB flows bypass the RL decision entirely, admitted via the existing
    deterministic reserved-bandwidth check (`cp_sat_common.admit_avb_flows`)
    -- matches the paper's own training setup ("AVB flows... scheduled
    statically... using ILP") and every other baseline's AVB handling.
  - r(B) (the BE-queue preemption-guard reward term) is fixed at 1 --
    this project's shared simulator has no BE-flow generation or
    credit-based-shaping simulation (same simplification noted for AVB in
    `simulator.py`'s own module docstring), so there is nothing for r(B)
    to check.
"""
from __future__ import annotations

import statistics

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.scheduling.simulator import (
    LinkOccupancy,
    ScheduleResult,
    compute_hyperperiod_ns,
    flow_intervals,
    resolve_flow_path,
)
from tsn_bench.serialization.graph_format import gcn_td3_node_features
from tsn_bench.solvers.cp_sat_common import admit_avb_flows
from tsn_bench.topology.models import Topology

# The paper's own omega=128 is a specific real switch's hardware limit
# (InnoRoute Trustnode) from their much smaller toy setup. Measured
# locally against this dataset (2026-09-22): base schedules alone --
# before any dynamically-arriving flow is even processed -- already sit
# at 136-632 entries on the busiest link (low load: 136-272, medium:
# 316-380, near_saturation: 456-632), because this dataset's flow
# generator produces far more TT flows sharing links than the paper's toy
# 2-switch/2-flow example ever did. Under the literal 128, every episode
# is already over threshold at reset, triggering an immediate ILP
# fallback with zero flows processed and zero RL training signal --
# confirmed empirically: a first training smoke test (job 52661) fell
# back on essentially every single step. Reusing 128 here would be
# transcribing a number calibrated for a different scale problem, not a
# faithful reproduction. `tsn_bench/scheduling/metrics.py::compute_metrics`
# already established a dataset-appropriate default (1024) for exactly
# this reason -- every other baseline in this project already measures
# GCL feasibility against 1024, not 128. Reusing that default here keeps
# B4 consistent with the rest of the project and gives real headroom
# above all observed base-schedule levels for training signal to
# accumulate before a fallback is genuinely warranted.
GCL_THRESHOLD_OMEGA = 1024

# Not in the paper -- added 2026-09-22 after a real training run (job
# 52672) collapsed to always-rejecting every TT flow for 650+ consecutive
# epochs. Root cause: eq. (2)'s reward gives an outright reject (a1=0) and
# a failed accept attempt (a1=1 but r_O*r_S*r_B=0, e.g. the chosen offset
# collided) the EXACT SAME score -- both zero the compliance term and
# differ only by the shared penalty terms. Correctly guessing a
# non-colliding offset in an already-dense schedule is genuinely hard, so
# TD3's deterministic policy gradient had a real incentive to collapse
# toward "always reject": a stable, predictable reward with no downside,
# versus a noisy, usually-failing attempt. Checkpoint inspection (job
# 52705) confirmed this wasn't a numerical divergence (no NaN/Inf, all
# weights small and unremarkable) -- it's a legitimate local optimum of
# the reward as originally transcribed from the paper. This penalty is
# subtracted ONLY on an outright reject (not on a failed accept attempt),
# so attempting and missing is no longer reward-equivalent to not trying.
REJECT_PENALTY = 0.1


class B4Environment:
    def __init__(
        self,
        topology: Topology,
        admitted_base_flows: list[Flow],
        base_schedule: ScheduleResult,
        gcl_threshold: int = GCL_THRESHOLD_OMEGA,
        reject_penalty: float = REJECT_PENALTY,
    ):
        self.topology = topology
        self.gcl_threshold = gcl_threshold
        self.reject_penalty = reject_penalty
        self.flow_paths = {f.flow_id: resolve_flow_path(topology, f) for f in admitted_base_flows}
        self.schedule = ScheduleResult(
            admitted=set(base_schedule.admitted), rejected=set(),
            offsets_ns=dict(base_schedule.offsets_ns), method="B4_gcn_td3",
        )
        self.link_occupancy: dict[str, LinkOccupancy] = {}
        self.avb_reserved_mbps: dict[str, float] = {}
        self.all_flows_seen: list[Flow] = list(admitted_base_flows)
        self.latencies_us: list[float] = []
        self.hyperperiod_ns = compute_hyperperiod_ns(list(self.flow_paths.values())) or 1
        self._seed_from_base()

    def _seed_from_base(self) -> None:
        for fid in self.schedule.admitted:
            fp = self.flow_paths.get(fid)
            if fp is None or fp.flow.flow_class != FlowClass.TT or fid not in self.schedule.offsets_ns:
                continue
            offset = self.schedule.offsets_ns[fid]
            for link_id, start, end in flow_intervals(fp, offset, self.hyperperiod_ns):
                self.link_occupancy.setdefault(link_id, LinkOccupancy()).insert(start, end)
            self.latencies_us.append(fp.total_delay_ns / 1000.0)

    def state_graph(self) -> dict:
        """GCN-TD3 Table II graph for the current partial schedule -- the
        TD3 agent's state (before GCN encoding)."""
        return gcn_td3_node_features(
            self.topology, self.all_flows_seen, schedule=self.schedule,
            flow_paths=self.flow_paths, gcl_events=self._gcl_events(),
        )

    def _gcl_events(self) -> dict[str, list[tuple[int, int]]]:
        events: dict[str, list[tuple[int, int]]] = {}
        for fid in self.schedule.admitted:
            fp = self.flow_paths.get(fid)
            if fp is None or fp.flow.flow_class != FlowClass.TT or fid not in self.schedule.offsets_ns:
                continue
            offset = self.schedule.offsets_ns[fid]
            for hop in fp.hops:
                switch_id = self.topology.links[hop.link.link_id].src
                sqid = f"{switch_id}__sq"
                start = offset + hop.pre_delay_ns
                events.setdefault(sqid, []).append((start, start + hop.trans_time_ns))
        return events

    def gcl_max_entries(self) -> int:
        return max((len(occ.starts) for occ in self.link_occupancy.values()), default=0)

    def gcl_threshold_exceeded(self) -> bool:
        return self.gcl_max_entries() > self.gcl_threshold

    def valid_offset_range(self, flow: Flow) -> tuple[int, int]:
        """[low, high) an accepted flow's release offset must fall in --
        same bound `cp_sat_common.schedule_tt_flows_cpsat` gives its
        offset variable (constraint 3: non-negative, transmission window
        fits within the flow's own period)."""
        from tsn_bench.scheduling.simulator import us_to_ns
        return 0, max(1, us_to_ns(flow.period_us))

    def step(self, flow: Flow, accept: bool, offset_ns: int | None = None) -> tuple[float, bool, dict]:
        """One dynamically-arriving flow. `offset_ns` is ignored for a
        rejected or AVB flow. Returns (reward, episode_done, info) --
        `episode_done` mirrors Algorithm 1's `until beta = omega`: True
        once any link's GCL length exceeds the threshold, at which point
        the caller (train.py) is expected to trigger the ILP fallback and
        start a fresh episode via `B4Environment(...)` on the
        now-larger flow set."""
        self.all_flows_seen.append(flow)

        if flow.flow_class == FlowClass.AVB:
            fp = resolve_flow_path(self.topology, flow)
            self.flow_paths[flow.flow_id] = fp
            admitted, _rejected, self.avb_reserved_mbps = admit_avb_flows(
                self.topology, {}, set(), {flow.flow_id: fp}, self.avb_reserved_mbps,
            )
            was_admitted = flow.flow_id in admitted
            (self.schedule.admitted if was_admitted else self.schedule.rejected).add(flow.flow_id)
            # No RL reward for AVB -- see module docstring.
            return 0.0, self.gcl_threshold_exceeded(), {"admitted": was_admitted, "flow_class": "avb"}

        fp = resolve_flow_path(self.topology, flow)
        self.flow_paths[flow.flow_id] = fp

        if not accept:
            self.schedule.rejected.add(flow.flow_id)
            reward = self._penalty_terms() - self.reject_penalty
            return reward, self.gcl_threshold_exceeded(), {"admitted": False, "flow_class": "tt"}

        assert offset_ns is not None
        low, high = self.valid_offset_range(flow)
        offset_ns = max(low, min(high - 1, offset_ns))

        r_o = 1.0 if fp.total_delay_ns <= fp.deadline_ns else 0.0
        r_s = 1.0
        for link_id, start, end in flow_intervals(fp, offset_ns, self.hyperperiod_ns):
            occ = self.link_occupancy.get(link_id)
            if occ is not None and occ.overlaps(start, end):
                r_s = 0.0
                break
        r_b = 1.0  # BE-queue check -- always satisfied, see module docstring

        valid = r_o == 1.0 and r_s == 1.0
        if valid:
            self.schedule.admitted.add(flow.flow_id)
            self.schedule.offsets_ns[flow.flow_id] = offset_ns
            for link_id, start, end in flow_intervals(fp, offset_ns, self.hyperperiod_ns):
                self.link_occupancy.setdefault(link_id, LinkOccupancy()).insert(start, end)
            self.latencies_us.append(fp.total_delay_ns / 1000.0)
        else:
            self.schedule.rejected.add(flow.flow_id)

        compliance = r_o * r_s * r_b  # a1 = 1 (this branch is only reached on accept)
        reward = compliance + self._penalty_terms()
        return reward, self.gcl_threshold_exceeded(), {
            "admitted": valid, "flow_class": "tt", "r_o": r_o, "r_s": r_s,
        }

    def _penalty_terms(self) -> float:
        """-beta/omega - Fd/(Fd+Fc) - J_sigma/J_mu, eq. (2)'s three penalty
        terms, using the schedule state as of right now (after this
        step's admit/reject decision has already been applied)."""
        beta_term = self.gcl_max_entries() / self.gcl_threshold
        fd = len(self.schedule.rejected)
        fc = len(self.schedule.admitted)
        drop_term = fd / (fd + fc) if (fd + fc) else 0.0
        if len(self.latencies_us) > 1:
            mean = statistics.fmean(self.latencies_us)
            jitter_term = (statistics.pstdev(self.latencies_us) / mean) if mean else 0.0
        else:
            jitter_term = 0.0
        return -beta_term - drop_term - jitter_term
