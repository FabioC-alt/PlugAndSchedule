# S3: SLM + GraphRAG

Isolates: does retrieving only the relevant subtopology (instead of the
whole network as text) change accuracy or decision latency versus S2,
holding the base model and tool constant?

## What it is

An S2 variant (`slm_future/s2_solver_tool/`) — same base model
(Qwen2.5-3B-Instruct, no LoRA), same `solve_subschedule` tool-calling loop
— where the model's *prompt* describes only a retrieved subtopology instead
of the full network. Retrieval never narrows what the *solver* schedules
against: `agent.py::run_agent_with_retrieval` calls S2's
`run_agent(..., prompt_topology=retrieved)`, passing the real, full
`topology` for the tool call and only the filtered one for the prompt text
(see `s2_solver_tool/agent.py::run_agent`'s `prompt_topology` param).

- `agent.py::retrieval_node_ids` — the retrieval center-node set: the newly
  added node(s) **plus every new flow's actual shortest-path route**
  (`topology.shortest_path(f.src, f.dst)`). Routes matter because the
  dataset generator only requires ONE endpoint of a new flow to touch a new
  node (`tsn_bench/flows/generator.py`) — the other endpoint can be
  anywhere in the topology, so centering retrieval on new nodes alone can
  silently omit a flow's talker or listener from the model's view.
- `Topology.k_hop_subview` (`tsn_bench/topology/models.py`) — expands a
  k-hop buffer around that center set (`nx.ego_graph`, mirroring
  `b2_incremental_ilp.k_hop_affected_flows`'s pattern applied to nodes).
- `evaluate.py` — same CLI/JSONL schema as S2's `evaluate.py`, plus
  `k_hop`, `retrieved_node_count`, `full_node_count`,
  `context_reduction_pct`.

## Finding: k_hop=0 is the only setting with real reduction

Measured locally against `held_out_test` (16-switch instances, all load
factors/scenarios): once retrieval correctly includes full flow routes (not
just new-node neighborhoods), a k-hop buffer washes out almost all context
savings, because a flow's shortest path alone can already span most of the
graph.

| k_hop | avg. nodes kept |
|---|---|
| 0 (routes only, no buffer) | 39% (low) – 75% (near_saturation) |
| 1 | 91% (low) – 100% (near_saturation) |
| 2 | ~100% across the board |

So `evaluate.py --k-hop` defaults to **0**: retrieval keeps exactly the
new node(s) and every new flow's route, nothing more. This is still a real
25–61% node reduction depending on load factor (more flows → more routes →
less reduction), and it's the setting actually worth comparing against S2's
full-topology prompt. `--k-hop 1`/`2` remain available for an ablation
showing the wash-out directly, but aren't expected to move accuracy or
latency much relative to S2 since they barely shrink the prompt.

## Result: matches S2's accuracy, real context savings, no latency win

Full `held_out_test` run (720 instances, job 52612, 2026-09-21) vs. S2's
existing held_out_test results, same split, same base model:

| | S3 (k_hop=0) | S2 |
|---|---|---|
| Admission accuracy (blended) | 95.4% | ~95.0% |
| — low / medium / near_saturation | 99.3% / 96.4% / 90.4% | 99.1% / 97.1% / 88.7% |
| Feasibility rate | 100.0% | 100.0% |
| Avg. context reduction | 40.3% (55.4% / 37.6% / 28.0%) | 0% (full topology) |
| Decision latency (blended) | ~22.8s | ~19.2s |
| — low / medium / near_saturation | 9.76s / 22.36s / 36.38s | 9.31s / 12.30s / 36.03s |

Admission accuracy and feasibility match S2 almost exactly — retrieval
doesn't cost quality. Context reduction is real and confirmed at scale,
shrinking as load increases (more flows → more routes → less of the graph
is prunable), matching the local measurement above.

**Latency is not improved**, and medium load shows an unexplained ~1.8x
slowdown (22.4s vs 12.3s) at *identical* candidate-flow counts on both
sides, with low variance on both runs (not a couple of outliers). Not
explained by prompt length (S3's prompt is shorter, if anything) or flow
count (identical). Low and near_saturation loads are close to parity.
Open question, not chased further this session — worth a follow-up if S3
becomes a real candidate method rather than an ablation.

**Net**: S3 is a quality-preserving context-reduction technique, not a
latency win. Useful for the "scale to topologies too large to fit in
context" half of the original question — not directly testable with the
current dataset's 16-switch max (which already fits unfiltered, ~2KB) —
but not for speed at the sizes tested here.

Raw results: `results/s3_eval_held_out_test.jsonl`.
