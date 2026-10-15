# S4: full agent — GNN-tool + solver-tool

Combines S2's solver tool with S3's retrieval mechanism into one agent
that has BOTH available and decides for itself when to use each, instead
of S3's design where retrieval always happens automatically before the
model ever sees the prompt. This is the actual research question S4
asks: does giving the model that choice change quality or latency,
compared to S2 (never retrieve) and S3 (always retrieve)?

(Note on naming: the original research plan called this "GNN-tool +
solver-tool," but the actual retrieval mechanism built for S3 — and
reused here — is a plain k-hop graph-traversal retriever, not a
GNN-based reranker; the plan's "option (b)" GNN retriever was never
built, see `s3_graphrag/README.md`. S4 pairs the solver tool with
whichever retrieval tool actually exists, which is this one.)

## What it is

- `tool_schema.py::RETRIEVE_SUBGRAPH_TOOL` / `retrieve_subgraph_tool` —
  a second tool, wrapping S3's exact retrieval mechanism
  (`retrieval_node_ids` + `k_hop_subview`, k_hop=0) as a callable the
  model can invoke mid-conversation instead of having it applied
  automatically.
- `agent.py::run_agent` — a new two-tool loop (not a thin wrapper around
  S2's, since S2's loop hardcodes single-tool dispatch): the model is
  offered `retrieve_subgraph` (optional) and `solve_subschedule`
  (required, same semantics as S2 — must be called once with every
  candidate flow ID). Dispatches by the tool call's `name` field each
  round; a `retrieve_subgraph` call appends the narrowed topology text
  as a tool result and the loop continues, a `solve_subschedule` call
  is handled identically to S2's (same collision/deadline-checked
  solver, same bookkeeping for context flows across calls).
- `evaluate.py` — same CLI/JSONL schema as S2/S3, plus `retrieved`
  (bool) and `retrieval_calls_made`, logging whether the model chose to
  retrieve on each instance.

## Status

Pipeline built (2026-09-25), grounded in S2's tool-calling loop and
S3's retrieval mechanism, both already validated at full scale. Local
mechanics test confirmed (`retrieve_subgraph_tool`'s underlying
retrieval logic — the same `k_hop_subview`/`retrieval_node_ids` pair
S3 already validated — produces the expected narrowed topology text).
Not yet smoke-tested with real torch execution on disi.

## Not yet done

1. Smoke test on disi (`scripts/cluster/submit_s4_eval.sbatch`, a
   handful of instances) to confirm the two-tool dispatch actually
   works end-to-end with the real model.
2. Full held_out_test / test evaluation.
3. The actual comparison: does the model's own retrieve-or-not choice
   correlate with anything useful (e.g. retrieving more on
   near_saturation instances where the full topology is denser), and
   does having the choice change quality/latency versus S2's never and
   S3's always?

## Compute

Same tier as S2/S3 (GPU, `disi`'s `l40s` partition, same `.venv-gpu`).
