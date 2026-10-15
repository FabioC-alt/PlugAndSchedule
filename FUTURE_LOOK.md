# Future Look

Forward-looking notes: external tools, industry directions, and ideas worth
tracking that aren't concrete enough for `TEST_PLAN.md` yet. Newest entry
first, kept short — one card per idea. The web UI's Future Look tab
renders this file as a grid of cards.

## 2026-09-18 — OpenNod: became S2b, confirmed at full scale, strictly dominates S2

github.com/TheoLeeCJ/OpenNod reads typed option logits in one forward
pass instead of generating text — an open, MIT-licensed reproduction of
typesafe.ai's Jev pattern (see entry below). Claims independently
reproduced, prototyped as a drop-in for S2's slowest step, then
productionized as `slm_future/s2b_opennod/` and run on the full 946
instances (2026-09-19): **99.8-100% admission accuracy, 100%
feasibility, ~1.8-2.0s/instance** — vs. S2's 69.0%/100%/9.9s. Strictly
dominates S2 on every metric. This thread is closed out; see
`MILESTONES.md` Phase 4 and `slm_future/s2b_opennod/README.md` for the
full result. Caveat that still stands: OpenNod's base model (Qwen3.5-4B)
differs from S2's (Qwen2.5-3B-Instruct) — a mechanism win, not yet a
validated same-model ablation.

## 2026-09-18 — openjev: interesting primitive, not a direct fit

huggingface.co/AlexWortega/openjev is an MIT cross-encoder for NLI
(entailment scoring), demoed reranking and even playing Doom. No solid
benchmark table, just radar charts. Using it here would mean reframing
admission as entailment scoring — real rework, not a drop-in. Lower
priority than OpenNod; revisit only if OpenNod's prototype doesn't pan
out.

## 2026-09-18 — Qwen-2.5-1B-RLCD: not a fit for this infrastructure

huggingface.co/harshatheg/Qwen-2.5-1B-RLCD claims 5.6-7x speedup, but
it's Apple Silicon/MLX-specific — this project runs NVIDIA/CUDA via
PyTorch. The card doesn't actually explain "RLCD"; may be branding
rather than a real link to typesafe.ai's method. Skip.

## 2026-09-17 — TypeSafe AI / Jev: fast "typed decision" models

typesafe.ai's Jev targets exactly S2's problem (structured decisions
generated token-by-token are slow) via "RLCD" instead of RLHF, claiming
193.6x speedup. Not adopting yet: numbers are vendor-only (no paper, no
third-party benchmark), it's closed/hosted (this project needs
open/local), and it wouldn't resolve the deeper open question of whether
the SLM's judgment adds value over the solver at all. **Verdict**: watch,
don't pivot toward it — see OpenNod above for an open alternative that's
actually actionable now.
