# Plug and Schedule

Code, results and a reviewer package for the paper *Plug and Schedule: Can Language Models Admit New
Machines into Time-Sensitive Networks?*

The paper compares nine methods for admitting new nodes into a running Time-Sensitive Network (TSN):
three classical baselines (B1 full ILP, B2 incremental ILP, B3 greedy heuristic), a reinforcement learning
policy (B4, a reimplementation of GCN-TD3), and five language-model configurations (S1 fine-tuned model, S2
model with a solver tool, S2b yes/no scoring per flow, S3 narrowed prompt, S4 optional retrieval). All methods
are scored against the schedule of a full ILP solver on 2160 generated instances.

## Check the results in a few seconds

```bash
python reviewer-package/reproduce.py
```

This recomputes the paper's tables and statistics from the per-instance results in
`reviewer-package/results/` and compares them with the numbers printed in the paper. It needs only the Python
standard library (`scipy` is optional, for the p-values). Details, the dataset check and the commands to
re-run every method are in [`reviewer-package/README.md`](reviewer-package/README.md).

## Layout

```
reviewer-package/     recompute the paper's numbers, dataset manifest and fingerprints, cluster commands
tsn_bench/            dataset generator, solvers B1 to B3, evaluation tools
baselines_future/     B4 (GCN-TD3 reimplementation)
slm_future/           S1 to S4 (directory names are historical: s2b_opennod holds S2b)
configs/              dataset configuration (seeds, topology sizes, flow profiles, load levels)
scripts/cluster/      SLURM job scripts used for every run in the paper
PROGRESS.md, MILESTONES.md, TEST_PLAN.md, FUTURE_LOOK.md   development log kept while the work was done
```

The development log uses the earlier name "OpenNod" for the library now called SemIf-OpenJev (the repository
was renamed; it is the same code), which S2b uses for its yes/no scoring.

## Setup

```bash
pip install -r requirements.txt           # CPU: dataset generation, B1 to B3
pip install -r slm_future/s1_lora/requirements-gpu.txt   # GPU: B4, S1 to S4
```
