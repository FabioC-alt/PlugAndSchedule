# B5/B6: RL-architecture and routing ablations (not implemented yet)

- **B5 (GCN-DDQN) / B6-secondary (GCN-DDPG)**: same GCN encoder as B4, swap
  the RL head. Only worth building once B4 exists and is validated — these
  are internal ablations, not a separate reimplementation effort.
- **B6 (optional)**: GCN-based joint routing+scheduling DRL (Yang et al.,
  IEEE IoT Journal 2022). Only in scope if routing (not just scheduling)
  becomes part of the research question — right now
  `tsn_bench/topology/models.py::Topology.shortest_path` fixes routing via
  plain BFS, which is sufficient for the scheduling-only question this repo
  currently targets.

Nothing to do here until B4 (`../b4_gcn_td3/`) exists.
