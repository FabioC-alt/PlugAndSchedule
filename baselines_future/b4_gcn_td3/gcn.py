"""GCN state encoder for B4 (arXiv:2405.05019, Sec. IV-B / Fig. 3).

Two-layer GCN (64 -> 32 neurons, ReLU, Kipf & Welling propagation rule)
over queue-level graph nodes, producing one fixed-size graph embedding per
instance -- the TD3 agent's state vector.

Design note on "variable-sized node features" (the problem Sec. IV-B
names): a raw Table II feature vector's length depends on how many flows
or gate events are attached to THAT SPECIFIC node instance (see
`tsn_bench/serialization/graph_format.py::gcn_td3_node_features`), not
just its type -- two talkers of the same type can have different vector
lengths. A single per-type `nn.Linear` can't consume that directly. The
paper's Fig. 3 says only "specialized input layers designed for each node
type," not how they handle the per-instance length difference within a
type. We resolve it with a DeepSets-style permutation-invariant pass: each
node's scalar prefix and each repeated per-item block (per-flow, or
per-gate-event) get their own small shared MLP, item blocks are
mean-pooled, then concatenated and projected to a common hidden size --
this consumes any flow/gate-event count without an arbitrary padding cap,
so it generalizes to instances with more flows than any single training
example, rather than truncating or erroring on them.

Similarly, Fig. 3's final output width ("22") is specific to the paper's
own toy example (it falls out of their example's n/s/b, not a universal
constant -- their own Fig. 3 summary formula doesn't even reconcile
digit-for-digit against Table II's marked columns for the BE row, so it
isn't safe to treat as an architectural constant either way). We expose
the graph embedding width as `graph_embedding_dim` instead of hardcoding
22.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Raw feature layout per node type, matching
# graph_format.py::gcn_td3_node_features exactly:
#   (n_scalar_prefix_features, [(per_item_block_size, item_kind), ...])
NODE_TYPE_LAYOUT = {
    "talker": (1, [(10, "flow")]),
    "listener": (1, [(2, "flow")]),
    "scheduled_queue": (7, [(2, "flow"), (2, "gate_event")]),
    "be_queue": (7, [(2, "gate_event")]),
}


class _DeepSetsNodeEncoder(nn.Module):
    """One node type's input layer: scalar prefix -> small MLP; each
    repeated item block -> a shared per-item MLP, mean-pooled over items
    (permutation-invariant, count-invariant); concatenate and project to
    `hidden_dim`. Handles zero items for a block (a node with no flows or
    no gate events yet) by feeding a zero vector -- the natural "nothing
    observed yet" representation, consistent with how
    `gcn_td3_node_features` itself zeros out an unresolved flow's timing.
    """

    def __init__(self, n_scalar: int, item_blocks: list[tuple[int, str]], hidden_dim: int):
        super().__init__()
        self.n_scalar = n_scalar
        self.item_blocks = item_blocks
        item_hidden = hidden_dim // 2
        self.scalar_mlp = nn.Sequential(nn.Linear(max(n_scalar, 1), item_hidden), nn.ReLU())
        self.item_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(block_size, item_hidden), nn.ReLU())
            for block_size, _kind in item_blocks
        ])
        self.project = nn.Linear(item_hidden * (1 + len(item_blocks)), hidden_dim)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """`raw`: 1D tensor, this node's exact Table-II feature vector
        (length = n_scalar + sum(block_size * count_i) for each block)."""
        device = raw.device
        scalar = raw[: self.n_scalar] if self.n_scalar > 0 else torch.zeros(1, device=device)
        parts = [self.scalar_mlp(scalar)]
        offset = self.n_scalar
        for (block_size, _kind), mlp in zip(self.item_blocks, self.item_mlps):
            remaining = raw[offset:]
            n_items = remaining.numel() // block_size
            if n_items == 0:
                pooled = torch.zeros(mlp[0].out_features, device=device)
            else:
                items = remaining[: n_items * block_size].view(n_items, block_size)
                pooled = mlp(items).mean(dim=0)
            parts.append(pooled)
            offset += n_items * block_size
        return self.project(torch.cat(parts))


class GCNLayer(nn.Module):
    """One Kipf & Welling propagation step: H' = ReLU(D^-1/2 (A+I) D^-1/2 H W)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        return F.relu(norm_adj @ self.linear(x))


def normalized_adjacency(n_nodes: int, edges: list[tuple[int, int]], device=None) -> torch.Tensor:
    """Dense symmetric-normalized adjacency with self-loops. Topologies
    here are small (well under 200 queue-level nodes even at 16 switches),
    so a dense [N,N] matrix is simplest and fast enough -- no need for
    sparse ops or a torch_geometric dependency for a model this size."""
    a = torch.eye(n_nodes, device=device)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0  # undirected message passing -- a flow's path direction is
        # already captured by the graph structure itself (talker->...->listener
        # edges only exist in that direction in the input), but state
        # aggregation benefits from both neighbors seeing each other, same
        # as the paper's own "aggregate feature information from
        # neighbouring nodes" (Sec. IV-B) without a stated directionality
        # restriction.
    deg = a.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    d = torch.diag(deg_inv_sqrt)
    return d @ a @ d


class GCNStateEncoder(nn.Module):
    """Full encoder: per-type DeepSets input layers -> 2 GCN layers
    (64, 32) -> global average pooling -> FC to `graph_embedding_dim`."""

    def __init__(self, hidden_dim: int = 64, conv2_dim: int = 32, graph_embedding_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_encoders = nn.ModuleDict({
            node_type: _DeepSetsNodeEncoder(n_scalar, blocks, hidden_dim)
            for node_type, (n_scalar, blocks) in NODE_TYPE_LAYOUT.items()
        })
        self.conv1 = GCNLayer(hidden_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, conv2_dim)
        self.output_fc = nn.Linear(conv2_dim, graph_embedding_dim)

    def forward(self, graph: dict) -> torch.Tensor:
        """`graph`: the dict returned by
        `graph_format.gcn_td3_node_features` (node_ids/features/edges,
        numpy arrays). Returns a single `[graph_embedding_dim]` vector."""
        device = next(self.parameters()).device
        ordered_ids: list[str] = []
        type_of: dict[str, str] = {}
        for node_type, ids in graph["node_ids"].items():
            for nid in ids:
                ordered_ids.append(nid)
                type_of[nid] = node_type
        index = {nid: i for i, nid in enumerate(ordered_ids)}
        n = len(ordered_ids)

        x = torch.zeros(n, self.hidden_dim, device=device)
        for nid in ordered_ids:
            raw = torch.as_tensor(graph["features"][nid], dtype=torch.float32, device=device)
            x[index[nid]] = self.input_encoders[type_of[nid]](raw)

        edge_idx = [(index[a], index[b]) for a, b in graph["edges"] if a in index and b in index]
        norm_adj = normalized_adjacency(n, edge_idx, device=device)

        h = self.conv1(x, norm_adj)
        h = self.conv2(h, norm_adj)
        graph_repr = h.mean(dim=0)  # Global Average Pooling
        return self.output_fc(graph_repr)
