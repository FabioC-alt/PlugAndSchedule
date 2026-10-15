"""Flow-set generator: TT profiles + AVB profile from the GCN-TD3 paper's
traffic distribution, randomized priority/src-dst/arrival order, and a
load-factor knob that controls how close to link saturation the instance is.

Default (`capacity_scale_links=None`): load factor is defined relative to
a single link's rated capacity (the bottleneck-link proxy): sum(flow
bandwidth) / link_rate_mbps. This is a coarse network-wide proxy, not a
per-link exact utilization, and it has a real consequence documented in
PROGRESS.md's 2026-09-17 node/flow-count scaling entry: since the target
is one fixed link's capacity regardless of topology size, flow count
stays roughly flat across switch counts 2-16 (~287 flows either way) —
bigger topologies don't get proportionally more candidate flows.

Optional `capacity_scale_links`: multiplies the target by that many
links' worth of capacity, so load factor becomes relative to an
*aggregate* capacity proxy that grows with the topology instead of a
fixed single link. Confirmed via the `scaling_probe` dataset (see
PROGRESS.md's 2026-09-17 entries) that pushing the target well past a
single link's capacity is what actually produces admission-rate/latency
stress in B1-B3 -- this parameter is how that gets combined with
topology size instead of staying a topology-size-independent multiplier.
Kept opt-in (default None => old behavior, byte-identical to before) so
every existing dataset/result stays meaningful; see
configs/aggregate_scaling.yaml for the dataset that actually uses it.
"""
from __future__ import annotations

import random

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.topology.models import NodeType, Topology

TTProfile = dict  # {"size_bytes": int, "period_us": float, "deadline_us": float}


def generate_flows(
    topology: Topology,
    tt_profiles: list[TTProfile],
    avb_profile: dict,
    avb_fraction: float,
    link_rate_mbps: float,
    load_factor: float,
    rng: random.Random,
    restrict_src_dst_to: set[str] | None = None,
    id_prefix: str = "f",
    added_at_step: int = 0,
    start_arrival_order: int = 0,
    max_attempts: int = 2000,
    capacity_scale_links: int | None = None,
) -> list[Flow]:
    """Generate flows until requested bandwidth reaches `load_factor` of the
    target capacity -- one link's, or `capacity_scale_links` links' worth if
    given (see module docstring). If `restrict_src_dst_to` is given, every
    generated flow has its talker OR listener endpoint in that set (used for
    node-addition scenarios, where the new flows involve the newly added
    node).

    `max_attempts` bounds how many flows can be generated in one call (each
    attempt almost always succeeds, so this is effectively a flow-count cap).
    The default (2000) is never reached by the main dataset's load factors
    (top out at near_saturation=0.85, no topology scaling); raised
    explicitly by the scaling probe and by aggregate-scaled configs, which
    both push the effective target well past a single link's capacity.
    """
    talkers = [n.node_id for n in topology.talkers()]
    listeners = [n.node_id for n in topology.listeners()]
    if not talkers or not listeners:
        return []

    capacity_links = capacity_scale_links if capacity_scale_links else 1
    target_bw = load_factor * link_rate_mbps * capacity_links
    flows: list[Flow] = []
    used_bw = 0.0
    idx = 0
    arrival_order = start_arrival_order
    attempts = 0

    while used_bw < target_bw and attempts < max_attempts:
        attempts += 1
        is_avb = rng.random() < avb_fraction
        if is_avb:
            size = rng.randint(*avb_profile["size_bytes_range"])
            period = avb_profile["period_us"]
            deadline = avb_profile["deadline_us"]
            flow_class = FlowClass.AVB
        else:
            profile = rng.choice(tt_profiles)
            size = profile["size_bytes"]
            period = profile["period_us"]
            deadline = profile["deadline_us"]
            flow_class = FlowClass.TT

        if restrict_src_dst_to:
            anchor = rng.choice(list(restrict_src_dst_to))
            anchor_node = topology.nodes[anchor]
            if anchor_node.node_type == NodeType.TALKER:
                src = anchor
                dst = rng.choice(listeners)
            elif anchor_node.node_type == NodeType.LISTENER:
                src = rng.choice(talkers)
                dst = anchor
            else:
                # newly added switch: pick one of its own attached end-stations if present,
                # else fall back to any talker/listener pair
                candidates_t = [t for t in talkers if t in restrict_src_dst_to]
                candidates_l = [l for l in listeners if l in restrict_src_dst_to]
                src = rng.choice(candidates_t) if candidates_t else rng.choice(talkers)
                dst = rng.choice(candidates_l) if candidates_l else rng.choice(listeners)
        else:
            src = rng.choice(talkers)
            dst = rng.choice(listeners)

        if src == dst:
            continue

        flow = Flow(
            flow_id=f"{id_prefix}{idx}",
            src=src,
            dst=dst,
            flow_class=flow_class,
            size_bytes=size,
            period_us=period,
            deadline_us=deadline,
            pcp=rng.randint(0, 7),
            arrival_order=arrival_order,
            added_at_step=added_at_step,
        )
        flows.append(flow)
        used_bw += flow.bandwidth_mbps()
        idx += 1
        arrival_order += 1

    rng.shuffle(flows)
    for i, f in enumerate(flows):
        f.arrival_order = start_arrival_order + i
    return flows
