"""Shared time-slotted GCL scheduling model, used identically by B1/B2/B3 (and
later S1-S4) so every method is compared on the same problem representation.

Model (documented simplification vs. a full OMNeT++/INET simulation — see
MILESTONES.md "Known simplifications"):
  - "No-wait" hop-by-hop forwarding: a frame is retransmitted on its next hop
    immediately after store-and-forward transmission + link propagation delay,
    with no extra queuing slack beyond that. This means a TT flow's entire
    path timing is a deterministic function of a single decision variable:
    its release offset within its own period.
  - All TT flow periods in this dataset are harmonic (250/500/1000us all
    divide 4000us), so the schedule only needs to be verified over one
    hyperperiod = max(period) to guarantee correctness for all repetitions.
  - A physical link can transmit one frame at a time regardless of queue/PCP,
    so link contention is modeled as a single non-overlap resource per
    (link, direction) — physically accurate for the transmission itself.
  - AVB flows are admitted via a reserved-bandwidth budget check on every
    link they cross, not a full credit-based-shaping (CBS) simulation.

All time values are integer nanoseconds internally to keep CP-SAT exact
(sub-microsecond transmission times at 1Gbps would otherwise be fractional).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.topology.models import Link, Topology

US_TO_NS = 1000


def us_to_ns(value_us: float) -> int:
    return round(value_us * US_TO_NS)


@dataclass
class HopTiming:
    link: Link
    trans_time_ns: int
    prop_delay_ns: int
    pre_delay_ns: int  # cumulative delay (from flow release) at the START of this hop's transmission


@dataclass
class FlowPath:
    flow: Flow
    node_path: list[str]
    hops: list[HopTiming]

    @property
    def total_delay_ns(self) -> int:
        if not self.hops:
            return 0
        last = self.hops[-1]
        return last.pre_delay_ns + last.trans_time_ns + last.prop_delay_ns

    @property
    def period_ns(self) -> int:
        return us_to_ns(self.flow.period_us)

    @property
    def deadline_ns(self) -> int:
        return us_to_ns(self.flow.deadline_us)


def resolve_flow_path(topology: Topology, flow: Flow) -> FlowPath:
    node_path = topology.shortest_path(flow.src, flow.dst)
    hops: list[HopTiming] = []
    cumulative = 0
    for a, b in zip(node_path[:-1], node_path[1:]):
        link = topology.link_between(a, b)
        if link is None:
            raise ValueError(f"no link {a}->{b} on path for flow {flow.flow_id}")
        trans_time_ns = round((flow.size_bytes * 8 * 1000) / link.rate_mbps)
        prop_delay_ns = us_to_ns(link.propagation_delay_us)
        hops.append(HopTiming(link=link, trans_time_ns=trans_time_ns, prop_delay_ns=prop_delay_ns,
                               pre_delay_ns=cumulative))
        cumulative += trans_time_ns + prop_delay_ns
    return FlowPath(flow=flow, node_path=node_path, hops=hops)


def compute_hyperperiod_ns(flow_paths: list[FlowPath]) -> int:
    periods = {fp.period_ns for fp in flow_paths}
    if not periods:
        return 1
    h = 1
    for p in periods:
        h = math.lcm(h, p)
    return h


class LinkOccupancy:
    """Sorted, non-overlapping interval set for one link, supporting O(log n)
    collision checks and inserts. Shared by every method that needs to
    validate "does this offset collide with what's already scheduled on this
    link" — B3's greedy placement and the SLM evaluation feasibility check
    both need exactly this, not two separate implementations of it."""

    __slots__ = ("starts", "ends")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect.bisect_right(self.starts, start)
        if i > 0 and self.ends[i - 1] > start:
            return True
        if i < len(self.starts) and self.starts[i] < end:
            return True
        return False

    def insert(self, start: int, end: int) -> None:
        i = bisect.bisect_right(self.starts, start)
        self.starts.insert(i, start)
        self.ends.insert(i, end)


def flow_intervals(fp: FlowPath, offset_ns: int, hyperperiod_ns: int) -> list[tuple[str, int, int]]:
    """Every (link_id, start_ns, end_ns) transmission interval a TT flow
    occupies across one hyperperiod, given a release offset."""
    repeats = max(1, hyperperiod_ns // fp.period_ns)
    out = []
    for hop in fp.hops:
        for k in range(repeats):
            start = offset_ns + k * fp.period_ns + hop.pre_delay_ns
            out.append((hop.link.link_id, start, start + hop.trans_time_ns))
    return out


@dataclass
class ScheduleResult:
    admitted: set[str] = field(default_factory=set)
    rejected: set[str] = field(default_factory=set)
    offsets_ns: dict[str, int] = field(default_factory=dict)   # TT flows only
    method: str = ""
    decision_latency_s: float = 0.0
    solver_status: str = ""

    def admission_rate(self, total_flows: int) -> float:
        return len(self.admitted) / total_flows if total_flows else 0.0
