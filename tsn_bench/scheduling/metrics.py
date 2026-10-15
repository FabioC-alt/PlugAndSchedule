"""Metrics computed for every method (B1-B3 today, B4-B6/S1-S5 later), per
research-plan section 5.1. All latency/jitter figures are in microseconds.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from tsn_bench.scheduling.simulator import FlowPath, ScheduleResult


@dataclass
class Metrics:
    method: str
    total_flows: int
    admitted_flows: int
    admission_rate: float
    deadline_compliance_rate: float   # among admitted flows
    mean_latency_us: float
    jitter_us: float                  # std dev of latency across admitted flows
    gcl_total_entries: int
    gcl_max_per_link: int
    decision_latency_s: float
    gcl_threshold_exceeded: bool

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "total_flows": self.total_flows,
            "admitted_flows": self.admitted_flows,
            "admission_rate": self.admission_rate,
            "deadline_compliance_rate": self.deadline_compliance_rate,
            "mean_latency_us": self.mean_latency_us,
            "jitter_us": self.jitter_us,
            "gcl_total_entries": self.gcl_total_entries,
            "gcl_max_per_link": self.gcl_max_per_link,
            "decision_latency_s": self.decision_latency_s,
            "gcl_threshold_exceeded": self.gcl_threshold_exceeded,
        }


def gcl_entries_per_link(result: ScheduleResult, flow_paths: dict[str, FlowPath],
                          hyperperiod_ns: int) -> dict[str, int]:
    """Number of gate-open/close events scheduled per link across one
    hyperperiod — a standard proxy for hardware GCL-table feasibility."""
    counts: dict[str, int] = {}
    for flow_id in result.admitted:
        fp = flow_paths.get(flow_id)
        if fp is None or fp.flow.flow_class.value != "tt":
            continue
        repeats = hyperperiod_ns // fp.period_ns
        for hop in fp.hops:
            counts[hop.link.link_id] = counts.get(hop.link.link_id, 0) + repeats
    return counts


def compute_metrics(
    method: str,
    result: ScheduleResult,
    flow_paths: dict[str, FlowPath],
    hyperperiod_ns: int,
    gcl_threshold: int = 1024,
) -> Metrics:
    total = len(result.admitted) + len(result.rejected)
    latencies_us: list[float] = []
    compliant = 0
    for flow_id in result.admitted:
        fp = flow_paths.get(flow_id)
        if fp is None:
            continue
        # latency (release -> delivery) is offset-independent by construction
        # of the no-wait hop-by-hop model — see simulator.py docstring.
        latency_ns = fp.total_delay_ns
        latency_us = latency_ns / 1000.0
        latencies_us.append(latency_us)
        if latency_us <= fp.flow.deadline_us:
            compliant += 1

    gcl_counts = gcl_entries_per_link(result, flow_paths, hyperperiod_ns)
    gcl_total = sum(gcl_counts.values())
    gcl_max = max(gcl_counts.values()) if gcl_counts else 0

    return Metrics(
        method=method,
        total_flows=total,
        admitted_flows=len(result.admitted),
        admission_rate=result.admission_rate(total),
        deadline_compliance_rate=(compliant / len(result.admitted)) if result.admitted else 0.0,
        mean_latency_us=statistics.fmean(latencies_us) if latencies_us else 0.0,
        jitter_us=statistics.pstdev(latencies_us) if len(latencies_us) > 1 else 0.0,
        gcl_total_entries=gcl_total,
        gcl_max_per_link=gcl_max,
        decision_latency_s=result.decision_latency_s,
        gcl_threshold_exceeded=gcl_max > gcl_threshold,
    )
