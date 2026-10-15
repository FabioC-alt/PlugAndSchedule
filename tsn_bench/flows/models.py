"""Flow data model. A Flow is one talker->listener stream to be admitted and
scheduled. Single-destination only (no multicast) — a documented simplification."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlowClass(str, Enum):
    TT = "tt"     # time-triggered, hard deadline, scheduled table entry (GCL)
    AVB = "avb"   # audio-video bridging, credit-based-shaping class (budget-checked, not fully simulated)


@dataclass
class Flow:
    flow_id: str
    src: str            # talker node_id
    dst: str             # listener node_id
    flow_class: FlowClass
    size_bytes: int
    period_us: float
    deadline_us: float
    pcp: int = 0                 # 802.1Q priority code point, 0-7
    arrival_order: int = 0       # simulated arrival index, for "flows arriving after node add"
    added_at_step: int = 0       # 0 = present from the start; >0 = requested after that node-addition step

    def bandwidth_mbps(self) -> float:
        return (self.size_bytes * 8) / self.period_us  # bytes*8 bits / us == Mbps

    def to_dict(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "src": self.src,
            "dst": self.dst,
            "flow_class": self.flow_class.value,
            "size_bytes": self.size_bytes,
            "period_us": self.period_us,
            "deadline_us": self.deadline_us,
            "pcp": self.pcp,
            "arrival_order": self.arrival_order,
            "added_at_step": self.added_at_step,
        }

    @staticmethod
    def from_dict(d: dict) -> "Flow":
        return Flow(
            flow_id=d["flow_id"],
            src=d["src"],
            dst=d["dst"],
            flow_class=FlowClass(d["flow_class"]),
            size_bytes=d["size_bytes"],
            period_us=d["period_us"],
            deadline_us=d["deadline_us"],
            pcp=d.get("pcp", 0),
            arrival_order=d.get("arrival_order", 0),
            added_at_step=d.get("added_at_step", 0),
        )
