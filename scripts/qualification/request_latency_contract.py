"""Canonical fixed-cardinality request latency field names."""

from __future__ import annotations


LATENCY_PHASE_NAMES = (
    "actor_queue",
    "actor_admission",
    "tokenization",
    "prefill",
    "decode",
    "actor_cycle_idle",
    "sampling",
    "readback",
    "response_delivery",
    "handler_queue",
    "client_delivery",
    "gpu_lock_wait",
    "graph_capture",
    "graph_replay",
    "synchronization",
    "resize",
    "trim",
    "adapter",
    "training",
    "unexplained",
)
LATENCY_PHASE_FIELDS = tuple(f"{phase}_ms" for phase in LATENCY_PHASE_NAMES)

LATENCY_STALL_REASON_FIELDS = (
    "actor_queue",
    "actor_admission",
    "actor_prefill",
    "actor_decode",
    "actor_cycle_idle",
    "response_delivery",
    "handler_queue",
    "client_delivery",
    "sampling",
    "readback",
    "gpu_lock_wait",
    "graph_capture",
    "graph_replay",
    "synchronization",
    "resize",
    "trim",
    "adapter",
    "training",
    "unexplained",
)
