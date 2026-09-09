# SPDX-License-Identifier: Apache-2.0
"""Small process-local guards for device work shared by host threads."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any


class FairDeviceExecutionGuard:
    """Serialize device submissions in FIFO order across host threads.

    A condition/ticket lock is used instead of ``threading.Lock`` so a fast
    generation loop cannot repeatedly reacquire the device while a background
    encoder batch is already waiting.
    """

    def __init__(self, *, completion_fence: Callable[[], None] | None = None) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._completion_fence = completion_fence
        self._stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "acquire_count": 0,
                "wait_total_ns": 0,
                "wait_max_ns": 0,
                "hold_total_ns": 0,
                "hold_max_ns": 0,
            }
        )

    @contextlib.contextmanager
    def hold(self, *, label: str = "default") -> Iterator[tuple[int, int]]:
        """Yield ``(ticket, wait_ns)`` after acquiring exclusive execution."""
        wait_started_ns = time.monotonic_ns()
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
            acquired_ns = time.monotonic_ns()
            wait_ns = acquired_ns - wait_started_ns
            stats = self._stats[label]
            stats["acquire_count"] += 1
            stats["wait_total_ns"] += wait_ns
            stats["wait_max_ns"] = max(stats["wait_max_ns"], wait_ns)
        try:
            yield ticket, wait_ns
        finally:
            try:
                if self._completion_fence is not None:
                    self._completion_fence()
            finally:
                # Never strand later tickets when a diagnostic fence reports a
                # device error. The current caller still observes that error.
                with self._condition:
                    hold_ns = time.monotonic_ns() - acquired_ns
                    stats = self._stats[label]
                    stats["hold_total_ns"] += hold_ns
                    stats["hold_max_ns"] = max(stats["hold_max_ns"], hold_ns)
                    self._serving_ticket += 1
                    self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return a lock-consistent, JSON-ready aggregate timing snapshot."""
        with self._condition:
            labels = {
                label: {
                    "acquire_count": values["acquire_count"],
                    "wait_total_ms": values["wait_total_ns"] / 1e6,
                    "wait_max_ms": values["wait_max_ns"] / 1e6,
                    "hold_total_ms": values["hold_total_ns"] / 1e6,
                    "hold_max_ms": values["hold_max_ns"] / 1e6,
                }
                for label, values in sorted(self._stats.items())
            }
            return {
                "next_ticket": self._next_ticket,
                "serving_ticket": self._serving_ticket,
                "outstanding": self._next_ticket - self._serving_ticket,
                "labels": labels,
            }
