# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

from sglang_omni.utils.execution_guard import FairDeviceExecutionGuard


def test_fair_device_execution_guard_serves_waiters_in_ticket_order() -> None:
    guard = FairDeviceExecutionGuard()
    first_entered = threading.Event()
    release_first = threading.Event()
    order: list[int] = []

    def run(worker_id: int) -> None:
        with guard.hold():
            order.append(worker_id)
            if worker_id == 0:
                first_entered.set()
                assert release_first.wait(timeout=2)

    first = threading.Thread(target=run, args=(0,))
    first.start()
    assert first_entered.wait(timeout=2)

    waiters = [threading.Thread(target=run, args=(worker_id,)) for worker_id in (1, 2)]
    waiters[0].start()
    # Starting the second waiter only after the first is blocked makes ticket
    # assignment deterministic without relying on scheduler timing.
    while guard._next_ticket < 2:
        pass
    waiters[1].start()
    release_first.set()

    first.join(timeout=2)
    for waiter in waiters:
        waiter.join(timeout=2)
    assert not first.is_alive()
    assert all(not waiter.is_alive() for waiter in waiters)
    assert order == [0, 1, 2]


def test_completion_fence_runs_before_the_next_ticket() -> None:
    order: list[str] = []
    guard = FairDeviceExecutionGuard(
        completion_fence=lambda: order.append("fence")
    )

    with guard.hold():
        order.append("first")
    with guard.hold():
        order.append("second")

    assert order == ["first", "fence", "second", "fence"]


def test_completion_fence_failure_does_not_strand_later_tickets() -> None:
    calls = 0

    def fence() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("device fence failed")

    guard = FairDeviceExecutionGuard(completion_fence=fence)

    with pytest.raises(RuntimeError, match="device fence failed"):
        with guard.hold():
            pass
    with guard.hold() as (ticket, _wait_ns):
        assert ticket == 1


def test_guard_snapshot_reports_labeled_balanced_intervals() -> None:
    guard = FairDeviceExecutionGuard()

    with guard.hold(label="generation_decode_graph"):
        pass
    with guard.hold(label="encoder"):
        pass

    snapshot = guard.snapshot()
    assert snapshot["next_ticket"] == 2
    assert snapshot["serving_ticket"] == 2
    assert snapshot["outstanding"] == 0
    assert snapshot["labels"]["generation_decode_graph"]["acquire_count"] == 1
    assert snapshot["labels"]["encoder"]["acquire_count"] == 1
    assert snapshot["labels"]["generation_decode_graph"]["hold_total_ms"] >= 0


def test_barrier_only_mode_does_not_serialize_but_keeps_the_fence() -> None:
    # 910C-066 bench: no mutual exclusion, one barrier per handoff, and a
    # snapshot that stays empty so an operator can attest serialization was off.
    fence_calls = 0

    def fence() -> None:
        nonlocal fence_calls
        fence_calls += 1

    guard = FairDeviceExecutionGuard(completion_fence=fence, serialize=False)
    both_inside = threading.Barrier(2, timeout=5)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with guard.hold(label="encoder"):
                # Serialized callers could never satisfy this barrier.
                both_inside.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the asserts.
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert fence_calls == 2
    snapshot = guard.snapshot()
    assert snapshot["next_ticket"] == 0
    assert snapshot["serving_ticket"] == 0
    assert snapshot["outstanding"] == 0
    assert snapshot["labels"] == {}
