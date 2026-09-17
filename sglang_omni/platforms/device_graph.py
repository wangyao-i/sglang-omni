# SPDX-License-Identifier: Apache-2.0
"""Graph capture for the model-owned graph paths, one backend per accelerator.

A model that captures its own graphs asks its platform for the backend instead of
naming torch.cuda. The graph object and the capture context's keywords differ per
accelerator, and SGLang resolves them explicitly for the same reason, so the
choice belongs on the platform rather than in a per-model branch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

import torch


class DeviceGraphBackend(Protocol):
    """Records a model-owned graph on one accelerator."""

    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> AbstractContextManager[Any]:
        """Open a capture and yield the graph it records into."""
        ...

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Replay a recorded graph."""
        ...


class CudaDeviceGraphBackend:
    """CUDA, and the backends that present through torch.cuda: HIP and MUSA."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        if allow_host_input_update:
            raise ValueError("CUDA graphs do not support host input updates")
        graph = torch.cuda.CUDAGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(cuda_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        del device
        if host_input_updates is not None:
            raise ValueError("CUDA graphs do not support host input updates")
        graph.replay()


class NpuDeviceGraphBackend:
    """Ascend NPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        graph = torch.npu.NPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        if thread_local_errors:
            kwargs["capture_error_mode"] = "thread_local"
        if allow_host_input_update:
            kwargs["auto_dispatch_capture"] = True
        with torch.npu.graph(npu_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        if host_input_updates is None:
            graph.replay()
            return
        if device is None:
            raise ValueError("NPU host input updates require the graph device")

        # Keep update and replay on the caller's thread. A second host submitter
        # can leave the Ascend graph update waiting indefinitely.
        torch.npu.set_device(device)
        graph.update(cpu_update_input=host_input_updates)
        graph.replay()


class XpuDeviceGraphBackend:
    """Intel XPU."""

    @contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
        allow_host_input_update: bool = False,
    ) -> Iterator[Any]:
        # Note (siju): XPU's graph context declares no capture_error_mode and
        # rejects it as a TypeError, so the request is dropped, not translated.
        del thread_local_errors
        if allow_host_input_update:
            raise ValueError("XPU graphs do not support host input updates")
        graph = torch.xpu.XPUGraph()
        kwargs: dict[str, Any] = {}
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        with torch.xpu.graph(xpu_graph=graph, **kwargs):
            yield graph

    def replay(
        self,
        graph: Any,
        *,
        host_input_updates: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        del device
        if host_input_updates is not None:
            raise ValueError("XPU graphs do not support host input updates")
        graph.replay()


__all__ = [
    "CudaDeviceGraphBackend",
    "DeviceGraphBackend",
    "NpuDeviceGraphBackend",
    "XpuDeviceGraphBackend",
]
