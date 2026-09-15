# Qwen3-ASR NPU host-input graph update handoff

## Status and decision

- Status: **核心机制已通过；聚焦测试兼容项待复验**（诊断分支，不是
  #2160 交付结论）。
- Root-cause owner: local Qwen3-ASR graph-key workstream.
- Single causal question: on one encoder token bucket, can `NPUGraph.update()`
  replace the Ascend attention host-side cumulative window boundaries between
  layout A and layout B while both graph results remain numerically aligned
  with eager execution?
- Decision unlocked by a pass: an NPU encoder graph key may use the token bucket
  instead of the complete effective window layout.
- Decision on a failure: retain the exact-layout key and classify the first
  failure before attempting another variant.

## Gate 0 contract

| Fact | State | Contract |
|---|---|---|
| SGLang-Omni candidate | known | `codex/qwen3-asr-npu-host-input-update-probe`, code commit `a851d1f886c3ef6e72de215000d71c13d2ecf278` |
| Candidate ancestry | known | `1638c5dd` (#2160 head) -> `757c2fbe` -> `e336149b` -> `a851d1f8` |
| SGLang dependency | hard constraint | release `v0.5.19`, actual commit `0bcd822377da7b5718e674eaf9c870d349424dd1` |
| Hardware | hard constraint | one idle Ascend 910C device; do not substitute 910B evidence |
| Server node identity and topology | server must verify | keep hostname/address private; return only hardware model, visible-device count, and process-to-device mapping |
| Runtime versions | server must verify | exact Python, CANN, `torch`, and `torch_npu` versions |
| Model/data | hard constraint | repository's synthetic two-layer accelerator test only; no private audio or model weights are required |
| Checkout | hard constraint | use a separate worktree/process; do not modify or stop the #2016 all-graph/realtime validation checkout |
| Network/data boundary | hard constraint | no private paths, hostnames, addresses, credentials, raw logs, or input contents leave the server |
| Remote availability | known | `origin/codex/qwen3-asr-npu-host-input-update-probe` was published at `fcaa07b9b51975ece8ec65dfde0a5c6d9c61a8c4` |

The server agent first signs this contract by returning the actual checkout,
HEAD, clean/dirty state, dependency HEAD, runtime, hardware, and whether the
isolated worktree requirement can be met. Any material mismatch stops the gate.

## Scope

In scope:

- `auto_dispatch_capture=True` for the model-owned NPU graph;
- concurrent `graph.update(cpu_update_input=...)` and `graph.replay()` with the
  target NPU selected inside the update thread;
- updating both `actual_seq_lengths` and `actual_seq_lengths_kv`;
- reuse of one captured graph by two distinct layouts in the same token bucket;
- eager/graph parity for both layouts.

Out of scope:

- end-to-end service, realtime transport, throughput, latency, concurrency,
  cache eviction or capacity tuning, all-graph qualification, and changes to
  the #2160 delivery branch. The existing global `max_graphs` entry cap remains;
  because entries are now bucket-keyed, exhaustion keeps its eager-fallback
  behavior but the cap mechanically counts buckets rather than exact layouts;
- server-side compatibility guards, silent eager fallback, or edits to
  installed packages.

## Prior art and local evidence

SGLang `v0.5.19` uses `auto_dispatch_capture=True` and pairs host input update
with replay in a background thread in
`npu_cudagraph_backend.py::replay_with_input_update`. The candidate adapts that
ordering in the existing SGLang-Omni `DeviceGraphBackend`; it does not copy the
LLM graph runner or dynamically probe for the interface.

Local evidence at `a851d1f8`:

- Black 24.10 check: passed for the four changed Python files.
- Ruff: passed for the four changed Python files.
- `py_compile`: passed for the four changed Python files.
- device-graph backend tests: `14 passed` using a package-isolation shim so the
  backend module can be tested without importing Linux-only SGLang runtime.
- `git diff --check`: passed.
- Full Qwen3-ASR test collection: not run locally. Windows cannot import the
  Linux/GPU SGLang stack (`resource`, then `triton` are unavailable).

These results prove the control-flow contract only. They do not prove that
`torch_npu` updates all captured attention nodes or that numerical results are
correct on 910C.

## 910C result at `fcaa07b9`

The server reported the following exact matrix:

- SGLang-Omni `fcaa07b9b51975ece8ec65dfde0a5c6d9c61a8c4` on
  `codex/qwen3-asr-npu-host-input-update-probe`;
- SGLang `0bcd822377da7b5718e674eaf9c870d349424dd1`, clean;
- `torch 2.10.0`, `torch_npu 2.10.0.post2`, CANN 9.0.1, Ascend 910C;
- both focused encoder tests passed;
- `test_graph_matches_eager_tower` passed in 14.47 seconds: `[500]` captured
  the graph, `[450]` reused the same bucket entry, both stayed below the
  predeclared `3e-2` maximum absolute error, and update/replay completed with
  no eager fallback, capture failure, ACL error, or device error.

This closes the single causal question: on this exact runtime matrix, host-side
window boundaries can change while one bucket-keyed encoder graph is reused and
remains aligned with eager execution.

Two qualification deviations remain explicit:

1. `test_device_graph.py` produced 13 passes and one failure because this NPU
   PyTorch distribution has no `torch.xpu.graph`. The failing test combined
   CUDA and XPU API introspection even though the XPU API is optional in that
   distribution. The repository-owned fix splits the assertions and skips only
   the unavailable XPU introspection; the NPU mechanism code is unchanged.
2. The device had a concurrent workload, so the server did not perform an
   attributable HBM observation. Synchronization completed, but this run is not
   memory-health, stability, service, concurrency, or performance evidence.

Because Gate 1 failed before Gate 2, the server execution did not follow the
predeclared first-failure stop rule. The later mechanism result is retained as
valid diagnostic evidence for the causal question, but the complete task is not
recorded as an unconditional qualification pass. On a descendant containing
only the test/documentation correction, rerun the focused contract tests from a
fresh process. The 910C mechanism gate need not be repeated unless production
code, dependency HEAD, runtime, or hardware changes.

## Mode audit

| Mode/change | Source | Needed for this question | Last known basis | Removal/review condition |
|---|---|---:|---|---|
| Encoder layer-stack graph | #2160 | yes | exact-layout graph candidate | retain independent of this diagnostic result |
| NPU lazy capture | #2160 | yes | avoids synthetic NPU layouts at startup | review separately; not varied here |
| `auto_dispatch_capture` | SGLang v0.5.19 NPU runner | yes | mature host-input update path | remove if the 910C gate fails |
| bucket-only NPU key | this diagnostic candidate | yes, sole policy change | passed the `fcaa07b9` 910C mechanism gate | eligible for a separate delivery change after focused-test closure and current-head review |
| all-graph/realtime modes | separate #2016 validation line | no | separate branch and evidence | must stay disabled/outside this probe |

## Execution and stop policy

Run [the companion task](qwen3_asr_npu_host_input_update_task.md) in order.
Stop at the first collection, capture, update, replay, parity, timeout, ACL, or
device-health failure. Preserve the first complete traceback and the raw log on
the server. Do not continue into service or performance tests.

Pass requires all of the following on the exact heads above:

1. focused backend tests pass;
2. the synthetic encoder accelerator test completes from a fresh process;
3. layouts `[500]` and `[450]` use the same bucket and do not increase graph
   registry size on the second call;
4. each layout's graph output has `max_abs_diff < 3e-2` versus eager;
5. no eager fallback, hang, update/replay exception, ACL/device error, or
   unhealthy device state occurs.

## Result template

```text
Task: Qwen3-ASR NPU host-input graph update
Status: passed / failed / blocked

SGLang-Omni branch / before HEAD / tested HEAD / worktree state:
SGLang branch or tag / actual HEAD / worktree state:
Python / CANN / torch / torch_npu:
Hardware model / visible devices / process-to-device mapping:

Gate 0 differences:
Focused backend tests:
Accelerator test:
Layout A [500] bucket / graph count / max_abs_diff:
Layout B [450] bucket / graph count / max_abs_diff:
Target capture/update/replay evidence:
Fallback count:
Forbidden error count:
Post-run device health:

First complete failure and classification (if any):
Server-only artifact directory:
Server repair commit (normally none):
Recommended next single experiment:
```

Keep raw logs and environment dumps in the server-only artifact directory.
Return only this redacted summary and the smallest necessary error excerpt.
