# Qwen3-ASR Omni-only repair — validation O1.1

Status: candidate, not hardware qualified. This packet does not authorize
dependency edits, installation, device reset, merge, or unattended NPU runs.
Native CPU preflight has passed on the pinned pair below. Obtain operator
approval before proceeding to the separately bounded hardware gate below.

## Fixed identities and hypothesis

- Omni code: `555c6906dd38078d128267b7d4f9d34975c3d55c`, branch
  `qwen3-asr/omni-owned-graph-submission`. A later packet-only commit is allowed;
  record both HEAD and this production-code identity.
- SGLang: upstream v0.5.19 `0bcd822377da7b5718e674eaf9c870d349424dd1`, clean,
  **without #40059 or any local decoder/graph modifications**. This is the
  corrected pinned target preserves the server's release lineage; O1's earlier
  db39b7f961 main baseline is superseded, not an instruction to change versions.
- Keep the existing reported environment: Python 3.11.10, torch 2.10.0,
  torch_npu 2.10.0.post2, CANN 9.0.1 and the same assigned Ascend hardware.
  If the baseline is unavailable or differs, report before changing anything.

Question: are encoder-private update transactions plus encoder-owned cache H2D
sufficient when decoder uses its original update-thread/replay/join path?
The six old successful runs used patched SGLang and do not answer this question.
No SGLang monkeypatch, queue-flag workaround or graph-disable fallback is allowed.
Only Omni owns the repair. Failure rejects sufficiency on this pinned pair;
retain evidence instead of automatically restoring #40059 and calling it fixed.

## CPU preflight — run first, no serving workload

Use clean checkouts and preserve existing working services. A shared dependency
environment is permitted without installs/edits if actual import resolution is
verified and the launch environment is preserved; a separate venv is not required.
Record Git HEAD/status, Python executable, actual imported package paths,
torch/torch_npu/CANN/driver identities and SHA256 for the imported encoder runner,
encoder service, request builder, graph backend, SGLang NPU graph runner/backend,
and device stream helpers. Verify the imported SGLang files match the upstream
commit, not just the shell cwd or package version. Check no old diagnostic
startup hooks or tracing/serialization environment settings are active.

From the Omni candidate with the unmodified SGLang environment selected:

```bash
python -m pytest \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  tests/unit_test/qwen3_asr/test_request_builders.py \
  tests/unit_test/platforms/test_device_graph.py \
  -q -m 'not accelerator and not benchmark'
```

Require successful collection and all selected cases passing. Return exact
counts, deselections/skips with reasons, command and identities. In particular,
the new 5 submission-failure cases, partial-capture restoration, 3 terminal
service cases and 4 cached-transfer cases must execute. Do not use the old
4/4 cached-transfer receipt as qualification for this candidate. Do not install
or modify SGLang to make collection pass; return the first incompatibility.

Local evidence: 17 source-isolated CPU cases and targeted pre-commit passed.
Native collection on the author's Windows host failed because sglang/torchaudio
are unavailable; this is not a statement about the server environment.
Source-isolated tests replace import boundaries and do not qualify native APIs,
NPU synchronization, graph execution, or installed dependency compatibility.

O1 server receipt: 562 passed, 1 XPU API skip, 2 accelerator deselections,
14 warnings in 15.32 seconds; targeted repeat 14/14 passed. The six reported
Omni/SGLang runner/backend file hashes independently match committed Git blobs.
CPU results are operator-reported, not independently rerun on the server.
No CPU rerun is needed solely for this packet-only correction.

## Import-resolution guard for the hardware launcher

The default editable install resolves to b950878e03, which contains the earlier
SGLang async patch and is NOT an acceptable Omni-only baseline. O1 prepended
the clean 0bcd8223 checkout's python directory to PYTHONPATH; retain that exact
override and the remaining environment (including tbe paths) in the service
launcher and inherited worker environment. Do not replace the entire PYTHONPATH
or reinstall packages. Checking a different interactive shell is insufficient.

Before client load, verify from the actual model-worker context (existing
startup diagnostics/debug facility, not a production monkeypatch) the imported
SGLang and Omni roots and critical source digests. If that evidence cannot be
obtained with existing facilities, stop before load and report the precise gap;
do not silently substitute the successful pytest process's import identity.
Retain the full launch command/environment mapping in server-owned evidence.

## Hardware gate — only after CPU preflight and operator approval

Use the retained launcher and evaluation harness, recording their full commands
and digests rather than inventing new server paths. Preserve the existing
model, weights, generation parameters, device assignment, seed and cache policy.

- Compile disabled; encoder graph enabled; prefill breakable; decode full;
  `disable_cuda_graph=false`, pre-LM encoder worker active,
  `max_running_requests=64`, `cuda_graph_max_bs=64`.
- Record effective parsed values, actual process imports, queue environment and
  run output locations. No #40059 marker is expected on this unmodified decoder.
- Frozen SeedTTS EN 140 input digest:
  `9f631ab78d8bf3e19ab82a9c099826a850a0f103ae08d96db62561a71ec14815`.
  Verify the retained ordered input, not a replacement corpus or prefix match.
- First run: fresh process, cold concurrency 8 / 140, no warmup. At most one
  initial run; stop on any failure. Two further fresh repetitions may be
  separately approved after this first discriminating result. No performance
  claim or statistical long-run-stability claim follows from these checks.
- Allow at most 15 minutes for readiness, 10 minutes for the workload and
  60 seconds without a completed response (including the first). Enforce with
  an independent watchdog. Stop new load on the first failure.
- Require 140/140 HTTP success, no empty/failed/high-WER samples, zero per-sample
  disagreement against the retained reference and corpus WER 0.011373. Preserve
  raw outputs for independent recomputation; do not waive correctness to pass.
- Require positive execution evidence for encoder/prefill/decode graph paths
  and cache miss/hit admission paths. Capture messages alone do not attest
  replay, and absence of warnings is not a zero-fallback counter. If the existing
  evidence mechanism cannot attest a required path, report unknown before
  expanding repetitions; do not add runtime hooks during a qualification run.
- Require normal owned-process shutdown and return to the recorded resource
  baseline. Forced cleanup is a failure/diagnostic outcome, not a passing run.
  Preserve unknown/shared processes; never reset the device or kill by broad name.

On a failure retain first error, progress, imports/configuration, all existing
logs and cleanup result. No automatic order/lock/queue variants or reruns.
Decide the next diagnostic only after analyzing which dependency actually failed.

## Return

O1 stage (CPU or hardware), code/dependency identities and clean-state checks;
actual commands; test counts or request counts; correctness and graph/cache path
evidence; timeouts/errors; cleanup; retained server artifact identifiers and
redacted report digest. Mark unknowns explicitly. Raw server data stays on server.
