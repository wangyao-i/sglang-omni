# Encoder-owned cached embedding transfer candidate

Status: local candidate, **not hardware qualified**. This document does not
authorize a service launch or consume the remaining hardware opportunity.

## Evidence and bounded hypothesis

The supplied D1 / D1-OFFLINE report places the scheduler in a Host mutex wait
inside `NPUStream::stream()`, called by `graph_task_update_end`, before the CANN
End call. A request-builder thread is in an eventfd wait in the same local
function from a real CPU-to-NPU embedding copy. Identical mutex/repository
ownership is unknown; 451 threads lack native stacks. The absence of GIL or
CANN synchronization frames in the captured subset does not exclude them in
missing threads. Neither the queue-consumer dependency nor the complete cycle
has been established.

Candidate question: does removing request-builder device submission for cached
embeddings eliminate this stall while preserving admission and cache semantics?
Moving work to the encoder worker does not itself prove that a backend queue
dependency has been fixed; the encoder and decoder still execute concurrently.

## Baseline and scope

- SGLang remains `b950878e03f07c63bdca050e8a854503ebfa7058`.
- Omni parent: `424ea1c4b822ebb63b12a3ff7261e9e438096887` (B private encoder
  update stream). Candidate branch: `qwen3-asr/encoder-owned-cache-transfer`.
- No change to graph update/replay order, RLock, events, stream ownership,
  encoder graph capture, dependencies or queue environment variables.
- No D1 monkeypatch hooks carried into this candidate.
- No shipping PR update or merge; previous experiment identities remain intact.

## Implementation contract

Early request-builder cache hits still skip feature extraction. A typed transfer
job retains the selected CPU tensor and enters the existing encoder service
queue; it never invokes the audio tower. The request returns `DeferredAdmission`
even when its ordinary encode-wait policy would block. The worker copies on its
existing private stream, preserves default-consumer allocator registration and
synchronizes before completing the future. Request admission therefore cannot
precede transfer completion. Transfer failures fail the admission future.

Both cache-hit branches inside `submit_item` use the same transfer path,
including the single-flight cache recheck. A single-flight follower publishes
only the leader's already synchronized, consumer-registered device tensor; its
callback performs no `.to()` or `record_stream()` on a caller thread.

Mixed batches transfer cache hits first, then encode only misses; returned
results retain queue order. Existing worker per-item retries remain in use.
Transfer failures cannot force already encoded items with cleared mel tensors
to re-encode. Sources remain held by the transfer jobs through synchronization,
including after cache eviction. Existing service lifecycle checks and future
cancellation handling apply. Stage batch/item/time statistics now include
transfer jobs and must not be interpreted as audio-tower-only work.

## Local checks and limits

Run from this candidate checkout:

```sh
python scripts/npu/test_cache_transfer_ownership.py
python scripts/npu/test_private_encoder_update_stream.py
python <sglang-root>/scripts/npu/test_async_submission.py --omni-root <omni-root>
python -m pytest tests/unit_test/qwen3_asr/test_encoder_service.py tests/unit_test/qwen3_asr/test_request_builders.py -q
```

The first test suite executes the actual service, shared worker and cache source
with SGLang imports substituted and CPU tensors. It checks ownership, readiness,
mixed jobs, failures, closed service, eviction, late cache hits, follower metadata
publication and the request-builder deferred branch. It does **not** execute
NPU stream synchronization, validate device allocator behavior or replace native
tests. The Windows environment lacks `sglang` and `torchaudio`; native suites
cannot collect locally. They remain a preflight requirement on the installed
server environment, without starting a service or workload.

## Remaining hardware decision (not issued here)

After local review and server preflight, separately authorize at most one cold
conc8/140 run with the unchanged all-graph profile and SeedTTS evaluation hash
`9f631ab78d8bf3e19ab82a9c099826a850a0f103ae08d96db62561a71ec14815`.
Pin the candidate full SHA, imported file hashes and clean tracked state first.
Keep independent no-progress detection and bounded cleanup. Do not spend a run
merely to repair hooks or collect a second baseline.

Success requires 140/140 complete responses, complete correctness evaluation,
no timeout/error/stall and explicit graph-path evidence. Capture logs alone do
not qualify replay; missing counters remain unknown. A single passing run is a
bounded gate, not general stability proof or proof of the precise backend root
cause. Failure rejects this ownership change as a sufficient mitigation; retain
evidence and stop. No automatic retries, alternative queue flags or further
lock/order experiments are authorized.
