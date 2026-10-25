# Concurrent GLM execution

This extends the existing cache-rocm branch. The upstream integration is at 1a0b56cec (100 commits beyond the earlier 8baabd53b base); concurrency behavior comes from production source 85ce7b07d and is adapted to the newer batch, scheduler and multimodal APIs. The public port has its own validation; production-lineage measurements must not be treated as measurements of a different build.

## Execution model

Ordinary server slots can be combined into one decode and still traverse layer-split GPUs serially. This implementation gives GLM5NEXT requests independent execution contexts, schedulers, graph buffers and workers while sharing model weights and physical KV storage. Independent requests can occupy different parts of the layer pipeline at the same time.

Each request still obeys its own layer and token dependencies. Sampling a completed lane can submit its next token while another lane is prefilling. Shared state planning, cache mutation and final release retain coordinator boundaries; this is not a fully asynchronous distributed scheduler. The continuation window is bounded to 512 iterations.

The six-worker reference configuration has capacities 512/512/1/1/1/1. Two workers accept substantial prefills; idle prefill workers can also decode. Logical request slots and worker workspaces are separate. Priority is respected, with available capacity eligible for lower-priority work. Larger batches consume additional workspace; they are not free just because KV storage is shared.

This path is specific to GLM5NEXT with unified KV and supported adapters/output modes. Custom expert-parallel experiments remain disabled. It does not prune experts, offload experts to CPU, speculate across future tokens, or create two NUMA-local model replicas. Multimodal preparation and causal media chunks can overlap other work; unsupported media geometry retains the ordinary helper path.

## Shared state and capacity

Matching prompts can share physical prefix cells. The physical context budget counts unique resident cells, while each request retains its own logical context and output reservation. Capacity accounting includes idle GPU heads; reclaiming idle device residency preserves RAM/SSD checkpoints. Active-capacity pressure must defer restore without invalidating a valid checkpoint.

For simultaneous cold requests with a common prefix, one producer can publish reusable pages while followers wait outside execution slots. Matching includes canonical token/media identity and adapter eligibility. A follower's priority can raise the producer's execution priority. Cancellation releases the dependency and permits a successor to reuse already published pages. Only complete eligible pages are reusable; a shared head shorter than a page does not guarantee a hit.

Native progress includes shared-prefix waiting. The private hosting scheduler has additional admission/UI integration that is not part of this repository. Native execution, cache storage, request controls and the CPU tests below do not require that hosting service.

## Configuration and observability

The root README includes the full six-MI50 example, including unified KV, worker capacities, stream priorities and transfer settings. These are hardware-specific settings, not universal defaults. Without opt-in execution lanes, normal server behavior remains available.

`POST /v1/internal/cache-probe` is a read-only snapshot with `execution_memory`, `execution_timing`, worker profiles and cache counters. Memory observations distinguish shared storage from per-worker compute allocations. GPU utilization alone does not distinguish useful overlap, synchronization waits and bandwidth saturation.

Optional bounded expert instrumentation:

```sh
export LLAMA_GLM5NEXT_ROUTE_STATS=1
export LLAMA_GLM5NEXT_ROUTE_STATS_DIR=/var/cache/llama/expert-stats
```

Use a writable directory owned by the server user. Collectors report numeric expert activations and router weights per worker. Compare deltas within a single process lifetime and separate benchmark traffic from production traffic. Low activation frequency alone does not establish that an expert can be pruned safely.

## CPU reproduction

Use a CPU-only build and synthetic model. Do not point these tests at a production service. The HTTP tests own isolated loopback ports 18087 and 18088, and terminate only the server they start. Run them sequentially with those ports free. Python 3 and pytest are needed for the storage suite.

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target llama-server test-fork-llama-archs \
  test-backend-owned-input test-moe-route-stats -j 4
build-cpu/bin/test-backend-owned-input
build-cpu/bin/test-moe-route-stats
build-cpu/bin/test-fork-llama-archs -a glm5next -s 42
mkdir -p /tmp/glm53-concurrency-fixture
build-cpu/bin/test-fork-llama-archs -a glm5next -s 42 \
  -o /tmp/glm53-concurrency-fixture
python3 tools/server/tests/unit/test_glm53_expanded_concurrency.py \
  --binary "$PWD/build-cpu/bin/llama-server" \
  --fixture /tmp/glm53-concurrency-fixture/glm5next-moe.gguf \
  --slots 6 --heterogeneous --owner-ubatch 16 --logical-batch 8192 \
  --worker-batches 512,512,1,1,1,1 --require-worker-profile \
  --require-transport-profile --require-route-buffer --log /tmp/glm-six-lane.log
python3 tools/server/tests/unit/test_glm53_prefix_flight.py \
  --binary "$PWD/build-cpu/bin/llama-server" \
  --fixture /tmp/glm53-concurrency-fixture/glm5next-moe.gguf \
  --log /tmp/glm-prefix-flight.log
```

The suites check exact serial output equivalence on the fixture, shared heads, parking/resume, cancellation isolation, decode progress beside prefill, resident-KV pressure, checkpoint restore and prefix-producer cancellation. These are functional checks, not full-model quality tests or GPU throughput measurements. They do not establish that QPI is or is not the remaining bottleneck.


For the synthetic image regression, use the same binary as both inputs. The
baseline process explicitly disables execution lanes; the candidate uses the
requested workers. The mock vision endpoint supplies deterministic embeddings,
not a real vision model.

```sh
LLAMA_GLM5NEXT_EXECUTION_LANES=3 LLAMA_GLM5NEXT_PREFILL_WORKERS=2 \
LLAMA_GLM5NEXT_WORKER_BATCHES=512,512,1 LLAMA_GLM5NEXT_PREFILL_BATCH=512 \
python3 tools/server/tests/unit/test_glm53_media_fairness.py \
  --binary "$PWD/build-cpu/bin/llama-server" \
  --baseline "$PWD/build-cpu/bin/llama-server" \
  --fixture /tmp/glm53-concurrency-fixture/glm5next-moe.gguf \
  --log /tmp/glm-media.log
```

The HTTP fixtures explicitly override their tiny model's advertised context to
4096 positions. This permits long synthetic sequences under the newer server's
training-context cap; it is not a recommendation to override real model limits.

## Initial public integration validation (d9595f1fc, 2026-10-25)

The integrated source was built in Release mode with HIP and CUDA disabled.
Checks included the GLM fork architecture suite, the upstream Llama architecture
suite, deferred mock-GPU input ownership and transfer ordering, and bounded
router statistics.

The six-worker HTTP fixture used 512/512/1/1/1/1 workers and a 16-token owner
microbatch. It reached six active requests, matched serial token output, moved
requests between workers, parked requests for capacity, isolated cancellation,
and completed 651 decode tokens while prefills were pending. Its shared-head
check used 556 physical cells for 3116 logical cells across six active sequences.
These counts describe one synthetic CPU run, not a throughput benchmark.

The prefix-dependency fixture reused all 1536 shared tokens before evaluating an
8-token tail. After producer cancellation, its successor reused 384 already
published tokens. A 1500-token checkpoint restored 1499 tokens under resident
capacity pressure and beside an active request, with no cache failure. The
restart fixture retained exact output and reused 95 of 96 tokens after both a
RAM-preserving restart and SSD-only recovery.

All eight isolated storage tests passed. The synthetic media suite matched
serial outputs for image/text and image/image execution, advanced decode during
partial image prefill, and prepared queued media with three occupied slots.
An identical image reused 1193 tokens; changed image content reused only the
128-token text prefix.

This publication did not replace the production server or run the integrated
public source on the production GPUs. Prior MI50 measurements apply to the
production source lineage, not to this newly integrated binary.


## Upstream synchronization (2026-10-25)

Upstream `1a0b56cec73dac938e0602c974b21dc8e2862830` adds 100 commits after
the previous upstream base. This is a history-preserving merge into `cache-rocm`.
The integration retains the fork's GLM execution workers, cache dependencies,
external-media adapter and instrumentation.

Conflict resolution preserves upstream's chunked CUDA graphs, fused top-k mask,
new model architectures, tokenizer whitespace behavior and server role split.
The fork's graph diagnostics and internal cache routes follow the new APIs.
DeepSeek 4.1 attention calls explicitly select the ordinary, non-Turbo3 basis.
The fork RPC patch version is incremented for the combined operation table.

ROCm 6.3.3 / gfx906 host-and-device syntax checks passed for `ggml-cuda.cu`,
`mmq.cu`, `top-k-mask.cu` and `topk-moe.cu`, with HIP graphs enabled.
These are compile checks, not GPU execution or full HIP link validation.


The merged Release CPU build passed the fork GLM architecture/state suite,
upstream Llama and DeepSeek 4.1 architecture fixtures, deferred mock-GPU input
ownership/transfer ordering, bounded router statistics, and all 28 upstream
TOP_K_MASK cases. All eight storage regressions passed.

The six-worker HTTP suite retained exact serial output, worker reassignment,
capacity parking, cancellation isolation and shared-head reuse (556 physical
cells for 3116 logical cells). Prefix-producer cancellation/reuse, image/text
and image/image overlap, queued media preparation, image cache identity, and
RAM-preserving/SSD-only restart recovery also passed. This validation used only
synthetic CPU fixtures; production runtime and GPU qualification are unchanged.
