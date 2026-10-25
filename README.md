# llama.cpp: concurrent GLM inference, tiered checkpoints and gfx906 tuning

An independent fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for running large models on older AMD GPUs, retaining useful prompt state across requests and model unloads, and making cache behavior observable. Take the ideas, adapt the code, or use the fork directly under the existing [MIT license](LICENSE).

The additions were developed around GLM-5.3-Flash on six Radeon Instinct MI50 cards. This is a research and deployment fork, not a proposal that upstream adopt the entire patch set. Upstream attribution and license notices are retained. AI coding assistance was used in developing and integrating this fork.

See [concurrent execution](docs/CONCURRENT_GLM_INFERENCE.md) for design, configuration, tests and limitations.

## What is different

- **Independent execution lanes.** Six GLM workers share weights and KV storage while owning separate execution state. Two substantial prefills can overlap with decode; idle prefill workers can decode too.
- **Shared-prefix dependencies.** Followers can wait for an active producer to publish matching pages. Capacity checks include idle resident heads and preserve valid checkpoints under active pressure.
- **Expert instrumentation.** Bounded per-worker router counts and weights support workload analysis without recording prompt text.
- **Range checkpoints, not just whole prompts.** Shared prefix nodes and branch tails retain target attention, indexer and recurrent state, with exact token/media boundaries and optional draft state.
- **Persistent RAM plus asynchronous SSD backing.** Linux tmpfs mappings survive the server process. A bounded background writer backs immutable payloads on disk; recovery verifies descriptors and loads cold payloads when needed.
- **Explicit cache lifecycle.** Requests can identify a branch, distinguish interactive/background work, and choose retained, ephemeral or one-shot behavior. Age, bounded reuse frequency and recomputation cost influence eviction; shared ancestors and active leases remain protected.
- **ROCm/gfx906 work.** HIP peer-transfer controls, scheduler alias reuse, model-specific attention and quantized matrix kernels, and transport instrumentation. These optimizations are workload-dependent.
- **Additional experimental paths.** GLM conversion/model support, Turbo3 cache experiments, expert-parallel fixtures, hidden-state extraction and an optional external vision embedding bridge. Presence in the tree does not mean a feature is enabled in the reference profile.

The cache is implemented in `llama-server`. TGWUI is an optional frontend; a client can send `cache_control` directly to the native HTTP API. The examples below do not require a private scheduler or repository.

## Reference configuration

This is the configured deployment profile qualified through 2026-10-24, not a claim that every setting is optimal elsewhere. The integrated public source is tested separately from the production build; historical GPU measurements are not a performance guarantee for this port or other hardware. Hardware: two Xeon E5-2697A v4 CPUs (32 physical cores / 64 threads total), approximately 128 GiB system RAM, six 32 GiB MI50/gfx906 GPUs across two NUMA nodes, and local SSD storage. The established toolchain is ROCm 6.3.3.

| Setting | Reference value |
|---|---|
| Model | GLM-5.3-Flash UD-IQ4_XS, five local GGUF shards |
| Projector | GLM-5.3-Flash F16 |
| Context / simultaneous slots | 266240 shared physical positions / 6; 204800 per-request cap |
| GPU layers | 47 |
| CPU threads / batch threads | 48 / 48 |
| Logical batch / coordinator microbatch | 8192 / 16 |
| Worker microbatches | 512, 512, 1, 1, 1, 1 |
| KV cache / flash attention | F16 / enabled |
| Tensor split | `1.2,0.85,0.9,1.05,0.8,1.2` |
| Explicit placement | Layer 45, token embeddings and layer 31 down experts on `ROCm4`; projector on `ROCm2` |
| Speculative decoding | Disabled (`--spec-type none`) |
| Memory mapping / NUMA | `--no-mmap` / `--numa distribute` |
| Automatic fitting | Disabled |
| Whole-prompt RAM cache | Disabled (`--cache-ram 0`) |
| Range-page RAM budget / spacing | 98304 MiB / 2048 tokens |
| SSD budget / free-space reserve | 384 GiB / 64 GiB |
| SSD pending-write budget | 1 GiB |

GPU indices and layer placement describe one physical topology. Re-measure them on your machine. The RAM budget is capped by tmpfs capacity with metadata headroom, and other RAM users still matter. The context value is a configured limit, not a general hardware limit.

### Build

See [upstream build instructions](docs/build.md) for dependencies and other platforms. A HIP build for the reference toolchain starts with:

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-6.3.3/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.3/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.3 \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP_RCCL=ON
cmake --build build-hip --target llama-server -j 8
```

Use a matching RCCL installation when enabling RCCL. Build into a new directory; do not mix libraries from unrelated revisions. A CPU build can exercise the storage contract without a model or GPU allocation:

```sh
cmake -S . -B build-cpu -DGGML_HIP=OFF -DGGML_CUDA=OFF
cmake --build build-cpu --target llama-server -j 8
LLAMA_PAGE_RAM_TEST_LIB_DIR="$PWD/build-cpu/bin" \
  python3 -m pytest --confcutdir tools/server/tests/unit \
  tools/server/tests/unit/test_page_ram.py -q
```

### Model-side tuning

These are the reference profile's environment overrides:

```sh
export LLAMA_GLM5NEXT_EXECUTION_LANES=6
export LLAMA_GLM5NEXT_PREFILL_WORKERS=2
export LLAMA_GLM5NEXT_WORKER_BATCHES=512,512,1,1,1,1
export LLAMA_GLM5NEXT_PREFILL_BATCH=512
export GGML_CUDA_DISABLE_GRAPHS=1
export GPU_MAX_HW_QUEUES=32
export GGML_SCHED_READY_COPY_MIN_BYTES=1
export LLAMA_EXECUTION_READY_READBACK=1
export LLAMA_EXECUTION_GPU_PRIORITY=1
export GGML_HIP_NO_PEER_STAGING=1
export GGML_HIP_NO_PEER_STAGING_MAX_BYTES=33554432
export GGML_HIP_NO_PEER_STAGING_TIMING=0
export GGML_SCHED_ALIAS_COPIES=1
export GGML_SCHED_ALIAS_SUBVIEWS=1
export LLAMA_GLM5NEXT_CONSUMER_HC_NORM=1
export LLAMA_GLM5NEXT_TURBO3_MLA=0
export LLAMA_GLM5NEXT_SPARSE_SEGMENTED=1
export LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK=14
export LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER=0
```

With local model/projector paths substituted:

```sh
build-hip/bin/llama-server \
  --model /models/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  --mmproj /models/mmproj-GLM-5.3-Flash-F16.gguf \
  --ctx-size 266240 --parallel 6 --kv-unified --kv-unified-per-slot 204800 --n-gpu-layers 47 \
  --threads 48 --threads-batch 48 --batch-size 8192 --ubatch-size 16 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --tensor-split 1.2,0.85,0.9,1.05,0.8,1.2 \
  --override-tensor '(blk\.45\..*|token_embd\.weight)=ROCm4' \
  --override-tensor 'blk\.31\.ffn_down_exps\.weight=ROCm4' \
  --mmproj-device ROCm2 --spec-type none --fit off \
  --no-mmap --numa distribute \
  --cache-ram 0 --cache-page-ram 98304 --cache-page-tokens 2048
```

### Persistent storage

Without the persistence environment variables, the enabled page cache is process-local. For persistence, use a dedicated tmpfs directory and a dedicated disk directory:

```sh
export LLAMA_SERVER_PAGE_CACHE_RAM_DIR=/dev/shm/llama-page-cache
export LLAMA_SERVER_PAGE_CACHE_SSD_DIR=/var/cache/llama/page-cache
export LLAMA_SERVER_PAGE_CACHE_SSD_BYTES=412316860416
export LLAMA_SERVER_PAGE_CACHE_SSD_RESERVE_BYTES=68719476736
export LLAMA_SERVER_PAGE_CACHE_SSD_QUEUE_BYTES=1073741824
```

Also set `LLAMA_SERVER_PAGE_CACHE_NAMESPACE` to a 64-character lowercase SHA256 digest of a deterministic identity manifest containing the verified model shards, projector, chat template, runtime libraries and state-affecting launch settings. Do not use a fixed example digest or carry a namespace across a changed state format. Model filenames alone are not compatibility identities. Create the disk directory with permissions for the server user and size tmpfs for the chosen RAM budget.

RAM mappings survive process restarts, not reboots. SSD persistence covers completed writes; it does not make an in-flight capture durable. See [the storage contract](tools/server/PERSISTENT_RAM_CACHE.md) for recovery, locks, corruption handling and replacement policy.

### Request controls

Include this extension alongside the normal completion/chat request fields:

```json
{
  "cache_control": {
    "priority": "interactive",
    "lifecycle": "retained",
    "branch": "my-conversation"
  }
}
```

Priority is `interactive` or `background`; lifecycle is `retained`, `ephemeral` or `one_shot`. A one-shot request cannot name a branch and does not update retained cache history. A branch is a retention claim, not permission to reuse different prompt content. Matching still uses exact prompt and media identity.

## Reproducing the cache checks

The fork retains its upstream integration at `8baabd53b77d1c48c6ef8d3b620436292f3f6231`. Concurrency changes were ported from production source `85ce7b07da2e6d733ea2f39653d6b952a19e65f9` onto that base. The CPU storage suite, synthetic GLM target/draft state tests, and HTTP restart tests are independent of the large reference model. The HTTP test checks cold prefill, live reuse, process restart with tmpfs intact, and recovery after deleting its temporary RAM cache while retaining SSD backups. It asserts identical output and restored-token counts.

```sh
cmake -S . -B build-cpu -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --target llama-server test-fork-llama-archs -j 8
build-cpu/bin/test-fork-llama-archs -a glm5next
mkdir -p /tmp/llama-test-models
build-cpu/bin/test-fork-llama-archs -a glm5next -o /tmp/llama-test-models
python3 tools/server/tests/unit/test-page-cache-http.py \
  --server build-cpu/bin/llama-server \
  --model /tmp/llama-test-models/glm5next-moe.gguf \
  --output /tmp/llama-cache-test-results
```

Use a CPU build for that script: disabling GPU layers alone does not prevent a GPU-enabled library from initializing devices. These checks establish checkpoint correctness on the fixture; they are not production-model throughput or quality benchmarks.

## Where to look

| Area | Starting point |
|---|---|
| Checkpoint graph and restore | `tools/server/server-context.cpp` |
| tmpfs storage and byte descriptors | `tools/server/server-page-ram.h` |
| SSD writer and pressure handling | `tools/server/server-page-ssd.h` |
| Replacement weights | `tools/server/server-cache-policy.h` |
| Response lifecycle and leases | `tools/server/server-responses-control.h`, `server-responses-leases.h` |
| State range export | `src/llama-ext.h`, `src/llama-context.cpp`, `src/llama-memory-hybrid.cpp` |
| GLM-specific graph | `src/models/glm5next.cpp` |
| HIP and quantized kernels | `ggml/src/ggml-cuda/`, `ggml/src/ggml-hip/` |
| CPU cache contract tests | `tools/server/tests/unit/test_page_ram.py` |

Measure end-to-end prefill, decode and cache-hit behavior. A faster kernel fixture is not necessarily a faster whole model. The reference profile disables MTP and Turbo3; expert-parallel and global forced-MMQ experiments are not advertised as deployment wins. No general speedup claim is implied by the source being public.

---

## Upstream llama.cpp documentation

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
