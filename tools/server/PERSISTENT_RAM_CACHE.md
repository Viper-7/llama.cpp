# Process-persistent range-state RAM cache

This extension adds process-persistent checkpoint storage and optional asynchronous SSD backing. Existing target attention/indexer/recurrent state, media-aware exact identities, optional draft state, exact prompt endpoint capture, branch policy and one-shot exclusion remain the cache contract.

Configure both environment variables:

- `LLAMA_SERVER_PAGE_CACHE_RAM_DIR`: a dedicated tmpfs directory. Ordinary disk filesystems are rejected.
- `LLAMA_SERVER_PAGE_CACHE_NAMESPACE`: 64 lowercase hexadecimal digits from a SHA256 identity covering the checksum-verified model shards, projector/template, immutable runtime closure and state-affecting launch configuration. Changing this identity makes existing checkpoint state incompatible.

The existing `--cache-page-ram` and `--cache-page-tokens` options still enable and size the cache. The effective RAM limit is capped at the tmpfs total capacity minus up to 128 MiB of metadata headroom. A root lock enforces that all managed namespaces together stay below this effective byte budget. Other tmpfs users can still reduce available capacity; `posix_fallocate` fails cleanly when insufficient pages remain, before any GPU state export.

Target, draft and diagnostic reference bytes are exported directly into shared mmap files, avoiding a second full heap copy. Each completed capture records a fast XXH3 checksum and then atomically replaces the graph manifest. Graph metadata includes exact parent coordinates, branch claims and all three state payloads. Eviction publishes the reduced graph and removes unreferenced byte files before allocating the next capture, so a full cache can accept replacements. Process termination keeps the shared tmpfs pages; a host reboot does not. The optional SSD tier described below retains completed writes on disk; pending writes are not durable checkpoints.

An exclusive namespace lock prevents concurrent owners from mutating a graph. Reopen verifies namespace, format, checksum algorithm, limit, page geometry, parent chains and every payload checksum. Corrupt or incomplete nodes and their dependent descendants are skipped. The cleaned valid graph is published and orphan files are removed. A disabled, locked or invalid store leaves the existing cold heap cache available; `POST /v1/internal/cache-probe` reports persistence disabled and failures. It must not be described as a restored cache.

Incompatible namespace cleanup is scoped to this dedicated root: only inactive, 64-hex, non-symlink directories containing recognized cache files are removed, after acquiring their namespace lock. Active namespaces and directories with unrelated files are preserved. The current namespace survives model unloads and process restarts. Branch release removes ownership and persists immediately. Now-unowned pages receive a persisted 30-second `release_after_ms` grace: native and dormant GC exclude them until it expires, allowing a fingerprint replacement to adopt the unchanged prefix. Adoption clears the marker and supplies its new retention claim. Repeated release of an already-removed claim cannot extend grace. Unadopted pages have zero ownership weight and become ordinary graph-safe pressure GC candidates after expiry; this is GC eligibility, not a hard deletion timer. Parents referenced by shared retained children remain intact. Recovery bounds future deadlines to 30 seconds.

The existing exact endpoint capture occurs after prompt prefill and before decode. Completed prompt state is therefore already retained before a model is unloaded. An unfinished prompt or generated continuation after its last checkpoint is not synthesized into a checkpoint during shutdown.

`POST /v1/internal/cache-probe` includes `page_cache.persistence` counters for enabled state, recovered/rejected nodes, atomic graph commits, failures, orphan collection and incompatible namespace collection. Captures/hits/restore counters retain their existing meaning.

CPU-only validation:

```sh
LLAMA_PAGE_RAM_TEST_LIB_DIR=/path/to/cpu-build/bin \
  python3 -m pytest --confcutdir tools/server/tests/unit \
  tools/server/tests/unit/test_page_ram.py -q
```

The tests fork and terminate a writer without destructors, reopen identical bytes, reject corruption and incomplete parent chains, verify locks and namespace isolation, preserve active namespaces and shared branch ancestors, reject non-tmpfs storage, collect abandoned captures and prove allocation replacement progresses at the byte budget boundary. They instantiate the real cache graph with synthetic state bytes and never load a model or allocate GPU memory.

Build the executable and all llama/ggml libraries from the same revision. Record their checksums in the cache identity, and use a new namespace after changing runtime or state-affecting settings. CPU storage tests do not establish GPU restore correctness for a particular model; test that model and its exact launch configuration separately.

Cache telemetry: `GET /props` bypasses sleeping and does not expose live cache state. The existing cache probe accepts a plain synthetic `{"prompt":""}` body and tokenizes without generation or GPU state capture. Its normal response constructor can wake a sleeping runtime, so use it only against an already awake, resident child.

## Age-aware replacement

`age_frequency_density_v1` replaces permanent priority tiers. Eligible leaves are
ranked by `max(hint_multiplier * 2^(-hint_age / half_life)) *
(1 + log2(1 + min(retrievals, 15))) * recompute_tokens / (resident_bytes + 1)`.
The lowest score is evicted first. Last actual access breaks equal-score ties;
manifest order preserves exact ties. Capture ancestors and graph parents remain
ineligible. These are pressure-time weights, not expiry timers or reserved RAM.

| Ownership hint | Fresh multiplier | Half-life |
|---|---:|---:|
| Incidental retained interactive | 4 | 30 minutes |
| Named retained interactive | 4 | 1 hour |
| Named retained background | 2 | 8 hours |
| Incidental retained background | 1 | 1 hour |
| Ephemeral interactive (named or incidental) | 2 | 15 minutes |
| Ephemeral background (named or incidental) | 0.5 | 15 minutes |

The hints age separately: a background access cannot renew an old interactive
claim. Retrievals count distinct native requests that actually adopt an existing
page, including matched graph prefixes already resident in a live slot; new capture and repeated capture
within the creating request do not inflate frequency. Frequency saturates at 15
and never eliminates age decay. One-shot requests still do not touch history,
retention or LRU. Explicit branch release and graph-safe parent handling remain.

History consists of optional numeric fields in graph version 1. Unknown legacy
activity is cold, not fabricated as current usage. Reopen validates payloads as
before and preserves the original manifest LRU despite topological load ordering.
Invalid optional policy fields are ignored instead of rejecting valid payloads.
The cache-probe reports replacement policy, known/cold page counts, bounded
retrieval counts and the decay horizons. Changing only replacement weights need not change model-state bytes, but this fork does not provide a cross-build compatibility guarantee. Use a fresh namespace when upgrading. The modernized runtime has different upstream state APIs from the earlier deployment build.

External dormant-cache collectors can use the published policy header as a reference. Such a collector must honor the same namespace locks and graph dependencies; no external host daemon is required or included here.

## Sparse live-prefix endpoint capture

A live slot may reuse a prefix longer than its retained coarse graph. If an
endpoint's nominal coarse parent is absent, capture selects the longest rooted
matching compatible ancestor at or before that boundary, and serializes one
wider range from there to the current tail. With no surviving compatible
ancestor it uses a root range, subject to the same byte/budget checks. The
selected ancestry stays protected during eviction. Exact endpoints including
complete media chunks retain their media identity and position geometry. The
read-only cache probe reports successful `capture_bridges`. CPU graph tests
use synthetic state bytes and real media/token identity; actual model-state
restore correctness remains separately observable through natural traffic.

## Optional SSD backing

Set `LLAMA_SERVER_PAGE_CACHE_SSD_DIR` to a dedicated disk-backed directory. The RAM and SSD stores use the same compatibility namespace. Symlink roots and tmpfs SSD roots are rejected.

| Variable | Default | Meaning |
|---|---:|---|
| `LLAMA_SERVER_PAGE_CACHE_SSD_BYTES` | 412316860416 | Disk cache byte budget (384 GiB) |
| `LLAMA_SERVER_PAGE_CACHE_SSD_RESERVE_BYTES` | 68719476736 | Free disk space reserve (64 GiB) |
| `LLAMA_SERVER_PAGE_CACHE_SSD_QUEUE_BYTES` | 1073741824 | Bounded asynchronous write queue (1 GiB) |

The writer copies immutable checkpoint payloads in the background. Recovery uses verified descriptors and restores payloads lazily. Pressure reclamation respects namespace locks, canonical response leases and branch release grace. Dedicated cache directories contain model state and should be protected like inference data.
