// Included inside the fixture's anonymous namespace. No serving/backend hooks.
// All arenas are retained until the complete sequence ends: no cache eviction,
// no recycled first-node pointers, no borrowed invocation-lifetime leaves.
struct sequence_budget {
    const config & c;
    std::vector<size_t> bytes, entries;
    explicit sequence_budget(const config & c_) : c(c_), bytes(c.devices.size()), entries(c.devices.size()) {}
    void allocate(arena & a, ggml_backend_t backend, size_t owner) {
        const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(
            a.ctx, ggml_backend_get_default_buffer_type(backend));
        require(entries[owner] < c.cache_entries, "sequence arena entry cap exceeded before allocation");
        require(needed <= c.cache_bytes - bytes[owner], "sequence arena byte cap exceeded before allocation");
        a.allocate(backend);
        const size_t actual = ggml_backend_buffer_get_size(a.buffer);
        require(actual == needed, "sequence allocator size disagrees with preflight");
        bytes[owner] += actual; ++entries[owner];
    }
};

std::string sequence_hash_file(const std::string & path, size_t limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good() && input.tellg() >= 0 && uint64_t(input.tellg()) <= limit,
        "unreadable/oversized provenance file " + path);
    input.seekg(0);
    sha256_t state; sha256_init(&state);
    std::array<unsigned char, 65536> block;
    while (input.read(reinterpret_cast<char *>(block.data()), block.size()) || input.gcount()) {
        sha256_update(&state, block.data(), size_t(input.gcount()));
    }
    require(input.eof() && !input.bad(), "provenance read failed");
    unsigned char digest[32]; sha256_final(&state, digest);
    std::string result;
    for (auto b : digest) { char hex[3]; std::snprintf(hex, sizeof(hex), "%02x", b); result += hex; }
    return result;
}

json sequence_binary_provenance() {
#ifdef __linux__
    json result = {{"executable_sha256", sequence_hash_file("/proc/self/exe", size_t(2) << 30)}};
    std::ifstream maps("/proc/self/maps");
    require(maps.good(), "cannot verify loaded ggml libraries");
    std::map<std::string, std::string> libraries;
    std::string line;
    while (std::getline(maps, line)) {
        const size_t start = line.find('/');
        if (start == std::string::npos) { continue; }
        const std::string path = line.substr(start);
        if (path.find("libggml") != std::string::npos && path.find(".so") != std::string::npos) {
            libraries.emplace(path, "");
        }
    }
    require(!maps.bad(), "loaded library provenance read failed");
    for (auto & lib : libraries) { lib.second = sequence_hash_file(lib.first, size_t(8) << 30); }
    result["loaded_ggml_libraries_sha256"] = libraries;
    return result;
#else
    throw std::runtime_error("sequence provenance currently requires Linux /proc");
#endif
}

std::vector<uint32_t> sequence_bits(const std::vector<float> & values) {
    std::vector<uint32_t> bits(values.size());
    std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
    return bits;
}

// The reference contains exact F32 bit patterns, not lossy decimal outputs.
// Input capture, selected bank bytes, executable and loaded ggml libraries all
// have independently recomputed SHA256 identities before candidate execution.
struct sequence_reference {
    static constexpr size_t max_bytes = 128 * 1024 * 1024;
    json identity;
    std::vector<std::vector<float>> outputs;
    FILE * output_file = nullptr;
    explicit sequence_reference(const config & c, const capture_data & capture,
            const std::array<bank_source, 3> & src, const std::string & bank_hash) {
        size_t elements = 0;
        for (const auto & item : capture.cases) { elements += size_t(c.d) * item.q; }
        require(elements <= 8 * 1024 * 1024, "reference output geometry exceeds bounded format");
        identity = {{"schema", "glm53-chronological-reference-v1"}, {"D", c.d}, {"F", c.f},
            {"experts", c.experts}, {"top_k", c.used}, {"layer", c.layer}, {"clamp", c.clamp},
            {"capture_sha256", capture.sha256},
            {"bank_bytes_sha256_gate_up_down", bank_hash}, {"devices", c.devices},
            {"activation_recipe", "mt19937_123_plus_q_plus_65537_case_normal_0_0.5_v1"},
            {"binary", sequence_binary_provenance()}, {"cases", capture.cases.size()}};
        for (const auto & b : src) { identity["bank_types"].push_back(ggml_type_name(b.type)); }
        outputs.resize(capture.cases.size());
        if (c.sequence == "native") {
            output_file = std::fopen(c.reference_out.c_str(), "wx");
            require(output_file != nullptr, "reference output must be a new exclusively created file");
        } else {
            std::ifstream in(c.reference_in, std::ios::binary | std::ios::ate);
            require(in.good() && in.tellg() > 0 && uint64_t(in.tellg()) <= max_bytes,
                "reference input unreadable/oversized");
            in.seekg(0);
            json doc = json::parse(in, [](int depth, json::parse_event_t, json &) {
                require(depth <= 16, "reference nesting exceeds format cap");
                return true;
            });
            in >> std::ws;
            require(in.eof() && !in.bad() && doc.size() == 3 && doc.at("complete") == true &&
                doc.at("identity") == identity && doc.at("outputs").is_array() &&
                doc.at("outputs").size() == outputs.size(), "reference provenance/completion mismatch");
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto & row = doc.at("outputs").at(i);
                require(row.is_array() && row.size() == size_t(c.d) * capture.cases[i].q,
                    "reference output shape mismatch");
                outputs[i].resize(row.size());
                for (size_t j = 0; j < row.size(); ++j) {
                    const uint32_t bits = uint32_t(count_field(json{{"bits", row[j]}}, "bits", 0, UINT32_MAX));
                    std::memcpy(&outputs[i][j], &bits, sizeof(bits));
                    require(std::isfinite(outputs[i][j]), "nonfinite reference output");
                }
            }
        }
    }
    ~sequence_reference() { if (output_file) { std::fclose(output_file); } }
    void finish() {
        if (!output_file) { return; }
        json doc = {{"identity", identity}, {"complete", true}, {"outputs", json::array()}};
        for (const auto & row : outputs) { doc["outputs"].push_back(sequence_bits(row)); }
        const std::string data = doc.dump() + "\n";
        require(data.size() <= max_bytes, "reference serialization exceeds byte cap");
        require(std::fwrite(data.data(), 1, data.size(), output_file) == data.size() &&
            std::fflush(output_file) == 0, "reference output write failed");
        FILE * closing = output_file; output_file = nullptr;
        require(std::fclose(closing) == 0, "reference output close failed");
    }
};

struct sequence_entry {
    arena mem;
    ggml_tensor * metadata, * result = nullptr;
    std::vector<int32_t> upload;
    std::vector<size_t> counts;
    sequence_entry(const config & c, int q, const std::vector<assignments> & buckets,
            const banks & bank, ggml_tensor * x, ggml_tensor * coeff,
            ggml_backend_t backend, size_t owner, sequence_budget & budget) {
        counts.resize(c.used + 1);
        size_t ints = 0;
        for (int r = 1; r <= c.used; ++r) {
            counts[r] = buckets[r].token.size();
            ints += (2 * r + 1) * counts[r];
        }
        require(ints > 0, "empty sequence entry");
        metadata = ggml_new_tensor_1d(mem.ctx, GGML_TYPE_I32, ints);
        upload.reserve(ints);
        ggml_tensor * all_coeff = ggml_reshape_2d(mem.ctx, coeff, 1, c.used * q);
        size_t offset = 0;
        for (int r = 1; r <= c.used; ++r) {
            const size_t n = counts[r];
            if (!n) { continue; }
            ggml_tensor * ids = ggml_view_2d(mem.ctx, metadata, r, n, r * sizeof(int32_t), offset);
            offset += r * n * sizeof(int32_t);
            ggml_tensor * tokens = ggml_view_1d(mem.ctx, metadata, n, offset);
            offset += n * sizeof(int32_t);
            ggml_tensor * slots = ggml_view_1d(mem.ctx, metadata, r * n, offset);
            offset += r * n * sizeof(int32_t);
            ggml_tensor * inputs = ggml_get_rows(mem.ctx, x, tokens);
            ggml_tensor * weights = ggml_get_rows(mem.ctx, all_coeff, slots);
            ggml_tensor * ff = expert_ffn(mem, bank, ggml_reshape_3d(mem.ctx, inputs, c.d, 1, n), ids, c.clamp);
            ff = ggml_mul(mem.ctx, ff, ggml_reshape_3d(mem.ctx, weights, 1, r, n));
            ggml_tensor * sum = ggml_view_2d(mem.ctx, ff, c.d, n, ff->nb[2], 0);
            for (int s = 1; s < r; ++s) {
                sum = ggml_add(mem.ctx, sum, ggml_view_2d(mem.ctx, ff, c.d, n, ff->nb[2], s * ff->nb[1]));
            }
            ggml_tensor * partial = ggml_get_rows_back(mem.ctx, sum, tokens, x);
            result = result ? ggml_add(mem.ctx, result, partial) : partial;
        }
        require(offset == ggml_nbytes(metadata), "sequence metadata geometry mismatch");
        ggml_build_forward_expand(mem.graph, result);
        budget.allocate(mem, backend, owner);
    }
    void set(const std::vector<assignments> & buckets, ggml_backend_t backend) {
        upload.clear();
        for (size_t r = 1; r < counts.size(); ++r) {
            require(buckets[r].token.size() == counts[r], "sequence key changed shape");
            upload.insert(upload.end(), buckets[r].expert.begin(), buckets[r].expert.end());
            upload.insert(upload.end(), buckets[r].token.begin(), buckets[r].token.end());
            upload.insert(upload.end(), buckets[r].slot.begin(), buckets[r].slot.end());
        }
        require(upload.size() * sizeof(int32_t) == ggml_nbytes(metadata), "sequence metadata upload mismatch");
        ggml_backend_tensor_set_async(backend, metadata, upload.data(), 0, ggml_nbytes(metadata));
    }
};

struct sequence_owner {
    arena leaves{false}, collector;
    ggml_tensor * x, * coeff, * zero_tensor, * result = nullptr;
    std::vector<ggml_tensor *> slots;
    std::vector<float> zero;
    sequence_owner(const config & c, int q, ggml_tensor * root_x, ggml_tensor * root_coeff,
            size_t owner, ggml_backend_t backend, sequence_budget & budget) {
        x = owner == 0 ? root_x : ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_F32, c.d, q);
        coeff = owner == 0 ? root_coeff : ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_F32, c.used, q);
        const int nslots = c.sequence == "bucket" ? c.used : 1;
        for (int i = 0; i < nslots; ++i) {
            auto * slot = ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_F32, c.d, q);
            slots.push_back(slot);
            result = result ? ggml_add(collector.ctx, result, slot) : slot;
        }
        zero_tensor = ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_F32, c.d, q);
        result = ggml_cont(collector.ctx, result);
        ggml_build_forward_expand(collector.graph, result);
        budget.allocate(leaves, backend, owner);
        budget.allocate(collector, backend, owner);
        zero.resize(size_t(c.d) * q, 0.0f);
        // One upload in timed first-q setup. Subsequent inactive-slot clears
        // stay on this backend; do not reupload host zeros on every dispatch.
        ggml_backend_tensor_set(zero_tensor, zero.data(), 0, ggml_nbytes(zero_tensor));
    }
};

struct sequence_q {
    arena inputs{false}, graph;
    ggml_tensor * x, * coeff, * ids_storage, * ids, * result = nullptr;
    std::unique_ptr<arena> readback;
    ggml_tensor * route_ids = nullptr;
    std::vector<int32_t> host_ids;
    std::vector<std::unique_ptr<sequence_owner>> owners;
    std::vector<ggml_tensor *> partials;
    bool native;
    sequence_q(const config & c, int q, const banks & full,
            const std::vector<ggml_backend_t> & backends, sequence_budget & budget)
        : native(c.sequence == "native" || q < 6) {
        x = ggml_new_tensor_2d(inputs.ctx, GGML_TYPE_F32, c.d, q);
        coeff = ggml_new_tensor_2d(inputs.ctx, GGML_TYPE_F32, c.used, q);
        ids_storage = ggml_new_tensor_2d(inputs.ctx, GGML_TYPE_I32, c.experts, q);
        ids = ggml_view_2d(inputs.ctx, ids_storage, c.used, q, ids_storage->nb[1], 0);
        budget.allocate(inputs, backends[0], 0);
        if (native) {
            ggml_tensor * ff = expert_ffn(graph, full, ggml_reshape_3d(graph.ctx, x, c.d, 1, q), ids, c.clamp);
            ff = ggml_mul(graph.ctx, ff, ggml_reshape_3d(graph.ctx, coeff, 1, c.used, q));
            result = ggml_view_2d(graph.ctx, ff, c.d, q, ff->nb[2], 0);
            for (int s = 1; s < c.used; ++s) {
                result = ggml_add(graph.ctx, result, ggml_view_2d(graph.ctx, ff, c.d, q, ff->nb[2], s * ff->nb[1]));
            }
        } else {
            readback.reset(new arena);
            route_ids = ggml_cont(readback->ctx, ids);
            ggml_build_forward_expand(readback->graph, route_ids);
            budget.allocate(*readback, backends[0], 0);
            host_ids.resize(size_t(c.used) * q);
            for (size_t p = 0; p < backends.size(); ++p) {
                owners.emplace_back(new sequence_owner(c, q, x, coeff, p, backends[p], budget));
                // Real permanent leaves, not views of whichever cached producer ran.
                auto * partial = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, c.d, q);
                partials.push_back(partial);
                result = result ? ggml_add(graph.ctx, result, partial) : partial;
            }
            result = ggml_cont(graph.ctx, result);
        }
        ggml_build_forward_expand(graph.graph, result);
        budget.allocate(graph, backends[0], 0);
    }
};

struct sequence_runner {
    const config & c;
    const std::vector<int> & bounds;
    const std::vector<ggml_backend_t> & backends;
    const banks & full;
    const std::vector<std::unique_ptr<banks>> & weights;
    sequence_budget budget;
    // Declaration order makes cache arenas die before their input leaves.
    std::map<int, std::unique_ptr<sequence_q>> shapes;
    std::vector<std::map<std::vector<size_t>, std::unique_ptr<sequence_entry>>> caches;
    sequence_runner(const config & c_, const std::vector<int> & bounds_,
            const std::vector<ggml_backend_t> & backends_, const banks & full_,
            const std::vector<std::unique_ptr<banks>> & weights_)
        : c(c_), bounds(bounds_), backends(backends_), full(full_), weights(weights_), budget(c_), caches(backends_.size()) {}
    ~sequence_runner() {
        // Includes exception paths after async uploads: no leaf/vector frees in flight.
        for (auto backend : backends) { ggml_backend_synchronize(backend); }
    }
    ggml_tensor * run(const replay_case & item, size_t index, json & event) {
        const auto start = clock_type::now();
        auto found = shapes.find(item.q);
        const bool q_miss = found == shapes.end();
        if (q_miss) {
            found = shapes.emplace(item.q, std::unique_ptr<sequence_q>(
                new sequence_q(c, item.q, full, backends, budget))).first;
        }
        sequence_q & state = *found->second;
        const double q_setup_ms = elapsed_ms(start);
        const auto injection_start = clock_type::now();
        std::mt19937 rng(123 + item.q + 65537 * uint32_t(index));
        std::normal_distribution<float> normal(0, 0.5f);
        std::vector<float> x(size_t(c.d) * item.q);
        for (float & v : x) { v = normal(rng); }
        std::vector<int32_t> padded(size_t(c.experts) * item.q, 0);
        for (int t = 0; t < item.q; ++t) {
            std::copy_n(item.ids.begin() + t * c.used, c.used, padded.begin() + t * c.experts);
        }
        ggml_backend_tensor_set(state.x, x.data(), 0, ggml_nbytes(state.x));
        ggml_backend_tensor_set(state.coeff, item.coeff.data(), 0, ggml_nbytes(state.coeff));
        ggml_backend_tensor_set(state.ids_storage, padded.data(), 0, ggml_nbytes(state.ids_storage));
        ggml_backend_synchronize(backends[0]);
        const double injection_ms = elapsed_ms(injection_start);
        auto phase = clock_type::now();
        double route_ms = 0, lookup_ms = 0, miss_ms = 0, dispatch_ms = 0, submit_ms = 0;
        size_t zero_upload_bytes = 0, clear_bytes = 0, clear_copies = 0;
        json keys = json::array();
        if (!state.native) {
            if (q_miss) {
                for (const auto & owner : state.owners) { zero_upload_bytes += ggml_nbytes(owner->zero_tensor); }
            }
            require(ggml_backend_graph_compute_async(backends[0], state.readback->graph) == GGML_STATUS_SUCCESS, "sequence route gather failed");
            ggml_backend_tensor_get_async(backends[0], state.route_ids, state.host_ids.data(), 0, state.host_ids.size() * sizeof(int32_t));
            ggml_backend_synchronize(backends[0]);
            const auto parts = partition(state.host_ids, item.q, c.used, bounds);
            route_ms = elapsed_ms(phase);
            struct pending { size_t owner, slot; sequence_entry * entry; std::vector<assignments> buckets; };
            std::vector<pending> work;
            std::vector<std::vector<bool>> active_slots(backends.size());
            for (size_t p = 0; p < backends.size(); ++p) {
                active_slots[p].resize(state.owners[p]->slots.size(), false);
            }
            phase = clock_type::now();
            for (size_t p = 0; p < backends.size(); ++p) {
                if (parts[p].expert.empty()) { continue; }
                auto buckets = cardinality_buckets(parts[p], c.used);
                const size_t iterations = c.sequence == "bucket" ? size_t(c.used) : 1;
                for (size_t slot = 0; slot < iterations; ++slot) {
                    std::vector<assignments> selected = buckets;
                    std::vector<size_t> key{size_t(item.q)};
                    if (c.sequence == "bucket") {
                        if (buckets[slot + 1].token.empty()) { continue; }
                        key.push_back(slot + 1); key.push_back(buckets[slot + 1].token.size());
                        for (size_t r = 1; r < selected.size(); ++r) { if (r != slot + 1) { selected[r] = {}; } }
                    } else {
                        for (size_t r = 1; r < buckets.size(); ++r) { key.push_back(buckets[r].token.size()); }
                    }
                    auto cached = caches[p].find(key);
                    const bool miss = cached == caches[p].end();
                    if (miss) {
                        const auto miss_start = clock_type::now();
                        cached = caches[p].emplace(key, std::unique_ptr<sequence_entry>(new sequence_entry(c, item.q,
                            selected, *weights[p], state.owners[p]->x, state.owners[p]->coeff, backends[p], p, budget))).first;
                        miss_ms += elapsed_ms(miss_start);
                    }
                    keys.push_back({{"owner", p}, {"key", key}, {"host_hit", !miss}});
                    active_slots[p][slot] = true;
                    work.push_back({p, slot, cached->second.get(), std::move(selected)});
                }
            }
            lookup_ms = elapsed_ms(phase);
            phase = clock_type::now();
            // Fan-out and metadata submit before any root-owner FFN compute.
            for (size_t p = 0; p < backends.size(); ++p) {
                auto & owner = *state.owners[p];
                for (size_t s = 0; s < owner.slots.size(); ++s) {
                    // Active scatter results materialize the entire [D,q]
                    // slot, including zero rows. Only inactive slots need a
                    // clear; a same-backend copy preserves stream ordering.
                    if (!active_slots[p][s]) {
                        auto * slot = owner.slots[s];
                        ggml_backend_tensor_copy_async(backends[p], backends[p], owner.zero_tensor, slot);
                        clear_bytes += ggml_nbytes(slot);
                        ++clear_copies;
                    }
                }
                if (p && !parts[p].expert.empty()) {
                    ggml_backend_tensor_copy_async(backends[0], backends[p], state.x, owner.x);
                    ggml_backend_tensor_copy_async(backends[0], backends[p], state.coeff, owner.coeff);
                }
            }
            for (auto & w : work) { w.entry->set(w.buckets, backends[w.owner]); }
            dispatch_ms = elapsed_ms(phase);
            phase = clock_type::now();
            for (auto & w : work) {
                const auto backend = backends[w.owner];
                require(ggml_backend_graph_compute_async(backend, w.entry->mem.graph) == GGML_STATUS_SUCCESS, "sequence bucket compute failed");
                ggml_backend_tensor_copy_async(backend, backend, w.entry->result, state.owners[w.owner]->slots[w.slot]);
            }
            for (size_t p = 0; p < backends.size(); ++p) {
                auto & owner = *state.owners[p];
                require(ggml_backend_graph_compute_async(backends[p], owner.collector.graph) == GGML_STATUS_SUCCESS, "sequence owner collector failed");
                ggml_backend_tensor_copy_async(backends[p], backends[0], owner.result, state.partials[p]);
            }
            submit_ms = elapsed_ms(phase);
        }
        phase = clock_type::now();
        require(ggml_backend_graph_compute_async(backends[0], state.graph.graph) == GGML_STATUS_SUCCESS, "sequence root compute failed");
        for (auto backend : backends) { ggml_backend_synchronize(backend); }
        const double collect_ms = elapsed_ms(phase);
        const double inclusive_ms = elapsed_ms(start);
        event = {{"q_shape_miss", q_miss}, {"q_setup_lookup_ms", q_setup_ms},
            {"input_injection_ms", injection_ms}, {"inclusive_with_input_ms", inclusive_ms},
            {"routed_region_plus_all_preparation_ms", inclusive_ms - injection_ms},
            {"route_readback_partition_ms", route_ms}, {"lookup_and_misses_ms", lookup_ms},
            {"miss_construction_allocation_ms", miss_ms}, {"dispatch_clear_upload_ms", dispatch_ms},
            {"bucket_owner_collect_submit_ms", submit_ms}, {"root_collect_all_sync_ms", collect_ms},
            {"policy", state.native ? "native_full_bank" : "hybrid_ep_q6_full_bank_resident"},
            {"route_D2H_bytes", state.native ? 0 : state.host_ids.size() * sizeof(int32_t)},
            {"initial_zero_tensor_upload_bytes", zero_upload_bytes},
            {"inactive_clear_same_backend_bytes", clear_bytes}, {"inactive_clear_copies", clear_copies},
            {"keys", keys}, {"arena_bytes_per_device", budget.bytes}, {"arena_entries_per_device", budget.entries}};
        return state.result;
    }
};

void run_sequence(const config & c, const capture_data & capture,
        const std::array<bank_source, 3> & src, const std::string & bank_hash,
        const std::vector<int> & bounds, const std::vector<ggml_backend_t> & backends,
        const banks & full, const std::vector<std::unique_ptr<banks>> & weights, double tolerance) {
    sequence_reference reference(c, capture, src, bank_hash);
    size_t full_bank_bytes = 0;
    std::vector<size_t> weight_bytes(backends.size());
    for (const auto & b : src) {
        const size_t expert_bytes = ggml_row_size(b.type, b.k) * b.m;
        full_bank_bytes += expert_bytes * b.e;
        for (size_t p = 0; p < backends.size(); ++p) {
            weight_bytes[p] += expert_bytes * (bounds[p + 1] - bounds[p]);
        }
    }
    weight_bytes[0] += full_bank_bytes;
    const json contract = {{"kind", "sequence_contract"}, {"mode", c.sequence}, {"passes", 3},
        {"warmups", 0}, {"cases", capture.cases.size()}, {"reference_identity", reference.identity},
        {"nmse_tolerance", tolerance}, {"full_bank_tensor_bytes", full_bank_bytes},
        {"resident_full_plus_shard_tensor_bytes_per_device", weight_bytes},
        {"cache_entries_per_device_cap", c.cache_entries}, {"cache_bytes_per_device_cap", c.cache_bytes},
        {"entry_accounting", "all_persistent_arenas_including_per_q_io_and_collectors"},
        {"allocation_accounting", "arena_buffers_only_weights_reported_separately"},
        {"host_metadata_bytes", "unobserved_bounded_by_arena_entry_count"},
        {"backend_workspace_staging_executable_bytes", "unobserved_not_in_arena_cap"},
        {"device_graph_cache_events", "unobserved_host_hits_do_not_prove_HIP_hits"},
        {"zero_clear_policy", "one_initial_upload_per_owner_q_then_inactive_only_same_backend_copies"},
        {"schedule", "back_to_back_layer_only_optimistic_residency_not_original_elapsed_gaps"},
        {"small_q_policy", "native_q_less_than_6_before_route_readback_requires_full_resident_bank"},
        {"quality_scope", "captured_routes_real_or_synthetic_weights_synthetic_varying_activations"}};
    std::puts(contract.dump().c_str());
    sequence_runner runner(c, bounds, backends, full, weights);
    const char * labels[] = {"first_sequence", "first_repeated_sequence", "second_repeated_sequence"};
    for (int pass = 0; pass < 3; ++pass) {
        const auto pass_start = clock_type::now();
        double total = 0, injection = 0, validation = 0;
        for (size_t i = 0; i < capture.cases.size(); ++i) {
            const auto & item = capture.cases[i];
            json event;
            ggml_tensor * output = runner.run(item, i, event);
            total += event.at("routed_region_plus_all_preparation_ms").get<double>();
            injection += event.at("input_injection_ms").get<double>();
            const auto check_start = clock_type::now();
            std::vector<float> values(ggml_nelements(output));
            ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
            double sq = 0, norm = 0, max_abs = 0;
            if (c.sequence == "native" && pass == 0) { reference.outputs[i] = values; }
            for (size_t j = 0; j < values.size(); ++j) {
                require(std::isfinite(values[j]), "sequence nonfinite output");
                const double expected = reference.outputs[i][j];
                const double diff = double(values[j]) - expected;
                sq += diff * diff; norm += expected * expected;
                max_abs = std::max(max_abs, std::abs(diff));
            }
            const double nmse = sq / std::max(norm, 1e-30);
            require(nmse <= tolerance, "sequence numerical gate failed");
            const double validation_ms = elapsed_ms(check_start);
            validation += validation_ms;
            event.update({{"kind", "sequence_sample"}, {"pass", pass}, {"pass_label", labels[pass]},
                {"capture_index", i}, {"call", item.call}, {"layer_call", item.layer_call},
                {"q", item.q}, {"replay", item.replay}, {"coefficient_stage", item.stage},
                {"nmse", nmse}, {"max_abs", max_abs}, {"validation_readback_cpu_ms_excluded", validation_ms},
                {"cumulative_routed_region_plus_preparation_ms", total}});
            std::puts(event.dump().c_str());
        }
        const json summary = {{"kind", "sequence_result"}, {"pass", pass}, {"pass_label", labels[pass]},
            {"routed_region_plus_all_preparation_ms", total}, {"input_injection_ms", injection},
            {"validation_readback_cpu_ms_excluded", validation}, {"pass_wall_ms", elapsed_ms(pass_start)},
            {"cases", capture.cases.size()}};
        std::puts(summary.dump().c_str()); std::fflush(stdout);
    }
    reference.finish();
}
