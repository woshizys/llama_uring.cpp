#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#include <vector>

enum class MoePart : int32_t {
    Up     = 0,
    Gate   = 1,
    Down   = 2,
    GateUp = 3,
};

enum class TensorLayoutKind : int32_t {
    Contiguous    = 0,
    NonContiguous = 1,
};

struct TensorLayoutEntry {
    MoePart part = MoePart::Up;
    size_t offset = 0;
    size_t size = 0;
    size_t slot_size = 0;
};

struct TensorLayout {
    TensorLayoutKind kind = TensorLayoutKind::NonContiguous;
    std::vector<MoePart> parts = { MoePart::Up, MoePart::Gate, MoePart::Down, MoePart::GateUp };
    std::vector<TensorLayoutEntry> entries;
    size_t slot_size = 0;
};

struct ExpertKey {
    int32_t layer  = -1;
    int32_t expert = -1;

    bool operator==(const ExpertKey & other) const {
        return layer == other.layer && expert == other.expert;
    }

    bool operator<(const ExpertKey & other) const {
        return layer < other.layer || (layer == other.layer && expert < other.expert);
    }
};

struct ExpertSlice {
    MoePart part = MoePart::Up;

    int32_t file_id = -1;
    uint64_t file_offset = 0;
    uint64_t file_size = 0;
};

struct ExpertPrediction {
    int32_t layer = -1;
    int32_t expert = -1;
    float probability = 0.0f;
    uint64_t deadline_ns = 0;
};

inline ExpertSlice make_expert_slice(
        MoePart part,
        int32_t file_id,
        uint64_t file_offset,
        uint64_t file_size) {
    ExpertSlice slice;
    slice.part = part;
    slice.file_id = file_id;
    slice.file_offset = file_offset;
    slice.file_size = file_size;
    return slice;
}

extern "C" {

struct llama_expert_manager_ffi;
struct llama_expert_handle_ffi;
struct llama_expert_ticket_ffi;

struct llama_expert_slice_ffi {
    int32_t part;
    int32_t file_id;
    uint64_t file_offset;
    uint64_t file_size;
};

struct llama_expert_prediction_ffi {
    int32_t layer;
    int32_t expert;
    float probability;
    uint64_t deadline_ns;
};

// Implemented by the Rust expert cache crate. ensure() pins the expert until
// the returned opaque handle is released.
llama_expert_manager_ffi * llama_expert_manager_new(
        size_t capacity,
        size_t hidden_dim,
        size_t intermediate_dim,
        size_t precision_bits);

llama_expert_manager_ffi * llama_expert_manager_new_with_slot_size(
        size_t capacity,
        size_t slot_size);

llama_expert_manager_ffi * llama_expert_manager_new_with_layout(
        size_t capacity,
        size_t slot_size,
        int32_t layout_kind,
        const int32_t * layout_parts,
        size_t part_count);

llama_expert_handle_ffi * llama_expert_manager_ensure(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        int32_t expert);

int32_t llama_expert_manager_ensure_many(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        const int32_t * experts,
        size_t count,
        llama_expert_handle_ffi ** handles_out);

int32_t llama_expert_manager_prefetch_many(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        const int32_t * experts,
        size_t count);

int32_t llama_expert_manager_prefetch_next_layer_mandatory(
        llama_expert_manager_ffi * manager,
        int32_t current_layer,
        int32_t * target_layer_out);

int32_t llama_expert_manager_predict_prefetch_next(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        const int32_t * experts,
        size_t count,
        const float * router_scores,
        size_t max_predictions,
        uint64_t current_token_id,
        uint64_t deadline_ns,
        llama_expert_prediction_ffi * predictions_out,
        size_t predictions_capacity);

llama_expert_ticket_ffi * llama_expert_manager_submit_batch_async(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        const int32_t * experts,
        size_t count,
        uint64_t token_id);

llama_expert_handle_ffi * llama_expert_ticket_get_handle(
        const llama_expert_ticket_ffi * ticket,
        int32_t layer,
        int32_t expert);

llama_expert_handle_ffi * llama_expert_ticket_try_get_handle(
        const llama_expert_ticket_ffi * ticket,
        int32_t layer,
        int32_t expert);

llama_expert_handle_ffi * llama_expert_ticket_wait_any_ready(
        const llama_expert_ticket_ffi * ticket,
        int32_t layer,
        const int32_t * experts,
        size_t count,
        int32_t * expert_out);

void llama_expert_ticket_release(llama_expert_ticket_ffi * ticket);

void llama_expert_ticket_free(llama_expert_ticket_ffi * ticket);

void llama_expert_manager_release(
        llama_expert_manager_ffi * manager,
        llama_expert_handle_ffi * handle);

uint8_t * llama_expert_handle_host_ptr(
        const llama_expert_handle_ffi * handle,
        int32_t part);

uint8_t * llama_expert_handle_host_base_ptr(
        const llama_expert_handle_ffi * handle);

const uint8_t * llama_expert_handle_device_ptr(
        const llama_expert_handle_ffi * handle,
        int32_t part);

const uint8_t * llama_expert_handle_device_base_ptr(
        const llama_expert_handle_ffi * handle);

size_t llama_expert_handle_part_size(
        const llama_expert_handle_ffi * handle,
        int32_t part);

int32_t llama_expert_handle_slot_id(
        const llama_expert_handle_ffi * handle);

uint64_t llama_expert_handle_generation(
        const llama_expert_handle_ffi * handle);

int32_t llama_expert_manager_register_slice(
        llama_expert_manager_ffi * manager,
        int32_t layer,
        int32_t expert,
        const llama_expert_slice_ffi * slice);

int32_t llama_expert_manager_register_file(
        llama_expert_manager_ffi * manager,
        int32_t file_id,
        const char * path);


int32_t llama_expert_manager_write_background_file(
        llama_expert_manager_ffi * manager,
        const char * path,
        const uint8_t * data,
        size_t data_size,
        int32_t kv_slot_id,
        size_t * written_out);
const char * llama_expert_manager_last_error_message();

int32_t llama_expert_manager_stats(
        llama_expert_manager_ffi * manager,
        size_t * hit_out,
        size_t * miss_out);

void llama_expert_manager_free(llama_expert_manager_ffi * manager);

}

inline llama_expert_slice_ffi to_ffi(const ExpertSlice & slice) {
    llama_expert_slice_ffi out;
    out.part = static_cast<int32_t>(slice.part);
    out.file_id = slice.file_id;
    out.file_offset = slice.file_offset;
    out.file_size = slice.file_size;
    return out;
}

class ExpertManager;
class ExpertHandle;

class ExpertTicket {
public:
    ExpertTicket() = default;

    ExpertTicket(ExpertManager * manager, llama_expert_ticket_ffi * ticket)
        : manager_(manager), ticket_(ticket) {}

    ExpertTicket(const ExpertTicket &) = delete;
    ExpertTicket & operator=(const ExpertTicket &) = delete;

    ExpertTicket(ExpertTicket && other) noexcept {
        manager_ = other.manager_;
        ticket_ = other.ticket_;
        other.manager_ = nullptr;
        other.ticket_ = nullptr;
    }

    ExpertTicket & operator=(ExpertTicket && other) noexcept {
        if (this != &other) {
            reset();
            manager_ = other.manager_;
            ticket_ = other.ticket_;
            other.manager_ = nullptr;
            other.ticket_ = nullptr;
        }
        return *this;
    }

    ~ExpertTicket() {
        reset();
    }

    explicit operator bool() const {
        return ticket_ != nullptr;
    }

    ExpertHandle get_handle(int32_t layer, int32_t expert) const;
    ExpertHandle try_get_handle(int32_t layer, int32_t expert) const;
    std::pair<int32_t, ExpertHandle> wait_any_ready(int32_t layer, const int32_t * experts, size_t count) const;

    void release();
    void reset();

private:
    ExpertManager * manager_ = nullptr;
    llama_expert_ticket_ffi * ticket_ = nullptr;

    friend class ExpertManager;
};

class ExpertHandle {
public:
    ExpertHandle() = default;

    ExpertHandle(ExpertManager * manager, llama_expert_handle_ffi * handle)
        : manager_(manager), handle_(handle) {}

    ExpertHandle(const ExpertHandle &) = delete;
    ExpertHandle & operator=(const ExpertHandle &) = delete;

    ExpertHandle(ExpertHandle && other) noexcept {
        manager_ = other.manager_;
        handle_ = other.handle_;
        other.manager_ = nullptr;
        other.handle_ = nullptr;
    }

    ExpertHandle & operator=(ExpertHandle && other) noexcept {
        if (this != &other) {
            reset();
            manager_ = other.manager_;
            handle_ = other.handle_;
            other.manager_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~ExpertHandle() {
        reset();
    }

    explicit operator bool() const {
        return handle_ != nullptr;
    }

    uint8_t * host_ptr(MoePart part) const;

    uint8_t * host_base_ptr() const {
        return llama_expert_handle_host_base_ptr(handle_);
    }

    const uint8_t * device_ptr(MoePart part) const;

    const uint8_t * device_base_ptr() const {
        return llama_expert_handle_device_base_ptr(handle_);
    }

    size_t part_size(MoePart part) const;

    int32_t slot_id() const {
        return llama_expert_handle_slot_id(handle_);
    }

    uint64_t generation() const {
        return llama_expert_handle_generation(handle_);
    }

    void reset();

private:
    ExpertManager * manager_ = nullptr;
    llama_expert_handle_ffi * handle_ = nullptr;

    friend class ExpertManager;
};

class ExpertManager {
public:
    static ExpertManager create(
            size_t capacity,
            size_t hidden_dim,
            size_t intermediate_dim,
            size_t precision_bits) {
        llama_expert_manager_ffi * impl = llama_expert_manager_new(
                capacity,
                hidden_dim,
                intermediate_dim,
                precision_bits);
        if (impl == nullptr) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to create MoE expert manager: ") + err);
            }
            throw std::runtime_error("failed to create MoE expert manager: Rust FFI is not linked or returned null without an error");
        }
        return ExpertManager(impl, true, TensorLayout{}, capacity);
    }

    static ExpertManager create_with_slot_size(
            size_t capacity,
            size_t slot_size) {
        return create_with_layout(capacity, slot_size, TensorLayout{});
    }

    static ExpertManager create_with_layout(
            size_t capacity,
            size_t slot_size,
            const TensorLayout & layout) {
        std::vector<int32_t> parts;
        parts.reserve(layout.parts.size());
        for (MoePart part : layout.parts) {
            parts.push_back(static_cast<int32_t>(part));
        }

        llama_expert_manager_ffi * impl = llama_expert_manager_new_with_layout(
                capacity,
                slot_size,
                static_cast<int32_t>(layout.kind),
                parts.data(),
                parts.size());
        if (impl == nullptr) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to create MoE expert manager: ") + err);
            }
            throw std::runtime_error("failed to create MoE expert manager: Rust FFI is not linked or returned null without an error");
        }
        return ExpertManager(impl, true, layout, capacity);
    }

    explicit ExpertManager(
            llama_expert_manager_ffi * impl,
            bool owns_impl = false,
            TensorLayout layout = TensorLayout{},
            size_t capacity = 0)
        : impl_(impl), owns_impl_(owns_impl), layout_(std::move(layout)), capacity_(capacity) {
        if (impl_ == nullptr) {
            throw std::invalid_argument("ExpertManager requires a non-null Rust manager");
        }
    }

    ExpertManager(const ExpertManager &) = delete;
    ExpertManager & operator=(const ExpertManager &) = delete;

    ExpertManager(ExpertManager && other) noexcept {
        impl_ = other.impl_;
        owns_impl_ = other.owns_impl_;
        layout_ = std::move(other.layout_);
        capacity_ = other.capacity_;
        other.impl_ = nullptr;
        other.owns_impl_ = false;
        other.capacity_ = 0;
    }

    ExpertManager & operator=(ExpertManager && other) noexcept {
        if (this != &other) {
            close();
            impl_ = other.impl_;
            owns_impl_ = other.owns_impl_;
            layout_ = std::move(other.layout_);
            capacity_ = other.capacity_;
            other.impl_ = nullptr;
            other.owns_impl_ = false;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~ExpertManager() {
        close();
    }

    llama_expert_manager_ffi * get() const {
        return impl_;
    }

    const TensorLayout & layout() const {
        return layout_;
    }

    size_t capacity() const {
        return capacity_;
    }

    const TensorLayoutEntry * layout_entry(MoePart part) const {
        for (const TensorLayoutEntry & entry : layout_.entries) {
            if (entry.part == part) {
                return &entry;
            }
        }
        return nullptr;
    }

    uint8_t * handle_host_ptr(const ExpertHandle & handle, MoePart part) const {
        const TensorLayoutEntry * entry = layout_entry(part);
        if (entry != nullptr) {
            uint8_t * base = handle.host_base_ptr();
            if (base != nullptr) {
                return base + entry->offset;
            }
        }
        return llama_expert_handle_host_ptr(handle.handle_, static_cast<int32_t>(part));
    }

    const uint8_t * handle_device_ptr(const ExpertHandle & handle, MoePart part) const {
        const TensorLayoutEntry * entry = layout_entry(part);
        if (entry != nullptr) {
            const uint8_t * base = handle.device_base_ptr();
            if (base != nullptr) {
                return base + entry->offset;
            }
        }
        return llama_expert_handle_device_ptr(handle.handle_, static_cast<int32_t>(part));
    }

    size_t handle_part_size(const ExpertHandle & handle, MoePart part) const {
        const TensorLayoutEntry * entry = layout_entry(part);
        if (entry != nullptr) {
            return entry->size;
        }
        return llama_expert_handle_part_size(handle.handle_, static_cast<int32_t>(part));
    }

    ExpertHandle ensure(int32_t layer, int32_t expert) {
        llama_expert_handle_ffi * handle = llama_expert_manager_ensure(impl_, layer, expert);
        if (handle == nullptr) {
            throw std::runtime_error("failed to ensure MoE expert");
        }
        return ExpertHandle(this, handle);
    }

    std::vector<ExpertHandle> ensure_many(std::vector<ExpertKey> experts) {
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());

        std::vector<ExpertHandle> handles;
        handles.reserve(experts.size());
        for (const ExpertKey & key : experts) {
            handles.emplace_back(ensure(key.layer, key.expert));
        }
        return handles;
    }

    std::unordered_map<int32_t, ExpertHandle> ensure_map(std::vector<ExpertKey> experts) {
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());

        std::unordered_map<int32_t, ExpertHandle> handles;
        handles.reserve(experts.size());
        if (experts.empty()) {
            return handles;
        }

        const int32_t layer = experts.front().layer;
        std::vector<int32_t> expert_ids;
        expert_ids.reserve(experts.size());
        for (const ExpertKey & key : experts) {
            if (key.layer != layer) {
                throw std::runtime_error("failed to batch ensure MoE experts from multiple layers");
            }
            expert_ids.push_back(key.expert);
        }

        std::vector<llama_expert_handle_ffi *> raw_handles(expert_ids.size(), nullptr);
        const int32_t rc = llama_expert_manager_ensure_many(
                impl_,
                layer,
                expert_ids.data(),
                expert_ids.size(),
                raw_handles.data());
        if (rc != 0) {
            throw std::runtime_error("failed to batch ensure MoE experts");
        }

        for (size_t i = 0; i < expert_ids.size(); ++i) {
            handles.emplace(expert_ids[i], ExpertHandle(this, raw_handles[i]));
        }

        return handles;
    }

    void prefetch_many(std::vector<ExpertKey> experts) {
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        if (experts.empty()) {
            return;
        }

        const int32_t layer = experts.front().layer;
        std::vector<int32_t> expert_ids;
        expert_ids.reserve(experts.size());
        for (const ExpertKey & key : experts) {
            if (key.layer != layer) {
                throw std::runtime_error("failed to prefetch MoE experts from multiple layers");
            }
            expert_ids.push_back(key.expert);
        }

        const int32_t rc = llama_expert_manager_prefetch_many(
                impl_,
                layer,
                expert_ids.data(),
                expert_ids.size());
        if (rc != 0) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to prefetch MoE experts: ") + err);
            }
            throw std::runtime_error("failed to prefetch MoE experts");
        }
    }

    std::pair<int32_t, size_t> prefetch_next_layer_mandatory(int32_t current_layer) {
        int32_t target_layer = -1;
        const int32_t count = llama_expert_manager_prefetch_next_layer_mandatory(
                impl_, current_layer, &target_layer);
        if (count < 0) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(
                        std::string("failed to prefetch mandatory next MoE layer: ") + err);
            }
            throw std::runtime_error("failed to prefetch mandatory next MoE layer");
        }
        if (count == 0) {
            return {-1, 0};
        }
        return {target_layer, static_cast<size_t>(count)};
    }

    std::vector<ExpertPrediction> predict_prefetch_next(
            int32_t layer,
            std::vector<int32_t> experts,
            std::vector<float> router_scores,
            size_t max_predictions,
            uint64_t current_token_id,
            uint64_t deadline_ns) {
        if (!router_scores.empty() && router_scores.size() != experts.size()) {
            throw std::invalid_argument("router score count must match expert count");
        }
        if (router_scores.empty()) {
            std::sort(experts.begin(), experts.end());
            experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        } else {
            std::vector<std::pair<int32_t, float>> routes;
            routes.reserve(experts.size());
            for (size_t i = 0; i < experts.size(); ++i) {
                routes.emplace_back(experts[i], router_scores[i]);
            }
            std::sort(routes.begin(), routes.end(),
                    [](const auto & left, const auto & right) {
                        return left.first < right.first;
                    });
            experts.clear();
            router_scores.clear();
            for (const auto & route : routes) {
                if (!experts.empty() && experts.back() == route.first) {
                    router_scores.back() = std::max(router_scores.back(), route.second);
                } else {
                    experts.push_back(route.first);
                    router_scores.push_back(route.second);
                }
            }
        }
        if (experts.empty() || max_predictions == 0) {
            return {};
        }

        std::vector<llama_expert_prediction_ffi> raw(max_predictions);
        const int32_t count = llama_expert_manager_predict_prefetch_next(
                impl_,
                layer,
                experts.data(),
                experts.size(),
                router_scores.empty() ? nullptr : router_scores.data(),
                max_predictions,
                current_token_id,
                deadline_ns,
                raw.data(),
                raw.size());
        if (count < 0) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to predict/prefetch MoE experts: ") + err);
            }
            throw std::runtime_error("failed to predict/prefetch MoE experts");
        }

        std::vector<ExpertPrediction> predictions;
        predictions.reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) {
            predictions.push_back({
                raw[static_cast<size_t>(i)].layer,
                raw[static_cast<size_t>(i)].expert,
                raw[static_cast<size_t>(i)].probability,
                raw[static_cast<size_t>(i)].deadline_ns,
            });
        }
        return predictions;
    }

    ExpertTicket submit_batch_async(
            std::vector<ExpertKey> experts,
            uint64_t token_id = UINT64_MAX) {
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());

        if (experts.empty()) {
            return ExpertTicket(this, nullptr);
        }

        const int32_t layer = experts.front().layer;
        std::vector<int32_t> expert_ids;
        expert_ids.reserve(experts.size());
        for (const ExpertKey & key : experts) {
            if (key.layer != layer) {
                throw std::runtime_error("failed to batch submit MoE experts from multiple layers");
            }
            expert_ids.push_back(key.expert);
        }

        llama_expert_ticket_ffi * ticket = llama_expert_manager_submit_batch_async(
                impl_,
                layer,
                expert_ids.data(),
                expert_ids.size(),
                token_id);
        if (ticket == nullptr) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to submit MoE expert batch: ") + err);
            }
            throw std::runtime_error("failed to submit MoE expert batch");
        }

        return ExpertTicket(this, ticket);
    }

    void register_slice(int32_t layer, int32_t expert, const ExpertSlice & slice) {
        llama_expert_slice_ffi ffi_slice = to_ffi(slice);
        const int32_t rc = llama_expert_manager_register_slice(impl_, layer, expert, &ffi_slice);
        if (rc != 0) {
            throw std::runtime_error("failed to register MoE expert slice");
        }
    }

    void register_file(int32_t file_id, const std::string & path) {
        const int32_t rc = llama_expert_manager_register_file(impl_, file_id, path.c_str());
        if (rc != 0) {
            throw std::runtime_error("failed to register MoE expert file");
        }
    }

    size_t write_background_file(
            const std::string & path,
            const std::vector<uint8_t> & data,
            int32_t kv_slot_id) {
        size_t written = 0;
        const int32_t rc = llama_expert_manager_write_background_file(
                impl_,
                path.c_str(),
                data.data(),
                data.size(),
                kv_slot_id,
                &written);
        if (rc != 0) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(
                        std::string("failed to write KV state through P4 scheduler: ") + err);
            }
            throw std::runtime_error("failed to write KV state through P4 scheduler");
        }
        if (written != data.size()) {
            throw std::runtime_error("P4 scheduler returned a short KV state write");
        }
        return written;
    }

    void release(llama_expert_handle_ffi * handle) {
        if (handle != nullptr) {
            llama_expert_manager_release(impl_, handle);
        }
    }

    ExpertHandle ticket_get_handle(const ExpertTicket & ticket, int32_t layer, int32_t expert) {
        (void) ticket;
        llama_expert_handle_ffi * handle = llama_expert_ticket_get_handle(ticket.ticket_, layer, expert);
        if (handle == nullptr) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to get MoE expert handle from ticket: ") + err);
            }
            throw std::runtime_error("failed to get MoE expert handle from ticket");
        }
        return ExpertHandle(this, handle);
    }

    ExpertHandle ticket_try_get_handle(const ExpertTicket & ticket, int32_t layer, int32_t expert) {
        (void) ticket;
        llama_expert_handle_ffi * handle = llama_expert_ticket_try_get_handle(ticket.ticket_, layer, expert);
        if (handle == nullptr) {
            return ExpertHandle();
        }
        return ExpertHandle(this, handle);
    }

    std::pair<int32_t, ExpertHandle> ticket_wait_any_ready(
            const ExpertTicket & ticket,
            int32_t layer,
            const int32_t * experts,
            size_t count) {
        (void) ticket;
        int32_t ready_expert = -1;
        llama_expert_handle_ffi * handle = llama_expert_ticket_wait_any_ready(
                ticket.ticket_,
                layer,
                experts,
                count,
                &ready_expert);
        if (handle == nullptr) {
            const char * err = llama_expert_manager_last_error_message();
            if (err != nullptr && err[0] != '\0') {
                throw std::runtime_error(std::string("failed to wait for ready MoE expert from ticket: ") + err);
            }
            throw std::runtime_error("failed to wait for ready MoE expert from ticket");
        }
        return { ready_expert, ExpertHandle(this, handle) };
    }

    void release_ticket(llama_expert_ticket_ffi * ticket) {
        if (ticket != nullptr) {
            llama_expert_ticket_release(ticket);
            llama_expert_ticket_free(ticket);
        }
    }

    std::pair<size_t, size_t> stats() const {
        size_t hit = 0;
        size_t miss = 0;
        if (llama_expert_manager_stats(impl_, &hit, &miss) != 0) {
            return {0, 0};
        }
        return {hit, miss};
    }

private:
    void close() {
        if (impl_ != nullptr && owns_impl_) {
            llama_expert_manager_free(impl_);
        }
        impl_ = nullptr;
        owns_impl_ = false;
        capacity_ = 0;
    }

    llama_expert_manager_ffi * impl_ = nullptr;
    bool owns_impl_ = false;
    TensorLayout layout_;
    size_t capacity_ = 0;

    friend class ExpertTicket;
    friend class ExpertHandle;
};

inline uint8_t * ExpertHandle::host_ptr(MoePart part) const {
    if (manager_ != nullptr && handle_ != nullptr) {
        return manager_->handle_host_ptr(*this, part);
    }
    return nullptr;
}

inline const uint8_t * ExpertHandle::device_ptr(MoePart part) const {
    if (manager_ != nullptr && handle_ != nullptr) {
        return manager_->handle_device_ptr(*this, part);
    }
    return nullptr;
}

inline size_t ExpertHandle::part_size(MoePart part) const {
    if (manager_ != nullptr && handle_ != nullptr) {
        return manager_->handle_part_size(*this, part);
    }
    return 0;
}

inline void ExpertHandle::reset() {
    if (manager_ != nullptr && handle_ != nullptr) {
        manager_->release(handle_);
    }
    manager_ = nullptr;
    handle_ = nullptr;
}

inline ExpertHandle ExpertTicket::get_handle(int32_t layer, int32_t expert) const {
    if (manager_ == nullptr || ticket_ == nullptr) {
        throw std::runtime_error("MoE expert ticket is not valid");
    }
    return manager_->ticket_get_handle(*this, layer, expert);
}

inline ExpertHandle ExpertTicket::try_get_handle(int32_t layer, int32_t expert) const {
    if (manager_ == nullptr || ticket_ == nullptr) {
        return ExpertHandle();
    }
    return manager_->ticket_try_get_handle(*this, layer, expert);
}

inline std::pair<int32_t, ExpertHandle> ExpertTicket::wait_any_ready(
        int32_t layer,
        const int32_t * experts,
        size_t count) const {
    if (manager_ == nullptr || ticket_ == nullptr) {
        throw std::runtime_error("MoE expert ticket is not valid");
    }
    return manager_->ticket_wait_any_ready(*this, layer, experts, count);
}

inline void ExpertTicket::release() {
    if (ticket_ != nullptr) {
        llama_expert_ticket_release(ticket_);
    }
}

inline void ExpertTicket::reset() {
    if (manager_ != nullptr && ticket_ != nullptr) {
        manager_->release_ticket(ticket_);
    }
    manager_ = nullptr;
    ticket_ = nullptr;
}
