#include "llama-expert-pack.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

MoePart parse_part(const std::string & value) {
    if (value == "up") {
        return MoePart::Up;
    }
    if (value == "gate") {
        return MoePart::Gate;
    }
    if (value == "down") {
        return MoePart::Down;
    }
    if (value == "gate_up") {
        return MoePart::GateUp;
    }
    throw std::runtime_error("unsupported Expert pack part: " + value);
}

bool valid_sha256(const std::string & value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        });
}

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

template<typename T>
T required(const json & value, const char * name) {
    if (!value.contains(name)) {
        throw std::runtime_error(std::string("Expert pack manifest is missing '") + name + "'");
    }
    return value.at(name).get<T>();
}

fs::path resolve_pack_path(const fs::path & manifest_path, const json & pack) {
    fs::path configured(required<std::string>(pack, "path"));
    if (!configured.is_absolute()) {
        configured = manifest_path.parent_path() / configured;
    }
    if (fs::is_regular_file(configured)) {
        return fs::canonical(configured);
    }

    if (pack.contains("file_name")) {
        fs::path colocated = manifest_path.parent_path() /
            pack.at("file_name").get<std::string>();
        if (fs::is_regular_file(colocated)) {
            return fs::canonical(colocated);
        }
    }
    throw std::runtime_error("Expert pack file does not exist: " + configured.string());
}

} // namespace

uint64_t ExpertPackManifest::key(int32_t layer, int32_t expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(expert);
}

ExpertPackManifest ExpertPackManifest::load(const std::string & manifest_path_text) {
    fs::path manifest_path = fs::absolute(manifest_path_text);
    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error(
            "failed to open Expert pack manifest: " + manifest_path.string());
    }

    json root;
    try {
        input >> root;
    } catch (const json::exception & error) {
        throw std::runtime_error(
            "failed to parse Expert pack manifest: " + std::string(error.what()));
    }

    if (required<int>(root, "schema_version") != 1 ||
        required<std::string>(root, "format") != "pdcat-expert-pack") {
        throw std::runtime_error("unsupported Expert pack manifest version or format");
    }
    if (!root.contains("verification") ||
        !root.at("verification").value("valid", false)) {
        throw std::runtime_error("Expert pack manifest is not marked as verified");
    }

    ExpertPackManifest result;
    result.manifest_path_ = fs::canonical(manifest_path).string();
    result.alignment_ = required<size_t>(root, "alignment");
    if (result.alignment_ < 4096 || !is_power_of_two(result.alignment_)) {
        throw std::runtime_error(
            "Expert pack alignment must be a power of two and at least 4096");
    }

    const json & model = root.at("model");
    result.model_size_bytes_ = required<uint64_t>(model, "size_bytes");
    result.model_sha256_ = required<std::string>(model, "sha256");
    result.architecture_ = required<std::string>(model, "architecture");
    if (!valid_sha256(result.model_sha256_)) {
        throw std::runtime_error("Expert pack model SHA-256 is invalid");
    }

    const json & pack = root.at("pack");
    fs::path pack_path = resolve_pack_path(manifest_path, pack);
    result.pack_path_ = pack_path.string();
    result.pack_size_bytes_ = required<uint64_t>(pack, "size_bytes");
    if (fs::file_size(pack_path) != result.pack_size_bytes_) {
        throw std::runtime_error("Expert pack file size does not match manifest");
    }
    const std::string pack_sha256 = required<std::string>(pack, "sha256");
    if (!valid_sha256(pack_sha256)) {
        throw std::runtime_error("Expert pack SHA-256 is invalid");
    }

    const size_t declared_count = required<size_t>(root, "object_count");
    const json & objects = root.at("objects");
    if (!objects.is_array() || objects.size() != declared_count || objects.empty()) {
        throw std::runtime_error("Expert pack object count is invalid");
    }

    uint64_t previous_end = 0;
    std::vector<MoePart> expected_part_order;
    result.objects_.reserve(objects.size());
    result.object_index_.reserve(objects.size());
    for (const json & value : objects) {
        ExpertPackObject object;
        object.layer = required<int32_t>(value, "layer");
        object.expert = required<int32_t>(value, "expert");
        object.offset = required<uint64_t>(value, "offset");
        object.valid_length = required<size_t>(value, "valid_length");
        object.padded_length = required<size_t>(value, "padded_length");
        object.valid_sha256 = required<std::string>(value, "valid_sha256");

        if (object.layer < 0 || object.expert < 0 ||
            object.offset % result.alignment_ != 0 ||
            object.padded_length % result.alignment_ != 0 ||
            object.valid_length == 0 ||
            object.valid_length > object.padded_length ||
            object.offset < previous_end ||
            object.offset > result.pack_size_bytes_ ||
            object.padded_length > result.pack_size_bytes_ - object.offset ||
            !valid_sha256(object.valid_sha256)) {
            throw std::runtime_error("invalid Expert pack object bounds or checksum");
        }

        const json & parts = value.at("parts");
        if (!parts.is_array() || parts.empty()) {
            throw std::runtime_error("Expert pack object has no parts");
        }
        uint64_t next_part_offset = object.offset;
        size_t part_bytes = 0;
        std::vector<MoePart> this_part_order;
        std::unordered_set<int32_t> seen_parts;
        for (const json & part_value : parts) {
            ExpertPackPart part;
            part.part = parse_part(required<std::string>(part_value, "part"));
            part.tensor_name = required<std::string>(part_value, "tensor_name");
            part.tensor_type = required<std::string>(part_value, "tensor_type");
            part.source_offset = required<uint64_t>(part_value, "source_offset");
            part.pack_offset = required<uint64_t>(part_value, "pack_offset");
            part.valid_length = required<size_t>(part_value, "valid_length");
            part.sha256 = required<std::string>(part_value, "sha256");

            if (part.tensor_name.empty() || part.tensor_type.empty() ||
                part.valid_length == 0 || part.pack_offset != next_part_offset ||
                part.pack_offset > result.pack_size_bytes_ ||
                part.valid_length > result.pack_size_bytes_ - part.pack_offset ||
                !valid_sha256(part.sha256) ||
                !seen_parts.insert(static_cast<int32_t>(part.part)).second) {
                throw std::runtime_error("invalid Expert pack part");
            }
            next_part_offset += part.valid_length;
            part_bytes += part.valid_length;
            this_part_order.push_back(part.part);
            object.parts.push_back(std::move(part));
        }
        if (part_bytes != object.valid_length ||
            next_part_offset != object.offset + object.valid_length) {
            throw std::runtime_error("Expert pack object parts are not contiguous");
        }
        if (expected_part_order.empty()) {
            expected_part_order = this_part_order;
        } else if (expected_part_order != this_part_order) {
            throw std::runtime_error(
                "Expert pack v1 requires a consistent part order for every object");
        }

        const uint64_t object_key = key(object.layer, object.expert);
        if (!result.object_index_.emplace(
                object_key, result.objects_.size()).second) {
            throw std::runtime_error("duplicate layer/expert in Expert pack manifest");
        }
        previous_end = object.offset + object.padded_length;
        result.objects_.push_back(std::move(object));
    }

    result.part_order_ = std::move(expected_part_order);
    return result;
}

const ExpertPackObject * ExpertPackManifest::find(
        int32_t layer,
        int32_t expert) const {
    auto it = object_index_.find(key(layer, expert));
    return it == object_index_.end() ? nullptr : &objects_[it->second];
}

const ExpertPackPart * ExpertPackManifest::find_part(
        int32_t layer,
        int32_t expert,
        MoePart part) const {
    const ExpertPackObject * object = find(layer, expert);
    if (object == nullptr) {
        return nullptr;
    }
    for (const ExpertPackPart & candidate : object->parts) {
        if (candidate.part == part) {
            return &candidate;
        }
    }
    return nullptr;
}
