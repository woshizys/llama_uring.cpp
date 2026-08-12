#pragma once

#include "llama-expert-manager.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ExpertPackPart {
    MoePart part = MoePart::Up;
    std::string tensor_name;
    std::string tensor_type;
    uint64_t source_offset = 0;
    uint64_t pack_offset = 0;
    size_t valid_length = 0;
    std::string sha256;
};

struct ExpertPackObject {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t offset = 0;
    size_t valid_length = 0;
    size_t padded_length = 0;
    std::string valid_sha256;
    std::vector<ExpertPackPart> parts;
};

class ExpertPackManifest {
public:
    static ExpertPackManifest load(const std::string & manifest_path);

    const ExpertPackObject * find(int32_t layer, int32_t expert) const;
    const ExpertPackPart * find_part(
            int32_t layer,
            int32_t expert,
            MoePart part) const;

    const std::string & manifest_path() const {
        return manifest_path_;
    }

    const std::string & pack_path() const {
        return pack_path_;
    }

    const std::string & model_sha256() const {
        return model_sha256_;
    }

    const std::string & architecture() const {
        return architecture_;
    }

    uint64_t model_size_bytes() const {
        return model_size_bytes_;
    }

    uint64_t pack_size_bytes() const {
        return pack_size_bytes_;
    }

    size_t alignment() const {
        return alignment_;
    }

    const std::vector<MoePart> & part_order() const {
        return part_order_;
    }

    const std::vector<ExpertPackObject> & objects() const {
        return objects_;
    }

private:
    static uint64_t key(int32_t layer, int32_t expert);

    std::string manifest_path_;
    std::string pack_path_;
    std::string model_sha256_;
    std::string architecture_;
    uint64_t model_size_bytes_ = 0;
    uint64_t pack_size_bytes_ = 0;
    size_t alignment_ = 0;
    std::vector<MoePart> part_order_;
    std::vector<ExpertPackObject> objects_;
    std::unordered_map<uint64_t, size_t> object_index_;
};
