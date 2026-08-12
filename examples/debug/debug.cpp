#include "debug.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <nlohmann/json.hpp>


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <unordered_set>

static void print_usage(int /*argc*/, char ** argv) {
    const std::string usage_template = R"(
        example usage:

          Print tensors:

          {prog} -m model.gguf -p "Hello my name is" --verbose

          The tensors to be printed can be filtered with --tensor-filter option.

          Save logits/embeddings:

          {prog} -m model.gguf -p "Hello my name is" --save-logits

          Add --embedding to save embeddings)" "\n";

    // Fix the source code indentation above that is introduced by the raw string literal.
    std::string usage = std::regex_replace(usage_template, std::regex("\\n {8}"), "\n");
    usage = std::regex_replace(usage, std::regex("\\{prog\\}"), argv[0]);
    LOG("%s\n", usage.c_str());
}

static bool has_pooling(llama_context * ctx) {
    switch (llama_pooling_type(ctx)) {
        case LLAMA_POOLING_TYPE_NONE:
        case LLAMA_POOLING_TYPE_UNSPECIFIED:
            return false;
        default:
            return true;
    }
}

struct output_data {
    float *                  data_ptr    = nullptr;
    int                      data_size   = 0;
    std::string              type_suffix;
    std::vector<float>       embd_norm;
    std::string              prompt;
    std::vector<llama_token> tokens;

    output_data(llama_context * ctx, const llama_model * model, const common_params & params) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        tokens = common_tokenize(ctx, params.prompt, add_bos);
        prompt = params.prompt;

        if (params.embedding) {
            const int n_embd       = llama_model_n_embd_out(model);
            const bool pooling     = has_pooling(ctx);
            const int n_embd_count = pooling ? 1 : tokens.size();
            const int n_floats     = n_embd * n_embd_count;

            float * embd_raw = pooling ? llama_get_embeddings_seq(ctx, 0) : llama_get_embeddings(ctx);
            if (embd_raw == nullptr) {
                throw std::runtime_error("failed to get embeddings from the model");
            }

            LOG_DBG("pooling_enabled: %s\n", pooling ? "true" : "false");
            LOG_DBG("n_embd: %d\n", n_embd);
            LOG_DBG("n_floats: %d\n", n_floats);
            LOG_DBG("n_embd_count: %d\n", n_embd_count);

            data_ptr    = embd_raw;
            data_size   = n_floats;
            type_suffix = "-embeddings";

            if (params.embd_normalize >= 0) {
                embd_norm.resize(n_floats);
                for (int i = 0; i < n_embd_count; i++) {
                    common_embd_normalize(embd_raw+i*n_embd, embd_norm.data()+i*n_embd, n_embd, params.embd_normalize);
                }
                data_ptr = embd_norm.data();
            }
        } else {
            const float * logits = llama_get_logits_ith(ctx, tokens.size() - 1);
            const int n_logits = llama_vocab_n_tokens(vocab);

            data_ptr = const_cast<float*>(logits);
            data_size = n_logits;
            type_suffix = "";
        }
    }
};

static void save_output_data(const output_data & output, const std::string & model_name, const std::string & output_dir) {
    std::filesystem::create_directory(output_dir);
    auto base_path = std::filesystem::path{output_dir} / ("llamacpp-" + model_name + output.type_suffix);

    // Save logits/embeddings to binary file.
    {
        std::filesystem::path filepath{base_path.string() + ".bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open binary output file: " + filepath.string());
        }
        file.write(reinterpret_cast<const char*>(output.data_ptr), output.data_size * sizeof(float));
        LOG("Data saved to %s\n", filepath.c_str());
    }

    // Save logits/embeddings to text file.
    {
        std::filesystem::path filepath{base_path.string() + ".txt"};
        std::ofstream file{filepath};
        if (!file) {
            throw std::runtime_error("failed to open text output file: " + filepath.string());
        }
        for (int i = 0; i < output.data_size; i++) {
            file << i << ": " << output.data_ptr[i] << '\n';
        }
        LOG("Data saved to %s\n", filepath.c_str());
    }

    // Save prompt and tokens to text file.
    {
        std::filesystem::path filepath{base_path.string() + "-prompt.txt"};
        std::ofstream file{filepath};
        if (!file) {
            throw std::runtime_error("failed to open prompt output file: " + filepath.string());
        }

        file << "prompt: " << output.prompt << '\n';
        file << "n_tokens: " << output.tokens.size() << '\n';

        file << "token ids: ";
        for (size_t i = 0; i < output.tokens.size(); i++) {
            file << output.tokens[i];
            if (i + 1 < output.tokens.size()) {
                file << ", ";
            }
        }
        file << '\n';
        LOG("Prompt saved to %s\n", filepath.c_str());
    }

    // Save token ids to binary file.
    {
        std::filesystem::path filepath{base_path.string() + "-tokens.bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open tokens binary file: " + filepath.string());
        }
        file.write(reinterpret_cast<const char*>(output.tokens.data()), output.tokens.size() * sizeof(llama_token));
        LOG("Tokens saved to %s\n", filepath.c_str());
    }

}

static void save_generation_trace(
        const std::vector<llama_token> & generated,
        const std::vector<float> & logits,
        int n_vocab,
        const std::string & model_name,
        const std::string & output_dir) {
    std::filesystem::create_directory(output_dir);
    const auto base_path = std::filesystem::path{output_dir} /
            ("llamacpp-" + model_name + "-generation");

    {
        const std::filesystem::path filepath{base_path.string() + "-logits.bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open generation logits file: " + filepath.string());
        }
        file.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
        LOG("Generation logits saved to %s\n", filepath.c_str());
    }

    {
        const std::filesystem::path filepath{base_path.string() + "-tokens.bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open generated tokens file: " + filepath.string());
        }
        file.write(
                reinterpret_cast<const char *>(generated.data()),
                generated.size() * sizeof(llama_token));
        LOG("Generated tokens saved to %s\n", filepath.c_str());
    }

    {
        const std::filesystem::path filepath{base_path.string() + ".txt"};
        std::ofstream file{filepath};
        if (!file) {
            throw std::runtime_error("failed to open generation trace file: " + filepath.string());
        }
        file << "n_steps: " << generated.size() << '\n';
        file << "n_vocab: " << n_vocab << '\n';
        for (size_t step = 0; step < generated.size(); ++step) {
            file << step << ": token=" << generated[step]
                 << " logits_offset=" << step * static_cast<size_t>(n_vocab) << '\n';
        }
        LOG("Generation trace saved to %s\n", filepath.c_str());
    }
}

static void print_tokenized_prompt(llama_context * ctx, const std::vector<llama_token> & tokens, const std::string & prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    LOG("Model add_bos: %s\n", llama_vocab_get_add_bos(vocab) ? "true" : "false");
    LOG("Input prompt: \"%s\"\n", prompt.c_str());
    LOG("Token ids (%zu):\n", tokens.size());

    for (auto id : tokens) {
        std::string piece(128, '\0');
        int n = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, true);
        if (n < 0) {
            LOG_ERR("failed to convert token %d to piece\n", id);
            continue;
        }
        piece.resize(n);
        LOG("%s(%d) ", piece.c_str(), id);
    }
    LOG("\n");
}
using json = nlohmann::ordered_json;

static void set_gate_environment(const char * name, const std::string * value) {
#if defined(_WIN32)
    if (_putenv_s(name, value == nullptr ? "" : value->c_str()) != 0) {
        throw std::runtime_error(std::string("failed to update environment variable ") + name);
    }
#else
    const int rc = value == nullptr
            ? unsetenv(name)
            : setenv(name, value->c_str(), 1);
    if (rc != 0) {
        throw std::runtime_error(std::string("failed to update environment variable ") + name);
    }
#endif
}

static std::vector<json> load_gate_prompts(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Gate A dataset: " + path.string());
    }

    std::vector<json> rows;
    std::unordered_set<std::string> ids;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        json row;
        try {
            row = json::parse(line);
        } catch (const json::exception & error) {
            throw std::runtime_error(
                    path.string() + ":" + std::to_string(line_number) + ": " + error.what());
        }
        if (!row.is_object()
                || !row.contains("prompt_id") || !row["prompt_id"].is_string()
                || !row.contains("prompt") || !row["prompt"].is_string()) {
            throw std::runtime_error(
                    path.string() + ":" + std::to_string(line_number) +
                    ": prompt_id and prompt strings are required");
        }
        const std::string id = row["prompt_id"].get<std::string>();
        const std::string prompt = row["prompt"].get<std::string>();
        if (id.empty() || prompt.empty()
                || !std::regex_match(id, std::regex("^[A-Za-z0-9_.-]+$"))) {
            throw std::runtime_error(
                    path.string() + ":" + std::to_string(line_number) +
                    ": invalid prompt_id or empty prompt");
        }
        if (!ids.insert(id).second) {
            throw std::runtime_error("duplicate Gate A prompt_id: " + id);
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::runtime_error("Gate A dataset is empty: " + path.string());
    }
    return rows;
}

static bool run_gate_dataset(
        llama_context * ctx,
        const common_params & params,
        const std::filesystem::path & dataset_path) {
    if (params.n_predict <= 0) {
        LOG_ERR("Gate A batch mode requires --n-predict > 0\n");
        return false;
    }

    std::vector<json> rows;
    try {
        rows = load_gate_prompts(dataset_path);
    } catch (const std::exception & error) {
        LOG_ERR("%s\n", error.what());
        return false;
    }

    if (const char * raw_limit = std::getenv("PDCAT_GATE_A_MAX_PROMPTS")) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(raw_limit, &end, 10);
        if (end == raw_limit || end == nullptr || end[0] != '\0' || parsed == 0) {
            LOG_ERR("PDCAT_GATE_A_MAX_PROMPTS must be a positive integer\n");
            return false;
        }
        rows.resize(std::min<size_t>(rows.size(), static_cast<size_t>(parsed)));
    }

    const std::filesystem::path output_dir(params.logits_output_dir);
    std::filesystem::create_directories(output_dir);
    std::ofstream logits_output(output_dir / "gate_a_logits.f32", std::ios::binary | std::ios::trunc);
    std::ofstream tokens_output(output_dir / "gate_a_tokens.i32", std::ios::binary | std::ios::trunc);
    std::ofstream results_output(output_dir / "gate_a_results.jsonl", std::ios::out | std::ios::trunc);
    if (!logits_output || !tokens_output || !results_output) {
        LOG_ERR("failed to create Gate A outputs in %s\n", output_dir.c_str());
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto milliseconds = [](auto start, auto finish) {
        return std::chrono::duration<double, std::milli>(finish - start).count();
    };

    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const json & row = rows[row_index];
        const std::string prompt_id = row["prompt_id"].get<std::string>();
        const std::string prompt = row["prompt"].get<std::string>();
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt, add_bos);
        if (prompt_tokens.empty()) {
            LOG_ERR("Gate A prompt %s tokenized to an empty sequence\n", prompt_id.c_str());
            return false;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        try {
            const std::string prefill = "prefill";
            set_gate_environment("PDCAT_REQUEST_ID", &prompt_id);
            set_gate_environment("PDCAT_PHASE", &prefill);
            set_gate_environment("PDCAT_TOKEN_ID", nullptr);
        } catch (const std::exception & error) {
            LOG_ERR("%s\n", error.what());
            return false;
        }

        const auto prompt_start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
            LOG_ERR("Gate A prompt %s failed during prefill\n", prompt_id.c_str());
            return false;
        }
        const auto prompt_finish = std::chrono::steady_clock::now();

        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sampler == nullptr) {
            LOG_ERR("Gate A prompt %s could not create greedy sampler\n", prompt_id.c_str());
            return false;
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        const uint64_t logits_byte_offset = static_cast<uint64_t>(logits_output.tellp());
        const uint64_t tokens_byte_offset = static_cast<uint64_t>(tokens_output.tellp());
        std::vector<llama_token> generated;
        std::vector<double> decode_input_ms;
        generated.reserve(static_cast<size_t>(params.n_predict));

        for (int32_t step = 0; step < params.n_predict; ++step) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                LOG_ERR("Gate A prompt %s has no logits at step %d\n", prompt_id.c_str(), step);
                llama_sampler_free(sampler);
                return false;
            }
            logits_output.write(
                    reinterpret_cast<const char *>(logits),
                    static_cast<std::streamsize>(n_vocab) * sizeof(float));

            const llama_token token = llama_sampler_sample(sampler, ctx, -1);
            llama_sampler_accept(sampler, token);
            generated.push_back(token);
            if (llama_vocab_is_eog(vocab, token) || step + 1 == params.n_predict) {
                break;
            }

            try {
                const std::string decode = "decode";
                const std::string token_id = std::to_string(step);
                set_gate_environment("PDCAT_PHASE", &decode);
                set_gate_environment("PDCAT_TOKEN_ID", &token_id);
            } catch (const std::exception & error) {
                LOG_ERR("%s\n", error.what());
                llama_sampler_free(sampler);
                return false;
            }
            const auto decode_start = std::chrono::steady_clock::now();
            if (llama_decode(ctx, llama_batch_get_one(&generated.back(), 1))) {
                LOG_ERR("Gate A prompt %s failed during decode step %d\n", prompt_id.c_str(), step);
                llama_sampler_free(sampler);
                return false;
            }
            decode_input_ms.push_back(milliseconds(
                    decode_start, std::chrono::steady_clock::now()));
        }
        llama_sampler_free(sampler);

        tokens_output.write(
                reinterpret_cast<const char *>(generated.data()),
                static_cast<std::streamsize>(generated.size() * sizeof(llama_token)));
        if (!logits_output || !tokens_output) {
            LOG_ERR("failed to write Gate A raw output for %s\n", prompt_id.c_str());
            return false;
        }

        json result = {
            {"schema_version", 1},
            {"prompt_id", prompt_id},
            {"prompt_sha256", row.value("prompt_sha256", "")},
            {"source_id", row.value("source_id", "")},
            {"split", row.value("split", "")},
            {"category", row.value("category", "")},
            {"prompt_token_count", prompt_tokens.size()},
            {"generated_tokens", generated},
            {"n_vocab", n_vocab},
            {"logits_step_count", generated.size()},
            {"logits_byte_offset", logits_byte_offset},
            {"logits_byte_count", generated.size() * static_cast<size_t>(n_vocab) * sizeof(float)},
            {"tokens_byte_offset", tokens_byte_offset},
            {"tokens_byte_count", generated.size() * sizeof(llama_token)},
            {"prefill_ms", milliseconds(prompt_start, prompt_finish)},
            {"decode_input_ms", decode_input_ms},
        };
        results_output << result.dump() << '\n';
        results_output.flush();
        LOG_INF(
                "[PDCAT][gate-a] prompt=%zu/%zu id=%s prompt_tokens=%zu generated=%zu\n",
                row_index + 1, rows.size(), prompt_id.c_str(),
                prompt_tokens.size(), generated.size());
    }

    LOG_INF(
            "[PDCAT][gate-a] completed prompts=%zu dataset=%s output=%s\n",
            rows.size(), dataset_path.c_str(), output_dir.c_str());
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    print_tokenized_prompt(ctx, tokens, params.prompt);

    if (params.save_logits) {
        try {
            output_data output {ctx, model, params};
            std::filesystem::path model_path{params.model.path};
            std::string model_name{model_path.stem().string()};
            save_output_data(output, model_name, params.logits_output_dir);
        } catch (const std::exception & e) {
            LOG_ERR("%s : error saving logits: %s\n", __func__, e.what());
        }
    }

    const int32_t n_predict = std::max<int32_t>(0, params.n_predict);
    if (n_predict > 0) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> generated;
        std::vector<float> generation_logits;
        generated.reserve(static_cast<size_t>(n_predict));
        generation_logits.reserve(static_cast<size_t>(n_predict) * static_cast<size_t>(n_vocab));

        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sampler == nullptr) {
            LOG_ERR("%s : failed to create greedy correctness sampler\n", __func__);
            return false;
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        for (int32_t step = 0; step < n_predict; ++step) {
            const float * step_logits = llama_get_logits_ith(ctx, -1);
            if (step_logits == nullptr) {
                LOG_ERR("%s : missing logits at generation step %d\n", __func__, step);
                llama_sampler_free(sampler);
                return false;
            }
            generation_logits.insert(generation_logits.end(), step_logits, step_logits + n_vocab);

            const llama_token token = llama_sampler_sample(sampler, ctx, -1);
            llama_sampler_accept(sampler, token);
            generated.push_back(token);
            if (llama_vocab_is_eog(vocab, token) || step + 1 == n_predict) {
                break;
            }
            if (llama_decode(ctx, llama_batch_get_one(&generated.back(), 1))) {
                LOG_ERR("%s : failed to eval generated token at step %d\n", __func__, step);
                llama_sampler_free(sampler);
                return false;
            }
        }
        llama_sampler_free(sampler);

        LOG("Generated token ids (%zu):", generated.size());
        for (const llama_token token : generated) {
            LOG(" %d", token);
        }
        LOG("\n");

        if (params.save_logits) {
            try {
                const std::filesystem::path model_path{params.model.path};
                save_generation_trace(
                        generated,
                        generation_logits,
                        n_vocab,
                        model_path.stem().string(),
                        params.logits_output_dir);
            } catch (const std::exception & e) {
                LOG_ERR("%s : error saving generation trace: %s\n", __func__, e.what());
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    std::optional<common_debug_cb_user_data> cb_data;
    if (!params.save_logits) {
        cb_data.emplace(params, params.tensor_filter);
    }

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    const char * gate_dataset = std::getenv("PDCAT_GATE_A_DATASET");
    if (gate_dataset != nullptr && gate_dataset[0] != '\0') {
        if (!run_gate_dataset(ctx, params, gate_dataset)) {
            return 1;
        }
    } else if (!run(ctx, params)) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
