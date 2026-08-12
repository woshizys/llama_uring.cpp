llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-expert-pack.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PRIVATE ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_EXPERT_CACHE)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    find_library(LLAMA_EXPERT_CACHE_URING_LIBRARY uring REQUIRED)
    set(LLAMA_EXPERT_CACHE_FFI_MANIFEST ${CMAKE_SOURCE_DIR}/rust/expert-cache-ffi/Cargo.toml)
    set(LLAMA_EXPERT_CACHE_FFI_LIB ${CMAKE_SOURCE_DIR}/rust/expert-cache-ffi/target/release/libllama_expert_cache_ffi.a)

    add_custom_target(llama-expert-cache-ffi-build
        COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path ${LLAMA_EXPERT_CACHE_FFI_MANIFEST}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust/expert-cache-ffi
        COMMENT "Building Rust io-scheduler expert cache FFI"
        VERBATIM)

    add_library(llama-expert-cache-ffi STATIC IMPORTED GLOBAL)
    set_target_properties(llama-expert-cache-ffi PROPERTIES IMPORTED_LOCATION ${LLAMA_EXPERT_CACHE_FFI_LIB})
    add_dependencies(llama-expert-cache-ffi llama-expert-cache-ffi-build)

    target_compile_definitions(llama PRIVATE LLAMA_EXPERT_CACHE_ENABLED)
    target_link_libraries(llama PUBLIC llama-expert-cache-ffi)
    if (UNIX AND NOT APPLE)
        target_link_libraries(llama PUBLIC ${LLAMA_EXPERT_CACHE_URING_LIBRARY} dl pthread m)
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
