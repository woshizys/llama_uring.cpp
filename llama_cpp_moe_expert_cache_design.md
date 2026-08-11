# llama.cpp MoE Expert Cache 设计与当前实现

本文档记录当前 `llama_uring.cpp` 与 `InterfaceIO/io-scheduler` 的 MoE expert cache 接入设计和最新实现状态。

## 1. 目标

目标是在 llama.cpp 推理 MoE 模型时，以 `(layer, expert)` 为粒度显式管理 expert 参数缓存：

- 低 batch decode 场景下，在 expert 权重真正参与 `GGML_OP_MUL_MAT_ID` 计算前执行 `ensure()`；
- cache hit 时直接返回 RAII handle，计算期间 pin 住 expert，避免被 LRU 驱逐；
- cache miss 时由 Rust `io-scheduler` 通过 uring 从 GGUF 文件加载 expert 参数；
- cache 容量小于总 expert 参数量，通过 expert 粒度 LRU 替换；
- llama.cpp 只负责模型元信息提取、推理路径 hook、FFI 调用，不再实现重复的 cache / LRU / I/O 逻辑；
- `io-scheduler` 是唯一的 expert cache 管理后端。

核心原则：

```text
llama.cpp:
  识别 MoE tensor -> 记录 GGUF file path / offset -> 推理时调用 FFI ensure/get_ptr/release

Rust FFI:
  C ABI 边界 -> tokio runtime -> io_scheduler::LlamaExpertManager

io-scheduler:
  ExpertMeta 合并 -> file_id 到 fd 映射 -> uring 加载 -> LRU slot cache -> RAII handle
```

## 2. 当前总体架构

```text
llama_model_load
  |
  | load_hparams() 得到 n_embd / n_ff_exp / n_expert
  |
  | 如果 params.expert_cache_capacity > 0 且没有外部 manager:
  |   自动创建 Rust ExpertManager
  |
  v
llama_model::load_tensors()
  |
  | 禁用 mmap expert cache 路径
  | 标记 CPU MoE expert tensor 为 external，并 skip 全量加载
  | 注册 GGUF file_id -> file_path
  | 注册每个 (layer, expert, part) -> file_id / offset / size
  v
推理执行 GGML_OP_MUL_MAT_ID
  |
  | thread 0 从 ids 收集本次 op 用到的 experts
  | ExpertManager.ensure(layer, expert)
  | hit: 返回 handle
  | miss: io-scheduler 通过 uring 加载到 LRU slot
  v
GGML CPU mul_mat_id
  |
  | 按 expert id 和 part 获取 host_ptr(part)
  | 使用 cache slot 中的 expert 参数计算
  v
op 结束
  |
  | 释放 C++ op handle
  | Rust opaque handle drop
  | io-scheduler LRU 引用计数 -1
```

## 3. 关键数据模型

### 3.1 ExpertKey

LRU key 是 `(layer, expert)`：

```rust
pub struct ExpertKey(usize, usize);
```

每个 key 表示一个逻辑 expert，包含该 expert 所需的所有 MoE part，例如 `Up`、`Gate`、`Down`、`GateUp`。

### 3.2 MoePart

当前统一使用 `io-scheduler` 中的定义，FFI 不再维护重复 enum：

```rust
#[repr(i32)]
pub enum MoePart {
    Up = 0,
    Gate = 1,
    Down = 2,
    GateUp = 3,
}
```

### 3.3 Slice 与 TensorMeta

llama.cpp 注册给 Rust 的 slice 只包含文件位置，不包含目标内存地址，也不包含 direct I/O 对齐计划：

```rust
pub struct Slice {
    pub file_id: i32,
    pub file_offset: usize,
    pub file_size: usize,
}

pub struct TensorMeta {
    pub part: MoePart,
    pub layer: usize,
    pub expert: usize,
    pub slice: Slice,
}
```

direct I/O 的 `align_down / align_up / bounce buffer` 由 `io-scheduler` worker 在加载时统一处理。llama.cpp 不再维护读取对齐计划。

### 3.4 ExpertMeta

`ExpertMeta` 按 `(layer, expert)` 聚合多个 part：

- `add_part()`：加入一个 `TensorMeta`；
- 自动合并同 file 上连续或重叠的 slice；
- `tensors_index` 记录每个 part 在合并后 buffer 中的位置；
- `offset_in_buffer` 与最终 cache slot 中的内存布局一致；
- `get_part(part)` 返回 part 的 `TensorIndex`。

加载时 worker 按合并后的 `slices` 执行 I/O；计算时 `ExpertHandle::get_part(part)` 使用 `offset_in_buffer` 返回 part 起始指针。

## 4. llama.cpp 侧接入

### 4.1 模型参数

`llama_model_params` 当前包含：

```c
void * expert_manager;
size_t expert_cache_capacity;
bool expert_manager_owned;
```

语义：

- `expert_manager != NULL`：使用外部传入的 Rust manager；
- `expert_cache_capacity > 0 && expert_manager == NULL`：llama.cpp 自动创建 manager；
- `expert_manager_owned == true`：`llama_model` 析构时释放 manager。

默认值：

```c
expert_manager = NULL;
expert_cache_capacity = 0;
expert_manager_owned = false;
```

默认不启用 expert cache，不改变 llama.cpp 原始行为。

### 4.2 自动创建 ExpertManager

自动创建发生在 `llama_model::load_tensors()` 开始处。此时 hparams 与 GGUF tensor metadata 已经加载，可以直接读取 MoE tensor 的实际 stride。

当满足以下条件时自动创建：

```text
params.expert_manager == nullptr
params.expert_cache_capacity > 0
hparams.n_expert > 0
hparams.n_ff_exp > 0
```

创建参数不再用 `precision_bits` 估算，而是按每层每个 expert 的实际 tensor stride 计算：

```cpp
slot_size = max_over_layers(
    ffn_up_exps.nb[2] +
    ffn_gate_exps.nb[2] +
    ffn_down_exps.nb[2] +
    ffn_gate_up_exps.nb[2]);

ExpertManager::create_with_slot_size(params.expert_cache_capacity, slot_size);
```

其中 `nb[2]` 是 GGML/GGUF 中一个 expert 在该 tensor 内的真实字节跨度，包含量化 block 的 scale、布局和 padding。因此 Q8_0 模型会按 Q8_0 tensor 的实际大小创建 slot。创建成功后安装 ggml MoE expert callback。

### 4.3 file_id 到 file_path

`file_id` 来自 llama.cpp 的 `llama_model_loader::files` 下标，不是 OS fd。

因此 `llama_model_loader` 保存：

```cpp
std::vector<std::string> file_paths;
```

规则：

- 主 GGUF 文件加入 `files` 后，记录 `fname`；
- split GGUF 文件加入 `files` 后，记录 `fname_split`；
- `file_paths[i]` 与 `files[i]`、`llama_tensor_weight.idx` 保持一致。

在注册 expert slice 前，llama.cpp 先执行：

```cpp
expert_manager->register_file(file_id, ml.file_paths[file_id]);
```

Rust 后端通过 uring open 文件并保存 `file_id -> fd` 映射。

如果模型通过 `FILE *` 加载而没有 path，expert cache 当前不支持该路径，因为无法稳定建立 `file_id -> path -> fd` 映射。

### 4.4 expert tensor skip-load

启用 expert cache 时，CPU 上的 MoE expert tensors 会被标记为 external，并加入 loader skip 列表：

- `ffn_up_exps`
- `ffn_gate_exps`
- `ffn_down_exps`
- `ffn_gate_up_exps`

这样 llama.cpp 不会为全量 expert tensor 分配常规 backend buffer，也不会在模型加载阶段把所有 expert 参数读入内存。

### 4.5 注册 expert slice

模型加载完成 tensor metadata 后，llama.cpp 根据 `weights_map` 获取每个 MoE tensor 的：

- `file_id = weight.idx`
- `tensor_file_offset = weight.offs`
- `expert_stride = tensor->nb[2]`
- `n_expert = tensor->ne[2]`

然后为每个 expert 注册：

```cpp
ExpertSlice {
    part,
    file_id,
    file_offset = tensor_file_offset + expert_stride * expert,
    file_size = expert_stride,
}
```

### 4.6 推理时 ensure / get pointer / release

`GGML_OP_MUL_MAT_ID` CPU compute path 使用 callback：

1. thread 0 从 `ids` tensor 收集本次 op 用到的 expert id；
2. 对每个 `(layer, expert)` 调用 `ExpertManager::ensure_map()`；
3. C++ op handle 保存 `expert_id -> ExpertHandle`；
4. worker 线程计算每个 expert 时调用 `host_ptr(part)`；
5. `host_ptr(part)` 通过 Rust FFI 返回 cache slot 中该 part 的真实起始地址；
6. op 完成后释放 op handle；
7. Rust handle drop，LRU 引用计数递减。

### 4.7 external tensor 分配边界

MoE expert tensor 被标记为 `GGML_TENSOR_FLAG_EXTERNAL` 后，必须同时从以下路径排除：

- 模型权重 backend buffer 分配；
- `ggml_backend_alloc_ctx_tensors_from_buft_impl()` 的 ctx tensor 预估与分配；
- graph allocator 的 `sched_reserve` / `ggml_gallocr_*` 预留、初始化、释放与重分配判断；
- CUDA backend 的 op support / fusion 检测。

最新修复点：

- `ggml/src/ggml-alloc.c` 已将 external tensor 视为已有外部存储，不再为其分配 graph compute buffer；
- `ggml/src/ggml-cuda/ggml-cuda.cu` 已禁止 CUDA backend 接管 external `GGML_OP_MUL_MAT_ID`，并在 fusion 判断中保护 `src0->buffer == nullptr` 的情况。

这解决了一个关键问题：loader 已经 skip 掉 MoE 权重后，`sched_reserve` 仍可能把 external MoE tensor 作为 graph leaf 重新计入 compute buffer，导致 `capacity=400` 仍然出现异常内存压力。

## 5. Rust FFI 层

位置：

```text
/workspace/llama_uring.cpp/rust/expert-cache-ffi
```

当前 FFI crate 已直接连接 `io-scheduler`。

### 5.1 依赖

```toml
io-scheduler = { path = "../../../InterfaceIO/io-scheduler" }
tokio = { version = "1.0", features = ["rt-multi-thread"] }
```

### 5.2 manager 与 runtime

`llama_expert_manager_ffi` 持有：

```rust
manager: LlamaExpertManager,
runtime: tokio::runtime::Runtime,
```

构造时创建 tokio multi-thread runtime，并在 runtime context 中创建 `LlamaExpertManager`，保证 `io-scheduler` 内部 worker 的 `tokio::spawn` 可用。

### 5.3 C ABI

当前导出的主要 ABI：

```c
llama_expert_manager_ffi * llama_expert_manager_new(
    size_t capacity,
    size_t hidden_dim,
    size_t intermediate_dim,
    size_t precision_bits);

int32_t llama_expert_manager_register_file(
    llama_expert_manager_ffi * manager,
    int32_t file_id,
    const char * path);

int32_t llama_expert_manager_register_slice(
    llama_expert_manager_ffi * manager,
    int32_t layer,
    int32_t expert,
    const llama_expert_slice_ffi * slice);

llama_expert_handle_ffi * llama_expert_manager_ensure(
    llama_expert_manager_ffi * manager,
    int32_t layer,
    int32_t expert);

uint8_t * llama_expert_handle_host_ptr(
    const llama_expert_handle_ffi * handle,
    int32_t part);

const uint8_t * llama_expert_handle_device_ptr(
    const llama_expert_handle_ffi * handle,
    int32_t part);

size_t llama_expert_handle_part_size(
    const llama_expert_handle_ffi * handle,
    int32_t part);

int32_t llama_expert_handle_slot_id(
    const llama_expert_handle_ffi * handle);

void llama_expert_manager_release(
    llama_expert_manager_ffi * manager,
    llama_expert_handle_ffi * handle);

void llama_expert_manager_free(
    llama_expert_manager_ffi * manager);
```

错误信息通过 thread-local `llama_expert_manager_last_error_message()` 暴露。

### 5.4 FFI 职责边界

FFI 的职责只保留 ABI 转换、错误隔离、panic 捕获和 runtime 调度。expert cache 的元数据合并、I/O、LRU、slot 管理和 RAII release 语义全部由 `io-scheduler` 实现。

## 6. io-scheduler 后端

位置：

```text
/workspace/InterfaceIO/io-scheduler/src/expert_manager
```

### 6.1 LlamaExpertManager

对 llama.cpp 暴露的核心接口：

```rust
pub fn new(capacity, hidden_dim, intermediate_dim, precision_bits) -> Self;
pub async fn register_file_core(&self, file_id: i32, path: String) -> SchedulerResult<()>;
pub async fn register_tensor_core(&mut self, tensor_meta: TensorMeta);
pub async fn ensure(&self, expert: ExpertKey) -> SchedulerResult<ExpertHandle>;
```

`ensure()` 返回上层 `ExpertHandle`，它同时持有：

- RAII `CachedExpertHandle`；
- 对应 expert 的 `ExpertMeta`。

这样 `get_part(part)` 可以根据 `TensorIndex.offset_in_buffer` 返回 part 的正确指针。

### 6.2 LocalLoader

`LocalLoader` 负责 uring I/O：

```rust
register_file(file_id, path) -> open path -> fd -> 保存 file_id 到 fd 映射
load_registered(file_id, offset, size, dst) -> resolve fd -> read
```

当前注意点：

- 重复注册同一 `file_id` 会覆盖映射，当前未关闭旧 fd；正常模型加载只注册一次；
- 当前 open flags 是 `O_RDONLY | O_DIRECT`。

### 6.3 ExpertCache

`ExpertCache` 是真正的 LRU slot cache：

- slot 数量由 `capacity` 决定；
- 每个 slot 大小由 `hidden_dim * intermediate_dim * precision_bits / 8 * 3` 决定；
- 当前使用 host pool，并通过 CUDA host register 获取 device pointer；
- cache item 被 handle 引用时引用计数增加；
- handle drop 时 release，引用计数递减；
- LRU 只能驱逐引用计数为 0 的 item。

### 6.4 miss 加载流程

miss 时：

1. `ensure()` 查不到 cache item；
2. worker acquire 一个可用 slot；
3. 按 `ExpertMeta.slices()` 读取合并后的文件 slice；
4. 第一个 slice 可直接读入最终 slot；
5. 后续 slice 使用 `AlignedBuffer` 作为临时 direct I/O buffer；
6. 将 payload 复制到最终 slot 的连续布局；
7. `finish()` 把 slot 插入 LRU cache；
8. `ensure()` 返回 RAII handle。

`AlignedBuffer` 是 owned RAII allocation，避免裸 `*mut u8` 跨 `await` 导致 future 非 `Send`。

## 7. 内存与容量策略

### 7.1 llama.cpp 可用内存查询

ggml backend 存在设备内存查询接口：

```cpp
ggml_backend_dev_memory(device, &free, &total);
ggml_backend_dev_get_props(device, &props); // props.memory_free / props.memory_total
```

llama.cpp 也有 context 级 memory breakdown：

```cpp
llama_get_memory_breakdown(ctx);
```

但该 breakdown 依赖已创建 context，不能作为模型加载前 expert cache 容量的直接输入。

### 7.2 当前为何不用显存 free 自动设置 capacity

当前 `io-scheduler` cache 使用 host allocation + `cudaHostRegister` 映射给 CUDA。它主要消耗 host pinned/mapped memory，而不是普通 ggml backend buffer 或纯 VRAM buffer。

因此仅用 `ggml_backend_dev_memory()` 的显存 free 来决定 expert cache capacity 不可靠。当前采用显式 capacity 参数驱动：

```cpp
params.expert_cache_capacity = N;
```

后续如果要自动容量，建议策略是：

1. 查询系统可用 host memory；
2. 查询 backend/GPU memory 作为辅助信息；
3. 预留安全比例，例如只使用可用 host memory 的 20%-40%；
4. 复用当前自动创建逻辑，按 MoE tensor metadata 的 `nb[2]` 计算真实 `slot_bytes`；
5. `capacity = budget / slot_bytes`；
6. 如果 `cudaHostRegister` 失败，降低 capacity 重试。

## 8. 使用方式

### 8.1 自动创建方式

推荐方式：

```cpp
llama_model_params params = llama_model_default_params();
params.expert_cache_capacity = 16;

llama_model * model = llama_model_load_from_file(path, params);
```

llama.cpp 会在加载 hparams 后自动创建 ExpertManager。

### 8.2 外部传入方式

仍支持外部创建 manager：

```cpp
params.expert_manager = llama_expert_manager_new(capacity, hidden_dim, intermediate_dim, precision_bits);
params.expert_manager_owned = true;
```

如果外部传入了 `expert_manager`，llama.cpp 不会自动创建第二个 manager。

### 8.3 当前推荐命令

端到端验证建议使用 `--single-turn`，避免 `llama-cli` 在 `-p` 首轮回答后继续进入聊天输入循环：

```bash
./tools/io_bw_monitor.sh -i 1 -d nvme1n1p1 -o uring_io.csv -- \
  ./build-expert-cache/bin/llama-cli \
  -m /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf \
  -p "hello, please introduce yourself" \
  -n 64 -c 4096 \
  -b 128 -ub 64 \
  --cpu-moe --expert-cache-capacity 400 \
  --single-turn \
  -dio -lv 3
```

如果需要交互式多轮聊天，可以去掉 `--single-turn`。非交互脚本或监控包装器中建议保留 `--single-turn`。

## 9. 当前限制

- 当前完整接入重点是 CPU `GGML_OP_MUL_MAT_ID` 路径；
- GPU expert tensor 当前仍按原逻辑加载，后续需要 CUDA kernel 使用 `device_ptr(part)` 并用 CUDA event 延迟 release；
- `FILE *` 模型加载路径没有 file path，当前 expert cache 不支持；
- 旧 `llama_expert_manager_new()` / 外部手动创建路径仍有 `precision_bits` 参数；当前 CLI 自动创建路径已移除 `--expert-cache-precision-bits`，并使用 GGUF tensor 的真实 `nb[2]` stride；
- 自动容量尚未实现，目前 capacity 由参数显式指定；
- 外部手动创建路径若仍使用 `hidden_dim * intermediate_dim * precision_bits / 8 * 3`，对量化 tensor 仍只是近似估算；
- `LocalLoader` 重复注册同一 `file_id` 时仍需要补关闭旧 fd 的逻辑；
- 低容量场景可能因为单个 op 同时访问的 expert 数超过 capacity 而 `ensure` 失败，`capacity=16` 仅适合探针，不适合 DeepSeek-V2-Lite 正常推理。

## 10. 验证状态

已完成的构建与验证：

```text
cargo check --offline --manifest-path /workspace/InterfaceIO/io-scheduler/Cargo.toml --lib --message-format=short
cargo check --offline --manifest-path /workspace/llama_uring.cpp/rust/expert-cache-ffi/Cargo.toml --lib --message-format=short
cargo test  --offline --manifest-path /workspace/llama_uring.cpp/rust/expert-cache-ffi/Cargo.toml --lib --message-format=short
git diff --check -- include/llama.h src/llama-model.cpp src/llama-expert-manager.h rust/expert-cache-ffi/src/lib.rs
cmake --build /workspace/llama_uring.cpp/build-expert-cache --target llama-cli -j2
```

最新端到端探针：

```text
model: /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf
params: -p hello -n 1 -c 512 -b 16 -ub 16 --cpu-moe --expert-cache-capacity 400 -dio -lv 3
result: load / sched_reserve / warmup / prompt eval 成功，输出 Hello
```

关键内存日志：

```text
auto-created MoE expert cache: capacity=400, slot_size=8.77 MiB, total_cache=3506.25 MiB
MoE expert cache externalized 78 tensor(s), skipped 14586.00 MiB from llama weight buffers
CUDA0 model buffer size = 1126.47 MiB
CUDA_Host model buffer size = 212.50 MiB
sched_reserve: CUDA0 compute buffer size = 7.38 MiB
sched_reserve: CUDA_Host compute buffer size = 1.52 MiB
```

结论：

- model weight buffer 中没有再加载完整 MoE 参数副本；
- `sched_reserve` 不再把 external MoE tensors 重新计入 compute buffer；
- CUDA backend 不再尝试调度 external MoE `MUL_MAT_ID`；
- `capacity=400` 在小规模 prompt eval 下可正常推理。

尚未完成的验证：

- 原始 `-n 64 -c 4096` 长输出压力测试；
- 长时间多轮交互下的 LRU 驱逐与 fd 生命周期验证；
- GPU expert cache 路径验证。

### 10.1 CLI 退出问题

`llama-cli` 默认是聊天 CLI。使用 `-p` 后首轮回答结束，如果没有 `--single-turn`，程序会继续进入输入循环。此前在 stdin EOF、监控脚本或 Ctrl+C 取消生成后，外层循环可能继续打印空 prompt/空行。

当前修复：

- `common/console.*` 增加 `console::input_eof()`，让 `llama-cli` 能识别 EOF / Ctrl+D 并退出；
- `tools/cli/cli.cpp` 增加 `generation_interrupted`，Ctrl+C 取消生成后不再继续进入空输入循环，而是走正常清理退出；
- 非交互式运行仍建议显式加 `--single-turn`。

## 11. 后续工作

优先级建议：

1. 处理重复 `file_id` 注册时的 fd 覆盖 close；
2. 补全 `ExpertMeta::is_complete()`，在 ensure 前检查必要 MoE part 是否都已注册；
3. 从 GGUF tensor type 推导 `precision_bits` 或直接按实际 registered part size 计算 slot size；
4. 增加 host memory 自动 capacity 策略；
5. 为 GPU expert cache 增加 CUDA event 延迟 release；
6. 执行原始长输出配置的端到端压力测试；
7. 增加 `llama-cli -p` + EOF / Ctrl+C 的回归测试，防止再次出现空行循环。
