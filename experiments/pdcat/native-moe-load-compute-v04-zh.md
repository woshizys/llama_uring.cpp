# 原生 llama.cpp MoE prefill 加载/计算实测（v0.4）

## 结论

在 Jetson Orin NX、DeepSeek-V2-Lite-Q8_0、原生 mmap + `--cpu-moe` 路径上，
128/256-token prefill 都明显受 expert 加载支配。四个冷启动样本中，路由 expert
从 GGUF mmap 缺页并搬运到 CUDA buffer 的时间占完整 prompt eval 的
81.51%–84.26%；CUDA expert GEMM 只占 8.86%–11.06%。以已经被 trace 覆盖的
expert 加载与 GEMM 两阶段为分母，加载占 88.15%–90.49%。

因此，在这台设备和该模型上，prefill 并不是“计算过长导致 I/O 收益被掩盖”；
相反，原生路径仍有很大的 I/O 优化空间。128 和 256 token 都适合作为后续
prefetch/overlap 实验点，目前没有因为原生计算占主导而迁移到其他 GPU 的必要。

## 实测结果

每个格子均为一次独立冷启动测量（`n=1`），不是估算。
这些数值来自在 load/compute 边界同步的分解 trace；无 trace 对照及其限制见后文。

| cgroup 限额 | prompt | prompt eval | prompt tok/s | expert 加载 | 加载 / prompt | expert CUDA | CUDA / prompt | 其余阶段 | 加载 / (加载+expert CUDA) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | 128 | 14,275.80 ms | 8.97 | 12,029.06 ms | 84.26% | 1,264.68 ms | 8.86% | 982.07 ms (6.88%) | 90.49% |
| 4 GiB | 256 | 16,540.17 ms | 15.48 | 13,481.31 ms | 81.51% | 1,803.66 ms | 10.90% | 1,255.20 ms (7.59%) | 88.20% |
| 8 GiB | 128 | 12,041.60 ms | 10.63 | 10,036.39 ms | 83.35% | 1,248.75 ms | 10.37% | 756.47 ms (6.28%) | 88.93% |
| 8 GiB | 256 | 16,135.85 ms | 15.87 | 13,275.99 ms | 82.28% | 1,784.77 ms | 11.06% | 1,075.09 ms (6.66%) | 88.15% |

这里的“其余阶段”由 `prompt eval - expert 加载 - expert CUDA` 得到，包含 attention、
router、非 expert 算子、backend 调度，以及未进入 75 个 full-prompt CUDA expert
算子的末层/末 token 路径；没有把它伪装成已细分的计算时间。

## 每层时间与冷页证据

full-prompt 路径覆盖 `blk.1`–`blk.25`，每层 gate/up/down 三个 expert tensor，
所以每个样本严格得到 75 条加载记录和 75 条 CUDA 计算记录。所有记录的 token
字段分别严格为 128 或 256。

| 限额 / prompt | 平均每层加载 | 每层加载中位数 / P95 | 平均每层 expert CUDA | 每层 CUDA 中位数 / P95 | 加载前冷页比例 | major faults | 路由 expert 字节总量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GiB / 128 | 481.16 ms | 487.95 / 647.27 ms | 50.59 ms | 50.02 / 62.81 ms | 92.06% | 21,462 | 4,154,523,648 B |
| 4 GiB / 256 | 539.25 ms | 524.12 / 742.59 ms | 72.15 ms | 71.51 / 86.74 ms | 92.07% | 23,799 | 4,733,583,360 B |
| 8 GiB / 128 | 401.46 ms | 417.52 / 609.35 ms | 49.95 ms | 50.01 / 62.70 ms | 81.51% | 19,076 | 4,154,523,648 B |
| 8 GiB / 256 | 531.04 ms | 568.82 / 665.61 ms | 71.39 ms | 71.37 / 86.98 ms | 87.34% | 23,310 | 4,733,583,360 B |

128 到 256 token 时，expert CUDA 约从 1.25 s 增至 1.79–1.80 s；但加载仍为
10.04–13.48 s。256 token 路由到的 expert 更多，累计需搬运的 expert 数据从
4.155 GB 增至 4.734 GB，因此 token 增多并不会自动让加载占比消失。

4 GiB 相比 8 GiB 的 128-token 样本多花约 1.99 s 在加载，且加载前冷页比例从
81.51% 升到 92.06%。256-token 两个单样本的加载差只有约 0.21 s；由于当前每格
只有一次冷启动，不能把该差值当成稳定的容量收益，正式论文数值仍应至少重复三次。

## 方法与口径

- 设备：NVIDIA Jetson Orin NX Engineering Reference Developer Kit，8 核
  Cortex-A78AE；模型位于 `/dev/nvme1n1p1` 的 ext4 文件系统。
- 模型：DeepSeek-V2-Lite-Q8_0，GGUF 大小 16,702,520,096 B。
- 基线参数：`--gpu-layers all --cpu-moe --mmap --no-direct-io
  --expert-cache-capacity 0 --no-repack --no-warmup --no-conversation`，batch/ubatch
  均为 512，线程数 8，`n_predict=1`。PDCat ExpertManager 没有参与。
- prompt 用 BOS 加重复的单-token ` hello` 构造；llama 性能输出再次核验为严格
  128/256 prompt tokens。
- 每次运行前，在没有存活 llama 进程时对整个 GGUF 执行
  `POSIX_FADV_DONTNEED`，清理该文件的 clean page cache；没有清理全机 cache。
- 每个样本运行在一次性 NVIDIA Docker sibling 容器中。Docker 在宿主 cgroup v2
  设置 `memory.max=4/8 GiB`、`memory.swap.max=0`。观测到的
  `memory.current` 峰值均约为相应上限；四组 `oom=0`、`oom_kill=0`。
- cgroup 口径严格覆盖其匿名内存和文件页缓存；本报告不额外断言 Jetson 驱动的
  每一类 nvmap 内部分配都由 memcg 完整计费。
- scheduler trace 在真实 router IDs 可用后，对 CPU_Mapped GGUF expert 页做
  `mincore`，同步计时原生已有的“仅复制命中 expert”路径；该时间包含 mmap 缺页
  和 CPU_Mapped→CUDA buffer 搬运。
- CUDA trace 在 copy 完成后同步计时原始 `MUL_MAT_ID` kernel。trace 为了拆开两段
  在边界处增加同步，因此表中的分项适合判断优化上限；它不是“未插桩异步流水”的
  端到端重叠时间。
- `/proc/self/io.read_bytes` 对 mmap/CUDA 触发的文件缺页未提供有效归因（四组均为
  0），所以报告使用 wall time、`mincore` 驻留度和 major faults，不把 0 误报成
  没有 NVMe I/O。

## 无 trace 扰动检查与无效样本

- `8 GiB / 128` 在相同冷缓存、prompt 和 cgroup 配置下关闭全部 trace，原生
  prompt eval 为 11,916.92 ms；同步 trace 为 12,041.60 ms，扰动为 +124.68 ms
  （+1.05%）。该控制支持 128-token 分解结果。
- `8 GiB / 256` 无 trace 控制运行期间，内核在 09:29:51 和 09:30:27 两次报告
  `nvme nvme1: I/O ... timeout, completion polled`；其 prompt eval 为
  47,831.91 ms。这个数值是硬件 timeout 污染样本，不能当作有效 baseline，也不能
  用来计算 trace 加速比。
- timeout 后 `/proc/diskstats` 显示设备已无 in-flight I/O，也没有遗留 llama 或
  实验容器；但环境没有 `nvme-cli`，无法读取 SMART/error log。出于设备安全考虑，
  本轮没有继续补 4 GiB 无 trace 控制。
- 因此，四格表是实际执行得到的同步分解结果，不是估算；其中只有 8G/128 额外通过
  了无 trace 扰动校验。发布论文前应在 NVMe 稳定后补齐无 trace 控制和至少三次重复。

## nvme1 timeout 根因与 oracle 预取复现

### timeout 根因

`nvme1` 是 Acer SSD N5000M 512GB（固件 `X0430L`），内核参数
`nvme_core.io_timeout=30`。四次异常均为：

```text
nvme nvme1: I/O ... QID 6/7 timeout, completion polled
```

Linux NVMe timeout handler 会先主动轮询 completion queue；只有轮询后发现原请求已经
完成，才打印 `completion polled`。因此这里的 30 秒不是 SSD 直到第 30 秒才读完，
而是 CQE 已存在但正常完成中断没有让驱动及时收割请求。

本机证据也与完成中断异常一致：

- `nvme1` 的 8 个 I/O MSI 队列在一次快照中分别有 2–964 次 `unhandled`；作为对照，
  同机稳定的 `nvme0` 各队列只有 1–4 次。
- PCIe AER correctable/nonfatal/fatal 计数全部为 0；链路稳定为 PCIe 4.0 x2。
- 控制器保持 `live`、PCI D0，runtime suspend 累计为 0；timeout 后没有 controller
  reset、abort、块 I/O error 或 ext4 error。

所以当前最可能的故障点是 N5000M 端点固件或它所在 PCIe/MSI 路径的 CQE/MSI 顺序、
丢中断问题。现有观测不能进一步区分“SSD 没有发/过早发 MSI”和“Jetson root complex
丢失 MSI”，但可以排除把该现象解释为正常的 30 秒 NVMe 服务时间。APST 仍启用，
不过设备在持续读取时保持 active，当前没有证据把 APST 作为主因。

### 严格 oracle 重试

第一次固定输入的 75 条 load trace 保存了每层 gate/up/down 的精确 expert IDs 和 GGUF
offset。第二次相同输入先读取这些精确范围，再启动原生 llama 路径。为降低已知的
NVMe 中断风险，预取只使用一个同步 reader，并仅在预取期间固定到 CPU4/QID5；
预取后恢复原 CPU affinity。运行仍使用 8 GiB、swap=0、128 token 和相同 trace 口径。

- 75/75 个 tensor 的 expert ID 集合完全一致，route mismatch 为 0。
- 预取读取 4,155,021,312 B、999 个范围，耗时 4.668 s，吞吐 890.12 MB/s。
- 本次运行退出码为 0，`oom=0`、`oom_kill=0`，内核没有新增 NVMe timeout/reset/error。

| 指标 | 冷启动 baseline | 全量 oracle 预取 | 变化 |
| --- | ---: | ---: | ---: |
| prompt eval / TTFT 代理 | 12,041.60 ms | 33,029.05 ms | +174.3%（2.743x） |
| 加上请求前预取的端到端时间 | 12,041.60 ms | 37,697.00 ms | +213.1%（3.130x） |
| expert 加载 | 10,036.39 ms | 31,009.13 ms | +209.0% |
| expert CUDA | 1,248.75 ms | 1,251.38 ms | +0.21% |
| 加载前 resident 比例 | 18.49% | 43.31% | +24.82 pp |
| load major faults | 19,076 | 30,696 | +60.9% |
| `workingset_refault_file` | 1,128,122 | 1,612,872 | +43.0% |
| cgroup `pgscan` | 3,518,513 | 3,957,260 | +12.5% |

oracle 没有加速，原因也不是路由错误或 CUDA 计算波动。模型文件为 16.70 GB，而实验
cgroup 只有 8 GiB；在 llama 初始化非 expert 权重时，提前放入 page cache 的 3.87 GiB
expert 页被部分驱逐。剩下的 resident 页形成碎片化空洞，使同步 expert copy 在少量
缺页处频繁进入 major fault 和 memcg 回收；预取反而破坏了冷启动时较连续的文件
readahead。CUDA 时间几乎不变，约 50 ms/层，全部退化来自加载/回收路径。

逐层结果进一步验证该解释：第 9、10、23 层在使用时仍为 100% resident，加载从
baseline 的 215–589 ms 降至 34–43 ms；但第 5–8 层即使开始时约 93% resident，
零散缺页和回收仍使每层加载升到 3.49–4.00 s。对当前同步 copy 路径，接近但不到
100% 的覆盖并不等于接近 100% 的收益。

| 层 | baseline load | oracle load | baseline CUDA | oracle CUDA | oracle resident |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 321.12 ms | 183.95 ms | 37.11 ms | 36.46 ms | 0.5% |
| 2 | 410.30 ms | 291.13 ms | 42.91 ms | 42.64 ms | 0.9% |
| 3 | 370.26 ms | 363.63 ms | 44.68 ms | 44.63 ms | 1.2% |
| 4 | 609.35 ms | 1,172.55 ms | 48.08 ms | 48.12 ms | 25.2% |
| 5 | 417.52 ms | 4,001.27 ms | 55.23 ms | 55.52 ms | 92.8% |
| 6 | 519.82 ms | 3,761.78 ms | 53.27 ms | 53.23 ms | 92.7% |
| 7 | 534.18 ms | 3,490.62 ms | 50.01 ms | 50.32 ms | 92.8% |
| 8 | 411.45 ms | 3,654.91 ms | 53.19 ms | 53.16 ms | 92.6% |
| 9 | 588.56 ms | 42.56 ms | 62.64 ms | 62.62 ms | 100.0% |
| 10 | 399.92 ms | 40.84 ms | 60.65 ms | 60.61 ms | 100.0% |
| 11 | 446.92 ms | 1,509.76 ms | 64.75 ms | 64.77 ms | 16.8% |
| 12 | 432.14 ms | 1,266.22 ms | 57.56 ms | 57.61 ms | 8.3% |
| 13 | 433.19 ms | 984.68 ms | 47.93 ms | 48.00 ms | 8.1% |
| 14 | 291.77 ms | 928.13 ms | 42.69 ms | 42.68 ms | 7.8% |
| 15 | 452.69 ms | 1,133.52 ms | 53.17 ms | 53.14 ms | 8.6% |
| 16 | 622.84 ms | 1,577.57 ms | 62.70 ms | 62.68 ms | 8.0% |
| 17 | 468.95 ms | 1,387.23 ms | 53.15 ms | 53.13 ms | 8.4% |
| 18 | 345.43 ms | 884.54 ms | 40.48 ms | 40.49 ms | 7.9% |
| 19 | 474.63 ms | 1,436.55 ms | 53.13 ms | 53.25 ms | 7.8% |
| 20 | 412.84 ms | 1,019.12 ms | 44.76 ms | 44.84 ms | 7.8% |
| 21 | 298.80 ms | 860.09 ms | 40.45 ms | 40.53 ms | 7.8% |
| 22 | 493.85 ms | 919.80 ms | 47.91 ms | 47.92 ms | 25.6% |
| 23 | 215.04 ms | 34.01 ms | 50.04 ms | 50.49 ms | 100.0% |
| 24 | 41.92 ms | 42.63 ms | 47.23 ms | 62.53 ms | 100.0% |
| 25 | 22.89 ms | 22.02 ms | 35.07 ms | 22.04 ms | 100.0% |

每层 load 的 baseline / oracle 均值分别为 401.46 / 1,240.37 ms，中位数为
417.52 / 984.68 ms，P95 为 609.35 / 3,761.78 ms；每层 CUDA 均值仍为
49.95 / 50.06 ms。

这个负结果说明后续 PDCat 不能在 8 GiB 下把整个请求的 25 层 expert 一次性放入
page cache。正确实现应在模型初始化完成后，按层和带宽预算 just-in-time 预取，限制
窗口并在消费后释放；同时把“required range 是否完整 ready”作为调度条件。全量 oracle
不是可用优化，但 100% resident 的层已经实测展示了单层加载降至约 34–43 ms 的上限。

## 逐层 explicit-slot oracle 实现状态

全请求 mmap 预取被证伪后，下一步已改为模型初始化后的逐层 JIT 方案，不再依赖
page cache：

- `export_native_moe_oracle_trace.py` 将第一次 128-token 原生 trace 转为层间精确路由
  JSONL。导出器会把full-prompt记录后的连续single-token末层suffix安全补入，并在层号
  wrap时停止；当前版本化oracle覆盖`blk.1`–`blk.26`，共458个expert标签。
- 第二次运行在当前层 router 结果可见后，先提交当前层 P0，再把精确的下一层 expert
  作为 P2 异步读取到 ExpertManager 的 O_DIRECT slot；不会改变 native router 选择。
- oracle predictor 采用 exact-only 语义。trace 不包含的转移返回空，不使用层热度或
  “下一层同 expert id”伪造命中。
- 旧 P1 路径每层无条件读取64个expert（约561 MiB）；未预算的oracle会读取下一层实际
  选择的12–24个（约105–211 MiB），最终带宽预算profile进一步截断到128-token每层9个、
  256-token每层13个，并复用P0到来时的queued-P2→P0 promotion。
- 原调度器把 P2 硬编码为单请求串行，因此仅提高全局 QD 不会增加预取并发。现已增加
  独立 `PDCAT_IO_MAX_P2_QD` 与 P0 留位约束。后述实测表明一个 9,191,424-B
  O_DIRECT 请求已经达到约 2.1 GiB/s，最终 profile 因而收敛为全局 QD=2、P2 QD=1，
  同时为 P0 保留一个 submission slot 和 20 MiB 字节预算。

软件验证已完成：InterfaceIO 26/26 单测通过（包括严格 oracle 缺失转移、逐 token
decode oracle、P2 QD 为
P0 留位、prefill/decode residency 淘汰顺序和真实 io_uring P4 测试），Rust FFI
2/2单测通过，`llama-completion` 与 `llama-cli` Release/CUDA 构建通过。

受控 8 GiB/128-token 修正对照已经执行：baseline 有效，oracle 新增一次
`timeout, completion polled`，因此按授权限制立即停止且不重试。该次 oracle 不能用于
加速比；staging 淘汰修复后的硬件复验仍会读取 nvme1，属于新的风险操作，需要再次
明确批准。复验应首先报告 P2 聚合带宽、每层 ready 数、P0 promotion/blocked time、
prediction hit、TTFT 与是否新增 NVMe timeout，再决定是否扩大到 256 token。

首次启动该矩阵时，安全脚本只完成 baseline 就在 oracle 导出前停止：
`llama-completion` 自动启用了模型 chat template，把 127 次 ` hello` 包装成了实际
135-token prompt。该样本 prompt eval 为 5,651.53 ms（23.89 tok/s），P0 共读取
6,957,907,968 B，活动时间窗聚合带宽 1,220.53 MiB/s；26 层最大 P0-use blocked
time 的中位数为 126.79 ms。运行退出码为 0，前后没有新增 nvme1 timeout/reset/error。
由于 token 数错误，这些数值只用于发现 conversation-mode 问题，不进入 128-token
baseline/oracle 对照。

矩阵入口现已强制 `--no-conversation`，并在任何 oracle 导出或第二次运行之前从
`run_manifest.json` 硬校验 `prompt_eval.count`；不等于请求的 128/256 就立即停止。
脚本还会在每次容器前后比较 nvme1 内核错误，出现新事件即标记 contaminated、停止且
不重试。逐层汇总工具会排除后续 decode pass，报告 P0/P2 聚合带宽、prediction hit、
ready-before-router、on-time use 和每层最大 blocked time。

### 修正后的 8 GiB / 128-token 对照与新发现

`runs/prefill-oracle-8g-p128-v04-20260812-r02` 严格使用
`--no-conversation`，baseline 与 oracle 的 manifest 都确认 prompt 为 128 tokens；
baseline 导出的 oracle 覆盖 26 层、458 个 expert 标签。baseline 是有效样本：

- prompt eval 3,546.42 ms，36.09 tok/s；峰值 RSS 1,211,248,640 B。
- 458 个 P0 expert 共读取 4,209,672,192 B，首笔服务到末笔 CQE 的跨度为
  3,402.02 ms（含层间计算空窗，1,180.08 MiB/s）。合并真实 I/O 活跃区间后为
  1,961.36 ms，即 2,046.87 MiB/s。
- 每层最大 P0-use blocked time 的中位数/均值/P95/最大值为
  77.17/75.55/100.48/101.92 ms。

oracle 运行退出码为 0，但内核新增：

```text
nvme nvme1: I/O 228 QID 3 timeout, completion polled
```

因此 35,158.67 ms 的 prompt eval **是硬件污染样本，不能用于加速比**，且按授权没有
重试。不过 timeout 前后的细粒度 trace 揭示了另一个确定性软件问题：

- 446/458 actual uses 带有精确 prediction ledger 命中；缺少的 12 个正是无法由前层
  预测的第1层，说明 oracle 集合语义正确。
- 285 个 P2 读取在目标层 router 之前已经完成，但真正 demand 到来时只有 36 个仍是
  on-time cache hit；249 个“已 ready 但未命中”的 expert 在使用前被淘汰。
- 原因是 demand expert 使用后被标为 `Persistent`，而所有 P2 都是 `Speculative`；
  slot 满时缓存无条件优先淘汰 speculative，于是新一层 P2 不断淘汰同层更早完成的
  P2，却长期保留已经消费的旧层 demand 项。
- 该自淘汰使物理读取达到 6,516,719,616 B，较458次实际使用放大到 1.548x。
  第2--4层在 cache 尚未饱和时，每层最大 blocked time 从 baseline 的
  63.93/67.82/71.38 ms 降至 13.62/32.33/20.05 ms；第5层后 on-time hit 几乎归零，
  blocked time 回到 baseline 水平。第25层 31.91 s 的阻塞来自上述 NVMe timeout。

淘汰逻辑已修复：prefill P2 进入 `PrefillStaging`，slot 选择只优先丢弃真正的 decode
`Speculative`；没有 decode speculative 时使用普通 LRU，从而淘汰更老、已消费且未 pin
的 demand 项，保留更新的下一层 staging。decode speculative 仍优先自淘汰，保持原有
cache-pollution 防护。两个单元测试分别证明“prefill staging 淘汰旧 demand”和
“decode speculative 先于 demand 被淘汰”。

QD 数据也给出了清晰结论：旧 decode trace 的串行 P2 单个 9.19 MB 读取中位数约
4.15 ms，即约 2.11 GiB/s；本次 P2 QD=2 时每请求中位数约 8.49 ms，两笔合计仍约
2.06 GiB/s；baseline QD=3 与 oracle（排除 >1 s timeout I/O）的总 clean-union
带宽分别为 2,046.87 与 2,042.30 MiB/s。QD2 没有提高设备总带宽，只把相同带宽分给
两笔请求。最终实验因此改用全局 QD=2/P2 QD=1，既保持约2.1 GiB/s上限，又减少
outstanding 请求和 MSI 压力；不能据单次 timeout 宣称 QD2 是硬件故障原因。

### 修复后待复验配置

下一次硬件复验不再无界排入 oracle 的全部下一层专家。预算公式为
`bandwidth × phase window × utilization`：按实测约 2,048 MiB/s、prefill 50 ms、
80% 利用率得到 81.92 MiB；每个 padded expert object 为 9,191,424 B，因此向下取整为
每层最多9个。decode 独立使用12 ms窗口；该设置主要用于后续 decode 实验，不影响本次
prefill pair。全局 QD=2、P2 QD=1，另一 submission slot 与20 MiB预算保留给P0。

256-token矩阵按原生trace约72 ms的每层计算窗口单独配置，因此每层上限为13个expert。
旧256-token native trace只有25个full-prompt MoE层的count信息；在该不完整口径下预算为
312个/2,867,724,288 B，覆盖可预测expert集合62.03%。它只用于运行前容量规划，论文值
必须由新矩阵导出的26层严格route（包括末层single-token suffix）重新计算。

用 `tools/estimate_pdcat_prefill_budget.py` 重放本次128-token baseline route：总需求为
458个expert/4,209,672,192 B，其中除第1层外可预测446个；预算会选择222个、
2,040,496,128 B，上界覆盖可预测需求49.78%、全部需求48.47%。理想情况下这些字节仅从
P0移动到P2，总物理读取仍为4,209,672,192 B、放大1.0x；若复验仍显著高于该值，应优先
检查重复读取、过早淘汰或跨层残留队列，而不能把额外字节算作overlap收益。

新的 `predictor_submission` JSONL 事件记录 source/target layer、phase、候选ID与概率、
window、带宽、利用率、预算字节、expert字节和deadline。汇总器据此直接报告 candidate/
byte precision、recall、ready recall、late-prefetch、full-demand-ready及错误预取字节，
不再依赖可能被并发日志交错破坏的stderr文本。

矩阵还新增强制可比性检查：两次运行必须具有相同命令、源码工作树、配置哈希、模型与
pack元数据，以及实际 executable/本地动态库 SHA-256；native router路径和生成输出
必须完全一致。若进程成功退出但dmesg出现新NVMe异常，脚本会先只读验证并生成summary，
随后将矩阵标记为contaminated并停止，不会启动下一项或自动重试。predictor phase
也已从进程级环境变量改为每次
C++→FFI→InterfaceIO调用显式传递，确保同一请求中prefill使用`PrefillStaging`而decode
恢复`Speculative`语义。router trace同时记录每个expert承担的token-slot次数，使汇总器
能够计算token-weighted coverage。当前Python验证为17/17；尚无新的NVMe数据，因此不能
宣称已获得端到端加速。

### 原生 llama.cpp `auto + fit` 对照

为验证“把一部分MoE层常驻GPU能否减少mmap加载”，另用本地原生
`/workspace/llama.cpp` 的b8185二进制（commit `2afcdb977`）运行严格128-token、
8 GiB cgroup、swap=0、冷mmap对照；没有传入PDCat的expert cache/pack参数，也没有
启用`--cpu-moe`。runner还显式使用`--no-warmup --no-repack -b 512 -ub 512`，因此它
只保留了GPU fit策略的默认值，并不是完整CLI默认配置。

该测点是“8 GiB强约束下的冷启动压力测试”，不是原生`llama-cli`的一般性能基线。
其cgroup采样内存距8 GiB上限仅45,056 B，`memory.events.max=23,335`；同时记录到
9,328,822次page scan、6,454,901次page steal和2,833,054次file workingset refault。
按4 KiB页折算分别约35.59、24.62和10.81 GiB的页事件规模。这些计数不能直接等同于
NVMe读取字节，但客观表明该运行发生了严重回收/反复驻留，因此54.23 s不能外推成
“原生mmap的正常prefill耗时”。

GPU fit默认参数`--gpu-layers auto --fit on`的结果为28/28层offload，其中11层存在
CPU overflow；规划使用10,844 MiB CUDA内存并保留1,197 MiB，实际CUDA model buffer
为10,470.63 MiB。MoE布局为`blk.1`--`blk.16`完整常驻GPU、`blk.17`的down留在CPU、
`blk.18`--`blk.26`的expert tensor留在CPU。按本文既有TTFT口径，prompt eval为
54,228.18 ms（2.36 tok/s）；从进程启动、fit、冷模型加载到退出的wall time为
98.98 s。cgroup采样峰值为8 GiB，RSS峰值约7.99 GiB，整机MemAvailable最低约
2.47 GiB；OOM、OOM kill与0.75 GiB紧急保护均未触发，运行前后无新增NVMe异常。

该结果没有支持“常驻更多层即可显著降低TTFT”。虽然前16个MoE层无需在请求期间从
mmap加载，但overflow路径不是按router结果只搬选中的expert：实际对`blk.18`--
`blk.25`搬运全部64个expert的gate/up/down tensor，并额外搬运`blk.17`的down，共25个
完整tensor、约4,902,092,800 B；scheduler报告CUDA I/O为4,058.28 ms。最后的
`blk.26` MoE在CPU执行，形成新的长尾。相比之下，PDCat 128-token demand-only只读取
实际命中的458个expert、4,209,672,192 B。因此auto-fit减少了发生动态加载的层数，却
没有减少动态搬运字节，还失去了expert粒度加载。

用户随后在host主机、无8 GiB cgroup限制、默认warmup/repack/batch和`llama-cli`
conversation路径报告`Prompt: 24.2 tok/s`。源码确认默认warmup会先执行BOS/EOS
dummy decode，MoE图在warmup状态把`n_expert_used`扩成全部expert，并在完成后重置
performance counter；所以日志中的`Prompt`吞吐不包含这次全expert first-touch/
repack预热成本。这是一条有价值的反证：若按128--135个
实际prompt token估算，请求prefill约为5.29--5.58 s，与上述压力点相差近10倍；但由于
`llama-cli`拒绝`--no-conversation`并继续使用chat template、实际token数和cache冷热未
写入原始manifest，该数字暂记为“用户报告的待复现实测”，不与严格128-token TTFT直接
计算加速比。代理随后只在无内存上限的Docker容器中尝试近似复现，并非host同口径；
该次运行未主动清cache，但于2026-08-12 13:21:20出现新的
`nvme nvme1: I/O 193 QID 7 timeout, completion polled`，已立即终止并作废，未启动后续
`llama-completion`对照。

额外的`--fit-target 4096`安全点产生16个overflow层、7,472.29 MiB CUDA model buffer，
prompt eval为56,332.19 ms（2.27 tok/s），只比默认点慢3.88%。随后尝试用同一原生
binary补`--gpu-layers all --cpu-moe`版本内对照时，新增一次
`nvme1: I/O 194 QID 6 timeout, completion polled`，且单个split卡住30.64 s；其
92,398.06 ms结果无效，按安全规则停止且未自动重试。因此不能用它计算auto-fit相对
原生CPU-MoE的正式加速比，也不能把b8185的54.23 s与更新版llama_uring.cpp的11.92 s
直接归因给参数差异。

### Decode strict oracle 软件状态

decode 不能复用原来只按“层号+当前专家集合”索引的 prefill trace：同一层在多个生成
token 中会重复执行，而且相同当前集合可能对应不同下一层集合。当前实现已把
`token_id` 加入 `PredictionRequest`、trace record 和 exact transition key，并保留对旧
prefill JSONL（无 token_id）的兼容。层26到下一 token 层1的 cyclic transition 使用
`target_token_id = current_token_id + 1`；其他层间转移保持同一 token_id。router trace
也在注册路径上使用 predictor generation 显式写 token_id，不再依赖进程级环境变量。

严格性保护包括：

- token-aware 导出器保留完整有序的 prefill 尾部与 decode 路由，不按层折叠；
- 缺 token_id、同一位置路由冲突、token 未增加就发生层号回绕、多个 request 都会拒绝；
- exact predictor 若同一 `(token, layer, current experts)` 对应两个目标，加载 trace 时
  直接报错；运行时在线 observe 携带 token_id，不会学习一个无 token 的旁路 fallback；
- baseline/oracle 必须逐 token 路由、生成 stdout、prompt/decode 计数和运行产物哈希
  完全相同，才会生成 TPOT/阻塞/带宽 summary。

`tools/run_pdcat_decode_oracle_matrix.py` 已覆盖 4/8 GiB × 128/256 prompt token，默认
固定生成64 token（`--ignore-eos --seed 42 --temp 0`），并使用12 ms、2048 MiB/s、80%
预算，即每个层间窗口最多预取2个expert。它与 prefill runner 共用“首个新增 nvme1
异常立即停止、绝不重试”的安全门，真实执行仍必须显式传入风险确认参数。当前仅完成
dry-run、17个Python测试、26个InterfaceIO测试、2个FFI测试和Release/CUDA编译；由于
r02授权已在新timeout后耗尽，尚未执行decode硬件对照，也没有新的TPOT加速结论。

## 原始产物

- `runs/native-moe-profile-4g-p128-v04-20260812-r01`
- `runs/native-moe-profile-4g-p256-v04-20260812-r01`
- `runs/native-moe-profile-8g-p128-v04-20260812-r05`
- `runs/native-moe-profile-8g-p256-v04-20260812-r01`
- `runs/native-moe-control-8g-p128-v04-20260812-r01`（有效无 trace 控制）
- `runs/native-moe-control-8g-p256-v04-20260812-r01`（NVMe timeout，无效）
- `runs/native-moe-oracle-source-8g-p128-v04-20260812-r01`（路由来源；计时受 timeout 污染）
- `runs/native-moe-oracle-prefetch-8g-p128-v04-20260812-r01`（有效严格 oracle 重试）
- `runs/upstream-llama-auto-fit-default-8g-p128-v04-20260812-r01`（有效8 GiB冷启动压力点；不是一般性能基线）
- `runs/upstream-llama-auto-fit-target4096-8g-p128-v04-20260812-r02`（有效4 GiB余量点）
- `runs/upstream-llama-cpu-moe-8g-p128-v04-20260812-r01`（NVMe timeout，无效）

每个目录包含 `run_manifest.json`、`native_moe_profile.jsonl`、
`cgroup_samples.jsonl`、`stdout.log` 和 `stderr.log`。早期 `r01`–`r04`/GDB 目录是
trace 边界定位样本，未纳入表格；最终有效 run ID 已在上面明确列出。

## 对后续 PDCat 实验的直接含义

1. 128 token 是更敏感的容量/预取对照点：4→8 GiB 的单样本差异主要体现在加载。
2. 256 token 仍然有约 82% prompt 时间落在 expert 加载，足以验证 overlap；它同时
   提供更长的 CUDA 计算窗口（每层约 71–72 ms），更适合检验带宽预算型预取。
3. 后续 baseline 与 PDCat 必须使用相同的 cgroup 限额、冷缓存步骤、prompt tokens
   和 trace 同步口径；否则 TTFT/加载比例不可直接比较。
4. 当前数据支持继续在 Jetson 上实现和验证，不支持“必须迁移到更强 GPU 才能看到
   prefill 收益”的判断。正式论文表格应在锁定功耗/频率后每格至少重复三次，并报告
   中位数与范围。
