# Qwen3-ASR Ascend 910B roadmap

## 目标

项目同时有两个验收目标，缺一不可。

### 1. 全特性支持

在单张 Ascend 910B/910C 上同时开启：

- Encoder Graph；
- Prefill Graph；
- Decode Graph；
- Torch Compile；
- NPU Execution Guard。

要求输出正确、所有目标路径实际执行、无 eager/graph fallback、无捕获失败、
无死锁，并能正常释放进程和 NPU HBM。

### 2. 性能达标

固定 exact-10-second 数据集，700 个测量样本、并发 70：

- latency p95 `< 0.500 s`；
- throughput `>= 140 requests/s`；
- WER 保持在当前已认证的约 `0.016` 水平；
- 请求全部成功，结果可由 fresh process 重复。

### 3. 代码可上库

将有效修改分别提交到 SGLang 和 SGLang-Omni。社区 PR 只包含产品代码、
测试和必要文档，不包含 `910C` 历史、临时诊断、服务器路径或已失败的实验方案。

## 当前进展

| 项目 | 当前结果 |
|---|---|
| 全特性正确性 | 已在 broad `forward` guard 下通过 |
| 最佳配置 | `max_total_tokens=32768`、`mem_fraction_static=0.80`、`torch_compile_bs=[1,2,70]` |
| 最佳性能 | p95 `1.442 s`、throughput `59.88/s`、WER `0.0164` |
| 距离目标 | p95 还差 `2.88x`，throughput 还差 `2.34x` |
| 当前图模式任务 | `910C-059` 已证明 ordered update/replay 和 `graph` guard 可在 C70 排空；准确率与延迟指标待从现有产物补齐 |

当前唯一有效性能基线是上述 M1c 配置。全 13 bucket compile 的首次结果
没有优于 `[1,2,70]`，且后续 fresh-process warm-up 卡住，因此不作为基线。

## 当前识别到的问题

### 1. NPU Graph update/replay 修复待正式固化

旧 NPU backend 的 Python update 线程可能阻塞。`910C-059` 改为同线程顺序执行
`NPUGraph.update()` 和 `replay()` 后，`forward` 和 `graph` scope 均完成 C70，数千次
update/replay 全部配对且没有 update-thread 标记。下一步是补齐现有结果的 WER、乱码、
p95/p99 和吞吐量，再决定默认模式和上库形式；CUDA graph 逻辑未修改。

### 2. Guard 范围过大

当前正式性能基线仍用 guard 包住整个 generation forward。ordered update 下，缩到
`graph` scope 已通过请求完成和锁配对检查，但缺少完整端到端指标，尚不能宣称性能受益。
`model` scope 已拒绝，不再验证。

### 3. Attention 编译不连续

Torch Compile 正确性已经修复，但运行中仍有大量 graph break/recompile，且很多
decode forward 没有得到有效编译收益。单纯编译全部 bucket 没有继续提升性能，
下一步重点是减少 attention 路径的编译断点和 host/device 间隙。

### 4. Encoder 队列和设备临界区较长

历史数据中 encoder queue wait 和 guard hold 存在长尾。Encoder Graph 已能 capture/
replay，但还需要保证签名在测量前稳定，并缩小 encoder 真正需要占用设备 guard 的区域。

### 5. 开发分支混入过多实验内容

当前总分支同时包含产品实现、benchmark、诊断工具、历史 handoff 和失败候选，不能直接
作为社区 PR。需要从最新 upstream 基线重建小提交，而不是整体 cherry-pick 实验历史。

## 后续分为四部分

### 第一部分：固化图更新与并发修复

- 从 `910C-059` 现有产物补齐准确率和性能指标，不重跑硬件；
- 若指标有效，保留 ordered update 和 `graph` guard 作为上库候选；
- 再做跨 NPU graph runner 回归和 fresh-process 复现后固化默认行为。

完成标准：全特性正确、C70 排空、update/replay 全配对、guard 无 outstanding。

### 第二部分：整理并上库有效代码

SGLang 侧拆分：

1. stateful decode attention 的 Torch Compile 正确性；
2. sparse compile bucket；
3. 外部 execution guard hook；
4. 验证后的 NPU update/replay 修复。

SGLang-Omni 侧拆分：

1. Qwen3-ASR NPU execution guard；
2. Encoder Graph 与 bounded signature cache；
3. graph/compile model-info；
4. compile bucket 配置传递；
5. exact10 benchmark、语料生成和 NPU monitor。

诊断脚本、wheel/import probe、capture defer/release 和 `910C` 文档只保留在开发分支。

### 第三部分：性能优化

按以下顺序推进，每次只保留正确且端到端性能有收益的修改：

1. 缩小 guard 临界区；
2. 减少 attention graph break/recompile；
3. 降低 encoder queue wait 和 device hold；
4. 优化 prefill/decode admission 与 batch 形成；
5. 最后根据 NPU profile 优化明确的热点算子。

阶段目标：先达到 p95 `<=1.20 s`、throughput `>=70/s`，再推进到
p95 `<=0.80 s`、throughput `>=100/s`，最终达到客户目标。

### 第四部分：最终认证

- 在最终 rebased 代码上开启全部特性；
- 运行正确性、容量 8→70 和 exact10 C70；
- 进行三次 fresh-process 性能复现；
- 验证无 fallback、无错误、正常排空、HBM 回到基线；
- 更新用户文档并关闭开发 handoff。

## 当前下一步

1. 服务器只读取 `910C-059` 已有 JSON/JSONL/log，补回 WER、garbled、p95/p99、
   throughput、RTFx 和分 label guard 指标；
2. 本地从最新 upstream 准备第一批干净 PR；
3. 指标确认后，开始 attention 编译连续性和 encoder 临界区优化。

详细运行命令和历史证据继续保存在：

- [hardware handoff](qwen3_asr_ascend_910b_handoff.md)
- [validation task](qwen3_asr_ascend_910b_validation_task.md)
- [performance task](qwen3_asr_ascend_910b_performance_task.md)
