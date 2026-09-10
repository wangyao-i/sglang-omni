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
| 当前图模式任务 | SGLang `e4d18390a`，等待 `910C-059` 验证 ordered update/replay |

当前唯一有效性能基线是上述 M1c 配置。全 13 bucket compile 的首次结果
没有优于 `[1,2,70]`，且后续 fresh-process warm-up 卡住，因此不作为基线。

## 当前识别到的问题

### 1. NPU Graph update/replay 存在阻塞风险

当前 NPU backend 使用 Python 后台线程执行 `NPUGraph.update()`，主线程执行
`replay()` 后再等待该线程。已观察到 replay 返回，但 update 线程不返回，最终使
持有 generation guard 的线程和等待 encoder guard 的请求全部停住。

这不是 FIFO ticket lock 本身损坏。当前修复候选改为同线程顺序执行
`NPUGraph.update()` 和 `replay()`，只影响 NPU backend，不修改 CUDA graph 逻辑。

### 2. Guard 范围过大

当前正确配置用 guard 包住整个 generation forward，能够避免 encoder 与 graph
设备操作冲突，但也串行化了较大的设备执行区域。缩到 graph/model scope 的旧候选
曾发生卡死，必须先解决 update/replay 所有权，再重新验证缩圈。

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

### 第一部分：关闭图更新与并发问题

- 完成 `910C-059`：验证 ordered update/replay；
- 先使用已认证的 `forward` guard 验证正确性和 C70；
- 通过后再验证 `graph` guard；
- 如果高层 `NPUGraph.update()` 顺序调用仍阻塞，直接改为显式 NPU update stream +
  graph-task event，不再尝试新的 Python 线程组合。

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

1. 服务器按 [performance task](qwen3_asr_ascend_910b_performance_task.md)
   执行 `910C-059`；
2. 本地同时从最新 upstream 准备第一批干净 PR；
3. `910C-059` 通过后立即开始 guard 缩圈和 attention 编译连续性优化。

详细运行命令和历史证据继续保存在：

- [hardware handoff](qwen3_asr_ascend_910b_handoff.md)
- [validation task](qwen3_asr_ascend_910b_validation_task.md)
- [performance task](qwen3_asr_ascend_910b_performance_task.md)
