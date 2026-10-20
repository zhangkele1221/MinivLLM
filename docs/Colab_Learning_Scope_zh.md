# Colab 学习范围与后续 GPU 测试清单

更新时间：2026-10-20

当前阶段的策略是：先在 Google Colab 单卡环境中学习 MinivLLM 的核心推理链路，性能复现、多 GPU 和需要额外硬件验证的内容留到后续租赁 NVIDIA GPU。

## 1. 总体结论

Colab 能覆盖这个项目最重要的单卡学习内容，包括模型层、Attention、KV Cache、BlockManager、Scheduler、ModelRunner 和 LLMEngine。免费 Colab 不能验证多 GPU 张量并行，也不能直接复现项目基于 A6000 得到的性能数据。

配套入口：[MinivLLM_Colab.ipynb](../MinivLLM_Colab.ipynb)

## 2. Colab 中可学习的内容

| 内容 | Colab 使用方式 |
| --- | --- |
| Qwen3-0.6B 单卡推理 | 直接运行 Colab Notebook |
| Embedding、RMSNorm、Linear、Sampler | 阅读代码并运行单卡推理 |
| FlashAttention、PagedAttention Triton Kernel | 单卡运行；性能结论仅适用于当前 GPU |
| Sequence、BlockManager、Scheduler | 运行单元测试和完整推理流程 |
| Prefix Cache、抢占和停止条件 | 阅读代码、修改测试场景 |
| `LLMEngine → ModelRunner → Qwen3` 调用链 | 通过 Notebook 端到端观察 |
| CUDA Graph 的原理 | 阅读代码；当前 Notebook 为稳定性默认关闭 |

## 3. Colab 不足或暂未验证的内容

| 内容 | 原因 | 后续处理 |
| --- | --- | --- |
| 多 GPU 张量并行 | 免费 Colab 只有一张 GPU | 租用至少 2 张相同的 NVIDIA GPU |
| A6000 性能基准复现 | T4 与 A6000 的计算能力和显存不同 | 在 A6000 或相近 GPU 上重跑 |
| CUDA Graph 实际运行 | 当前示例使用 `enforce_eager=True`，且配置字段存在不一致 | 修复后在支持 CUDA Graph 的 NVIDIA GPU 上验证 |
| 多 GPU 权重分片加载 | 当前加载器把完整合并权重复制到可能已切片的参数 | 双卡验证并修正 `weight_loader` 调用 |
| vLLM 吞吐对比 | `vllm` 较重，且可能与 Colab 预装 PyTorch/Triton 版本冲突 | 使用独立虚拟环境和干净 GPU 镜像 |
| 长时间稳定性 | Colab 会断开且 `/content` 可能被清空 | 在租赁 GPU 上运行长时间测试 |
| 默认规模基准测试 | T4 可能 OOM 或超时 | 先减小 batch/sequence length，再跑完整参数 |

### 已知代码风险

1. `src/myvllm/engine/model_runner.py` 的 CUDA Graph 路径读取 `max_num_seqs`，但示例配置使用 `max_num_sequences`。当前 `enforce_eager=True` 会绕过该路径。
2. `src/myvllm/utils/loader.py` 在 QKV 和 gate/up 合并后直接执行完整张量的 `copy_`，没有调用自定义分片 `weight_loader`。单卡大概率正常，多卡需要重点验证。
3. 基准脚本的默认矩阵规模可能超过 T4 的显存或免费运行时长，学习时应先降低参数。

## 4. 后续租赁 GPU 测试计划

### 阶段 A：单张 NVIDIA GPU

建议使用 A6000、A100、L40S 或至少 24 GiB 显存的 GPU。目标是验证完整单卡能力，而不是只看服务能否启动。

1. 使用 Linux、Python 3.11.14 和匹配 CUDA 的 PyTorch/Triton。
2. 运行 Scheduler 测试。
3. 运行 `main.py`，确认生成正常且所有关键参数成功加载。
4. 修复 `max_num_seqs` / `max_num_sequences` 不一致，再启用 CUDA Graph。
5. 先以小 batch 和小序列长度运行 Attention 基准，再逐步恢复到默认参数。
6. 记录 GPU 型号、显存、CUDA、PyTorch、Triton 和 Git commit。

### 阶段 B：两张或更多 GPU

建议租用至少 2 张同型号 NVIDIA GPU。目标是验证：

1. `world_size > 1` 时 NCCL 进程组初始化。
2. Column/Row Parallel Linear 的 all-reduce 结果。
3. QKV 和 gate/up 权重的分片加载。
4. VocabParallelEmbedding 和 LM Head 的 gather。
5. 不同 GPU 显存不一致时 KV Cache block 数取最小值。
6. Scheduler、Prefill、Decode 在多卡下的正确性。
7. 单卡与多卡输出的一致性和吞吐变化。

## 5. GPU 环境检查命令

```bash
python --version
nvidia-smi
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Git 仓库在 GPU 机器上准备好后：

```bash
uv sync
uv run pytest -q
uv run python main.py
```

先保留单卡 `world_size=1` 和 `enforce_eager=True`。确认正确输出后，再进行 CUDA Graph 与多卡实验。

## 6. 验收记录

每次 GPU 测试至少记录：

- Git commit 和分支
- GPU 型号、数量、显存
- NVIDIA Driver、CUDA、PyTorch、Triton 版本
- 使用的 `world_size`、`block_size`、batch 和序列长度
- 是否出现 OOM、NCCL 错误或未加载参数警告
- 输出文本/Token ID 是否合理
- Prefill、Decode 的吞吐和延迟

当前 Colab Notebook 已完成 JSON、Python 语法和 Qwen3 模型配置字段检查，但尚未在真实 T4 上完成端到端运行。首次 Colab 运行结果也应补录到本文件。
