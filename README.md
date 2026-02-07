# 🚀 NanoGPT Training Experiment: AdamW vs. Muon

> A comparative study of the classic AdamW optimizer and the state-of-the-art Muon optimizer (NeurIPS 2024/2025) on NanoGPT training.

## 📖 项目背景 (Project Overview)

大语言模型是当前深度学习的研究热点。本可项目旨在在资源受限的环境下（单卡 GPU），复现并训练 NanoGPT（124M 参数），并对比分析经典优化器 **AdamW** 与顶会最新提出的 **Muon (Momentum Orthogonalized)** 优化器的性能差异。

**核心目标：**
1.  **基础任务**：使用 AdamW 预训练 NanoGPT，总结单卡大模型训练的显存优化与调试技巧。
2.  **进阶任务**：复现 Muon 优化器，分析其在小规模模型上的收敛特性与效率。

---

## 🛠️ 实验环境 (Environment)

* **Hardware**: NVIDIA GeForce RTX 5090 (32GB VRAM) @ AutoDL
* **OS**: Ubuntu 22.04 LTS
* **Python**: 3.10 / 3.12
* **Framework**: PyTorch 2.4.0 + CUDA 12.4
* **Dataset**: OpenWebText

**模型架构 (NanoGPT):**
* Parameters: 124M (GPT-2 Small scale)
* Layers: 12, Heads: 12, Embedding Dim: 768
* Context Length: 1024
* **Batch Size**: 480 (Implemented via Gradient Accumulation: 32 * 15)
* Precision: `bfloat16`

---

## 📊 实验一：AdamW 基准训练 (Baseline)

### 1. 算法设置
* **Optimizer**: PyTorch Native `AdamW`
* **Learning Rate**: 6e-4 (Cosine Decay)
* **Weight Decay**: 0.1
* **Betas**: (0.9, 0.95)

### 2. 训练结果
模型在 40,000 步内收敛良好，未出现梯度爆炸。
* **Final Train Loss**: 2.967
* **Convergence**: 快速下降期 (0-2k steps) -> 平稳下降期 (20k+ steps)

<div align="center">
  <img src="assets/adamw_loss_curve.svg" width="80%">
  <p>图 1：AdamW 优化器训练 Loss 曲线 (Train)</p>
</div>

### 3. 核心调试技巧 (Optimization Tips)
在单卡训练中，我们采用了以下策略来保障效率与稳定性：

* **显存优化**: 使用 **Gradient Accumulation** (小批次 x 多次累积) 模拟大 Batch Size；全程启用 **`bfloat16`** 混合精度。
* **效率提升**: 启用 **`torch.compile()`**，显著提升 MFU (Model Flops Utilization)。
* **稳定性**: 设置 `grad_clip=1.0` 防止梯度爆炸；设置 2000 步 **Warmup** 避免初期陷入局部最优。
* **监控**: 利用 WandB 实时监控 GPU SM Clock 与利用率，确保算力充分释放。

<div align="center">
  <img src="assets/wandb_monitor.png" width="80%">
  <p>图 2：WandB 硬件资源监控面板</p>
</div>

---

## 🧪 实验二：Muon 优化器复现 (Reproduction)

### 1. 算法原理
复现了 **Muon (Momentum Orthogonalized)** 优化器。采用**分层优化策略**：
* **矩阵参数 (2D Weights)**: 使用 Muon (Newton-Schulz 迭代正交化)，LR = 0.02。
* **向量参数 (Embeddings/Bias)**: 使用 AdamW，LR ≈ 3e-4。

### 2. 关键代码片段

**优化器配置 (Parameter Grouping):**
```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    from muon import SingleDeviceMuonWithAuxAdam
    muon_params = []
    adam_params = []
    
    for name, p in self.named_parameters():
        if not p.requires_grad: continue
        # 核心逻辑：维度>=2 且非 Embedding/Head 层使用 Muon
        if p.ndim >= 2 and 'wte' not in name and 'wpe' not in name and 'lm_head' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_group = {'params': muon_params, 'use_muon': True, 'lr': 0.02, 'momentum': 0.95}
    adam_group = {'params': adam_params, 'use_muon': False, 'lr': 3e-4, 'betas': betas}
    
    return SingleDeviceMuonWithAuxAdam([muon_group, adam_group])

```

### 3. 训练结果

Muon 训练过程收敛正常，验证了算法复现的正确性。

<div align="center">
<img src="assets/muon_loss_curve.svg" width="80%">
<p>图 3：Muon 优化器训练 Loss 曲线</p>
</div>

---

## 🆚 结果对比与分析 (Comparison)

我们将 AdamW 与 Muon 的验证集 Loss 进行了对比：

| 训练阶段 | Iteration | AdamW Loss (Baseline) | Muon Loss | 差异分析 |
| --- | --- | --- | --- | --- |
| **热身结束** | 2,000 | 3.791 | **3.774** | **Muon 领先** (样本效率极高) |
| **中期** | 20,000 | **3.069** | 3.258 | AdamW 反超 |
| **最终收敛** | 40,000 | **2.967** | 3.051 | **AdamW 最终获胜** |

<div align="center">
<img src="assets/comparison_chart.svg" width="80%">
<p>图 4：AdamW vs Muon 对比折线图 (Blue: AdamW, Red: Muon)</p>
</div>

### 🧐 深度分析：为何 Muon 在此任务中未击败 AdamW？

尽管 Muon 是前沿算法，但在本次 NanoGPT (124M) 实验中，AdamW 取得了更低的最终 Loss。原因分析如下：

1. **初期爆发 vs 后期乏力**: Muon 在前 2000 步表现优异，验证了其**高样本效率**的特性。但在精细调整阶段，AdamW 的自适应一阶矩估计表现出更强的挖掘能力。
2. **模型规模错配 (Scale Mismatch)**: Muon 设计初衷是优化 **7B+** 超大模型。在 124M 小模型上，强制的矩阵正交化约束可能限制了参数解空间的灵活性。
3. **超参数敏感性**: Muon 使用了默认 LR=0.02，对于小模型可能略大，导致后期在极小值附近震荡。

---

## 📝 总结 (Conclusion)

本实验成功在 RTX 5090 上完成了 NanoGPT 的全流程训练与算法复现。

* **工程实践**: 验证了混合精度、图编译等技巧在单卡训练中的重要性。
* **算法洞察**: 实验表明，虽然 Muon 是前沿算法，但在**小规模模型**场景下，经典的 **AdamW 依然是稳健性与性价比的首选**。这提示我们在选择优化器时需充分考虑模型规模的适配性。

---

*Project by [Chongxuan Liu]*
