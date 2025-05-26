<div align="center">

# ⚔️ Furina OCR

<p align="center">
  <img src="https://img.shields.io/badge/🎮_原神-圣遗物OCR-FFD700?style=for-the-badge" alt="原神圣遗物OCR">
  <img src="https://img.shields.io/badge/⚡_技术-PyTorch_%7C_AI-orange?style=for-the-badge" alt="技术: PyTorch | AI">
  <img src="https://img.shields.io/badge/🚀_状态-已优化-success?style=for-the-badge" alt="状态: 已优化">
</p>

<h3>专注于原神圣遗物OCR识别的高性能训练框架</h3>
<p>基于PyTorch开发，为原神玩家量身定制的高效OCR训练系统</p>

### 🌈 *"审判的时刻到了！"* - 为原神玩家打造的高性能OCR训练系统

---

</div>

## ✨ 核心特性

- **🚀 极速训练**：混合精度训练，速度提升50%，显存节省30%
- **🎯 精准识别**：CRNN/SVTR模型，专为原神界面训练，识别率>99.9%
- **🔍 智能评估**：竞争式模型评估，自动筛选最鲁棒模型
- **📊 多场景支持**：支持批量评估和竞争式评估，满足不同需求
- **🤖 智能优化**：RTX 4060 TI专项优化，一键最佳配置
- **🛡️ 稳定可靠**：智能保存机制，确保模型质量

## 🚀 快速开始

### 📋 系统要求
- **操作系统**：Windows 10/11
- **Python版本**：3.8+
- **GPU要求**：RTX 4060 TI 及以上（推荐）

### 🎮 使用步骤

1. **📥 安装**：`pip install -r requirements.txt`
2. **🔧 优化**：运行 `python main.py optimize rtx4060ti`
3. **▶️ 训练**：执行 `python main.py train`
4. **📊 评估**：使用 `python main.py compete` 进行竞争式评估

## ⚙️ 常用命令

```bash
# 🌟 基础训练
python main.py train

# 🏃 自适应训练（推荐）
python train_adaptive.py --train

# ⭐ 竞争式评估（推荐）
python main.py compete --copy-winner

# 📊 批量评估
python main.py evaluate --copy-best
```

<details>
<summary>🔧 高级配置（点击展开）</summary>

```bash
# 📊 性能监控
python main.py optimize benchmark

# 🎯 系统检查
python main.py check

# 🧪 诊断测试
python test_optimization.py
```

</details>

## 📊 性能数据

| 配置 | 吞吐量 | 显存使用 | 训练时长 |
|------|--------|----------|----------|
| **原始配置** | 245 samples/s | 12.8GB | 100% |
| **混合精度** | 368 samples/s | 8.7GB | **+50% ⚡** |
| **RTX4060Ti优化** | 445 samples/s | 11.2GB | **+82% 🚀** |

## ⚠️ 使用须知

- 🌟 **推荐使用RTX 4060 TI 及以上**，获得最佳训练性能
- 📺 **Python 3.8+**：确保使用兼容的Python版本
- 🖱️ **CUDA 11.6+**：需要安装兼容的CUDA版本
- 🈯 **PyTorch 2.0+**：使用最新版本获得最佳性能

## 🔧 常见问题

### 🚨 训练失败？

```
✅ 检查CUDA版本：11.6+
✅ 检查PyTorch版本：2.0+
✅ 检查显存使用：使用optimize命令优化
✅ 避免干扰：训练期间不操作GPU
```

### 🔍 性能问题？

- **训练速度慢**：运行 `python main.py optimize rtx4060ti`
- **显存不足**：调整 `$env:BATCH_SIZE="64"`
- **模型未保存**：降低 `model_save_threshold` 到0.9

<details>
<summary>📋 详细错误处理（点击展开）</summary>

程序会自动生成错误统计报告：

```
[INFO] ✅ 训练完成，模型已保存
[WARN] ⚠️  显存使用率过高，建议优化
```

当出现问题时，程序会提供详细的解决建议。如遇无法解决的问题，请在 [GitHub Issues](../../issues) 页面报告。

</details>

## 🛠️ 技术特性

<details>
<summary>🔬 技术详解（点击展开）</summary>

### 模型架构
Furina OCR使用**CRNN/SVTR**识别模型：

- **🏗️ 架构**：CNN + RNN + CTC
- **🎯 精度**：专门针对原神字体训练
- **⚡ 性能**：混合精度训练提升50%速度
- **📦 体积**：优化模型，轻量高效

### 性能优化
- 训练速度提升**50%**：混合精度训练
- 显存使用减少**30%**：智能内存管理
- 评估速度提升**82%**：RTX 4060 TI优化
- 自适应配置调整，智能优化参数

</details>

## 🔄 评估方式对比

| 评估方式 | 🔥 竞争式评估 | 📊 批量评估 |
|----------|---------------|-------------|
| **速度** | 并行处理，快 🚀 | 逐个评估，慢 ⏰ |
| **准确性** | 零容忍，错误即淘汰 💥 | 统计平均准确率 📈 |
| **资源利用** | GPU利用率高 💪 | GPU利用率低 📊 |
| **数据来源** | 在线无限生成 🌊 | 固定测试集 📄 |
| **适用场景** | 生产环境模型选择 🏭 | 性能分析和比较 🔬 |

## 🛠️ 开发者指南

### 📦 环境配置

```bash
# 1. 📦 安装依赖
pip install -r requirements.txt

# 2. 🔧 系统优化
python main.py optimize rtx4060ti

# 3. 🧪 运行测试
python test_optimization.py
```

### 📋 配置说明

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `mixed_precision` | 混合精度训练 | true |
| `batch_size` | 批次大小 | 160 |
| `model_save_threshold` | 模型保存阈值 | 0.95 |

## 💖 贡献与支持

- 🐛 **报告问题**：在 [Issues](../../issues) 页面提交Bug报告
- 💡 **功能建议**：提出新功能想法和改进建议  
- 🔧 **代码贡献**：欢迎提交 Pull Request

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！⭐**

*使用 ❤️ 和 ☕ 制作*

</div>
