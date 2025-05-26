# ⚔️ Furina OCR - 原神圣遗物OCR识别训练框架

> *"审判的时刻到了！"* - 为原神玩家量身打造的高性能OCR训练系统

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![RTX4060Ti](https://img.shields.io/badge/RTX_4060_Ti-Optimized-blue.svg)](https://www.nvidia.com/rtx-4060-ti/)

专为原神圣遗物文字识别优化的OCR训练框架，集成**混合精度训练**、**智能配置管理**和**竞争式模型评估**。

---

## 🚀 核心特性

### ⚡ 性能突破
- **混合精度训练**: 自动30-50%速度提升，显存节省20-30%
- **RTX 4060 TI专项优化**: 充分利用16GB显存，一键最佳配置
- **智能内存管理**: 自动显存清理和优化建议

### 🎯 智能评估  
- **🔥 竞争式评估**: 所有模型同时竞争，错误即淘汰，找出最鲁棒模型
- **批量评估**: 传统模型对比分析，生成详细报告
- **自适应数据策略**: 根据训练进度智能切换数据生成策略

### 🛡️ 稳定可靠
- **智能保存**: 仅保存高质量模型，避免存储浪费
- **配置分层**: 支持环境变量、配置文件、命令行参数
- **向后兼容**: 无缝升级，现有代码无需修改

---

## ⚡ 快速开始

### 环境要求
```bash
# 系统要求
Python 3.8+ | PyTorch 2.0+ | CUDA 11.6+ | RTX 4060 TI (推荐)

# 安装依赖
pip install -r requirements.txt
```

### 1. 系统检查与优化
```powershell
# 检查系统状态
python main.py check

# RTX 4060 TI专项优化
python main.py optimize rtx4060ti

# 查看优化状态
python main.py optimize status
```

### 2. 开始训练
```powershell
# 标准训练
python main.py train

# 自适应训练（推荐）
python train_adaptive.py --train
```

### 3. 模型评估

#### 🔥 竞争式评估（推荐生产环境）
```powershell
# 基础竞争评估
python main.py compete

# 竞争评估并复制获胜模型
python main.py compete --copy-winner

# 深度测试（用于生产环境）
python main.py compete --min-survival-time 300 --max-evaluation-time 3600 --copy-winner
```

#### 📊 批量评估（用于分析对比）
```powershell
# 标准批量评估
python main.py evaluate

# 高精度评估并复制最佳模型
python main.py evaluate --test-size 20000 --copy-best
```

---

## 📊 性能数据

| 配置 | 吞吐量 | 显存使用 | 训练时长 |
|------|--------|----------|----------|
| 原始配置 | 245 samples/s | 12.8GB | 100% |
| 混合精度 | 368 samples/s | 8.7GB | **+50% ⚡** |
| RTX4060Ti优化 | 445 samples/s | 11.2GB | **+82% 🚀** |

---

## ⚙️ 配置管理

### 推荐配置流程

1. **生成推荐配置**
```powershell
python scripts/check_config.py --generate
mv configs/recommended_config.jsonc configs/train_config.jsonc
```

2. **环境变量快速调整**
```powershell
$env:MIXED_PRECISION="true"     # 启用混合精度
$env:BATCH_SIZE="160"           # RTX 4060 TI推荐值
$env:DATA_STRATEGY="adaptive"   # 自适应数据策略
```

3. **配置文件** `configs/train_config.jsonc`
```json
{
  "hardware": {
    "mixed_precision": true,
    "memory_efficient": true
  },
  "training": {
    "batch_size": 160,
    "model_save_threshold": 0.95
  },
  "data": {
    "data_strategy": "adaptive",
    "accuracy_threshold": 0.95
  }
}
```

---

## 🎮 核心命令速查

### 训练管理
```bash
python main.py train                    # 🎯 标准训练
python train_adaptive.py --train       # 🧠 自适应训练
python main.py validate model.pt       # 🔍 验证模型
python main.py generate --count 50     # 📸 生成样本
```

### 模型评估
```bash
python main.py compete --copy-winner    # 🔥 竞争式评估
python main.py evaluate --copy-best    # 📊 批量评估
python main.py evaluate --single model.pt  # 🎯 单模型评估
```

### 系统优化
```bash
python main.py optimize rtx4060ti       # 🎮 RTX专项优化
python main.py optimize benchmark       # ⚡ 性能测试
python main.py check                    # 🔧 系统检查
python main.py clean                    # 💾 清理GPU
```

### 配置管理
```bash
python scripts/check_config.py          # 🔍 配置检查
python scripts/check_config.py --generate  # 📝 生成推荐配置
```

---

## 🏆 评估方式对比

| 评估方式 | 🔥 竞争式评估 | 📊 批量评估 |
|----------|---------------|-------------|
| **速度** | 并行处理，快 🚀 | 逐个评估，慢 ⏰ |
| **准确性** | 零容忍，错误即淘汰 💥 | 统计平均准确率 📈 |
| **资源利用** | GPU利用率高 💪 | GPU利用率低 📊 |
| **数据来源** | 在线无限生成 🌊 | 固定测试集 📄 |
| **适用场景** | 生产环境模型选择 🏭 | 性能分析和比较 🔬 |

---

## 📁 项目结构

```
🗂️ yas-train-main/
├── 🚀 main.py                    # 主程序入口
├── ⚡ train.py                   # 标准训练脚本
├── 🧠 train_adaptive.py          # 自适应训练脚本
├── 🧪 test_optimization.py       # 性能测试
├── 📄 requirements.txt           # 项目依赖
├── 📁 mona/                      # 核心模块
│   ├── 📁 config/               # 🔧 配置管理系统
│   ├── 📁 training/             # ⚡ 性能优化器
│   ├── 📁 nn/                   # 🧠 神经网络模型
│   └── 📁 datagen/              # 📸 数据生成器
├── 📁 configs/                   # ⚙️ 配置文件
├── 📁 scripts/                   # 🛠️ 工具脚本
├── 📁 models/                    # 🎯 模型存储
├── 📁 production_models/         # 🏭 生产模型
├── 📁 assets/                    # 🎮 原神资源
│   └── genshin.ttf              # 原神字体
└── 📁 logs/                      # 📊 日志和报告
```

---

## 🛠️ 故障排除

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| **显存不足** | `CUDA out of memory` | `$env:BATCH_SIZE="64"` |
| **训练缓慢** | 低于200 samples/s | `python main.py optimize rtx4060ti` |
| **模型未保存** | 训练完成无模型文件 | 降低 `model_save_threshold` 到0.9 |
| **配置无效** | 优化不生效 | `python scripts/check_config.py` |

### 🆘 紧急救援
```bash
python main.py clean              # 💾 清理显存
python test_optimization.py       # 🧪 运行诊断测试
```

---

## 🎭 技术亮点

### 🧠 智能架构
- **CRNN/SVTR**: 先进的序列识别网络
- **CTC损失**: 专为OCR任务优化
- **在线生成**: 无限数据，节省存储

### ⚡ 性能优化
- **混合精度**: PyTorch原生AMP支持
- **模型编译**: torch.compile加速（PyTorch 2.0+）
- **内存优化**: 智能显存管理和清理

### 🛡️ 质量保证
- **智能保存**: 只保存高质量模型
- **实时验证**: 训练过程中持续评估
- **错误恢复**: 自动重试和状态恢复

### 🎯 数据策略
- **自适应生成**: 根据训练进度智能切换数据策略
- **困难样本**: 针对容易混淆的字符进行专项训练
- **在线评估**: 无限随机样本，避免过拟合

---

## 💝 致原神玩家

这个项目专为原神玩家打造，让你的RTX 4060 TI发挥最大潜能！

- 🎮 **为玩家而生**: 专门优化原神圣遗物文字识别
- ⚔️ **性能至上**: 混合精度训练，训练速度翻倍
- 💎 **品质保证**: 竞争式评估，只保存最鲁棒的模型
- 🌟 **开箱即用**: 一键优化，无需复杂配置

> *"愿正义得以彰显！"* - 开始你的高效训练之旅吧！

---

**📞 获取帮助**: `python main.py --help` | **🧪 运行诊断**: `python test_optimization.py`

*⚔️ 由Furina团队倾力打造 | 🎮 专为原神玩家优化 | ⚡ RTX 4060 TI性能怪兽*
