"""
分层配置管理系统
支持环境变量、配置文件和命令行参数的多层级配置
整合现有配置并添加性能优化选项
"""

import os
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器 - 支持多层级配置加载"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为configs/train_config.jsonc
        """
        # 更新默认配置文件为JSONC格式
        self.config_file = config_file or "configs/train_config.jsonc"
        self._config = self._load_base_config()
        self._apply_env_overrides()
    
    def _remove_json_comments(self, content: str) -> str:
        """移除JSONC格式的注释"""
        # 移除单行注释 // ...
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # 移除多行注释 /* ... */
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 移除末尾逗号（JSONC允许，但标准JSON不允许）
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        return content
    
    def _load_json_with_comments(self, file_path: str) -> dict:
        """加载支持注释的JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除注释
        clean_content = self._remove_json_comments(content)
        
        # 解析JSON
        return json.loads(clean_content)
    
    def _load_base_config(self) -> Dict[str, Any]:
        """加载基础配置，整合原有配置"""
        # 🎯 简化默认配置 - 仅保留核心必需项，其他通过配置文件设置
        base_config = {
            # 模型参数 - 架构相关，一般不变
            "model": {
                "height": 32,                 # 固定输入高度
                "train_width": 384,           # 默认训练宽度
                "lexicon_size": None,         # 动态计算
            },
            
            # 训练参数 - 建议通过配置文件调整
            "training": {
                "batch_size": 64,             # 🔧 保守默认值，适配大多数显卡
                "epoch": 50,
                "learning_rate": 1.0,         # Adadelta默认学习率
                "optimizer": "adadelta",
                "gradient_clip_norm": 1.0,
                "print_per": 100,             # 日志输出频率
                "save_per": 600,              # 验证保存频率
                "model_save_threshold": 0.95, # 🎯 默认95%即保存，便于调试
                "early_stopping_patience": 10,
                "unfreeze_backbone_epoch": 0,
            },
            
            # 数据参数 - 建议通过配置文件调整
            "data": {
                "train_size": 50000,          # 🔧 减少默认值，加快调试
                "validate_size": 5000,
                "dataloader_workers": 4,      # 🔧 保守默认值
                "pin_memory": True,
                "online_train": True,
                "online_val": True,
                # 🚀 自适应数据策略配置
                "data_strategy": "adaptive",
                "accuracy_threshold": 0.95,
                "difficult_samples_count": 5000,
                "difficult_samples_ratio": 0.3,
            },
            
            # 数据增强 - 算法相关，一般不变
            "augmentation": {
                "gaussian_blur_prob": 0.5,
                "random_crop_prob": 0.5,
                "gaussian_noise_std": 1/255,
                "rotation_range": 2,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
            },
            
            # 硬件优化 - 智能默认值
            "hardware": {
                "mixed_precision": True,      # 🚀 现代显卡默认启用
                "compile_model": False,       # 🔧 稳定性优先，可选启用
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True,
                "cuda_benchmark": True,       # 🔥 固定尺寸，启用优化
                "memory_efficient": False,    # 🔧 默认关闭，显存不足时启用
            },
            
            # 模型保存和检查点
            "checkpoint": {
                "save_best_only": False,
                "save_top_k": 3,
                "monitor_metric": "accuracy",
                "checkpoint_dir": "checkpoints",
                "auto_resume": True,
                "save_interval": 600,
            },
            
            # 预训练模型
            "pretrain": {
                "enabled": False,
                "model_path": "models/genshin_model.pt",
                "freeze_backbone": False,
                "unfreeze_epoch": 0,
            },
            
            # 实验跟踪
            "experiment": {
                "enabled": False,
                "project_name": "genshin-ocr",
                "experiment_name": None,
                "tags": [],
                "log_model": True,
            },
            
            # 调试和监控
            "debug": {
                "profile_training": False,
                "log_memory_usage": True,
                "save_sample_images": True,
                "validate_data_integrity": False,
                "performance_monitoring": True,
            }
        }
        
        # 尝试加载配置文件
        if os.path.exists(self.config_file):
            try:
                file_config = self._load_json_with_comments(self.config_file)
                base_config = self._deep_merge(base_config, file_config)
                print(f"✅ 配置文件已加载: {self.config_file}")
                print(f"📝 配置覆盖说明: 文件配置 > 默认配置 > 环境变量")
            except Exception as e:
                print(f"⚠️ 配置文件加载失败: {self.config_file}, 错误: {e}")
                print("🔄 使用默认配置")
        else:
            print(f"📝 配置文件不存在: {self.config_file}")
            print("💡 建议: 复制 configs/train_config.jsonc.example 并根据需要修改")
        
        return base_config
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_mappings = {
            # 训练参数
            "BATCH_SIZE": ("training", "batch_size", int),
            "LEARNING_RATE": ("training", "learning_rate", float),
            "EPOCH": ("training", "epoch", int),
            "MODEL_SAVE_THRESHOLD": ("training", "model_save_threshold", float),
            
            # 数据参数
            "TRAIN_SIZE": ("data", "train_size", int),
            "WORKERS": ("data", "dataloader_workers", int),
            
            # 硬件优化 - 关键环境变量
            "MIXED_PRECISION": ("hardware", "mixed_precision", lambda x: x.lower() == 'true'),
            "COMPILE_MODEL": ("hardware", "compile_model", lambda x: x.lower() == 'true'),
            "CUDA_BENCHMARK": ("hardware", "cuda_benchmark", lambda x: x.lower() == 'true'),
            "MEMORY_EFFICIENT": ("hardware", "memory_efficient", lambda x: x.lower() == 'true'),
        }
        
        for env_key, (section, key, converter) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    self._config[section][key] = converter(env_value)
                    print(f"✅ 环境变量覆盖: {env_key} = {self._config[section][key]}")
                except (ValueError, KeyError) as e:
                    print(f"⚠️ 环境变量 {env_key} 解析失败: {e}")
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, section: str, key: str = None, default=None):
        """获取配置值"""
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """设置配置值"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        print(f"✅ 配置更新: {section}.{key} = {value}")
    
    def save_config(self, path: Optional[str] = None):
        """保存当前配置到文件"""
        save_path = path or self.config_file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        print(f"💾 配置已保存: {save_path}")
    
    def get_flat_config(self) -> dict:
        """获取扁平化配置（向后兼容原有config.py）"""
        flat_config = {}
        
        # 直接映射原有配置键
        flat_config.update({
            # 模型参数
            "height": self._config["model"]["height"],
            "train_width": self._config["model"]["train_width"],
            
            # 训练参数
            "batch_size": self._config["training"]["batch_size"],
            "epoch": self._config["training"]["epoch"],
            "print_per": self._config["training"]["print_per"],
            "save_per": self._config["training"]["save_per"],
            "model_save_threshold": self._config["training"]["model_save_threshold"],
            "unfreeze_backbone_epoch": self._config["training"]["unfreeze_backbone_epoch"],
            
            # 数据参数
            "train_size": self._config["data"]["train_size"],
            "validate_size": self._config["data"]["validate_size"],
            "dataloader_workers": self._config["data"]["dataloader_workers"],
            "online_train": self._config["data"]["online_train"],
            "online_val": self._config["data"]["online_val"],
            
            # 预训练
            "pretrain": self._config["pretrain"]["enabled"],
            "pretrain_name": self._config["pretrain"]["model_path"].split('/')[-1],
            
            # 新增性能优化参数
            "mixed_precision": self._config["hardware"]["mixed_precision"],
            "compile_model": self._config["hardware"]["compile_model"],
            "cuda_benchmark": self._config["hardware"]["cuda_benchmark"],
            "memory_efficient": self._config["hardware"]["memory_efficient"],
        })
        
        return flat_config
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("🚀 原神OCR训练配置摘要")
        print("=" * 50)
        
        # 训练配置
        training = self._config["training"]
        print(f"📊 训练配置:")
        print(f"   批次大小: {training['batch_size']}")
        print(f"   训练轮数: {training['epoch']}")
        print(f"   保存阈值: {training['model_save_threshold']}")
        
        # 硬件优化
        hardware = self._config["hardware"]
        print(f"🔧 硬件优化:")
        print(f"   混合精度: {'✅' if hardware['mixed_precision'] else '❌'}")
        print(f"   模型编译: {'✅' if hardware['compile_model'] else '❌'}")
        print(f"   CUDA优化: {'✅' if hardware['cuda_benchmark'] else '❌'}")
        
        # 数据配置
        data = self._config["data"]
        print(f"📦 数据配置:")
        print(f"   训练样本: {data['train_size']:,}")
        print(f"   验证样本: {data['validate_size']:,}")
        print(f"   工作线程: {data['dataloader_workers']}")
        
        print("=" * 50)
    
    @property
    def training(self) -> dict:
        """训练配置"""
        return self._config["training"]
    
    @property
    def data(self) -> dict:
        """数据配置"""
        return self._config["data"]
    
    @property
    def model(self) -> dict:
        """模型配置"""
        return self._config["model"]
    
    @property
    def hardware(self) -> dict:
        """硬件配置"""
        return self._config["hardware"]


# 创建全局配置实例
config_manager = ConfigManager()

# 向后兼容的配置字典
config = config_manager.get_flat_config()

# 打印配置摘要
if __name__ == "__main__":  # 只在直接运行模块时打印，不在导入时打印
    config_manager.print_config_summary() 