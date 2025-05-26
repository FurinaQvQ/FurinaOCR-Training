"""
配置验证模块
"""
from typing import Dict, Any, List, Union
import os
from pathlib import Path


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigValidator:
    """配置验证器"""
    
    REQUIRED_FIELDS = [
        "height", "train_width", "batch_size", "epoch",
        "train_size", "validate_size", "dataloader_workers"
    ]
    
    VALID_RANGES = {
        "height": (16, 64),
        "train_width": (128, 1024),
        "batch_size": (1, 512),
        "epoch": (1, 1000),
        "train_size": (100, 10000000),
        "validate_size": (10, 1000000),
        "dataloader_workers": (0, 32),
        "print_per": (1, 10000),
        "save_per": (1, 50000)
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """验证完整配置"""
        cls._check_required_fields(config)
        cls._check_value_ranges(config)
        cls._check_logical_consistency(config)
        cls._check_file_dependencies(config)
    
    @classmethod
    def _check_required_fields(cls, config: Dict[str, Any]) -> None:
        """检查必需字段"""
        missing_fields = []
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigValidationError(f"缺少必需配置字段: {missing_fields}")
    
    @classmethod
    def _check_value_ranges(cls, config: Dict[str, Any]) -> None:
        """检查数值范围"""
        for field, (min_val, max_val) in cls.VALID_RANGES.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    raise ConfigValidationError(f"配置字段 {field} 必须是数字，当前值: {value}")
                
                if not (min_val <= value <= max_val):
                    raise ConfigValidationError(
                        f"配置字段 {field} 超出有效范围 [{min_val}, {max_val}]，当前值: {value}"
                    )
    
    @classmethod
    def _check_logical_consistency(cls, config: Dict[str, Any]) -> None:
        """检查逻辑一致性"""
        # 验证样本数量合理性
        if config.get("validate_size", 0) > config.get("train_size", 0):
            raise ConfigValidationError("验证集大小不应大于训练集大小")
        
        # 验证保存频率合理性
        if config.get("save_per", 0) < config.get("print_per", 0):
            raise ConfigValidationError("模型保存频率应大于等于打印频率")
        
        # 验证批次大小与数据集大小的关系
        if config.get("batch_size", 0) > config.get("train_size", 0):
            raise ConfigValidationError("批次大小不应大于训练集大小")
    
    @classmethod
    def _check_file_dependencies(cls, config: Dict[str, Any]) -> None:
        """检查文件依赖"""
        # 检查字体文件
        font_path = Path("assets/genshin.ttf")
        if not font_path.exists():
            raise ConfigValidationError(f"原神字体文件不存在: {font_path}")
        
        # 检查预训练模型（如果启用）
        if config.get("pretrain", False):
            model_path = Path(f"models/{config.get('pretrain_name', '')}")
            if not model_path.exists():
                raise ConfigValidationError(f"预训练模型文件不存在: {model_path}")
        
        # 检查必需目录
        required_dirs = ["models", "samples"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def get_optimization_suggestions(cls, config: Dict[str, Any]) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        # 批次大小建议
        batch_size = config.get("batch_size", 128)
        if batch_size < 32:
            suggestions.append("批次大小较小，可能影响训练稳定性，建议至少设为32")
        elif batch_size > 256:
            suggestions.append("批次大小较大，注意显存使用情况")
        
        # 数据加载器线程数建议
        workers = config.get("dataloader_workers", 0)
        cpu_count = os.cpu_count() or 1
        if workers > cpu_count:
            suggestions.append(f"数据加载器线程数 ({workers}) 超过CPU核心数 ({cpu_count})")
        elif workers == 0:
            suggestions.append("建议设置适当的数据加载器线程数以提升性能")
        
        # 训练样本数建议
        train_size = config.get("train_size", 0)
        if train_size < 10000:
            suggestions.append("训练样本数较少，可能影响模型性能")
        
        return suggestions


def validate_and_suggest(config: Dict[str, Any]) -> List[str]:
    """验证配置并返回优化建议"""
    ConfigValidator.validate_config(config)
    return ConfigValidator.get_optimization_suggestions(config) 