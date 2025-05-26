"""
原神OCR训练配置模块 - 统一版
整合分层配置管理系统和便捷函数，保持向后兼容性
"""

from .base import config_manager, ConfigManager

# 导出向后兼容的config字典
config = config_manager.get_flat_config()

# 配置管理器访问接口
def get_config_manager():
    """获取配置管理器实例"""
    return config_manager

def update_config(section: str, key: str, value):
    """更新配置值"""
    config_manager.set(section, key, value)
    # 同步更新全局config字典
    global config
    config = config_manager.get_flat_config()

def enable_mixed_precision():
    """快速启用混合精度训练"""
    update_config("hardware", "mixed_precision", True)
    print("🚀 混合精度训练已启用")

def enable_model_compilation():
    """快速启用模型编译优化"""
    update_config("hardware", "compile_model", True)
    print("🔥 模型编译优化已启用")

def optimize_for_rtx4060ti():
    """RTX 4060 TI 专项优化"""
    print("🎯 正在应用RTX 4060 TI专项优化...")
    
    # 启用所有性能优化
    update_config("hardware", "mixed_precision", True)
    update_config("hardware", "cuda_benchmark", True)
    update_config("hardware", "memory_efficient", True)
    
    # 调整批次大小以充分利用16GB显存
    current_batch = config_manager.get("training", "batch_size")
    if current_batch < 160:  # 适合16GB显存的批次大小
        update_config("training", "batch_size", 160)
        print(f"   批次大小调整: {current_batch} → 160")
    
    # 优化数据加载
    update_config("data", "dataloader_workers", 8)
    update_config("data", "pin_memory", True)
    
    print("✅ RTX 4060 TI优化完成")

def print_optimization_status():
    """打印当前优化状态"""
    hardware = config_manager.hardware
    print("\n🔧 当前优化状态:")
    print(f"   混合精度训练: {'✅ 已启用' if hardware['mixed_precision'] else '❌ 未启用'}")
    print(f"   模型编译优化: {'✅ 已启用' if hardware['compile_model'] else '❌ 未启用'}")
    print(f"   CUDA基准测试: {'✅ 已启用' if hardware['cuda_benchmark'] else '❌ 未启用'}")
    print(f"   内存优化模式: {'✅ 已启用' if hardware['memory_efficient'] else '❌ 未启用'}")
    print(f"   当前批次大小: {config_manager.get('training', 'batch_size')}")

def disable_mixed_precision():
    """禁用混合精度训练"""
    update_config("hardware", "mixed_precision", False)
    print("🔻 混合精度训练已禁用")

def reset_to_defaults():
    """重置为默认配置"""
    print("🔄 重置配置为默认值...")
    # 重新创建配置管理器来加载默认值
    global config_manager, config
    from .base import ConfigManager
    config_manager = ConfigManager()
    config = config_manager.get_flat_config()
    print("✅ 配置已重置为默认值")

def save_current_config(path: str = None):
    """保存当前配置到文件"""
    config_manager.save_config(path)

def show_config_summary():
    """显示配置摘要"""
    config_manager.print_config_summary()
    print_optimization_status()

# 导出所有公共接口
__all__ = [
    # 核心组件
    'config_manager', 'ConfigManager', 'config',
    
    # 配置管理函数
    'get_config_manager', 'update_config', 'save_current_config', 'reset_to_defaults',
    
    # 优化控制函数
    'enable_mixed_precision', 'disable_mixed_precision', 'enable_model_compilation',
    'optimize_for_rtx4060ti',
    
    # 状态查看函数
    'print_optimization_status', 'show_config_summary'
]

# ========== RTX 4060 TI 性能调优说明 ==========
"""
🚀 混合精度训练和配置管理系统已启用！

🎯 快速优化命令:
  from mona.config import enable_mixed_precision, optimize_for_rtx4060ti
  
  enable_mixed_precision()    # 启用混合精度
  optimize_for_rtx4060ti()    # RTX 4060TI专项优化

🔧 环境变量控制:
  $env:MIXED_PRECISION="true"     # 启用混合精度
  $env:BATCH_SIZE="160"           # 调整批次大小  
  $env:COMPILE_MODEL="true"       # 启用模型编译
  $env:WORKERS="8"                # 调整工作线程

📊 预期性能提升:
  - 训练速度: 30-50% ⬆️
  - 显存效率: 20-30% ⬆️
  - 训练稳定性: 显著提升

💡 使用示例:
  python main.py check                    # 系统检查
  python main.py optimize rtx4060ti       # 一键优化
  python main.py optimize status          # 查看状态
  python main.py train                    # 开始训练
""" 