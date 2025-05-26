"""
配置冲突检查工具
检测和解决base.py与train_config.jsonc之间的配置问题
"""

import os
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from mona.config import get_config_manager, config
from mona.utils import logger


def check_config_conflicts():
    """检查配置冲突和问题"""
    logger.info("🔍 开始配置冲突检查...")
    logger.info("=" * 50)
    
    config_manager = get_config_manager()
    issues_found = []
    warnings = []
    recommendations = []
    
    # 1. 检查配置文件存在性
    config_file = "configs/train_config.jsonc"
    if not os.path.exists(config_file):
        issues_found.append(f"❌ 配置文件不存在: {config_file}")
        recommendations.append("💡 建议创建配置文件以自定义训练参数")
    else:
        logger.info(f"✅ 配置文件存在: {config_file}")
    
    # 2. 检查关键配置值
    batch_size = config_manager.get("training", "batch_size")
    train_size = config_manager.get("data", "train_size")
    model_threshold = config_manager.get("training", "model_save_threshold")
    
    logger.info("\n📊 当前关键配置:")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"   训练样本数: {train_size:,}")
    logger.info(f"   保存阈值: {model_threshold}")
    
    # 3. 检查配置合理性
    if batch_size > 256:
        warnings.append(f"⚠️ 批次大小过大 ({batch_size})，可能导致显存不足")
        recommendations.append("💡 建议降低batch_size到128-160（RTX 4060 TI）")
    
    if batch_size < 16:
        warnings.append(f"⚠️ 批次大小过小 ({batch_size})，训练效率较低")
        recommendations.append("💡 建议提高batch_size到64-128")
    
    if train_size < 10000:
        warnings.append(f"⚠️ 训练样本数过少 ({train_size:,})，可能影响模型性能")
        recommendations.append("💡 建议增加train_size到50000+")
    
    if model_threshold == 1.0:
        warnings.append("⚠️ 保存阈值为100%，可能导致训练很久才保存模型")
        recommendations.append("💡 调试时建议设置model_save_threshold为0.95")
    
    # 4. 检查硬件优化配置
    mixed_precision = config_manager.get("hardware", "mixed_precision")
    compile_model = config_manager.get("hardware", "compile_model")
    memory_efficient = config_manager.get("hardware", "memory_efficient")
    
    logger.info(f"\n⚡ 硬件优化状态:")
    logger.info(f"   混合精度: {'✅ 已启用' if mixed_precision else '❌ 未启用'}")
    logger.info(f"   模型编译: {'✅ 已启用' if compile_model else '❌ 未启用'}")
    logger.info(f"   内存优化: {'✅ 已启用' if memory_efficient else '❌ 未启用'}")
    
    if not mixed_precision:
        recommendations.append("🚀 强烈建议启用mixed_precision以提升训练速度")
    
    # 5. 检查数据策略配置
    data_strategy = config_manager.get("data", "data_strategy", "online")
    accuracy_threshold = config_manager.get("data", "accuracy_threshold", 0.95)
    
    logger.info(f"\n🧠 数据策略:")
    logger.info(f"   策略模式: {data_strategy}")
    logger.info(f"   切换阈值: {accuracy_threshold}")
    
    if data_strategy not in ["online", "fixed", "adaptive"]:
        issues_found.append(f"❌ 无效的数据策略: {data_strategy}")
    
    # 6. 检查环境变量覆盖
    env_overrides = []
    env_vars = ["BATCH_SIZE", "MIXED_PRECISION", "TRAIN_SIZE", "MODEL_SAVE_THRESHOLD"]
    
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value is not None:
            env_overrides.append(f"   {env_var}={value}")
    
    if env_overrides:
        logger.info(f"\n🌍 环境变量覆盖:")
        for override in env_overrides:
            logger.info(override)
    else:
        logger.info(f"\n🌍 无环境变量覆盖")
    
    # 7. 检查目录结构
    required_dirs = ["models", "samples", "logs", "checkpoints"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            warnings.append(f"⚠️ 目录不存在: {dir_name}")
            recommendations.append(f"💡 建议创建目录: mkdir {dir_name}")
    
    # 8. 输出检查结果
    logger.info("\n" + "=" * 50)
    logger.info("🎯 检查结果汇总")
    logger.info("=" * 50)
    
    if issues_found:
        logger.info("❌ 发现问题:")
        for issue in issues_found:
            logger.info(f"   {issue}")
    else:
        logger.info("✅ 未发现严重配置问题")
    
    if warnings:
        logger.info("\n⚠️ 警告事项:")
        for warning in warnings:
            logger.info(f"   {warning}")
    
    if recommendations:
        logger.info("\n💡 优化建议:")
        for rec in recommendations:
            logger.info(f"   {rec}")
    
    # 9. 生成配置报告
    report = {
        "config_check_summary": {
            "issues_count": len(issues_found),
            "warnings_count": len(warnings),
            "recommendations_count": len(recommendations),
            "config_file_exists": os.path.exists(config_file),
            "current_config": {
                "batch_size": batch_size,
                "train_size": train_size,
                "model_save_threshold": model_threshold,
                "mixed_precision": mixed_precision,
                "data_strategy": data_strategy
            }
        }
    }
    
    # 保存报告
    os.makedirs("logs", exist_ok=True)
    with open("logs/config_check_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📄 详细报告已保存: logs/config_check_report.json")
    
    return len(issues_found) == 0


def generate_recommended_config():
    """生成推荐配置文件"""
    logger.info("\n🔧 生成推荐配置...")
    
    # 检测显卡类型（简单启发式）
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # 根据显卡推荐配置
            if "4060" in gpu_name or "4070" in gpu_name:
                recommended_batch_size = 128
                recommended_workers = 8
            elif "3060" in gpu_name or "1660" in gpu_name:
                recommended_batch_size = 64
                recommended_workers = 6
            elif "4090" in gpu_name or "3090" in gpu_name:
                recommended_batch_size = 256
                recommended_workers = 12
            else:
                recommended_batch_size = 64
                recommended_workers = 4
            
            logger.info(f"🎮 检测到显卡: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"💡 推荐batch_size: {recommended_batch_size}")
            
        else:
            logger.info("⚠️ 未检测到CUDA设备，使用CPU默认配置")
            recommended_batch_size = 32
            recommended_workers = 4
    except:
        recommended_batch_size = 64
        recommended_workers = 4
    
    recommended_config = {
        "training": {
            "batch_size": recommended_batch_size,
            "model_save_threshold": 0.95,
            "epoch": 50
        },
        "data": {
            "train_size": 200000,
            "validate_size": 10000,
            "dataloader_workers": recommended_workers,
            "data_strategy": "adaptive"
        },
        "hardware": {
            "mixed_precision": True,
            "compile_model": False,
            "memory_efficient": False
        }
    }
    
    # 保存推荐配置
    output_file = "configs/recommended_config.jsonc"
    os.makedirs("configs", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("// 自动生成的推荐配置\n")
        f.write("// 基于您的硬件环境优化\n")
        json.dump(recommended_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 推荐配置已保存: {output_file}")
    logger.info("💡 您可以将此文件重命名为 train_config.jsonc 使用")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="配置冲突检查工具")
    parser.add_argument("--generate", action="store_true", help="生成推荐配置文件")
    parser.add_argument("--fix", action="store_true", help="自动修复常见问题")
    
    args = parser.parse_args()
    
    # 执行配置检查
    is_config_ok = check_config_conflicts()
    
    if args.generate:
        generate_recommended_config()
    
    if args.fix:
        logger.info("\n🔧 自动修复功能暂未实现")
        logger.info("💡 请根据上述建议手动调整配置")
    
    # 返回状态码
    return 0 if is_config_ok else 1


if __name__ == "__main__":
    exit(main()) 