"""
原神OCR训练主程序 - 优化版
整合训练、验证、转换等核心功能，添加配置管理和性能优化支持
"""

import argparse
import torch
import torch.nn as nn
import gc
import os
import sys
from pathlib import Path

from mona.config import config, get_config_manager, optimize_for_rtx4060ti, print_optimization_status
from mona.utils import logger
from train import train as train_model


def cleanup_gpu():
    """GPU资源清理 - 使用优化版本"""
    if torch.cuda.is_available():
        # 使用性能优化器的清理功能
        try:
            from mona.training.performance_optimizer import PerformanceOptimizer
            optimizer = PerformanceOptimizer({})
            optimizer.cleanup_memory(aggressive=True)
            logger.info("✅ GPU资源已清理（优化版）")
        except ImportError:
            # 回退到基础清理
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("✅ GPU资源已清理（基础版）")
    else:
        logger.info("ℹ️ 未检测到CUDA设备")


def system_check():
    """增强的系统检查"""
    logger.info("🔧 系统状态检查")
    logger.info("=" * 40)
    
    # 获取配置管理器
    config_manager = get_config_manager()
    
    # GPU检查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"🎮 GPU: {gpu_name}")
        logger.info(f"💾 显存: {gpu_memory:.1f}GB (已使用: {memory_used:.1f}GB)")
        
        # RTX 4060 TI检测
        if "4060" in gpu_name:
            logger.info("🎯 检测到RTX 4060 TI - 可启用专项优化")
    else:
        logger.warning("⚠️ 未检测到CUDA设备")
    
    # PyTorch版本检查
    torch_version = torch.__version__
    logger.info(f"🔥 PyTorch版本: {torch_version}")
    
    # 检查torch.compile支持
    if hasattr(torch, 'compile') and torch_version >= "2.0":
        logger.info("✅ 支持torch.compile优化")
        if not config_manager.hardware["compile_model"]:
            logger.info("💡 建议启用模型编译: python main.py optimize --enable-compile")
    else:
        logger.info("⚠️ 不支持torch.compile（需要PyTorch 2.0+）")
    
    # 混合精度检查
    if config_manager.hardware["mixed_precision"]:
        logger.info("🚀 混合精度训练: 已启用")
    else:
        logger.info("💡 混合精度训练: 未启用 - 建议启用以提升性能")
    
    # 配置检查
    logger.info(f"📦 Batch Size: {config_manager.get('training', 'batch_size')}")
    logger.info(f"🔄 训练轮数: {config_manager.get('training', 'epoch')}")
    logger.info(f"💾 模型保存阈值: {config_manager.get('training', 'model_save_threshold')}")
    
    logger.info("=" * 40)


def generate_samples(count=10):
    """生成训练样本"""
    from mona.datagen.datagen import DataGen
    from mona.text import get_lexicon
    from PIL import ImageFont
    
    # 初始化
    lexicon = get_lexicon()
    fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
    datagen = DataGen(config, fonts, lexicon)
    
    # 创建输出目录
    os.makedirs("samples", exist_ok=True)
    
    logger.info(f"📸 生成 {count} 个样本...")
    for i in range(count):
        im, text = datagen.generate_image()
        im.save(f"samples/sample_{i:03d}_{text}.png")
    
    logger.info(f"✅ 已生成 {count} 个样本到 samples/ 目录")


def validate_model(model_path):
    """验证模型"""
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        return
    
    from mona.nn.model2 import Model2
    from mona.text import get_lexicon
    from mona.nn import predict as predict_net
    from mona.datagen.datagen import DataGen
    from PIL import ImageFont
    import torchvision.transforms as transforms
    
    # 加载模型
    lexicon = get_lexicon()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Model2(lexicon.lexicon_size(), 1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    
    # 生成测试数据
    fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
    datagen = DataGen(config, fonts, lexicon)
    
    correct = 0
    total = 20  # 测试20个样本
    
    logger.info("🔍 开始验证...")
    with torch.no_grad():
        for i in range(total):
            im, true_text = datagen.generate_image()
            tensor = transforms.ToTensor()(im).unsqueeze(0).to(device)
            
            predicted = predict_net(net, tensor, lexicon)
            if predicted[0] == true_text:
                correct += 1
            
            if i < 5:  # 显示前5个结果
                logger.info(f"真实: {true_text} | 预测: {predicted[0]} | {'✓' if predicted[0] == true_text else '✗'}")
    
    accuracy = correct / total
    logger.info(f"📊 验证完成: {correct}/{total} ({accuracy:.2%})")


def convert_to_onnx(model_path, output_path=None):
    """转换模型为ONNX格式"""
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        return
    
    from mona.nn.model2 import Model2
    from mona.text import get_lexicon
    
    # 设置输出路径
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    lexicon = get_lexicon()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    net = Model2(lexicon.lexicon_size(), 1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 32, 384).to(device)
    
    # 导出ONNX
    torch.onnx.export(
        net,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"📄 ONNX模型已保存: {output_path}")


def list_models():
    """列出所有模型文件"""
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("⚠️ models目录不存在")
        return
    
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        logger.info("📂 未找到模型文件")
        return
    
    logger.info("📋 可用模型:")
    for model_file in sorted(model_files):
        size = model_file.stat().st_size / 1024 / 1024  # MB
        logger.info(f"  📄 {model_file.name} ({size:.1f}MB)")


def evaluate_models(args):
    """批量评估模型 - 在线数据生成版"""
    try:
        from scripts.model_evaluator import ModelEvaluator
        
        logger.info("🚀 启动模型评估（在线数据生成版）...")
        
        # 初始化评估器
        evaluator = ModelEvaluator("models")
        
        if args.single:
            # 评估单个模型
            model_path = Path(args.single)
            if not model_path.exists():
                logger.error(f"❌ 模型文件不存在: {model_path}")
                return
            
            result = evaluator.evaluate_single_model(model_path, args.test_size)
            if result:
                logger.info(f"\n🎉 单模型评估完成:")
                logger.info(f"   📄 模型: {result['model_name']}")
                logger.info(f"   🎯 准确率: {result['exact_accuracy']:.6f}")
                logger.info(f"   📝 字符准确率: {result['char_accuracy']:.6f}")
                logger.info(f"   ⚡ 速度: {result['avg_inference_time_ms']:.2f}ms")
                logger.info(f"   🏆 综合评分: {result['overall_score']:.6f}")
                logger.info(f"   📊 实际测试样本: {result['total_samples']:,}")
        else:
            # 批量评估
            results = evaluator.evaluate_all_models(args.test_size)
            
            if results:
                # 生成报告
                report_file = evaluator.generate_report("logs")
                
                # 显示最佳模型
                best_model = evaluator.get_best_model()
                logger.info(f"\n🏆 最佳模型: {best_model['model_name']}")
                logger.info(f"   🎯 准确率: {best_model['exact_accuracy']:.6f}")
                logger.info(f"   📝 字符准确率: {best_model['char_accuracy']:.6f}")
                logger.info(f"   ⚡ 速度: {best_model['avg_inference_time_ms']:.2f}ms")
                logger.info(f"   🏆 综合评分: {best_model['overall_score']:.6f}")
                logger.info(f"   📊 测试样本: {best_model['total_samples']:,}")
                
                # 显示模型差异统计
                if len(results) > 1:
                    accuracies = [r["exact_accuracy"] for r in results]
                    worst_model = results[-1]
                    accuracy_diff = best_model["exact_accuracy"] - worst_model["exact_accuracy"]
                    logger.info(f"\n📈 模型性能差异:")
                    logger.info(f"   最佳 vs 最差: {accuracy_diff:.6f} ({accuracy_diff*100:.4f}%)")
                    logger.info(f"   准确率标准差: {__import__('numpy').std(accuracies):.6f}")
                
                # 复制最佳模型
                if args.copy_best:
                    evaluator.copy_best_model()
                
                logger.info(f"\n📊 详细报告已生成: {report_file}")
            else:
                logger.error("❌ 评估失败或未找到模型")
    
    except ImportError as e:
        logger.error(f"❌ 导入评估模块失败: {e}")
        logger.info("💡 请确保所有依赖已安装: pip install pandas matplotlib seaborn")
    except Exception as e:
        logger.error(f"❌ 评估过程出错: {e}")


def competitive_evaluate_models(args):
    """竞争式并行模型评估 - 错误即淘汰"""
    try:
        from scripts.competitive_evaluator import CompetitiveEvaluator
        
        logger.info("🏁 启动竞争式模型评估...")
        logger.info("💥 规则: 所有模型同时竞争，出错即淘汰，最后存活者获胜")
        logger.info("🌊 无限在线数据生成，真正的鲁棒性测试")
        
        # 初始化竞争式评估器
        evaluator = CompetitiveEvaluator("models", args.max_workers)
        
        # 运行竞争式评估
        report = evaluator.run_competitive_evaluation(
            min_survival_time=args.min_survival_time,
            max_evaluation_time=args.max_evaluation_time
        )
        
        if report:
            # 保存报告
            evaluator.save_report(report, "logs")
            
            # 复制获胜模型
            if args.copy_winner:
                evaluator.copy_winner_model(report)
        else:
            logger.error("❌ 竞争式评估失败")
    
    except ImportError as e:
        logger.error(f"❌ 导入竞争式评估模块失败: {e}")
        logger.info("💡 请确保项目结构正确，scripts/competitive_evaluator.py 存在")
    except Exception as e:
        logger.error(f"❌ 竞争式评估过程出错: {e}")


def handle_optimization_commands(args):
    """处理优化相关命令"""
    config_manager = get_config_manager()
    
    if args.action == "status":
        config_manager.print_config_summary()
        print_optimization_status()
    
    elif args.action == "enable-mixed-precision":
        from mona.config import enable_mixed_precision
        enable_mixed_precision()
    
    elif args.action == "enable-compile":
        from mona.config import enable_model_compilation
        enable_model_compilation()
    
    elif args.action == "rtx4060ti":
        optimize_for_rtx4060ti()
    
    elif args.action == "benchmark":
        run_performance_benchmark()
    
    else:
        logger.error(f"❌ 未知的优化命令: {args.action}")


def run_performance_benchmark():
    """运行性能基准测试"""
    logger.info("🏃 开始性能基准测试...")
    
    try:
        from mona.training.performance_optimizer import PerformanceOptimizer
        config_manager = get_config_manager()
        
        optimizer = PerformanceOptimizer(config_manager.hardware)
        
        # 创建测试数据
        test_tensor = torch.randn(config["batch_size"], 1, 32, 384)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
        
        import time
        
        # 预热
        for _ in range(10):
            with optimizer.autocast_context():
                _ = test_tensor * 2
        
        # 基准测试
        start_time = time.time()
        for i in range(100):
            with optimizer.autocast_context():
                result = test_tensor * 2
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        logger.info(f"⚡ 平均处理时间: {avg_time*1000:.2f}ms")
        logger.info(f"📊 理论吞吐量: {config['batch_size']/avg_time:.1f} samples/s")
        
        # 内存使用情况
        memory_info = optimizer.get_memory_info()
        if "cuda_utilization" in memory_info:
            logger.info(f"💾 显存使用率: {memory_info['cuda_utilization']:.1f}%")
        
        logger.info("✅ 基准测试完成")
        
    except Exception as e:
        logger.error(f"❌ 基准测试失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="原神OCR训练工具 - 优化版")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证模型')
    validate_parser.add_argument('model', help='模型文件路径')
    
    # 生成样本命令
    generate_parser = subparsers.add_parser('generate', help='生成训练样本')
    generate_parser.add_argument('--count', type=int, default=10, help='生成样本数量')
    
    # 转换命令
    convert_parser = subparsers.add_parser('convert', help='转换模型为ONNX')
    convert_parser.add_argument('model', help='模型文件路径')
    convert_parser.add_argument('--output', help='输出文件路径')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='性能优化管理')
    optimize_parser.add_argument('action', choices=[
        'status', 'enable-mixed-precision', 'enable-compile', 'rtx4060ti', 'benchmark'
    ], help='优化操作')
    
    # 其他命令
    subparsers.add_parser('check', help='检查系统状态')
    subparsers.add_parser('clean', help='清理GPU资源')
    subparsers.add_parser('list', help='列出所有模型')
    
    # 新增模型评估命令
    eval_parser = subparsers.add_parser('evaluate', help='批量评估模型（在线数据生成）')
    eval_parser.add_argument('--test-size', type=int, default=10000, help='测试样本数量（在线生成，默认10000）')
    eval_parser.add_argument('--copy-best', action='store_true', help='复制最佳模型')
    eval_parser.add_argument('--single', help='评估单个模型文件')
    
    # 竞争式评估命令
    compete_parser = subparsers.add_parser('compete', help='竞争式并行评估（错误即淘汰）')
    compete_parser.add_argument('--min-survival-time', type=int, default=60, help='最小存活时间(秒)')
    compete_parser.add_argument('--max-evaluation-time', type=int, default=1800, help='最大评估时间(秒)')
    compete_parser.add_argument('--max-workers', type=int, default=None, help='最大并行工作数')
    compete_parser.add_argument('--copy-winner', action='store_true', help='复制获胜模型到models/models文件夹')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'validate':
        validate_model(args.model)
    elif args.command == 'generate':
        generate_samples(args.count)
    elif args.command == 'convert':
        convert_to_onnx(args.model, args.output)
    elif args.command == 'optimize':
        handle_optimization_commands(args)
    elif args.command == 'check':
        system_check()
    elif args.command == 'clean':
        cleanup_gpu()
    elif args.command == 'list':
        list_models()
    elif args.command == 'evaluate':
        evaluate_models(args)
    elif args.command == 'compete':
        competitive_evaluate_models(args)
    else:
        parser.print_help()
        
        # 如果没有参数，显示快速入门
        print("\n" + "="*50)
        print("🚀 快速入门:")
        print("  python main.py check                    # 系统检查")
        print("  python main.py optimize status          # 查看优化状态")
        print("  python main.py optimize rtx4060ti       # RTX 4060TI专项优化")
        print("  python main.py train                    # 开始训练")
        print("  python main.py evaluate                 # 批量评估模型（在线生成10K样本）")
        print("  python main.py evaluate --test-size 20000  # 使用2万样本精确评估")
        print("  python main.py evaluate --copy-best     # 评估并复制最佳模型")
        print("  python main.py compete                  # 🔥竞争式评估（错误即淘汰）")
        print("  python main.py compete --copy-winner    # 竞争式评估并复制冠军到models/models文件夹")
        print("  python main.py list                     # 列出所有模型")
        print("  python main.py optimize benchmark       # 性能基准测试")
        print("="*50)


if __name__ == "__main__":
    main() 