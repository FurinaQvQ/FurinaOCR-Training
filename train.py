"""
原神OCR训练脚本 - 优化版
集成混合精度训练、性能优化器和高级监控功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import ImageFont
from typing import Tuple
import datetime
import gc

from mona.datagen.datagen import DataGen
from mona.config import config, get_config_manager
from mona.nn import predict as predict_net
from mona.nn.model2 import Model2
from mona.text import get_lexicon
from mona.utils import logger

# 导入性能优化器
from mona.training.performance_optimizer import PerformanceOptimizer, TrainingMonitor

# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# 获取配置管理器
config_manager = get_config_manager()

# 初始化性能优化器
performance_optimizer = PerformanceOptimizer(config_manager.hardware)
logger.info("✅ 性能优化器已初始化")

# 初始化
lexicon = get_lexicon()
fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
datagen = DataGen(config, fonts, lexicon)

logger.info(f"词汇表大小: {lexicon.lexicon_size()}")

# 显示优化状态
if config_manager.hardware["mixed_precision"]:
    logger.info("🚀 混合精度训练已启用")
if config_manager.hardware["compile_model"]:
    logger.info("🔥 模型编译优化已启用")


class AddGaussianNoise(nn.Module):
    """高斯噪声数据增强"""
    def __init__(self, mean: float = 0.0, std: float = 1/255):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return tensor


class OnlineDataSet(Dataset):
    """在线数据集"""
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        im, text = datagen.generate_image()
        tensor = transforms.ToTensor()(im)
        return tensor, text


def get_target(s: list, lexicon) -> Tuple[torch.Tensor, torch.Tensor]:
    """转换目标字符串为张量"""
    target_length = []
    target_vector = []
    
    for target in s:
        target_length.append(len(target))
        for char in target:
            index = lexicon.word_to_index.get(char, 0)
            target_vector.append(index)

    return torch.LongTensor(target_vector), torch.LongTensor(target_length)


def validate_model(net: nn.Module, validate_loader: DataLoader, lexicon, device: str) -> float:
    """模型验证"""
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, label in validate_loader:
            x = x.to(device)
            
            # 使用混合精度进行推理
            with performance_optimizer.autocast_context():
                predict = predict_net(net, x, lexicon)
            
            correct += sum([1 if predict[i] == label[i] else 0 for i in range(len(label))])
            total += len(label)

    accuracy = correct / total if total > 0 else 0.0
    net.train()
    return accuracy


def cleanup_resources():
    """清理资源"""
    performance_optimizer.cleanup_memory(aggressive=True)
    logger.info("💾 内存清理完成")


def train():
    """优化的训练函数"""
    logger.info("开始优化训练...")
    
    # 模型初始化
    net = Model2(lexicon.lexicon_size(), 1).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in net.parameters()):,}")
    
    # 应用性能优化
    net = performance_optimizer.optimize_model(net)
    
    # 预训练模型加载
    if config.get("pretrain", False):
        try:
            checkpoint = torch.load(f"models/{config['pretrain_name']}", 
                                  map_location=device, weights_only=True)
            net.load_state_dict(checkpoint)
            logger.info(f"✅ 加载预训练模型: {config['pretrain_name']}")
        except Exception as e:
            logger.warning(f"⚠️ 加载预训练模型失败: {e}")

    # 数据增强
    data_aug_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(1, 1),
                transforms.GaussianBlur(3, 3),
                transforms.GaussianBlur(5, 5),
            ])], p=0.5),
        transforms.RandomApply([
            transforms.RandomCrop(size=(31, 383)),
            transforms.Resize((32, 384), antialias=True),
        ], p=0.5),
        AddGaussianNoise(mean=0, std=1/255),
    ])
    
    # 数据集和数据加载器 - 应用性能优化
    train_dataset = OnlineDataSet(config['train_size'])
    validate_dataset = OnlineDataSet(config['validate_size'])

    # 优化数据加载器设置
    train_loader_kwargs = {
        "dataset": train_dataset,
        "shuffle": True,
        "batch_size": config["batch_size"],
    }
    train_loader_kwargs = performance_optimizer.optimize_dataloader(train_loader_kwargs)
    train_loader = DataLoader(**train_loader_kwargs)
    
    validate_loader_kwargs = {
        "dataset": validate_dataset,
        "batch_size": config["batch_size"],
    }
    validate_loader_kwargs = performance_optimizer.optimize_dataloader(validate_loader_kwargs)
    validate_loader = DataLoader(**validate_loader_kwargs)

    # 优化器和损失函数
    optimizer = optim.Adadelta(net.parameters())
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)
    
    # 初始化训练监控器
    training_monitor = TrainingMonitor(log_interval=config["print_per"])
    
    # 训练循环
    total_epochs = config["epoch"]
    batch_count = 1
    best_accuracy = 0.0
    start_time = datetime.datetime.now()
    
    logger.info(f"🚀 训练设置: {total_epochs} epochs, batch_size={config['batch_size']}")
    
    # 显示性能优化建议
    suggestions = performance_optimizer.suggest_optimizations()
    for suggestion in suggestions:
        logger.info(f"💡 {suggestion}")
    
    for epoch in range(1, total_epochs + 1):
        for batch_idx, (x, label) in enumerate(train_loader, 1):
            step_start_time = datetime.datetime.now().timestamp()
            
            try:
                optimizer.zero_grad()
                target_vector, target_lengths = get_target(label, lexicon)
                target_vector, target_lengths = target_vector.to(device), target_lengths.to(device)
                x = x.to(device)

                # 数据增强
                x = data_aug_transform(x)
                batch_size = x.size(0)

                # 前向传播 - 使用混合精度
                with performance_optimizer.autocast_context():
                    y = net(x)
                    input_lengths = torch.full((batch_size,), 24, device=device, dtype=torch.long)
                    loss = ctc_loss(y, target_vector, input_lengths, target_lengths)
                
                # 反向传播 - 使用优化的反向传播
                performance_optimizer.backward_step(loss, optimizer)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                # 性能分析
                throughput = performance_optimizer.profile_step(step_start_time, batch_size)
                
                # 记录训练步骤
                current_lr = optimizer.param_groups[0].get('lr', config.get('learning_rate', 1.0))
                training_monitor.log_step(
                    loss=loss.item(),
                    lr=current_lr,
                    batch_size=batch_size,
                    throughput=throughput
                )

                # 验证和保存
                if batch_count % config["save_per"] == 0:
                    logger.info("🔍 验证模型...")
                    val_accuracy = validate_model(net, validate_loader, lexicon, device)
                    logger.info(f"📊 验证准确率: {val_accuracy:.6f} ({val_accuracy*100:.2f}%)")
                    
                    # 性能统计
                    memory_info = performance_optimizer.get_memory_info()
                    if "cuda_utilization" in memory_info:
                        logger.info(f"💾 显存使用率: {memory_info['cuda_utilization']:.1f}%")
                    
                    # 模型保存策略
                    threshold = config.get("model_save_threshold", 1.0)
                    
                    if val_accuracy >= threshold:
                        # 保存正式模型
                        model_path = f"models/model_acc{val_accuracy:.6f}_epoch{epoch}.pt"
                        torch.save(net.state_dict(), model_path)
                        logger.info(f"🎉 达到阈值 {threshold:.3f}！模型已保存: {model_path}")
                        
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                    else:
                        logger.info(f"📊 未达到保存阈值 {threshold:.3f}")
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            logger.info(f"🔄 更新最佳准确率: {best_accuracy:.6f}")
                    
                    # 保存训练检查点
                    torch.save(net.state_dict(), "models/model_training.pt")
                    logger.info("💾 训练检查点已保存")
                    
                    # 清理资源
                    cleanup_resources()

                batch_count += 1
                
            except Exception as e:
                logger.error(f"❌ 训练出错: {e}")
                cleanup_resources()
                continue
    
    # 训练完成统计
    logger.info("=" * 50)
    logger.info("🎉 训练完成!")
    logger.info(f"🏆 最佳准确率: {best_accuracy:.6f}")
    
    # 性能统计摘要
    perf_summary = performance_optimizer.get_performance_summary()
    if perf_summary:
        logger.info(f"⚡ 平均批次时间: {perf_summary.get('avg_batch_time', 0):.3f}s")
        logger.info(f"💾 平均内存使用: {perf_summary.get('avg_memory_mb', 0):.1f}MB")
    
    training_stats = training_monitor.get_training_stats()
    if training_stats:
        logger.info(f"📊 平均吞吐量: {training_stats.get('avg_throughput', 0):.1f} samples/s")
        logger.info(f"⏱️ 总训练时间: {training_stats.get('total_time', 0):.1f}s")
    
    logger.info("=" * 50)
    
    # 最终验证
    final_accuracy = validate_model(net, validate_loader, lexicon, device)
    logger.info(f"🎯 最终验证准确率: {final_accuracy:.6f}")
    
    threshold = config.get("model_save_threshold", 1.0)
    if final_accuracy >= threshold:
        final_model_path = f"models/final_model_acc{final_accuracy:.6f}.pt"
        torch.save(net.state_dict(), final_model_path)
        logger.info(f"🎉 最终模型已保存: {final_model_path}")
    else:
        logger.info(f"📊 最终准确率未达到保存阈值 {threshold:.3f}")
    
    # 显示优化建议
    final_suggestions = performance_optimizer.suggest_optimizations()
    if final_suggestions:
        logger.info("\n💡 性能优化建议:")
        for suggestion in final_suggestions:
            logger.info(f"   {suggestion}")


if __name__ == "__main__":
    train() 