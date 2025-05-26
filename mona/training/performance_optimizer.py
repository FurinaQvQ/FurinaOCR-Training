"""
训练性能优化器
集成混合精度、模型编译、内存优化等高级性能优化技术
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any
import psutil
import time
import gc
from contextlib import contextmanager

from mona.utils import logger


class PerformanceOptimizer:
    """训练性能优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化性能优化器
        
        Args:
            config: 配置字典，包含硬件和训练相关配置
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 混合精度训练
        self.use_amp = config.get("mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 模型编译（PyTorch 2.0+）
        self.use_compile = config.get("compile_model", False)
        
        # 性能监控
        self.performance_stats = {
            "batch_times": [],
            "memory_usage": [],
            "gpu_utilization": [],
        }
        
        # CUDA优化设置
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if config.get("cuda_benchmark", True):
                torch.backends.cudnn.benchmark = True
                logger.info("启用CUDNN benchmark模式")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """优化模型"""
        # 模型编译优化
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
                logger.info("✅ 模型编译优化已启用 (torch.compile)")
            except Exception as e:
                logger.warning(f"模型编译失败: {e}")
        
        # 混合精度优化
        if self.use_amp:
            logger.info("✅ 混合精度训练已启用 (AMP)")
        
        return model
    
    def optimize_dataloader(self, dataloader_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """优化数据加载器设置"""
        if self.device.type == "cuda":
            dataloader_kwargs.update({
                "pin_memory": self.config.get("dataloader_pin_memory", True),
                "persistent_workers": self.config.get("dataloader_persistent_workers", True),
                "num_workers": min(self.config.get("dataloader_workers", 8), psutil.cpu_count()),
            })
        
        # 动态调整batch_size以适应内存
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            current_batch_size = dataloader_kwargs.get("batch_size", 128)
            
            # 简单的内存估算和批次大小调整
            memory_per_sample = 32 * 384 * 4  # 估算每个样本的内存使用
            max_batch_size = min(available_memory // (memory_per_sample * 10), 512)  # 保留90%内存
            
            if current_batch_size > max_batch_size:
                dataloader_kwargs["batch_size"] = max_batch_size
                logger.warning(f"批次大小调整为 {max_batch_size} 以适应显存限制")
        
        return dataloader_kwargs
    
    @contextmanager
    def autocast_context(self):
        """自动混合精度上下文管理器"""
        if self.use_amp:
            with autocast():
                yield
        else:
            yield
    
    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """优化的反向传播步骤"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    def profile_step(self, start_time: float, batch_size: int):
        """性能分析步骤"""
        step_time = time.time() - start_time
        self.performance_stats["batch_times"].append(step_time)
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.performance_stats["memory_usage"].append(memory_mb)
        
        # 计算吞吐量
        throughput = batch_size / step_time
        return throughput
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存使用信息"""
        info = {}
        
        # CUDA内存
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            info.update({
                "cuda_allocated_gb": memory_allocated,
                "cuda_reserved_gb": memory_reserved,
                "cuda_total_gb": memory_total,
                "cuda_utilization": memory_allocated / memory_total * 100,
            })
        
        # 系统内存
        system_memory = psutil.virtual_memory()
        info.update({
            "system_used_gb": system_memory.used / 1024**3,
            "system_total_gb": system_memory.total / 1024**3,
            "system_utilization": system_memory.percent,
        })
        
        return info
    
    def cleanup_memory(self, aggressive: bool = False):
        """内存清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
        
        gc.collect()
        
        if aggressive:
            # 强制Python垃圾回收
            for _ in range(3):
                gc.collect()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_stats["batch_times"]:
            return {}
        
        import numpy as np
        
        batch_times = np.array(self.performance_stats["batch_times"])
        memory_usage = np.array(self.performance_stats["memory_usage"])
        
        summary = {
            "avg_batch_time": float(np.mean(batch_times)),
            "median_batch_time": float(np.median(batch_times)),
            "max_batch_time": float(np.max(batch_times)),
            "min_batch_time": float(np.min(batch_times)),
            "std_batch_time": float(np.std(batch_times)),
        }
        
        if len(memory_usage) > 0:
            summary.update({
                "avg_memory_mb": float(np.mean(memory_usage)),
                "max_memory_mb": float(np.max(memory_usage)),
                "min_memory_mb": float(np.min(memory_usage)),
            })
        
        return summary
    
    def suggest_optimizations(self) -> list[str]:
        """基于性能数据建议优化措施"""
        suggestions = []
        
        memory_info = self.get_memory_info()
        perf_summary = self.get_performance_summary()
        
        # 内存优化建议
        if "cuda_utilization" in memory_info:
            cuda_util = memory_info["cuda_utilization"]
            if cuda_util > 90:
                suggestions.append("🔴 显存使用率过高(>90%)，建议降低batch_size")
            elif cuda_util < 60:
                suggestions.append("🟡 显存使用率较低(<60%)，可以尝试增加batch_size")
            else:
                suggestions.append("🟢 显存使用率适中")
        
        # 性能优化建议
        if not self.use_amp and torch.cuda.is_available():
            suggestions.append("💡 建议启用混合精度训练以提升性能")
        
        if not self.use_compile and hasattr(torch, 'compile'):
            suggestions.append("💡 建议启用torch.compile以提升推理速度")
        
        # 数据加载优化建议
        if perf_summary:
            avg_time = perf_summary.get("avg_batch_time", 0)
            if avg_time > 1.0:  # 每批次超过1秒
                suggestions.append("⚡ 训练速度较慢，检查数据加载器num_workers设置")
        
        return suggestions


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_interval: int = 100):
        """
        初始化训练监控器
        
        Args:
            log_interval: 日志记录间隔
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # 监控指标
        self.metrics = {
            "losses": [],
            "learning_rates": [],
            "batch_times": [],
            "throughputs": [],
        }
    
    def log_step(self, loss: float, lr: float, batch_size: int, throughput: float):
        """记录训练步骤"""
        self.step_count += 1
        current_time = time.time()
        
        # 更新指标
        self.metrics["losses"].append(loss)
        self.metrics["learning_rates"].append(lr)
        self.metrics["throughputs"].append(throughput)
        
        # 定期输出日志
        if self.step_count % self.log_interval == 0:
            elapsed_time = current_time - self.start_time
            recent_throughput = sum(self.metrics["throughputs"][-self.log_interval:]) / self.log_interval
            
            logger.info(
                f"Step {self.step_count:6d} | "
                f"Loss: {loss:.6f} | "
                f"LR: {lr:.2e} | "
                f"Throughput: {recent_throughput:.1f} samples/s | "
                f"Elapsed: {elapsed_time:.1f}s"
            )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.metrics["losses"]:
            return {}
        
        import numpy as np
        
        return {
            "total_steps": self.step_count,
            "avg_loss": float(np.mean(self.metrics["losses"])),
            "recent_loss": float(np.mean(self.metrics["losses"][-100:])) if len(self.metrics["losses"]) >= 100 else None,
            "avg_throughput": float(np.mean(self.metrics["throughputs"])),
            "total_time": time.time() - self.start_time,
        } 