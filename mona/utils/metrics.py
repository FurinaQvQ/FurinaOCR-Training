"""
训练指标监控和可视化模块
"""
import time
import psutil
import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    batch: int
    loss: float
    accuracy: float
    learning_rate: float
    gpu_memory_mb: float
    cpu_percent: float
    samples_per_second: float
    timestamp: float


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[TrainingMetrics] = []
        self.loss_window = deque(maxlen=window_size)
        self.accuracy_window = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=window_size)
        
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.total_samples_processed = 0
    
    def record_batch(self, epoch: int, batch: int, loss: float, 
                    accuracy: float, lr: float, batch_size: int) -> None:
        """记录批次指标"""
        current_time = time.time()
        
        # 计算吞吐量
        time_delta = current_time - self.last_log_time
        throughput = batch_size / time_delta if time_delta > 0 else 0
        
        # 获取系统资源使用情况
        gpu_memory = self._get_gpu_memory_usage()
        cpu_percent = psutil.cpu_percent()
        
        # 创建指标记录
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            loss=loss,
            accuracy=accuracy,
            learning_rate=lr,
            gpu_memory_mb=gpu_memory,
            cpu_percent=cpu_percent,
            samples_per_second=throughput,
            timestamp=current_time
        )
        
        # 更新历史记录和滑动窗口
        self.metrics_history.append(metrics)
        self.loss_window.append(loss)
        self.accuracy_window.append(accuracy)
        self.throughput_window.append(throughput)
        
        self.total_samples_processed += batch_size
        self.last_log_time = current_time
    
    def get_recent_stats(self) -> Dict[str, float]:
        """获取最近的统计信息"""
        if not self.loss_window:
            return {}
        
        return {
            "avg_loss": sum(self.loss_window) / len(self.loss_window),
            "avg_accuracy": sum(self.accuracy_window) / len(self.accuracy_window),
            "avg_throughput": sum(self.throughput_window) / len(self.throughput_window),
            "total_samples": self.total_samples_processed,
            "training_time_hours": (time.time() - self.start_time) / 3600
        }
    
    def get_best_metrics(self) -> Dict[str, float]:
        """获取最佳指标"""
        if not self.metrics_history:
            return {}
        
        best_accuracy = max(m.accuracy for m in self.metrics_history)
        best_loss = min(m.loss for m in self.metrics_history)
        max_throughput = max(m.samples_per_second for m in self.metrics_history)
        
        return {
            "best_accuracy": best_accuracy,
            "best_loss": best_loss,
            "max_throughput": max_throughput
        }
    
    def save_metrics(self, filepath: str) -> None:
        """保存指标到文件"""
        metrics_data = {
            "training_summary": {
                **self.get_recent_stats(),
                **self.get_best_metrics()
            },
            "history": [
                {
                    "epoch": m.epoch,
                    "batch": m.batch,
                    "loss": m.loss,
                    "accuracy": m.accuracy,
                    "learning_rate": m.learning_rate,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "cpu_percent": m.cpu_percent,
                    "samples_per_second": m.samples_per_second,
                    "timestamp": m.timestamp
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    def _get_gpu_memory_usage(self) -> float:
        """获取GPU内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def print_progress(self, epoch: int, total_epochs: int, 
                      batch: int, total_batches: int) -> None:
        """打印训练进度"""
        recent_stats = self.get_recent_stats()
        
        # 计算进度百分比
        epoch_progress = (batch / total_batches) * 100
        total_progress = ((epoch - 1) * total_batches + batch) / (total_epochs * total_batches) * 100
        
        print(f"\r[Epoch {epoch:3d}/{total_epochs}] "
              f"[Batch {batch:6d}/{total_batches}] "
              f"Progress: {epoch_progress:5.1f}% | "
              f"Overall: {total_progress:5.1f}% | "
              f"Loss: {recent_stats.get('avg_loss', 0):.6f} | "
              f"Acc: {recent_stats.get('avg_accuracy', 0):.4f} | "
              f"Throughput: {recent_stats.get('avg_throughput', 0):.1f} samples/s", 
              end="", flush=True)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = 0.0
        self.should_stop = False
    
    def __call__(self, accuracy: float) -> bool:
        """检查是否应该早停"""
        if accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_accuracy = 0.0
        self.should_stop = False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer, mode: str = 'plateau', factor: float = 0.5, 
                 patience: int = 5, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = float('inf') if mode == 'min' else 0.0
        self.counter = 0
    
    def step(self, metric: float) -> bool:
        """更新学习率，返回是否发生了调整"""
        is_better = (metric < self.best_metric) if self.mode == 'min' else (metric > self.best_metric)
        
        if is_better:
            self.best_metric = metric
            self.counter = 0
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self._reduce_lr()
            self.counter = 0
            return True
        
        return False
    
    def _reduce_lr(self):
        """降低学习率"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if new_lr != old_lr:
                print(f"\n学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")


# 全局指标收集器
metrics_collector = MetricsCollector() 