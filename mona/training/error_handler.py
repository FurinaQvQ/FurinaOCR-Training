"""
训练错误处理和恢复系统
提供自动错误恢复、训练状态保存、异常监控等功能
"""

import torch
import pickle
import time
import traceback
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from contextlib import contextmanager
import signal
import sys

from mona.utils import logger


class TrainingCheckpoint:
    """训练检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 检查点文件路径
        self.state_file = self.checkpoint_dir / "training_state.pkl"
        self.metadata_file = self.checkpoint_dir / "training_metadata.json"
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       batch_idx: int,
                       loss: float,
                       best_accuracy: float,
                       **kwargs):
        """
        保存训练检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前轮数
            batch_idx: 当前批次索引
            loss: 当前损失
            best_accuracy: 最佳准确率
            **kwargs: 其他需要保存的状态
        """
        try:
            # 保存模型和优化器状态
            checkpoint_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': loss,
                'best_accuracy': best_accuracy,
                'timestamp': datetime.now().isoformat(),
                'extra_state': kwargs
            }
            
            # 保存状态文件
            torch.save(checkpoint_state, self.state_file)
            
            # 保存元数据
            metadata = {
                'last_saved': datetime.now().isoformat(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': float(loss),
                'best_accuracy': float(best_accuracy),
                'file_size': self.state_file.stat().st_size
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 检查点已保存: epoch={epoch}, batch={batch_idx}, loss={loss:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return False
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
        """
        加载训练检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            
        Returns:
            训练状态字典，如果加载失败返回None
        """
        if not self.state_file.exists():
            logger.info("未找到检查点文件")
            return None
        
        try:
            checkpoint = torch.load(self.state_file, map_location='cpu', weights_only=False)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"✅ 检查点已加载: epoch={checkpoint['epoch']}, batch={checkpoint['batch_idx']}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """清理旧的检查点文件"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoint_files) > keep_count:
            # 按修改时间排序，删除最旧的文件
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            for file_to_remove in checkpoint_files[:-keep_count]:
                file_to_remove.unlink()
                logger.info(f"删除旧检查点: {file_to_remove.name}")


class ErrorHandler:
    """训练错误处理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        """
        初始化错误处理器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_log = []
        
        # 错误统计
        self.error_counts = {
            'cuda_oom': 0,
            'data_loading': 0,
            'model_forward': 0,
            'optimizer_step': 0,
            'checkpoint_save': 0,
            'unknown': 0
        }
    
    def classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        error_str = str(error).lower()
        
        if 'cuda out of memory' in error_str or 'out of memory' in error_str:
            return 'cuda_oom'
        elif 'dataloader' in error_str or 'dataset' in error_str:
            return 'data_loading'
        elif 'forward' in error_str or 'input' in error_str:
            return 'model_forward'
        elif 'optimizer' in error_str or 'backward' in error_str:
            return 'optimizer_step'
        elif 'checkpoint' in error_str or 'save' in error_str:
            return 'checkpoint_save'
        else:
            return 'unknown'
    
    def handle_cuda_oom(self, batch_size: int) -> int:
        """处理CUDA内存不足错误"""
        logger.warning("检测到CUDA内存不足，正在清理内存...")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 建议新的批次大小
        new_batch_size = max(batch_size // 2, 1)
        logger.warning(f"建议将批次大小从 {batch_size} 调整为 {new_batch_size}")
        
        return new_batch_size
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误信息"""
        error_type = self.classify_error(error)
        self.error_counts[error_type] += 1
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        logger.error(f"训练错误 [{error_type}]: {error} (上下文: {context})")
        
        # 如果错误频繁，输出警告
        if self.error_counts[error_type] > 5:
            logger.warning(f"错误类型 '{error_type}' 已发生 {self.error_counts[error_type]} 次")
    
    @contextmanager
    def error_context(self, context: str, auto_retry: bool = True):
        """错误处理上下文管理器"""
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                yield
                break  # 成功执行，退出重试循环
                
            except Exception as e:
                self.log_error(e, context)
                
                if not auto_retry or retry_count >= self.max_retries:
                    raise  # 重新抛出异常
                
                retry_count += 1
                logger.warning(f"重试第 {retry_count}/{self.max_retries} 次，{self.retry_delay}秒后继续...")
                time.sleep(self.retry_delay)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误统计摘要"""
        total_errors = sum(self.error_counts.values())
        
        summary = {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_log[-5:] if self.error_log else [],
            'most_frequent_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if total_errors > 0 else None
        }
        
        return summary


class GracefulInterruptHandler:
    """优雅中断处理器"""
    
    def __init__(self, checkpoint_callback: Optional[Callable] = None):
        """
        初始化中断处理器
        
        Args:
            checkpoint_callback: 中断时的检查点保存回调函数
        """
        self.checkpoint_callback = checkpoint_callback
        self.interrupted = False
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        if self.interrupted:
            logger.warning("强制退出...")
            sys.exit(1)
        
        self.interrupted = True
        logger.info("收到中断信号，正在优雅退出...")
        
        if self.checkpoint_callback:
            try:
                logger.info("保存中断检查点...")
                self.checkpoint_callback()
                logger.info("✅ 检查点已保存")
            except Exception as e:
                logger.error(f"保存检查点失败: {e}")
    
    def check_interrupt(self) -> bool:
        """检查是否被中断"""
        return self.interrupted


class TrainingStateManager:
    """训练状态管理器 - 整合检查点、错误处理和中断管理"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 auto_save_interval: int = 600,  # 自动保存间隔（批次数）
                 max_retries: int = 3):
        """
        初始化训练状态管理器
        
        Args:
            checkpoint_dir: 检查点目录
            auto_save_interval: 自动保存间隔
            max_retries: 最大重试次数
        """
        self.checkpoint_manager = TrainingCheckpoint(checkpoint_dir)
        self.error_handler = ErrorHandler(max_retries=max_retries)
        self.interrupt_handler = GracefulInterruptHandler(self._emergency_save)
        
        self.auto_save_interval = auto_save_interval
        self.last_batch_idx = 0
        
        # 训练状态
        self.training_state = {
            'model': None,
            'optimizer': None,
            'epoch': 0,
            'batch_idx': 0,
            'best_accuracy': 0.0
        }
    
    def initialize_training(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        初始化训练，尝试从检查点恢复
        
        Args:
            model: 模型
            optimizer: 优化器
            
        Returns:
            训练状态字典
        """
        self.training_state['model'] = model
        self.training_state['optimizer'] = optimizer
        
        # 尝试加载检查点
        checkpoint = self.checkpoint_manager.load_checkpoint(model, optimizer)
        
        if checkpoint:
            self.training_state.update({
                'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'best_accuracy': checkpoint['best_accuracy']
            })
            logger.info(f"从检查点恢复训练: epoch={checkpoint['epoch']}, batch={checkpoint['batch_idx']}")
        else:
            logger.info("开始新的训练")
        
        return self.training_state
    
    def save_if_needed(self, current_batch_idx: int, loss: float, force: bool = False):
        """根据需要保存检查点"""
        self.last_batch_idx = current_batch_idx
        
        if force or (current_batch_idx % self.auto_save_interval == 0):
            self.checkpoint_manager.save_checkpoint(
                model=self.training_state['model'],
                optimizer=self.training_state['optimizer'],
                epoch=self.training_state['epoch'],
                batch_idx=current_batch_idx,
                loss=loss,
                best_accuracy=self.training_state['best_accuracy']
            )
    
    def _emergency_save(self):
        """紧急保存（中断时调用）"""
        if self.training_state['model'] is not None:
            self.checkpoint_manager.save_checkpoint(
                model=self.training_state['model'],
                optimizer=self.training_state['optimizer'],
                epoch=self.training_state['epoch'],
                batch_idx=self.last_batch_idx,
                loss=0.0,  # 中断时可能没有当前损失
                best_accuracy=self.training_state['best_accuracy'],
                interrupted=True
            )
    
    def should_stop_training(self) -> bool:
        """检查是否应该停止训练"""
        return self.interrupt_handler.check_interrupt()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'training_state': {k: v for k, v in self.training_state.items() 
                             if k not in ['model', 'optimizer']},
            'error_summary': self.error_handler.get_error_summary(),
            'interrupted': self.interrupt_handler.interrupted
        } 