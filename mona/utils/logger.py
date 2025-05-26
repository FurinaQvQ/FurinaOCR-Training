"""
统一的日志系统
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """训练专用日志器"""
    
    def __init__(self, name: str = "genshin_ocr", log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def log_training_start(self, config: dict):
        """记录训练开始信息"""
        self.info("=" * 50)
        self.info("开始训练原神OCR模型")
        self.info("=" * 50)
        self.info(f"训练配置: {config}")
    
    def log_epoch(self, epoch: int, loss: float, accuracy: float, lr: float):
        """记录epoch信息"""
        self.info(f"Epoch {epoch:3d} | Loss: {loss:.6f} | Acc: {accuracy:.4f} | LR: {lr:.6f}")
    
    def log_validation(self, accuracy: float, total_samples: int):
        """记录验证信息"""
        self.info(f"验证结果 - 准确率: {accuracy:.4f} | 样本数: {total_samples}")
    
    def log_model_save(self, path: str, accuracy: float):
        """记录模型保存信息"""
        self.info(f"模型已保存: {path} | 准确率: {accuracy:.4f}")


# 全局日志器实例
logger = TrainingLogger() 