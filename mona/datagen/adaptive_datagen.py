"""
自适应数据生成器
根据训练阶段和准确率动态调整数据生成策略
"""

import random
import pickle
import os
from typing import List, Tuple, Optional
from PIL import Image

from .datagen import DataGen
from ..utils.logger import logger


class AdaptiveDataGen:
    """
    自适应数据生成器
    - 训练初期：使用在线随机数据生成
    - 高准确率阶段：使用困难样本重复训练
    """
    
    def __init__(self, config, fonts, lexicon):
        self.config = config
        self.base_datagen = DataGen(config, fonts, lexicon)
        self.difficult_samples = []
        self.cache_file = "samples/difficult_samples.pkl"
        self.current_mode = "online"  # "online" 或 "fixed"
        self.accuracy_threshold = 0.95
        
        # 加载已保存的困难样本
        self.load_difficult_samples()
    
    def set_mode(self, mode: str, accuracy: float = None):
        """
        设置数据生成模式
        
        Args:
            mode: "online" 在线生成 / "fixed" 固定困难样本 / "adaptive" 自适应
            accuracy: 当前验证准确率
        """
        if mode == "adaptive" and accuracy is not None:
            if accuracy < self.accuracy_threshold:
                self.current_mode = "online"
                logger.info(f"📊 准确率 {accuracy:.3f} < {self.accuracy_threshold}，使用在线数据生成")
            else:
                self.current_mode = "fixed"
                logger.info(f"🎯 准确率 {accuracy:.3f} ≥ {self.accuracy_threshold}，切换到困难样本训练")
                if len(self.difficult_samples) == 0:
                    self.generate_difficult_samples()
        else:
            self.current_mode = mode
            logger.info(f"🔧 手动设置数据模式: {mode}")
    
    def generate_image(self) -> Tuple[Image.Image, str]:
        """根据当前模式生成图像"""
        if self.current_mode == "online":
            return self.base_datagen.generate_image()
        elif self.current_mode == "fixed":
            if len(self.difficult_samples) == 0:
                logger.warning("⚠️ 困难样本为空，回退到在线生成")
                return self.base_datagen.generate_image()
            return random.choice(self.difficult_samples)
        else:
            return self.base_datagen.generate_image()
    
    def generate_difficult_samples(self, count: int = 5000):
        """
        生成困难样本集
        重点生成容易识别错误的样本类型
        """
        logger.info(f"🎯 开始生成 {count} 个困难样本...")
        self.difficult_samples = []
        
        # 困难样本类型
        difficult_patterns = [
            # 容易混淆的字符组合
            ["0", "O", "o"],  # 数字0和字母O
            ["1", "l", "I"],  # 数字1和字母l、I
            ["6", "9"],       # 6和9
            [",", ".", "。"], # 标点符号
            ["雷", "电"],     # 相似汉字
            ["%", "℅"],      # 百分号
        ]
        
        for i in range(count):
            # 30% 概率生成困难模式
            if random.random() < 0.3:
                # 选择困难字符
                pattern = random.choice(difficult_patterns)
                char = random.choice(pattern)
                
                # 手动构造包含该字符的文本
                if random.random() < 0.5:
                    text = char  # 单字符
                else:
                    # 随机组合
                    base_text = self.base_datagen.lexicon.generate_text()
                    pos = random.randint(0, len(base_text))
                    text = base_text[:pos] + char + base_text[pos:]
                
                # 生成对应图像
                img = self._generate_specific_text_image(text)
            else:
                # 70% 使用正常随机生成
                img, text = self.base_datagen.generate_image()
            
            self.difficult_samples.append((img, text))
            
            if (i + 1) % 1000 == 0:
                logger.info(f"📊 已生成困难样本: {i + 1}/{count}")
        
        # 保存到文件
        self.save_difficult_samples()
        logger.info(f"✅ 困难样本生成完成，共 {len(self.difficult_samples)} 个")
    
    def _generate_specific_text_image(self, text: str) -> Image.Image:
        """为特定文本生成图像"""
        # 复用原有的图像生成逻辑，但使用指定文本
        from .datagen import rand_color_1, rand_color_2
        from .pre_process import pre_process
        from PIL import ImageDraw, ImageFont
        
        color1 = rand_color_1()
        color2 = rand_color_2()
        
        img = Image.new("RGB", (1200, 120), color1)
        draw = ImageDraw.Draw(img)
        x = random.randint(0, 20)
        y = random.randint(0, 5)
        
        font = random.choice(self.base_datagen.fonts)
        draw.text((x, y), text, color2, font=font)
        
        thr = random.uniform(0.5, 0.6)
        img = pre_process(img, thr)
        return img
    
    def save_difficult_samples(self):
        """保存困难样本到文件"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.difficult_samples, f)
        logger.info(f"💾 困难样本已保存: {self.cache_file}")
    
    def load_difficult_samples(self):
        """从文件加载困难样本"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.difficult_samples = pickle.load(f)
                logger.info(f"📂 已加载困难样本: {len(self.difficult_samples)} 个")
            except Exception as e:
                logger.warning(f"⚠️ 加载困难样本失败: {e}")
                self.difficult_samples = []
        else:
            logger.info("📝 困难样本文件不存在，将在需要时生成")
    
    def get_mode_info(self) -> dict:
        """获取当前模式信息"""
        return {
            "current_mode": self.current_mode,
            "difficult_samples_count": len(self.difficult_samples),
            "accuracy_threshold": self.accuracy_threshold,
            "cache_file_exists": os.path.exists(self.cache_file)
        }
    
    def clear_difficult_samples(self):
        """清除困难样本缓存"""
        self.difficult_samples = []
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("🗑️ 困难样本缓存已清除")
    
    def analyze_training_progress(self, recent_accuracies: List[float]) -> str:
        """
        分析训练进度，给出数据策略建议
        
        Args:
            recent_accuracies: 最近几次验证的准确率列表
            
        Returns:
            建议信息
        """
        if len(recent_accuracies) < 3:
            return "数据不足，继续训练"
        
        current_acc = recent_accuracies[-1]
        trend = recent_accuracies[-1] - recent_accuracies[-3]
        
        if current_acc < 0.9:
            return "准确率较低，建议继续在线数据训练增加多样性"
        elif current_acc < 0.95:
            if trend < 0.01:  # 进步缓慢
                return "进步缓慢，可考虑生成困难样本进行针对性训练"
            else:
                return "稳步提升中，继续当前策略"
        elif current_acc < 1.0:
            return "接近目标，建议切换到困难样本模式进行精细调优"
        else:
            return "已达到完美准确率！"


class ProgressiveDataSet:
    """渐进式数据集，根据训练进度自动调整"""
    
    def __init__(self, size: int, adaptive_datagen: AdaptiveDataGen):
        self.size = size
        self.adaptive_datagen = adaptive_datagen
        self.current_accuracy = 0.0
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index: int):
        from torchvision import transforms
        im, text = self.adaptive_datagen.generate_image()
        tensor = transforms.ToTensor()(im)
        return tensor, text
    
    def update_accuracy(self, accuracy: float):
        """更新当前准确率，自动调整数据生成策略"""
        self.current_accuracy = accuracy
        self.adaptive_datagen.set_mode("adaptive", accuracy) 