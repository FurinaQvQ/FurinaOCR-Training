"""
数据质量验证器
验证训练数据的完整性、格式正确性和质量指标
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from collections import Counter
import cv2

from mona.utils import logger


class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self, output_dir: str = "validation_reports"):
        """
        初始化数据质量验证器
        
        Args:
            output_dir: 验证报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.validation_results = {
            'image_stats': {},
            'text_stats': {},
            'quality_issues': [],
            'recommendations': []
        }
    
    def validate_image_batch(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        验证图像批次的质量
        
        Args:
            images: 图像张量 (B, C, H, W)
            
        Returns:
            验证结果字典
        """
        batch_size, channels, height, width = images.shape
        
        # 转换为numpy数组进行分析
        images_np = images.detach().cpu().numpy()
        
        # 基本统计信息
        stats = {
            'batch_size': batch_size,
            'shape': (channels, height, width),
            'dtype': str(images.dtype),
            'device': str(images.device),
            'mean': float(np.mean(images_np)),
            'std': float(np.std(images_np)),
            'min': float(np.min(images_np)),
            'max': float(np.max(images_np)),
        }
        
        # 检查异常值
        issues = []
        
        # 检查值域
        if stats['min'] < -1.0 or stats['max'] > 1.0:
            issues.append("图像像素值超出预期范围 [-1, 1]")
        
        # 检查是否有NaN或Inf
        if np.isnan(images_np).any():
            issues.append("检测到NaN值")
        
        if np.isinf(images_np).any():
            issues.append("检测到Inf值")
        
        # 检查对比度
        contrast_scores = []
        for i in range(batch_size):
            img = images_np[i, 0]  # 假设是灰度图
            contrast = np.std(img)
            contrast_scores.append(contrast)
        
        avg_contrast = np.mean(contrast_scores)
        if avg_contrast < 0.1:
            issues.append(f"平均对比度过低: {avg_contrast:.3f}")
        
        # 检查亮度分布
        brightness_scores = [np.mean(images_np[i, 0]) for i in range(batch_size)]
        brightness_std = np.std(brightness_scores)
        
        if brightness_std > 0.5:
            issues.append(f"亮度分布不均匀: std={brightness_std:.3f}")
        
        stats['contrast_scores'] = contrast_scores
        stats['brightness_scores'] = brightness_scores
        stats['quality_issues'] = issues
        
        return stats
    
    def validate_text_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        验证文本批次的质量
        
        Args:
            texts: 文本字符串列表
            
        Returns:
            验证结果字典
        """
        if not texts:
            return {'error': '文本列表为空'}
        
        # 基本统计
        text_lengths = [len(text) for text in texts]
        
        stats = {
            'batch_size': len(texts),
            'total_chars': sum(text_lengths),
            'avg_length': np.mean(text_lengths),
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'length_std': np.std(text_lengths),
        }
        
        # 字符频率统计
        all_chars = ''.join(texts)
        char_counter = Counter(all_chars)
        
        stats['unique_chars'] = len(char_counter)
        stats['char_frequency'] = dict(char_counter.most_common(20))
        
        # 检查异常
        issues = []
        
        # 检查空字符串
        empty_count = sum(1 for text in texts if len(text) == 0)
        if empty_count > 0:
            issues.append(f"发现 {empty_count} 个空字符串")
        
        # 检查长度异常
        if stats['length_std'] > stats['avg_length'] * 0.5:
            issues.append("文本长度差异过大")
        
        # 检查特殊字符
        unusual_chars = set()
        for char in char_counter:
            if ord(char) < 32 or ord(char) > 126:  # 非可打印ASCII字符
                if char not in ['\n', '\t', ' ']:  # 排除常见空白字符
                    unusual_chars.add(char)
        
        if unusual_chars:
            issues.append(f"发现异常字符: {list(unusual_chars)[:10]}")  # 只显示前10个
        
        stats['quality_issues'] = issues
        
        return stats
    
    def validate_image_text_pairs(self, 
                                 images: torch.Tensor, 
                                 texts: List[str]) -> Dict[str, Any]:
        """
        验证图像-文本对的一致性
        
        Args:
            images: 图像张量
            texts: 文本列表
            
        Returns:
            验证结果字典
        """
        if len(images) != len(texts):
            return {'error': f'图像数量({len(images)})与文本数量({len(texts)})不匹配'}
        
        batch_size = len(images)
        
        # 验证图像和文本的复杂度相关性
        image_complexities = []
        text_complexities = []
        
        images_np = images.detach().cpu().numpy()
        
        for i in range(batch_size):
            # 图像复杂度（使用边缘检测）
            img = images_np[i, 0]  # 灰度图
            img_uint8 = ((img + 1) * 127.5).astype(np.uint8)  # 转换到0-255
            edges = cv2.Canny(img_uint8, 50, 150)
            img_complexity = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            image_complexities.append(img_complexity)
            
            # 文本复杂度（字符种类数）
            text_complexity = len(set(texts[i])) / max(len(texts[i]), 1)
            text_complexities.append(text_complexity)
        
        # 计算相关性
        if len(image_complexities) > 1:
            correlation = np.corrcoef(image_complexities, text_complexities)[0, 1]
        else:
            correlation = 0.0
        
        stats = {
            'batch_size': batch_size,
            'image_complexities': image_complexities,
            'text_complexities': text_complexities,
            'complexity_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'avg_image_complexity': np.mean(image_complexities),
            'avg_text_complexity': np.mean(text_complexities),
        }
        
        # 检查异常
        issues = []
        
        if abs(correlation) < 0.1:
            issues.append("图像复杂度与文本复杂度相关性较低")
        
        # 检查是否有图像过于简单或复杂
        simple_images = sum(1 for x in image_complexities if x < 0.01)
        complex_images = sum(1 for x in image_complexities if x > 0.5)
        
        if simple_images > batch_size * 0.2:
            issues.append(f"过多简单图像: {simple_images}/{batch_size}")
        
        if complex_images > batch_size * 0.1:
            issues.append(f"过多复杂图像: {complex_images}/{batch_size}")
        
        stats['quality_issues'] = issues
        
        return stats
    
    def generate_quality_report(self, 
                              validation_data: List[Tuple[torch.Tensor, List[str]]], 
                              sample_count: int = 100) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Args:
            validation_data: 验证数据列表 [(images, texts), ...]
            sample_count: 采样数量
            
        Returns:
            完整的质量报告
        """
        logger.info(f"开始生成数据质量报告，采样数量: {sample_count}")
        
        all_image_stats = []
        all_text_stats = []
        all_pair_stats = []
        
        sample_indices = np.random.choice(len(validation_data), 
                                        min(sample_count, len(validation_data)), 
                                        replace=False)
        
        for idx in sample_indices:
            images, texts = validation_data[idx]
            
            # 验证图像
            img_stats = self.validate_image_batch(images)
            all_image_stats.append(img_stats)
            
            # 验证文本
            text_stats = self.validate_text_batch(texts)
            all_text_stats.append(text_stats)
            
            # 验证图像-文本对
            pair_stats = self.validate_image_text_pairs(images, texts)
            all_pair_stats.append(pair_stats)
        
        # 汇总统计信息
        report = self._aggregate_validation_results(all_image_stats, all_text_stats, all_pair_stats)
        
        # 保存报告
        report_file = self.output_dir / f"quality_report_{len(sample_indices)}_samples.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"质量报告已保存: {report_file}")
        
        return report
    
    def _aggregate_validation_results(self, 
                                    image_stats: List[Dict], 
                                    text_stats: List[Dict], 
                                    pair_stats: List[Dict]) -> Dict[str, Any]:
        """汇总验证结果"""
        
        # 汇总图像统计
        image_summary = {
            'total_batches': len(image_stats),
            'avg_mean': np.mean([s['mean'] for s in image_stats]),
            'avg_std': np.mean([s['std'] for s in image_stats]),
            'avg_contrast': np.mean([np.mean(s['contrast_scores']) for s in image_stats]),
            'total_issues': sum(len(s['quality_issues']) for s in image_stats),
        }
        
        # 汇总文本统计
        text_summary = {
            'total_batches': len(text_stats),
            'avg_length': np.mean([s['avg_length'] for s in text_stats if 'avg_length' in s]),
            'total_chars': sum(s['total_chars'] for s in text_stats if 'total_chars' in s),
            'total_issues': sum(len(s['quality_issues']) for s in text_stats if 'quality_issues' in s),
        }
        
        # 汇总配对统计
        pair_summary = {
            'total_pairs': len(pair_stats),
            'avg_correlation': np.mean([s['complexity_correlation'] for s in pair_stats]),
            'total_issues': sum(len(s['quality_issues']) for s in pair_stats if 'quality_issues' in s),
        }
        
        # 生成建议
        recommendations = self._generate_recommendations(image_summary, text_summary, pair_summary)
        
        return {
            'timestamp': np.datetime64('now').astype(str),
            'image_summary': image_summary,
            'text_summary': text_summary,
            'pair_summary': pair_summary,
            'recommendations': recommendations,
            'overall_score': self._calculate_overall_score(image_summary, text_summary, pair_summary)
        }
    
    def _generate_recommendations(self, 
                                image_summary: Dict, 
                                text_summary: Dict, 
                                pair_summary: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 图像相关建议
        if image_summary['avg_contrast'] < 0.15:
            recommendations.append("🔧 建议增强图像对比度以提高OCR识别率")
        
        if image_summary['total_issues'] > image_summary['total_batches'] * 0.1:
            recommendations.append("⚠️ 图像质量问题较多，建议检查数据生成流程")
        
        # 文本相关建议
        if text_summary['avg_length'] < 3:
            recommendations.append("📝 文本长度偏短，可能影响模型学习")
        
        if text_summary['avg_length'] > 50:
            recommendations.append("📝 文本长度偏长，考虑是否需要截断")
        
        # 配对相关建议
        if pair_summary['avg_correlation'] < 0.1:
            recommendations.append("🔗 图像-文本复杂度相关性低，检查数据生成逻辑")
        
        if len(recommendations) == 0:
            recommendations.append("✅ 数据质量良好，可以继续训练")
        
        return recommendations
    
    def _calculate_overall_score(self, 
                               image_summary: Dict, 
                               text_summary: Dict, 
                               pair_summary: Dict) -> float:
        """计算整体质量分数 (0-100)"""
        score = 100.0
        
        # 图像质量扣分
        if image_summary['avg_contrast'] < 0.1:
            score -= 20
        elif image_summary['avg_contrast'] < 0.15:
            score -= 10
        
        # 文本质量扣分
        if text_summary['avg_length'] < 2:
            score -= 15
        elif text_summary['avg_length'] > 100:
            score -= 10
        
        # 配对质量扣分
        if pair_summary['avg_correlation'] < 0:
            score -= 15
        elif pair_summary['avg_correlation'] < 0.1:
            score -= 10
        
        # 问题数量扣分
        total_batches = max(image_summary['total_batches'], 1)
        issue_rate = (image_summary['total_issues'] + text_summary['total_issues'] + 
                     pair_summary['total_issues']) / (total_batches * 3)
        
        score -= min(issue_rate * 50, 30)  # 最多扣30分
        
        return max(score, 0.0)
    
    def visualize_quality_metrics(self, report: Dict[str, Any]) -> str:
        """
        可视化质量指标
        
        Args:
            report: 质量报告
            
        Returns:
            可视化文件路径
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('数据质量分析报告', fontsize=16)
            
            # 整体分数
            ax1 = axes[0, 0]
            score = report['overall_score']
            colors = ['red' if score < 60 else 'yellow' if score < 80 else 'green']
            ax1.bar(['整体质量分数'], [score], color=colors)
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('分数')
            ax1.set_title(f'整体质量分数: {score:.1f}')
            
            # 问题统计
            ax2 = axes[0, 1]
            issue_counts = [
                report['image_summary']['total_issues'],
                report['text_summary']['total_issues'],
                report['pair_summary']['total_issues']
            ]
            ax2.bar(['图像', '文本', '配对'], issue_counts)
            ax2.set_ylabel('问题数量')
            ax2.set_title('质量问题统计')
            
            # 图像统计
            ax3 = axes[1, 0]
            img_metrics = [
                report['image_summary']['avg_mean'],
                report['image_summary']['avg_std'],
                report['image_summary']['avg_contrast']
            ]
            ax3.bar(['平均值', '标准差', '对比度'], img_metrics)
            ax3.set_ylabel('数值')
            ax3.set_title('图像质量指标')
            
            # 文本统计
            ax4 = axes[1, 1]
            text_metrics = [
                report['text_summary']['avg_length'],
                report['pair_summary']['avg_correlation'] * 20  # 放大显示
            ]
            ax4.bar(['平均长度', '相关性×20'], text_metrics)
            ax4.set_ylabel('数值')
            ax4.set_title('文本质量指标')
            
            plt.tight_layout()
            
            # 保存图片
            viz_file = self.output_dir / "quality_visualization.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"质量可视化已保存: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            logger.error(f"可视化生成失败: {e}")
            return "" 