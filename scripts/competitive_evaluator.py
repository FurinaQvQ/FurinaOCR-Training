"""
竞争式并行模型评估器
所有模型同时竞争，出错即淘汰，最终选出最鲁棒的模型
"""

import os
import torch
import torch.nn as nn
import time
import json
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from PIL import ImageFont
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

import sys
sys.path.append(str(Path(__file__).parent.parent))

from mona.config import config, get_config_manager
from mona.nn.model2 import Model2
from mona.nn import predict as predict_net
from mona.text import get_lexicon
from mona.datagen.datagen import DataGen
from mona.utils import logger
import torchvision.transforms as transforms


class ModelCompetitor:
    """单个模型的竞争者"""
    
    def __init__(self, model_path: Path, model_id: int, device: str):
        self.model_path = model_path
        self.model_name = model_path.stem
        self.model_id = model_id
        self.device = device
        
        # 统计信息
        self.correct_count = 0
        self.total_count = 0
        self.error_count = 0
        self.inference_times = []
        self.is_active = True
        self.elimination_reason = None
        self.elimination_time = None
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """加载模型"""
        try:
            lexicon = get_lexicon()
            net = Model2(lexicon.lexicon_size(), 1).to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            net.load_state_dict(state_dict)
            net.eval()
            return net
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {self.model_name}: {e}")
            self.is_active = False
            self.elimination_reason = f"加载失败: {e}"
            return None
    
    def predict(self, tensor: torch.Tensor, lexicon) -> Tuple[str, float]:
        """模型预测"""
        if not self.is_active or self.model is None:
            return "", 0.0
        
        try:
            start_time = time.time()
            with torch.no_grad():
                predicted = predict_net(self.model, tensor, lexicon)
            inference_time = time.time() - start_time
            
            self.inference_times.append(inference_time)
            pred_text = predicted[0] if predicted else ""
            return pred_text, inference_time
            
        except Exception as e:
            self.eliminate(f"预测错误: {e}")
            return "", 0.0
    
    def update_stats(self, is_correct: bool):
        """更新统计信息"""
        if not self.is_active:
            return
            
        self.total_count += 1
        if is_correct:
            self.correct_count += 1
        else:
            self.error_count += 1
    
    def eliminate(self, reason: str):
        """淘汰模型"""
        self.is_active = False
        self.elimination_reason = reason
        self.elimination_time = datetime.now()
        
        # 释放GPU内存
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
    
    @property
    def accuracy(self) -> float:
        """当前准确率"""
        return self.correct_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def avg_inference_time(self) -> float:
        """平均推理时间"""
        return np.mean(self.inference_times) if self.inference_times else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "is_active": self.is_active,
            "correct_count": self.correct_count,
            "total_count": self.total_count,
            "error_count": self.error_count,
            "accuracy": self.accuracy,
            "avg_inference_time_ms": self.avg_inference_time * 1000,
            "elimination_reason": self.elimination_reason,
            "elimination_time": self.elimination_time.isoformat() if self.elimination_time else None
        }


class CompetitiveEvaluator:
    """竞争式并行模型评估器"""
    
    def __init__(self, models_dir: str = "models", max_workers: int = None):
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lexicon = get_lexicon()
        self.config_manager = get_config_manager()
        
        # 并行配置
        self.max_workers = max_workers or min(torch.cuda.device_count() if torch.cuda.is_available() else 1, 4)
        
        # 初始化数据生成器
        fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
        self.datagen = DataGen(config, fonts, self.lexicon)
        
        # 竞争者列表
        self.competitors: List[ModelCompetitor] = []
        self.data_queue = queue.Queue(maxsize=1000)  # 数据队列
        self.stop_generation = threading.Event()
        
        # 统计信息
        self.start_time = None
        self.total_samples_generated = 0
        self.evaluation_log = []
        
        logger.info(f"🏁 竞争式评估器初始化完成")
        logger.info(f"🎮 设备: {self.device}")
        logger.info(f"🔥 最大并行数: {self.max_workers}")
    
    def discover_and_load_models(self) -> List[ModelCompetitor]:
        """发现并加载所有模型"""
        if not self.models_dir.exists():
            logger.error(f"❌ 模型目录不存在: {self.models_dir}")
            return []
        
        model_files = list(self.models_dir.glob("*.pt"))
        
        # 过滤掉临时文件
        filtered_models = []
        for model_file in model_files:
            if not any(skip in model_file.name.lower() for skip in 
                      ["training", "checkpoint", "temp", "tmp"]):
                filtered_models.append(model_file)
        
        logger.info(f"📂 发现模型文件: {len(filtered_models)} 个")
        
        # 创建竞争者
        competitors = []
        for i, model_path in enumerate(filtered_models):
            competitor = ModelCompetitor(model_path, i, self.device)
            if competitor.is_active:
                competitors.append(competitor)
                size_mb = model_path.stat().st_size / 1024 / 1024
                logger.info(f"   🎯 加载成功: {competitor.model_name} ({size_mb:.1f}MB)")
            else:
                logger.error(f"   ❌ 加载失败: {model_path.name}")
        
        logger.info(f"🏁 参赛模型数: {len(competitors)}")
        return competitors
    
    def data_generator_worker(self):
        """数据生成器工作线程"""
        batch_size = 50  # 每批生成50个样本
        
        while not self.stop_generation.is_set():
            try:
                # 生成一批数据
                batch_data = []
                for _ in range(batch_size):
                    if self.stop_generation.is_set():
                        break
                    
                    im, text = self.datagen.generate_image()
                    tensor = transforms.ToTensor()(im).unsqueeze(0).to(self.device)
                    batch_data.append((tensor, text))
                    self.total_samples_generated += 1
                
                # 将数据放入队列
                if batch_data:
                    self.data_queue.put(batch_data, timeout=1)
                    
            except queue.Full:
                continue
            except Exception as e:
                logger.error(f"❌ 数据生成错误: {e}")
                time.sleep(0.1)
    
    def evaluate_competitor_batch(self, competitor: ModelCompetitor, data_batch: List[Tuple[torch.Tensor, str]]) -> bool:
        """评估单个竞争者的一批数据"""
        if not competitor.is_active:
            return False
        
        for tensor, true_text in data_batch:
            if not competitor.is_active:
                break
            
            try:
                # 预测
                pred_text, inference_time = competitor.predict(tensor, self.lexicon)
                
                # 检查结果
                is_correct = (pred_text == true_text)
                competitor.update_stats(is_correct)
                
                # 如果预测错误，淘汰该模型
                if not is_correct:
                    competitor.eliminate(f"预测错误: 真实='{true_text}', 预测='{pred_text}'")
                    return False
                    
            except Exception as e:
                competitor.eliminate(f"评估异常: {e}")
                return False
        
        return True
    
    def run_competitive_evaluation(self, min_survival_time: int = 60, 
                                 max_evaluation_time: int = 1800) -> Dict[str, Any]:
        """运行竞争式评估"""
        logger.info("🚀 开始竞争式模型评估...")
        logger.info("🏁 规则: 预测错误即淘汰，最后存活的模型获胜")
        logger.info("=" * 60)
        
        # 加载所有模型
        self.competitors = self.discover_and_load_models()
        if not self.competitors:
            logger.error("❌ 没有可用的竞争模型")
            return {}
        
        self.start_time = time.time()
        
        # 启动数据生成线程
        logger.info("🌊 启动在线数据生成...")
        data_thread = threading.Thread(target=self.data_generator_worker, daemon=True)
        data_thread.start()
        
        # 显示初始状态
        logger.info(f"🎮 参赛选手:")
        for competitor in self.competitors:
            logger.info(f"   🎯 {competitor.model_name}")
        
        logger.info(f"\n⏱️ 开始无限评估...")
        
        try:
            # 主评估循环
            while True:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # 检查时间限制
                if elapsed_time > max_evaluation_time:
                    logger.info(f"⏰ 达到最大评估时间 {max_evaluation_time}s，停止评估")
                    break
                
                # 获取活跃的竞争者
                active_competitors = [c for c in self.competitors if c.is_active]
                
                if len(active_competitors) == 0:
                    logger.info("💥 所有模型都被淘汰了！")
                    break
                elif len(active_competitors) == 1 and elapsed_time >= min_survival_time:
                    logger.info(f"🏆 找到最终获胜者！")
                    break
                
                # 尝试获取数据批次
                try:
                    data_batch = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # 并行评估活跃的竞争者
                with ThreadPoolExecutor(max_workers=min(len(active_competitors), self.max_workers)) as executor:
                    future_to_competitor = {
                        executor.submit(self.evaluate_competitor_batch, competitor, data_batch): competitor
                        for competitor in active_competitors
                    }
                    
                    # 处理结果
                    eliminated_this_round = []
                    for future in as_completed(future_to_competitor):
                        competitor = future_to_competitor[future]
                        try:
                            success = future.result()
                            if not success and competitor.is_active:
                                eliminated_this_round.append(competitor)
                        except Exception as e:
                            competitor.eliminate(f"执行异常: {e}")
                            eliminated_this_round.append(competitor)
                
                # 报告淘汰情况
                if eliminated_this_round:
                    for competitor in eliminated_this_round:
                        logger.info(f"💥 淘汰: {competitor.model_name} - {competitor.elimination_reason}")
                        logger.info(f"     存活样本: {competitor.correct_count:,}, 准确率: {competitor.accuracy:.6f}")
                
                # 定期状态报告
                if int(elapsed_time) % 10 == 0 and len(active_competitors) > 1:
                    self._print_status_report(elapsed_time, active_competitors)
        
        except KeyboardInterrupt:
            logger.info("⏹️ 用户中断评估")
        finally:
            # 停止数据生成
            self.stop_generation.set()
        
        # 生成最终报告
        return self._generate_final_report()
    
    def _print_status_report(self, elapsed_time: float, active_competitors: List[ModelCompetitor]):
        """打印状态报告"""
        logger.info(f"\n📊 第 {int(elapsed_time)}s 状态报告:")
        logger.info(f"🏁 剩余选手: {len(active_competitors)}")
        
        # 按正确数量排序
        sorted_competitors = sorted(active_competitors, key=lambda x: x.correct_count, reverse=True)
        
        for i, competitor in enumerate(sorted_competitors[:5], 1):  # 只显示前5名
            logger.info(f"   {i}. {competitor.model_name}: "
                       f"{competitor.correct_count:,}样本, "
                       f"准确率{competitor.accuracy:.6f}, "
                       f"速度{competitor.avg_inference_time*1000:.1f}ms")
        
        logger.info(f"📈 总样本已生成: {self.total_samples_generated:,}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终评估报告"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        # 获取所有竞争者的统计信息
        all_stats = [competitor.get_stats() for competitor in self.competitors]
        
        # 找到获胜者（活跃且正确数量最多的）
        active_competitors = [c for c in self.competitors if c.is_active]
        winner = None
        
        if active_competitors:
            winner = max(active_competitors, key=lambda x: x.correct_count)
        
        # 生成报告
        report = {
            "evaluation_summary": {
                "total_competitors": len(self.competitors),
                "final_survivors": len(active_competitors),
                "winner": winner.model_name if winner else None,
                "total_evaluation_time": total_time,
                "total_samples_generated": self.total_samples_generated,
                "evaluation_mode": "competitive_elimination",
                "evaluation_time": datetime.now().isoformat()
            },
            "winner_details": winner.get_stats() if winner else None,
            "all_competitors": all_stats,
            "elimination_timeline": [
                {
                    "model_name": c.model_name,
                    "elimination_time": c.elimination_time.isoformat() if c.elimination_time else None,
                    "elimination_reason": c.elimination_reason,
                    "survival_samples": c.correct_count,
                    "accuracy": c.accuracy
                }
                for c in self.competitors if not c.is_active
            ]
        }
        
        # 打印最终结果
        logger.info("\n" + "🏆" * 30)
        logger.info("🎉 竞争式评估完成!")
        logger.info("🏆" * 30)
        
        if winner:
            logger.info(f"\n🥇 最终获胜者: {winner.model_name}")
            logger.info(f"   🎯 存活样本数: {winner.correct_count:,}")
            logger.info(f"   📊 准确率: {winner.accuracy:.6f}")
            logger.info(f"   ⚡ 平均速度: {winner.avg_inference_time*1000:.2f}ms")
            logger.info(f"   ⏱️ 总评估时间: {total_time:.1f}s")
        else:
            logger.info("💥 没有模型存活到最后!")
        
        logger.info(f"\n📈 评估统计:")
        logger.info(f"   🎮 参赛模型: {len(self.competitors)}")
        logger.info(f"   🏁 最终存活: {len(active_competitors)}")
        logger.info(f"   📊 总样本数: {self.total_samples_generated:,}")
        logger.info(f"   ⏱️ 总用时: {total_time:.1f}s")
        
        # 显示淘汰顺序
        eliminated = [c for c in self.competitors if not c.is_active]
        if eliminated:
            logger.info(f"\n💥 淘汰顺序 (从早到晚):")
            eliminated_sorted = sorted(eliminated, 
                                     key=lambda x: x.elimination_time if x.elimination_time else datetime.min)
            
            for i, competitor in enumerate(eliminated_sorted, 1):
                elapsed = (competitor.elimination_time - datetime.fromtimestamp(self.start_time)).total_seconds() if competitor.elimination_time else 0
                logger.info(f"   {i}. {competitor.model_name} "
                           f"(第{elapsed:.1f}s淘汰, 存活{competitor.correct_count:,}样本)")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "logs") -> str:
        """保存评估报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"competitive_evaluation_{timestamp}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 详细报告已保存: {report_file}")
        return str(report_file)
    
    def copy_winner_model(self, report: Dict[str, Any]) -> str:
        """复制获胜模型到models/models文件夹，保持原文件名，并自动导出ONNX和index_2_word.json"""
        winner_name = report["evaluation_summary"].get("winner")
        if not winner_name:
            logger.error("❌ 没有获胜者，无法复制模型")
            return None
        
        # 找到获胜者的模型文件
        winner_competitor = None
        for competitor in self.competitors:
            if competitor.model_name == winner_name:
                winner_competitor = competitor
                break
        
        if not winner_competitor:
            logger.error(f"❌ 找不到获胜者模型: {winner_name}")
            return None
        
        # 创建目标目录 models/models
        import shutil
        from pathlib import Path
        target_dir = Path("models") / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 保持原始文件名
        source_path = winner_competitor.model_path
        target_path = target_dir / source_path.name
        
        # 复制模型
        shutil.copy2(source_path, target_path)
        logger.info(f"🏆 冠军模型已复制: {target_path}")
        logger.info(f"📂 源文件: {source_path}")
        logger.info(f"📊 模型性能: 存活{winner_competitor.correct_count:,}样本, "
                    f"准确率{winner_competitor.accuracy:.6f}")
        
        # ========== 自动导出ONNX ==========
        try:
            from mona.nn.model2 import Model2
            from mona.text import get_lexicon
            import torch
            lexicon = get_lexicon()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net = Model2(lexicon.lexicon_size(), 1).to(device)
            net.load_state_dict(torch.load(target_path, map_location=device, weights_only=True))
            net.eval()
            dummy_input = torch.randn(1, 1, 32, 384).to(device)
            onnx_path = target_dir / "model_training.onnx"
            torch.onnx.export(
                net,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"✅ 已导出ONNX模型: {onnx_path}")
        except Exception as e:
            logger.error(f"❌ 导出ONNX失败: {e}")
        
        # ========== 导出index_2_word.json ==========
        try:
            index2word = get_lexicon().index_to_word
            # 转为str-key的dict
            index2word_str = {str(k): v for k, v in index2word.items()}
            json_path = target_dir / "index_2_word.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(index2word_str, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 已导出index_2_word.json: {json_path}")
        except Exception as e:
            logger.error(f"❌ 导出index_2_word.json失败: {e}")
        
        return str(target_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="竞争式并行模型评估器")
    parser.add_argument("--models-dir", default="models", help="模型文件目录")
    parser.add_argument("--min-survival-time", type=int, default=60, help="最小存活时间(秒)")
    parser.add_argument("--max-evaluation-time", type=int, default=1800, help="最大评估时间(秒)")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并行工作数")
    parser.add_argument("--copy-winner", action="store_true", help="复制获胜模型到models/models文件夹")
    parser.add_argument("--output-dir", default="logs", help="报告输出目录")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = CompetitiveEvaluator(args.models_dir, args.max_workers)
    
    # 运行竞争式评估
    report = evaluator.run_competitive_evaluation(
        min_survival_time=args.min_survival_time,
        max_evaluation_time=args.max_evaluation_time
    )
    
    if report:
        # 保存报告
        evaluator.save_report(report, args.output_dir)
        
        # 复制获胜模型
        if args.copy_winner:
            evaluator.copy_winner_model(report)
    
    return 0


if __name__ == "__main__":
    exit(main()) 