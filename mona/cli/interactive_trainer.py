"""
交互式训练控制台
提供实时监控、动态参数调整、训练控制等用户友好的功能
"""

import time
import threading
import queue
import curses
import signal
import sys
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import json

from mona.utils import logger


class InteractiveTrainer:
    """交互式训练控制台"""
    
    def __init__(self, training_callback: Callable, config: Dict[str, Any]):
        """
        初始化交互式训练器
        
        Args:
            training_callback: 训练回调函数
            config: 配置字典
        """
        self.training_callback = training_callback
        self.config = config
        
        # 训练状态
        self.training_state = {
            'running': False,
            'paused': False,
            'current_epoch': 0,
            'current_batch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'lr': 0.0,
            'throughput': 0.0,
            'elapsed_time': 0,
            'eta': 0,
        }
        
        # 控制队列
        self.control_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # 界面控制
        self.screen = None
        self.running = True
        
        # 训练统计
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'timestamps': [],
        }
        
        # 可调整参数
        self.adjustable_params = {
            'learning_rate': {'min': 1e-6, 'max': 1e-1, 'current': config.get('learning_rate', 1.0)},
            'batch_size': {'min': 16, 'max': 512, 'current': config.get('batch_size', 128)},
            'save_interval': {'min': 100, 'max': 2000, 'current': config.get('save_per', 600)},
        }
    
    def start_interactive_training(self):
        """启动交互式训练"""
        try:
            # 初始化curses界面
            self.screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.screen.keypad(True)
            self.screen.nodelay(True)
            
            # 启动训练线程
            training_thread = threading.Thread(target=self._training_worker, daemon=True)
            training_thread.start()
            
            # 启动界面更新循环
            self._ui_loop()
            
        except KeyboardInterrupt:
            logger.info("用户中断训练")
        finally:
            self._cleanup()
    
    def _training_worker(self):
        """训练工作线程"""
        try:
            self.training_state['running'] = True
            start_time = time.time()
            
            # 调用训练回调
            self.training_callback(
                progress_callback=self._update_training_status,
                control_queue=self.control_queue
            )
            
        except Exception as e:
            logger.error(f"训练线程错误: {e}")
            self.training_state['running'] = False
    
    def _update_training_status(self, **kwargs):
        """更新训练状态"""
        self.training_state.update(kwargs)
        
        # 更新历史记录
        if 'loss' in kwargs:
            self.training_history['losses'].append(kwargs['loss'])
            self.training_history['timestamps'].append(time.time())
        
        if 'accuracy' in kwargs:
            self.training_history['accuracies'].append(kwargs['accuracy'])
    
    def _ui_loop(self):
        """界面更新循环"""
        while self.running:
            try:
                # 处理用户输入
                self._handle_user_input()
                
                # 更新显示
                self._update_display()
                
                # 控制更新频率
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"界面更新错误: {e}")
                break
    
    def _handle_user_input(self):
        """处理用户输入"""
        try:
            key = self.screen.getch()
            
            if key == ord('q') or key == 27:  # q键或ESC
                self.running = False
                self.control_queue.put({'action': 'stop'})
            
            elif key == ord('p'):  # 暂停/继续
                if self.training_state['paused']:
                    self.control_queue.put({'action': 'resume'})
                    self.training_state['paused'] = False
                else:
                    self.control_queue.put({'action': 'pause'})
                    self.training_state['paused'] = True
            
            elif key == ord('s'):  # 保存检查点
                self.control_queue.put({'action': 'save'})
            
            elif key == ord('r'):  # 重置统计
                self.training_history = {'losses': [], 'accuracies': [], 'timestamps': []}
            
            elif key == ord('h'):  # 显示帮助
                self._show_help()
            
            # 参数调整
            elif key == ord('1'):  # 增加学习率
                self._adjust_parameter('learning_rate', 1.2)
            elif key == ord('2'):  # 减少学习率
                self._adjust_parameter('learning_rate', 0.8)
            
            elif key == ord('3'):  # 增加批次大小
                self._adjust_parameter('batch_size', 1.2)
            elif key == ord('4'):  # 减少批次大小
                self._adjust_parameter('batch_size', 0.8)
                
        except curses.error:
            pass  # 无输入时的正常情况
    
    def _adjust_parameter(self, param_name: str, factor: float):
        """调整参数"""
        if param_name in self.adjustable_params:
            param_info = self.adjustable_params[param_name]
            new_value = param_info['current'] * factor
            
            # 限制范围
            new_value = max(param_info['min'], min(param_info['max'], new_value))
            
            if param_name == 'batch_size':
                new_value = int(new_value)
            
            param_info['current'] = new_value
            
            # 发送参数更新命令
            self.control_queue.put({
                'action': 'update_param',
                'param': param_name,
                'value': new_value
            })
    
    def _update_display(self):
        """更新显示内容"""
        try:
            self.screen.clear()
            height, width = self.screen.getmaxyx()
            
            # 标题
            title = "🚀 原神OCR交互式训练控制台"
            self.screen.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
            
            # 训练状态区域
            self._draw_training_status(2, 2)
            
            # 性能指标区域
            self._draw_performance_metrics(2, width // 2)
            
            # 参数控制区域
            self._draw_parameter_controls(height // 2, 2)
            
            # 历史图表区域（简化版）
            self._draw_history_chart(height // 2, width // 2)
            
            # 控制说明
            self._draw_controls_help(height - 5, 2)
            
            self.screen.refresh()
            
        except curses.error as e:
            pass  # 终端大小问题时忽略
    
    def _draw_training_status(self, start_row: int, start_col: int):
        """绘制训练状态"""
        status_lines = [
            f"📊 训练状态: {'运行中' if self.training_state['running'] else '已停止'}",
            f"⏸️  暂停状态: {'是' if self.training_state['paused'] else '否'}",
            f"🔄 当前轮次: {self.training_state['current_epoch']}",
            f"📦 当前批次: {self.training_state['current_batch']}",
            f"📉 当前损失: {self.training_state['loss']:.6f}",
            f"🎯 当前准确率: {self.training_state['accuracy']:.4f}",
            f"⚡ 吞吐量: {self.training_state['throughput']:.1f} samples/s",
        ]
        
        for i, line in enumerate(status_lines):
            try:
                self.screen.addstr(start_row + i, start_col, line)
            except curses.error:
                break
    
    def _draw_performance_metrics(self, start_row: int, start_col: int):
        """绘制性能指标"""
        # 计算统计信息
        recent_losses = self.training_history['losses'][-20:] if self.training_history['losses'] else []
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
        
        elapsed = time.time() - (self.training_history['timestamps'][0] if self.training_history['timestamps'] else time.time())
        
        metrics_lines = [
            f"📈 平均损失(20批): {avg_loss:.6f}",
            f"⏱️  训练时长: {self._format_duration(elapsed)}",
            f"🔥 历史批次数: {len(self.training_history['losses'])}",
            f"📊 最佳准确率: {max(self.training_history['accuracies'], default=0):.4f}",
            f"💾 学习率: {self.training_state['lr']:.2e}",
        ]
        
        for i, line in enumerate(metrics_lines):
            try:
                self.screen.addstr(start_row + i, start_col, line)
            except curses.error:
                break
    
    def _draw_parameter_controls(self, start_row: int, start_col: int):
        """绘制参数控制"""
        control_lines = [
            "🎛️  参数控制:",
            f"   学习率: {self.adjustable_params['learning_rate']['current']:.2e} (1/2键调整)",
            f"   批次大小: {self.adjustable_params['batch_size']['current']} (3/4键调整)",
            f"   保存间隔: {self.adjustable_params['save_interval']['current']} 批次",
        ]
        
        for i, line in enumerate(control_lines):
            try:
                self.screen.addstr(start_row + i, start_col, line)
            except curses.error:
                break
    
    def _draw_history_chart(self, start_row: int, start_col: int):
        """绘制简化历史图表"""
        if not self.training_history['losses']:
            return
        
        chart_lines = [
            "📈 损失趋势 (最近20批):",
        ]
        
        # 简化的ASCII图表
        recent_losses = self.training_history['losses'][-20:]
        if len(recent_losses) > 1:
            min_loss = min(recent_losses)
            max_loss = max(recent_losses)
            
            if max_loss > min_loss:
                for i, loss in enumerate(recent_losses[-10:]):  # 只显示最近10个
                    normalized = (loss - min_loss) / (max_loss - min_loss)
                    bar_length = int(normalized * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    chart_lines.append(f"   {i+1:2d}: {bar} {loss:.4f}")
        
        for i, line in enumerate(chart_lines):
            try:
                self.screen.addstr(start_row + i, start_col, line)
            except curses.error:
                break
    
    def _draw_controls_help(self, start_row: int, start_col: int):
        """绘制控制说明"""
        help_lines = [
            "🎮 控制键: [Q]退出 [P]暂停/继续 [S]保存 [R]重置统计 [H]帮助",
            "⚙️  参数: [1/2]学习率 [3/4]批次大小",
        ]
        
        for i, line in enumerate(help_lines):
            try:
                self.screen.addstr(start_row + i, start_col, line, curses.A_REVERSE)
            except curses.error:
                break
    
    def _show_help(self):
        """显示详细帮助"""
        help_text = [
            "========== 交互式训练控制台帮助 ==========",
            "",
            "🎮 基本控制:",
            "  Q / ESC    - 停止训练并退出",
            "  P          - 暂停/恢复训练",
            "  S          - 立即保存检查点",
            "  R          - 重置训练统计数据",
            "  H          - 显示此帮助信息",
            "",
            "⚙️ 参数调整:",
            "  1          - 增加学习率 (×1.2)",
            "  2          - 减少学习率 (×0.8)",
            "  3          - 增加批次大小 (×1.2)",
            "  4          - 减少批次大小 (×0.8)",
            "",
            "📊 界面说明:",
            "  左上角显示训练状态和基本指标",
            "  右上角显示性能统计信息",
            "  左下角显示可调整参数",
            "  右下角显示损失趋势图",
            "",
            "按任意键返回..."
        ]
        
        self.screen.clear()
        for i, line in enumerate(help_text):
            try:
                self.screen.addstr(i, 2, line)
            except curses.error:
                break
        
        self.screen.refresh()
        self.screen.getch()  # 等待按键
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds//60:.0f}分{seconds%60:.0f}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}时{minutes:.0f}分"
    
    def _cleanup(self):
        """清理资源"""
        if self.screen:
            curses.nocbreak()
            self.screen.keypad(False)
            curses.echo()
            curses.endwin()
        
        self.running = False
        logger.info("交互式训练控制台已关闭")
    
    def save_training_log(self, filename: Optional[str] = None):
        """保存训练日志"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"
        
        log_data = {
            'training_history': self.training_history,
            'final_state': self.training_state,
            'parameters': self.adjustable_params,
            'config': self.config,
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"训练日志已保存: {filename}")
        except Exception as e:
            logger.error(f"保存训练日志失败: {e}")


def create_training_with_ui(training_function, config):
    """
    创建带UI的训练函数包装器
    
    Args:
        training_function: 原始训练函数
        config: 配置字典
        
    Returns:
        交互式训练器实例
    """
    
    def enhanced_training_callback(progress_callback=None, control_queue=None):
        """增强的训练回调，支持进度报告和控制"""
        
        # 修改原始训练函数以支持进度回调和控制
        original_train = training_function
        
        def monitored_train():
            try:
                # 这里需要根据具体的训练函数实现进度监控
                # 示例实现：
                for epoch in range(config.get('epoch', 50)):
                    for batch_idx in range(100):  # 假设每轮100批次
                        
                        # 检查控制命令
                        if control_queue and not control_queue.empty():
                            try:
                                command = control_queue.get_nowait()
                                if command['action'] == 'stop':
                                    return
                                elif command['action'] == 'pause':
                                    while True:  # 暂停循环
                                        if not control_queue.empty():
                                            resume_cmd = control_queue.get_nowait()
                                            if resume_cmd['action'] == 'resume':
                                                break
                                        time.sleep(0.1)
                            except queue.Empty:
                                pass
                        
                        # 模拟训练步骤
                        time.sleep(0.1)  # 模拟训练时间
                        
                        # 报告进度
                        if progress_callback:
                            progress_callback(
                                current_epoch=epoch,
                                current_batch=batch_idx,
                                loss=0.5 * (1 - (epoch * 100 + batch_idx) / 5000),  # 模拟递减损失
                                accuracy=min(0.99, (epoch * 100 + batch_idx) / 5000),  # 模拟递增准确率
                                lr=config.get('learning_rate', 1.0),
                                throughput=128 / 0.1,  # 模拟吞吐量
                            )
            
            except Exception as e:
                logger.error(f"训练过程出错: {e}")
        
        return monitored_train()
    
    return InteractiveTrainer(enhanced_training_callback, config) 