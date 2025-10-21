#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练损失曲线记录演示

功能说明：
    模拟模型训练并记录loss随epoch变化的趋势
    使用纯文本ASCII字符绘制实时训练曲线，不依赖图形库
    实现Early Stopping逻辑检测训练收敛点
    提供训练过程可视化与统计分析

作者：Ph.D. Rhino

执行流程：
    1. 检查并安装必要的依赖
    2. 初始化loss起始值和训练参数
    3. 模拟epoch循环，计算带随机波动的递减loss
    4. 用ASCII字符打印每轮loss值与趋势指示
    5. 计算最优epoch并触发Early Stopping
    6. 输出最终训练报告与统计数据

输入说明：
    无（程序自动模拟训练数据）

输出展示：
    - 实时训练过程可视化
    - ASCII图形化的损失曲线
    - Early Stopping触发提示
    - 最终训练统计报告
"""

import os
import random
import time
import math
from typing import List, Dict, Tuple, Optional


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    本程序主要使用Python标准库，无需额外依赖
    """
    print("正在检查依赖...")
    print("✓ 本程序仅使用Python标准库，无需额外依赖")
    print("依赖检查完成！")


# 移除单独的generate_training_losses函数，功能已整合到run_training_simulation中


# 移除平滑损失函数，简化实现


def detect_early_stopping(
    losses: List[Tuple[int, float]],
    patience: int = 3,
    min_delta: float = 0.01
) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    检测是否应该触发Early Stopping
    
    Args:
        losses: 损失列表
        patience: 容忍没有改善的轮数
        min_delta: 认为是改善的最小变化量
    
    Returns:
        (是否触发Early Stopping, 最优epoch, 最小loss)
    """
    if len(losses) < patience + 1:
        return False, None, None
    
    # 找出最小loss及其位置
    min_loss = float('inf')
    best_epoch = 0
    
    for epoch, loss in losses:
        if loss < min_loss:
            min_loss = loss
            best_epoch = epoch
    
    # 检查最近patience轮是否有改善
    recent_epochs = losses[-patience:]
    has_improvement = any(loss[1] < min_loss - min_delta for loss in recent_epochs)
    
    # 如果最近patience轮没有改善，触发Early Stopping
    if not has_improvement and best_epoch <= losses[-patience-1][0]:
        return True, best_epoch, min_loss
    
    return False, best_epoch, min_loss


# 移除复杂的ASCII绘图函数，简化为简单的文本展示


def print_training_progress(
    epoch: int,
    loss: float,
    losses: List[Tuple[int, float]],
    max_epochs: int
) -> None:
    """
    打印训练进度
    
    Args:
        epoch: 当前epoch
        loss: 当前loss值
        losses: 历史loss列表
        max_epochs: 最大epoch数
    """
    # 打印基本信息（不清除屏幕，避免兼容性问题）
    print(f"\n===== 训练进度监控 =====")
    print(f"Epoch {epoch}/{max_epochs}: loss={loss:.4f}")
    
    # 打印简单的进度条
    bar_length = 30
    filled_length = int(bar_length * epoch / max_epochs)
    bar = '▓' * filled_length + '-' * (bar_length - filled_length)
    print(f"[{bar}]")
    
    # 打印loss趋势指示
    if epoch > 1:
        prev_loss = losses[epoch-2][1] if epoch-2 >= 0 else loss
        trend = "↓" if loss < prev_loss else "↑" if loss > prev_loss else "→"
        print(f"趋势: {trend}")
    
    # 简化的文本形式展示loss趋势
    print("\n损失趋势:")
    # 计算合适的缩放因子
    max_loss = max(loss[1] for loss in losses)
    scale = 50 / max_loss if max_loss > 0 else 50
    
    for i, (e, l) in enumerate(losses[-min(10, epoch):]):
        # 使用'▓'字符长度表示loss大小
        bar_length = max(1, int(l * scale))
        bar = '▓' * bar_length
        print(f"  Epoch {e:2d}: loss={l:.4f} {bar}")


def run_training_simulation(
    num_epochs: int = 20,
    initial_loss: float = 1.2,
    patience: int = 3,
    min_delta: float = 0.02,
    delay: float = 0.3
) -> Dict[str, any]:
    """
    运行训练模拟
    
    Args:
        num_epochs: 最大训练轮数
        initial_loss: 初始损失值
        patience: Early Stopping耐心值
        min_delta: Early Stopping最小变化量
        delay: 每轮显示延迟时间（秒）
    
    Returns:
        训练统计结果
    """
    print(f"开始训练模拟 (最大{num_epochs}轮)...")
    
    # 训练历史
    training_history = []
    early_stopped = False
    best_epoch = None
    min_loss = float('inf')
    
    # 模拟训练循环
    for epoch in range(1, num_epochs + 1):
        # 生成当前轮次的loss（简化计算）
        # 基础衰减 + 随机波动
        base_loss = initial_loss * math.exp(-0.1 * epoch)
        noise = (random.random() - 0.5) * 0.1
        loss = max(0.1, base_loss + noise)
        
        # 添加一些随机性使曲线更真实
        if random.random() < 0.15:
            loss = min(loss * 1.3, initial_loss)
        
        # 更新训练历史
        training_history.append((epoch, loss))
        
        # 打印训练进度
        print_training_progress(epoch, loss, training_history, num_epochs)
        
        # 检查Early Stopping
        should_stop, current_best_epoch, current_min_loss = detect_early_stopping(
            training_history, patience, min_delta
        )
        
        if current_best_epoch and current_min_loss:
            best_epoch = current_best_epoch
            min_loss = current_min_loss
        
        if should_stop and not early_stopped:
            early_stopped = True
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}!")
            print(f"  最优模型在 epoch {best_epoch}, loss = {min_loss:.4f}")
            # 继续执行以展示完整结果
        
        # 暂停以模拟真实训练过程
        time.sleep(delay)
    
    # 计算统计数据
    total_epochs = len(training_history)
    final_loss = training_history[-1][1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    # 如果没有触发Early Stopping，找出最佳epoch
    if not early_stopped:
        min_loss = min(loss[1] for loss in training_history)
        best_epoch = next(epoch for epoch, loss in training_history if loss == min_loss)
    
    # 返回统计结果
    return {
        "total_epochs": total_epochs,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "min_loss": min_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "training_history": training_history
    }


def generate_training_report(stats: Dict[str, any]) -> str:
    """
    生成训练报告
    
    Args:
        stats: 训练统计结果
    
    Returns:
        格式化的训练报告字符串
    """
    report = []
    report.append("\n" + "="*50)
    report.append("训练完成报告")
    report.append("="*50)
    report.append(f"总训练轮数: {stats['total_epochs']}")
    report.append(f"是否提前停止: {'是' if stats['early_stopped'] else '否'}")
    report.append(f"最佳模型轮数: {stats['best_epoch']}")
    report.append(f"最小损失值: {stats['min_loss']:.6f}")
    report.append(f"最终损失值: {stats['final_loss']:.6f}")
    report.append(f"损失降低比例: {stats['loss_reduction']:.2f}%")
    
    # 计算收敛时间估计
    converged_epochs = stats['best_epoch']
    convergence_rate = stats['loss_reduction'] / converged_epochs if converged_epochs > 0 else 0
    report.append(f"平均每轮损失降低: {convergence_rate:.2f}%")
    
    # 评估训练稳定性
    losses = [loss[1] for loss in stats['training_history']]
    loss_std = 0
    if len(losses) > 1:
        mean_loss = sum(losses) / len(losses)
        loss_std = math.sqrt(sum((l - mean_loss)**2 for l in losses) / len(losses))
    
    stability = "稳定" if loss_std < 0.1 else "中等" if loss_std < 0.3 else "不稳定"
    report.append(f"训练稳定性: {stability} (标准差: {loss_std:.4f})")
    
    # 生成简化的ASCII最终曲线
    print("\n最终损失曲线:")
    # 使用简单的文本条形图
    max_loss = max(loss[1] for loss in stats['training_history'])
    scale = 50 / max_loss if max_loss > 0 else 50
    
    for epoch, loss in stats['training_history']:
        bar_length = max(1, int(loss * scale))
        bar = '▓' * bar_length
        print(f"  Epoch {epoch:2d}: loss={loss:.4f} {bar}")
    
    return '\n'.join(report)


def main():
    """
    主函数
    """
    print("===== 训练损失曲线记录演示程序 =====")
    
    # 检查依赖
    check_and_install_dependencies()
    
    # 设置随机种子以保证结果可复现
    random.seed(42)
    
    # 运行训练模拟
    stats = run_training_simulation(
        num_epochs=20,
        initial_loss=1.2,
        patience=3,
        min_delta=0.02,
        delay=0.3  # 为了演示加快速度
    )
    
    # 生成并打印训练报告
    report = generate_training_report(stats)
    print(report)
    
    # 打印总结
    print("\n训练监控总结:")
    print("- 使用ASCII字符绘制实时损失曲线，不依赖图形库")
    print("- 实现Early Stopping逻辑，自动检测训练收敛")
    print("- 提供详细的训练统计和稳定性分析")
    print("- 模拟真实训练过程中的损失波动")
    
    print("\n===== 训练损失曲线记录演示程序完成 =====")


if __name__ == "__main__":
    main()