#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Entropy 与 CosineDecay 示意

功能说明：
    演示交叉熵计算与余弦退火学习率生成
    使用Softmax计算概率分布，手动实现交叉熵公式
    搭配循环生成学习率衰减序列展示更新过程

作者：Ph.D. Rhino

执行流程：
    1. 定义logits和真实标签
    2. softmax转换并计算loss
    3. 按余弦曲线更新学习率
    4. 打印loss与lr序列
    5. 输出最优epoch估计

输入说明：
    无用户输入，程序自动生成演示数据

输出展示：
    每个epoch的学习率和损失值
    余弦退火学习率曲线的可视化
    最优epoch的估计结果

技术要点：
    - 手动实现softmax函数避免数值溢出
    - 交叉熵损失的精确计算
    - 余弦退火学习率调度策略实现
    - 模拟训练过程中的损失下降
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    需要安装：numpy（用于数值计算）、matplotlib（用于可视化）
    """
    print("正在检查依赖...")
    
    # 需要的依赖包列表
    required_packages = ["numpy", "matplotlib"]
    missing_packages = []
    
    # 检查每个包是否已安装
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    # 安装缺失的包
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "--upgrade", "pip"])
            subprocess.check_call(["pip", "install"] + missing_packages)
            print("依赖包安装成功！")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            raise
    else:
        print("所有必要的依赖包已安装完成。")


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    """
    安全的softmax函数实现，避免数值溢出
    
    Args:
        logits: 模型的原始输出（未归一化的对数概率）
    
    Returns:
        归一化后的概率分布
    """
    # 减去最大值以避免指数爆炸
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probabilities


def calculate_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    计算交叉熵损失
    
    Args:
        probabilities: softmax输出的概率分布
        labels: 真实标签的one-hot编码
    
    Returns:
        交叉熵损失值
    """
    # 为了数值稳定性，添加一个很小的epsilon以避免log(0)
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    
    # 交叉熵公式: -sum(labels * log(probabilities))
    cross_entropy = -np.sum(labels * np.log(probabilities))
    
    # 返回平均损失
    return cross_entropy / len(labels)


def cosine_decay_lr(initial_lr: float, current_epoch: int, total_epochs: int) -> float:
    """
    余弦退火学习率调度器
    
    Args:
        initial_lr: 初始学习率
        current_epoch: 当前轮次（从0开始）
        total_epochs: 总轮次
    
    Returns:
        当前轮次的学习率
    """
    # 余弦退火公式: lr = initial_lr * (1 + cos(π * current_epoch / total_epochs)) / 2
    lr = initial_lr * (1 + math.cos(math.pi * current_epoch / total_epochs)) / 2
    return lr


def generate_sample_logits_and_labels(num_samples: int = 100, num_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成示例logits和真实标签
    
    Args:
        num_samples: 样本数量
        num_classes: 类别数量
    
    Returns:
        (logits数组, one-hot编码的标签数组)
    """
    print(f"生成示例数据：{num_samples}个样本，{num_classes}个类别")
    
    # 设置随机种子以保证结果可复现
    np.random.seed(42)
    
    # 生成随机logits（范围从-5到5）
    logits = np.random.uniform(-5, 5, size=(num_samples, num_classes))
    
    # 生成随机的真实标签
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    
    # 转换为one-hot编码
    labels = np.zeros((num_samples, num_classes))
    labels[np.arange(num_samples), true_labels] = 1
    
    return logits, labels


def simulate_training_process(
    initial_lr: float = 0.01,
    total_epochs: int = 50,
    num_samples: int = 100,
    num_classes: int = 3
) -> Tuple[List[float], List[float]]:
    """
    模拟训练过程，计算每个epoch的学习率和损失值
    
    Args:
        initial_lr: 初始学习率
        total_epochs: 总训练轮次
        num_samples: 样本数量
        num_classes: 类别数量
    
    Returns:
        (学习率列表, 损失值列表)
    """
    print(f"\n开始模拟训练过程：{total_epochs}个epoch")
    
    # 生成初始数据
    logits, labels = generate_sample_logits_and_labels(num_samples, num_classes)
    
    # 存储每轮的学习率和损失值
    lr_history = []
    loss_history = []
    
    # 模拟训练过程
    for epoch in range(total_epochs):
        # 计算当前轮次的学习率（余弦退火）
        current_lr = cosine_decay_lr(initial_lr, epoch, total_epochs)
        
        # 应用softmax得到概率分布
        probabilities = safe_softmax(logits)
        
        # 计算交叉熵损失
        loss = calculate_cross_entropy(probabilities, labels)
        
        # 模拟模型更新：根据学习率调整logits，使其逐渐接近正确标签
        # 这是一个简化的模拟，实际训练中会使用反向传播和优化器
        logits += simulate_model_update(logits, labels, current_lr)
        
        # 记录学习率和损失
        lr_history.append(current_lr)
        loss_history.append(loss)
        
        # 每10轮或最后一轮打印信息
        if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
            print(f"Epoch {epoch + 1:2d} LR={current_lr:.4f} Loss={loss:.2f}")
    
    return lr_history, loss_history


def simulate_model_update(logits: np.ndarray, labels: np.ndarray, lr: float) -> np.ndarray:
    """
    模拟模型更新过程，根据学习率调整logits
    
    Args:
        logits: 当前的logits值
        labels: 真实标签
        lr: 当前学习率
    
    Returns:
        logits的更新量
    """
    # 计算梯度方向（简化版，真实情况需要通过反向传播计算）
    # 对于交叉熵损失，梯度大约等于 (probabilities - labels)
    probabilities = safe_softmax(logits)
    gradients = probabilities - labels
    
    # 添加一些随机噪声以模拟真实训练中的波动
    noise = np.random.normal(0, 0.01, size=logits.shape)
    
    # 计算更新量：-学习率 * 梯度 + 噪声
    update = -lr * gradients + noise
    
    return update


def find_optimal_epoch(loss_history: List[float]) -> int:
    """
    找到损失值最小的epoch
    
    Args:
        loss_history: 损失值历史记录
    
    Returns:
        最优epoch的索引（从1开始计数）
    """
    min_loss_index = np.argmin(loss_history)
    return min_loss_index + 1  # 转换为从1开始的计数


def visualize_training_process(lr_history: List[float], loss_history: List[float]):
    """
    可视化训练过程中的学习率和损失值变化
    
    Args:
        lr_history: 学习率历史记录
        loss_history: 损失值历史记录
    """
    try:
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制学习率曲线
        epochs = range(1, len(lr_history) + 1)
        ax1.plot(epochs, lr_history, 'b-', linewidth=2)
        ax1.set_title('余弦退火学习率调度')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('学习率')
        ax1.grid(True)
        
        # 绘制损失值曲线
        ax2.plot(epochs, loss_history, 'r-', linewidth=2)
        ax2.set_title('交叉熵损失变化')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.grid(True)
        
        # 找到并标记最优epoch
        optimal_epoch = find_optimal_epoch(loss_history)
        min_loss = min(loss_history)
        ax2.plot(optimal_epoch, min_loss, 'go', markersize=8, label=f'最优Epoch: {optimal_epoch}')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像而不显示（避免在某些环境下需要图形界面）
        plt.savefig('lr_loss_schedule.png', dpi=300, bbox_inches='tight')
        print(f"\n训练过程可视化已保存为 'lr_loss_schedule.png'")
        
        # 关闭图像以释放内存
        plt.close()
        
    except Exception as e:
        print(f"\n可视化过程中出现错误: {e}")
        print("跳过可视化步骤，但计算结果仍然可用")


def print_training_summary(lr_history: List[float], loss_history: List[float]):
    """
    打印训练过程总结
    
    Args:
        lr_history: 学习率历史记录
        loss_history: 损失值历史记录
    """
    print("\n" + "=" * 50)
    print("训练过程总结")
    print("=" * 50)
    
    # 计算统计信息
    initial_lr = lr_history[0]
    final_lr = lr_history[-1]
    lr_ratio = final_lr / initial_lr
    
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    min_loss = min(loss_history)
    
    # 找到最优epoch
    optimal_epoch = find_optimal_epoch(loss_history)
    
    # 计算损失下降百分比
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    
    print(f"初始学习率: {initial_lr:.4f}")
    print(f"最终学习率: {final_lr:.4f}")
    print(f"学习率衰减比例: {lr_ratio:.4f}x")
    print(f"初始损失: {initial_loss:.4f}")
    print(f"最终损失: {final_loss:.4f}")
    print(f"最小损失: {min_loss:.4f} (在第{optimal_epoch}轮达到)")
    print(f"损失下降百分比: {loss_reduction:.2f}%")
    
    # 分析学习率调度效果
    print("\n学习率调度分析:")
    if lr_ratio < 0.01:
        print("- 学习率衰减幅度较大，有助于模型在训练后期精细调整")
    elif lr_ratio < 0.1:
        print("- 学习率衰减适中，平衡了训练速度和收敛效果")
    else:
        print("- 学习率衰减较小，可能需要更多轮次才能充分收敛")
    
    # 分析损失下降情况
    print("\n损失下降分析:")
    if loss_reduction > 80:
        print("- 损失下降显著，模型训练效果良好")
    elif loss_reduction > 50:
        print("- 损失下降适中，模型有较好的学习能力")
    else:
        print("- 损失下降较小，可能需要调整学习率或增加训练轮次")


def print_ini_format_output(lr_history: List[float], loss_history: List[float]):
    """
    按用户要求的格式输出关键信息
    
    Args:
        lr_history: 学习率历史记录
        loss_history: 损失值历史记录
    """
    print("\n" + "=" * 50)
    print("INI格式输出示例")
    print("=" * 50)
    print("ini")
    
    # 输出第1轮、第10轮和最后一轮的数据
    print(f"Epoch 1 LR={lr_history[0]:.2f} Loss={loss_history[0]:.2f}")
    
    # 输出第10轮（如果存在）
    if len(lr_history) >= 10:
        print(f"Epoch 10 LR={lr_history[9]:.2f} Loss={loss_history[9]:.2f}")
    
    # 输出最后一轮
    print(f"Epoch {len(lr_history)} LR={lr_history[-1]:.4f} Loss={loss_history[-1]:.4f}")


def main():
    """
    主函数
    """
    print("===== Cross-Entropy 与 CosineDecay 演示程序 =====")
    
    # 检查依赖
    check_and_install_dependencies()
    
    # 设置训练参数
    initial_lr = 0.01
    total_epochs = 50
    num_samples = 100
    num_classes = 3
    
    # 打印参数信息
    print(f"\n训练参数:")
    print(f"初始学习率: {initial_lr}")
    print(f"总训练轮次: {total_epochs}")
    print(f"样本数量: {num_samples}")
    print(f"类别数量: {num_classes}")
    
    # 模拟训练过程
    lr_history, loss_history = simulate_training_process(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        num_samples=num_samples,
        num_classes=num_classes
    )
    
    # 按要求格式输出信息
    print_ini_format_output(lr_history, loss_history)
    
    # 可视化训练过程
    visualize_training_process(lr_history, loss_history)
    
    # 打印训练总结
    print_training_summary(lr_history, loss_history)
    
    print("\n===== 演示程序完成 =====")
    print("\n关键概念解释:")
    print("1. 交叉熵(Cross-Entropy): 衡量预测概率分布与真实分布之间的差异")
    print("2. 余弦退火学习率(CosineDecay): 根据余弦函数逐渐降低学习率")
    print("3. Softmax: 将logits转换为概率分布，确保所有类别的概率和为1")
    print("4. 最优Epoch: 损失值达到最小的训练轮次")


if __name__ == "__main__":
    main()