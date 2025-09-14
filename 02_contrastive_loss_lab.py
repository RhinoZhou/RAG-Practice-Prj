#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比学习与训练损失可视化实验

功能说明:
- 模拟InfoNCE损失函数在不同温度参数下的表现
- 模拟TripletLoss损失函数在不同边距(margin)参数下的表现
- 生成合成查询(query)、正样本(positive)和负样本(negative)相似度数据
- 展示难负样本(hard negatives)与随机负样本(random negatives)的损失差异
- 绘制损失曲线并保存为PNG图像

输入:
- 可选配置文件: configs/loss_demo.json (包含超参范围、样本规模等设置)

输出:
- outputs/infonce_curve.png: InfoNCE损失随温度变化的曲线
- outputs/triplet_curve.png: TripletLoss损失随边距变化的曲线
- 控制台输出: 难负样本与随机负样本的损失均值差异摘要

依赖包:
- numpy: 用于数值计算和随机数据生成
- matplotlib: 用于绘制损失曲线
- scipy: 用于统计计算


"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 设置中文字体，确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 检查并安装依赖包
def check_and_install_dependencies():
    """检查并自动安装必要的依赖包"""
    required_packages = [
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0"
    ]
    
    try:
        # 检查是否已安装所有必要的包
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy
        print("✓ 所有依赖包已安装完成")
    except ImportError:
        print("正在安装必要的依赖包...")
        # 使用pip安装缺失的包
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ 成功安装 {package}")
            except Exception as e:
                print(f"✗ 安装 {package} 失败: {str(e)}")
                raise RuntimeError(f"无法安装必要的依赖包，请手动安装: {package}") from e

# 创建输出目录
def create_output_directory(directory="outputs"):
    """创建输出目录，用于保存图像文件等"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ 创建输出目录: {directory}")
    return directory

# 加载配置文件
def load_config(config_path="configs/loss_demo.json"):
    """加载配置文件，如果不存在则使用默认配置"""
    # 默认配置
    default_config = {
        "num_samples": 1000,  # 样本数量
        "temperature_range": [0.1, 5.0],  # 温度参数范围
        "temperature_steps": 50,  # 温度参数步数
        "margin_range": [0.0, 2.0],  # 边距参数范围
        "margin_steps": 50,  # 边距参数步数
        "positive_similarity_mean": 0.8,  # 正样本相似度均值
        "positive_similarity_std": 0.1,  # 正样本相似度标准差
        "random_negative_mean": 0.2,  # 随机负样本相似度均值
        "random_negative_std": 0.2,  # 随机负样本相似度标准差
        "hard_negative_mean": 0.6,  # 难负样本相似度均值
        "hard_negative_std": 0.1  # 难负样本相似度标准差
    }
    
    # 检查是否有自定义配置文件
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 合并默认配置和自定义配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            print(f"✓ 从文件加载配置: {config_path}")
            return config
        except Exception as e:
            print(f"✗ 加载自定义配置失败: {str(e)}，使用默认配置")
    
    # 使用默认配置
    print("✓ 使用默认配置")
    return default_config

# 生成合成的三元组相似度数据
def generate_triplet_similarities(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成查询-正样本-负样本的相似度数据
    
    这个函数模拟了对比学习中常见的三种相似度分布：
    1. 正样本相似度：查询与相关文档的高相似度
    2. 随机负样本相似度：查询与不相关文档的低相似度  
    3. 难负样本相似度：查询与相似但不相关文档的中等相似度
    
    参数:
    - config: 配置字典，包含生成参数
    
    返回:
    - pos_similarities: 查询与正样本的相似度数组
    - random_neg_similarities: 查询与随机负样本的相似度数组
    - hard_neg_similarities: 查询与难负样本的相似度数组
    """
    np.random.seed(42)  # 设置随机种子，保证结果可复现
    
    # 生成查询与正样本的相似度
    pos_similarities = np.random.normal(
        loc=config["positive_similarity_mean"],
        scale=config["positive_similarity_std"],
        size=config["num_samples"]
    )
    # 确保相似度在合理范围内
    pos_similarities = np.clip(pos_similarities, 0.0, 1.0)
    
    # 生成查询与随机负样本的相似度
    random_neg_similarities = np.random.normal(
        loc=config["random_negative_mean"],
        scale=config["random_negative_std"],
        size=config["num_samples"]
    )
    random_neg_similarities = np.clip(random_neg_similarities, 0.0, 1.0)
    
    # 生成查询与难负样本的相似度
    hard_neg_similarities = np.random.normal(
        loc=config["hard_negative_mean"],
        scale=config["hard_negative_std"],
        size=config["num_samples"]
    )
    hard_neg_similarities = np.clip(hard_neg_similarities, 0.0, 1.0)
    
    print(f"✓ 生成了{config['num_samples']}个三元组相似度样本")
    print(f"  正样本相似度均值: {np.mean(pos_similarities):.4f}")
    print(f"  随机负样本相似度均值: {np.mean(random_neg_similarities):.4f}")
    print(f"  难负样本相似度均值: {np.mean(hard_neg_similarities):.4f}")
    
    return pos_similarities, random_neg_similarities, hard_neg_similarities

# 计算InfoNCE损失
def calculate_infonce_loss(pos_sim: np.ndarray, neg_sim: np.ndarray, temperature: float) -> float:
    """
    计算InfoNCE损失 (InfoNCE Loss)
    
    InfoNCE (Information Noise Contrastive Estimation) 是一种对比学习损失函数，
    通过最大化正样本对的相似度，同时最小化负样本对的相似度来学习表示。
    
    数学公式:
    InfoNCE = -log(exp(pos_sim/temperature) / (exp(pos_sim/temperature) + sum(exp(neg_sim/temperature))))
    
    参数:
    - pos_sim: 查询与正样本的相似度数组
    - neg_sim: 查询与负样本的相似度数组  
    - temperature: 温度参数，控制softmax的锐度，越小越锐利
    
    返回:
    - 平均InfoNCE损失值
    
    注意: 为简化计算，这里每个查询只使用一个负样本
    """
    # 防止除零错误
    temperature = max(temperature, 1e-8)
    
    # 计算正样本的指数项
    pos_exp = np.exp(pos_sim / temperature)
    
    # 计算负样本的指数项
    neg_exp = np.exp(neg_sim / temperature)
    
    # 计算InfoNCE损失
    loss = -np.log(pos_exp / (pos_exp + neg_exp))
    
    # 返回平均损失
    return np.mean(loss)

# 计算TripletLoss损失
def calculate_triplet_loss(pos_sim: np.ndarray, neg_sim: np.ndarray, margin: float) -> float:
    """
    计算TripletLoss损失 (Triplet Loss)
    
    TripletLoss是一种对比学习损失函数，通过三元组(anchor, positive, negative)来学习表示。
    目标是让正样本与anchor的相似度尽可能高，负样本与anchor的相似度尽可能低。
    
    数学公式:
    TripletLoss = max(0, margin + neg_sim - pos_sim)
    
    参数:
    - pos_sim: 查询与正样本的相似度数组
    - neg_sim: 查询与负样本的相似度数组
    - margin: 边距参数，控制正负样本之间的最小间隔
    
    返回:
    - 平均TripletLoss损失值
    
    注意: 这里我们使用相似度而非距离，所以公式有所调整
    """
    # 计算损失
    loss = np.maximum(0.0, margin + neg_sim - pos_sim)
    
    # 返回平均损失
    return np.mean(loss)

# 扫描超参数并计算损失
def scan_hyperparameters(
    pos_sim: np.ndarray,
    random_neg_sim: np.ndarray,
    hard_neg_sim: np.ndarray,
    config: Dict
) -> Tuple[Dict, Dict]:
    """
    扫描超参数（温度和边距）并计算相应的损失
    """
    # 生成温度参数列表
    temperatures = np.linspace(
        config["temperature_range"][0],
        config["temperature_range"][1],
        config["temperature_steps"]
    )
    
    # 生成边距参数列表
    margins = np.linspace(
        config["margin_range"][0],
        config["margin_range"][1],
        config["margin_steps"]
    )
    
    # 存储InfoNCE损失结果
    infonce_results = {
        "temperatures": temperatures,
        "random_neg_loss": [],
        "hard_neg_loss": []
    }
    
    # 存储TripletLoss损失结果
    triplet_results = {
        "margins": margins,
        "random_neg_loss": [],
        "hard_neg_loss": []
    }
    
    print("✓ 开始扫描超参数并计算损失...")
    
    # 计算InfoNCE损失
    for temp in temperatures:
        random_loss = calculate_infonce_loss(pos_sim, random_neg_sim, temp)
        hard_loss = calculate_infonce_loss(pos_sim, hard_neg_sim, temp)
        infonce_results["random_neg_loss"].append(random_loss)
        infonce_results["hard_neg_loss"].append(hard_loss)
    
    # 计算TripletLoss损失
    for margin in margins:
        random_loss = calculate_triplet_loss(pos_sim, random_neg_sim, margin)
        hard_loss = calculate_triplet_loss(pos_sim, hard_neg_sim, margin)
        triplet_results["random_neg_loss"].append(random_loss)
        triplet_results["hard_neg_loss"].append(hard_loss)
    
    print("✓ 超参数扫描完成")
    
    return infonce_results, triplet_results

# 绘制并保存InfoNCE损失曲线
def plot_infonce_curve(results: Dict, output_path: str = "outputs/infonce_curve.png"):
    """绘制并保存InfoNCE损失随温度变化的曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制随机负样本的损失曲线
    plt.plot(
        results["temperatures"],
        results["random_neg_loss"],
        label="随机负样本",
        linewidth=2,
        color="blue"
    )
    
    # 绘制难负样本的损失曲线
    plt.plot(
        results["temperatures"],
        results["hard_neg_loss"],
        label="难负样本",
        linewidth=2,
        color="red"
    )
    
    # 设置图表属性
    plt.title("InfoNCE损失随温度参数变化的曲线", fontsize=16)
    plt.xlabel("温度参数 (Temperature)", fontsize=14)
    plt.ylabel("平均损失值", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ InfoNCE损失曲线图已保存至: {output_path}")
    
    plt.close()

# 绘制并保存TripletLoss损失曲线
def plot_triplet_curve(results: Dict, output_path: str = "outputs/triplet_curve.png"):
    """绘制并保存TripletLoss损失随边距变化的曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制随机负样本的损失曲线
    plt.plot(
        results["margins"],
        results["random_neg_loss"],
        label="随机负样本",
        linewidth=2,
        color="blue"
    )
    
    # 绘制难负样本的损失曲线
    plt.plot(
        results["margins"],
        results["hard_neg_loss"],
        label="难负样本",
        linewidth=2,
        color="red"
    )
    
    # 设置图表属性
    plt.title("TripletLoss损失随边距参数变化的曲线", fontsize=16)
    plt.xlabel("边距参数 (Margin)", fontsize=14)
    plt.ylabel("平均损失值", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ TripletLoss损失曲线图已保存至: {output_path}")
    
    plt.close()

# 计算并打印损失差异摘要
def print_loss_comparison_summary(
    pos_sim: np.ndarray,
    random_neg_sim: np.ndarray,
    hard_neg_sim: np.ndarray
):
    """计算并打印难负样本与随机负样本的损失差异摘要"""
    print("\n=== 难负样本与随机负样本损失对比摘要 ===")
    
    # 使用默认超参数计算损失
    default_temperature = 0.5
    default_margin = 0.2
    
    # 计算InfoNCE损失
    infonce_random_loss = calculate_infonce_loss(pos_sim, random_neg_sim, default_temperature)
    infonce_hard_loss = calculate_infonce_loss(pos_sim, hard_neg_sim, default_temperature)
    infonce_diff = infonce_hard_loss - infonce_random_loss
    
    # 计算TripletLoss损失
    triplet_random_loss = calculate_triplet_loss(pos_sim, random_neg_sim, default_margin)
    triplet_hard_loss = calculate_triplet_loss(pos_sim, hard_neg_sim, default_margin)
    triplet_diff = triplet_hard_loss - triplet_random_loss
    
    # 打印对比结果
    print(f"\n默认温度参数: {default_temperature}")
    print(f"InfoNCE损失 - 随机负样本: {infonce_random_loss:.4f}")
    print(f"InfoNCE损失 - 难负样本: {infonce_hard_loss:.4f}")
    print(f"InfoNCE损失差异: {infonce_diff:.4f} ({infonce_diff/infonce_random_loss*100:.1f}%)")
    
    print(f"\n默认边距参数: {default_margin}")
    print(f"TripletLoss损失 - 随机负样本: {triplet_random_loss:.4f}")
    print(f"TripletLoss损失 - 难负样本: {triplet_hard_loss:.4f}")
    print(f"TripletLoss损失差异: {triplet_diff:.4f} ({triplet_diff/triplet_random_loss*100:.1f}%)")
    
    # 简单分析
    print("\n分析说明:")
    print("- 难负样本通常会导致更高的损失值，因为它们与查询的相似度更高")
    print("- 在训练过程中，使用难负样本可以提高模型的学习效率和性能")
    print("- InfoNCE损失对温度参数敏感，较低的温度会放大相似度差异")
    print("- TripletLoss的性能取决于边距参数的选择，合适的边距可以平衡正负样本的区分度")

# 主函数
def main():
    """主函数，执行完整流程"""
    print("\n===== 对比学习与训练损失可视化实验 =====")
    
    # 1. 检查并安装依赖
    check_and_install_dependencies()
    
    # 2. 创建输出目录
    output_dir = create_output_directory()
    
    # 3. 加载配置
    config = load_config()
    
    # 4. 生成三元组相似度数据
    pos_sim, random_neg_sim, hard_neg_sim = generate_triplet_similarities(config)
    
    # 5. 扫描超参数并计算损失
    infonce_results, triplet_results = scan_hyperparameters(
        pos_sim, random_neg_sim, hard_neg_sim, config
    )
    
    # 6. 绘制并保存损失曲线
    infonce_path = os.path.join(output_dir, "infonce_curve.png")
    triplet_path = os.path.join(output_dir, "triplet_curve.png")
    plot_infonce_curve(infonce_results, infonce_path)
    plot_triplet_curve(triplet_results, triplet_path)
    
    # 7. 打印损失对比摘要
    print_loss_comparison_summary(pos_sim, random_neg_sim, hard_neg_sim)
    
    print("\n===== 程序执行完成 =====")

# 程序入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)