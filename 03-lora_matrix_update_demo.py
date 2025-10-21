#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA 权重更新逻辑模拟

功能说明：
    展示 LoRA 如何用低秩矩阵 A、B 更新冻结权重
    构造简化矩阵乘法场景，展示 W' = W + A×B 的更新原理，并验证显存与计算量优势。

作者：Ph.D. Rhino

执行流程：
    1. 检查并安装必要的依赖
    2. 随机生成权重矩阵 W
    3. 初始化低秩矩阵 A, B
    4. 计算更新后的权重矩阵 W'
    5. 计算并输出矩阵差分范数
    6. 模拟LoRA层仅更新A,B的情况
    7. 统计内存占用差异
    8. 对比计算效率和参数减少比例

输入说明：
    程序会自动生成随机矩阵进行模拟

输出展示：
    - LoRA的秩（rank）值
    - 参数减少百分比
    - 矩阵差分的Frobenius范数
    - 内存占用对比
    - 计算效率分析
"""

import os
import time
import numpy as np
import psutil
from typing import Tuple, Dict, Any


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    需要安装：numpy, psutil
    """
    print("正在检查依赖...")
    
    # 需要的依赖包列表
    required_packages = ["numpy", "psutil"]
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


def get_memory_usage() -> int:
    """
    获取当前进程的内存使用情况（以字节为单位）
    
    Returns:
        内存使用量（字节）
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # 以字节为单位


def generate_weights(in_dim: int, out_dim: int, seed: int = 42) -> np.ndarray:
    """
    生成随机权重矩阵 W
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        seed: 随机种子，用于复现结果
    
    Returns:
        权重矩阵 W
    """
    np.random.seed(seed)
    # 生成均值为0，标准差为0.01的随机矩阵，模拟预训练权重
    W = np.random.normal(0, 0.01, (out_dim, in_dim))
    return W


def initialize_lora_matrices(in_dim: int, out_dim: int, rank: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    初始化LoRA低秩矩阵 A 和 B
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: LoRA的秩
        seed: 随机种子
    
    Returns:
        (矩阵A, 矩阵B) 的元组
    """
    np.random.seed(seed + 1)  # 使用不同的种子避免相关性
    
    # 初始化矩阵A：out_dim × rank，使用Kaiming初始化
    # 参考文献：https://arxiv.org/abs/1502.01852
    bound = np.sqrt(6.0 / out_dim)
    A = np.random.uniform(-bound, bound, (out_dim, rank))
    
    # 初始化矩阵B：rank × in_dim，初始化为较小的值
    # LoRA论文建议初始化为零，但这里使用极小的随机值以模拟训练开始
    B = np.random.normal(0, 1e-5, (rank, in_dim))
    
    return A, B


def calculate_lora_update(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    计算LoRA更新矩阵 delta_W = A × B
    
    Args:
        A: 矩阵A (out_dim × rank)
        B: 矩阵B (rank × in_dim)
    
    Returns:
        更新矩阵 delta_W = A × B
    """
    return np.dot(A, B)


def calculate_frobenius_norm(matrix: np.ndarray) -> float:
    """
    计算矩阵的Frobenius范数
    
    Args:
        matrix: 输入矩阵
    
    Returns:
        Frobenius范数
    """
    return np.linalg.norm(matrix, 'fro')


def compare_parameter_count(in_dim: int, out_dim: int, rank: int) -> Dict[str, Any]:
    """
    比较原始权重和LoRA权重的参数数量
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: LoRA的秩
    
    Returns:
        包含参数数量信息的字典
    """
    # 原始权重参数数量
    original_params = in_dim * out_dim
    
    # LoRA参数数量 (A + B)
    lora_params = in_dim * rank + out_dim * rank
    
    # 参数减少比例
    param_reduction = (1 - lora_params / original_params) * 100
    
    return {
        "original_params": original_params,
        "lora_params": lora_params,
        "param_reduction": param_reduction
    }


def simulate_forward_pass(W: np.ndarray, A: np.ndarray, B: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟前向传播
    
    Args:
        W: 原始权重矩阵
        A: LoRA矩阵A
        B: LoRA矩阵B
        x: 输入向量
    
    Returns:
        (原始输出, LoRA增强后的输出) 的元组
    """
    # 原始输出: W × x
    y_original = np.dot(W, x)
    
    # LoRA增强后的输出: (W + A×B) × x = W×x + A×(B×x)
    # 注意：实际实现中为了提高计算效率，会使用 A×(B×x) 而不是 (A×B)×x
    y_lora = y_original + np.dot(A, np.dot(B, x))
    
    return y_original, y_lora


def benchmark_computation(in_dim: int, out_dim: int, rank: int, num_runs: int = 10) -> Dict[str, float]:
    """
    对比原始方法和LoRA方法的计算效率
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: LoRA的秩
        num_runs: 运行次数
    
    Returns:
        包含计算时间的字典
    """
    # 使用批处理输入以更好地展示LoRA的优势
    batch_size = 32
    
    # 生成权重和批量输入
    W = generate_weights(in_dim, out_dim)
    A, B = initialize_lora_matrices(in_dim, out_dim, rank)
    x_batch = np.random.randn(batch_size, in_dim)
    
    # 计算完整的LoRA更新矩阵
    delta_W = calculate_lora_update(A, B)
    W_prime = W + delta_W
    
    # 预热运行
    for _ in range(3):
        np.dot(W_prime, x_batch[0])
        np.dot(W, x_batch[0]) + np.dot(A, np.dot(B, x_batch[0]))
    
    # 基准测试1：直接计算W'×x（模拟全参数微调的前向传播）
    start_time = time.time()
    for _ in range(num_runs):
        for x in x_batch:
            y1 = np.dot(W_prime, x)
    original_time = time.time() - start_time
    
    # 基准测试2：计算W×x + A×(B×x)（模拟LoRA的优化前向传播）
    start_time = time.time()
    for _ in range(num_runs):
        for x in x_batch:
            y2 = np.dot(W, x) + np.dot(A, np.dot(B, x))
    lora_time = time.time() - start_time
    
    # 基准测试3：预计算B×x的中间结果（实际LoRA实现常用优化）
    start_time = time.time()
    for _ in range(num_runs):
        for x in x_batch:
            bx = np.dot(B, x)
            axb = np.dot(A, bx)
            y3 = np.dot(W, x) + axb
    optimized_time = time.time() - start_time
    
    # 计算加速比
    speedup_vs_original = original_time / lora_time if lora_time > 0 else float('inf')
    speedup_optimized = original_time / optimized_time if optimized_time > 0 else float('inf')
    
    return {
        "original_time": original_time,
        "lora_time": lora_time,
        "optimized_time": optimized_time,
        "speedup": speedup_optimized  # 使用最佳优化版本的加速比
    }


def simulate_lora_training(in_dim: int, out_dim: int, rank: int) -> Dict[str, Any]:
    """
    模拟LoRA训练过程的内存使用
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: LoRA的秩
    
    Returns:
        包含内存使用情况的字典
    """
    # 对于小矩阵，直接计算理论内存使用量更准确
    # 每个浮点数占8字节
    bytes_per_float = 8
    
    # 1. 原始全参数微调和LoRA微调的理论内存计算
    # 原始微调：权重矩阵 + 梯度矩阵
    original_memory_bytes = 2 * in_dim * out_dim * bytes_per_float
    
    # LoRA微调：冻结权重矩阵(不计入可训练参数内存) + A矩阵 + B矩阵 + A的梯度 + B的梯度
    # 注意：冻结权重的内存仍然存在，但不需要用于反向传播更新，不计入可训练参数内存
    lora_memory_bytes = (in_dim * rank + out_dim * rank) * bytes_per_float * 2
    
    # 转换为MB
    original_memory_mb = original_memory_bytes / (1024 * 1024)
    lora_memory_mb = lora_memory_bytes / (1024 * 1024)
    
    # 计算内存节省比例（针对可训练参数）
    memory_reduction = (1 - lora_memory_bytes / original_memory_bytes) * 100
    
    return {
        "original_memory_mb": original_memory_mb,
        "lora_memory_mb": lora_memory_mb,
        "memory_reduction": memory_reduction
    }


def run_demo(in_dim: int = 1024, out_dim: int = 1024, rank: int = 8) -> None:
    """
    运行LoRA更新逻辑演示
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: LoRA的秩
    """
    print("===== LoRA 权重更新逻辑模拟演示 =====")
    print(f"\n配置参数:")
    print(f"  输入维度: {in_dim}")
    print(f"  输出维度: {out_dim}")
    print(f"  LoRA秩(rank): {rank}")
    
    # 1. 生成原始权重矩阵
    print("\n1. 生成原始权重矩阵 W...")
    W = generate_weights(in_dim, out_dim)
    print(f"  W形状: {W.shape}")
    print(f"  W的前3×3子矩阵:\n{W[:3, :3]}")
    
    # 2. 初始化LoRA矩阵
    print("\n2. 初始化LoRA低秩矩阵 A 和 B...")
    A, B = initialize_lora_matrices(in_dim, out_dim, rank)
    print(f"  A形状: {A.shape}")
    print(f"  B形状: {B.shape}")
    print(f"  A的前3×3子矩阵:\n{A[:3, :min(3, rank)]}")
    print(f"  B的前3×3子矩阵:\n{B[:min(3, rank), :3]}")
    
    # 3. 计算LoRA更新矩阵 delta_W = A × B
    print("\n3. 计算LoRA更新矩阵 delta_W = A × B...")
    delta_W = calculate_lora_update(A, B)
    print(f"  delta_W形状: {delta_W.shape}")
    print(f"  delta_W的前3×3子矩阵:\n{delta_W[:3, :3]}")
    
    # 4. 计算更新后的权重矩阵 W' = W + delta_W
    print("\n4. 计算更新后的权重矩阵 W' = W + delta_W...")
    W_prime = W + delta_W
    print(f"  W'的前3×3子矩阵:\n{W_prime[:3, :3]}")
    
    # 5. 计算矩阵差分的Frobenius范数
    print("\n5. 计算矩阵差分的Frobenius范数...")
    delta_norm = calculate_frobenius_norm(W_prime - W)
    print(f"  ||W' - W||_F = {delta_norm:.6f}")
    
    # 6. 参数数量对比
    print("\n6. 参数数量对比...")
    param_info = compare_parameter_count(in_dim, out_dim, rank)
    print(f"  原始权重参数数量: {param_info['original_params']:,}")
    print(f"  LoRA参数数量: {param_info['lora_params']:,} (A + B)")
    print(f"  参数减少比例: {param_info['param_reduction']:.1f}%")
    
    # 7. 内存使用对比
    print("\n7. 内存使用对比模拟...")
    # 使用当前维度计算理论内存使用
    memory_info = simulate_lora_training(in_dim, out_dim, rank)
    print(f"  原始全参数微调可训练参数内存: {memory_info['original_memory_mb']:.2f} MB")
    print(f"  LoRA微调可训练参数内存: {memory_info['lora_memory_mb']:.2f} MB")
    print(f"  可训练参数内存节省: {memory_info['memory_reduction']:.1f}%")
    print("  注：实际使用中还需考虑冻结权重的内存占用，但无需为其存储梯度")
    
    # 8. 计算效率对比
    print("\n8. 计算效率对比...")
    # 使用中等维度进行基准测试，以展示LoRA在批量数据上的优势
    test_dim = 512
    bench_info = benchmark_computation(test_dim, test_dim, rank)
    print(f"  原始方法前向传播时间: {bench_info['original_time']*1000:.2f} ms")
    print(f"  LoRA优化前向传播时间: {bench_info['lora_time']*1000:.2f} ms")
    print(f"  LoRA进一步优化时间: {bench_info['optimized_time']*1000:.2f} ms")
    print(f"  最佳加速比: {bench_info['speedup']:.2f}x")
    
    # 9. 总结
    print("\n9. LoRA优势总结:")
    print(f"  Rank={rank}, 参数减少: {param_info['param_reduction']:.1f}%")
    print(f"  ||W'-W||_F = {delta_norm:.6f}")
    print(f"  可训练参数内存节省: {memory_info['memory_reduction']:.1f}%")
    print(f"  计算加速: {bench_info['speedup']:.2f}x")
    
    print("\n===== LoRA 权重更新逻辑模拟演示完成 =====")
    print("\nLoRA的核心优势:")
    print("1. 参数效率高: 通过低秩分解大幅减少可训练参数")
    print("2. 内存占用低: 仅需存储和更新低秩矩阵A和B的梯度")
    print("3. 计算高效: 前向传播可以优化为 A×(B×x) 避免存储完整的delta_W")
    print("4. 可插拔性: 训练完成后可以将A×B合并到原始权重中")
    print("5. 多个LoRA适配器可以共享同一基础模型")


def main():
    """
    主函数：协调整个演示流程
    """
    # 检查依赖
    check_and_install_dependencies()
    
    # 运行不同秩的演示
    print("\n" + "="*60)
    print("运行不同秩（rank）的LoRA模拟对比")
    print("="*60)
    
    ranks = [4, 8, 16, 32]
    results = []
    
    # 使用较小的维度以加快演示速度
    in_dim, out_dim = 256, 256
    
    for rank in ranks:
        print(f"\n\n演示LoRA秩 = {rank}:")
        
        # 生成权重和LoRA矩阵
        W = generate_weights(in_dim, out_dim)
        A, B = initialize_lora_matrices(in_dim, out_dim, rank)
        
        # 计算参数减少比例
        param_info = compare_parameter_count(in_dim, out_dim, rank)
        
        # 计算矩阵差分范数
        delta_W = calculate_lora_update(A, B)
        delta_norm = calculate_frobenius_norm(delta_W)
        
        # 保存结果
        results.append({
            "rank": rank,
            "param_reduction": param_info["param_reduction"],
            "delta_norm": delta_norm
        })
        
        # 打印结果
        print(f"  Rank={rank}, 参数减少: {param_info['param_reduction']:.1f}%")
        print(f"  ||W'-W||_F = {delta_norm:.6f}")
    
    # 运行完整演示
    print("\n\n" + "="*60)
    print("运行完整LoRA演示")
    print("="*60)
    run_demo(in_dim=1024, out_dim=1024, rank=8)


if __name__ == "__main__":
    main()