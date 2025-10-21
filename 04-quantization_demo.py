#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-bit 权重量化与恢复示例

功能说明：
    模拟 32bit → 4bit → 恢复 的权重量化过程
    采用 Python 标准库整数映射展示量化公式，使学员理解浮点值如何以较低精度存储。

作者：Ph.D. Rhino

执行流程：
    1. 检查并安装必要的依赖
    2. 生成模拟浮点权重数组
    3. 执行线性量化：quantized = round((x - min) / scale)
    4. 反量化恢复为近似浮点数
    5. 计算误差均方差 MSE
    6. 输出内存压缩比与误差值

输入说明：
    程序会自动生成随机权重集合

输出展示：
    - 量化位数信息
    - 内存压缩比
    - 均方误差 (MSE)
    - 原始值与量化恢复值的对比
"""

import os
import math
import random
import struct
from typing import Tuple, List, Dict, Any
import numpy as np


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    需要安装：numpy
    """
    print("正在检查依赖...")
    
    # 需要的依赖包列表
    required_packages = ["numpy"]
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


def generate_weights(size: int = 10000, seed: int = 42) -> np.ndarray:
    """
    生成模拟的浮点权重数组
    
    Args:
        size: 数组大小
        seed: 随机种子
    
    Returns:
        浮点权重数组
    """
    print(f"正在生成{size}个模拟浮点权重...")
    
    # 设置随机种子以保证结果可复现
    np.random.seed(seed)
    
    # 生成权重，模拟神经网络中的权重分布
    # 使用截断正态分布，更接近实际权重分布
    weights = np.random.normal(0, 0.1, size)
    
    # 统计信息
    print(f"  权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"  权重均值: {weights.mean():.6f}")
    print(f"  权重标准差: {weights.std():.6f}")
    
    return weights


def linear_quantization(weights: np.ndarray, bits: int = 4) -> Tuple[np.ndarray, float, float]:
    """
    执行线性量化
    
    Args:
        weights: 原始浮点权重
        bits: 量化位数
    
    Returns:
        (量化后整数数组, 量化缩放因子, 最小值)
    """
    print(f"\n执行{bits}-bit线性量化...")
    
    # 计算权重的最小值和最大值
    min_val = weights.min()
    max_val = weights.max()
    
    # 计算量化范围
    # 对于n位量化，可以表示的整数值范围是 [0, 2^n - 1]
    q_min = 0
    q_max = 2 ** bits - 1
    
    # 计算缩放因子 (scale)
    # 避免除以零的情况
    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / (q_max - q_min)
    
    # 执行量化: quantized = round((x - min_val) / scale)
    # 将浮点数映射到整数范围
    quantized = np.round((weights - min_val) / scale)
    
    # 确保量化后的值在有效范围内
    quantized = np.clip(quantized, q_min, q_max).astype(np.int32)
    
    print(f"  量化参数:")
    print(f"    原始范围: [{min_val:.6f}, {max_val:.6f}]")
    print(f"    量化范围: [{q_min}, {q_max}]")
    print(f"    缩放因子: {scale:.6f}")
    print(f"    量化后样本: {quantized[:5]}...")
    
    return quantized, scale, min_val


def dequantization(quantized: np.ndarray, scale: float, min_val: float) -> np.ndarray:
    """
    执行反量化，恢复为近似浮点数
    
    Args:
        quantized: 量化后的整数数组
        scale: 量化缩放因子
        min_val: 原始最小值
    
    Returns:
        恢复后的近似浮点数数组
    """
    print("\n执行反量化恢复...")
    
    # 反量化公式: x ≈ quantized * scale + min_val
    restored = quantized.astype(np.float32) * scale + min_val
    
    print(f"  反量化样本: {restored[:5]}...")
    
    return restored


def calculate_metrics(original: np.ndarray, restored: np.ndarray) -> Dict[str, float]:
    """
    计算量化前后的误差指标
    
    Args:
        original: 原始浮点权重
        restored: 恢复后的近似浮点数
    
    Returns:
        包含各种误差指标的字典
    """
    print("\n计算量化误差指标...")
    
    # 计算误差
    error = original - restored
    
    # 计算均方误差 (MSE)
    mse = np.mean(error ** 2)
    
    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(mse)
    
    # 计算平均绝对误差 (MAE)
    mae = np.mean(np.abs(error))
    
    # 计算最大绝对误差
    max_abs_error = np.max(np.abs(error))
    
    # 计算信噪比 (SNR)，避免除以零
    signal_power = np.mean(original ** 2)
    if signal_power > 0:
        snr = 10 * np.log10(signal_power / mse)
    else:
        snr = float('inf')
    
    # 计算相对误差
    relative_error = np.mean(np.abs(error) / (np.abs(original) + 1e-10)) * 100  # 百分比
    
    print(f"  均方误差 (MSE): {mse:.6f}")
    print(f"  均方根误差 (RMSE): {rmse:.6f}")
    print(f"  平均绝对误差 (MAE): {mae:.6f}")
    print(f"  最大绝对误差: {max_abs_error:.6f}")
    print(f"  信噪比 (SNR): {snr:.2f} dB" if snr != float('inf') else "  信噪比 (SNR): ∞ dB")
    print(f"  平均相对误差: {relative_error:.4f}%")
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "snr": snr,
        "relative_error": relative_error
    }


def calculate_memory_savings(original_size: int, bits: int) -> Dict[str, float]:
    """
    计算内存节省情况
    
    Args:
        original_size: 原始数组元素数量
        bits: 量化位数
    
    Returns:
        内存节省信息
    """
    # 假设原始数据是32位浮点数 (4字节)
    original_bits = 32
    
    # 计算原始内存使用量 (字节)
    original_memory = original_size * (original_bits / 8)
    
    # 计算量化后内存使用量 (字节)
    quantized_memory = original_size * (bits / 8)
    
    # 加上元数据开销 (min_val和scale各需要4字节)
    metadata_overhead = 8  # 2个32位浮点数
    
    # 总量化内存使用量
    total_quantized_memory = quantized_memory + metadata_overhead
    
    # 计算压缩比
    compression_ratio = original_memory / total_quantized_memory
    
    # 计算内存节省百分比
    memory_saved_percent = (1 - total_quantized_memory / original_memory) * 100
    
    print("\n内存使用分析:")
    print(f"  原始数据: {original_memory:.2f} 字节 ({original_bits}-bit 浮点)")
    print(f"  量化数据: {quantized_memory:.2f} 字节 ({bits}-bit 整数)")
    print(f"  元数据开销: {metadata_overhead} 字节 (min_val + scale)")
    print(f"  总量化内存: {total_quantized_memory:.2f} 字节")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  内存节省: {memory_saved_percent:.1f}%")
    
    return {
        "original_memory": original_memory,
        "quantized_memory": quantized_memory,
        "total_quantized_memory": total_quantized_memory,
        "compression_ratio": compression_ratio,
        "memory_saved_percent": memory_saved_percent
    }


def compare_values(original: np.ndarray, restored: np.ndarray, num_samples: int = 10) -> None:
    """
    比较原始值和恢复值的样本
    
    Args:
        original: 原始浮点权重
        restored: 恢复后的近似浮点数
        num_samples: 显示的样本数量
    """
    print(f"\n原始值与恢复值对比 (前{num_samples}个样本):")
    print("  原始值       恢复值       误差")
    print("  -----------  -----------  -----------")
    
    # 随机选择一些样本进行显示
    sample_indices = np.random.choice(len(original), min(num_samples, len(original)), replace=False)
    
    for idx in sample_indices:
        orig_val = original[idx]
        rest_val = restored[idx]
        err_val = orig_val - rest_val
        print(f"  {orig_val:11.6f}  {rest_val:11.6f}  {err_val:11.6f}")


def simulate_bit_operations_demo():
    """
    演示底层位操作的量化过程
    展示单个浮点数如何被量化为4位整数
    """
    print("\n" + "="*60)
    print("位操作量化过程演示")
    print("="*60)
    
    # 选择一个示例浮点数
    example_float = 0.078125  # 这个值能被精确表示
    print(f"\n示例浮点数: {example_float}")
    
    # 定义量化参数
    bits = 4
    min_val = -0.1
    max_val = 0.1
    q_min = 0
    q_max = 2**bits - 1
    
    # 计算缩放因子
    scale = (max_val - min_val) / (q_max - q_min)
    
    print(f"量化参数:")
    print(f"  位数: {bits} bits")
    print(f"  范围: [{min_val}, {max_val}]")
    print(f"  量化整数范围: [{q_min}, {q_max}]")
    print(f"  缩放因子: {scale}")
    
    # 手动执行量化步骤
    print("\n量化步骤:")
    step1 = example_float - min_val
    step2 = step1 / scale
    step3 = round(step2)
    quantized = max(q_min, min(q_max, step3))
    
    print(f"  1. 偏移: {example_float} - ({min_val}) = {step1}")
    print(f"  2. 缩放: {step1} / {scale} = {step2}")
    print(f"  3. 取整: round({step2}) = {step3}")
    print(f"  4. 截断: max({q_min}, min({q_max}, {step3})) = {quantized}")
    print(f"  量化结果: {quantized} (二进制: {quantized:04b})")
    
    # 手动执行反量化步骤
    print("\n反量化步骤:")
    step4 = quantized * scale
    step5 = step4 + min_val
    
    print(f"  1. 缩放: {quantized} * {scale} = {step4}")
    print(f"  2. 偏移: {step4} + ({min_val}) = {step5}")
    print(f"  反量化结果: {step5}")
    print(f"  原始值: {example_float}")
    print(f"  误差: {example_float - step5}")


def run_demo_with_different_bits():
    """
    使用不同的量化位数进行对比演示
    """
    print("\n" + "="*60)
    print("不同量化位数对比")
    print("="*60)
    
    # 生成测试数据
    test_weights = generate_weights(size=10000)
    
    # 测试不同的量化位数
    bits_list = [8, 4, 2, 1]
    results = []
    
    for bits in bits_list:
        print(f"\n\n测试量化位数: {bits} bit")
        
        # 执行量化
        quantized, scale, min_val = linear_quantization(test_weights, bits=bits)
        
        # 执行反量化
        restored = dequantization(quantized, scale, min_val)
        
        # 计算误差指标
        metrics = calculate_metrics(test_weights, restored)
        
        # 计算内存节省
        memory_info = calculate_memory_savings(len(test_weights), bits)
        
        # 保存结果
        results.append({
            "bits": bits,
            "mse": metrics["mse"],
            "compression_ratio": memory_info["compression_ratio"],
            "relative_error": metrics["relative_error"]
        })
    
    # 展示对比结果
    print("\n" + "="*60)
    print("量化位数对比结果")
    print("="*60)
    print(f"{'位数':<5}{'压缩比':<10}{'均方误差(MSE)':<20}{'相对误差(%)':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['bits']:<5}{result['compression_ratio']:<10.2f}{result['mse']:<20.6f}{result['relative_error']:<15.4f}")


def main():
    """
    主函数：协调整个量化演示流程
    """
    print("===== 4-bit 权重量化与恢复示例程序 =====")
    
    # 检查依赖
    check_and_install_dependencies()
    
    # 生成模拟权重
    weights = generate_weights(size=10000)
    
    # 执行4-bit量化
    bits = 4
    quantized, scale, min_val = linear_quantization(weights, bits=bits)
    
    # 执行反量化
    restored = dequantization(quantized, scale, min_val)
    
    # 计算误差指标
    metrics = calculate_metrics(weights, restored)
    
    # 计算内存节省
    memory_info = calculate_memory_savings(len(weights), bits)
    
    # 对比原始值和恢复值
    compare_values(weights, restored)
    
    # 演示底层位操作
    simulate_bit_operations_demo()
    
    # 测试不同量化位数
    run_demo_with_different_bits()
    
    # 总结
    print("\n" + "="*60)
    print("量化总结")
    print("="*60)
    print(f"量化位数: {bits} bit")
    print(f"压缩比: {memory_info['compression_ratio']:.2f}x")
    print(f"均方误差 (MSE): {metrics['mse']:.6f}")
    print(f"内存节省: {memory_info['memory_saved_percent']:.1f}%")
    print("\n量化核心公式:")
    print("  量化: quantized = round((x - min_val) / scale)")
    print("  反量化: x ≈ quantized * scale + min_val")
    print("  其中: scale = (max_val - min_val) / (2^bits - 1)")
    
    print("\n===== 4-bit 权重量化与恢复示例程序完成 =====")


if __name__ == "__main__":
    main()