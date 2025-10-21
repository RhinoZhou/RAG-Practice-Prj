#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Token 缓存与批处理推理简化演示

功能说明：
    演示批处理与缓存机制减少重复计算的逻辑
    构造递归文本生成函数，记录各步缓存命中率
    用以说明 KV Cache 的性能效果

作者：Ph.D. Rhino

执行流程：
    1. 接收数条推理请求
    2. 检查缓存字典，命中则跳过再计算
    3. 批量执行生成函数
    4. 统计平均响应时延
    5. 输出前后性能对比

输入说明：
    模拟多句查询文本，包含相似前缀和不同后缀的请求

输出展示：
    缓存命中率统计
    使用缓存前后的性能对比
    批处理效果分析

技术要点：
    - 模拟KV Cache机制，缓存中间计算结果
    - 实现批处理请求合并与并行处理
    - 精确计时统计不同策略的性能差异
    - 可视化展示缓存命中率和性能提升
"""

import os
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict


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


class TokenGenerationSimulator:
    """
    Token生成模拟器，模拟大语言模型的推理过程
    """
    
    def __init__(self):
        # 初始化缓存字典，模拟KV Cache
        self.kv_cache = {}
        # 重置统计信息
        self.reset_stats()
        # 设置模拟参数
        self.base_token_time = 0.02  # 基础token生成时间（秒）
        self.context_scale = 0.001   # 上下文长度对时间的影响系数
        
    def reset_stats(self):
        """
        重置性能统计信息
        """
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_computation_time = 0
        
    def generate_tokens_without_cache(self, text: str, max_new_tokens: int = 10) -> Tuple[str, float]:
        """
        不使用缓存生成文本（模拟传统推理方式）
        
        Args:
            text: 输入文本
            max_new_tokens: 要生成的最大token数
            
        Returns:
            (生成的文本, 处理时间)
        """
        start_time = time.time()
        
        # 模拟完整计算过程，不使用任何缓存
        input_tokens = len(text.split())  # 简单模拟token数量
        total_computation = 0
        
        # 模拟每一步都进行完整计算
        for i in range(max_new_tokens):
            # 计算当前token的处理时间（与上下文长度相关）
            current_context = input_tokens + i
            token_time = self.base_token_time + (current_context * self.context_scale)
            
            # 模拟计算延迟
            time.sleep(token_time * 0.1)  # 实际延迟乘以0.1以加速演示
            total_computation += token_time
            
            # 更新统计信息
            self.total_tokens += 1
            self.cache_misses += 1
        
        end_time = time.time()
        actual_time = end_time - start_time
        self.total_computation_time += actual_time
        
        # 模拟生成的文本（简单重复输入的最后几个字符）
        words = text.split()
        if words:
            suffix = ' '.join(words[-2:]) + '...' if len(words) >= 2 else text + '...'
        else:
            suffix = text + '...'
        
        generated_text = text + " " + suffix * (max_new_tokens // 3)
        
        return generated_text[:100], actual_time  # 限制返回长度
    
    def generate_tokens_with_cache(self, text: str, max_new_tokens: int = 10) -> Tuple[str, float]:
        """
        使用KV Cache生成文本（模拟优化的推理方式）
        
        Args:
            text: 输入文本
            max_new_tokens: 要生成的最大token数
            
        Returns:
            (生成的文本, 处理时间)
        """
        start_time = time.time()
        
        # 分割文本为token序列
        input_tokens = text.split()
        num_input_tokens = len(input_tokens)
        
        # 查找缓存命中情况
        cache_key_parts = []
        cache_hits_in_request = 0
        
        # 尝试从最长前缀开始匹配缓存
        for i in range(num_input_tokens, 0, -1):
            prefix_key = ' '.join(input_tokens[:i])
            if prefix_key in self.kv_cache:
                # 找到缓存命中
                cache_key_parts = prefix_key.split()
                cache_hits_in_request = len(cache_key_parts)
                self.cache_hits += cache_hits_in_request
                self.total_tokens += cache_hits_in_request
                break
        
        # 计算未命中部分的处理时间
        new_tokens_needed = num_input_tokens - cache_hits_in_request
        total_computation = 0
        
        # 处理未命中的输入token
        if new_tokens_needed > 0:
            for i in range(new_tokens_needed):
                current_context = len(cache_key_parts) + i
                token_time = self.base_token_time + (current_context * self.context_scale)
                time.sleep(token_time * 0.1)  # 模拟延迟
                total_computation += token_time
                self.cache_misses += 1
                self.total_tokens += 1
                
            # 更新缓存
            new_cache_key = ' '.join(input_tokens[:num_input_tokens])
            self.kv_cache[new_cache_key] = True
        
        # 生成新token
        for i in range(max_new_tokens):
            current_context = num_input_tokens + i
            token_time = self.base_token_time + (current_context * self.context_scale)
            time.sleep(token_time * 0.1)  # 模拟延迟
            total_computation += token_time
            self.total_tokens += 1
        
        end_time = time.time()
        actual_time = end_time - start_time
        self.total_computation_time += actual_time
        
        # 模拟生成的文本
        words = text.split()
        if words:
            suffix = ' '.join(words[-2:]) + '...' if len(words) >= 2 else text + '...'
        else:
            suffix = text + '...'
        
        generated_text = text + " " + suffix * (max_new_tokens // 3)
        
        return generated_text[:100], actual_time
    
    def batch_generate_with_cache(self, texts: List[str], max_new_tokens: int = 10) -> Tuple[List[Tuple[str, float]], float]:
        """
        批量处理多个文本生成请求，使用缓存机制
        
        Args:
            texts: 文本列表
            max_new_tokens: 每个文本要生成的最大token数
            
        Returns:
            (生成结果列表, 总处理时间)
        """
        start_time = time.time()
        results = []
        
        # 按文本长度排序，优化批处理效率（相似长度的文本通常有相似的计算需求）
        sorted_texts = sorted(texts, key=len)
        
        # 处理每个文本
        for text in sorted_texts:
            result, _ = self.generate_tokens_with_cache(text, max_new_tokens)
            results.append((result, 0))  # 时间将在最后计算
        
        end_time = time.time()
        total_batch_time = end_time - start_time
        
        # 计算平均处理时间
        avg_time_per_request = total_batch_time / len(texts)
        results = [(text, avg_time_per_request) for text, _ in results]
        
        return results, total_batch_time
    
    def get_cache_hit_ratio(self) -> float:
        """
        计算缓存命中率
        
        Returns:
            缓存命中率（0-1之间的浮点数）
        """
        if self.total_tokens == 0:
            return 0.0
        return self.cache_hits / self.total_tokens


def generate_sample_queries(num_queries: int = 20) -> List[str]:
    """
    生成样本查询文本，包含共享前缀和不同后缀
    
    Args:
        num_queries: 生成的查询数量
        
    Returns:
        查询文本列表
    """
    print(f"生成{num_queries}个样本查询文本...")
    
    # 定义一些前缀和后缀
    prefixes = [
        "如何在Python中",
        "用PyTorch实现",
        "机器学习中的",
        "深度学习模型",
        "自然语言处理中"
    ]
    
    suffixes = [
        "优化性能",
        "处理大数据",
        "提高准确率",
        "减少内存使用",
        "加速训练过程",
        "实现自定义层",
        "处理分类问题",
        "进行回归分析",
        "处理时序数据",
        "实现注意力机制"
    ]
    
    queries = []
    
    # 生成查询，确保有一定比例的相似查询
    for i in range(num_queries):
        # 70%的查询使用相同前缀，增加缓存命中率
        if i < num_queries * 0.7:
            prefix = random.choice(prefixes[:3])  # 从常用前缀中选择
        else:
            prefix = random.choice(prefixes)
        
        suffix = random.choice(suffixes)
        query = prefix + " " + suffix
        queries.append(query)
    
    # 显示部分生成的查询
    print("示例查询:")
    for query in queries[:5]:
        print(f"  - {query}")
    print(f"  ... 等共{num_queries}条查询")
    
    return queries


def run_performance_test():
    """
    运行性能测试，比较有无缓存和批处理的性能差异
    """
    print("\n===== 开始性能测试 =====")
    
    # 创建模拟器实例
    simulator = TokenGenerationSimulator()
    
    # 生成样本查询
    queries = generate_sample_queries(num_queries=30)
    max_new_tokens = 15
    
    print(f"\n测试参数:")
    print(f"- 查询数量: {len(queries)}")
    print(f"- 每个查询生成的最大token数: {max_new_tokens}")
    print(f"- 基础token生成时间: {simulator.base_token_time}秒")
    print(f"- 上下文长度影响系数: {simulator.context_scale}")
    
    # 1. 测试不使用缓存的性能
    print("\n1. 测试不使用缓存的推理性能...")
    simulator.reset_stats()
    start_time = time.time()
    
    results_without_cache = []
    for i, query in enumerate(queries):
        result, _ = simulator.generate_tokens_without_cache(query, max_new_tokens)
        results_without_cache.append(result)
        # 每5个查询显示进度
        if (i + 1) % 5 == 0:
            print(f"  已处理 {i + 1}/{len(queries)} 个查询")
    
    total_time_without_cache = time.time() - start_time
    avg_time_without_cache = total_time_without_cache / len(queries)
    
    print(f"  总处理时间: {total_time_without_cache:.2f}秒")
    print(f"  平均每个查询时间: {avg_time_without_cache:.4f}秒")
    print(f"  总token数: {simulator.total_tokens}")
    
    # 2. 测试使用缓存但不批处理的性能
    print("\n2. 测试使用缓存但不批处理的推理性能...")
    simulator.reset_stats()
    start_time = time.time()
    
    results_with_cache = []
    for i, query in enumerate(queries):
        result, _ = simulator.generate_tokens_with_cache(query, max_new_tokens)
        results_with_cache.append(result)
        # 每5个查询显示进度
        if (i + 1) % 5 == 0:
            print(f"  已处理 {i + 1}/{len(queries)} 个查询")
    
    total_time_with_cache = time.time() - start_time
    avg_time_with_cache = total_time_with_cache / len(queries)
    cache_hit_ratio = simulator.get_cache_hit_ratio()
    
    print(f"  总处理时间: {total_time_with_cache:.2f}秒")
    print(f"  平均每个查询时间: {avg_time_with_cache:.4f}秒")
    print(f"  缓存命中率: {cache_hit_ratio:.2%}")
    print(f"  总缓存命中次数: {simulator.cache_hits}")
    print(f"  总缓存未命中次数: {simulator.cache_misses}")
    
    # 3. 测试使用缓存和批处理的性能
    print("\n3. 测试使用缓存和批处理的推理性能...")
    simulator.reset_stats()
    start_time = time.time()
    
    # 按批次处理查询，每批6个
    batch_size = 6
    results_with_batch = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_results, _ = simulator.batch_generate_with_cache(batch_queries, max_new_tokens)
        results_with_batch.extend(batch_results)
        print(f"  已处理批次 {i//batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}")
    
    total_time_with_batch = time.time() - start_time
    avg_time_with_batch = total_time_with_batch / len(queries)
    cache_hit_ratio_batch = simulator.get_cache_hit_ratio()
    
    print(f"  总处理时间: {total_time_with_batch:.2f}秒")
    print(f"  平均每个查询时间: {avg_time_with_batch:.4f}秒")
    print(f"  缓存命中率: {cache_hit_ratio_batch:.2%}")
    print(f"  总缓存命中次数: {simulator.cache_hits}")
    print(f"  总缓存未命中次数: {simulator.cache_misses}")
    
    # 计算性能提升
    time_reduction_cache = (1 - total_time_with_cache / total_time_without_cache) * 100
    time_reduction_batch = (1 - total_time_with_batch / total_time_without_cache) * 100
    latency_reduction_cache = avg_time_without_cache - avg_time_with_cache
    latency_reduction_batch = avg_time_without_cache - avg_time_with_batch
    
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print("\ncsharp")
    print(f"Cache hit: {cache_hit_ratio_batch:.0%}")
    print(f"Average latency reduced from {avg_time_without_cache:.1f}s → {avg_time_with_batch:.1f}s")
    print()
    print(f"性能提升:")
    print(f"- 仅使用缓存: 时间减少 {time_reduction_cache:.1f}%，平均延迟减少 {latency_reduction_cache:.3f}秒")
    print(f"- 缓存+批处理: 时间减少 {time_reduction_batch:.1f}%，平均延迟减少 {latency_reduction_batch:.3f}秒")
    print()
    print(f"批处理加速比: {avg_time_without_cache / avg_time_with_batch:.2f}x")
    print()
    
    # 生成可视化图表
    visualize_results(
        avg_time_without_cache, 
        avg_time_with_cache, 
        avg_time_with_batch,
        cache_hit_ratio, 
        cache_hit_ratio_batch
    )
    
    # 返回测试结果用于后续分析
    return {
        "without_cache": {
            "total_time": total_time_without_cache,
            "avg_time": avg_time_without_cache
        },
        "with_cache": {
            "total_time": total_time_with_cache,
            "avg_time": avg_time_with_cache,
            "hit_ratio": cache_hit_ratio
        },
        "with_cache_and_batch": {
            "total_time": total_time_with_batch,
            "avg_time": avg_time_with_batch,
            "hit_ratio": cache_hit_ratio_batch
        }
    }

def visualize_results(
    time_without_cache: float, 
    time_with_cache: float, 
    time_with_batch: float,
    hit_ratio: float, 
    hit_ratio_batch: float
):
    """
    可视化性能测试结果
    
    Args:
        time_without_cache: 不使用缓存的平均时间
        time_with_cache: 使用缓存但不批处理的平均时间
        time_with_batch: 使用缓存和批处理的平均时间
        hit_ratio: 使用缓存但不批处理的命中率
        hit_ratio_batch: 使用缓存和批处理的命中率
    """
    try:
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制平均延迟对比图
        labels = ['无缓存', '仅使用缓存', '缓存+批处理']
        times = [time_without_cache, time_with_cache, time_with_batch]
        
        ax1.bar(labels, times, color=['red', 'orange', 'green'])
        ax1.set_title('平均推理延迟对比')
        ax1.set_ylabel('平均时间 (秒)')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(times):
            ax1.text(i, v + 0.01, f'{v:.3f}s', ha='center')
        
        # 绘制缓存命中率对比图
        hit_labels = ['仅使用缓存', '缓存+批处理']
        hit_ratios = [hit_ratio, hit_ratio_batch]
        
        ax2.bar(hit_labels, hit_ratios, color=['orange', 'green'])
        ax2.set_title('缓存命中率对比')
        ax2.set_ylabel('命中率')
        ax2.set_ylim(0, 1)  # 设置y轴范围为0-1
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱状图上添加百分比标签
        for i, v in enumerate(hit_ratios):
            ax2.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('inference_cache_performance.png', dpi=300, bbox_inches='tight')
        print("\n性能对比可视化已保存为 'inference_cache_performance.png'")
        
        # 关闭图像以释放内存
        plt.close()
        
    except Exception as e:
        print(f"\n可视化过程中出现错误: {e}")
        print("跳过可视化步骤，但性能测试结果仍然可用")

def analyze_kv_cache_benefits():
    """
    分析KV Cache在不同场景下的优势
    """
    print("\n" + "=" * 60)
    print("KV Cache 优势分析")
    print("=" * 60)
    
    print("\n1. 工作原理:")
    print("   - KV Cache (Key-Value Cache) 存储注意力机制计算中的中间结果")
    print("   - 在自回归生成过程中，避免对已经处理过的token重复计算")
    print("   - 通过保存Key和Value矩阵，在生成长文本时显著加速")
    
    print("\n2. 性能优势:")
    print("   - 推理速度: 缓存命中时计算时间减少90%以上")
    print("   - 内存使用: 虽然增加了额外缓存，但总体计算效率大幅提升")
    print("   - 吞吐量: 批处理与缓存结合可提高系统整体吞吐量")
    
    print("\n3. 适用场景:")
    print("   - 连续对话: 相同对话历史的后续请求可充分利用缓存")
    print("   - 相似查询: 前缀相同的查询可共享计算结果")
    print("   - 长文本生成: 生成长文本时每步只计算新增token")
    
    print("\n4. 最佳实践:")
    print("   - 缓存大小: 根据模型规模和服务内存设置合理缓存大小")
    print("   - 批处理优化: 相似长度文本批处理效率更高")
    print("   - 缓存失效策略: 实现LRU等策略管理缓存生命周期")


def main():
    """
    主函数
    """
    print("===== Token 缓存与批处理推理简化演示 =====")
    
    # 检查依赖
    check_and_install_dependencies()
    
    # 运行性能测试
    results = run_performance_test()
    
    # 分析KV Cache优势
    analyze_kv_cache_benefits()
    
    print("\n===== 演示程序完成 =====")


if __name__ == "__main__":
    main()