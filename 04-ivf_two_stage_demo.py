#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF两阶段检索演示程序

作者：Ph.D. Rhino

功能说明：
此程序实现了基于随机簇心的倒排文件索引(IVF)的两阶段检索方法。
第一阶段使用随机初始化的簇心对向量进行分区，
第二阶段查询时通过探测nprobe个最近簇来获取候选向量，最后对候选向量进行余弦相似度精排并返回Top-K结果。

执行流程：
1. 初始化2D玩具数据和随机簇心
2. 将样本向量分配到最近的簇中
3. 查询时选择nprobe个最近的簇
4. 对选中簇中的所有向量进行余弦相似度精排
5. 返回Top-K结果

可调参数：
- nlist: 簇的数量，默认为8
- nprobe: 查询时探测的簇数量，默认为2
- topk: 返回的结果数量，默认为5
- dim: 向量维度，默认为2
- total_vectors: 总向量数量，默认为100

依赖项：
- numpy: 用于数值计算和向量操作
- matplotlib: 可选，用于可视化结果

注意事项：
此程序使用随机数据生成和随机簇心初始化，每次运行结果可能略有不同。
"""

import os
import sys
import time
import random
import math
import subprocess
from typing import List, Tuple, Dict, Any

# 检查并安装依赖
def check_dependencies():
    """检查必要的依赖包是否已安装，若未安装则自动安装"""
    required_packages = ['numpy']
    optional_packages = ['matplotlib']
    
    # 安装必要依赖
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装必要依赖包: {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # 尝试安装可选依赖
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装可选依赖包: {package}")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except:
                print(f"可选依赖包 {package} 安装失败，可视化功能将不可用")

# 导入依赖
check_dependencies()
import numpy as np

# 尝试导入matplotlib用于可视化
HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass

class IVFTwoStageDemo:
    """IVF两阶段检索演示类"""
    
    def __init__(self, nlist: int = 8, nprobe: int = 2, topk: int = 5, dim: int = 2, total_vectors: int = 100):
        """
        初始化IVF两阶段检索演示
        
        参数:
        nlist: 簇的数量
        nprobe: 查询时探测的簇数量
        topk: 返回的结果数量
        dim: 向量维度
        total_vectors: 总向量数量
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.topk = topk
        self.dim = dim
        self.total_vectors = total_vectors
        
        # 初始化数据和簇
        self.vectors = None  # 所有向量
        self.cluster_centroids = None  # 簇心
        self.inverted_index = None  # 倒排索引，记录每个簇包含的向量ID
        
        # 生成随机数据
        self._generate_random_data()
        
        # 初始化随机簇心
        self._initialize_random_centroids()
        
        # 构建倒排索引
        self._build_inverted_index()
    
    def _generate_random_data(self):
        """生成随机的2D向量数据"""
        # 生成随机向量，范围在[-1, 1]之间
        self.vectors = np.random.uniform(-1, 1, (self.total_vectors, self.dim))
        
        # 对向量进行归一化，使余弦相似度计算更直观
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        # 避免除零错误
        norms[norms == 0] = 1
        self.vectors = self.vectors / norms
    
    def _initialize_random_centroids(self):
        """随机初始化簇心"""
        # 从现有向量中随机选择nlist个作为簇心
        # 确保选择的簇心数量不超过总向量数
        nlist = min(self.nlist, self.total_vectors)
        random_indices = np.random.choice(self.total_vectors, nlist, replace=False)
        self.cluster_centroids = self.vectors[random_indices].copy()
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        # 使用numpy的点积函数计算余弦相似度
        # 由于向量已经归一化，点积直接等于余弦相似度
        return np.dot(vec1, vec2)
    
    def _find_nearest_centroid(self, vector: np.ndarray) -> int:
        """找到与给定向量最近的簇心"""
        max_similarity = -1
        nearest_centroid_idx = 0
        
        for i, centroid in enumerate(self.cluster_centroids):
            similarity = self._calculate_cosine_similarity(vector, centroid)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_centroid_idx = i
        
        return nearest_centroid_idx
    
    def _build_inverted_index(self):
        """构建倒排索引，将向量分配到最近的簇中"""
        # 初始化倒排索引
        self.inverted_index = {i: [] for i in range(len(self.cluster_centroids))}
        
        # 将每个向量分配到最近的簇中
        for i, vector in enumerate(self.vectors):
            centroid_idx = self._find_nearest_centroid(vector)
            self.inverted_index[centroid_idx].append(i)
    
    def _find_nearest_centroids(self, query: np.ndarray, k: int) -> List[int]:
        """找到与查询向量最近的k个簇心"""
        similarities = []
        
        for i, centroid in enumerate(self.cluster_centroids):
            similarity = self._calculate_cosine_similarity(query, centroid)
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个簇的索引
        return [idx for idx, _ in similarities[:k]]
    
    def search(self, query: np.ndarray) -> List[Tuple[int, float]]:
        """
        执行IVF两阶段检索
        
        参数:
        query: 查询向量
        
        返回:
        List[Tuple[int, float]]: 排序后的结果列表，每个元素为(向量索引, 余弦相似度分数)
        """
        # 第一阶段：找到与查询最近的nprobe个簇
        nearest_centroids = self._find_nearest_centroids(query, self.nprobe)
        
        # 收集候选向量
        candidate_indices = set()
        for centroid_idx in nearest_centroids:
            candidate_indices.update(self.inverted_index[centroid_idx])
        
        # 第二阶段：对候选向量进行余弦相似度精排
        results = []
        for vec_idx in candidate_indices:
            similarity = self._calculate_cosine_similarity(query, self.vectors[vec_idx])
            results.append((vec_idx, similarity))
        
        # 按相似度降序排序并返回Top-K结果
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:self.topk], len(candidate_indices)
    
    def visualize(self, query: np.ndarray, results: List[Tuple[int, float]]):
        """可视化IVF检索结果"""
        if not HAS_MATPLOTLIB:
            print("matplotlib未安装，无法进行可视化")
            return
        
        plt.figure(figsize=(10, 8))
        
        # 绘制所有向量
        for i in range(self.total_vectors):
            plt.scatter(self.vectors[i, 0], self.vectors[i, 1], c='lightblue', alpha=0.6)
        
        # 绘制簇心
        for centroid in self.cluster_centroids:
            plt.scatter(centroid[0], centroid[1], c='red', marker='x', s=100)
        
        # 绘制查询向量
        plt.scatter(query[0], query[1], c='green', marker='*', s=200, label='查询向量')
        
        # 绘制Top-K结果向量
        result_indices = [idx for idx, _ in results]
        for idx in result_indices:
            plt.scatter(self.vectors[idx, 0], self.vectors[idx, 1], c='orange', s=150, alpha=0.8)
            plt.annotate(f'{idx}', (self.vectors[idx, 0], self.vectors[idx, 1]), fontsize=12)
        
        plt.title(f'IVF检索结果 (nlist={self.nlist}, nprobe={self.nprobe})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        
        # 保存可视化结果到文件
        plt.savefig('ivf_demo_visualization.png')
        print("可视化结果已保存为 'ivf_demo_visualization.png'")
        
        # 显示图形
        plt.show()

    def run_demo(self):
        """运行IVF两阶段检索演示"""
        # 生成随机查询向量并归一化
        query = np.random.uniform(-1, 1, self.dim)
        query = query / np.linalg.norm(query)
        
        # 执行检索
        start_time = time.time()
        results, candidate_count = self.search(query)
        end_time = time.time()
        
        # 格式化输出结果
        print(f"IVF: nlist={self.nlist}, nprobe={self.nprobe}, candidates={candidate_count}, topk=[", end="")
        for i, (idx, score) in enumerate(results):
            if i > 0:
                print(", ", end="")
            print(f"(idx={idx}, score={score:.2f})", end="")
        print("]")
        
        # 输出执行时间
        execution_time = end_time - start_time
        print(f"执行时间: {execution_time:.4f}秒")
        
        # 可视化结果（如果有matplotlib）
        if HAS_MATPLOTLIB:
            self.visualize(query, results)

        return {
            'query': query,
            'results': results,
            'candidate_count': candidate_count,
            'execution_time': execution_time
        }

    def compare_parameters(self):
        """比较不同参数设置下的检索结果"""
        print("\n=== 参数比较实验 ===")
        
        # 生成固定的查询向量用于比较
        query = np.random.uniform(-1, 1, self.dim)
        query = query / np.linalg.norm(query)
        
        # 不同nprobe设置的比较
        original_nprobe = self.nprobe
        
        for nprobe in [1, 2, 4]:
            self.nprobe = nprobe
            
            # 执行检索
            start_time = time.time()
            results, candidate_count = self.search(query)
            end_time = time.time()
            
            # 输出结果
            print(f"\nnprobe={nprobe}:")
            print(f"  候选数量: {candidate_count}")
            print(f"  Top-3 结果: {results[:3]}")
            print(f"  执行时间: {(end_time - start_time):.4f}秒")
        
        # 恢复原始设置
        self.nprobe = original_nprobe

# 主函数
def main():
    """主函数"""
    # 创建IVF两阶段检索演示实例
    demo = IVFTwoStageDemo(nlist=8, nprobe=2, topk=5)
    
    # 运行基础演示
    print("=== IVF两阶段检索基础演示 ===")
    demo.run_demo()
    
    # 运行参数比较
    demo.compare_parameters()
    
    # 检查执行效率
    print("\n=== 执行效率测试 ===")
    total_time = 0
    num_runs = 10
    
    for i in range(num_runs):
        query = np.random.uniform(-1, 1, demo.dim)
        query = query / np.linalg.norm(query)
        
        start_time = time.time()
        demo.search(query)
        end_time = time.time()
        
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_runs
    print(f"平均检索时间 ({num_runs}次): {avg_time:.6f}秒")
    
    # 如果执行时间过长，尝试优化
    if avg_time > 0.1:
        print("\n注意：检索时间较长，建议优化")
        print("优化建议:")
        print("1. 减小nlist和nprobe值")
        print("2. 减少总向量数量")
        print("3. 使用更高效的相似度计算方法")

if __name__ == "__main__":
    main()