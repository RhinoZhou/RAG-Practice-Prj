#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热冷集路由检索演示程序

作者：Ph.D. Rhino

功能说明：
此程序实现了基于置信度的热集/冷集路由检索系统，主要功能包括：
1. 根据查询置信度自动路由到热集或冷集检索器
2. 支持双路径并发执行热集和冷集检索
3. 实现RRF（Reciprocal Rank Fusion）结果融合算法
4. 支持MaxSim（Maximum Similarity）去重策略

内容概述：
- 模拟两类检索器：热集检索器（低延迟、中等精度）和冷集检索器（高延迟、高精度）
- 实现单路径路由（基于置信度阈值）和双路径并发两种策略
- 提供可调节的融合参数和去重阈值
- 详细记录并展示路由路径、延迟和Top-K结果

执行流程：
1. 计算查询的置信度
2. 根据置信度和阈值决定路由策略（单路径或双路径并发）
3. 执行相应的热集/冷集检索
4. 对结果进行RRF融合
5. 应用MaxSim去重并截断到Top-K
6. 输出检索路径、延迟和最终结果

可调参数：
- confidence_threshold：置信度阈值，默认为0.7
- topk：返回结果数量，默认为5
- rerank_size：重排大小，默认为20
- maxsim_threshold：最大相似度去重阈值，默认为0.9
- rrf_k：RRF融合参数，默认为60

依赖项：
- numpy：用于数值计算
- concurrent.futures：用于实现并发执行（Python标准库）
- time：用于计时（Python标准库）
"""

import os
import sys
import time
import random
import math
import subprocess
import concurrent.futures
from typing import List, Tuple, Dict, Any

# 检查并安装依赖
def check_dependencies():
    """检查必要的依赖包是否已安装，若未安装则自动安装"""
    required_packages = ['numpy']
    
    # 安装必要依赖
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装必要依赖包: {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# 导入依赖
check_dependencies()
import numpy as np

class HotColdRoutingDemo:
    """热冷集路由检索演示类"""
    
    def __init__(self, confidence_threshold: float = 0.7, topk: int = 5,
                 rerank_size: int = 20, maxsim_threshold: float = 0.9, rrf_k: int = 60):
        """
        初始化热冷集路由检索演示
        
        参数:
        confidence_threshold: 置信度阈值，高于此值路由到热集，低于路由到冷集
        topk: 返回结果数量
        rerank_size: 重排大小
        maxsim_threshold: MaxSim去重阈值
        rrf_k: RRF融合参数
        """
        self.confidence_threshold = confidence_threshold
        self.topk = topk
        self.rerank_size = rerank_size
        self.maxsim_threshold = maxsim_threshold
        self.rrf_k = rrf_k
        
        # 模拟数据集：100个文档，每个文档有唯一ID和特征向量
        self.num_docs = 100
        self.doc_ids = [f"doc_{i}" for i in range(self.num_docs)]
        
        # 随机生成文档特征向量（用于计算相似度）
        np.random.seed(42)  # 设置随机种子以保证结果可复现
        self.doc_vectors = np.random.randn(self.num_docs, 10)  # 10维特征向量
        
        # 归一化向量以简化余弦相似度计算
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        self.doc_vectors = self.doc_vectors / norms
        
        # 模拟热集和冷集文档分布
        # 热集：包含30个高频访问文档
        self.hot_docs = set(random.sample(self.doc_ids, 30))
        # 冷集：包含所有文档
        self.cold_docs = set(self.doc_ids)
        
        # 记录热集和冷集的检索延迟（毫秒）
        self.hot_latency_base = 2.0  # 热集基础延迟
        self.cold_latency_base = 10.0  # 冷集基础延迟
    
    def _calculate_confidence(self, query: str) -> float:
        """
        计算查询的置信度
        
        参数:
        query: 查询文本
        
        返回:
        float: 置信度分数（0-1之间）
        """
        # 在实际应用中，置信度计算可能基于查询意图识别、历史点击率等
        # 这里为了演示，我们基于查询长度和某些关键词来模拟置信度
        
        # 查询长度越长，通常置信度越高
        length_factor = min(len(query) / 20, 1.0)
        
        # 包含某些特定关键词的查询通常置信度较高
        high_confidence_keywords = ["如何", "什么是", "步骤", "方法", "定义", "解释", "教程"]
        keyword_factor = 0.0
        for keyword in high_confidence_keywords:
            if keyword in query:
                keyword_factor += 0.1
        
        # 综合计算置信度（加入一些随机噪声增加真实感）
        confidence = 0.5 + 0.3 * length_factor + 0.2 * keyword_factor
        confidence += random.gauss(0, 0.05)  # 添加高斯噪声
        confidence = max(0.1, min(0.99, confidence))  # 限制在合理范围内
        
        return round(confidence, 2)
    
    def _hot_search(self, query: str) -> Tuple[List[Tuple[str, float]], float]:
        """
        模拟热集检索
        
        参数:
        query: 查询文本
        
        返回:
        Tuple[List[Tuple[str, float]], float]: (检索结果，延迟时间（毫秒）)
        """
        # 模拟检索延迟
        latency = self.hot_latency_base + random.uniform(-0.5, 0.5)
        time.sleep(latency / 1000)  # 转换为秒
        
        # 模拟生成查询向量
        query_vector = np.random.randn(10)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 只在热集中计算相似度
        hot_indices = [i for i, doc_id in enumerate(self.doc_ids) if doc_id in self.hot_docs]
        
        # 计算相似度
        similarities = []
        for idx in hot_indices:
            sim = np.dot(query_vector, self.doc_vectors[idx])
            similarities.append((self.doc_ids[idx], sim))
        
        # 排序并返回结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:self.rerank_size], latency
    
    def _cold_search(self, query: str) -> Tuple[List[Tuple[str, float]], float]:
        """
        模拟冷集检索
        
        参数:
        query: 查询文本
        
        返回:
        Tuple[List[Tuple[str, float]], float]: (检索结果，延迟时间（毫秒）)
        """
        # 模拟检索延迟
        latency = self.cold_latency_base + random.uniform(-2.0, 2.0)
        time.sleep(latency / 1000)  # 转换为秒
        
        # 模拟生成查询向量
        query_vector = np.random.randn(10)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 在所有文档中计算相似度
        similarities = []
        for i, doc_id in enumerate(self.doc_ids):
            sim = np.dot(query_vector, self.doc_vectors[i])
            similarities.append((doc_id, sim))
        
        # 排序并返回结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:self.rerank_size], latency
    
    def _rrf_fusion(self, results1: List[Tuple[str, float]], results2: List[Tuple[str, float]] = None) -> List[Tuple[str, float]]:
        """
        使用RRF（Reciprocal Rank Fusion）融合检索结果
        
        参数:
        results1: 第一组检索结果
        results2: 第二组检索结果（可选）
        
        返回:
        List[Tuple[str, float]]: 融合后的结果
        """
        # 创建结果字典
        fused_scores = {}
        
        # 处理第一组结果
        for rank, (doc_id, _) in enumerate(results1):
            # RRF公式: 1/(k + rank)，其中rank从0开始，所以加1
            fused_scores[doc_id] = 1.0 / (self.rrf_k + rank + 1)
        
        # 处理第二组结果（如果存在）
        if results2:
            for rank, (doc_id, _) in enumerate(results2):
                if doc_id in fused_scores:
                    fused_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
                else:
                    fused_scores[doc_id] = 1.0 / (self.rrf_k + rank + 1)
        
        # 转换为列表并排序
        fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _maxsim_deduplication(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        使用MaxSim（Maximum Similarity）策略去重
        
        参数:
        results: 需要去重的结果列表
        
        返回:
        List[Tuple[str, float]]: 去重后的结果
        """
        if not results:
            return []
        
        # 初始化去重后的结果列表
        dedup_results = [results[0]]
        
        for doc_id, score in results[1:]:
            # 计算与已选结果的最大相似度
            max_sim = 0.0
            doc_idx = self.doc_ids.index(doc_id)
            doc_vector = self.doc_vectors[doc_idx]
            
            for selected_doc_id, _ in dedup_results:
                selected_idx = self.doc_ids.index(selected_doc_id)
                selected_vector = self.doc_vectors[selected_idx]
                sim = np.dot(doc_vector, selected_vector)
                if sim > max_sim:
                    max_sim = sim
            
            # 如果最大相似度小于阈值，则保留该文档
            if max_sim < self.maxsim_threshold:
                dedup_results.append((doc_id, score))
                
                # 如果已经达到topk，则停止
                if len(dedup_results) >= self.topk:
                    break
        
        return dedup_results[:self.topk]
    
    def search_single_path(self, query: str) -> Dict[str, Any]:
        """
        单路径检索：根据置信度路由到热集或冷集
        
        参数:
        query: 查询文本
        
        返回:
        Dict[str, Any]: 检索结果和统计信息
        """
        # 计算置信度
        confidence = self._calculate_confidence(query)
        
        # 根据置信度决定路由路径
        if confidence >= self.confidence_threshold:
            route = "hot"
            results, latency = self._hot_search(query)
        else:
            route = "cold"
            results, latency = self._cold_search(query)
        
        # 应用MaxSim去重
        final_results = self._maxsim_deduplication(results)
        
        return {
            "route": route,
            "confidence": confidence,
            "results": final_results,
            "latency": round(latency, 2),
            "raw_results": results
        }
    
    def search_dual_path(self, query: str) -> Dict[str, Any]:
        """
        双路径并发检索：同时执行热集和冷集检索，然后融合结果
        
        参数:
        query: 查询文本
        
        返回:
        Dict[str, Any]: 检索结果和统计信息
        """
        # 计算置信度
        confidence = self._calculate_confidence(query)
        
        # 并发执行热集和冷集检索
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            hot_future = executor.submit(self._hot_search, query)
            cold_future = executor.submit(self._cold_search, query)
            
            # 获取结果
            hot_results, hot_latency = hot_future.result()
            cold_results, cold_latency = cold_future.result()
        
        # 融合结果
        fused_results = self._rrf_fusion(hot_results, cold_results)
        
        # 应用MaxSim去重
        final_results = self._maxsim_deduplication(fused_results)
        
        # 总延迟取最大值
        total_latency = max(hot_latency, cold_latency)
        
        return {
            "route": "dual",
            "confidence": confidence,
            "results": final_results,
            "latency": round(total_latency, 2),
            "hot_latency": round(hot_latency, 2),
            "cold_latency": round(cold_latency, 2),
            "hot_results": hot_results,
            "cold_results": cold_results,
            "fused_results": fused_results
        }
    
    def run_demo(self, queries: List[str] = None, use_dual_path: bool = False):
        """
        运行热冷集路由检索演示
        
        参数:
        queries: 查询列表
        use_dual_path: 是否使用双路径并发
        
        返回:
        List[Dict[str, Any]]: 所有查询的结果
        """
        if queries is None:
            # 默认查询列表
            queries = [
                "如何提高检索系统性能",
                "推荐系统的协同过滤算法",
                "机器学习基础教程",
                "RAG系统架构设计",
                "向量数据库比较"
            ]
        
        all_results = []
        total_latency = 0
        
        print(f"\n{'双路径并发' if use_dual_path else '单路径路由'}检索演示（阈值={self.confidence_threshold}）\n")
        
        for i, query in enumerate(queries):
            print(f"查询 {i+1}: {query}")
            
            # 执行检索
            if use_dual_path:
                result = self.search_dual_path(query)
                print(f"  route={result['route']}, conf={result['confidence']}, topk={len(result['results'])}, latency_ms={result['latency']}")
                print(f"  hot_latency_ms={result['hot_latency']}, cold_latency_ms={result['cold_latency']}")
            else:
                result = self.search_single_path(query)
                print(f"  route={result['route']}, conf={result['confidence']}, topk={len(result['results'])}, latency_ms={result['latency']}")
            
            # 格式化输出结果
            formatted_results = "[" + ",".join([f"({doc}, {score:.3f})" for doc, score in result['results'][:5]]) + "]"
            print(f"  merged={formatted_results}")
            print()
            
            all_results.append(result)
            total_latency += result['latency']
        
        # 计算平均延迟
        avg_latency = total_latency / len(queries)
        print(f"平均延迟: {avg_latency:.2f}毫秒")
        
        return all_results, avg_latency
    
    def compare_strategies(self):
        """
        比较单路径和双路径检索策略的性能
        """
        print("\n=== 检索策略比较实验 ===")
        
        # 默认查询列表
        queries = [
            "如何提高检索系统性能",
            "推荐系统的协同过滤算法",
            "机器学习基础教程",
            "RAG系统架构设计",
            "向量数据库比较"
        ]
        
        # 测试单路径策略
        print("\n1. 单路径路由策略:")
        single_results, single_avg_latency = self.run_demo(queries, use_dual_path=False)
        
        # 测试双路径策略
        print("\n2. 双路径并发策略:")
        dual_results, dual_avg_latency = self.run_demo(queries, use_dual_path=True)
        
        # 分析结果覆盖度
        print("\n=== 结果覆盖度分析 ===")
        for i in range(len(queries)):
            single_docs = set([doc for doc, _ in single_results[i]['results']])
            dual_docs = set([doc for doc, _ in dual_results[i]['results']])
            
            # 计算交集和差集
            intersection = single_docs.intersection(dual_docs)
            single_only = single_docs - dual_docs
            dual_only = dual_docs - single_docs
            
            print(f"\n查询 {i+1}: {queries[i]}")
            print(f"  单路径结果数: {len(single_docs)}, 双路径结果数: {len(dual_docs)}")
            print(f"  共同结果数: {len(intersection)}")
            print(f"  单路径特有: {len(single_only)}")
            print(f"  双路径特有: {len(dual_only)}")
        
        # 总结
        print(f"\n=== 性能总结 ===")
        print(f"单路径平均延迟: {single_avg_latency:.2f}毫秒")
        print(f"双路径平均延迟: {dual_avg_latency:.2f}毫秒")
        
        # 计算延迟比率
        latency_ratio = dual_avg_latency / single_avg_latency if single_avg_latency > 0 else 0
        print(f"双路径/单路径延迟比: {latency_ratio:.2f}x")
        
        if dual_avg_latency < single_avg_latency:
            print("结论: 在当前测试条件下，双路径策略具有更低的延迟。")
        elif dual_avg_latency > single_avg_latency:
            print("结论: 在当前测试条件下，单路径策略具有更低的延迟。")
        else:
            print("结论: 两种策略在延迟方面表现相当。")
    
    def threshold_sensitivity(self):
        """
        测试不同置信度阈值对路由策略的影响
        """
        print("\n=== 阈值敏感性分析 ===")
        
        # 默认查询
        query = "如何提高检索系统性能"
        print(f"测试查询: {query}")
        
        # 测试不同阈值
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            # 临时设置阈值
            original_threshold = self.confidence_threshold
            self.confidence_threshold = threshold
            
            # 执行单路径检索
            result = self.search_single_path(query)
            
            # 恢复原始阈值
            self.confidence_threshold = original_threshold
            
            print(f"\n阈值={threshold}:")
            print(f"  置信度: {result['confidence']}")
            print(f"  路由路径: {result['route']}")
            print(f"  延迟: {result['latency']}毫秒")
            
            # 格式化输出结果
            formatted_results = "[" + ",".join([f"({doc}, {score:.3f})" for doc, score in result['results'][:3]]) + "]"
            print(f"  Top-3结果: {formatted_results}")

# 主函数
def main():
    """主函数"""
    print("=== 热冷集路由检索演示程序 ===")
    
    # 创建演示实例
    demo = HotColdRoutingDemo(
        confidence_threshold=0.7,
        topk=5,
        rerank_size=20,
        maxsim_threshold=0.9,
        rrf_k=60
    )
    
    # 运行基础演示
    print("\n=== 基础功能演示 ===")
    demo.run_demo()
    
    # 运行策略比较
    demo.compare_strategies()
    
    # 运行阈值敏感性分析
    demo.threshold_sensitivity()
    
    # 检查执行效率
    print("\n=== 执行效率测试 ===")
    total_time = 0
    num_runs = 5
    
    for i in range(num_runs):
        query = f"测试查询 {i+1}: 如何优化检索系统性能"
        
        start_time = time.time()
        demo.search_single_path(query)
        end_time = time.time()
        
        run_time = (end_time - start_time) * 1000  # 转换为毫秒
        total_time += run_time
        print(f"运行 {i+1}: {run_time:.2f}毫秒")
    
    avg_time = total_time / num_runs
    print(f"平均运行时间 ({num_runs}次): {avg_time:.2f}毫秒")
    
    # 如果执行时间过长，尝试优化
    if avg_time > 50:
        print("\n注意：执行时间较长，建议优化")
        print("优化建议:")
        print("1. 减少模拟延迟时间")
        print("2. 减小rerank_size参数值")
        print("3. 优化向量相似度计算")

if __name__ == "__main__":
    main()