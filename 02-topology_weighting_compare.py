#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中心度/社群与文本相关性的加权比较

功能说明：对比"仅文本相关"与"图谱拓扑信号融合"的前列质量与稳定性。

内容概述：基于小图（节点、边、权威等级）计算简化中心度（如入度/出度代理）和社群分组；
对候选文档按文本相似与拓扑信号融合加权；输出NDCG/Hit@K对比与前列波动性（方差）。

执行流程：
1. 加载节点属性（权威、时间）与边列表。
2. 计算简化中心度、社群标签，归一化为拓扑得分。
3. 两种排序策略：仅文本 vs 文本×拓扑加权。
4. 评估NDCG@K、Hit@K与前列稳定性（多次抽样）。
5. 导出对比表与曲线CSV。

输入说明：--candidates candidates.csv --k 5 --alpha 0.5

作者：Ph.D. Rhino
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Any

# 设置随机种子以确保结果可复现
np.random.seed(42)


class TopologyWeightingComparer:
    """
    中心度/社群与文本相关性的加权比较器
    用于对比仅文本相关和图谱拓扑信号融合的排序效果
    """
    
    def __init__(self, candidates_file: str = None, k: int = 5, alpha: float = 0.5):
        """
        初始化比较器
        
        Args:
            candidates_file: 候选文档文件路径
            k: 评估的截断位置（如NDCG@K、Hit@K中的K）
            alpha: 文本相似性的权重系数，拓扑信号的权重系数为1-alpha
        """
        self.candidates_file = candidates_file
        self.k = k
        self.alpha = alpha
        
        # 数据存储
        self.nodes = {}
        self.edges = []
        self.candidates = []
        self.evaluations = []
        
        # 计算结果存储
        self.degrees = {}
        self.community_labels = {}
        self.topology_scores = {}
        
        # 评估结果存储
        self.results = {
            'text_only': {},
            'fused': {}
        }
        
    def check_dependencies(self):
        """
        检查并安装必要的依赖包
        """
        required_packages = ['numpy', 'pandas', 'matplotlib']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"安装缺失的依赖包: {', '.join(missing_packages)}")
            import subprocess
            for package in missing_packages:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        else:
            print("所有必要的依赖包已安装")
    
    def generate_sample_data(self):
        """
        生成示例数据，包括节点属性、边列表和候选文档
        """
        print("生成示例数据...")
        
        # 生成节点（文档）数据
        nodes_data = []
        for i in range(1, 21):
            node_id = f"doc_{i}"
            nodes_data.append({
                'id': node_id,
                'authority': np.random.uniform(0.3, 0.9),  # 权威等级
                'timestamp': f"2024-03-{i:02d}",  # 时间戳
                'topic': f"topic_{i % 5 + 1}"  # 主题分类
            })
        
        # 保存节点数据
        with open('node_attributes.json', 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        
        # 生成边数据（基于主题相似度）
        edges_data = []
        for i in range(1, 21):
            src = f"doc_{i}"
            # 为每个节点生成2-5条边，优先连接相同或相近主题的节点
            num_edges = np.random.randint(2, 6)
            possible_dsts = []
            
            # 优先选择相同主题的节点
            for j in range(1, 21):
                if i != j and j % 5 == i % 5:
                    possible_dsts.append(f"doc_{j}")
            
            # 如果相同主题的节点不足，添加其他主题的节点
            for j in range(1, 21):
                if i != j and f"doc_{j}" not in possible_dsts and len(possible_dsts) < num_edges:
                    possible_dsts.append(f"doc_{j}")
            
            # 生成边数据
            for dst in possible_dsts[:num_edges]:
                edges_data.append({
                    'source': src,
                    'target': dst,
                    'weight': np.random.uniform(0.6, 1.0),  # 边权重
                    'type': 'citation' if np.random.random() > 0.5 else 'reference'  # 边类型
                })
        
        # 保存边数据
        with open('edges.json', 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        
        # 生成候选文档数据
        candidates_data = []
        # 创建节点ID到索引的映射
        node_index_map = {node['id']: i for i, node in enumerate(nodes_data)}
        
        for i in range(1, 21):
            doc_id = f"doc_{i}"
            relevance = 0.0
            text_similarity = 0.0
            
            # 更合理地设置相关性和文本相似度
            if i <= 7:  # 增加相关文档的数量
                relevance = np.random.uniform(0.7, 0.95)  # 更高的相关性
                # 相关文档的文本相似度也较高，但有一定随机性
                text_similarity = np.random.uniform(0.6, 0.9)
            elif i <= 12:  # 部分相关文档
                relevance = np.random.uniform(0.4, 0.6)
                text_similarity = np.random.uniform(0.5, 0.8)
            else:  # 不相关文档
                relevance = np.random.uniform(0.05, 0.35)
                text_similarity = np.random.uniform(0.3, 0.6)
            
            # 为重要节点增加更高的相关性
            node_index = node_index_map.get(doc_id, -1)
            if node_index >= 0 and nodes_data[node_index].get('authority', 0) > 0.7:
                relevance = max(relevance, np.random.uniform(0.6, 0.9))
            
            candidates_data.append({
                'doc_id': doc_id,
                'text_similarity': text_similarity,
                'relevance': relevance,
                'query_id': 'query_001'
            })
        
        # 保存候选文档数据
        candidates_df = pd.DataFrame(candidates_data)
        candidates_df.to_csv('candidates.csv', index=False, encoding='utf-8-sig')
        
        print("示例数据生成完成：")
        print("- 节点属性: node_attributes.json")
        print("- 边数据: edges.json")
        print("- 候选文档: candidates.csv")
        
        # 更新文件路径
        self.candidates_file = 'candidates.csv'
    
    def load_data(self):
        """
        加载节点属性、边列表和候选文档数据
        """
        print("加载数据...")
        
        # 检查候选文档文件是否存在，如果不存在则生成示例数据
        if self.candidates_file is None or not os.path.exists(self.candidates_file):
            self.generate_sample_data()
        
        # 加载候选文档数据
        self.candidates = pd.read_csv(self.candidates_file)
        print(f"已加载候选文档数据，共 {len(self.candidates)} 条")
        
        # 加载节点属性
        try:
            with open('node_attributes.json', 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)
                for node in nodes_data:
                    self.nodes[node['id']] = node
            print(f"已加载节点属性数据，共 {len(self.nodes)} 个节点")
        except Exception as e:
            print(f"加载节点属性数据失败: {e}，将使用默认值")
        
        # 加载边数据
        try:
            with open('edges.json', 'r', encoding='utf-8') as f:
                self.edges = json.load(f)
            print(f"已加载边数据，共 {len(self.edges)} 条边")
        except Exception as e:
            print(f"加载边数据失败: {e}，将使用默认值")
    
    def calculate_degrees(self):
        """
        计算节点的度（简化中心度）
        包括入度、出度和总度
        """
        print("计算节点度（简化中心度）...")
        
        # 初始化度计数器
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        # 计算入度和出度
        for edge in self.edges:
            out_degree[edge['source']] += 1
            in_degree[edge['target']] += 1
        
        # 合并为总度
        all_nodes = set(in_degree.keys()) | set(out_degree.keys())
        for node in all_nodes:
            self.degrees[node] = {
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'total_degree': in_degree.get(node, 0) + out_degree.get(node, 0)
            }
        
        # 为没有边连接的节点设置默认值
        for doc_id in self.candidates['doc_id'].unique():
            if doc_id not in self.degrees:
                self.degrees[doc_id] = {
                    'in_degree': 0,
                    'out_degree': 0,
                    'total_degree': 0
                }
        
        print(f"已计算 {len(self.degrees)} 个节点的度")
    
    def assign_communities(self):
        """
        基于主题信息为节点分配社群标签
        如果没有主题信息，则基于边连接关系简单聚类
        """
        print("分配社群标签...")
        
        # 优先使用节点的主题信息作为社群标签
        for node_id, node_info in self.nodes.items():
            if 'topic' in node_info:
                self.community_labels[node_id] = node_info['topic']
        
        # 为没有主题信息的节点分配默认社群
        for doc_id in self.candidates['doc_id'].unique():
            if doc_id not in self.community_labels:
                # 基于文档ID简单分配社群
                community_id = f"community_{hash(doc_id) % 5 + 1}"
                self.community_labels[doc_id] = community_id
        
        print(f"已分配 {len(self.community_labels)} 个节点的社群标签")
    
    def calculate_topology_scores(self):
        """
        计算节点的拓扑得分
        综合考虑度中心性和权威等级，使用更稳定的归一化方法
        """
        print("计算拓扑得分...")
        
        # 收集所有节点的度和权威值
        all_degrees = []
        all_authorities = []
        
        for doc_id in self.candidates['doc_id'].unique():
            # 获取度值
            degree = self.degrees.get(doc_id, {}).get('total_degree', 0)
            all_degrees.append(degree)
            
            # 获取权威值
            authority = self.nodes.get(doc_id, {}).get('authority', 0.5)
            all_authorities.append(authority)
        
        # 使用更稳定的归一化方法（min-max归一化加上平滑处理）
        min_degree, max_degree = min(all_degrees), max(all_degrees)
        degree_range = max_degree - min_degree if max_degree > min_degree else 1
        
        min_authority, max_authority = min(all_authorities), max(all_authorities)
        authority_range = max_authority - min_authority if max_authority > min_authority else 1
        
        # 计算拓扑得分（度中心性 + 权威等级），加入平滑处理
        for doc_id in self.candidates['doc_id'].unique():
            # 归一化度值并平滑
            degree = self.degrees.get(doc_id, {}).get('total_degree', 0)
            norm_degree = 0.05 + 0.95 * (degree - min_degree) / degree_range
            
            # 归一化权威值并平滑
            authority = self.nodes.get(doc_id, {}).get('authority', 0.5)
            norm_authority = 0.05 + 0.95 * (authority - min_authority) / authority_range
            
            # 拓扑得分：度中心性（0.4） + 权威等级（0.6）
            topology_score = 0.4 * norm_degree + 0.6 * norm_authority
            self.topology_scores[doc_id] = topology_score
        
        print(f"已计算 {len(self.topology_scores)} 个节点的拓扑得分")
    
    def calculate_ndcg(self, rankings: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        计算NDCG@K
        
        Args:
            rankings: 排序后的文档ID列表
            relevance_scores: 文档ID到相关性分数的映射
            k: 截断位置
            
        Returns:
            NDCG@K值
        """
        # 截取前K个结果
        rankings_k = rankings[:k]
        
        # 计算DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(rankings_k):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 2)  # 位置从1开始
        
        # 计算IDCG@K（理想DCG）
        ideal_rankings = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        idcg = 0.0
        for i, (doc_id, rel) in enumerate(ideal_rankings[:k]):
            idcg += rel / np.log2(i + 2)
        
        # 避免除以0
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_hit_at_k(self, rankings: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        计算Hit@K
        
        Args:
            rankings: 排序后的文档ID列表
            relevant_docs: 相关文档ID集合
            k: 截断位置
            
        Returns:
            Hit@K值（0或1）
        """
        # 检查前K个结果中是否有相关文档
        for doc_id in rankings[:k]:
            if doc_id in relevant_docs:
                return 1.0
        return 0.0
    
    def evaluate_strategies(self):
        """
        评估两种排序策略：仅文本相关 vs 文本×拓扑加权
        """
        print("评估排序策略...")
        
        # 获取相关性分数
        relevance_scores = {}
        relevant_docs = set()
        
        for _, row in self.candidates.iterrows():
            doc_id = row['doc_id']
            rel = row.get('relevance', 0.0)
            relevance_scores[doc_id] = rel
            # 将相关性大于0.5的文档视为相关
            if rel > 0.5:
                relevant_docs.add(doc_id)
        
        # 策略1：仅文本相似性排序
        text_only_rankings = self.candidates.sort_values('text_similarity', ascending=False)['doc_id'].tolist()
        
        # 策略2：文本相似性和拓扑得分融合排序
        # 创建融合得分
        self.candidates['fused_score'] = self.candidates.apply(
            lambda row: self.alpha * row['text_similarity'] + 
                       (1 - self.alpha) * self.topology_scores.get(row['doc_id'], 0.5), 
            axis=1
        )
        fused_rankings = self.candidates.sort_values('fused_score', ascending=False)['doc_id'].tolist()
        
        # 计算评估指标
        self.results['text_only']['ndcg'] = self.calculate_ndcg(text_only_rankings, relevance_scores, self.k)
        self.results['fused']['ndcg'] = self.calculate_ndcg(fused_rankings, relevance_scores, self.k)
        
        self.results['text_only']['hit_at_k'] = self.calculate_hit_at_k(text_only_rankings, relevant_docs, self.k)
        self.results['fused']['hit_at_k'] = self.calculate_hit_at_k(fused_rankings, relevant_docs, self.k)
        
        # 计算Top-K文档集合
        text_only_topk = set(text_only_rankings[:self.k])
        fused_topk = set(fused_rankings[:self.k])
        
        # 计算共同的Top-K文档比例
        common_docs = text_only_topk.intersection(fused_topk)
        self.results['common_topk_ratio'] = len(common_docs) / self.k if self.k > 0 else 0
        
        print(f"排序策略评估完成:")
        print(f"仅文本策略 - NDCG@{self.k}: {self.results['text_only']['ndcg']:.4f}, Hit@{self.k}: {self.results['text_only']['hit_at_k']:.4f}")
        print(f"融合策略 - NDCG@{self.k}: {self.results['fused']['ndcg']:.4f}, Hit@{self.k}: {self.results['fused']['hit_at_k']:.4f}")
        print(f"共同Top-K文档比例: {self.results['common_topk_ratio']:.4f}")
    
    def evaluate_stability(self, n_samples: int = 30, sample_ratio: float = 0.9):
        """
        评估排序策略的稳定性（通过多次抽样）
        增加抽样次数和样本比例以提高评估稳定性
        
        Args:
            n_samples: 抽样次数
            sample_ratio: 每次抽样的比例
        """
        print(f"评估排序策略稳定性（{n_samples}次抽样）...")
        
        text_only_scores = []
        fused_scores = []
        text_only_hit_rates = []
        fused_hit_rates = []
        
        for i in range(n_samples):
            # 随机抽样候选文档，使用更大的样本比例
            sample_size = int(len(self.candidates) * sample_ratio)
            sample = self.candidates.sample(sample_size, replace=False)
            
            # 获取相关性分数
            relevance_scores = {}
            relevant_docs = set()
            
            for _, row in sample.iterrows():
                doc_id = row['doc_id']
                rel = row.get('relevance', 0.0)
                relevance_scores[doc_id] = rel
                if rel > 0.5:
                    relevant_docs.add(doc_id)
            
            # 仅文本相似性排序
            text_only_rankings = sample.sort_values('text_similarity', ascending=False)['doc_id'].tolist()
            
            # 融合排序
            sample['fused_score'] = sample.apply(
                lambda row: self.alpha * row['text_similarity'] + 
                           (1 - self.alpha) * self.topology_scores.get(row['doc_id'], 0.5), 
                axis=1
            )
            fused_rankings = sample.sort_values('fused_score', ascending=False)['doc_id'].tolist()
            
            # 计算指标并记录
            text_only_ndcg = self.calculate_ndcg(text_only_rankings, relevance_scores, self.k)
            fused_ndcg = self.calculate_ndcg(fused_rankings, relevance_scores, self.k)
            
            text_only_hit = self.calculate_hit_at_k(text_only_rankings, relevant_docs, self.k)
            fused_hit = self.calculate_hit_at_k(fused_rankings, relevant_docs, self.k)
            
            text_only_scores.append(text_only_ndcg)
            fused_scores.append(fused_ndcg)
            text_only_hit_rates.append(text_only_hit)
            fused_hit_rates.append(fused_hit)
        
        # 计算方差（稳定性指标）
        self.results['text_only']['variance'] = np.var(text_only_scores)
        self.results['fused']['variance'] = np.var(fused_scores)
        
        # 计算Hit@K的平均值
        self.results['text_only']['avg_hit_rate'] = np.mean(text_only_hit_rates)
        self.results['fused']['avg_hit_rate'] = np.mean(fused_hit_rates)
        
        print(f"排序策略稳定性评估完成:")
        print(f"仅文本策略 - NDCG@{self.k}方差: {self.results['text_only']['variance']:.6f}, 平均Hit@{self.k}: {self.results['text_only']['avg_hit_rate']:.4f}")
        print(f"融合策略 - NDCG@{self.k}方差: {self.results['fused']['variance']:.6f}, 平均Hit@{self.k}: {self.results['fused']['avg_hit_rate']:.4f}")
    
    def generate_ndcg_curves(self, max_k: int = 10):
        """
        生成不同K值下的NDCG曲线数据
        
        Args:
            max_k: 最大的K值
        """
        print(f"生成NDCG曲线数据（K=1到{max_k}）...")
        
        # 获取相关性分数
        relevance_scores = {}
        for _, row in self.candidates.iterrows():
            relevance_scores[row['doc_id']] = row.get('relevance', 0.0)
        
        # 仅文本相似性排序
        text_only_rankings = self.candidates.sort_values('text_similarity', ascending=False)['doc_id'].tolist()
        
        # 融合排序
        fused_rankings = self.candidates.sort_values('fused_score', ascending=False)['doc_id'].tolist()
        
        # 计算不同K值的NDCG
        curve_data = []
        for k in range(1, max_k + 1):
            text_only_ndcg = self.calculate_ndcg(text_only_rankings, relevance_scores, k)
            fused_ndcg = self.calculate_ndcg(fused_rankings, relevance_scores, k)
            
            curve_data.append({
                'K': k,
                'text_only_ndcg': text_only_ndcg,
                'fused_ndcg': fused_ndcg,
                'improvement': fused_ndcg - text_only_ndcg
            })
        
        # 保存曲线数据
        curves_df = pd.DataFrame(curve_data)
        curves_df.to_csv('ndcg_compare.csv', index=False, encoding='utf-8-sig')
        
        print(f"NDCG曲线数据已保存至 ndcg_compare.csv")
        return curves_df
    
    def generate_comparison_report(self):
        """
        生成比较报告并保存为CSV
        """
        print("生成比较报告...")
        
        # 计算改进幅度
        ndcg_improvement = self.results['fused']['ndcg'] - self.results['text_only']['ndcg']
        hit_improvement = self.results['fused']['hit_at_k'] - self.results['text_only']['hit_at_k']
        variance_reduction = self.results['text_only']['variance'] - self.results['fused']['variance']
        
        # 创建报告数据
        report_data = [
            {
                '指标': 'NDCG@' + str(self.k),
                '仅文本相关': self.results['text_only']['ndcg'],
                '图谱拓扑融合': self.results['fused']['ndcg'],
                '改进幅度': ndcg_improvement
            },
            {
                '指标': 'Hit@' + str(self.k),
                '仅文本相关': self.results['text_only']['hit_at_k'],
                '图谱拓扑融合': self.results['fused']['hit_at_k'],
                '改进幅度': hit_improvement
            },
            {
                '指标': f'Top-{self.k}方差',
                '仅文本相关': self.results['text_only']['variance'],
                '图谱拓扑融合': self.results['fused']['variance'],
                '改进幅度': variance_reduction
            },
            {
                '指标': f'平均Hit@{self.k}',
                '仅文本相关': self.results['text_only'].get('avg_hit_rate', self.results['text_only']['hit_at_k']),
                '图谱拓扑融合': self.results['fused'].get('avg_hit_rate', self.results['fused']['hit_at_k']),
                '改进幅度': self.results['fused'].get('avg_hit_rate', self.results['fused']['hit_at_k']) - 
                          self.results['text_only'].get('avg_hit_rate', self.results['text_only']['hit_at_k'])
            },
            {
                '指标': '共同Top-K比例',
                '仅文本相关': 1.0,
                '图谱拓扑融合': self.results['common_topk_ratio'],
                '改进幅度': self.results['common_topk_ratio'] - 1.0
            }
        ]
        
        # 保存报告
        report_df = pd.DataFrame(report_data)
        report_df.to_csv('topology_weighting_comparison.csv', index=False, encoding='utf-8-sig')
        
        print(f"比较报告已保存至 topology_weighting_comparison.csv")
    
    def run(self):
        """
        运行完整的评估流程
        """
        print(f"===== 开始中心度/社群与文本相关性的加权比较 =====")
        print(f"参数设置: K={self.k}, alpha={self.alpha}")
        
        # 检查依赖
        self.check_dependencies()
        
        # 加载数据
        self.load_data()
        
        # 计算拓扑特征
        self.calculate_degrees()
        self.assign_communities()
        self.calculate_topology_scores()
        
        # 评估排序策略
        self.evaluate_strategies()
        
        # 评估稳定性
        self.evaluate_stability()
        
        # 生成NDCG曲线
        self.generate_ndcg_curves()
        
        # 生成比较报告
        self.generate_comparison_report()
        
        # 输出总结结果
        print("\n===== 评估结果总结 =====")
        print(f"NDCG@{self.k}: text-only={self.results['text_only']['ndcg']:.3f}, fused={self.results['fused']['ndcg']:.3f} (+{self.results['fused']['ndcg'] - self.results['text_only']['ndcg']:.3f})")
        print(f"Hit@{self.k}:  {self.results['text_only']['hit_at_k']:.3f} -> {self.results['fused']['hit_at_k']:.3f} (+{self.results['fused']['hit_at_k'] - self.results['text_only']['hit_at_k']:.3f})")
        print(f"Top-{self.k} variance reduced: {self.results['text_only']['variance']:.6f} -> {self.results['fused']['variance']:.6f} (-{self.results['text_only']['variance'] - self.results['fused']['variance']:.6f})")
        print("Curves saved: ndcg_compare.csv")
        print("Report saved: topology_weighting_comparison.csv")
        print("\n===== 中心度/社群与文本相关性的加权比较完成 =====")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='中心度/社群与文本相关性的加权比较')
    parser.add_argument('--candidates', type=str, default=None, help='候选文档文件路径（可选，未指定时自动生成示例数据）')
    parser.add_argument('--k', type=int, default=5, help='评估的截断位置')
    parser.add_argument('--alpha', type=float, default=0.7, help='文本相似性的权重系数')
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 创建比较器实例并运行
    comparer = TopologyWeightingComparer(
        candidates_file=args.candidates,
        k=args.k,
        alpha=args.alpha
    )
    
    comparer.run()


if __name__ == "__main__":
    main()