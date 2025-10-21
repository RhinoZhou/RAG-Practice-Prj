#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图谱路径检索与融合排序演示

功能说明：演示"候选路径→代价剪枝→证据融合排序"的图谱增强检索流程。

作者：Ph.D. Rhino

内容概述：
- 用简化三元组数据（CSV/JSON）构建小型知识网络
- 根据关系白名单与跳数限制生成候选路径
- 用代价函数（边权×跳数×时效衰减×权威折扣）剪枝
- 将图分数与文本相似度（关键词重合度近似）加权输出前列结果
- 输出路径与证据回放清单

执行流程：
1. 读取三元组与来源等级、时间戳；加载关系白名单和最大跳数
2. 以起始实体集进行受限扩展，生成候选路径并计算代价
3. 剪枝保留Top-N最低代价路径；收集证据节点文本并近似文本相关
4. 融合排序：score = alpha*graph_score + (1-alpha)*text_sim
5. 输出排序列表、路径详情与证据回放表（含来源/位置/时间）

输入说明：
  必填：--triples triples.csv、--query "药物A与药物B交互"
  可选：--rel-allow interacts_with,treats --max-hop 2 --alpha 0.6

输出展示：
  Candidate paths kept: 4 (from 17)
  Best answer nodes: [Guideline_2024: DDI: Severe]
  Scores: graph=0.82 text=0.64 final=0.75
  Evidence playback saved: evidence_paths.csv
"""

import os
import sys
import csv
import json
import math
import argparse
import heapq
import datetime
from collections import defaultdict, deque
import re

# 自动安装依赖
def install_dependencies():
    """
    检查并安装必要的依赖包
    """
    required_packages = ['numpy']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装依赖: {package}...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"依赖 {package} 安装成功")
            except Exception as e:
                print(f"安装依赖 {package} 失败: {e}")
                print(f"请手动安装: pip install {package}")

class GraphSearchFusionDemo:
    """
    图谱搜索融合演示类
    实现基于知识图谱的路径检索、剪枝和融合排序功能
    """
    
    def __init__(self, triples_file=None, query=None, rel_allow=None, max_hop=2, alpha=0.6, top_n=5):
        """
        初始化图谱搜索融合演示器
        
        Args:
            triples_file: 三元组数据文件路径
            query: 查询字符串
            rel_allow: 允许的关系类型列表
            max_hop: 最大跳数限制
            alpha: 图分数权重（文本相似度权重为1-alpha）
            top_n: 保留的候选路径数量
        """
        self.triples_file = triples_file
        self.query = query
        self.rel_allow = set(rel_allow) if rel_allow else None
        self.max_hop = max_hop
        self.alpha = alpha
        self.top_n = top_n
        
        # 图谱数据结构
        self.graph = defaultdict(list)  # {subject: [(predicate, object, weight, source_level, timestamp)]}
        self.inverted_graph = defaultdict(list)  # {object: [(predicate, subject, weight, source_level, timestamp)]}
        self.node_attributes = defaultdict(dict)  # 节点属性，包含文本描述等
        
        # 结果数据
        self.candidate_paths = []
        self.pruned_paths = []
        self.final_results = []
    
    def generate_sample_data(self):
        """
        生成示例三元组数据和节点属性数据
        """
        print("生成示例三元组数据...")
        
        # 生成三元组数据
        triples_data = [
            # 药物相互作用数据
            ['Drug_A', 'interacts_with', 'Drug_B', 0.95, 'guideline', '2024-01-15'],
            ['Drug_A', 'interacts_with', 'Drug_C', 0.80, 'clinical_trial', '2023-11-10'],
            ['Drug_B', 'interacts_with', 'Drug_D', 0.75, 'case_report', '2023-09-05'],
            ['Drug_B', 'treats', 'Condition_X', 0.90, 'guideline', '2024-02-20'],
            ['Drug_C', 'treats', 'Condition_Y', 0.85, 'clinical_trial', '2023-08-12'],
            ['Drug_D', 'treats', 'Condition_X', 0.65, 'case_report', '2022-12-01'],
            ['Drug_A', 'has_side_effect', 'SideEffect_1', 0.70, 'package_insert', '2023-05-20'],
            ['Drug_B', 'has_side_effect', 'SideEffect_2', 0.60, 'package_insert', '2023-04-15'],
            
            # 间接连接
            ['Condition_X', 'is_associated_with', 'Protein_P', 0.85, 'research_paper', '2023-07-30'],
            ['Condition_Y', 'is_associated_with', 'Protein_Q', 0.70, 'research_paper', '2023-03-18'],
            ['Protein_P', 'interacts_with', 'Protein_Q', 0.65, 'research_paper', '2022-10-05'],
            
            # 指南和严重警告
            ['DDI_Guideline_2024', 'documents', 'Drug_A', 0.99, 'authority', '2024-03-01'],
            ['DDI_Guideline_2024', 'documents', 'Drug_B', 0.99, 'authority', '2024-03-01'],
            ['DDI_Guideline_2024', 'states', 'DDI_Severe', 0.99, 'authority', '2024-03-01'],
            ['Drug_A', 'mentioned_in', 'DDI_Guideline_2024', 0.99, 'authority', '2024-03-01'],
            ['Drug_B', 'mentioned_in', 'DDI_Guideline_2024', 0.99, 'authority', '2024-03-01'],
        ]
        
        # 保存三元组数据到CSV
        with open('triples.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['subject', 'predicate', 'object', 'weight', 'source_level', 'timestamp'])
            writer.writerows(triples_data)
        
        # 生成节点属性数据
        node_attributes = {
            'Drug_A': {'text': '药物A是一种常用的抗生素，用于治疗多种感染性疾病。', 'type': 'drug'},
            'Drug_B': {'text': '药物B是一种强效止痛药，常用于术后疼痛管理。', 'type': 'drug'},
            'Drug_C': {'text': '药物C是一种降压药，用于控制高血压。', 'type': 'drug'},
            'Drug_D': {'text': '药物D是一种抗抑郁药，用于治疗抑郁症。', 'type': 'drug'},
            'Condition_X': {'text': '疾病X是一种慢性炎症性疾病，需要长期治疗。', 'type': 'condition'},
            'Condition_Y': {'text': '疾病Y是一种代谢性疾病，影响患者的血糖水平。', 'type': 'condition'},
            'Protein_P': {'text': '蛋白质P在炎症反应中发挥重要作用。', 'type': 'protein'},
            'Protein_Q': {'text': '蛋白质Q参与细胞信号传导过程。', 'type': 'protein'},
            'SideEffect_1': {'text': '副作用1包括恶心、呕吐等胃肠道反应。', 'type': 'side_effect'},
            'SideEffect_2': {'text': '副作用2可能导致头晕和嗜睡。', 'type': 'side_effect'},
            'DDI_Guideline_2024': {'text': '2024年药物相互作用指南，提供最新的药物相互作用信息和临床建议。', 'type': 'guideline'},
            'DDI_Severe': {'text': '严重药物相互作用警告：可能导致严重不良反应甚至危及生命。', 'type': 'warning'},
        }
        
        # 保存节点属性到JSON
        with open('node_attributes.json', 'w', encoding='utf-8') as f:
            json.dump(node_attributes, f, ensure_ascii=False, indent=2)
        
        print(f"已生成示例三元组数据: triples.csv")
        print(f"已生成示例节点属性: node_attributes.json")
        
        # 更新文件路径
        if not self.triples_file:
            self.triples_file = 'triples.csv'
    
    def load_data(self):
        """
        加载三元组数据和节点属性数据
        """
        print(f"加载三元组数据: {self.triples_file}")
        
        # 如果未指定三元组文件或文件不存在，则生成示例数据
        if self.triples_file is None or not os.path.exists(self.triples_file):
            print("三元组文件未指定或不存在，正在生成示例数据...")
            # 设置默认输出文件名
            if self.triples_file is None:
                self.triples_file = "triples.csv"
            self.generate_sample_data()
        
        # 读取三元组CSV文件
        with open(self.triples_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject = row['subject']
                predicate = row['predicate']
                obj = row['object']
                weight = float(row['weight'])
                source_level = row['source_level']
                timestamp = row['timestamp']
                
                # 添加到正向图
                self.graph[subject].append((predicate, obj, weight, source_level, timestamp))
                # 添加到反向图（用于双向搜索）
                self.inverted_graph[obj].append((predicate, subject, weight, source_level, timestamp))
        
        # 加载节点属性
        if os.path.exists('node_attributes.json'):
            with open('node_attributes.json', 'r', encoding='utf-8') as f:
                self.node_attributes = json.load(f)
        
        print(f"图谱加载完成，包含 {len(self.graph)} 个节点")
    
    def extract_entities_from_query(self):
        """
        从查询中提取实体（简单规则）
        
        Returns:
            list: 提取的实体列表
        """
        # 简单的实体提取规则：查找Drug_开头的词或特定关键词
        entities = []
        
        # 检查Drug_A、Drug_B等模式
        drug_pattern = re.compile(r'Drug_[A-Z]')
        drugs = drug_pattern.findall(self.query)
        entities.extend(drugs)
        
        # 如果没有找到明确的药物实体，根据查询内容推断
        if not entities:
            if '药物A' in self.query or 'Drug_A' in self.query:
                entities.append('Drug_A')
            if '药物B' in self.query or 'Drug_B' in self.query:
                entities.append('Drug_B')
        
        # 如果还是没有找到实体，默认使用Drug_A和Drug_B
        if not entities:
            print("警告：无法从查询中提取实体，使用默认实体")
            entities = ['Drug_A', 'Drug_B']
        
        print(f"从查询中提取的实体: {entities}")
        return entities
    
    def calculate_path_cost(self, path):
        """
        计算路径代价
        代价函数：边权×跳数×时效衰减×权威折扣
        
        Args:
            path: 路径，格式为 [(subject, predicate, object, weight, source_level, timestamp), ...]
            
        Returns:
            float: 路径代价（越高代价越大）
        """
        # 基础代价计算
        total_weight = 1.0
        hop_count = len(path)
        
        # 计算路径总权重（边权乘积）
        for _, _, _, weight, _, _ in path:
            total_weight *= weight
        
        # 计算时效衰减
        max_age_days = 365 * 2  # 2年为最大时限
        time_decay = 1.0
        
        for _, _, _, _, _, timestamp in path:
            try:
                # 解析时间戳
                date = datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                current_date = datetime.datetime.now()
                age_days = (current_date - date).days
                # 时效衰减函数：e^(-age_days / max_age_days)
                decay = math.exp(-age_days / max_age_days)
                time_decay *= decay
            except:
                # 如果时间格式不正确，使用默认值
                pass
        
        # 计算权威折扣
        # 权威级别映射：authority > guideline > clinical_trial > research_paper > case_report > package_insert
        authority_levels = {
            'authority': 1.0,
            'guideline': 0.9,
            'clinical_trial': 0.8,
            'research_paper': 0.7,
            'case_report': 0.6,
            'package_insert': 0.5
        }
        
        authority_discount = 1.0
        for _, _, _, _, source_level, _ in path:
            discount = authority_levels.get(source_level, 0.5)
            authority_discount *= discount
        
        # 计算最终代价（注意：这里我们希望代价低的路径更好，所以使用1-总权重作为基础代价）
        base_cost = 1.0 - total_weight
        # 综合代价：基础代价 × 跳数惩罚 × 时间衰减倒数 × 权威折扣倒数
        # 注意：时间越久，衰减越大，所以我们用倒数让旧数据代价更高
        # 权威级别越低，折扣越大，所以用倒数让低权威数据代价更高
        cost = base_cost * (1 + hop_count * 0.1) * (1 / (time_decay + 0.01)) * (1 / (authority_discount + 0.01))
        
        return cost
    
    def generate_candidate_paths(self, start_entities, max_paths=100):
        """
        生成候选路径
        
        Args:
            start_entities: 起始实体列表
            max_paths: 最大路径数量限制
        """
        print(f"生成候选路径，起始实体: {start_entities}, 最大跳数: {self.max_hop}")
        
        all_paths = []
        
        # 对每个起始实体进行广度优先搜索
        for start_entity in start_entities:
            if start_entity not in self.graph and start_entity not in self.inverted_graph:
                print(f"警告：起始实体 {start_entity} 不在图谱中")
                continue
            
            # 使用广度优先搜索生成路径
            queue = deque()
            # 初始路径（起点）
            queue.append([(start_entity, None, start_entity, 1.0, 'start', datetime.datetime.now().strftime('%Y-%m-%d'))])
            
            while queue and len(all_paths) < max_paths:
                path = queue.popleft()
                last_node = path[-1][2]  # 获取路径的最后一个节点
                current_hop = len(path) - 1  # 当前跳数
                
                # 如果达到最大跳数，添加到结果中并继续
                if current_hop >= self.max_hop:
                    all_paths.append(path)
                    continue
                
                # 从正向图扩展
                if last_node in self.graph:
                    for predicate, obj, weight, source_level, timestamp in self.graph[last_node]:
                        # 检查关系是否在允许列表中
                        if self.rel_allow and predicate not in self.rel_allow:
                            continue
                        
                        # 避免循环（简单检查：不重复访问相同的节点）
                        if obj not in [node[0] for node in path]:
                            new_path = path.copy()
                            new_path.append((last_node, predicate, obj, weight, source_level, timestamp))
                            queue.append(new_path)
                            if len(all_paths) < max_paths:
                                all_paths.append(new_path)
                
                # 从反向图扩展
                if last_node in self.inverted_graph:
                    for predicate, subj, weight, source_level, timestamp in self.inverted_graph[last_node]:
                        # 检查关系是否在允许列表中
                        if self.rel_allow and predicate not in self.rel_allow:
                            continue
                        
                        # 避免循环
                        if subj not in [node[0] for node in path]:
                            new_path = path.copy()
                            new_path.append((last_node, predicate, subj, weight, source_level, timestamp))
                            queue.append(new_path)
                            if len(all_paths) < max_paths:
                                all_paths.append(new_path)
        
        # 去重路径（基于节点序列）
        unique_paths = []
        seen = set()
        
        for path in all_paths:
            # 创建路径标识（节点序列）
            path_nodes = tuple(node[0] for node in path)
            if path_nodes not in seen:
                seen.add(path_nodes)
                unique_paths.append(path)
        
        self.candidate_paths = unique_paths
        print(f"生成了 {len(self.candidate_paths)} 条候选路径")
    
    def prune_paths(self):
        """
        根据代价函数剪枝路径，保留Top-N最低代价路径
        """
        print(f"路径剪枝，保留Top-{self.top_n}最低代价路径")
        
        # 计算所有路径的代价
        paths_with_cost = []
        for path in self.candidate_paths:
            cost = self.calculate_path_cost(path)
            paths_with_cost.append((cost, path))
        
        # 按代价排序，保留最小的N个
        paths_with_cost.sort(key=lambda x: x[0])
        self.pruned_paths = [path for cost, path in paths_with_cost[:self.top_n]]
        
        print(f"剪枝后保留 {len(self.pruned_paths)} 条路径（从 {len(self.candidate_paths)} 条中）")
    
    def calculate_text_similarity(self, node_text, query):
        """
        简单的文本相似度计算（关键词重合度）
        
        Args:
            node_text: 节点文本
            query: 查询文本
            
        Returns:
            float: 相似度分数（0-1）
        """
        if not node_text or not query:
            return 0.0
        
        # 提取关键词（简单分词）
        def extract_keywords(text):
            # 移除标点符号，转换为小写
            text = re.sub(r'[^\w\s]', '', text.lower())
            # 分词
            keywords = set(text.split())
            # 过滤停用词（简单版）
            stop_words = {'的', '了', '是', '在', '有', '和', '与', '等', '之一', '可以', '一种', '用于'}
            return keywords - stop_words
        
        # 提取查询和节点文本的关键词
        query_keywords = extract_keywords(query)
        node_keywords = extract_keywords(node_text)
        
        if not query_keywords:
            return 0.0
        
        # 计算交集比例
        intersection = query_keywords & node_keywords
        similarity = len(intersection) / len(query_keywords)
        
        return similarity
    
    def collect_evidence_and_rank(self):
        """
        收集证据并进行融合排序
        """
        print("收集证据并进行融合排序...")
        
        # 收集每个路径的终点节点作为候选答案
        candidates = []
        
        for path in self.pruned_paths:
            # 获取路径终点
            end_node = path[-1][2]
            
            # 计算图分数（基于路径代价的倒数）
            path_cost = self.calculate_path_cost(path)
            # 转换代价为分数：cost越低，分数越高
            graph_score = 1.0 / (1.0 + path_cost)
            
            # 获取节点文本描述
            node_text = self.node_attributes.get(end_node, {}).get('text', '')
            
            # 计算文本相似度
            text_similarity = self.calculate_text_similarity(node_text, self.query)
            
            # 融合分数
            final_score = self.alpha * graph_score + (1 - self.alpha) * text_similarity
            
            # 收集路径信息用于证据回放
            path_info = {
                'end_node': end_node,
                'graph_score': graph_score,
                'text_similarity': text_similarity,
                'final_score': final_score,
                'path_length': len(path) - 1,
                'path_nodes': ' -> '.join([node[0] for node in path]),
                'path_relations': ' -> '.join([str(node[1]) for node in path[1:]]),
                'sources': ', '.join(set([node[4] for node in path[1:]])),  # 去重的来源
                'timestamps': ', '.join(set([node[5] for node in path[1:]]))  # 去重的时间戳
            }
            
            candidates.append(path_info)
        
        # 按最终分数排序
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        self.final_results = candidates
        
        # 输出排序结果
        print(f"\n===== 融合排序结果 =====")
        for i, result in enumerate(self.final_results[:5], 1):
            print(f"{i}. 节点: {result['end_node']}")
            print(f"   图分数: {result['graph_score']:.2f}, 文本相似度: {result['text_similarity']:.2f}, 最终分数: {result['final_score']:.2f}")
            print(f"   路径: {result['path_nodes']}")
            print(f"   关系: {result['path_relations']}")
            print(f"   来源: {result['sources']}")
            print()
        
        # 输出最佳答案节点
        if self.final_results:
            best_nodes = [result['end_node'] for result in self.final_results[:3]]
            print(f"Best answer nodes: {best_nodes}")
            print(f"Scores: graph={self.final_results[0]['graph_score']:.2f} text={self.final_results[0]['text_similarity']:.2f} final={self.final_results[0]['final_score']:.2f}")
    
    def save_evidence_playback(self):
        """
        保存证据回放表
        """
        output_file = 'evidence_paths.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'End_Node', 'Graph_Score', 'Text_Similarity', 'Final_Score', 'Path_Length', 'Path_Nodes', 'Path_Relations', 'Sources', 'Timestamps'])
            
            for i, result in enumerate(self.final_results, 1):
                writer.writerow([
                    i,
                    result['end_node'],
                    f"{result['graph_score']:.4f}",
                    f"{result['text_similarity']:.4f}",
                    f"{result['final_score']:.4f}",
                    result['path_length'],
                    result['path_nodes'],
                    result['path_relations'],
                    result['sources'],
                    result['timestamps']
                ])
        
        print(f"Evidence playback saved: {output_file}")
    
    def run(self):
        """
        运行完整的图谱搜索融合流程
        """
        print("=== 开始图谱路径检索与融合排序演示 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 从查询中提取实体
        start_entities = self.extract_entities_from_query()
        
        # 3. 生成候选路径
        self.generate_candidate_paths(start_entities)
        
        # 4. 路径剪枝
        self.prune_paths()
        
        # 5. 收集证据并排序
        self.collect_evidence_and_rank()
        
        # 6. 保存证据回放
        self.save_evidence_playback()
        
        print("\n=== 图谱路径检索与融合排序演示完成 ===")

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图谱路径检索与融合排序演示工具')
    parser.add_argument('--triples', type=str, default=None, help='三元组数据文件路径（可选，未指定时自动生成示例数据）')
    parser.add_argument('--query', type=str, default='药物A与药物B交互', help='查询字符串')
    parser.add_argument('--rel-allow', type=str, default='interacts_with,treats,mentioned_in,documents,states', help='允许的关系类型列表，逗号分隔')
    parser.add_argument('--max-hop', type=int, default=2, help='最大跳数限制')
    parser.add_argument('--alpha', type=float, default=0.6, help='图分数权重（0-1）')
    parser.add_argument('--top-n', type=int, default=5, help='保留的候选路径数量')
    args = parser.parse_args()
    
    # 安装依赖
    install_dependencies()
    
    # 解析关系白名单
    rel_allow = args.rel_allow.split(',') if args.rel_allow else None
    
    # 创建并运行演示器
    demo = GraphSearchFusionDemo(
        triples_file=args.triples,
        query=args.query,
        rel_allow=rel_allow,
        max_hop=args.max_hop,
        alpha=args.alpha,
        top_n=args.top_n
    )
    
    # 运行演示
    demo.run()

if __name__ == "__main__":
    main()