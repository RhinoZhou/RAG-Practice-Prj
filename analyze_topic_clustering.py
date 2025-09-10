#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题聚类分块结果分析工具
用于评估分块器的性能和分块质量
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from typing import List, Dict, Any

class TopicClusteringAnalyzer:
    """主题聚类分块结果分析器"""
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.chunks = []
        self.token_counts = []
        self.cluster_ids = []
        self.cluster_sizes = defaultdict(int)
        self.cluster_token_distributions = defaultdict(list)
        self.consecutive_same_cluster = 0  # 连续相同簇的分块计数
        self.max_consecutive_same_cluster = 0  # 最大连续相同簇的分块数

    def load_data(self) -> None:
        """从JSON文件加载分块数据"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunks = data
                
                # 提取必要的统计数据
                prev_cluster = None
                for chunk in self.chunks:
                    token_count = chunk.get('tokens_count', 0)
                    cluster_id = chunk.get('cluster_id', -1)
                    
                    self.token_counts.append(token_count)
                    self.cluster_ids.append(cluster_id)
                    self.cluster_sizes[cluster_id] += 1
                    self.cluster_token_distributions[cluster_id].append(token_count)
                    
                    # 计算连续相同簇的分块数
                    if prev_cluster == cluster_id:
                        self.consecutive_same_cluster += 1
                        self.max_consecutive_same_cluster = max(
                            self.max_consecutive_same_cluster, 
                            self.consecutive_same_cluster
                        )
                    else:
                        self.consecutive_same_cluster = 1
                    prev_cluster = cluster_id
                    
            print(f"成功加载 {len(self.chunks)} 个分块")
        except Exception as e:
            print(f"加载数据出错: {e}")

    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """计算基本统计指标"""
        if not self.chunks:
            return {}
        
        return {
            'total_chunks': len(self.chunks),
            'total_tokens': sum(self.token_counts),
            'avg_tokens_per_chunk': np.mean(self.token_counts),
            'median_tokens_per_chunk': np.median(self.token_counts),
            'min_tokens_per_chunk': min(self.token_counts),
            'max_tokens_per_chunk': max(self.token_counts),
            'unique_clusters': len(set(self.cluster_ids)),
            'max_consecutive_same_cluster': self.max_consecutive_same_cluster
        }

    def analyze_cluster_distribution(self) -> Dict[int, float]:
        """分析簇的分布情况"""
        total = len(self.chunks)
        return {cluster_id: (count / total) * 100 for cluster_id, count in self.cluster_sizes.items()}

    def analyze_token_distribution(self) -> Dict[str, Any]:
        """分析分块大小（token数）的分布"""
        if not self.token_counts:
            return {}
        
        # 计算分块大小分布区间
        bins = [0, 20, 40, 60, 80, 100, 150, 200, float('inf')]
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-150', '151-200', '200+']
        
        # 使用numpy的digitize函数进行区间统计
        bin_indices = np.digitize(self.token_counts, bins)
        bin_counts = Counter(bin_indices)
        
        # 转换为更友好的格式
        distribution = {labels[i]: bin_counts.get(i+1, 0) for i in range(len(labels))}
        
        # 计算分块大小的变异系数（标准差/均值）
        cv = np.std(self.token_counts) / np.mean(self.token_counts) if np.mean(self.token_counts) > 0 else 0
        
        return {
            'distribution': distribution,
            'coefficient_of_variation': cv
        }

    def generate_visualizations(self, output_dir: str) -> None:
        """生成可视化图表"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 1. Token数量分布直方图
        plt.figure(figsize=(12, 6))
        sns.histplot(self.token_counts, bins=30, kde=True)
        plt.title('分块大小分布（token数）')
        plt.xlabel('Token数量')
        plt.ylabel('分块数量')
        plt.savefig(os.path.join(output_dir, 'token_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 簇分布饼图
        cluster_data = self.analyze_cluster_distribution()
        labels = [f'簇 {cid} ({count:.1f}%)' for cid, count in cluster_data.items()]
        plt.figure(figsize=(10, 10))
        plt.pie(cluster_data.values(), labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('簇分布比例')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 簇大小与token分布散点图
        plt.figure(figsize=(12, 6))
        cluster_ids = list(self.cluster_token_distributions.keys())
        avg_tokens = [np.mean(tokens) for tokens in self.cluster_token_distributions.values()]
        cluster_sizes = [len(tokens) for tokens in self.cluster_token_distributions.values()]
        
        scatter = plt.scatter(cluster_ids, avg_tokens, s=[s*10 for s in cluster_sizes], alpha=0.7)
        plt.title('各簇的平均分块大小与簇大小关系')
        plt.xlabel('簇ID')
        plt.ylabel('平均token数')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加大小图例
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
        size_labels = [f'{int(s/10)}个分块' for s in sorted(set(cluster_sizes))]
        plt.legend(handles, size_labels, title="簇大小", loc="upper right")
        
        plt.savefig(os.path.join(output_dir, 'cluster_token_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 分块序列中的簇变化可视化
        plt.figure(figsize=(15, 6))
        chunk_indices = list(range(min(500, len(self.cluster_ids))))  # 限制显示前500个分块以避免图表过长
        cluster_subset = self.cluster_ids[:500]
        
        plt.scatter(chunk_indices, cluster_subset, s=10)
        plt.plot(chunk_indices, cluster_subset, alpha=0.3)
        plt.title('分块序列中的簇变化（前500个分块）')
        plt.xlabel('分块索引')
        plt.ylabel('簇ID')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'cluster_sequence.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def display_summary(self) -> None:
        """显示分析摘要"""
        basic_stats = self.calculate_basic_statistics()
        token_dist = self.analyze_token_distribution()
        cluster_dist = self.analyze_cluster_distribution()
        
        print("=" * 60)
        print("主题聚类分块结果分析摘要")
        print("=" * 60)
        
        # 基本统计信息
        print(f"分块总数: {basic_stats.get('total_chunks', 0)}")
        print(f"总token数: {basic_stats.get('total_tokens', 0)}")
        print(f"平均分块大小: {basic_stats.get('avg_tokens_per_chunk', 0):.2f} tokens")
        print(f"中位数分块大小: {basic_stats.get('median_tokens_per_chunk', 0)} tokens")
        print(f"最小分块大小: {basic_stats.get('min_tokens_per_chunk', 0)} tokens")
        print(f"最大分块大小: {basic_stats.get('max_tokens_per_chunk', 0)} tokens")
        print(f"分块大小变异系数: {token_dist.get('coefficient_of_variation', 0):.4f}")
        print(f"唯一簇数量: {basic_stats.get('unique_clusters', 0)}")
        print(f"最大连续相同簇分块数: {basic_stats.get('max_consecutive_same_cluster', 0)}")
        print("-" * 60)
        
        # 主要簇分布
        sorted_clusters = sorted(cluster_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        print("主要簇分布（前10个）:")
        for cluster_id, percentage in sorted_clusters:
            print(f"  簇 {cluster_id}: {percentage:.2f}% ({self.cluster_sizes[cluster_id]}个分块)")
        print("-" * 60)
        
        # 分块大小分布
        print("分块大小分布:")
        for range_label, count in token_dist.get('distribution', {}).items():
            percentage = (count / basic_stats.get('total_chunks', 1)) * 100
            print(f"  {range_label} tokens: {count}个分块 ({percentage:.2f}%)")
        print("=" * 60)
        
        # 分析结论
        self._generate_conclusion(basic_stats, token_dist, cluster_dist)

    def _generate_conclusion(self, basic_stats: Dict[str, Any], 
                             token_dist: Dict[str, Any], 
                             cluster_dist: Dict[int, float]) -> None:
        """生成分析结论"""
        print("分析结论:")
        
        # 1. 分块大小评估
        avg_token = basic_stats.get('avg_tokens_per_chunk', 0)
        cv = token_dist.get('coefficient_of_variation', 0)
        
        if avg_token < 30:
            size_analysis = "分块过小，可能会导致上下文断裂，建议调整max_tokens参数或考虑更粗粒度的聚类。"
        elif avg_token > 150:
            size_analysis = "分块过大，可能会包含过多不相关信息，建议调整聚类参数增加簇数量。"
        else:
            size_analysis = "分块大小较为合理，适合大多数RAG应用场景。"
            
        if cv > 1.0:
            size_analysis += " 分块大小变异较大，分布不均匀。"
        elif cv < 0.5:
            size_analysis += " 分块大小分布较为均匀。"
        
        print(f"  1. {size_analysis}")
        
        # 2. 簇分布评估
        main_cluster_percentage = max(cluster_dist.values()) if cluster_dist else 0
        unique_clusters = basic_stats.get('unique_clusters', 0)
        
        if main_cluster_percentage > 50:
            cluster_analysis = f"存在主导簇（簇 {max(cluster_dist, key=cluster_dist.get)}，占比{main_cluster_percentage:.1f}%），可能导致主题区分不明显。"
        elif main_cluster_percentage < 10:
            cluster_analysis = "簇分布较为均匀，主题区分度较好。"
        else:
            cluster_analysis = "簇分布合理，有明显的主题区分。"
            
        if unique_clusters < 5:
            cluster_analysis += " 簇数量较少，可能未能充分区分文档主题。"
        elif unique_clusters > 30:
            cluster_analysis += " 簇数量较多，可能导致过度分割。"
            
        print(f"  2. {cluster_analysis}")
        
        # 3. 连贯性评估
        max_consecutive = basic_stats.get('max_consecutive_same_cluster', 0)
        avg_consecutive = len(self.chunks) / len(set(self.cluster_ids)) if len(set(self.cluster_ids)) > 0 else 0
        
        if max_consecutive > avg_consecutive * 3:
            coherence_analysis = "存在较长的连续相同簇序列，主题连贯性较好，但可能存在局部过度聚类。"
        elif max_consecutive < avg_consecutive / 2:
            coherence_analysis = "连续相同簇序列较短，主题转换频繁，可能影响上下文连贯性。"
        else:
            coherence_analysis = "主题转换较为合理，既保持了上下文连贯性，又避免了过长的单一主题序列。"
            
        print(f"  3. {coherence_analysis}")
        
        # 4. 总体评估
        print("  4. 总体而言，")
        if (avg_token >= 30 and avg_token <= 150 and cv <= 1.0 and 
            unique_clusters >= 5 and unique_clusters <= 30 and 
            main_cluster_percentage <= 50):
            print("     分块结果质量良好，适合用于需要保留文档主题结构的RAG系统。")
        else:
            print("     分块结果基本可用，但建议根据具体应用场景和上述分析调整参数以获得更好的效果。")
            
        # 5. 优化建议
        print("  5. 优化建议:")
        if avg_token < 30 or avg_token > 150:
            print("     - 调整max_tokens参数以控制分块大小。")
        if main_cluster_percentage > 50 or unique_clusters < 5:
            print("     - 增加n_clusters参数值以获得更细粒度的主题划分。")
        if unique_clusters > 30 or max_consecutive < avg_consecutive / 2:
            print("     - 考虑使用DBSCAN聚类算法并调整eps和min_samples参数。")
        print("     - 在实际应用中，建议结合下游任务的性能指标进行参数调优。")

    def check_index_continuity(self, tolerance: int = 5) -> bool:
        """检查分块之间的索引连续性，确保没有文本丢失或重叠"""
        if len(self.chunks) < 2:
            return True
        
        is_continuous = True
        for i in range(1, len(self.chunks)):
            prev_end = self.chunks[i-1].get('end_index', 0)
            curr_start = self.chunks[i].get('start_index', 0)
            
            if abs(curr_start - prev_end) > tolerance:
                print(f"警告：分块 {i-1} 和分块 {i} 之间存在不连续，差值为 {curr_start - prev_end}")
                is_continuous = False
        
        if is_continuous:
            print(f"索引连续性检查通过（容差：{tolerance}字符）")
        
        return is_continuous

    def find_representative_chunks(self, n_per_cluster: int = 1) -> List[Dict[str, Any]]:
        """为每个簇找到代表性的分块（平均大小附近的分块）"""
        representative_chunks = []
        
        for cluster_id, token_counts in self.cluster_token_distributions.items():
            if not token_counts:
                continue
                
            avg_tokens = np.mean(token_counts)
            # 找到最接近平均值的分块
            closest_chunk = None
            min_diff = float('inf')
            
            for chunk in self.chunks:
                if chunk.get('cluster_id') == cluster_id:
                    diff = abs(chunk.get('token_count', 0) - avg_tokens)
                    if diff < min_diff:
                        min_diff = diff
                        closest_chunk = chunk
                        
            if closest_chunk:
                representative_chunks.append(closest_chunk)
        
        return representative_chunks

    def display_representative_chunks(self, n_per_cluster: int = 1) -> None:
        """显示每个簇的代表性分块"""
        reps = self.find_representative_chunks(n_per_cluster)
        
        print(f"\n各簇代表性分块示例（共 {len(reps)} 个）:")
        print("=" * 60)
        
        for i, chunk in enumerate(reps[:5]):  # 只显示前5个以避免输出过多
            cluster_id = chunk.get('cluster_id', -1)
            token_count = chunk.get('token_count', 0)
            text_preview = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            
            print(f"簇 {cluster_id} 的代表性分块（{token_count} tokens）:")
            print(f"  {text_preview}")
            print("-" * 60)

    def full_analysis(self, output_dir: str = None) -> None:
        """执行完整的分析流程"""
        self.load_data()
        if not self.chunks:
            print("没有数据可供分析")
            return
        
        # 检查索引连续性
        self.check_index_continuity()
        
        # 显示分析摘要
        self.display_summary()
        
        # 显示代表性分块
        self.display_representative_chunks()
        
        # 生成可视化图表（如果指定了输出目录）
        if output_dir:
            self.generate_visualizations(output_dir)
            print(f"\n可视化图表已保存至: {output_dir}")

# 安装依赖
def install_dependencies():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("正在安装必要的依赖库...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib", "seaborn"])

def main():
    # 安装依赖
    install_dependencies()
    
    # 设置文件路径
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    json_path = os.path.join(results_dir, "topic_cluster_chunks.json")
    
    # 创建分析器并执行分析
    analyzer = TopicClusteringAnalyzer(json_path)
    analyzer.full_analysis(results_dir)

if __name__ == "__main__":
    main()