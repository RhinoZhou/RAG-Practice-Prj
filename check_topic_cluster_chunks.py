#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题聚类分块结果检查脚本
用于分析和验证TopicClusterBoundaryChunker生成的分块结果
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class TopicClusterChunksAnalyzer:
    """主题聚类分块结果分析器"""
    def __init__(self, chunks_path: str, index_tolerance: int = 5):
        """
        初始化分析器
        
        参数:
        - chunks_path: 分块结果JSON文件路径
        - index_tolerance: 索引连续性检查的容差（字符数）
        """
        self.chunks_path = chunks_path
        self.index_tolerance = index_tolerance
        self.chunks = []
        self.load_chunks()
        
    def load_chunks(self) -> None:
        """加载分块结果"""
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"分块结果文件不存在: {self.chunks_path}")
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"成功加载 {len(self.chunks)} 个分块")
    
    def check_index_continuity(self) -> bool:
        """检查索引连续性"""
        if len(self.chunks) <= 1:
            return True
        
        continuity_passed = True
        issues = []
        
        for i in range(1, len(self.chunks)):
            prev_end = self.chunks[i-1]['end_index']
            curr_start = self.chunks[i]['start_index']
            gap = curr_start - prev_end
            
            if abs(gap) > self.index_tolerance:
                continuity_passed = False
                issues.append({
                    'prev_chunk_id': self.chunks[i-1]['chunk_id'],
                    'curr_chunk_id': self.chunks[i]['chunk_id'],
                    'prev_end': prev_end,
                    'curr_start': curr_start,
                    'gap': gap
                })
        
        # 打印连续性检查结果
        print(f"\n索引连续性检查（容差: {self.index_tolerance} 字符）:")
        if continuity_passed:
            print("✓ 索引连续性检查通过")
        else:
            print(f"✗ 索引连续性检查失败，发现 {len(issues)} 处索引不连续")
            # 打印前5个问题
            for issue in issues[:5]:
                print(f"  分块 {issue['prev_chunk_id']} (end={issue['prev_end']}) 与分块 {issue['curr_chunk_id']} (start={issue['curr_start']}) 之间的差距: {issue['gap']} 字符")
            if len(issues) > 5:
                print(f"  ... 还有 {len(issues) - 5} 处问题未显示")
        
        return continuity_passed
    
    def analyze_token_distribution(self) -> Dict:
        """分析token分布"""
        if not self.chunks:
            return {}
        
        tokens = [chunk['tokens_count'] for chunk in self.chunks]
        stats = {
            'total': sum(tokens),
            'average': np.mean(tokens),
            'median': np.median(tokens),
            'min': min(tokens),
            'max': max(tokens),
            'std': np.std(tokens)
        }
        
        # 打印token分布统计
        print("\ntoken分布统计:")
        print(f"  总token数: {stats['total']}")
        print(f"  平均token数: {stats['average']:.2f}")
        print(f"  中位数token数: {stats['median']}")
        print(f"  最小token数: {stats['min']}")
        print(f"  最大token数: {stats['max']}")
        print(f"  标准差: {stats['std']:.2f}")
        
        return stats
    
    def analyze_cluster_distribution(self) -> Dict:
        """分析簇分布"""
        if not self.chunks:
            return {}
        
        cluster_ids = [chunk['cluster_id'] for chunk in self.chunks]
        unique_clusters = set(cluster_ids)
        cluster_counts = {cid: cluster_ids.count(cid) for cid in unique_clusters}
        
        # 打印簇分布统计
        print("\n簇分布统计:")
        print(f"  唯一簇数量: {len(unique_clusters)}")
        
        # 按簇出现次数排序并打印前10个
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        print("  主要簇分布（前10个）:")
        for i, (cid, count) in enumerate(sorted_clusters[:10]):
            percentage = (count / len(self.chunks)) * 100
            print(f"    簇 {cid}: {count} 个分块 ({percentage:.1f}%)")
        
        if len(sorted_clusters) > 10:
            remaining_count = len(self.chunks) - sum(count for _, count in sorted_clusters[:10])
            remaining_percentage = (remaining_count / len(self.chunks)) * 100
            print(f"    其他簇: {remaining_count} 个分块 ({remaining_percentage:.1f}%)")
        
        return cluster_counts
    
    def analyze_sentences_per_chunk(self) -> Dict:
        """分析每个分块的句子数量"""
        if not self.chunks:
            return {}
        
        sentences = [chunk['sentences_count'] for chunk in self.chunks]
        stats = {
            'average': np.mean(sentences),
            'median': np.median(sentences),
            'min': min(sentences),
            'max': max(sentences)
        }
        
        # 打印句子数量统计
        print("\n每个分块的句子数量统计:")
        print(f"  平均句子数: {stats['average']:.2f}")
        print(f"  中位数句子数: {stats['median']}")
        print(f"  最小句子数: {stats['min']}")
        print(f"  最大句子数: {stats['max']}")
        
        return stats
    
    def display_sample_chunks(self, num_samples: int = 3) -> None:
        """显示样例分块"""
        print(f"\n{num_samples} 个分块样例:")
        samples = self.chunks[:num_samples]
        
        for i, chunk in enumerate(samples):
            print(f"\n分块 {chunk['chunk_id']}:")
            print(f"  簇ID: {chunk['cluster_id']}")
            print(f"  token数量: {chunk['tokens_count']}")
            print(f"  句子数量: {chunk['sentences_count']}")
            print(f"  位置范围: {chunk['start_index']} - {chunk['end_index']}")
            # 打印前150个字符作为预览
            preview = chunk['text'][:150] + ("..." if len(chunk['text']) > 150 else "")
            print(f"  内容预览: {preview}")
    
    def run_complete_analysis(self) -> Dict:
        """运行完整的分析流程"""
        print(f"\n======== 主题聚类分块结果分析 ========")
        print(f"分析文件: {self.chunks_path}")
        print(f"分块总数: {len(self.chunks)}")
        
        # 运行各项分析
        results = {
            'index_continuity': self.check_index_continuity(),
            'token_distribution': self.analyze_token_distribution(),
            'cluster_distribution': self.analyze_cluster_distribution(),
            'sentences_per_chunk': self.analyze_sentences_per_chunk()
        }
        
        # 显示样例分块
        self.display_sample_chunks()
        
        print("\n======== 分析完成 ========")
        return results

def main():
    """主函数，运行主题聚类分块结果分析"""
    # 设置文件路径
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    chunks_path = os.path.join(results_dir, "topic_cluster_chunks.json")
    
    # 创建分析器并运行分析
    analyzer = TopicClusterChunksAnalyzer(chunks_path, index_tolerance=5)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()