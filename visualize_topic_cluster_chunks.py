#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题聚类分块结果可视化脚本
用于生成主题聚类分块结果的图表，直观展示分析结果
"""

import json
import os
import numpy as np
from typing import List, Dict

# 尝试导入必要的库，如果失败则安装
def install_dependencies():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("正在安装必要的依赖...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib", "seaborn"])
        print("依赖安装完成")

# 安装依赖
install_dependencies()

# 导入已安装的库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

class TopicClusterChunksVisualizer:
    """主题聚类分块结果可视化工具"""
    def __init__(self, chunks_path: str, output_dir: str):
        """
        初始化可视化工具
        
        参数:
        - chunks_path: 分块结果JSON文件路径
        - output_dir: 图表输出目录
        """
        self.chunks_path = chunks_path
        self.output_dir = output_dir
        self.chunks = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载分块数据
        self.load_chunks()
    
    def load_chunks(self) -> None:
        """加载分块结果"""
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"分块结果文件不存在: {self.chunks_path}")
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"成功加载 {len(self.chunks)} 个分块用于可视化")
    
    def plot_token_distribution(self) -> None:
        """绘制token数量分布直方图"""
        if not self.chunks:
            return
        
        tokens = [chunk['tokens_count'] for chunk in self.chunks]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(tokens, bins=30, kde=True, color='skyblue')
        plt.title('分块Token数量分布', fontsize=16)
        plt.xlabel('Token数量', fontsize=14)
        plt.ylabel('频率', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        
        # 添加统计信息到图表
        stats_text = f"""总分块数: {len(tokens)}
平均分块大小: {np.mean(tokens):.2f} tokens
中位数分块大小: {np.median(tokens)} tokens
最大分块大小: {np.max(tokens)} tokens
最小分块大小: {np.min(tokens)} tokens"""
        plt.figtext(0.95, 0.5, stats_text, fontsize=12, va='center', ha='right', bbox=dict(facecolor='white', alpha=0.8))
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'token_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Token分布直方图已保存至: {output_path}")
    
    def plot_cluster_distribution(self) -> None:
        """绘制簇分布饼图"""
        if not self.chunks:
            return
        
        cluster_ids = [chunk['cluster_id'] for chunk in self.chunks]
        unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
        
        # 按数量排序
        sorted_indices = np.argsort(counts)[::-1]
        unique_clusters = unique_clusters[sorted_indices]
        counts = counts[sorted_indices]
        
        # 为较小的簇创建'其他'类别
        threshold = 0.03  # 3%的阈值
        main_clusters = []
        main_counts = []
        other_count = 0
        
        for cid, count in zip(unique_clusters, counts):
            if count / len(cluster_ids) >= threshold:
                main_clusters.append(f'簇 {cid}')
                main_counts.append(count)
            else:
                other_count += count
        
        if other_count > 0:
            main_clusters.append('其他簇')
            main_counts.append(other_count)
        
        # 创建饼图
        plt.figure(figsize=(12, 8))
        wedges, texts, autotexts = plt.pie(main_counts, labels=main_clusters, autopct='%1.1f%%',
                                          startangle=90, shadow=False, explode=[0.05]*len(main_clusters))
        
        # 设置文本样式
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
        
        plt.title('分块簇分布', fontsize=16)
        plt.axis('equal')  # 保证饼图是圆的
        
        # 添加总簇数信息
        plt.figtext(0.5, 0.01, f'总唯一簇数: {len(unique_clusters)}', ha='center', fontsize=12)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'cluster_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"簇分布饼图已保存至: {output_path}")
    
    def plot_chunk_size_vs_cluster(self) -> None:
        """绘制分块大小与簇ID的关系散点图"""
        if not self.chunks:
            return
        
        cluster_ids = [chunk['cluster_id'] for chunk in self.chunks]
        tokens = [chunk['tokens_count'] for chunk in self.chunks]
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=cluster_ids, y=tokens, alpha=0.6, s=100, color='purple')
        plt.title('分块大小与簇ID的关系', fontsize=16)
        plt.xlabel('簇ID', fontsize=14)
        plt.ylabel('Token数量', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(cluster_ids, tokens, 1)
        p = np.poly1d(z)
        plt.plot(sorted(set(cluster_ids)), p(sorted(set(cluster_ids))), "r--", linewidth=2)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'chunk_size_vs_cluster.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"分块大小与簇ID关系散点图已保存至: {output_path}")
    
    def plot_cumulative_distribution(self) -> None:
        """绘制分块大小的累积分布函数图"""
        if not self.chunks:
            return
        
        tokens = [chunk['tokens_count'] for chunk in self.chunks]
        tokens.sort()
        cumulative = np.arange(1, len(tokens) + 1) / len(tokens)
        
        plt.figure(figsize=(12, 6))
        plt.plot(tokens, cumulative, marker='o', linestyle='-', markersize=4, color='green')
        plt.title('分块大小累积分布函数', fontsize=16)
        plt.xlabel('Token数量', fontsize=14)
        plt.ylabel('累积概率', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yticks(np.arange(0, 1.1, 0.1))
        
        # 添加参考线显示分位数
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            percentile_value = np.percentile(tokens, p)
            plt.axvline(x=percentile_value, color='red', linestyle='--', alpha=0.5)
            plt.text(percentile_value, 0.05, f'{p}%分位数: {percentile_value:.1f}', 
                     rotation=90, verticalalignment='bottom', fontsize=10)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'cumulative_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"累积分布函数图已保存至: {output_path}")
    
    def plot_sentences_per_chunk(self) -> None:
        """绘制每个分块的句子数量分布"""
        if not self.chunks:
            return
        
        sentences = [chunk['sentences_count'] for chunk in self.chunks]
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x=sentences, color='orange')
        plt.title('每个分块的句子数量分布', fontsize=16)
        plt.xlabel('句子数量', fontsize=14)
        plt.ylabel('分块数', fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        
        # 限制x轴范围，避免过多的类别
        if max(sentences) > 10:
            plt.xlim(-0.5, 10.5)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'sentences_per_chunk.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"句子数量分布图已保存至: {output_path}")
    
    def create_all_visualizations(self) -> None:
        """创建所有可视化图表"""
        print("\n======== 开始生成可视化图表 ========")
        
        # 生成各个图表
        self.plot_token_distribution()
        self.plot_cluster_distribution()
        self.plot_chunk_size_vs_cluster()
        self.plot_cumulative_distribution()
        self.plot_sentences_per_chunk()
        
        print("\n======== 所有可视化图表生成完成 ========")

def main():
    """主函数，生成主题聚类分块结果的可视化图表"""
    # 设置文件路径
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    chunks_path = os.path.join(results_dir, "topic_cluster_chunks.json")
    
    # 创建可视化工具并生成所有图表
    visualizer = TopicClusterChunksVisualizer(chunks_path, results_dir)
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()