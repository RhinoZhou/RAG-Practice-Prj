# -*- coding: utf-8 -*-
"""相似度分块结果检查器

用于验证Markdown结构感知分块器的分块结果，检查分块的质量、连贯性和完整性。
"""

import os
import json
import numpy as np
from typing import List, Dict, Any

class SimilarityChunkChecker:
    """相似度分块结果检查器"""
    
    def __init__(self):
        """初始化检查器"""
        pass
    
    def load_chunks(self, chunks_path: str) -> List[Dict[str, Any]]:
        """从JSON文件加载分块结果
        
        参数:
            chunks_path: 分块结果JSON文件路径
            
        返回:
            分块结果列表
        """
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"分块结果文件不存在: {chunks_path}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        return chunks
    
    def load_split_indices(self, indices_path: str) -> List[int]:
        """从JSON文件加载分割点索引
        
        参数:
            indices_path: 分割点索引JSON文件路径
            
        返回:
            分割点索引列表
        """
        if not os.path.exists(indices_path):
            raise FileNotFoundError(f"分割点索引文件不存在: {indices_path}")
        
        with open(indices_path, 'r', encoding='utf-8') as f:
            split_indices = json.load(f)
        
        return split_indices
    
    def check_index_continuity(self, chunks: List[Dict[str, Any]], tolerance: int = 2) -> bool:
        """检查分块索引的连续性
        
        参数:
            chunks: 分块结果列表
            tolerance: 允许的字符偏差容差
            
        返回:
            是否连续
        """
        if not chunks:
            return True
        
        # 按起始索引排序
        sorted_chunks = sorted(chunks, key=lambda x: x['start_index'])
        
        # 检查是否连续（允许一定的容差）
        is_continuous = True
        for i in range(1, len(sorted_chunks)):
            diff = sorted_chunks[i]['start_index'] - sorted_chunks[i-1]['end_index']
            if diff > tolerance:
                print(f"索引不连续: 前一块结束于 {sorted_chunks[i-1]['end_index']}, 后一块开始于 {sorted_chunks[i]['start_index']}，偏差 {diff} 字符")
                is_continuous = False
        
        return is_continuous
    
    def check_token_limits(self, chunks: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """检查分块是否超过最大token限制
        
        参数:
            chunks: 分块结果列表
            max_tokens: 最大token限制
            
        返回:
            超过限制的分块列表
        """
        oversized_chunks = []
        
        for chunk in chunks:
            if chunk['token_count'] > max_tokens:
                oversized_chunks.append(chunk)
        
        return oversized_chunks
    
    def calculate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算分块的统计信息
        
        参数:
            chunks: 分块结果列表
            
        返回:
            统计信息字典
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_tokens': 0,
                'avg_tokens': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'median_tokens': 0
            }
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        sentence_counts = [chunk.get('sentence_count', 0) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens': np.mean(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'median_tokens': np.median(token_counts),
            'avg_sentences_per_chunk': np.mean(sentence_counts)
        }
    
    def analyze_split_indices(self, split_indices: List[int]) -> Dict[str, Any]:
        """分析分割点索引的分布
        
        参数:
            split_indices: 分割点索引列表
            
        返回:
            分割点分析结果
        """
        if not split_indices:
            return {
                'total_indices': 0,
                'avg_distance': 0,
                'min_distance': 0,
                'max_distance': 0
            }
        
        # 计算相邻分割点之间的距离
        distances = []
        sorted_indices = sorted(split_indices)
        
        for i in range(1, len(sorted_indices)):
            distances.append(sorted_indices[i] - sorted_indices[i-1])
        
        return {
            'total_indices': len(split_indices),
            'avg_distance': np.mean(distances) if distances else 0,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0
        }
    
    def print_chunk_samples(self, chunks: List[Dict[str, Any]], num_samples: int = 3) -> None:
        """打印分块样例
        
        参数:
            chunks: 分块结果列表
            num_samples: 要打印的样例数量
        """
        print(f"\n--- 分块样例 ({min(num_samples, len(chunks))} 个) ---")
        
        # 选择前几个和后几个样例
        sample_indices = list(range(min(num_samples, len(chunks))))
        
        for i in sample_indices:
            chunk = chunks[i]
            print(f"\n分块 {i+1} (ID: {chunk.get('chunk_id', i)}):")
            print(f"Token数量: {chunk['token_count']}")
            print(f"句子数量: {chunk.get('sentence_count', 0)}")
            print(f"位置: {chunk['start_index']}-{chunk['end_index']}")
            # 打印前100个字符作为预览
            preview = chunk['text'][:100] + ('...' if len(chunk['text']) > 100 else '')
            print(f"内容预览: {preview}")
    
    def run_complete_check(self, chunks_path: str, indices_path: str, max_tokens: int = 500, index_tolerance: int = 25) -> Dict[str, Any]:
        """运行完整的检查流程
        
        参数:
            chunks_path: 分块结果JSON文件路径
            indices_path: 分割点索引JSON文件路径
            max_tokens: 最大token限制
            index_tolerance: 索引连续性检查的容差
            
        返回:
            检查结果汇总
        """
        # 加载数据
        chunks = self.load_chunks(chunks_path)
        split_indices = self.load_split_indices(indices_path)
        
        # 检查索引连续性
        index_continuity = self.check_index_continuity(chunks, index_tolerance)
        
        # 检查token限制
        oversized_chunks = self.check_token_limits(chunks, max_tokens)
        
        # 计算统计信息
        chunk_stats = self.calculate_statistics(chunks)
        index_stats = self.analyze_split_indices(split_indices)
        
        # 打印结果
        print("=== 分块结果检查报告 ===")
        print(f"\n基本信息:")
        print(f"- 分块总数: {chunk_stats['total_chunks']}")
        print(f"- 分割点总数: {index_stats['total_indices']}")
        print(f"- 总token数: {chunk_stats['total_tokens']}")
        
        print(f"\n分块统计:")
        print(f"- 平均token数: {chunk_stats['avg_tokens']:.2f}")
        print(f"- 最小token数: {chunk_stats['min_tokens']}")
        print(f"- 最大token数: {chunk_stats['max_tokens']}")
        print(f"- 中位数token数: {chunk_stats['median_tokens']}")
        print(f"- 平均每块句子数: {chunk_stats['avg_sentences_per_chunk']:.2f}")
        
        print(f"\n分割点统计:")
        print(f"- 平均分割距离: {index_stats['avg_distance']:.2f}")
        print(f"- 最小分割距离: {index_stats['min_distance']}")
        print(f"- 最大分割距离: {index_stats['max_distance']}")
        
        print(f"\n质量检查:")
        print(f"- 索引连续性: {'通过' if index_continuity else '失败'} (容差: {index_tolerance} 字符)")
        print(f"- 超过token限制的分块数: {len(oversized_chunks)}")
        """运行完整的检查流程
        
        参数:
            chunks_path: 分块结果JSON文件路径
            indices_path: 分割点索引JSON文件路径
            max_tokens: 最大token限制
            index_tolerance: 索引连续性检查的容差
            
        返回:
            检查结果汇总
        """
        # 加载数据
        chunks = self.load_chunks(chunks_path)
        split_indices = self.load_split_indices(indices_path)
        
        # 检查索引连续性
        index_continuity = self.check_index_continuity(chunks, index_tolerance)
        
        # 检查token限制
        oversized_chunks = self.check_token_limits(chunks, max_tokens)
        
        # 计算统计信息
        chunk_stats = self.calculate_statistics(chunks)
        index_stats = self.analyze_split_indices(split_indices)
        
        # 打印结果
        print("=== 分块结果检查报告 ===")
        print(f"\n基本信息:")
        print(f"- 分块总数: {chunk_stats['total_chunks']}")
        print(f"- 分割点总数: {index_stats['total_indices']}")
        print(f"- 总token数: {chunk_stats['total_tokens']}")
        
        print(f"\n分块统计:")
        print(f"- 平均token数: {chunk_stats['avg_tokens']:.2f}")
        print(f"- 最小token数: {chunk_stats['min_tokens']}")
        print(f"- 最大token数: {chunk_stats['max_tokens']}")
        print(f"- 中位数token数: {chunk_stats['median_tokens']}")
        print(f"- 平均每块句子数: {chunk_stats['avg_sentences_per_chunk']:.2f}")
        
        print(f"\n分割点统计:")
        print(f"- 平均分割距离: {index_stats['avg_distance']:.2f}")
        print(f"- 最小分割距离: {index_stats['min_distance']}")
        print(f"- 最大分割距离: {index_stats['max_distance']}")
        
        print(f"\n质量检查:")
        print(f"- 索引连续性: {'通过' if index_continuity else '失败'} (容差: {index_tolerance} 字符)")
        print(f"- 超过token限制的分块数: {len(oversized_chunks)}")
        """运行完整的检查流程
        
        参数:
            chunks_path: 分块结果JSON文件路径
            indices_path: 分割点索引JSON文件路径
            max_tokens: 最大token限制
            index_tolerance: 索引连续性检查的容差
            
        返回:
            检查结果汇总
        """
        # 加载数据
        chunks = self.load_chunks(chunks_path)
        split_indices = self.load_split_indices(indices_path)
        
        # 检查索引连续性
        index_continuity = self.check_index_continuity(chunks, index_tolerance)
        
        # 检查token限制
        oversized_chunks = self.check_token_limits(chunks, max_tokens)
        
        # 计算统计信息
        chunk_stats = self.calculate_statistics(chunks)
        index_stats = self.analyze_split_indices(split_indices)
        
        # 打印结果
        print("=== 分块结果检查报告 ===")
        print(f"\n基本信息:")
        print(f"- 分块总数: {chunk_stats['total_chunks']}")
        print(f"- 分割点总数: {index_stats['total_indices']}")
        print(f"- 总token数: {chunk_stats['total_tokens']}")
        
        print(f"\n分块统计:")
        print(f"- 平均token数: {chunk_stats['avg_tokens']:.2f}")
        print(f"- 最小token数: {chunk_stats['min_tokens']}")
        print(f"- 最大token数: {chunk_stats['max_tokens']}")
        print(f"- 中位数token数: {chunk_stats['median_tokens']}")
        print(f"- 平均每块句子数: {chunk_stats['avg_sentences_per_chunk']:.2f}")
        
        print(f"\n分割点统计:")
        print(f"- 平均分割距离: {index_stats['avg_distance']:.2f}")
        print(f"- 最小分割距离: {index_stats['min_distance']}")
        print(f"- 最大分割距离: {index_stats['max_distance']}")
        
        print(f"\n质量检查:")
        print(f"- 索引连续性: {'通过' if index_continuity else '失败'} (容差: {index_tolerance} 字符)")
        print(f"- 超过token限制的分块数: {len(oversized_chunks)}")
        
        # 打印分块样例
        self.print_chunk_samples(chunks)
        
        # 返回检查结果
        return {
            'index_continuity': index_continuity,
            'oversized_chunks_count': len(oversized_chunks),
            'chunk_statistics': chunk_stats,
            'index_statistics': index_stats,
            'chunks': chunks,
            'split_indices': split_indices
        }

def main():
    """主函数"""
    # 设置文件路径
    chunks_path = os.path.join("results", "markdown_structure_chunks.json")
    indices_path = os.path.join("results", "split_indices.json")
    max_tokens = 500
    
    try:
        # 初始化检查器
        checker = SimilarityChunkChecker()
        
        # 运行完整检查
        results = checker.run_complete_check(chunks_path, indices_path, max_tokens)
        
        print("\n=== 检查完成 ===")
        
    except Exception as e:
        print(f"检查过程中出错: {e}")

if __name__ == "__main__":
    main()