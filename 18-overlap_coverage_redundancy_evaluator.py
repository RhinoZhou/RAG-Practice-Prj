#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重叠率与信息覆盖评估工具

功能：
- 在给定overlap值下，计算覆盖率、n-gram冗余率
- 进行关键句覆盖、n-gram重复度检测和阈值报警
- 给出分块优化建议

使用示例：
```python
from overlap_coverage_redundancy_evaluator import OverlapCoverageEvaluator

# 评估不同overlap值下的分块效果
evaluator = OverlapCoverageEvaluator()
results = evaluator.evaluate(
    text="你的文本内容...",
    chunk_size=200,
    overlap_values=[0, 20, 40, 60]
)

# 查看详细评估结果
for result in results:
    print(f"重叠值: {result['overlap']}")
    print(f"覆盖率: {result['coverage']:.2f}%")
    print(f"冗余率: {result['redundancy']:.2f}%")
    print(f"建议: {result['recommendation']}")
    print("---")
```

依赖：
- numpy
- re (用于中文句子分割)
- jieba (可选，用于中文分词)
"""

import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple, Any


class OverlapCoverageEvaluator:
    """重叠率与信息覆盖评估器
    
    该类用于评估不同重叠值下的文本分块效果，主要计算覆盖率和冗余率指标，
    并基于预设阈值给出优化建议，帮助用户选择最佳的分块策略。
    """
    
    def __init__(self,
                 n_gram_size: int = 3,
                 coverage_threshold: float = 90.0,
                 redundancy_warning_threshold: float = 30.0,
                 redundancy_danger_threshold: float = 50.0):
        """
        初始化评估器
        
        参数:
            n_gram_size: n-gram的大小，用于计算冗余率
            coverage_threshold: 覆盖率阈值，低于此值会提示增加重叠
            redundancy_warning_threshold: 冗余率警告阈值
            redundancy_danger_threshold: 冗余率危险阈值，高于此值会提示减少重叠
        """
        self.n_gram_size = n_gram_size
        self.coverage_threshold = coverage_threshold
        self.redundancy_warning_threshold = redundancy_warning_threshold
        self.redundancy_danger_threshold = redundancy_danger_threshold
        
        # 尝试导入jieba分词库，如果可用则使用
        self.use_jieba = False
        try:
            import jieba
            self.jieba = jieba
            self.use_jieba = True
        except ImportError:
            print("警告: jieba库未安装，将使用简单的空格分词法处理中文文本。")
            print("建议安装jieba: pip install jieba")
            
    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        根据chunk_size和overlap生成分块
        
        参数:
            text: 待分块的文本
            chunk_size: 每块的大小（字符数）
            overlap: 重叠部分的大小（字符数）
        
        返回:
            分块后的文本列表
        
        异常:
            ValueError: 当重叠大小大于或等于分块大小时抛出
        """
        if overlap >= chunk_size:
            raise ValueError("重叠大小不能大于或等于分块大小")
            
        chunks = []
        start = 0
        text_length = len(text)
        
        # 使用滑动窗口生成分块，考虑重叠
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            
            # 如果已经到文本末尾，停止循环
            if end >= text_length:
                break
                
            # 移动到下一个块的起始位置，考虑重叠
            start += (chunk_size - overlap)
            
        return chunks
        
    def _extract_key_sentences(self, text: str) -> List[str]:
        """
        从文本中提取关键句（针对中文文本优化的实现）
        
        参数:
            text: 输入文本
        
        返回:
            关键句列表
        """
        # 针对中文文本的句子分割正则表达式
        # 使用中文句子结束符：句号、问号、感叹号、省略号等
        sentence_pattern = re.compile(r'[^。！？.!?]+[。！？.!?]')
        sentences = sentence_pattern.findall(text)
        
        # 简单过滤：移除过短的句子（可能是不完整的句子）
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # 如果使用正则表达式没有分割出句子，尝试按换行符分割
        if not key_sentences:
            sentences = text.split('\n')
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
        return key_sentences
        
    def _calculate_coverage(self, original_text: str, chunks: List[str]) -> float:
        """
        计算覆盖率：关键句被覆盖的比例
        
        参数:
            original_text: 原始文本
            chunks: 分块后的文本列表
        
        返回:
            覆盖率百分比
        
        覆盖率计算逻辑:
        1. 从原始文本提取关键句
        2. 检查每个关键句是否完整地出现在任何一个分块中
        3. 覆盖率 = (被覆盖的关键句数量 / 总关键句数量) * 100
        """
        key_sentences = self._extract_key_sentences(original_text)
        
        if not key_sentences:
            # 如果无法提取关键句，返回默认覆盖率（表示无法评估）
            return 0.0
            
        covered_count = 0
        for sentence in key_sentences:
            # 检查句子是否在任何一个块中出现
            for chunk in chunks:
                if sentence in chunk:
                    covered_count += 1
                    break
                    
        coverage = (covered_count / len(key_sentences)) * 100
        return coverage
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        文本分词处理（针对中文优化）
        
        参数:
            text: 输入文本
        
        返回:
            分词后的词列表
        
        分词策略:
        1. 优先使用jieba分词（如果可用）
        2. 如果jieba不可用或出错，使用正则表达式提取中文字符和英文单词
        """
        # 移除多余的空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 如果安装了jieba，使用jieba进行中文分词
        if self.use_jieba:
            try:
                tokens = list(self.jieba.cut(text))
                # 过滤掉纯空格的token
                tokens = [token.strip() for token in tokens if token.strip()]
                return tokens
            except Exception as e:
                print(f"Jieba分词出错: {e}")
                # 出错时回退到简单分词
        
        # 简单的分词方法：保留中文字符和英文单词，其他字符作为分隔符
        # 这是一个简化的分词实现，实际应用中建议使用专门的中文分词库
        # 将文本转换为小写
        text = text.lower()
        
        # 使用正则表达式提取中文字符和英文单词
        # 中文字符范围：\u4e00-\u9fa5
        # 英文单词：[a-z0-9]+\'?[a-z0-9]*
        pattern = re.compile(r'([\u4e00-\u9fa5]+|[a-z0-9]+\'?[a-z0-9]*)')
        tokens = pattern.findall(text)
        
        return tokens
        
    def _generate_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """
        生成n-gram序列
        
        参数:
            tokens: 分词后的词列表
            n: n-gram的大小
        
        返回:
            n-gram元组列表
        
        n-gram生成逻辑:
        从token序列中，以滑动窗口的方式截取连续的n个token，形成n-gram元组
        """
        if len(tokens) < n:
            return []
            
        ngrams_list = []
        for i in range(len(tokens) - n + 1):
            ngrams_list.append(tuple(tokens[i:i+n]))
            
        return ngrams_list
        
    def _calculate_redundancy(self, chunks: List[str]) -> float:
        """
        计算n-gram冗余率
        
        参数:
            chunks: 分块后的文本列表
        
        返回:
            冗余率百分比
        
        冗余率计算逻辑:
        1. 对每个分块进行分词处理
        2. 生成每个分块的n-gram序列
        3. 统计所有n-gram的出现次数
        4. 计算重复出现的n-gram占总n-gram数量的比例
        """
        if len(chunks) <= 1:
            return 0.0  # 如果只有一个块，不存在冗余
            
        # 提取所有块的n-grams
        all_ngrams = []
        for chunk in chunks:
            # 分词处理
            tokens = self._tokenize_text(chunk)
            
            # 计算当前块的n-grams
            chunk_ngrams = self._generate_ngrams(tokens, self.n_gram_size)
            all_ngrams.extend(chunk_ngrams)
            
        # 计算重复的n-grams比例
        if not all_ngrams:
            return 0.0  # 如果没有有效的n-gram，返回0
            
        ngram_counts = Counter(all_ngrams)
        # 计算出现次数大于1的n-gram数量
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        
        # 计算冗余率（重复n-gram占总n-gram的比例）
        redundancy = (repeated_ngrams / len(ngram_counts)) * 100
        return redundancy
        
    def _generate_recommendation(self, coverage: float, redundancy: float) -> str:
        """
        根据覆盖率和冗余率生成优化建议
        
        参数:
            coverage: 覆盖率百分比
            redundancy: 冗余率百分比
        
        返回:
            建议字符串
        
        建议生成策略:
        1. 根据覆盖率与阈值比较，判断信息完整性是否达标
        2. 根据冗余率与阈值比较，判断重复内容是否过多
        3. 对于覆盖率不足且冗余率过高的矛盾情况，提供特殊建议
        """
        recommendations = []
        
        # 覆盖率评估
        if coverage < self.coverage_threshold:
            recommendations.append(f"覆盖率({coverage:.1f}%)低于目标阈值({self.coverage_threshold}%)，建议增加重叠值以提高信息完整性")
        else:
            recommendations.append(f"覆盖率({coverage:.1f}%)良好，信息完整性得到保障")
            
        # 冗余率评估
        if redundancy > self.redundancy_danger_threshold:
            recommendations.append(f"冗余率({redundancy:.1f}%)过高，存在大量重复内容，建议大幅减少重叠值或考虑合并块")
        elif redundancy > self.redundancy_warning_threshold:
            recommendations.append(f"冗余率({redundancy:.1f}%)偏高，存在一定重复内容，建议适当减少重叠值")
        else:
            recommendations.append(f"冗余率({redundancy:.1f}%)在合理范围内，重复内容可控")
            
        # 综合建议
        if coverage < self.coverage_threshold and redundancy > self.redundancy_danger_threshold:
            recommendations.append("当前配置存在矛盾：覆盖率不足但冗余率过高，建议调整分块策略或尝试不同的分块大小")
            
        return "\n".join(recommendations)
        
    def evaluate(self, text: str, chunk_size: int, overlap_values: List[int]) -> List[Dict[str, Any]]:
        """
        评估不同overlap值下的分块效果
        
        参数:
            text: 待评估的文本
            chunk_size: 分块大小
            overlap_values: 要测试的重叠值列表
        
        返回:
            包含每个重叠值评估结果的列表
        
        评估流程:
        1. 对每个重叠值，使用滑动窗口生成分块
        2. 计算每个分块方案的覆盖率和冗余率
        3. 根据覆盖率和冗余率生成优化建议
        4. 汇总所有评估结果
        """
        results = []
        
        for overlap in overlap_values:
            # 生成分块
            chunks = self._split_into_chunks(text, chunk_size, overlap)
            
            # 计算覆盖率
            coverage = self._calculate_coverage(text, chunks)
            
            # 计算冗余率
            redundancy = self._calculate_redundancy(chunks)
            
            # 生成建议
            recommendation = self._generate_recommendation(coverage, redundancy)
            
            # 存储结果
            results.append({
                'overlap': overlap,
                'chunks_count': len(chunks),
                'coverage': coverage,
                'redundancy': redundancy,
                'recommendation': recommendation,
                'chunks': chunks  # 包含实际分块，便于进一步分析
            })
            
        return results
        
    def evaluate_and_print(self, text: str, chunk_size: int, overlap_values: List[int]) -> None:
        """
        评估不同overlap值并打印结果
        
        参数:
            text: 待评估的文本
            chunk_size: 分块大小
            overlap_values: 要测试的重叠值列表
        
        功能：
        调用evaluate方法进行评估，并格式化打印评估结果，方便用户查看
        """
        results = self.evaluate(text, chunk_size, overlap_values)
        
        print("===== 重叠率与信息覆盖评估结果 =====")
        print(f"分块大小: {chunk_size}字符")
        print(f"评估的重叠值: {overlap_values}")
        print(f"n-gram大小: {self.n_gram_size}\n")
        
        for i, result in enumerate(results):
            print(f"--- 评估 #{i+1} (重叠={result['overlap']}) ---")
            print(f"分块数量: {result['chunks_count']}")
            print(f"覆盖率: {result['coverage']:.2f}%")
            print(f"冗余率: {result['redundancy']:.2f}%")
            print(f"建议: {result['recommendation']}")
            print()
            
        print("====================================")


# 示例用法
if __name__ == "__main__":
    # 示例文本（关于人工智能的简单介绍）
    sample_text = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在开发能够模拟和执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、感知、理解自然语言、识别模式、做出决策等。

人工智能的发展可以追溯到20世纪50年代，当时计算机科学家开始探索如何让机器模拟人类的思维过程。早期的AI研究主要集中在符号逻辑和问题解决上，但进展相对缓慢，这一时期被称为"AI寒冬"。

直到21世纪初，随着计算能力的提升和大数据的出现，AI研究迎来了新的春天。特别是深度学习技术的突破，使得机器能够从大量数据中学习复杂的模式和表示，从而在图像识别、语音识别、自然语言处理等领域取得了重大进展。

今天，AI技术已经广泛应用于各个领域，如医疗保健、金融服务、交通运输、教育、娱乐等。它正在改变我们的工作方式、生活方式和社会结构。然而，AI的快速发展也带来了一系列伦理和社会问题，如隐私保护、就业影响、算法偏见等，需要我们认真思考和应对。"""
    
    # 创建评估器实例
    evaluator = OverlapCoverageEvaluator(
        n_gram_size=3,
        coverage_threshold=90.0,
        redundancy_warning_threshold=30.0,
        redundancy_danger_threshold=50.0
    )
    
    # 评估不同重叠值
    evaluator.evaluate_and_print(
        text=sample_text,
        chunk_size=200,
        overlap_values=[0, 20, 40, 60]
    )