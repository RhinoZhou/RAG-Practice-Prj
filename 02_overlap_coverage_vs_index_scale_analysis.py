#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统中的分块重叠-覆盖率-索引规模分析实验
评估不同 chunk_overlap 对关键信息覆盖率和索引规模的影响

主要功能:
- 从语料中抽取关键句集（基于TF-IDF）
- 测试不同overlap值（0, 10%, 15%, 20%, 30%）的表现
- 计算关键信息覆盖率、索引规模和冗余度
- 生成覆盖率-规模曲线图表和结果数据

输入：文本语料、关键句抽取阈值
输出：CSV/JSON（overlap、coverage、chunks）
"""

# 导入必要的库
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class ExperimentConfig:
    """实验配置类
    
    属性:
        overlap_ratios: 要测试的重叠比例列表
        chunk_size: 固定的分块大小
        top_k_sentences: 抽取的关键句数量
        corpus_file: 语料文件路径
        output_dir: 结果输出目录
    """
    overlap_ratios: List[float] = None
    chunk_size: int = 512
    top_k_sentences: int = 20
    corpus_file: str = "corpus.txt"
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.overlap_ratios is None:
            self.overlap_ratios = [0.0, 0.1, 0.15, 0.2, 0.3]

@dataclass
class ExperimentResult:
    """实验结果类
    
    属性:
        overlap_ratio: 重叠比例
        overlap_size: 重叠字符数
        num_chunks: 分块数量
        coverage: 关键信息覆盖率
        redundancy: 信息冗余度
        avg_chunk_size: 平均块大小
    """
    overlap_ratio: float
    overlap_size: int
    num_chunks: int
    coverage: float
    redundancy: float
    avg_chunk_size: float

class KeySentenceExtractor:
    """关键句抽取类
    
    功能：从文本中抽取关键句子（基于TF-IDF算法）
    实现细节：使用TF-IDF（词频-逆文档频率）算法计算句子的重要性，提取最具代表性的句子
    
    优化提示：当前实现使用简单的TF-IDF计算，可以改进为使用更先进的方法，
    如TextRank或预训练语言模型来提取关键句子
    """
    
    def __init__(self, top_k: int = 20):
        """初始化关键句抽取器
        
        参数:
            top_k: 要提取的关键句数量
        """
        self.top_k = top_k
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', 
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', 
            '没有', '看', '好', '自己', '这', '我们', '他们', '这个', '那个'
        }
    
    def extract_key_sentences(self, text: str) -> List[str]:
        """从文本中抽取关键句子
        
        参数:
            text: 输入文本
        
        返回:
            关键句子列表，按重要性降序排列
        
        实现说明：
        1. 使用正则表达式将文本分割成句子
        2. 过滤空句子和太短的句子（长度小于10字符）
        3. 计算每个句子的TF-IDF向量
        4. 计算句子向量的模长作为句子重要性得分
        5. 按得分排序并提取前top_k个句子作为关键句
        
        优化提示：可以扩展分句逻辑以支持更多标点符号和复杂句式，
        或者使用更复杂的评分方法来更好地反映句子的重要性
        """
        # 分句（基于中文句号、问号、感叹号等）
        sentences = re.split(r'[。！？；]', text)
        # 过滤空句子和太短的句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        # 使用TF-IDF计算句子重要性
        # 简单实现TF-IDF
        sentence_vectors = self._tfidf_vectors(sentences)
        
        # 计算每个句子的得分（向量模长）
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # 计算句子向量的模长作为得分
            score = sum(sentence_vectors[i].values())
            sentence_scores.append((sentence, score))
        
        # 按得分排序并取top_k
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_sentences = [sentence for sentence, _ in sentence_scores[:self.top_k]]
        
        return top_k_sentences
    
    def _tfidf_vectors(self, sentences: List[str]) -> List[Dict[str, float]]:
        """计算句子的TF-IDF向量
        
        参数:
            sentences: 句子列表
        
        返回:
            句子TF-IDF向量列表，每个向量是词汇到TF-IDF值的映射
        
        实现说明：
        1. 对每个句子进行分词并过滤停用词
        2. 计算词频（TF）：每个词在句子中出现的频率
        3. 计算逆文档频率（IDF）：衡量词的重要性
        4. 计算TF-IDF：TF值乘以IDF值
        
        注意：这里使用了简单的TF-IDF实现，对于更复杂的场景，可以使用scikit-learn的TfidfVectorizer
        """
        # 分词并过滤停用词
        tokenized_sentences = []
        for sentence in sentences:
            # 简单分词
            words = re.findall(r'\w+', sentence.lower())
            # 过滤停用词
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
            tokenized_sentences.append(filtered_words)
        
        # 计算TF-IDF
        # 1. 计算词频（TF）
        tf_matrix = []
        for tokens in tokenized_sentences:
            tf = Counter(tokens)
            total_words = len(tokens)
            if total_words > 0:
                # 归一化TF
                for word in tf:
                    tf[word] /= total_words
            tf_matrix.append(tf)
        
        # 2. 计算逆文档频率（IDF）
        # 统计包含每个词的文档数
        df = defaultdict(int)
        for tokens in tokenized_sentences:
            for word in set(tokens):
                df[word] += 1
        
        # 计算IDF
        idf = {}
        total_docs = len(sentences)
        for word, doc_count in df.items():
            idf[word] = math.log(total_docs / (doc_count + 1)) + 1  # 加1平滑
        
        # 3. 计算TF-IDF
        tfidf_vectors = []
        for tf in tf_matrix:
            tfidf = {}
            for word, tf_value in tf.items():
                tfidf[word] = tf_value * idf.get(word, 0)
            tfidf_vectors.append(tfidf)
        
        return tfidf_vectors

class TextChunker:
    """文本分块器类
    
    功能：将长文本按照指定大小和重叠比例分割成多个文本块
    """
    
    def __init__(self, chunk_size: int, overlap_ratio: float = 0.1):
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分割成块
        
        参数:
            text: 要分块的文本
        
        返回:
            文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在句号处分割
            if end < len(text):
                # 寻找最近的句号
                period_pos = text.rfind('。', start, end)
                if period_pos > start + self.chunk_size // 2:  # 确保不会太短
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个块的起始位置（考虑重叠）
            start = end - self.overlap_size
            if start >= len(text):
                break
        
        return chunks

class CoverageAnalyzer:
    """覆盖率分析器类
    
    功能：计算关键信息覆盖率和冗余度，评估分块策略的有效性
    实现细节：基于关键词覆盖的方法，分析不同分块重叠比例下的关键信息保留情况和冗余程度
    
    核心指标：
    - 覆盖率：衡量关键信息被保留的程度
    - 冗余度：衡量信息重复的程度
    - 平均块大小：评估分块的一致性
    """
    
    @staticmethod
    def calculate_coverage(key_sentences: List[str], chunks: List[str]) -> Tuple[float, float, float]:
        """计算覆盖率、冗余度和平均块大小
        
        参数:
            key_sentences: 关键句子列表（由KeySentenceExtractor提取）
            chunks: 文本块列表（由TextChunker生成）
        
        返回:
            (覆盖率, 冗余度, 平均块大小)元组
        
        实现说明：
        1. 从关键句子中提取所有关键词
        2. 分析文本块集合中包含的关键词
        3. 计算覆盖率：覆盖的关键词数 / 总关键词数
        4. 计算冗余度：总词数 / 唯一词数
        5. 计算平均块大小：所有块的平均字符数
        
        覆盖率解释：值为1表示所有关键信息都被保留在至少一个文本块中
        冗余度解释：值为1表示没有重复信息，值越大表示重复信息越多
        
        优化提示：可以考虑关键词的重要性权重，对不同重要性的关键词给予不同的权重，
        以获得更准确的覆盖率评估
        """
        if not key_sentences or not chunks:
            return 0.0, 0.0, 0.0
        
        # 将关键句子转换为关键词集合
        key_words = set()
        for sentence in key_sentences:
            words = re.findall(r'\w+', sentence.lower())
            key_words.update(words)
        
        # 计算每个块覆盖的关键词和所有块包含的唯一词
        covered_words = set()  # 被至少一个块覆盖的关键词
        all_words = set()      # 所有块包含的唯一词
        
        for chunk in chunks:
            words = re.findall(r'\w+', chunk.lower())
            chunk_words = set(words)
            all_words.update(chunk_words)
            # 找出与关键句子的交集
            covered_words.update(chunk_words & key_words)
        
        # 计算覆盖率 - 衡量关键信息被保留的程度
        coverage = len(covered_words) / len(key_words) if key_words else 0.0
        
        # 计算冗余度（所有块的总词数/唯一词数）- 衡量信息重复的程度
        total_words = sum(len(re.findall(r'\w+', chunk.lower())) for chunk in chunks)
        unique_words = len(all_words)
        redundancy = total_words / unique_words if unique_words else 0.0
        
        # 计算平均块大小 - 评估分块的一致性
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0
        
        return coverage, redundancy, avg_chunk_size

class OverlapCoverageExperiment:
    """重叠覆盖率实验主类
    
    功能：执行分块重叠-覆盖率-索引规模分析实验
    实验目的：探索分块重叠比例对RAG系统性能的影响，找出最佳重叠配置
    
    实现细节：
    - 基于不同的重叠比例对文本进行分块
    - 分析每个重叠比例下的覆盖率、生成的块数、冗余度等指标
    - 绘制重叠比例与各指标的关系曲线
    - 总结最佳重叠配置建议
    """
    
    def __init__(self, config: ExperimentConfig):
        """初始化实验对象
        
        参数:
            config: 实验配置对象，包含实验所需的各项参数
        
        属性初始化:
            results: 存储实验结果的列表
            key_sentences: 存储从语料中提取的关键句子
            corpus: 存储加载的语料文本
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        self.key_sentences: List[str] = []
        self.corpus: str = ""
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def load_data(self) -> str:
        """加载语料
        
        返回:
            语料文本
        
        注意事项：确保corpus_file路径正确，文件格式符合预期的文本结构
        """
        with open(self.config.corpus_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_key_sentences(self) -> List[str]:
        """提取关键句子
        
        返回:
            关键句子列表
        
        实现细节：
        1. 创建KeySentenceExtractor实例，使用配置的top_k_sentences参数
        2. 调用extract_key_sentences方法从加载的语料中提取关键句子
        3. 这些关键句子将作为评估覆盖率的基准
        
        优化提示：可以根据语料类型调整提取器参数，提高关键句子提取的准确性
        """
        extractor = KeySentenceExtractor(self.config.top_k_sentences)
        return extractor.extract_key_sentences(self.corpus)
    
    def run_experiment(self) -> List[ExperimentResult]:
        """执行重叠-覆盖率-索引规模分析实验
        
        功能：遍历不同的重叠比例，分析覆盖率、块数和冗余度
        
        实验流程：
        1. 加载语料文本数据
        2. 从语料中提取关键句子
        3. 打印实验配置和初始信息
        4. 遍历每个重叠比例执行以下操作：
           a. 创建TextChunker实例进行分块
           b. 对语料文本进行分块处理
           c. 统计块数
           d. 使用CoverageAnalyzer计算覆盖率和冗余度
           e. 创建实验结果对象并存储
           f. 打印当前重叠比例的实验结果
        5. 实验完成后打印总结分割线
        6. 返回实验结果列表
        
        实现细节：
        - 重叠大小 = 块大小 × 重叠比例
        - 每个重叠比例会生成独立的分块器实例
        - 实验结果以表格形式实时输出，方便观察
        
        返回:
            实验结果列表，包含每个重叠比例的覆盖率、块数、冗余度等指标
        """
        # 加载语料
        self.corpus = self.load_data()
        
        # 提取关键句子
        self.key_sentences = self.extract_key_sentences()
        
        print(f"加载语料: {len(self.corpus)} 字符")
        print(f"提取关键句子: {len(self.key_sentences)} 句")
        print("开始实验...")
        print("=" * 100)
        print(f"{'重叠比例':<10} {'重叠字符数':<10} {'块数':<10} {'覆盖率':<10} {'冗余度':<10} {'平均块大小':<12}")
        print("=" * 100)
        
        # 对每个重叠比例进行实验
        for overlap_ratio in self.config.overlap_ratios:
            # 创建分块器
            chunker = TextChunker(self.config.chunk_size, overlap_ratio)
            
            # 分块
            chunks = chunker.chunk_text(self.corpus)
            num_chunks = len(chunks)
            
            # 计算覆盖率和冗余度
            coverage, redundancy, avg_chunk_size = CoverageAnalyzer.calculate_coverage(
                self.key_sentences, chunks
            )
            
            # 创建实验结果对象
            result = ExperimentResult(
                overlap_ratio=overlap_ratio,
                overlap_size=chunker.overlap_size,
                num_chunks=num_chunks,
                coverage=coverage,
                redundancy=redundancy,
                avg_chunk_size=avg_chunk_size
            )
            
            # 添加到结果列表
            self.results.append(result)
            
            # 打印结果
            print(f"{result.overlap_ratio:<10.2f} {result.overlap_size:<10} {result.num_chunks:<10} "
                  f"{result.coverage:<10.4f} {result.redundancy:<10.4f} {result.avg_chunk_size:<12.2f}")
        
        print("=" * 100)
        
        return self.results
    
    def save_results(self) -> None:
        """保存实验结果到文件
        
        功能：将实验结果以JSON和CSV两种格式保存到指定的输出目录
        
        实现细节：
        1. 根据配置获取结果保存目录路径
        2. 创建结果数据字典列表
        3. 以JSON格式保存结果，方便后续分析和可视化
        4. 将结果转换为pandas DataFrame并保存为CSV格式，便于表格处理
        5. 打印保存成功的信息
        
        数据字段：
        - overlap_ratio: 重叠比例
        - overlap_size: 重叠字符数
        - num_chunks: 块数
        - coverage: 覆盖率
        - redundancy: 冗余度
        - avg_chunk_size: 平均块大小
        
        保存路径：根据配置的output_dir参数确定
        """
        # 保存为JSON文件
        results_dir = Path(self.config.output_dir)
        json_path = results_dir / "overlap_coverage_results.json"
        
        results_dict = []
        for result in self.results:
            results_dict.append({
                "overlap_ratio": result.overlap_ratio,
                "overlap_size": result.overlap_size,
                "num_chunks": result.num_chunks,
                "coverage": result.coverage,
                "redundancy": result.redundancy,
                "avg_chunk_size": result.avg_chunk_size
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        # 保存为CSV文件
        csv_path = results_dir / "overlap_coverage_results.csv"
        df = pd.DataFrame(results_dict)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"实验结果已保存到 {json_path} 和 {csv_path}")
    
    def plot_results(self) -> None:
        """绘制实验结果图表
        
        功能：可视化重叠比例与覆盖率、块数、冗余度的关系，并绘制覆盖率-规模曲线
        
        实现细节：
        1. 获取结果保存目录路径
        2. 从实验结果中提取所需数据
        3. 创建一个2x2的子图布局
        4. 绘制四个不同的关系图
        5. 调整子图间距，确保清晰显示
        6. 保存图表并关闭
        7. 打印保存成功的信息
        
        图表内容：
        - 覆盖率与重叠比例关系图：蓝色曲线，展示重叠比例对关键信息保留的影响
        - 块数与重叠比例关系图：红色曲线，展示重叠比例对索引规模的影响
        - 冗余度与重叠比例关系图：绿色曲线，展示重叠比例对信息重复的影响
        - 覆盖率-规模曲线图：紫色曲线，展示索引规模（块数）与覆盖率的关系，并标注不同的重叠比例
        
        图表特点：
        - 所有图表均包含网格线，便于数据读取
        - 覆盖率相关图表Y轴限制在0-1.1，确保比例一致性
        - 重叠比例转换为百分比显示，增强可读性
        - 覆盖率-规模曲线图使用标注显示每个点对应的重叠比例
        - 优化了图表标题和图例，使结果更加直观
        
        图表保存位置：根据配置的output_dir参数确定
        """
        # 创建图表目录
        results_dir = Path(self.config.output_dir)
        plot_path = results_dir / "overlap_coverage_plots.png"
        
        # 提取数据
        overlap_ratios = [result.overlap_ratio * 100 for result in self.results]  # 转换为百分比
        coverages = [result.coverage for result in self.results]  
        num_chunks = [result.num_chunks for result in self.results]  
        redundancies = [result.redundancy for result in self.results]  
        
        # 创建一个包含多个子图的图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制覆盖率与重叠比例的关系 - 改进显示效果
        axs[0, 0].plot(overlap_ratios, coverages, 'o-', color='blue', markersize=8, linewidth=2)
        for i, txt in enumerate(coverages):
            axs[0, 0].annotate(f'{txt:.4f}', (overlap_ratios[i], coverages[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axs[0, 0].set_title('关键信息覆盖率与重叠比例的关系', fontsize=12)
        axs[0, 0].set_xlabel('重叠比例(%)', fontsize=10)
        axs[0, 0].set_ylabel('覆盖率', fontsize=10)
        axs[0, 0].set_ylim(max(0, min(coverages) - 0.05), min(1.1, max(coverages) + 0.05))
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 绘制块数与重叠比例的关系
        axs[0, 1].plot(overlap_ratios, num_chunks, 'o-', color='red', markersize=8, linewidth=2)
        for i, txt in enumerate(num_chunks):
            axs[0, 1].annotate(f'{txt}', (overlap_ratios[i], num_chunks[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axs[0, 1].set_title('块数与重叠比例的关系', fontsize=12)
        axs[0, 1].set_xlabel('重叠比例(%)', fontsize=10)
        axs[0, 1].set_ylabel('块数', fontsize=10)
        axs[0, 1].set_ylim(max(0, min(num_chunks) - 1), max(num_chunks) + 1)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 绘制冗余度与重叠比例的关系
        axs[1, 0].plot(overlap_ratios, redundancies, 'o-', color='green', markersize=8, linewidth=2)
        for i, txt in enumerate(redundancies):
            axs[1, 0].annotate(f'{txt:.4f}', (overlap_ratios[i], redundancies[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axs[1, 0].set_title('信息冗余度与重叠比例的关系', fontsize=12)
        axs[1, 0].set_xlabel('重叠比例(%)', fontsize=10)
        axs[1, 0].set_ylabel('冗余度', fontsize=10)
        axs[1, 0].set_ylim(max(0, min(redundancies) - 0.1), max(redundancies) + 0.1)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 绘制覆盖率与块数的关系（覆盖率-规模曲线）
        axs[1, 1].plot(num_chunks, coverages, 'o-', color='purple', markersize=8, linewidth=2)
        for i, txt in enumerate(overlap_ratios):
            axs[1, 1].annotate(f'{txt}%', (num_chunks[i], coverages[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axs[1, 1].set_title('覆盖率-规模曲线（不同重叠比例）', fontsize=12)
        axs[1, 1].set_xlabel('块数（索引规模）', fontsize=10)
        axs[1, 1].set_ylabel('覆盖率', fontsize=10)
        axs[1, 1].set_ylim(max(0, min(coverages) - 0.05), min(1.1, max(coverages) + 0.05))
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 添加整体标题
        fig.suptitle('RAG系统分块重叠-覆盖率-索引规模分析', fontsize=16, y=0.98)
        
        # 调整子图之间的间距
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"实验图表已保存到 {plot_path}")
    
    def print_summary(self) -> None:
        """打印实验结果总结
        
        功能：分析实验结果并打印关键发现，包括嵌入质量、召回率调控和延迟-吞吐权衡
        
        实现细节：
        1. 检查是否有实验结果可总结
        2. 找出覆盖率最高的实验结果
        3. 打印最高覆盖率及其对应的重叠比例和块数
        4. 分析并打印重叠比例与块数的关系趋势
        5. 分析并打印重叠比例与冗余度的关系趋势
        6. 添加关于嵌入质量的分析
        7. 添加关于召回率调控的分析
        8. 添加关于延迟与吞吐权衡的分析
        
        总结内容：
        - 最高覆盖率及其最优配置
        - 重叠比例增加对块数的影响
        - 重叠比例增加对冗余度的影响
        - 嵌入质量分析
        - 召回率调控分析
        - 延迟与吞吐权衡分析
        
        设计目的：提供全面的实验结果概览，帮助理解重叠比例对RAG系统多方面性能的影响
        """
        if not self.results:
            print("没有实验结果可供总结")
            return
        
        # 找到最高覆盖率的结果
        best_coverage_result = max(self.results, key=lambda x: x.coverage)
        
        print("实验总结:")
        print(f"- 最高覆盖率: {best_coverage_result.coverage:.4f} (重叠比例: {best_coverage_result.overlap_ratio:.2f}, 块数: {best_coverage_result.num_chunks})")
        print(f"- 重叠比例与块数关系: 随着重叠比例增加，块数从 {self.results[0].num_chunks} 增加到 {self.results[-1].num_chunks}")
        print(f"- 冗余度变化: 随着重叠比例增加，冗余度从 {self.results[0].redundancy:.4f} 增加到 {self.results[-1].redundancy:.4f}")
        
        # 添加关于嵌入质量的分析
        print("\n深度分析 - 嵌入质量考量:")
        print("  嵌入质量并不只取决于模型本身，它同样依赖输入片段的语义纯度。")
        print(f"  - 当使用固定块大小({self.config.chunk_size}字符)时，粒度过细可能会把完整的概念拆散")
        print(f"  - 目前设置的块大小({self.config.chunk_size}字符)是一个平衡点，需根据具体领域和查询特点进行调整")
        print("  - 建议：可以通过段落边界检测来优化分块，确保概念的完整性")
        
        # 添加关于召回率调控的分析
        print("\n深度分析 - 召回率的调控:")
        print(f"  - 重叠滑窗是增强召回率的常用手段，实验中覆盖比例从{self.results[0].overlap_ratio*100:.0f}%增加到{self.results[-1].overlap_ratio*100:.0f}%")
        print("  - 重叠能减小'被切断信息'带来的漏检，但同时增加了存储与索引规模")
        print(f"  - 本次实验中，在重叠比例为{best_coverage_result.overlap_ratio*100:.0f}%时达到最高覆盖率{best_coverage_result.coverage:.4f}")
        print("  - 建议：结合段落相似度或主题边界检测，可以实现更智能的动态分块策略")
        
        # 添加关于延迟与吞吐权衡的分析
        print("\n深度分析 - 延迟与吞吐的权衡:")
        print("  - 短块通常带来更低的单块匹配延迟，但会增加检索次序与合并成本")
        print("  - 长块相反，单块处理时间较长但检索次数较少")
        print(f"  - 本次实验中，随着块数从{self.results[0].num_chunks}增加到{self.results[-1].num_chunks}，存储成本和处理时间也相应增加")
        print(f"  - 冗余度从{self.results[0].redundancy:.4f}增长到{self.results[-1].redundancy:.4f}，表明存储开销与重叠率几乎线性增长")
        print("  - 建议：工程上应使用目标延迟SLA约束，去反推chunk_size与top-k的最佳组合"),

def main():
    """主函数"""
    print("RAG系统分块重叠-覆盖率-索引规模分析实验")
    print("="*50)
    
    # 创建实验配置
    config = ExperimentConfig(
        overlap_ratios=[0.0, 0.1, 0.15, 0.2, 0.3],  # 0, 10%, 15%, 20%, 30%
        chunk_size=512,
        top_k_sentences=20,
        corpus_file="corpus.txt",
        output_dir="results"
    )
    
    # 创建实验实例
    experiment = OverlapCoverageExperiment(config)
    
    try:
        # 运行实验
        results = experiment.run_experiment()
        
        # 保存结果
        experiment.save_results()
        
        # 绘制图表
        experiment.plot_results()
        
        # 打印总结
        experiment.print_summary()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保语料文件存在")
    except Exception as e:
        print(f"实验过程中出现错误: {e}")

# 创建results目录（确保程序运行时目录存在）
Path("results").mkdir(exist_ok=True)

if __name__ == "__main__":
    main()