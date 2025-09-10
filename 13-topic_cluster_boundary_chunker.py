#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题聚类驱动边界分块器
基于简化BERTopic的实现，通过句子向量聚类和簇序列变化来确定分块边界
"""

import json
import os
import re
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Tuple

# 尝试导入必要的库，如果失败则安装
def install_dependencies():
    try:
        import sklearn
        from sklearn.cluster import KMeans, DBSCAN
        import tiktoken
        import PyPDF2
    except ImportError:
        print("正在安装必要的依赖...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn", "numpy", "tiktoken", "PyPDF2"])
        print("依赖安装完成")

# 安装依赖
install_dependencies()

# 导入已安装的库
from sklearn.cluster import KMeans, DBSCAN
import tiktoken
import PyPDF2

@dataclass
class ClusterChunk:
    """表示一个主题聚类分块的数据类"""
    chunk_id: int
    text: str
    start_index: int
    end_index: int
    tokens_count: int
    sentences_count: int
    cluster_id: int  # 新增：分块所属的簇ID

class SentenceVectorEncoder:
    """句子向量编码器（简化版）"""
    def __init__(self):
        # 使用简单的词频统计作为向量表示（实际应用中可替换为预训练模型如BERT）
        self.vocab = {}
        self.vocab_size = 1000  # 限制词汇表大小
        self.punct_pattern = re.compile(r'[.!?，。！？]')
        
    def tokenize(self, text: str) -> List[str]:
        """简单的分词方法"""
        # 基本分词，仅按空格分割并转为小写
        return re.findall(r'\b\w+\b', text.lower())
        
    def encode(self, sentences: List[str]) -> np.ndarray:
        """将句子列表编码为向量矩阵"""
        # 构建词汇表
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for token in tokens:
                if token not in self.vocab:
                    if len(self.vocab) < self.vocab_size:
                        self.vocab[token] = len(self.vocab)
        
        # 创建向量矩阵
        vectors = np.zeros((len(sentences), self.vocab_size))
        for i, sentence in enumerate(sentences):
            tokens = self.tokenize(sentence)
            for token in tokens:
                if token in self.vocab:
                    vectors[i, self.vocab[token]] += 1
        
        # 归一化向量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零错误
        return vectors / norms

class TopicClusterBoundaryChunker:
    """主题聚类驱动的边界分块器"""
    def __init__(self, 
                 cluster_method: str = "kmeans", 
                 n_clusters: int = 10, 
                 eps: float = 0.5, 
                 min_samples: int = 5,
                 max_tokens: int = 500):
        """
        初始化主题聚类分块器
        
        参数:
        - cluster_method: 聚类方法，可选 'kmeans' 或 'dbscan'
        - n_clusters: KMeans聚类的簇数量（当cluster_method为'kmeans'时有效）
        - eps: DBSCAN聚类的密度参数（当cluster_method为'dbscan'时有效）
        - min_samples: DBSCAN聚类的最小样本数（当cluster_method为'dbscan'时有效）
        - max_tokens: 每个分块的最大token数量
        """
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.max_tokens = max_tokens
        self.encoder = SentenceVectorEncoder()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.cluster_model = None
        
    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """将文本分割成句子，并记录每个句子的起始和结束位置"""
        sentences_with_positions = []
        sentences = re.split(r'([.!?，。！？])', text)
        
        current_pos = 0
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加标点符号
            
            # 跳过空句子
            if sentence.strip() == "":
                current_pos += len(sentence)
                continue
            
            start_pos = current_pos
            end_pos = current_pos + len(sentence)
            sentences_with_positions.append((sentence.strip(), start_pos, end_pos))
            current_pos = end_pos
        
        return sentences_with_positions
    
    def cluster_sentences(self, sentences: List[str]) -> List[int]:
        """对句子进行聚类，返回每个句子的簇标签"""
        if len(sentences) == 0:
            return []
        
        # 编码句子
        vectors = self.encoder.encode(sentences)
        
        # 根据选择的方法进行聚类
        if self.cluster_method == "kmeans":
            # 调整簇数量，确保不大于句子数量
            n_clusters = min(self.n_clusters, len(sentences))
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = self.cluster_model.fit_predict(vectors)
        elif self.cluster_method == "dbscan":
            self.cluster_model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = self.cluster_model.fit_predict(vectors)
        else:
            raise ValueError("聚类方法必须是 'kmeans' 或 'dbscan'")
        
        return labels.tolist()
    
    def create_chunks_by_cluster_boundaries(self, 
                                           sentences_with_positions: List[Tuple[str, int, int]], 
                                           cluster_labels: List[int]) -> List[ClusterChunk]:
        """根据簇标签的变化创建分块"""
        if not sentences_with_positions or not cluster_labels:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_cluster = cluster_labels[0]
        current_start_index = sentences_with_positions[0][1]
        current_token_count = 0
        chunk_id = 0
        
        for i, (sentence, start_pos, end_pos) in enumerate(sentences_with_positions):
            sentence_token_count = len(self.tokenizer.encode(sentence))
            current_label = cluster_labels[i]
            
            # 检查是否需要创建新分块
            # 1. 簇标签变化
            # 2. 当前分块的token数量超过最大限制
            if (current_label != current_cluster or 
                current_token_count + sentence_token_count > self.max_tokens) and current_chunk_sentences:
                # 创建新分块
                chunk_text = " ".join(current_chunk_sentences)
                chunk_end_index = sentences_with_positions[i-1][2]
                
                chunks.append(ClusterChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_index=current_start_index,
                    end_index=chunk_end_index,
                    tokens_count=current_token_count,
                    sentences_count=len(current_chunk_sentences),
                    cluster_id=current_cluster
                ))
                
                # 重置当前分块
                chunk_id += 1
                current_chunk_sentences = [sentence]
                current_start_index = start_pos
                current_token_count = sentence_token_count
                current_cluster = current_label
            else:
                # 继续添加到当前分块
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_token_count
        
        # 添加最后一个分块
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_end_index = sentences_with_positions[-1][2]
            
            chunks.append(ClusterChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_index=current_start_index,
                end_index=chunk_end_index,
                tokens_count=current_token_count,
                sentences_count=len(current_chunk_sentences),
                cluster_id=current_cluster
            ))
        
        return chunks
    
    def chunk_text(self, text: str) -> List[ClusterChunk]:
        """对输入文本进行分块"""
        # 分句
        sentences_with_positions = self.split_into_sentences(text)
        sentences = [sent[0] for sent in sentences_with_positions]
        
        # 聚类
        cluster_labels = self.cluster_sentences(sentences)
        
        # 处理DBSCAN的-1标签（噪声点）
        # 策略：将噪声点分配给上下文最相关的簇，以保持主题连贯性
        for i, label in enumerate(cluster_labels):
            if label == -1 and i > 0 and cluster_labels[i-1] != -1:
                # 如果前一个不是噪声点，直接继承其簇标签
                cluster_labels[i] = cluster_labels[i-1]
            elif label == -1 and i > 0 and cluster_labels[i-1] == -1:
                # 如果前一个也是噪声点，寻找最近的非噪声点
                j = i - 1
                while j >= 0 and cluster_labels[j] == -1:
                    j -= 1
                if j >= 0:
                    # 找到前面的非噪声点
                    cluster_labels[i] = cluster_labels[j]
                else:
                    # 如果前面没有非噪声点，寻找后面的第一个非噪声点
                    j = i + 1
                    while j < len(cluster_labels) and cluster_labels[j] == -1:
                        j += 1
                    if j < len(cluster_labels):
                        cluster_labels[i] = cluster_labels[j]
        
        # 根据簇边界创建分块
        chunks = self.create_chunks_by_cluster_boundaries(sentences_with_positions, cluster_labels)
        
        return chunks
    
    def save_chunks_to_json(self, chunks: List[ClusterChunk], output_path: str) -> None:
        """将分块结果保存为JSON文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换为字典并保存
        chunks_dict = [asdict(chunk) for chunk in chunks]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)

def read_pdf_text(pdf_path: str) -> str:
    """从PDF文件中读取文本"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def main():
    """主函数，演示主题聚类分块器的使用"""
    # 设置文件路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    pdf_path = os.path.join(data_dir, "大脑中动脉狭窄与闭塞致脑梗死的影像特点及发病机制的研究.pdf")
    output_json_path = os.path.join(results_dir, "topic_cluster_chunks.json")
    
    # 读取PDF文本
    print(f"正在读取PDF文件: {pdf_path}")
    text = read_pdf_text(pdf_path)
    print(f"文本读取完成，总长度: {len(text)} 字符")
    
    # 创建分块器并进行分块
    print("正在进行主题聚类分块...")
    chunker = TopicClusterBoundaryChunker(
        cluster_method="kmeans",  # 使用KMeans聚类
        n_clusters=20,  # 设置簇数量
        max_tokens=500  # 每个分块的最大token数量
    )
    
    chunks = chunker.chunk_text(text)
    
    # 保存结果
    chunker.save_chunks_to_json(chunks, output_json_path)
    
    # 打印统计信息
    total_tokens = sum(chunk.tokens_count for chunk in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    
    print(f"分块完成！")
    print(f"分块总数: {len(chunks)}")
    print(f"总token数: {total_tokens}")
    print(f"平均每个分块的token数: {avg_tokens:.2f}")
    print(f"分块结果已保存至: {output_json_path}")
    
    # 打印前3个分块的信息作为示例
    print("\n前3个分块示例:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n分块 {i+1} (簇ID: {chunk.cluster_id}, tokens: {chunk.tokens_count}):")
        # 打印前100个字符
        preview = chunk.text[:100] + ("..." if len(chunk.text) > 100 else "")
        print(f"{preview}")

if __name__ == "__main__":
    main()