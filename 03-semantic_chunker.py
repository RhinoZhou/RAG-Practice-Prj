#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统语义分块工具

支持多种分块策略：
1. 语义分块：基于句子语义相似度进行分块
2. 句段分块：基于句子边界进行分块
3. 窗口化分块：基于固定窗口大小进行分块

输出格式：chunks 列表，每个chunk包含chunk_id、offset、anchor等信息，便于追溯
"""

import os
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SemanticChunker")

class ChunkingMode(Enum):
    """分块模式枚举"""
    SEMANTIC = "semantic"  # 语义分块
    SENTENCE = "sentence"  # 句段分块
    WINDOW = "window"      # 窗口化分块

class SemanticChunker:
    """语义分块器，支持多种分块策略"""
    
    def __init__(self,
                 mode: str = "semantic",
                 window_size: int = 3,
                 overlap_size: int = 1,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """初始化语义分块器
        
        参数:
            mode: 分块模式，可选值: "semantic", "sentence", "window"
            window_size: 语义分块时的窗口大小
            overlap_size: 窗口化分块时的重叠大小
            chunk_size: 句段分块和窗口化分块时的块大小
            chunk_overlap: 句段分块和窗口化分块时的重叠大小
            embedding_model: 用于语义分块的嵌入模型名称
        """
        # 验证模式参数
        if mode not in [m.value for m in ChunkingMode]:
            raise ValueError(f"不支持的分块模式: {mode}，支持的模式: {[m.value for m in ChunkingMode]}")
        
        self.mode = ChunkingMode(mode)
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入模型（仅用于语义分块）
        if self.mode == ChunkingMode.SEMANTIC:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"无法加载HuggingFace嵌入模型: {e}")
                logger.info("将使用假嵌入模型进行演示")
                # 创建一个简单的假嵌入模型
                from langchain.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=384)
        
        # 初始化文本分割器
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子
        
        参数:
            text: 输入文本
        返回:
            句子列表
        """
        # 使用正则表达式分割句子
        # 支持中文和英文的句子分隔符
        pattern = r'(。|！|？|\.|!|\?)'
        sentences = []
        last_end = 0
        
        for match in re.finditer(pattern, text):
            start, end = match.span()
            sentence = text[last_end:end].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end
        
        # 处理最后一个句子
        if last_end < len(text):
            last_sentence = text[last_end:].strip()
            if last_sentence:
                sentences.append(last_sentence)
        
        return sentences
    
    def _calculate_similarity_scores(self, sentences: List[str]) -> List[float]:
        """计算句子之间的余弦相似度分数
        
        参数:
            sentences: 句子列表
        返回:
            相似度分数列表，列表长度为len(sentences)-1
        """
        if len(sentences) <= 1:
            return []
        
        print(f"\n计算 {len(sentences)} 个句子之间的相似度...")
        
        # 获取句子嵌入
        try:
            sentence_embeddings = self.embeddings.embed_documents(sentences)
        except Exception as e:
            logger.error(f"计算句子嵌入失败: {e}")
            # 返回默认的相似度分数
            return [0.1] * (len(sentences) - 1)
        
        # 计算相邻句子之间的余弦相似度
        similarity_scores = []
        for i in range(len(sentence_embeddings) - 1):
            embedding1 = np.array(sentence_embeddings[i])
            embedding2 = np.array(sentence_embeddings[i + 1])
            
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            ) if np.linalg.norm(embedding1) > 0 and np.linalg.norm(embedding2) > 0 else 0
            
            similarity_scores.append(similarity)
            
            # 添加相似度计算的输出
            if i < 5:  # 只显示前5个相似度分数，避免输出过多
                print(f"  句子 {i+1}-{i+2} 相似度: {similarity:.4f}")
        
        if len(similarity_scores) > 5:
            print(f"  ... 共 {len(similarity_scores)} 个相似度分数")
        
        return similarity_scores
    
    def _find_split_points(self, similarity_scores: List[float]) -> List[int]:
        """根据相似度分数找到分块点
        
        算法说明: 
        1. 使用滑动窗口计算局部区域的平均相似度
        2. 对边界区域进行特殊处理
        3. 使用统计学方法确定分块阈值
        4. 选择相似度低于阈值的点作为分块边界
        
        参数:
            similarity_scores: 相似度分数列表
        返回:
            分块点索引列表
        """
        if not similarity_scores:
            return []
        
        # 使用窗口大小计算局部相似度
        split_points = []
        window_scores = []
        
        for i in range(len(similarity_scores) - self.window_size + 1):
            window_avg = sum(similarity_scores[i:i + self.window_size]) / self.window_size
            window_scores.append((i + self.window_size // 2, window_avg))
        
        # 添加边界窗口
        for i in range(self.window_size // 2):
            if i < len(similarity_scores):
                window_scores.append((i, similarity_scores[i]))
        
        for i in range(len(similarity_scores) - self.window_size // 2, len(similarity_scores)):
            if i >= 0:
                window_scores.append((i, similarity_scores[i]))
        
        # 按相似度排序，取最低的作为分块点
        window_scores.sort(key=lambda x: x[1])
        
        # 选择阈值以下的点作为分块点
        mean_score = np.mean([s for _, s in window_scores])
        std_score = np.std([s for _, s in window_scores])
        threshold = mean_score - 0.5 * std_score
        split_indices = {i for i, s in window_scores if s < threshold}
        
        # 将分块点排序
        split_points = sorted(split_indices)
        
        print(f"\n分块点检测结果: ")
        print(f"  平均相似度: {mean_score:.4f}")
        print(f"  相似度标准差: {std_score:.4f}")
        print(f"  分块阈值: {threshold:.4f}")
        print(f"  找到 {len(split_points)} 个分块点: {split_points}")
        
        return split_points
    
    def _create_chunks_with_metadata(self, text: str, chunks: List[str], split_points: List[int] = None) -> List[Dict[str, Any]]:
        """为分块添加元数据
        
        参数:
            text: 原始文本
            chunks: 分块列表
            split_points: 分块点索引（可选）
        返回:
            带元数据的分块列表
        """
        result_chunks = []
        current_offset = 0
        
        for i, chunk in enumerate(chunks):
            # 查找chunk在原始文本中的偏移量
            # 注意：这是一个简单实现，对于复杂情况可能需要更精确的匹配
            chunk_start = text.find(chunk, current_offset)
            if chunk_start == -1:
                # 如果找不到精确匹配，使用当前偏移量
                chunk_start = current_offset
            
            chunk_end = chunk_start + len(chunk)
            current_offset = chunk_end
            
            # 提取锚点（chunk的前几个词和后几个词）
            words = chunk.strip().split()
            anchor = {
                "start_words": words[:3] if len(words) >= 3 else words,
                "end_words": words[-3:] if len(words) >= 3 else words,
                "start_offset": chunk_start,
                "end_offset": chunk_end
            }
            
            # 创建chunk对象
            result_chunk = {
                "chunk_id": f"chunk_{i}",
                "content": chunk,
                "offset": {
                    "start": chunk_start,
                    "end": chunk_end
                },
                "anchor": anchor,
                "length": len(chunk),
                "word_count": len(words),
                "mode": self.mode.value
            }
            
            result_chunks.append(result_chunk)
        
        return result_chunks
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """对文本进行分块处理
        
        参数:
            text: 输入文本
        返回:
            带元数据的分块列表，每个块包含chunk_id、content、offset、anchor等信息
        """
        if not text or not isinstance(text, str):
            logger.warning("无效的输入文本")
            return []
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            print(f"\n开始分块处理...")
            print(f"  输入文本长度: {len(text)} 字符")
            print(f"  使用模式: {self.mode.value}")
            
            if self.mode == ChunkingMode.SEMANTIC:
                # 1. 语义分块
                logger.info("使用语义分块模式")
                
                # 将文本分割成句子
                sentences = self._split_into_sentences(text)
                logger.info(f"分割出 {len(sentences)} 个句子")
                print(f"  分割出 {len(sentences)} 个句子")
                
                if len(sentences) <= 1:
                    # 如果只有一个句子，直接返回
                    chunks = [text]
                    print(f"  只有一个句子，无需进一步分块")
                else:
                    # 计算句子相似度
                    similarity_scores = self._calculate_similarity_scores(sentences)
                    
                    # 找到分块点
                    split_points = self._find_split_points(similarity_scores)
                    logger.info(f"找到 {len(split_points)} 个分块点")
                    
                    # 根据分块点组合句子
                    chunks = []
                    start_idx = 0
                    
                    for split_idx in split_points:
                        chunk_sentences = sentences[start_idx:split_idx + 1]
                        chunks.append(" ".join(chunk_sentences))
                        start_idx = split_idx + 1
                    
                    # 处理最后一个块
                    if start_idx < len(sentences):
                        chunk_sentences = sentences[start_idx:]
                        chunks.append(" ".join(chunk_sentences))
            
            elif self.mode == ChunkingMode.SENTENCE:
                # 2. 句段分块
                logger.info("使用句段分块模式")
                print(f"  分块配置: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
                
                # 使用LangChain的文本分割器
                documents = self.sentence_splitter.create_documents([text])
                chunks = [doc.page_content for doc in documents]
                
            elif self.mode == ChunkingMode.WINDOW:
                # 3. 窗口化分块
                logger.info("使用窗口化分块模式")
                print(f"  分块配置: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
                
                # 使用固定窗口大小进行分块
                chunks = []
                text_len = len(text)
                current_pos = 0
                
                while current_pos < text_len:
                    end_pos = min(current_pos + self.chunk_size, text_len)
                    chunk = text[current_pos:end_pos]
                    chunks.append(chunk)
                    current_pos += self.chunk_size - self.chunk_overlap
            
            # 为分块添加元数据
            result_chunks = self._create_chunks_with_metadata(text, chunks)
            
            # 记录结束时间
            end_time = time.time()
            logger.info(f"分块完成，共生成 {len(result_chunks)} 个块，耗时 {end_time - start_time:.2f} 秒")
            
            print(f"\n分块完成！")
            print(f"  生成块数量: {len(result_chunks)}")
            print(f"  平均块长度: {sum(len(chunk['content']) for chunk in result_chunks) / len(result_chunks):.1f} 字符")
            print(f"  耗时: {end_time - start_time:.2f} 秒")
            
            return result_chunks
            
        except Exception as e:
            logger.error(f"分块处理失败: {e}")
            print(f"分块处理失败: {e}")
            # 发生错误时，返回原始文本作为单个块
            return self._create_chunks_with_metadata(text, [text])

if __name__ == "__main__":
    """示例用法：演示三种分块模式的功能和输出"""
    # 配置日志级别为INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 示例文本
    sample_text = """
    RAG系统是一种检索增强生成技术，它结合了检索和生成的优势。
    
    RAG的主要流程包括：文本加载、文本清洗、文本分块、向量化、存储到向量数据库、检索相关文档、生成回答等步骤。
    
    语义分块是RAG系统中的重要环节，它能够根据文本的语义内容进行合理的分块，有助于提高检索的准确性和生成的质量。
    
    不同的分块策略适用于不同类型的文本和应用场景。语义分块基于句子间的语义相似度进行分块，句段分块基于句子边界进行分块，窗口化分块则基于固定的窗口大小进行分块。
    """
    
    print("#" * 60)
    print("#" * 20 + " RAG系统语义分块工具演示 " + "#" * 20)
    print("#" * 60)
    print(f"\n示例文本预览:")
    print("  " + sample_text.strip())
    
    # 1. 测试语义分块
    print("\n\n" + "=" * 50)
    print("1. 测试语义分块模式:")
    print("=" * 50)
    try:
        print("初始化语义分块器...")
        semantic_chunker = SemanticChunker(mode="semantic", window_size=3)
        print("执行语义分块...")
        chunks = semantic_chunker.chunk(sample_text)
        
        print("\n\n语义分块结果详情:")
        for i, chunk in enumerate(chunks):
            print(f"\n{'-' * 40}")
            print(f"块 {i+1} ({chunk['offset']['start']}-{chunk['offset']['end']})")
            print(f"{'-' * 40}")
            print(f"内容: {chunk['content']}")
            print(f"锚点: 开始='{' '.join(chunk['anchor']['start_words'])}', 结束='{' '.join(chunk['anchor']['end_words'])}'")
            print(f"长度: {chunk['length']} 字符, {chunk['word_count']} 词")
    except Exception as e:
        print(f"语义分块测试失败: {e}")
    
    # 2. 测试句段分块
    print("\n\n" + "=" * 50)
    print("2. 测试句段分块模式:")
    print("=" * 50)
    try:
        print("初始化句段分块器...")
        sentence_chunker = SemanticChunker(mode="sentence", chunk_size=200, chunk_overlap=50)
        print("执行句段分块...")
        chunks = sentence_chunker.chunk(sample_text)
        
        print("\n\n句段分块结果详情:")
        for i, chunk in enumerate(chunks):
            print(f"\n{'-' * 40}")
            print(f"块 {i+1} ({chunk['offset']['start']}-{chunk['offset']['end']})")
            print(f"{'-' * 40}")
            print(f"内容: {chunk['content']}")
            print(f"锚点: 开始='{' '.join(chunk['anchor']['start_words'])}', 结束='{' '.join(chunk['anchor']['end_words'])}'")
            print(f"长度: {chunk['length']} 字符, {chunk['word_count']} 词")
    except Exception as e:
        print(f"句段分块测试失败: {e}")
    
    # 3. 测试窗口化分块
    print("\n\n" + "=" * 50)
    print("3. 测试窗口化分块模式:")
    print("=" * 50)
    try:
        print("初始化窗口化分块器...")
        window_chunker = SemanticChunker(mode="window", chunk_size=100, chunk_overlap=20)
        print("执行窗口化分块...")
        chunks = window_chunker.chunk(sample_text)
        
        print("\n\n窗口化分块结果详情:")
        for i, chunk in enumerate(chunks):
            print(f"\n{'-' * 40}")
            print(f"块 {i+1} ({chunk['offset']['start']}-{chunk['offset']['end']})")
            print(f"{'-' * 40}")
            print(f"内容: {chunk['content']}")
            print(f"锚点: 开始='{' '.join(chunk['anchor']['start_words'])}', 结束='{' '.join(chunk['anchor']['end_words'])}'")
            print(f"长度: {chunk['length']} 字符, {chunk['word_count']} 词")
    except Exception as e:
        print(f"窗口化分块测试失败: {e}")
    
    print("\n\n" + "=" * 60)
    print("\n===== 所有分块模式测试完成 =====")
    print("\n三种分块策略对比:")
    print("- 语义分块: 基于句子语义相似度，智能划分语义边界")
    print("- 句段分块: 基于自然句子边界，保留完整语义单元")
    print("- 窗口化分块: 基于固定窗口大小，适合结构化数据")
    print("\n请根据您的实际需求选择合适的分块策略！")
    print("\n" + "=" * 60)