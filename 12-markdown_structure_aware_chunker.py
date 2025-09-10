# -*- coding: utf-8 -*-
"""Markdown结构感知分块器

实现相邻句子向量相似度序列的局部谷值检测作为分割点。
支持句向量编码、相似度序列计算、谷值阈值与窗口设置等功能。
"""

import os
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# 尝试导入PyMuPDF库用于PDF处理
try:
    import fitz
    HAS_FITZ = True
    print("成功导入PyMuPDF库")
except ImportError:
    HAS_FITZ = False
    print("未找到PyMuPDF库，请使用pip install pymupdf安装")

# 尝试导入tiktoken库用于token计算
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    print("未找到tiktoken库，将使用简单的字符tokenizer")

@dataclass
class SimilarityChunk:
    """表示一个相似度分块及其元数据
    
    属性:
        text: 块文本内容
        start_index: 在原文中的起始位置
        end_index: 在原文中的结束位置
        chunk_id: 块ID
        token_count: token数量
        sentences: 包含的句子列表
    """
    text: str
    start_index: int
    end_index: int
    chunk_id: int = 0
    token_count: int = 0
    sentences: Optional[List[str]] = None

class SimpleCharacterTokenizer:
    """简单的字符tokenizer实现，用于在tiktoken不可用时作为替代"""
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为字符级别token"""
        return list(text)
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenize(text))

class SentenceVectorEncoder:
    """句子向量编码器
    
    提供句子向量化功能，使用简单的TF-IDF类似的实现
    实际应用中可以替换为更复杂的编码器如Sentence-BERT
    """
    
    def __init__(self):
        """初始化句子向量编码器"""
        self.vocab = {}
        self.vocab_size = 0
    
    def _build_vocab(self, sentences: List[str]) -> None:
        """构建词汇表
        
        参数:
            sentences: 句子列表
        """
        word_set = set()
        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())
            word_set.update(words)
        
        self.vocab = {word: idx for idx, word in enumerate(word_set)}
        self.vocab_size = len(self.vocab)
    
    def encode(self, sentences: List[str]) -> np.ndarray:
        """将句子列表编码为向量矩阵
        
        参数:
            sentences: 句子列表
            
        返回:
            句子向量矩阵，形状为[句子数量, 向量维度]
        """
        if not self.vocab:
            self._build_vocab(sentences)
        
        vectors = np.zeros((len(sentences), self.vocab_size))
        
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\w+', sentence.lower())
            for word in words:
                if word in self.vocab:
                    vectors[i, self.vocab[word]] += 1
            # 归一化
            if np.linalg.norm(vectors[i]) > 0:
                vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
        
        return vectors

class MarkdownStructureAwareChunker:
    """Markdown结构感知分块器
    
    实现基于相邻句子相似度谷值的分块策略，主要流程包括：
    1. 将文本分割为句子
    2. 对句子进行向量编码
    3. 计算相邻句子向量的相似度
    4. 在相似度序列中检测局部谷值作为分割点
    5. 根据分割点对文本进行分块，确保每个块不超过最大token限制
    """
    """Markdown结构感知分块器
    
    实现基于相邻句子相似度谷值的分块策略
    """
    
    def __init__(self, tokenizer_name: str = "cl100k_base", use_simple_tokenizer: bool = False):
        """初始化分块器
        
        参数:
            tokenizer_name: 使用的tokenizer名称
            use_simple_tokenizer: 是否强制使用简单的字符tokenizer
        """
        self.tokenizer_name = tokenizer_name
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        self.tokenizer = self._load_tokenizer()
        self.vector_encoder = SentenceVectorEncoder()
        # 用于分句的正则表达式 - 匹配中文和英文的句末标点符号
        self.sentence_pattern = re.compile(r'(?<=[。！？.?!])\s*')
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        if self.use_simple_tokenizer or not HAS_TIKTOKEN:
            return SimpleCharacterTokenizer()
        else:
            try:
                return tiktoken.get_encoding(self.tokenizer_name)
            except Exception as e:
                print(f"加载tiktoken失败: {e}，将使用简单的字符tokenizer")
                return SimpleCharacterTokenizer()
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        参数:
            text: 输入文本
            
        返回:
            token数量
        """
        if isinstance(self.tokenizer, SimpleCharacterTokenizer):
            return self.tokenizer.count_tokens(text)
        else:
            return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子
        
        参数:
            text: 输入文本
            
        返回:
            句子列表
        """
        # 使用正则表达式分割句子
        sentences = self.sentence_pattern.split(text)
        # 过滤空句子
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        return sentences
    
    def compute_similarity(self, vectors: np.ndarray) -> List[float]:
        """计算相邻句子向量的相似度
        
        参数:
            vectors: 句子向量矩阵
            
        返回:
            相邻句子相似度列表
        """
        similarities = []
        for i in range(len(vectors) - 1):
            # 使用余弦相似度
            similarity = np.dot(vectors[i], vectors[i+1])
            similarities.append(float(similarity))
        return similarities
    
    def detect_valleys(self, similarities: List[float], window_size: int = 3, threshold: float = 0.3) -> List[int]:
        """在相似度序列中检测局部谷值
        
        参数:
            similarities: 相邻句子向量的相似度序列
            window_size: 局部窗口大小，用于确定谷值的局部范围
            threshold: 谷值阈值，低于此值的相似度才被认为是潜在的分割点
            
        返回:
            谷值位置索引列表，表示建议的分割位置
        """
        valleys = []  # 存储检测到的谷值位置
        n = len(similarities)
        
        # 遍历每个相似度值，判断是否为局部谷值
        for i in range(n):
            # 初始假设当前位置是谷值
            is_valley = True
            
            # 计算局部窗口范围
            window_start = max(0, i - window_size // 2)  # 窗口起始位置，确保不小于0
            window_end = min(n, i + window_size // 2 + 1)  # 窗口结束位置，确保不超过序列长度
            
            # 检查窗口内的所有元素，确认当前位置是否为最小值
            for j in range(window_start, window_end):
                if j != i and similarities[j] < similarities[i]:
                    is_valley = False
                    break  # 如果找到更小的值，当前位置不是谷值
            
            # 只有当该位置是窗口内的最小值且低于阈值时，才确认为谷值
            if is_valley and similarities[i] < threshold:
                valleys.append(i)
        
        return valleys
    
    def chunk_text(self, text: str, sentences: List[str], split_indices: List[int], max_tokens: int = 500) -> List[SimilarityChunk]:
        """根据分割点对文本进行分块
        
        参数:
            text: 原始文本
            sentences: 句子列表
            split_indices: 分割点索引
            max_tokens: 每个块的最大token数量
            
        返回:
            分块列表
        """
        chunks = []
        current_start = 0
        current_chunk_id = 0
        
        # 预处理：为每个句子创建在原始文本中的精确位置映射
        sentence_positions = []
        current_pos = 0  # 用于在文本中定位句子的起始位置
        
        for sentence in sentences:
            # 在文本中查找句子的精确位置，从当前位置开始搜索
            start_idx = text.find(sentence, current_pos)
            
            if start_idx != -1:  # 找到精确匹配
                end_idx = start_idx + len(sentence)  # 计算句子的结束位置
                sentence_positions.append((start_idx, end_idx))  # 存储句子的起始和结束位置
                current_pos = end_idx  # 更新当前位置，准备查找下一个句子
            else:  # 未找到精确匹配，使用估算的位置
                # 这种情况通常发生在文本处理过程中，如文本规范化、空格处理等导致的不匹配
                # 使用简单的估算方法，从上一个句子的结束位置开始
                sentence_positions.append((current_pos, current_pos + len(sentence)))
                current_pos += len(sentence)  # 更新当前位置
        
        # 确保分割点按顺序排列
        split_indices = sorted(split_indices)
        
        # 添加文本末尾作为最后一个分割点，确保处理完所有句子
        split_indices.append(len(sentences) - 1)
        
        for split_idx in split_indices:
            # 计算当前块的句子
            current_sentences = sentences[current_start:split_idx + 1]
            
            # 计算当前块的文本和token数量
            current_text = ''.join(current_sentences)
            current_tokens = self.count_tokens(current_text)
            
            # 使用句子位置映射来确定块的位置
            if current_start <= split_idx and current_start < len(sentence_positions) and split_idx < len(sentence_positions):
                start_pos = sentence_positions[current_start][0]
                end_pos = sentence_positions[split_idx][1]
            else:
                # 如果超出范围，使用估算的位置
                start_pos = 0 if not chunks else chunks[-1]['end_index']
                end_pos = start_pos + len(current_text)
            
            # 如果当前块的token数量超过最大限制，需要进一步分割
            if current_tokens > max_tokens:
                # 简单地将块平均分割
                mid_idx = current_start + (split_idx - current_start) // 2
                # 先处理前半部分
                first_half_sentences = sentences[current_start:mid_idx + 1]
                first_half_text = ''.join(first_half_sentences)
                first_half_tokens = self.count_tokens(first_half_text)
                
                # 确定前半部分的位置
                if current_start <= mid_idx and current_start < len(sentence_positions) and mid_idx < len(sentence_positions):
                    first_half_start_pos = sentence_positions[current_start][0]
                    first_half_end_pos = sentence_positions[mid_idx][1]
                else:
                    first_half_start_pos = start_pos
                    first_half_end_pos = start_pos + len(first_half_text)
                
                chunks.append(SimilarityChunk(
                    text=first_half_text,
                    start_index=first_half_start_pos,
                    end_index=first_half_end_pos,
                    chunk_id=current_chunk_id,
                    token_count=first_half_tokens,
                    sentences=first_half_sentences
                ))
                current_chunk_id += 1
                
                # 再处理后半部分
                second_half_sentences = sentences[mid_idx + 1:split_idx + 1]
                second_half_text = ''.join(second_half_sentences)
                second_half_tokens = self.count_tokens(second_half_text)
                
                # 确定后半部分的位置
                if mid_idx + 1 <= split_idx and mid_idx + 1 < len(sentence_positions) and split_idx < len(sentence_positions):
                    second_half_start_pos = sentence_positions[mid_idx + 1][0]
                    second_half_end_pos = sentence_positions[split_idx][1]
                else:
                    second_half_start_pos = first_half_end_pos
                    second_half_end_pos = first_half_end_pos + len(second_half_text)
                
                chunks.append(SimilarityChunk(
                    text=second_half_text,
                    start_index=second_half_start_pos,
                    end_index=second_half_end_pos,
                    chunk_id=current_chunk_id,
                    token_count=second_half_tokens,
                    sentences=second_half_sentences
                ))
                current_chunk_id += 1
            else:
                chunks.append(SimilarityChunk(
                    text=current_text,
                    start_index=start_pos,
                    end_index=end_pos,
                    chunk_id=current_chunk_id,
                    token_count=current_tokens,
                    sentences=current_sentences
                ))
                current_chunk_id += 1
            
            current_start = split_idx + 1
            
            # 如果已经处理完所有句子，跳出循环
            if current_start >= len(sentences):
                break
        
        return chunks
    
    def process_pdf_file(self, pdf_path: str, window_size: int = 3, threshold: float = 0.3, max_tokens: int = 500) -> Tuple[List[SimilarityChunk], List[int]]:
        """处理PDF文件并进行分块
        
        参数:
            pdf_path: PDF文件路径
            window_size: 局部窗口大小
            threshold: 谷值阈值
            max_tokens: 每个块的最大token数量
            
        返回:
            (分块列表, 分割点索引列表)
        """
        if not HAS_FITZ:
            raise ImportError("需要PyMuPDF库来处理PDF文件，请使用pip install pymupdf安装")
        
        # 读取PDF文件内容
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text()
        
        doc.close()
        
        # 分割句子
        sentences = self.split_into_sentences(full_text)
        
        # 如果没有句子，返回空结果
        if not sentences:
            return [], []
        
        # 对句子进行编码
        sentence_vectors = self.vector_encoder.encode(sentences)
        
        # 计算相邻句子的相似度
        similarities = self.compute_similarity(sentence_vectors)
        
        # 检测局部谷值作为分割点
        valley_indices = self.detect_valleys(similarities, window_size, threshold)
        
        # 根据分割点进行分块
        chunks = self.chunk_text(full_text, sentences, valley_indices, max_tokens)
        
        return chunks, valley_indices
    
    def save_chunks_to_json(self, chunks: List[SimilarityChunk], output_path: str) -> None:
        """将分块结果保存到JSON文件
        
        参数:
            chunks: 分块列表
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 转换分块对象为字典
        chunks_dict = []
        for chunk in chunks:
            chunks_dict.append({
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "sentence_count": len(chunk.sentences) if chunk.sentences else 0
            })
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)

def install_dependencies():
    """安装必要的依赖库"""
    try:
        import subprocess
        import sys
        
        # 检查并安装PyMuPDF
        try:
            import fitz
        except ImportError:
            print("安装PyMuPDF库...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
        
        # 检查并安装tiktoken（可选）
        try:
            import tiktoken
        except ImportError:
            print("安装tiktoken库（可选）...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            except Exception as e:
                print(f"安装tiktoken失败: {e}")
    except Exception as e:
        print(f"安装依赖库时出错: {e}")

def main():
    """主函数"""
    # 安装必要的依赖库
    install_dependencies()
    
    # 设置文件路径
    pdf_path = os.path.join("data", "大脑中动脉狭窄与闭塞致脑梗死的影像特点及发病机制的研究.pdf")
    output_path = os.path.join("results", "markdown_structure_chunks.json")
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    try:
        # 初始化分块器
        chunker = MarkdownStructureAwareChunker()
        
        # 设置参数
        window_size = 3
        threshold = 0.3
        max_tokens = 500
        
        # 处理PDF文件
        print(f"开始处理PDF文件: {pdf_path}")
        chunks, split_indices = chunker.process_pdf_file(pdf_path, window_size, threshold, max_tokens)
        
        # 打印分块结果摘要
        print(f"检测到 {len(split_indices)} 个分割点")
        print(f"生成了 {len(chunks)} 个分块")
        
        # 计算平均token数量
        avg_tokens = sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0
        print(f"平均每个分块的token数量: {avg_tokens:.2f}")
        
        # 保存结果
        chunker.save_chunks_to_json(chunks, output_path)
        print(f"分块结果已保存到: {output_path}")
        
        # 保存分割点索引
        split_indices_path = os.path.join("results", "split_indices.json")
        with open(split_indices_path, 'w', encoding='utf-8') as f:
            json.dump(split_indices, f, ensure_ascii=False, indent=2)
        print(f"分割点索引已保存到: {split_indices_path}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()