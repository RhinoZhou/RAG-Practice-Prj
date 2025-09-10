import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 导入需要的依赖
try:
    import numpy as np
except ImportError:
    print("请安装numpy库: pip install numpy")
    raise

# 尝试导入tiktoken库
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    print("tiktoken库未安装，将使用内置的简单tokenizer")
    HAS_TIKTOKEN = False

@dataclass
class TextChunk:
    """表示一个文本块及其元数据的数据包类
    
    该类用于存储分块后的文本内容及其相关元数据，包括token范围、字符位置范围和相邻块信息。
    
    属性:
        text: 块文本内容
        start_tok: 起始token索引
        end_tok: 结束token索引
        neighbors: 相邻块的索引列表
        chunk_id: 块ID（唯一标识符）
        start_pos: 原始文本中的起始字符位置
        end_pos: 原始文本中的结束字符位置
    """
    text: str  # 块文本内容
    start_tok: int  # 起始token索引
    end_tok: int  # 结束token索引
    neighbors: List[int]  # 相邻块的索引列表
    chunk_id: int  # 块ID
    start_pos: int  # 原始文本中的起始位置
    end_pos: int  # 原始文本中的结束位置

class SimpleCharacterTokenizer:
    """简单的基于字符的tokenizer实现，不需要网络连接
    
    这个tokenizer将每个字符映射为其Unicode码点，作为替代tiktoken的离线解决方案。
    在无法访问网络或tiktoken库未安装时使用。
    """
    
    def __init__(self):
        self.name = "simple_character_tokenizer"
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token序列（简单地将每个字符映射为其Unicode码点）
        
        参数:
            text: 要编码的文本
            
        返回:
            token序列（字符的Unicode码点列表）
        """
        return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """将token序列解码回文本
        
        参数:
            tokens: token序列（Unicode码点列表）
            
        返回:
            解码后的文本字符串
        """
        return ''.join([chr(t) if 0 <= t <= 0x10FFFF else '�' for t in tokens])

class TokenFixedOverlapChunker:
    """按tokens固定长度与重叠滑窗的文本分块器"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base", use_simple_tokenizer: bool = False):
        """初始化分块器
        
        参数:
            tokenizer_name: 使用的tokenizer名称，默认使用cl100k_base（GPT-4使用的tokenizer）
            use_simple_tokenizer: 是否强制使用简单的字符tokenizer，不依赖网络
        """
        self.tokenizer_name = tokenizer_name
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self):
        """加载指定的tokenizer"""
        if self.use_simple_tokenizer:
            print("使用简单的字符tokenizer")
            return SimpleCharacterTokenizer()
        
        if not HAS_TIKTOKEN:
            print("tiktoken库未安装，自动使用简单的字符tokenizer")
            return SimpleCharacterTokenizer()
        
        try:
            print(f"尝试加载tiktoken: {self.tokenizer_name}")
            return tiktoken.get_encoding(self.tokenizer_name)
        except Exception as e:
            print(f"加载tiktoken失败: {e}")
            print("切换到简单的字符tokenizer")
            return SimpleCharacterTokenizer()
    
    def tokenize(self, text: str) -> List[int]:
        """将文本转换为token序列
        
        参数:
            text: 要分词的文本
            
        返回:
            token序列
        """
        return self.tokenizer.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """将token序列转换回文本
        
        参数:
            tokens: token序列
            
        返回:
            文本
        """
        return self.tokenizer.decode(tokens)
    
    def chunk_text(self, text: str, chunk_size: int, overlap_ratio: float = 0.2) -> List[TextChunk]:
        """按固定长度和重叠率对文本进行分块的核心方法
        
        该方法实现了滑动窗口分块策略，使用固定大小的窗口和指定的重叠比例来处理文本。
        分块过程包括分词、计算滑动步长、生成文本块、计算字符位置和填充相邻块信息。
        
        参数:
            text: 要分块的文本
            chunk_size: 每个块的token数量
            overlap_ratio: 重叠比例，默认为0.2（20%）
            
        返回:
            TextChunk对象列表，包含所有分块结果及其元数据
        
        异常:
            ValueError: 当chunk_size不为正数或overlap_ratio不在有效范围内时抛出
        """
        if chunk_size <= 0:
            raise ValueError("块大小必须为正数")
        
        if overlap_ratio < 0 or overlap_ratio >= 1:
            raise ValueError("重叠比例必须在[0, 1)范围内")
        
        # 分词
        tokens = self.tokenize(text)
        if not tokens:
            return []
        
        # 计算重叠token数量
        overlap_size = int(chunk_size * overlap_ratio)
        
        # 计算步长
        step_size = chunk_size - overlap_size
        
        # 生成块
        chunks = []
        num_tokens = len(tokens)
        
        # 确保重叠大小不小于1，除非块大小为1
        if step_size <= 0:
            step_size = 1
        
        # 分块处理
        chunk_id = 0
        pos = 0
        
        while pos < num_tokens:
            # 确定当前块的token范围
            end_pos = min(pos + chunk_size, num_tokens)
            chunk_tokens = tokens[pos:end_pos]
            
            # 转换回文本
            chunk_text = self.detokenize(chunk_tokens)
            
            # 对于简单的字符tokenizer，token和字符是一一对应的
            if isinstance(self.tokenizer, SimpleCharacterTokenizer):
                start_char_pos = pos
                end_char_pos = end_pos
            else:
                # 对于tiktoken等复杂tokenizer，需要近似计算字符位置
                if pos == 0:
                    start_char_pos = 0
                else:
                    # 尝试找到前一个块结束位置对应的字符位置
                    prev_chunk = chunks[-1]
                    # 搜索chunk_text在原始文本中的位置，从prev_chunk的结束位置开始
                    approx_pos = text.find(chunk_text, prev_chunk.end_pos - 50)  # 向前搜索50个字符
                    if approx_pos == -1:
                        # 如果没找到，使用近似位置
                        start_char_pos = prev_chunk.end_pos - overlap_size // 2
                    else:
                        start_char_pos = approx_pos
                    # 修正起始位置
                    start_char_pos = max(0, start_char_pos)
                
                # 计算结束字符位置
                end_char_pos = min(start_char_pos + len(chunk_text), len(text))
            
            # 创建TextChunk对象
            chunk = TextChunk(
                text=chunk_text,
                start_tok=pos,
                end_tok=end_pos,
                neighbors=[],  # 稍后填充
                chunk_id=chunk_id,
                start_pos=start_char_pos,
                end_pos=end_char_pos
            )
            chunks.append(chunk)
            
            # 更新位置
            pos += step_size
            chunk_id += 1
        
        # 填充相邻块信息
        self._fill_neighbor_info(chunks)
        
        return chunks
    
    def _fill_neighbor_info(self, chunks: List[TextChunk]) -> None:
        """填充块的相邻信息
        
        为每个文本块添加相邻块的ID信息，便于后续的上下文理解和块间关系分析。
        前一个块（如果存在）和后一个块（如果存在）被标记为相邻块。
        
        参数:
            chunks: TextChunk对象列表
        """
        for i, chunk in enumerate(chunks):
            # 添加前一个块（如果有）
            if i > 0:
                chunk.neighbors.append(i - 1)
            
            # 添加后一个块（如果有）
            if i < len(chunks) - 1:
                chunk.neighbors.append(i + 1)
    
    def convert_to_json(self, chunks: List[TextChunk]) -> List[Dict]:
        """将TextChunk对象列表转换为JSON可序列化的字典列表
        
        将TextChunk对象转换为包含相同信息的普通Python字典，便于JSON序列化和存储。
        
        参数:
            chunks: TextChunk对象列表
            
        返回:
            可序列化的字典列表
        """
        return [{
            "text": chunk.text,
            "start_tok": chunk.start_tok,
            "end_tok": chunk.end_tok,
            "neighbors": chunk.neighbors,
            "chunk_id": chunk.chunk_id,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos
        } for chunk in chunks]
    
    def save_chunks_to_json(self, chunks: List[TextChunk], output_path: str) -> None:
        """将块保存为JSON文件
        
        将分块结果转换为JSON格式并保存到指定路径，便于后续分析和使用。
        
        参数:
            chunks: TextChunk对象列表
            output_path: 输出文件路径
        """
        json_data = self.convert_to_json(chunks)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"分块结果已保存到: {output_path}")
    
    def print_chunk_summary(self, chunks: List[TextChunk]) -> None:
        """打印块的摘要信息
        
        参数:
            chunks: TextChunk对象列表
        """
        print(f"总块数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"块 {i}:")
            print(f"  Token范围: [{chunk.start_tok}, {chunk.end_tok})")
            print(f"  字符范围: [{chunk.start_pos}, {chunk.end_pos})")
            print(f"  相邻块: {chunk.neighbors}")
            print(f"  文本预览: {chunk.text[:100]}...")
            print("-")

# 示例使用函数
def main():
    """示例函数，展示如何使用TokenFixedOverlapChunker"""
    # 创建分块器实例 - 使用简单tokenizer，不依赖网络
    chunker = TokenFixedOverlapChunker(use_simple_tokenizer=True)
    
    # 示例文本
    sample_text = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
旨在创造能够执行通常需要人类智能的任务的智能机器。人工智能的研究领域包括机器学习、计算机视觉、
自然语言处理、机器人学、专家系统等多个方面。

机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习而不需要明确编程。
深度学习是机器学习的一个子集，它使用多层神经网络来模拟人类大脑的某些功能。
深度学习在图像识别、语音识别、自然语言处理等领域取得了显著的成功。

随着大数据和计算能力的提升，人工智能技术正在快速发展，并在医疗、金融、交通、教育等
多个行业得到广泛应用。人工智能的发展也带来了一系列伦理和社会问题，如隐私保护、
就业影响、算法偏见等，需要我们认真思考和应对。"""
    
    # 分块参数
    chunk_size = 100  # 每个块100个token
    overlap_ratio = 0.2  # 20%的重叠率
    
    # 执行分块
    print(f"使用块大小={chunk_size}, 重叠率={overlap_ratio}进行分块...")
    chunks = chunker.chunk_text(sample_text, chunk_size, overlap_ratio)
    
    # 打印摘要
    chunker.print_chunk_summary(chunks)
    
    # 保存结果
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "token_fixed_overlap_chunks.json"
    chunker.save_chunks_to_json(chunks, str(output_path))
    
    # 测试不同参数
    print("\n测试不同的块大小和重叠率组合...")
    
    # 参数组合测试
    test_params = [
        (50, 0.1),  # 块大小50，重叠率10%
        (150, 0.3),  # 块大小150，重叠率30%
        (200, 0.2)  # 块大小200，重叠率20%
    ]
    
    for i, (size, ratio) in enumerate(test_params):
        print(f"\n测试组合 {i+1}: 块大小={size}, 重叠率={ratio}")
        test_chunks = chunker.chunk_text(sample_text, size, ratio)
        print(f"  生成块数: {len(test_chunks)}")
        # 保存测试结果
        test_output_path = output_dir / f"token_fixed_overlap_chunks_test_{i+1}.json"
        chunker.save_chunks_to_json(test_chunks, str(test_output_path))
    
    # 提供一个从文件加载文本的示例
    print("\n提供一个从文件加载文本的示例...")
    # 如果corpus.txt文件存在，可以使用以下代码加载文本
    corpus_path = "corpus.txt"
    if Path(corpus_path).exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_text = f.read()
        print(f"从文件加载了{len(corpus_text)}个字符的文本")
        # 这里可以添加对corpus_text进行分块的代码
    else:
        print(f"文件{corpus_path}不存在")

if __name__ == "__main__":
    main()