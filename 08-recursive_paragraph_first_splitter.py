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
class ParagraphChunk:
    """表示一个段落块及其元数据
    
    该类用于存储分块后的文本内容及其相关元数据，包括文本内容、层级信息、父段落关系
    和位置信息等，便于后续对分块结果进行分析和使用。
    
    属性:
        text: 块文本内容
        level: 层级（1表示原始段落，2表示对段落的进一步切分）
        parent_para_id: 父段落ID，指示该块所属的原始段落
        chunk_id: 块ID（唯一标识符）
        start_pos: 原始文本中的起始字符位置
        end_pos: 原始文本中的结束字符位置
        token_count: token数量
    """
    text: str  # 块文本内容
    level: int  # 层级（1表示原始段落，2表示对段落的进一步切分）
    parent_para_id: int  # 父段落ID
    chunk_id: int  # 块ID
    start_pos: int  # 原始文本中的起始位置
    end_pos: int  # 原始文本中的结束位置
    token_count: int  # token数量

class SimpleCharacterTokenizer:
    """简单的基于字符的tokenizer实现，不需要网络连接"""
    
    def __init__(self):
        self.name = "simple_character_tokenizer"
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token序列（这里简单地将每个字符映射为其Unicode码点）"""
        return [ord(c) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """将token序列解码回文本"""
        return ''.join([chr(t) if 0 <= t <= 0x10FFFF else '�' for t in tokens])

class RecursiveParagraphFirstSplitter:
    """递归分割（段落优先）的文本分块器
    
    该分块器实现了段落优先的递归分割策略：首先按段落粗分，保留文本的自然语义结构；
    然后对超过最大token限制的段落进行二次切分，确保每个块不超过大小限制。
    这种方法能够在保持语义完整性和控制块大小之间取得良好平衡。
    """
    
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
        """将文本转换为token序列"""
        return self.tokenizer.encode(text)
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenize(text))
    
    def split_into_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """将文本分割为段落
        
        该方法是段落优先策略的第一步，通过识别连续的换行符将文本分割成多个段落，
        保留文本的自然语义结构。
        
        参数:
            text: 要分割的文本
            
        返回:
            段落列表，每个元素为(段落文本, 起始位置, 结束位置)
        """
        # 使用正则表达式分割段落，识别连续的换行符
        paragraphs = []
        # 匹配一个或多个换行符作为段落分隔符
        pattern = r'(.*?)(?:\n\s*\n|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            para_text = match.group(1).strip()
            if para_text:  # 跳过空段落
                start_pos = match.start(1)
                end_pos = match.end(1)
                paragraphs.append((para_text, start_pos, end_pos))
        
        return paragraphs
    
    def split_paragraph_into_chunks(self, para_text: str, max_tokens: int, overlap_ratio: float,
                                   parent_para_id: int, start_pos: int) -> List[ParagraphChunk]:
        """对单个段落进行二次切分（如果超过最大token数）
        
        该方法是递归分割的核心，当段落的token数超过最大限制时，使用滑窗算法
        对段落进行二次切分，同时保留父段落ID和层级信息。
        
        参数:
            para_text: 段落文本
            max_tokens: 最大token数
            overlap_ratio: 重叠比例
            parent_para_id: 父段落ID
            start_pos: 段落起始位置
            
        返回:
            ParagraphChunk对象列表
        """
        tokens = self.tokenize(para_text)
        total_tokens = len(tokens)
        
        # 如果段落token数不超过最大限制，直接返回整个段落
        if total_tokens <= max_tokens:
            return [ParagraphChunk(
                text=para_text,
                level=1,
                parent_para_id=parent_para_id,
                chunk_id=0,
                start_pos=start_pos,
                end_pos=start_pos + len(para_text),
                token_count=total_tokens
            )]
        
        # 否则，进行二次切分
        chunks = []
        overlap_size = int(max_tokens * overlap_ratio)
        step_size = max_tokens - overlap_size
        
        # 确保步长至少为1
        if step_size <= 0:
            step_size = 1
        
        pos = 0
        chunk_id = 0
        
        while pos < total_tokens:
            end_pos = min(pos + max_tokens, total_tokens)
            chunk_tokens = tokens[pos:end_pos]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # 计算在原始文本中的位置
            approx_char_pos = start_pos
            if pos > 0:
                # 对于简单的字符tokenizer，位置可以直接映射
                if isinstance(self.tokenizer, SimpleCharacterTokenizer):
                    approx_char_pos = start_pos + pos
                else:
                    # 对于复杂tokenizer，使用近似位置
                    # 搜索当前块在段落中的位置
                    prev_chunk_text = chunks[-1].text
                    # 从重叠部分开始搜索
                    search_start = max(0, len(prev_chunk_text) - overlap_size * 2)
                    approx_char_pos = start_pos + prev_chunk_text.find(chunk_text[:100], search_start)
                    if approx_char_pos < start_pos:
                        approx_char_pos = start_pos + pos  # 使用近似位置
            
            end_char_pos = min(approx_char_pos + len(chunk_text), start_pos + len(para_text))
            
            chunk = ParagraphChunk(
                text=chunk_text,
                level=2,  # 二级切分
                parent_para_id=parent_para_id,
                chunk_id=chunk_id,
                start_pos=approx_char_pos,
                end_pos=end_char_pos,
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)
            
            # 更新位置
            pos += step_size
            chunk_id += 1
        
        return chunks
    
    def split_text(self, text: str, max_tokens: int, overlap_ratio: float = 0.2) -> List[ParagraphChunk]:
        """递归分割文本（段落优先）的主方法
        
        实现完整的段落优先递归分割流程：
        1. 首先按段落对文本进行粗分（保持语义完整性）
        2. 然后对每个段落检查token数量
        3. 对超过最大token限制的段落进行二次切分
        4. 为每个块添加层级信息和父段落关系
        
        参数:
            text: 要分割的文本
            max_tokens: 每个块的最大token数
            overlap_ratio: 重叠比例，默认为0.2（20%）
            
        返回:
            ParagraphChunk对象列表，包含所有分块结果及其元数据
        
        异常:
            ValueError: 当max_tokens不为正数或overlap_ratio不在有效范围内时抛出
        """
        if max_tokens <= 0:
            raise ValueError("最大token数必须为正数")
        
        if overlap_ratio < 0 or overlap_ratio >= 1:
            raise ValueError("重叠比例必须在[0, 1)范围内")
        
        # 第一步：按段落粗分
        paragraphs = self.split_into_paragraphs(text)
        
        # 第二步：对超限段落进行二次切分
        all_chunks = []
        global_chunk_id = 0
        
        for para_id, (para_text, start_pos, end_pos) in enumerate(paragraphs):
            # 计算段落的token数
            token_count = self.count_tokens(para_text)
            
            # 如果段落token数超过最大限制，进行二次切分
            if token_count > max_tokens:
                para_chunks = self.split_paragraph_into_chunks(
                    para_text, max_tokens, overlap_ratio, para_id, start_pos
                )
                all_chunks.extend(para_chunks)
            else:
                # 否则，保留原始段落
                chunk = ParagraphChunk(
                    text=para_text,
                    level=1,
                    parent_para_id=para_id,
                    chunk_id=global_chunk_id,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    token_count=token_count
                )
                all_chunks.append(chunk)
            
            global_chunk_id += 1
        
        return all_chunks
    
    def convert_to_json(self, chunks: List[ParagraphChunk]) -> List[Dict]:
        """将ParagraphChunk对象列表转换为JSON可序列化的字典列表
        
        将ParagraphChunk对象转换为包含相同信息的普通Python字典，便于JSON序列化和存储。
        特别保留了level和parent_para_id等层级信息，以便后续分析块之间的关系。
        
        参数:
            chunks: ParagraphChunk对象列表
            
        返回:
            可序列化的字典列表，包含所有分块结果及其元数据
        """
        return [{
            "text": chunk.text,
            "level": chunk.level,
            "parent_para_id": chunk.parent_para_id,
            "chunk_id": chunk.chunk_id,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "token_count": chunk.token_count
        } for chunk in chunks]
    
    def save_chunks_to_json(self, chunks: List[ParagraphChunk], output_path: str) -> None:
        """将块保存为JSON文件
        
        将分块结果转换为JSON格式并保存到指定路径，便于后续分析和使用。
        
        参数:
            chunks: ParagraphChunk对象列表
            output_path: 输出文件路径
        """
        json_data = self.convert_to_json(chunks)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"分块结果已保存到: {output_path}")
    
    def print_chunk_summary(self, chunks: List[ParagraphChunk]) -> None:
        """打印块的摘要信息
        
        打印分块结果的关键信息，包括块数量、每个块的层级、父段落关系、token数量
        字符位置和文本预览，便于用户快速了解分块效果。
        
        参数:
            chunks: ParagraphChunk对象列表
        """
        print(f"总块数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"块 {i}:")
            print(f"  层级: {chunk.level}")
            print(f"  父段落ID: {chunk.parent_para_id}")
            print(f"  Token数量: {chunk.token_count}")
            print(f"  字符范围: [{chunk.start_pos}, {chunk.end_pos})"),
            print(f"  文本预览: {chunk.text[:100]}...")
            print("-")

# 示例使用函数
def main():
    """示例函数，展示如何使用RecursiveParagraphFirstSplitter"""
    # 创建分块器实例 - 使用简单tokenizer，不依赖网络
    splitter = RecursiveParagraphFirstSplitter(use_simple_tokenizer=True)
    
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
    max_tokens = 100  # 每个块的最大token数
    overlap_ratio = 0.2  # 20%的重叠率
    
    # 执行分块
    print(f"使用最大token数={max_tokens}, 重叠率={overlap_ratio}进行分块...")
    chunks = splitter.split_text(sample_text, max_tokens, overlap_ratio)
    
    # 打印摘要
    splitter.print_chunk_summary(chunks)
    
    # 保存结果
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "recursive_paragraph_chunks.json"
    splitter.save_chunks_to_json(chunks, str(output_path))
    
    # 测试不同参数
    print("\n测试不同的最大token数...")
    
    # 参数组合测试
    test_params = [
        50,   # 较小的token限制，会触发更多的二次切分
        200,  # 较大的token限制，可能保留更多原始段落
        300   # 更大的token限制
    ]
    
    for i, max_tok in enumerate(test_params):
        print(f"\n测试 {i+1}: 最大token数={max_tok}")
        test_chunks = splitter.split_text(sample_text, max_tok, overlap_ratio)
        print(f"  生成块数: {len(test_chunks)}")
        # 统计不同层级的块数
        level1_count = sum(1 for chunk in test_chunks if chunk.level == 1)
        level2_count = sum(1 for chunk in test_chunks if chunk.level == 2)
        print(f"  一级块数(原始段落): {level1_count}")
        print(f"  二级块数(二次切分): {level2_count}")
        # 保存测试结果
        test_output_path = output_dir / f"recursive_paragraph_chunks_test_{i+1}.json"
        splitter.save_chunks_to_json(test_chunks, str(test_output_path))
    
    # 提供一个从文件加载文本的示例
    print("\n提供一个从文件加载文本的示例...")
    corpus_path = "corpus.txt"
    if Path(corpus_path).exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_text = f.read()
        print(f"从文件加载了{len(corpus_text)}个字符的文本")
        corpus_chunks = splitter.split_text(corpus_text, max_tokens, overlap_ratio)
        print(f"对加载的文本生成了{len(corpus_chunks)}个块")
        # 保存结果
        corpus_output_path = output_dir / "recursive_paragraph_chunks_corpus.json"
        splitter.save_chunks_to_json(corpus_chunks, str(corpus_output_path))
    else:
        print(f"文件{corpus_path}不存在")

if __name__ == "__main__":
    main()