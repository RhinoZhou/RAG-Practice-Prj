#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于Markdown/PDF的结构化分块演示

实现基于标题层级（H2/H3）的结构化分块，保护代码块与表格，生成section_path路径信息。
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# 尝试导入tiktoken库，如果不存在则使用简单的tokenizer
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    print("未找到tiktoken库，将使用简单的字符tokenizer")

@dataclass
class HierarchicalChunk:
    """表示一个层次化分块及其元数据
    
    属性:
        text: 块文本内容
        section_path: 标题路径，如 ["主标题", "子标题"]
        type: 块类型 ("heading", "code_block", "table", "paragraph")
        level: 标题级别，如果不是标题则为0
        chunk_id: 块ID
        start_pos: 原始文本中的起始位置
        end_pos: 原始文本中的结束位置
        token_count: token数量
    """
    text: str
    section_path: List[str]
    type: str
    level: int = 0
    chunk_id: int = 0
    start_pos: int = 0
    end_pos: int = 0
    token_count: int = 0

class SimpleCharacterTokenizer:
    """简单的字符tokenizer实现，用于在tiktoken不可用时作为替代"""
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为字符级别token
        
        参数:
            text: 输入文本
            
        返回:
            token列表
        """
        # 简单的字符级别分割，保留空白符和标点符号
        return list(text)
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        参数:
            text: 输入文本
            
        返回:
            token数量
        """
        return len(self.tokenize(text))
    
    def decode(self, tokens: List[str]) -> str:
        """将token列表解码回文本
        
        参数:
            tokens: token列表
            
        返回:
            解码后的文本
        """
        return ''.join(tokens)

class MarkdownHierarchicalSplitter:
    """基于Markdown的层次化分块器"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base", use_simple_tokenizer: bool = False):
        """初始化分块器
        
        参数:
            tokenizer_name: 使用的tokenizer名称
            use_simple_tokenizer: 是否强制使用简单的字符tokenizer
        """
        self.tokenizer_name = tokenizer_name
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        if self.use_simple_tokenizer or not HAS_TIKTOKEN:
            print("使用简单的字符tokenizer")
            return SimpleCharacterTokenizer()
        else:
            try:
                print(f"使用tiktoken: {self.tokenizer_name}")
                return tiktoken.get_encoding(self.tokenizer_name)
            except Exception as e:
                print(f"加载tiktoken失败: {e}，将使用简单的字符tokenizer")
                return SimpleCharacterTokenizer()
    
    def tokenize(self, text: str) -> List[str]:
        """分词方法
        
        参数:
            text: 输入文本
            
        返回:
            token列表
        """
        # 直接检查tokenizer的类型，避免条件判断错误
        if isinstance(self.tokenizer, SimpleCharacterTokenizer):
            return self.tokenizer.tokenize(text)
        else:
            # 对于tiktoken的tokenizer，使用encode方法
            return self.tokenizer.encode(text, disallowed_special=())
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        参数:
            text: 输入文本
            
        返回:
            token数量
        """
        # 直接检查tokenizer的类型，避免条件判断错误
        if isinstance(self.tokenizer, SimpleCharacterTokenizer):
            return self.tokenizer.count_tokens(text)
        else:
            # 对于tiktoken的tokenizer，使用encode方法计算长度
            return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def parse_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """解析Markdown文本，提取标题、代码块、表格和段落
        
        参数:
            markdown_text: Markdown文本
            
        返回:
            元素列表，每个元素包含类型、内容、位置和级别等信息
        """
        elements = []
        current_pos = 0
        
        # 正则表达式模式
        heading_pattern = r'^(#{2,6})\s+([^\n]+)\n'
        code_block_pattern = r'```(\w*?)\n(.*?)\n```'
        table_pattern = r'^\|.*\|\n\|[-| :]+\|\n(?:\|.*\|\n)*'
        paragraph_pattern = r'(?:^[^\n]+\n)+'
        
        # 按顺序尝试匹配不同类型的内容
        while current_pos < len(markdown_text):
            # 尝试匹配标题
            heading_match = re.match(heading_pattern, markdown_text[current_pos:], re.MULTILINE)
            if heading_match:
                level = len(heading_match.group(1))
                content = heading_match.group(2).strip()
                end_pos = current_pos + heading_match.end()
                elements.append({
                    "type": "heading",
                    "content": content,
                    "level": level,
                    "start_pos": current_pos,
                    "end_pos": end_pos
                })
                current_pos = end_pos
                continue
            
            # 尝试匹配代码块
            code_block_match = re.match(code_block_pattern, markdown_text[current_pos:], re.DOTALL)
            if code_block_match:
                language = code_block_match.group(1)
                content = code_block_match.group(2)
                full_content = f"```{language}\n{content}\n```"
                end_pos = current_pos + code_block_match.end()
                elements.append({
                    "type": "code_block",
                    "content": full_content,
                    "language": language,
                    "start_pos": current_pos,
                    "end_pos": end_pos
                })
                current_pos = end_pos
                continue
            
            # 尝试匹配表格
            table_match = re.match(table_pattern, markdown_text[current_pos:], re.MULTILINE)
            if table_match:
                content = table_match.group(0)
                end_pos = current_pos + table_match.end()
                elements.append({
                    "type": "table",
                    "content": content,
                    "start_pos": current_pos,
                    "end_pos": end_pos
                })
                current_pos = end_pos
                continue
            
            # 尝试匹配段落
            paragraph_match = re.match(paragraph_pattern, markdown_text[current_pos:], re.MULTILINE)
            if paragraph_match:
                content = paragraph_match.group(0).strip()
                if content:  # 跳过空段落
                    end_pos = current_pos + paragraph_match.end()
                    elements.append({
                        "type": "paragraph",
                        "content": content,
                        "start_pos": current_pos,
                        "end_pos": end_pos
                    })
                current_pos = current_pos + paragraph_match.end()
                continue
            
            # 如果没有匹配到任何模式，向前移动一个字符
            current_pos += 1
        
        return elements
    
    def build_section_path(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为每个元素构建section_path
        
        参数:
            elements: 解析后的元素列表
            
        返回:
            添加了section_path的元素列表
        """
        section_stack = []  # 存储当前的标题层级
        result = []
        
        for element in elements:
            if element["type"] == "heading":
                level = element["level"]
                # 移除比当前级别高或相等的标题
                while section_stack and section_stack[-1]["level"] >= level:
                    section_stack.pop()
                # 添加当前标题
                section_stack.append({
                    "level": level,
                    "content": element["content"]
                })
                # 构建section_path
                section_path = [item["content"] for item in section_stack]
                element["section_path"] = section_path
            else:
                # 非标题元素使用当前的section_path
                element["section_path"] = [item["content"] for item in section_stack]
            
            result.append(element)
        
        return result
    
    def split_paragraph_into_chunks(self, para_content: str, section_path: List[str], 
                                  max_tokens: int, overlap_ratio: float, 
                                  start_pos: int, base_chunk_id: int) -> List[HierarchicalChunk]:
        """将段落按tokens分割成小块
        
        参数:
            para_content: 段落内容
            section_path: 段落所属的section_path
            max_tokens: 最大token数
            overlap_ratio: 重叠比例
            start_pos: 段落起始位置
            base_chunk_id: 基础块ID
            
        返回:
            HierarchicalChunk对象列表
        """
        chunks = []
        tokens = self.tokenize(para_content)
        token_count = len(tokens)
        
        if token_count <= max_tokens:
            # 如果段落token数不超过最大限制，直接作为一个块
            chunks.append(HierarchicalChunk(
                text=para_content,
                section_path=section_path,
                type="paragraph",
                level=0,
                chunk_id=base_chunk_id,
                start_pos=start_pos,
                end_pos=start_pos + len(para_content),
                token_count=token_count
            ))
        else:
            # 计算重叠token数
            overlap_tokens = int(max_tokens * overlap_ratio)
            # 计算块数量
            step = max_tokens - overlap_tokens
            num_chunks = (token_count + step - 1) // step  # 向上取整
            
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + max_tokens, token_count)
                
                # 解码token
                if isinstance(self.tokenizer, SimpleCharacterTokenizer):
                    chunk_text = self.tokenizer.decode(tokens[start_idx:end_idx])
                else:
                    chunk_text = self.tokenizer.decode(tokens[start_idx:end_idx])
                
                # 计算字符位置
                # 这是一个近似计算，在实际应用中可能需要更精确的映射
                avg_char_per_token = len(para_content) / token_count
                chunk_start_pos = start_pos + int(start_idx * avg_char_per_token)
                chunk_end_pos = start_pos + int(end_idx * avg_char_per_token)
                
                chunks.append(HierarchicalChunk(
                    text=chunk_text,
                    section_path=section_path,
                    type="paragraph",
                    level=0,
                    chunk_id=base_chunk_id + i,
                    start_pos=chunk_start_pos,
                    end_pos=chunk_end_pos,
                    token_count=end_idx - start_idx
                ))
        
        return chunks
    
    def split_text(self, markdown_text: str, max_tokens: int = 200, overlap_ratio: float = 0.2) -> List[HierarchicalChunk]:
        """层次化分块的主方法
        
        参数:
            markdown_text: Markdown文本
            max_tokens: 每个块的最大token数
            overlap_ratio: 重叠比例
            
        返回:
            HierarchicalChunk对象列表
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens必须为正数")
        if overlap_ratio < 0 or overlap_ratio >= 1:
            raise ValueError("overlap_ratio必须在[0, 1)范围内")
        
        # 解析Markdown文本
        elements = self.parse_markdown(markdown_text)
        # 构建section_path
        elements_with_path = self.build_section_path(elements)
        
        chunks = []
        chunk_id = 0
        
        for element in elements_with_path:
            if element["type"] in ["code_block", "table"]:
                # 代码块和表格整体保留
                token_count = self.count_tokens(element["content"])
                chunks.append(HierarchicalChunk(
                    text=element["content"],
                    section_path=element["section_path"],
                    type=element["type"],
                    level=0,
                    chunk_id=chunk_id,
                    start_pos=element["start_pos"],
                    end_pos=element["end_pos"],
                    token_count=token_count
                ))
                chunk_id += 1
            elif element["type"] == "heading":
                # 标题作为独立块
                token_count = self.count_tokens(element["content"])
                chunks.append(HierarchicalChunk(
                    text=element["content"],
                    section_path=element["section_path"],
                    type=element["type"],
                    level=element["level"],
                    chunk_id=chunk_id,
                    start_pos=element["start_pos"],
                    end_pos=element["end_pos"],
                    token_count=token_count
                ))
                chunk_id += 1
            elif element["type"] == "paragraph":
                # 段落可能需要二次切分
                para_chunks = self.split_paragraph_into_chunks(
                    para_content=element["content"],
                    section_path=element["section_path"],
                    max_tokens=max_tokens,
                    overlap_ratio=overlap_ratio,
                    start_pos=element["start_pos"],
                    base_chunk_id=chunk_id
                )
                chunks.extend(para_chunks)
                chunk_id += len(para_chunks)
        
        return chunks
    
    def convert_to_json(self, chunks: List[HierarchicalChunk]) -> List[Dict]:
        """将HierarchicalChunk对象列表转换为JSON可序列化的字典列表
        
        参数:
            chunks: HierarchicalChunk对象列表
            
        返回:
            可序列化的字典列表
        """
        return [{
            "text": chunk.text,
            "section_path": chunk.section_path,
            "type": chunk.type,
            "level": chunk.level,
            "chunk_id": chunk.chunk_id,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "token_count": chunk.token_count
        } for chunk in chunks]
    
    def save_chunks_to_json(self, chunks: List[HierarchicalChunk], output_path: str) -> None:
        """将块保存为JSON文件
        
        参数:
            chunks: HierarchicalChunk对象列表
            output_path: 输出文件路径
        """
        json_data = self.convert_to_json(chunks)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"分块结果已保存到: {output_path}")
    
    def print_chunk_summary(self, chunks: List[HierarchicalChunk]) -> None:
        """打印块的摘要信息
        
        参数:
            chunks: HierarchicalChunk对象列表
        """
        print(f"总块数: {len(chunks)}")
        
        # 按类型统计块数
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.type] = type_counts.get(chunk.type, 0) + 1
        print("块类型统计:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}个")
        
        print("\n块详情:")
        for i, chunk in enumerate(chunks):
            print(f"块 {i}:")
            print(f"  类型: {chunk.type}")
            print(f"  标题级别: {chunk.level}")
            print(f"  路径: {' > '.join(chunk.section_path)}")
            print(f"  Token数量: {chunk.token_count}")
            print(f"  字符范围: [{chunk.start_pos}, {chunk.end_pos})"),
            print(f"  文本预览: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
            print("-")

def load_markdown_file(file_path: str) -> str:
    """加载Markdown文件
    
    参数:
        file_path: 文件路径
        
    返回:
        文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_sample_markdown() -> str:
    """创建示例Markdown文本，用于演示
    
    返回:
        示例Markdown文本
    """
    return """# 人工智能简介

## 机器学习基础

机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习而不需要明确编程。

### 监督学习

监督学习是机器学习的一种方法，其中模型从标记的训练数据中学习。常见的监督学习算法包括：

- 线性回归
- 决策树
- 随机森林
- 神经网络

```python
def train_model(X, y):
    # 训练一个简单的线性回归模型
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    return model

# 示例用法
import numpy as np
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = train_model(X, y)
prediction = model.predict([[6]])  # 应该输出约12
```

### 无监督学习

无监督学习处理未标记的数据，目标是发现数据中的模式或结构。常见的无监督学习任务包括：

| 任务类型 | 算法示例 | 应用场景 |
|---------|---------|---------|
| 聚类 | K-means, DBSCAN | 客户分群, 异常检测 |
| 降维 | PCA, t-SNE | 数据可视化, 特征提取 |
| 关联规则 | Apriori, FP-Growth | 市场购物篮分析 |

## 深度学习

深度学习是机器学习的一个子集，它使用多层神经网络来模拟人类大脑的某些功能。深度学习在以下领域取得了显著的成功：

1. 图像识别
2. 语音识别
3. 自然语言处理
4. 强化学习

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 创建一个简单的神经网络模型
def create_neural_network(input_dim=10):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

# 模型摘要
model = create_neural_network()
model.summary()
```

## 人工智能的伦理问题

随着人工智能技术的快速发展，一系列伦理和社会问题也随之出现：

- 隐私保护：AI系统收集和使用大量个人数据，如何保护用户隐私
- 就业影响：自动化可能导致某些工作岗位消失
- 算法偏见：AI系统可能反映和放大训练数据中的偏见
- 安全风险：AI系统可能被用于恶意目的或遭受攻击

这些问题需要技术专家、政策制定者和社会各界共同思考和解决。"""

def main():
    """主函数，演示层次化分块功能"""
    print("基于Markdown的层次化分块演示")
    print("=" * 50)
    
    # 创建示例Markdown文本
    sample_markdown = create_sample_markdown()
    print(f"生成了{len(sample_markdown)}个字符的示例Markdown文本")
    
    # 初始化分块器
    use_simple = not HAS_TIKTOKEN
    print(f"使用{'简单的字符tokenizer' if use_simple else 'tiktoken'}")
    splitter = MarkdownHierarchicalSplitter(use_simple_tokenizer=use_simple)
    
    # 分块参数
    max_tokens = 150
    overlap_ratio = 0.2
    print(f"使用最大token数={max_tokens}, 重叠率={overlap_ratio}进行分块...")
    
    # 执行分块
    chunks = splitter.split_text(sample_markdown, max_tokens, overlap_ratio)
    
    # 打印摘要
    splitter.print_chunk_summary(chunks)
    
    # 保存结果
    output_path = "results/hierarchical_chunking_chunks.json"
    splitter.save_chunks_to_json(chunks, output_path)
    
    # 测试不同的最大token数
    print("\n测试不同的最大token数...")
    
    test_params = [
        (100, 0.2, "results/hierarchical_chunking_chunks_test_1.json"),  # 较小的块大小
        (300, 0.2, "results/hierarchical_chunking_chunks_test_2.json"),  # 中等的块大小
        (500, 0.2, "results/hierarchical_chunking_chunks_test_3.json")   # 较大的块大小
    ]
    
    for i, (max_tok, overlap, out_path) in enumerate(test_params, 1):
        print(f"\n测试 {i}: 最大token数={max_tok}")
        test_chunks = splitter.split_text(sample_markdown, max_tok, overlap)
        print(f"  生成块数: {len(test_chunks)}")
        
        # 按类型统计
        type_counts = {}
        for chunk in test_chunks:
            type_counts[chunk.type] = type_counts.get(chunk.type, 0) + 1
        print("  块类型统计:")
        for type_name, count in type_counts.items():
            print(f"    {type_name}: {count}个")
        
        # 保存结果
        splitter.save_chunks_to_json(test_chunks, out_path)
    
    print("\n分块演示完成！")

if __name__ == "__main__":
    main()