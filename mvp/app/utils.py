#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
提供各种通用工具函数
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import re
import json
import os
import uuid
import time
import hashlib
from datetime import datetime

# 请求ID生成函数
def generate_request_id() -> str:
    """
    生成唯一的请求ID
    
    Returns:
        唯一的请求ID字符串
    """
    return str(uuid.uuid4())

# 文本处理工具
def clean_text(text: str) -> str:
    """
    清理文本，去除多余空格、换行等
    
    Args:
        text: 输入文本
    
    Returns:
        清理后的文本
    """
    # TODO: 实现更复杂的文本清理逻辑
    # 去除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    text = text.strip()
    return text

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本分割成指定大小的块
    
    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap: 重叠大小
    
    Returns:
        文本块列表
    """
    # TODO: 实现更智能的文本分块（基于句子、段落等）
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        top_k: 返回的关键词数量
    
    Returns:
        关键词列表
    """
    # TODO: 实现更复杂的关键词提取算法
    # 简单的词频统计（忽略停用词）
    stop_words = set([
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
        '会', '着', '没有', '看', '好', '自己', '这', 'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at'
    ])
    
    # 分词
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 过滤停用词
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # 计算词频
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # 排序并返回top_k个词
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_k]]

# 文件处理工具
def read_file(file_path: str) -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return ""

def write_file(file_path: str, content: str) -> bool:
    """
    写入文件内容
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"写入文件失败 {file_path}: {e}")
        return False

def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    列出目录下的所有文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，None表示所有文件
    
    Returns:
        文件路径列表
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

# JSON处理工具
def load_json(file_path: str) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败 {file_path}: {e}")
        return {}

def save_json(file_path: str, data: Dict[str, Any]) -> bool:
    """
    保存数据为JSON文件
    
    Args:
        file_path: 文件路径
        data: 要保存的数据
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON文件失败 {file_path}: {e}")
        return False

# 时间和ID工具
def generate_id(prefix: str = "") -> str:
    """
    生成唯一ID
    
    Args:
        prefix: ID前缀
    
    Returns:
        唯一ID字符串
    """
    uid = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{uid}"
    return uid

def get_current_timestamp() -> float:
    """
    获取当前时间戳
    
    Returns:
        当前时间戳（秒）
    """
    return time.time()

def format_timestamp(timestamp: float) -> str:
    """
    格式化时间戳
    
    Args:
        timestamp: 时间戳
    
    Returns:
        格式化的时间字符串
    """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# 哈希和加密工具
def calculate_hash(data: Union[str, bytes]) -> str:
    """
    计算数据的哈希值
    
    Args:
        data: 输入数据
    
    Returns:
        哈希值字符串
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()

def hash_file(file_path: str) -> str:
    """
    计算文件的哈希值
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件哈希值
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"计算文件哈希失败 {file_path}: {e}")
        return ""

# 数学和统计工具
def calculate_mean(values: List[float]) -> float:
    """
    计算平均值
    
    Args:
        values: 数值列表
    
    Returns:
        平均值
    """
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_recall(relevant: List[Any], retrieved: List[Any]) -> float:
    """
    计算召回率
    
    Args:
        relevant: 相关项目列表
        retrieved: 检索到的项目列表
    
    Returns:
        召回率
    """
    if not relevant:
        return 1.0
    relevant_set = set(relevant)
    retrieved_set = set(retrieved)
    intersection = relevant_set.intersection(retrieved_set)
    return len(intersection) / len(relevant_set)

def calculate_precision(relevant: List[Any], retrieved: List[Any]) -> float:
    """
    计算精确率
    
    Args:
        relevant: 相关项目列表
        retrieved: 检索到的项目列表
    
    Returns:
        精确率
    """
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    retrieved_set = set(retrieved)
    intersection = relevant_set.intersection(retrieved_set)
    return len(intersection) / len(retrieved_set)

# TODO: 实现更多工具函数
# 1. 多语言支持工具
# 2. 并行处理工具
# 3. 进度条显示工具
# 4. 性能分析工具


# 文本分词工具
def tokenize(text: str, language: str = 'zh') -> List[str]:
    """
    对文本进行分词
    
    Args:
        text: 输入文本
        language: 语言类型，支持'zh'（中文）和'en'（英文）
    
    Returns:
        分词后的词语列表
    """
    if not text:
        return []
        
    if language == 'zh':
        # 中文分词 - 使用简单的正则分词（实际应用中可以集成jieba等分词库）
        # 基本分词逻辑：保留中文字符、英文单词、数字
        import re
        tokens = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+', text)
        return tokens
    else:
        # 英文分词 - 使用空格和标点符号分词
        import re
        # 移除非字母数字字符，并转换为小写
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # 按空格分词并过滤空字符串
        tokens = [token for token in text.split() if token]
        return tokens


# 时间处理工具
def format_time_duration(duration_ms: float) -> str:
    """
    格式化时间持续时间
    
    Args:
        duration_ms: 持续时间（毫秒）
    
    Returns:
        格式化的时间字符串
    """
    if duration_ms < 1:
        return f"{duration_ms:.2f}μs"
    elif duration_ms < 1000:
        return f"{duration_ms:.2f}ms"
    elif duration_ms < 60000:
        seconds = duration_ms / 1000
        return f"{seconds:.2f}s"
    else:
        minutes = duration_ms / 60000
        return f"{minutes:.2f}m"


def get_time_diff_str(start_time: float) -> str:
    """
    获取从开始时间到当前时间的差异字符串
    
    Args:
        start_time: 开始时间戳
    
    Returns:
        时间差异字符串
    """
    duration = (time.time() - start_time) * 1000  # 转换为毫秒
    return format_time_duration(duration)


# 相似度计算工具
def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
    
    Returns:
        余弦相似度值（-1到1之间）
    """
    import math
    
    # 检查向量长度是否相同
    if len(vec1) != len(vec2):
        raise ValueError("向量长度必须相同")
    
    # 计算点积
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # 计算向量长度
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    # 避免除以零
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算余弦相似度
    return dot_product / (norm1 * norm2)


def approximate_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    计算文本相似度的近似值
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        method: 相似度计算方法，支持'jaccard'（杰卡德系数）和'levenshtein'（编辑距离）
    
    Returns:
        相似度值（0到1之间）
    """
    if not text1 or not text2:
        return 0.0 if text1 != text2 else 1.0
        
    if method == 'jaccard':
        # 分词
        tokens1 = set(tokenize(text1))
        tokens2 = set(tokenize(text2))
        
        # 计算交集和并集
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        # 避免除以零
        if union == 0:
            return 0.0
        
        # 计算杰卡德系数
        return intersection / union
    elif method == 'levenshtein':
        # 简化版编辑距离计算
        # 这里使用快速计算，不考虑复杂的编辑操作
        import difflib
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")


# 文本处理增强工具
def truncate_text(text: str, max_length: int = 200, suffix: str = '...') -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后的后缀
    
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def normalize_text(text: str) -> str:
    """
    标准化文本
    
    Args:
        text: 输入文本
    
    Returns:
        标准化后的文本
    """
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 转换为小写
    text = text.lower()
    # 去除首尾空格
    return text.strip()