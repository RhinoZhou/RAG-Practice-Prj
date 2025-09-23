#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本预处理模块
负责文档清洗、分块等预处理操作
"""

from typing import List, Dict, Any, Optional
from app.models import Document

class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        初始化文本预处理类
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 块之间的重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        # TODO: 实现文本清洗功能
        # 1. 去除多余空格和换行符
        # 2. 去除特殊字符
        # 3. 规范化文本格式
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 原始文本
        
        Returns:
            分块后的文本列表
        """
        # TODO: 实现文本分块功能
        # 1. 按照chunk_size和chunk_overlap分块
        # 2. 考虑句子边界
        return [text]
    
    def preprocess_document(self, document: Document) -> List[Document]:
        """
        预处理单个文档
        
        Args:
            document: 原始文档
        
        Returns:
            预处理后的文档列表
        """
        # TODO: 实现文档预处理功能
        # 1. 清洗文本
        # 2. 分块处理
        # 3. 添加块ID等元数据
        cleaned_text = self.clean_text(document.text)
        chunks = self.chunk_text(cleaned_text)
        
        processed_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                id=f"{document.id}_chunk_{i}",
                text=chunk,
                metadata={
                    **document.metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            processed_docs.append(chunk_doc)
        
        return processed_docs
    
    def batch_preprocess(self, documents: List[Document]) -> List[Document]:
        """
        批量预处理文档
        
        Args:
            documents: 原始文档列表
        
        Returns:
            预处理后的文档列表
        """
        processed_docs = []
        for doc in documents:
            processed = self.preprocess_document(doc)
            processed_docs.extend(processed)
        
        return processed_docs

    def normalize(self, text: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        规范化用户查询文本
        
        Args:
            text: 用户查询文本
            user_profile: 用户画像信息，可选
        
        Returns:
            规范化后的文本
        """
        # 1. 清洗空白字符
        normalized_text = self.clean_text(text)
        
        # 2. 归一化时间表达
        normalized_text = self._normalize_time_expressions(normalized_text)
        
        # 3. 归一化数量表达
        normalized_text = self._normalize_quantity_expressions(normalized_text)
        
        # 4. 考虑用户画像的个性化处理
        if user_profile:
            normalized_text = self._personalize_based_on_profile(normalized_text, user_profile)
        
        return normalized_text
    
    def _normalize_time_expressions(self, text: str) -> str:
        """
        归一化时间表达
        
        Args:
            text: 原始文本
        
        Returns:
            归一化时间表达后的文本
        """
        now = datetime.now()
        normalized_text = text
        
        # 处理最新
        if self.time_regex_patterns['latest'].search(normalized_text):
            # 获取当前日期的字符串表示
            today_str = now.strftime('%Y-%m-%d')
            normalized_text = self.time_regex_patterns['latest'].sub(f'{today_str}之前', normalized_text)
        
        # 处理上个月
        if self.time_regex_patterns['last_month'].search(normalized_text):
            last_month = now.month - 1 if now.month > 1 else 12
            last_month_year = now.year if now.month > 1 else now.year - 1
            last_month_str = f'{last_month_year}-{last_month:02d}'
            normalized_text = self.time_regex_patterns['last_month'].sub(last_month_str, normalized_text)
        
        # 处理上一周
        if self.time_regex_patterns['last_week'].search(normalized_text):
            last_week = now - timedelta(days=7)
            last_week_str = last_week.strftime('%Y-%m-%d')
            normalized_text = self.time_regex_patterns['last_week'].sub(f'{last_week_str}到{now.strftime("%Y-%m-%d")}', normalized_text)
        
        # 处理本月
        if self.time_regex_patterns['this_month'].search(normalized_text):
            this_month_str = f'{now.year}-{now.month:02d}'
            normalized_text = self.time_regex_patterns['this_month'].sub(this_month_str, normalized_text)
        
        # 处理本周
        if self.time_regex_patterns['this_week'].search(normalized_text):
            # 获取本周一的日期
            monday = now - timedelta(days=now.weekday())
            monday_str = monday.strftime('%Y-%m-%d')
            normalized_text = self.time_regex_patterns['this_week'].sub(f'{monday_str}到{now.strftime("%Y-%m-%d")}', normalized_text)
        
        # 处理昨天
        if self.time_regex_patterns['yesterday'].search(normalized_text):
            yesterday = now - timedelta(days=1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            normalized_text = self.time_regex_patterns['yesterday'].sub(yesterday_str, normalized_text)
        
        # 处理今天
        if self.time_regex_patterns['today'].search(normalized_text):
            today_str = now.strftime('%Y-%m-%d')
            normalized_text = self.time_regex_patterns['today'].sub(today_str, normalized_text)
        
        return normalized_text
    
    def _normalize_quantity_expressions(self, text: str) -> str:
        """
        归一化数量表达
        
        Args:
            text: 原始文本
        
        Returns:
            归一化数量表达后的文本
        """
        # 处理一些常见的数量表达
        quantity_mappings = {
            '几个': '3-5个',
            '多个': '3个以上',
            '少量': '1-2个',
            '大量': '10个以上',
            '大部分': '70%以上',
            '小部分': '30%以下',
        }
        
        normalized_text = text
        for pattern, replacement in quantity_mappings.items():
            normalized_text = normalized_text.replace(pattern, replacement)
        
        return normalized_text
    
    def _personalize_based_on_profile(self, text: str, user_profile: Dict[str, Any]) -> str:
        """
        根据用户画像进行个性化处理
        
        Args:
            text: 原始文本
            user_profile: 用户画像信息
        
        Returns:
            个性化处理后的文本
        """
        # 可以根据用户画像中的信息进行个性化处理
        # 例如，用户所在地区、行业等
        # 这里仅作为示例，实际应用中可以根据具体需求扩展
        if 'region' in user_profile and user_profile['region']:
            region = user_profile['region']
            if '本地区' in text or '本地' in text:
                text = text.replace('本地区', region).replace('本地', region)
        
        return text

# 全局实例，方便直接调用
preprocessor = TextPreprocessor()

def normalize(text: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
    """
    全局规范化函数，清洗空白、将"最新/上个月/一周"等归一化为明确的时间/数量表达
    
    Args:
        text: 用户查询文本
        user_profile: 用户画像信息，可选
    
    Returns:
        规范化后的文本
    """
    return preprocessor.normalize(text, user_profile)

# TODO: 实现更多预处理功能
# 1. 分句处理
# 2. 关键词提取
# 3. 实体识别
# 4. 文本向量化预处理