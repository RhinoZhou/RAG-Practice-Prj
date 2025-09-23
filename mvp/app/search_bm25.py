#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25检索器
负责使用BM25算法进行文本检索
"""

import os
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.models import Evidence
from app.config import config

class BM25Searcher:
    """BM25检索器"""
    
    def __init__(self, index_path: str = None):
        """
        初始化BM25检索器
        
        Args:
            index_path: BM25索引文件路径
        """
        self.index_path = index_path or os.path.join(config.INDEX_DIR, "bm25.pkl")
        self.bm25 = None
        self.docs = []
        self.index_loaded = False
        
        # 加载索引
        self.load_index()
    
    def load_index(self) -> bool:
        """
        加载BM25索引
        
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(self.index_path):
                print(f"错误: BM25索引文件不存在 {self.index_path}")
                return False
            
            print(f"加载BM25索引: {self.index_path}")
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # 提取索引数据
            self.bm25 = index_data.get("bm25")
            self.docs = index_data.get("docs", [])
            
            if self.bm25 is None or not self.docs:
                print(f"错误: 索引数据不完整")
                return False
            
            self.index_loaded = True
            print(f"BM25索引加载成功，文档数: {len(self.docs)}")
            return True
            
        except Exception as e:
            print(f"加载BM25索引失败: {e}")
            return False
    
    def simple_tokenize(self, text: str) -> List[str]:
        """
        简单的文本分词函数
        中文按字符分割，英文按空格分割
        
        Args:
            text: 输入文本
        
        Returns:
            分词后的词汇列表
        """
        # TODO: 这里可以替换为更复杂的分词库如jieba
        tokens = []
        temp_english = ""
        
        for char in text:
            # 检查是否为英文字符或数字
            if re.match(r'[a-zA-Z0-9]', char):
                temp_english += char
            else:
                if temp_english:
                    tokens.append(temp_english.lower())
                    temp_english = ""
                if char.strip():
                    tokens.append(char)
        
        # 处理最后一个英文单词
        if temp_english:
            tokens.append(temp_english.lower())
        
        return tokens
    
    def search(self, query: str, params: Dict[str, Any], filters: Dict[str, Any] = None) -> List[Evidence]:
        """
        执行BM25检索
        
        Args:
            query: 搜索查询
            params: 检索参数，包含top_k等
            filters: 过滤条件，支持year和platform字段
        
        Returns:
            检索结果列表，每个元素为Evidence对象
        """
        # 检查索引是否加载成功
        if not self.index_loaded:
            print("警告: 索引未加载成功，无法执行搜索")
            return []
        
        # 检查输入参数
        if not query:
            return []
        
        # 获取top_k参数
        top_k = params.get("top_k", config.TOP_K)
        top_k = max(1, min(top_k, 100))  # 限制在1-100之间
        
        # 分词处理查询
        tokenized_query = self.simple_tokenize(query)
        
        # 如果查询分词后为空，直接返回空结果
        if not tokenized_query:
            return []
        
        try:
            # 使用BM25进行检索
            scores = self.bm25.get_scores(tokenized_query)
            
            # 按照分数从高到低排序，获取前top_k个结果的索引
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]  # 获取两倍的结果用于过滤
            
            # 构建初步结果列表
            raw_results = []
            for idx in top_indices:
                if idx < len(self.docs):
                    doc = self.docs[idx]
                    raw_results.append({
                        "doc": doc,
                        "score": scores[idx]
                    })
            
            # 应用过滤条件
            filtered_results = self._apply_filters(raw_results, filters)
            
            # 转换为Evidence对象列表
            evidence_results = []
            for result in filtered_results[:top_k]:  # 确保最终返回top_k个结果
                doc = result["doc"]
                
                # 创建Evidence对象
                evidence = Evidence(
                    id=doc.get("doc_id", ""),
                    content=doc.get("content", ""),
                    source=doc.get("source", ""),
                    title=doc.get("title", ""),
                    score=float(result["score"]),  # 将分数转换为float
                    metadata={
                        "year": doc.get("year", ""),
                        "platform": doc.get("platform", ""),
                        "region": doc.get("region", "")
                    }
                )
                
                evidence_results.append(evidence)
            
            return evidence_results
            
        except Exception as e:
            print(f"BM25检索过程中出错: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        应用过滤条件
        
        Args:
            results: 原始检索结果列表
            filters: 过滤条件字典，支持year和platform字段
        
        Returns:
            过滤后的结果列表
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            doc = result["doc"]
            # 检查是否满足所有过滤条件
            match = True
            
            # 检查年份过滤
            if "year" in filters:
                doc_year = doc.get("year", "")
                filter_year = str(filters["year"])
                if doc_year and filter_year not in doc_year:
                    match = False
            
            # 检查平台过滤
            if "platform" in filters and match:
                doc_platform = doc.get("platform", "")
                filter_platform = filters["platform"]
                # 支持多个平台的过滤条件
                if isinstance(filter_platform, list):
                    if doc_platform not in filter_platform:
                        match = False
                else:
                    if doc_platform != filter_platform:
                        match = False
            
            # 如果满足所有过滤条件，添加到结果列表
            if match:
                filtered_results.append(result)
        
        return filtered_results

# 创建全局实例以便其他模块使用
bm25_searcher = BM25Searcher()