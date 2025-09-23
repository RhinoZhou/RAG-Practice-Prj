#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量检索器
负责使用向量相似度进行文本检索
"""

from typing import List, Dict, Any, Optional
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from app.models import Evidence
from app.config import config

class VectorSearcher:
    """向量检索器"""
    
    def __init__(self, index_path: str = None, meta_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量检索器
        
        Args:
            index_path: FAISS索引文件路径
            meta_path: 元数据文件路径
            model_name: SentenceTransformer模型名称
        """
        self.index_path = index_path or os.path.join(config.INDEX_DIR, "vector.faiss")
        self.meta_path = meta_path or os.path.join(config.INDEX_DIR, "meta.jsonl")
        self.model_name = model_name
        
        self.index = None
        self.docs = []
        self.model = None
        self.index_loaded = False
        self.model_loaded = False
        
        # 加载索引和模型
        self.load_model()
        self.load_index()
    
    def load_model(self) -> bool:
        """
        加载SentenceTransformer模型
        
        Returns:
            是否加载成功
        """
        try:
            print(f"加载SentenceTransformer模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model_loaded = True
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model_loaded = False
            return False
    
    def load_index(self) -> bool:
        """
        加载FAISS索引和元数据
        
        Returns:
            是否加载成功
        """
        try:
            # 检查索引文件是否存在
            if not os.path.exists(self.index_path):
                print(f"错误: 索引文件不存在 {self.index_path}")
                return False
            
            # 检查元数据文件是否存在
            if not os.path.exists(self.meta_path):
                print(f"错误: 元数据文件不存在 {self.meta_path}")
                return False
            
            # 加载FAISS索引
            print(f"加载FAISS索引: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            # 加载元数据
            print(f"加载元数据: {self.meta_path}")
            self.docs = []
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        self.docs.append(doc)
                    except Exception as e:
                        print(f"解析元数据行失败: {e}")
            
            # 验证索引和元数据数量是否匹配
            if self.index.ntotal != len(self.docs):
                print(f"警告: 索引向量数量 ({self.index.ntotal}) 与元数据数量 ({len(self.docs)}) 不匹配")
            
            self.index_loaded = True
            print(f"索引加载成功，向量数: {self.index.ntotal}, 文档数: {len(self.docs)}")
            return True
            
        except Exception as e:
            print(f"加载索引失败: {e}")
            self.index_loaded = False
            return False
    
    def search(self, query: str, params: Dict[str, Any], filters: Dict[str, Any] = None) -> List[Evidence]:
        """
        执行向量检索
        
        Args:
            query: 搜索查询
            params: 检索参数，包含top_k等
            filters: 过滤条件，支持year和platform字段
        
        Returns:
            检索结果列表，每个元素为Evidence对象
        """
        # 检查索引和模型是否加载成功
        if not self.index_loaded or not self.model_loaded:
            print("警告: 索引或模型未加载成功，无法执行搜索")
            # 尝试重新加载
            self.load_index()
            self.load_model()
            if not self.index_loaded or not self.model_loaded:
                return []
        
        # 检查输入参数
        if not query:
            return []
        
        # 获取top_k参数
        top_k = params.get("top_k", config.TOP_K)
        top_k = max(1, min(top_k, 100))  # 限制在1-100之间
        
        # 使用3*k进行粗召回，然后应用过滤收缩到k
        coarse_top_k = min(top_k * 3, self.index.ntotal if self.index else 0)
        
        try:
            # 对查询进行编码
            query_vector = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
            
            # 确保向量是二维数组
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 归一化查询向量（与索引向量保持一致）
            faiss.normalize_L2(query_vector)
            
            # 执行向量检索，获取粗召回结果
            distances, indices = self.index.search(query_vector, coarse_top_k)
            
            # 构建初步结果列表
            raw_results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # 检查索引是否有效
                if 0 <= idx < len(self.docs):
                    doc = self.docs[idx]
                    # 注意：内积相似度，分数越高表示越相似
                    raw_results.append({
                        "doc": doc,
                        "score": float(distance),
                        "index": idx
                    })
            
            # 应用过滤条件
            filtered_results = self._apply_filters(raw_results, filters)
            
            # 按照分数从高到低排序，并取前top_k个结果
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            filtered_results = filtered_results[:top_k]
            
            # 转换为Evidence对象列表
            evidence_results = []
            for result in filtered_results:
                doc = result["doc"]
                
                # 创建Evidence对象
                evidence = Evidence(
                    id=doc.get("doc_id", ""),
                    content=doc.get("content", ""),
                    source=doc.get("source", ""),
                    title=doc.get("title", ""),
                    score=result["score"],
                    metadata={
                        "year": doc.get("year", ""),
                        "platform": doc.get("platform", ""),
                        "region": doc.get("region", "")
                    }
                )
                
                evidence_results.append(evidence)
            
            return evidence_results
            
        except Exception as e:
            print(f"向量检索过程中出错: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        应用过滤条件，当年份过滤缺失时放宽到近邻年份(±1)
        
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
            
            # 检查年份过滤 - 缺失时放宽到±1年
            if "year" in filters:
                filter_year = filters["year"]
                doc_year = doc.get("year", "")
                
                if doc_year:
                    # 尝试转换为整数比较
                    try:
                        filter_year_int = int(filter_year)
                        doc_year_int = int(doc_year)
                        
                        # 检查是否在±1范围内
                        if abs(doc_year_int - filter_year_int) > 1:
                            match = False
                    except ValueError:
                        # 无法转换为整数时使用字符串精确匹配
                        filter_year_str = str(filter_year)
                        if filter_year_str not in doc_year:
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
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        生成文本的向量嵌入
        
        Args:
            text: 输入文本
        
        Returns:
            向量嵌入
        """
        return self.model.encode([text])[0]

# 创建全局实例以便其他模块使用
vector_searcher = VectorSearcher()