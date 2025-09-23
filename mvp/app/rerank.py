#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果重排序模块
负责对检索结果进行精排
"""

import time
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache
from cachetools import TTLCache
from sentence_transformers import CrossEncoder
from app.models import Document, Query, Evidence

class Reranker:
    """结果重排序器"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", cache_size=1000, cache_ttl=300):
        """
        初始化重排序器
        
        Args:
            model_name: 用于重排序的模型名称
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间(秒)
        """
        # 初始化TTLCache - 结合LRU淘汰和超时过期
        self.cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        # 加载重排序模型
        try:
            self.model = CrossEncoder(model_name)
            print(f"成功加载重排序模型: {model_name}")
        except Exception as e:
            print(f"加载重排序模型失败: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5, timeout: float = 5.0) -> List[Document]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的文档数量
        
        Returns:
            重排序后的文档列表
        """
        # TODO: 实现重排序功能
        if self.model is None:
            # 如果模型未加载，返回原始文档列表（按原始score排序）
            sorted_docs = sorted(documents, key=lambda x: x.score or 0, reverse=True)
            return sorted_docs[:top_k]
        
        if not documents:
            return []
        
        # 生成缓存键 (query_text + 所有doc_id的组合)
        cache_key = f"{query}::{'-'.join([doc.id for doc in documents])}::{top_k}"
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 准备重排序输入
        pairs = [(query, doc.text) for doc in documents]
        
        # 计算相关性分数并添加超时控制
        start_time = time.time()
        scores = None
        try:
            # 预测分数
            scores = self.model.predict(pairs)
            
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"警告: 重排序超时({elapsed_time:.2f}秒 > {timeout}秒)")
                # 超时情况下仍返回结果但记录警告
        except Exception as e:
            print(f"重排序计算失败: {e}")
            # 出错时返回原始文档列表
            sorted_docs = sorted(documents, key=lambda x: x.score or 0, reverse=True)
            return sorted_docs[:top_k]
        
        # 根据分数对文档进行排序
        scored_docs = [(doc, float(score)) for doc, score in zip(documents, scores)]
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # 更新文档的rank和score
        result_docs = []
        for rank, (doc, score) in enumerate(sorted_docs[:top_k]):
            updated_doc = doc.copy()
            updated_doc.rank = rank + 1
            updated_doc.score = score
            # 保留原始检索的score作为元数据
            if hasattr(updated_doc, 'original_score'):
                pass  # 已经有original_score
            else:
                updated_doc.metadata['original_score'] = doc.score
            result_docs.append(updated_doc)
        
        # 存入缓存
        self.cache[cache_key] = result_docs
        
        return result_docs
    
    def batch_rerank(self, queries: List[str], documents_list: List[List[Document]], top_k: int = 5) -> List[List[Document]]:
        """
        批量执行重排序
        
        Args:
            queries: 查询文本列表
            documents_list: 每个查询对应的文档列表
            top_k: 返回的文档数量
        
        Returns:
            重排序后的文档列表列表
        """
        # TODO: 实现批量重排序功能
        results = []
        for query, documents in zip(queries, documents_list):
            results.append(self.rerank(query, documents, top_k))
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        对分数进行归一化
        
        Args:
            scores: 原始分数列表
        
        Returns:
            归一化后的分数列表
        """
        # TODO: 实现分数归一化功能
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # 避免除零错误
            return [0.5] * len(scores)
        
        # 线性归一化到[0, 1]范围
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized_scores

# TODO: 实现更多重排序功能
# 1. 多阶段重排序
# 2. 自定义重排序特征
# 3. 多样化重排序
# 4. 交互式重排序反馈


# 创建全局实例
reranker = Reranker()


def rerank(q: Query, evidences: List[Evidence], topn: int = 10) -> List[Evidence]:
    """
    使用Cross-Encoder对检索结果进行精排
    
    Args:
        q: 查询对象
        evidences: 待重排序的证据列表
        topn: 返回的证据数量
    
    Returns:
        重排序后的证据列表
    """
    # 参数验证
    if not q or not evidences:
        return evidences[:topn]  # 返回前topn个（如果有的话）
    
    # 确保topn在合理范围内
    topn = max(1, min(topn, 100))  # 限制在1-100之间
    
    # 为降级路径做准备，先保存原始顺序
    original_evidences = evidences.copy()
    
    # 对Top-50做判别打分
    # 首先对输入的evidences进行筛选，最多取前50个
    rerank_candidates = evidences[:50]
    
    # 检查是否有足够的候选进行重排序
    if len(rerank_candidates) == 0:
        return []
    
    # 如果只有一个候选，直接返回
    if len(rerank_candidates) == 1:
        return rerank_candidates[:topn]
    
    try:
        # 尝试加载CrossEncoder模型
        # 使用预定义的模型名称
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # 设置最大处理时间为5秒（防止耗时过长）
        start_time = time.time()
        max_processing_time = 5.0  # 5秒
        
        # 创建CrossEncoder模型实例
        model = CrossEncoder(model_name)
        
        # 准备重排序输入（查询文本 + 证据内容）
        pairs = [(q.text, evidence.content) for evidence in rerank_candidates]
        
        # 计算相关性分数
        scores = model.predict(pairs)
        
        # 检查处理时间，如果超过最大时间则降级
        elapsed_time = time.time() - start_time
        if elapsed_time > max_processing_time:
            print(f"警告: 重排序耗时过长({elapsed_time:.2f}秒)，使用降级路径")
            return original_evidences[:topn]
        
        # 根据分数对证据进行排序
        scored_evidences = [(evidence, float(score)) for evidence, score in zip(rerank_candidates, scores)]
        scored_evidences.sort(key=lambda x: x[1], reverse=True)  # 分数越高越相关
        
        # 更新证据的分数和元数据
        reranked_evidences = []
        for i, (evidence, score) in enumerate(scored_evidences[:topn], 1):
            # 创建一个新的Evidence对象以避免修改原始数据
            reranked_evidence = Evidence(
                id=evidence.id,
                content=evidence.content,
                source=evidence.source,
                score=score,  # 更新为重排序分数
                relevance=score,  # 也设置到relevance字段
                metadata=evidence.metadata.copy() if evidence.metadata else {},
                start_pos=evidence.start_pos,
                end_pos=evidence.end_pos
            )
            
            # 在metadata中记录重排序信息
            reranked_evidence.metadata['reranked'] = True
            reranked_evidence.metadata['rerank_rank'] = i
            reranked_evidence.metadata['rerank_model'] = model_name
            
            # 保留原始检索的分数（如果有）
            if hasattr(evidence, 'original_score'):
                reranked_evidence.metadata['original_score'] = evidence.original_score
            elif evidence.score is not None:
                reranked_evidence.metadata['original_score'] = evidence.score
            
            reranked_evidences.append(reranked_evidence)
        
        return reranked_evidences
        
    except Exception as e:
        # 捕获所有异常，包括模型未安装、预测失败等
        print(f"重排序过程出错: {e}")
        # 降级路径：返回输入的前topn个证据
        return original_evidences[:topn]