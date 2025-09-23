#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索结果融合模块
负责合并不同检索策略的结果
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.models import Document, Evidence

class ResultFuser:
    """结果融合器"""
    
    def __init__(self):
        """初始化结果融合器"""
        # TODO: 加载融合配置
        pass
    
    def fuse_results(self, results_list: List[Dict[str, Any]], method: str = "rrf", top_k: int = 10) -> Dict[str, Any]:
        """
        融合多个检索结果
        
        Args:
            results_list: 检索结果列表
            method: 融合方法 (rrf, linear, weighted, etc.)
            top_k: 返回的文档数量
        
        Returns:
            融合后的结果
        """
        # TODO: 实现结果融合功能
        if not results_list:
            return {
                "documents": [],
                "metadata": {
                    "status": "error",
                    "message": "无结果可融合"
                }
            }
        
        # 获取查询文本（假设所有结果对应同一个查询）
        query = results_list[0].get("query", "")
        
        # 根据选择的方法执行融合
        if method == "rrf":
            fused_documents = self._rrf_fusion(results_list, top_k)
        elif method == "linear":
            fused_documents = self._linear_fusion(results_list, top_k)
        elif method == "weighted":
            fused_documents = self._weighted_fusion(results_list, top_k)
        else:
            # 默认使用RRF融合
            fused_documents = self._rrf_fusion(results_list, top_k)
        
        return {
            "query": query,
            "documents": fused_documents,
            "metadata": {
                "total_results": len(fused_documents),
                "fusion_method": method,
                "source_count": len(results_list),
                "top_k": top_k
            }
        }
    
    def _rrf_fusion(self, results_list: List[Dict[str, Any]], top_k: int = 10, k: int = 60) -> List[Document]:
        """
        使用Reciprocal Rank Fusion (RRF) 方法融合结果
        
        Args:
            results_list: 检索结果列表
            top_k: 返回的文档数量
            k: RRF参数
        
        Returns:
            融合后的文档列表
        """
        # TODO: 实现RRF融合算法
        doc_scores = {}
        doc_info = {}
        
        # 计算每个文档的RRF分数
        for results in results_list:
            documents = results.get("documents", [])
            for i, doc in enumerate(documents):
                doc_id = doc.id
                # RRF公式: 1/(k + rank)
                score = 1 / (k + (i + 1))
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_info[doc_id] = doc
                
                doc_scores[doc_id] += score
        
        # 按分数排序并返回top_k个文档
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 重建文档列表并更新rank和score
        result_docs = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_info[doc_id].copy()
            doc.rank = rank + 1
            doc.score = float(score)
            result_docs.append(doc)
        
        return result_docs
    
    def _linear_fusion(self, results_list: List[Dict[str, Any]], top_k: int = 10) -> List[Document]:
        """
        线性融合方法
        
        Args:
            results_list: 检索结果列表
            top_k: 返回的文档数量
        
        Returns:
            融合后的文档列表
        """
        # TODO: 实现线性融合算法
        doc_scores = {}
        doc_info = {}
        
        # 简单线性融合：对每个文档的分数取平均
        for results in results_list:
            documents = results.get("documents", [])
            for doc in documents:
                doc_id = doc.id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                    doc_info[doc_id] = doc
                
                doc_scores[doc_id].append(doc.score)
        
        # 计算平均分数
        avg_scores = {}
        for doc_id, scores in doc_scores.items():
            avg_scores[doc_id] = sum(scores) / len(scores)
        
        # 按分数排序并返回top_k个文档
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in avg_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 重建文档列表并更新rank和score
        result_docs = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_info[doc_id].copy()
            doc.rank = rank + 1
            doc.score = float(score)
            result_docs.append(doc)
        
        return result_docs
    
    def _weighted_fusion(self, results_list: List[Dict[str, Any]], top_k: int = 10, weights: Optional[List[float]] = None) -> List[Document]:
        """
        加权融合方法
        
        Args:
            results_list: 检索结果列表
            top_k: 返回的文档数量
            weights: 每个检索结果的权重
        
        Returns:
            融合后的文档列表
        """
        # TODO: 实现加权融合算法
        if weights is None:
            # 默认权重：每个检索结果权重相等
            weights = [1.0 / len(results_list)] * len(results_list)
        
        doc_scores = {}
        doc_info = {}
        
        # 加权融合：对每个文档的分数乘以权重后求和
        for i, results in enumerate(results_list):
            weight = weights[i]
            documents = results.get("documents", [])
            for doc in documents:
                doc_id = doc.id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_info[doc_id] = doc
                
                doc_scores[doc_id] += doc.score * weight
        
        # 按分数排序并返回top_k个文档
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 重建文档列表并更新rank和score
        result_docs = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_info[doc_id].copy()
            doc.rank = rank + 1
            doc.score = float(score)
            result_docs.append(doc)
        
        return result_docs

# TODO: 实现更多融合功能
# 1. 基于机器学习的融合
# 2. 自适应融合策略
# 3. 时间衰减融合
# 4. 多样性感知融合


def rrf_and_aggregate(results: Dict[str, List[Evidence]], k: int = 60) -> List[Evidence]:
    """
    使用Reciprocal Rank Fusion融合不同检索器的结果，并采用MaxSim原则合并同文档ID的片段
    
    Args:
        results: 包含不同检索器结果的字典，键为检索器名称，值为Evidence列表
        k: RRF融合的超参数，默认为60
    
    Returns:
        融合后的Evidence列表，按融合分数降序排列
    """
    if not results or all(len(evidence_list) == 0 for evidence_list in results.values()):
        return []
    
    # 记录每个文档ID的所有证据
    doc_evidences: Dict[str, List[Evidence]] = {}
    
    # 为每个检索器的结果计算RRF分数
    for searcher_name, evidence_list in results.items():
        for rank, evidence in enumerate(evidence_list, 1):
            # 计算RRF分数
            rrf_score = 1.0 / (rank + k - 1)
            
            # 将RRF分数添加到evidence中
            if not hasattr(evidence, 'rrf_scores'):
                evidence.rrf_scores = {}
            evidence.rrf_scores[searcher_name] = rrf_score
            
            # 保存到doc_evidences中
            doc_id = evidence.id
            if doc_id not in doc_evidences:
                doc_evidences[doc_id] = []
            doc_evidences[doc_id].append(evidence)
    
    # 根据MaxSim原则合并同文档ID的证据，并计算最终融合分数
    final_results = []
    for doc_id, evidences in doc_evidences.items():
        # 找到该文档中RRF分数最高的证据（MaxSim原则）
        max_rrf_evidence = None
        max_total_rrf = -1
        
        for evidence in evidences:
            # 计算总RRF分数（所有检索器的RRF分数之和）
            total_rrf = sum(evidence.rrf_scores.values())
            
            # 更新最高分证据
            if total_rrf > max_total_rrf:
                max_total_rrf = total_rrf
                max_rrf_evidence = evidence
        
        # 确保找到了最高分证据
        if max_rrf_evidence:
            # 更新evidence的总分数
            max_rrf_evidence.score = max_total_rrf
            
            # 将metadata中添加融合信息
            if not max_rrf_evidence.metadata:
                max_rrf_evidence.metadata = {}
            max_rrf_evidence.metadata['fused_from'] = list(max_rrf_evidence.rrf_scores.keys())
            max_rrf_evidence.metadata['fused_details'] = max_rrf_evidence.rrf_scores
            
            # 添加到最终结果列表
            final_results.append(max_rrf_evidence)
    
    # 按融合分数降序排序
    final_results.sort(key=lambda x: x.score, reverse=True)
    
    return final_results


# 创建全局实例
result_fuser = ResultFuser()