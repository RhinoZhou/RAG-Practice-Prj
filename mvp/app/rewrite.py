#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询重写模块
负责优化和扩展用户查询
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import math
from collections import Counter

class QueryRewriter:
    """查询重写类"""
    
    def __init__(self):
        """初始化查询重写器"""
        # 初始化查询重写模板
        self.rewrite_templates = [
            # 同义词替换模板
            lambda q: f"{q}",  # 原始查询
            lambda q: self._synonym_replace(q),  # 同义词替换
            lambda q: self._expand_with_related_terms(q),  # 相关词扩展
            lambda q: self._reorder_keywords(q),  # 关键词重排序
            lambda q: self._add_search_operators(q),  # 添加搜索操作符
            lambda q: self._simplify_query(q),  # 简化查询
            lambda q: self._expand_abbreviations(q),  # 展开缩写
        ]
        
        # 预定义的同义词词典
        self.synonym_dict = {
            "获取": ["查询", "查看", "检索"],
            "最新": ["最近", "最新版本", "最新的"],
            "政策": ["规定", "办法", "条例"],
            "流程": ["步骤", "过程", "程序"],
            "申请": ["提交", "请求", "办理"],
            "条件": ["要求", "资格", "标准"],
        }
        
        # 预定义的缩写词典
        self.abbreviation_dict = {
            "API": "应用程序编程接口",
            "SDK": "软件开发工具包",
            "FAQ": "常见问题解答",
            "SOP": "标准操作流程",
            "KPI": "关键绩效指标",
        }
    
    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        重写用户查询
        
        Args:
            query: 原始用户查询
        
        Returns:
            重写结果字典，包含原始查询、重写查询、扩展关键词等
        """
        # 生成候选查询
        candidates = self.generate_candidates(query, n=5)
        
        # 评分候选查询
        scored_candidates = self.score_candidates(candidates)
        
        # 选择最佳候选
        if scored_candidates:
            best_candidate = max(scored_candidates, key=lambda x: x['score'])
            rewritten_query = best_candidate['query']
        else:
            rewritten_query = query
        
        expanded_keywords = self._extract_keywords(query)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "expanded_keywords": expanded_keywords,
            "candidates": scored_candidates,
            "metadata": {
                "rewrite_method": "template_based",
                "confidence": self._calculate_confidence(query, rewritten_query)
            }
        }
    
    def generate_candidates(self, text: str, n: int = 5) -> List[str]:
        """
        基于模板生成3-5条查询候选
        
        Args:
            text: 原始查询文本
            n: 生成的候选数量
        
        Returns:
            查询候选列表
        """
        if not text:
            return []
        
        # 确保n在3-5之间
        n = max(3, min(5, n))
        
        candidates = set()  # 使用集合避免重复
        candidates.add(text)  # 始终包含原始查询
        
        # 生成其他候选
        for i, template in enumerate(self.rewrite_templates):
            try:
                candidate = template(text)
                if candidate and candidate != text:
                    candidates.add(candidate)
                # 如果已收集足够的候选，停止生成
                if len(candidates) >= n:
                    break
            except Exception as e:
                # 如果某个模板执行失败，继续尝试其他模板
                continue
        
        # 如果生成的候选不足n个，重复填充
        while len(candidates) < n:
            for candidate in list(candidates):
                # 尝试对现有候选进行进一步变换
                transformed = self._additional_transformation(candidate)
                if transformed and transformed not in candidates:
                    candidates.add(transformed)
                    if len(candidates) >= n:
                        break
            # 如果仍然不足，就用现有的候选填充
            if len(candidates) < n:
                break
        
        return list(candidates)[:n]  # 返回指定数量的候选
    
    def score_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """
        用简易BM25分数或启发式评分来评估候选查询
        
        Args:
            candidates: 查询候选列表
        
        Returns:
            带有评分的候选列表
        """
        if not candidates:
            return []
        
        scored_candidates = []
        
        # 计算每个候选的评分
        for candidate in candidates:
            # 计算简易BM25评分
            bm25_score = self._calculate_simple_bm25(candidate)
            
            # 计算启发式评分
            heuristic_score = self._calculate_heuristic_score(candidate)
            
            # 综合评分
            total_score = 0.7 * bm25_score + 0.3 * heuristic_score
            
            scored_candidates.append({
                'query': candidate,
                'score': total_score,
                'bm25_score': bm25_score,
                'heuristic_score': heuristic_score
            })
        
        # 排序候选
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_candidates
    
    def _synonym_replace(self, query: str) -> str:
        """\同义词替换"""
        for word, synonyms in self.synonym_dict.items():
            for synonym in synonyms:
                # 简单的同义词替换
                query = query.replace(word, synonym)
                break  # 每个词只替换一次
        return query
    
    def _expand_with_related_terms(self, query: str) -> str:
        """用相关词扩展查询"""
        # 这里使用简单的策略：在查询中添加相关词
        keywords = self._extract_keywords(query)
        if keywords:
            related_terms = []
            for keyword in keywords[:2]:  # 只处理前两个关键词
                if keyword in self.synonym_dict and self.synonym_dict[keyword]:
                    related_terms.append(self.synonym_dict[keyword][0])
            
            if related_terms:
                return f"{query} {' '.join(related_terms)}"
        return query
    
    def _reorder_keywords(self, query: str) -> str:
        """重排关键词顺序"""
        words = query.split()
        if len(words) > 1:
            # 简单的重排序策略：将长度较长的词放在前面
            words.sort(key=len, reverse=True)
            return ' '.join(words)
        return query
    
    def _add_search_operators(self, query: str) -> str:
        """添加搜索操作符"""
        # 简单的策略：为长查询添加引号表示精确匹配
        if len(query) > 15:
            return f'"{query}"'  # 添加引号表示精确匹配
        return query
    
    def _simplify_query(self, query: str) -> str:
        """简化查询，移除冗余词"""
        redundant_words = ['的', '了', '在', '是', '我', '你', '他', '她', '它']
        simplified = []
        for word in query.split():
            if word not in redundant_words:
                simplified.append(word)
        return ' '.join(simplified) if simplified else query
    
    def _expand_abbreviations(self, query: str) -> str:
        """展开缩写"""
        for abbr, full in self.abbreviation_dict.items():
            query = query.replace(abbr, full)
        return query
    
    def _additional_transformation(self, query: str) -> str:
        """额外的查询变换"""
        # 简单的变换：添加查询前缀
        prefixes = ['如何', '什么是', '怎样', '为什么']
        for prefix in prefixes:
            if not query.startswith(prefix):
                return f"{prefix}{query}"
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键词
        
        Args:
            query: 用户查询
        
        Returns:
            关键词列表
        """
        if not query:
            return []
        
        # 简单的关键词提取策略：移除停用词，保留有意义的词
        stop_words = ['的', '了', '在', '是', '我', '你', '他', '她', '它', '和', '或', '但', '而', '与']
        
        # 分词并过滤停用词
        words = [word for word in query.split() if word not in stop_words and len(word) > 1]
        
        # 如果分词结果为空，返回原始查询
        if not words:
            return [query]
        
        return words
    
    def _calculate_simple_bm25(self, query: str) -> float:
        """
        计算简易的BM25评分
        这里使用简化版本，基于词频和逆文档频率的近似计算
        """
        # 这里使用简化的BM25计算，假设文档集合是预定义的
        # 在实际应用中，应该使用真实的文档集合计算IDF
        
        # 假设的平均文档长度
        avg_doc_len = 100
        
        # BM25参数
        k1 = 1.5  # 控制词频饱和度
        b = 0.75  # 控制文档长度归一化的程度
        
        # 分词
        query_terms = self._extract_keywords(query)
        if not query_terms:
            return 0.0
        
        # 计算每个词的词频
        term_freq = Counter(query_terms)
        
        # 计算文档长度
        doc_len = len(query)
        
        # 计算BM25分数
        score = 0.0
        for term, freq in term_freq.items():
            # 简化的IDF计算，假设所有词的IDF相同
            idf = 1.0  # 实际应用中应该使用真实的IDF值
            
            # BM25公式
            term_score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_doc_len))
            score += term_score
        
        # 归一化分数
        return min(1.0, score / len(query_terms))
    
    def _calculate_heuristic_score(self, query: str) -> float:
        """
        计算启发式评分
        基于查询长度、关键词数量等特征
        """
        # 查询长度得分：适中长度的查询得分更高
        query_len = len(query)
        if 5 <= query_len <= 30:
            len_score = 1.0
        elif query_len < 5:
            len_score = 0.5
        else:
            len_score = 0.7
        
        # 关键词数量得分：关键词数量适中得分更高
        keywords = self._extract_keywords(query)
        keyword_count = len(keywords)
        if 2 <= keyword_count <= 5:
            keyword_score = 1.0
        elif keyword_count == 1:
            keyword_score = 0.6
        else:
            keyword_score = 0.8
        
        # 综合评分
        return 0.5 * len_score + 0.5 * keyword_score
    
    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        """\计算重写的置信度"""
        # 简单的置信度计算：基于原始查询和重写查询的相似性
        # 这里使用字符重叠率作为简单的度量
        original_chars = set(original)
        rewritten_chars = set(rewritten)
        overlap = len(original_chars.intersection(rewritten_chars))
        max_len = max(len(original_chars), len(rewritten_chars))
        
        if max_len == 0:
            return 0.0
        
        return overlap / max_len

# 全局实例，方便直接调用
rewriter = QueryRewriter()
query_rewriter = rewriter  # 同时提供别名以兼容测试脚本

def generate_candidates(text: str, n: int = 5) -> List[str]:
    """
    全局函数：基于模板生成3-5条查询候选
    
    Args:
        text: 原始查询文本
        n: 生成的候选数量
    
    Returns:
        查询候选列表
    """
    return rewriter.generate_candidates(text, n)

def rewrite(text: str) -> str:
    """
    全局函数：重写用户查询
    
    Args:
        text: 用户原始查询文本
    
    Returns:
        重写后的查询文本
    """
    return rewriter.rewrite_query(text)

def score_candidates(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    全局函数：用简易BM25分数或启发式评分来评估候选查询
    
    Args:
        query: 原始查询文本
        candidates: 查询候选列表
    
    Returns:
        带有评分的候选列表，格式为 [(候选查询, 分数), ...]
    """
    return rewriter.score_candidates(query, candidates)