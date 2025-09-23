#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回答生成模块
负责基于检索结果生成最终回答
"""

from typing import List, Dict, Any, Optional, Tuple
from app.models import Document, Evidence, Query
from app.config import config

class AnswerGenerator:
    """回答生成器"""
    
    def __init__(self):
        """初始化回答生成器"""
        # TODO: 初始化回答生成模型
        pass
    
    def generate_answer(self, query: str, documents: List[Document], **kwargs) -> Dict[str, Any]:
        """
        基于检索结果生成回答
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            **kwargs: 其他参数
        
        Returns:
            包含回答和相关信息的字典
        """
        # TODO: 实现回答生成功能
        # 1. 构建提示
        # 2. 调用生成模型
        # 3. 格式化回答
        
        # 当前返回一个简单的模板回答
        answer_text = self._generate_template_answer(query, documents)
        
        # 提取引用来源
        sources = self._extract_sources(documents)
        
        return {
            "answer": answer_text,
            "sources": sources,
            "metadata": {
                "generation_method": "template_based",
                "source_count": len(sources)
            }
        }
    
    def _generate_template_answer(self, query: str, documents: List[Document]) -> str:
        """
        生成模板式回答（临时实现）
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
        
        Returns:
            模板式回答文本
        """
        # TODO: 替换为实际的回答生成模型
        if not documents:
            return f"抱歉，我没有找到与 '{query}' 相关的信息。"
        
        # 简单地从文档中提取关键信息作为回答
        key_points = []
        for doc in documents[:3]:  # 只使用前3个文档
            # 提取文档的前50个字符作为关键点
            snippet = doc.text[:100].replace('\n', ' ').strip()
            if snippet:
                key_points.append(snippet)
        
        if not key_points:
            return f"关于 '{query}'，我找到了一些相关信息，但无法提取具体内容。"
        
        # 构建回答
        answer_parts = [
            f"针对问题 '{query}'，根据检索到的信息：\n"
        ]
        
        for i, point in enumerate(key_points, 1):
            answer_parts.append(f"{i}. {point}...\n")
        
        answer_parts.append("\n请注意，以上回答是基于检索到的信息生成的，仅供参考。")
        
        return ''.join(answer_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        从文档中提取引用来源
        
        Args:
            documents: 检索到的文档列表
        
        Returns:
            来源信息列表
        """
        sources = []
        for doc in documents:
            source_info = {
                "id": doc.id,
                "score": doc.score,
                "rank": doc.rank,
                "metadata": doc.metadata or {}
            }
            
            # 添加文档类型和来源路径（如果有）
            if "type" in doc.metadata:
                source_info["type"] = doc.metadata["type"]
            if "path" in doc.metadata:
                source_info["path"] = doc.metadata["path"]
            
            sources.append(source_info)
        
        return sources
    
    def _build_prompt(self, query: str, documents: List[Document]) -> str:
        """
        构建生成回答的提示
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
        
        Returns:
            提示文本
        """
        # TODO: 实现更复杂的提示构建逻辑
        prompt = f"问题: {query}\n"
        prompt += "请根据以下信息回答问题:\n"
        
        for i, doc in enumerate(documents, 1):
            prompt += f"信息 {i}: {doc.text[:200]}...\n"
        
        prompt += "\n回答:"
        
        return prompt

# TODO: 实现更多回答生成功能
# 1. 集成LLM模型
# 2. 多轮对话支持
# 3. 回答质量评估
# 4. 多语言回答生成


# 全局回答生成器实例
answer_generator = AnswerGenerator()


def compose(q: Query, evidences: List[Evidence]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    模板式生成答案（关键点列表 + 引文编号）
    
    Args:
        q: 查询对象
        evidences: 证据列表
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (生成的答案文本, 引用列表)
    """
    # 准备答案内容
    answer_parts = []
    citations = []
    
    # 添加问题作为开头
    answer_parts.append(f"针对问题：{q.text}\n\n")
    
    # 生成关键点列表和引文
    key_points = []
    for i, evidence in enumerate(evidences[:5], 1):  # 最多使用前5个证据
        # 提取关键点
        content = evidence.content or evidence.text or ""
        if not content:
            continue
            
        # 截取适当长度的内容作为关键点
        key_content = content[:200].replace('\n', ' ').strip()
        if len(content) > 200:
            key_content += "..."
            
        # 添加关键点，包含引用编号
        key_points.append(f"{i}. {key_content} [1]")
        
        # 添加引用信息
        citation = {
            "id": f"1",
            "evidence_id": evidence.id,
            "score": evidence.score,
            "source": evidence.source,
            "content": content[:500],  # 存储完整的内容片段
            "metadata": evidence.metadata or {}
        }
        citations.append(citation)
    
    # 构建答案
    if key_points:
        answer_parts.extend(key_points)
        answer_parts.append("\n\n综上所述，以上是对您问题的解答。")
    else:
        answer_parts.append("抱歉，没有找到相关信息来回答您的问题。")
    
    # 添加引用说明
    if citations:
        answer_parts.append("\n\n参考文献：")
        for i, citation in enumerate(citations, 1):
            source_info = citation["source"] or "未知来源"
            answer_parts.append(f"[{i}] 来源：{source_info}")
    
    return '\n'.join(answer_parts), citations