#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic数据模型模块
定义项目中使用的各种数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class Document(BaseModel):
    """文档数据模型"""
    id: str = Field(..., description="文档唯一标识符")
    text: str = Field(..., description="文档文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    score: Optional[float] = Field(default=None, description="文档相关性分数")
    rank: Optional[int] = Field(default=None, description="文档排名")

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询文本")
    top_k: Optional[int] = Field(default=10, description="返回的文档数量")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="过滤条件")
    use_rag: Optional[bool] = Field(default=True, description="是否使用RAG")

class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""
    queries: List[QueryRequest] = Field(..., description="查询请求列表")
    parallel: Optional[bool] = Field(default=False, description="是否并行处理")

class QueryResponse(BaseModel):
    """查询响应模型"""
    request_id: str = Field(..., description="请求ID")
    query: str = Field(..., description="用户查询文本")
    answer: Optional[str] = Field(default=None, description="生成的回答")
    documents: List[Document] = Field(default_factory=list, description="检索到的文档")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源")
    status: str = Field(..., description="处理状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")

class BatchQueryResponse(BaseModel):
    """批量查询响应模型"""
    batch_id: str = Field(..., description="批次ID")
    results: List[QueryResponse] = Field(..., description="查询结果列表")
    total_queries: int = Field(..., description="总查询数量")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    total_processing_time: float = Field(..., description="总处理时间(秒)")

class ConfigUpdateRequest(BaseModel):
    """配置更新请求模型"""
    config_key: str = Field(..., description="配置键")
    config_value: Any = Field(..., description="配置值")
    namespace: Optional[str] = Field(default="global", description="配置命名空间")

class LogEntry(BaseModel):
    """日志条目模型"""
    timestamp: datetime = Field(..., description="日志时间戳")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    source: str = Field(..., description="日志来源")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class Query(BaseModel):
    """查询模型"""
    text: str = Field(..., description="查询文本内容")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="过滤条件")
    rewrite_candidates: Optional[List[str]] = Field(default_factory=list, description="查询重写候选列表")
    top_k: Optional[int] = Field(default=10, description="返回结果数量")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")

class SearchParams(BaseModel):
    """搜索参数模型"""
    query: str = Field(..., description="搜索查询文本")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="过滤条件")
    top_k: Optional[int] = Field(default=10, description="返回结果数量")
    strategy: Optional[str] = Field(default="hybrid", description="检索策略")
    min_score: Optional[float] = Field(default=0.0, description="最小分数阈值")
    include_metadata: Optional[bool] = Field(default=True, description="是否包含元数据")

class Evidence(BaseModel):
    """证据模型"""
    id: str = Field(..., description="证据ID")
    content: str = Field(..., description="证据内容")
    source: str = Field(..., description="证据来源")
    score: Optional[float] = Field(default=None, description="证据分数")
    relevance: Optional[float] = Field(default=None, description="相关性评分")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")
    start_pos: Optional[int] = Field(default=None, description="在原文中的起始位置")
    end_pos: Optional[int] = Field(default=None, description="在原文中的结束位置")

class RouteDecision(BaseModel):
    """路由决策模型"""
    target_tools: List[str] = Field(default_factory=list, description="目标工具列表")
    destination_weights: Dict[str, float] = Field(default_factory=dict, description="目标权重映射")
    rules_applied: List[str] = Field(default_factory=list, description="应用的规则列表")
    confidence: Optional[float] = Field(default=None, description="决策置信度")
    strategy: Optional[str] = Field(default=None, description="选择的策略")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")

class PipelineResult(BaseModel):
    """流水线结果模型"""
    query: str = Field(..., description="原始查询")
    answer: Optional[str] = Field(default=None, description="生成的回答")
    evidences: List[Evidence] = Field(default_factory=list, description="支持证据列表")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="处理日志")
    timings_ms: Dict[str, float] = Field(default_factory=dict, description="各阶段处理时间(毫秒)")
    status: str = Field(..., description="处理状态")
    error: Optional[str] = Field(default=None, description="错误信息")
    route_decision: Optional[RouteDecision] = Field(default=None, description="路由决策结果")
    model_usage: Optional[Dict[str, float]] = Field(default=None, description="模型使用情况")