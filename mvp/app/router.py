#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询路由模块
负责根据查询特征选择合适的检索策略
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import yaml
from pydantic import Field
from typing import List, Dict, Any
from app.config import config
from app.models import Query, SearchParams, RouteDecision

class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    CROSS_ENCODER_ONLY = "cross_encoder_only"

class ExtendedRouteDecision(RouteDecision):
    """扩展的路由决策模型，添加了规则应用列表和Top-2合并目标"""
    rules_applied: List[str] = []
    top_k_destinations: List[Dict[str, Any]] = Field(default_factory=list)  # Top-2合并目标及权重

class QueryRouter:
    """查询路由器"""
    
    def __init__(self):
        """初始化查询路由器"""
        # 加载路由规则
        self.routing_rules = self._load_routing_rules()
        # 初始化检索器原型向量（简化实现，使用关键词映射）
        self.retriever_prototypes = {
            "bm25": ["关键词", "精确匹配", "短查询", "结构化"],
            "vector": ["语义理解", "概念匹配", "长查询", "上下文"]
        }
    
    def _load_routing_rules(self) -> Dict[str, Any]:
        """
        加载路由规则配置
        
        Returns:
            路由规则字典
        """
        try:
            with open(config.ROUTING_RULES_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # 如果无法加载配置文件，返回默认规则
            print(f"无法加载路由规则文件: {e}")
            return {
                "short_query_threshold": 10,
                "keyword_density_threshold": 0.3,
                "default_strategy": "hybrid",
                "policy_priority": False,
                "degrade_mode": False
            }
    
    def select_strategy(self, query: str, query_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        根据查询选择合适的检索策略
        
        Args:
            query: 用户查询
            query_features: 查询特征（可选）
        
        Returns:
            策略选择结果
        """
        # 简单的策略选择逻辑
        query_length = len(query)
        strategy = RetrievalStrategy.HYBRID.value  # 默认使用混合检索
        
        # 根据查询长度选择策略
        if query_length < self.routing_rules.get("short_query_threshold", 10):
            # 短查询更适合BM25
            strategy = RetrievalStrategy.BM25_ONLY.value
        elif query_length > 100:
            # 长查询更适合向量检索
            strategy = RetrievalStrategy.VECTOR_ONLY.value
        
        return {
            "strategy": strategy,
            "query_length": query_length,
            "features": query_features or {},
            "metadata": {
                "routing_method": "rule_based"
            }
        }
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        提取查询特征
        
        Args:
            query: 用户查询
        
        Returns:
            查询特征字典
        """
        return {
            "length": len(query),
            "word_count": len(query.split()),
            # 更多特征...
        }
    
    def _calculate_keyword_score(self, query: str, keywords: List[str]) -> float:
        """
        计算查询与关键词列表的匹配分数
        
        Args:
            query: 用户查询
            keywords: 关键词列表
        
        Returns:
            匹配分数 [0, 1]
        """
        if not keywords:
            return 0.0
            
        query_lower = query.lower()
        matched_count = sum(1 for keyword in keywords if keyword.lower() in query_lower)
        return matched_count / len(keywords)
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        归一化分数到 [0, 1] 范围
        
        Args:
            scores: 原始分数字典
        
        Returns:
            归一化后的分数字典
        """
        if not scores:
            return {}
            
        total = sum(scores.values())
        if total == 0:
            return {k: 0.0 for k in scores}
            
        return {k: v / total for k, v in scores.items()}
    
    def decide(self, q: Query, params: SearchParams, flags: dict) -> ExtendedRouteDecision:
        """
        路由决策核心函数
        
        Args:
            q: 查询对象
            params: 搜索参数
            flags: 标志字典
        
        Returns:
            扩展的路由决策对象
        """
        rules_applied = []
        scores = {}
        
        # 1. 语义候选：使用关键词打分
        for retriever, keywords in self.retriever_prototypes.items():
            scores[retriever] = self._calculate_keyword_score(q.text, keywords)
        
        # 2. 规则命中：根据filters/user_profile/flags调整权重
        # 检查policy_priority
        if (self.routing_rules.get("policy_priority", False) or 
            (q.filters and "platform" in q.filters and q.filters["platform"] == "policy")):
            # 如果开启了policy_priority或平台是policy，提高vector权重（因为policy领域配置为vector_only）
            if "vector" in scores:
                scores["vector"] = min(1.0, scores["vector"] * 1.5)
            rules_applied.append("policy_priority")
        
        # 检查degrade模式
        if flags.get("degrade", False) or self.routing_rules.get("degrade_mode", False):
            # 降级模式只使用bm25
            scores = {"bm25": 1.0}
            rules_applied.append("degrade_mode")
        
        # 简化MoE：归一化分数
        normalized_weights = self._normalize_scores(scores)
        
        # 生成Top-2合并目标及权重
        top_destinations = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)[:2]
        top_k_destinations = [
            {"name": name, "weight": float(weight), "rank": i+1}
            for i, (name, weight) in enumerate(top_destinations)
        ]
        
        # 确定selected_retrievers
        if flags.get("degrade", False) or self.routing_rules.get("degrade_mode", False):
            selected_retrievers = ["bm25"]
        else:
            selected_retrievers = list(normalized_weights.keys())
            # 默认至少包含["bm25", "vector"]
            if not selected_retrievers:
                selected_retrievers = ["bm25", "vector"]
            elif "bm25" not in selected_retrievers:
                selected_retrievers.append("bm25")
            elif "vector" not in selected_retrievers:
                selected_retrievers.append("vector")
        
        # 5. 返回ExtendedRouteDecision
        return ExtendedRouteDecision(
            target_tools=selected_retrievers,
            destination_weights=normalized_weights,
            rules_applied=rules_applied,
            top_k_destinations=top_k_destinations
        )

# 全局路由器实例
router = QueryRouter()

def decide(q: Query, params: SearchParams, flags: dict) -> RouteDecision:
    """
    全局路由决策函数，供外部调用
    
    Args:
        q: 查询对象
        params: 搜索参数
        flags: 标志字典
    
    Returns:
        路由决策对象
    """
    return router.decide(q, params, flags)

# 创建扩展的RouteDecision类，添加rules_applied字段
from pydantic import Field

# 修改decide方法返回ExtendedRouteDecision
# 重新定义QueryRouter.decide方法以返回正确的类型
def patch_decide_method():
    original_decide = QueryRouter.decide
    
    def patched_decide(self, q: Query, params: SearchParams, flags: dict) -> ExtendedRouteDecision:
        result = original_decide(self, q, params, flags)
        # 确保返回的是ExtendedRouteDecision对象
        if not isinstance(result, ExtendedRouteDecision):
            # 如果原始返回值是RouteDecision，转换为ExtendedRouteDecision
            if hasattr(result, "dict"):
                result_dict = result.dict()
                # 提取rules_applied字段（如果存在）
                rules_applied = result_dict.pop("rules_applied", [])
                return ExtendedRouteDecision(**result_dict, rules_applied=rules_applied)
            else:
                # 创建新的ExtendedRouteDecision对象
                return ExtendedRouteDecision(
                    target_tools=getattr(result, "target_tools", []),
                    destination_weights=getattr(result, "destination_weights", {}),
                    rules_applied=getattr(result, "rules_applied", [])
                )
        return result
    
    # 替换原始方法
    QueryRouter.decide = patched_decide

# 执行补丁
patch_decide_method()

# 更新全局decide函数的返回类型注解
def patched_global_decide(q: Query, params: SearchParams, flags: dict) -> ExtendedRouteDecision:
    return router.decide(q, params, flags)

decide = patched_global_decide