#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自查询解析模块
负责从用户查询中提取结构化查询条件
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

class SelfQueryParser:
    """自查询解析器"""
    
    def __init__(self):
        """初始化自查询解析器"""
        # 初始化正则表达式模式用于提取过滤条件
        self.platform_patterns = {
            'web': re.compile(r'(网页|Web|网站|浏览器)', re.IGNORECASE),
            'mobile_app': re.compile(r'(移动|手机|APP|应用)', re.IGNORECASE),
            'desktop_app': re.compile(r'(桌面|电脑|客户端)', re.IGNORECASE),
            'api': re.compile(r'(API|接口|编程)', re.IGNORECASE),
            'third_party': re.compile(r'(第三方|外部)', re.IGNORECASE),
        }
        
        self.year_pattern = re.compile(r'(20\d{2}|\d{4}年)')
        
        self.region_patterns = {
            'north_china': re.compile(r'(华北|北京|天津|河北|山西|内蒙古)', re.IGNORECASE),
            'east_china': re.compile(r'(华东|上海|江苏|浙江|安徽|福建|江西|山东)', re.IGNORECASE),
            'south_china': re.compile(r'(华南|广东|广西|海南)', re.IGNORECASE),
            'central_china': re.compile(r'(华中|河南|湖北|湖南)', re.IGNORECASE),
            'west_china': re.compile(r'(西北|西南|重庆|四川|贵州|云南|西藏|陕西|甘肃|青海|宁夏|新疆)', re.IGNORECASE),
            'northeast_china': re.compile(r'(东北|辽宁|吉林|黑龙江)', re.IGNORECASE),
            'overseas': re.compile(r'(海外|国际|国外|境外)', re.IGNORECASE),
        }
        
        self.order_status_patterns = {
            'pending': re.compile(r'(待处理|未处理|待审核|待审批)', re.IGNORECASE),
            'processing': re.compile(r'(处理中|审核中|进行中)', re.IGNORECASE),
            'completed': re.compile(r'(已完成|已处理|已审核|已审批)', re.IGNORECASE),
            'rejected': re.compile(r'(已拒绝|被拒绝|驳回)', re.IGNORECASE),
            'cancelled': re.compile(r'(已取消|取消|撤销)', re.IGNORECASE),
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        解析用户查询，提取结构化条件
        
        Args:
            query: 用户查询
        
        Returns:
            解析结果字典，包含查询文本和提取的约束条件
        """
        # 提取过滤条件
        filters = self.extract_filters(query)
        
        # 应用退避策略
        filters_with_backoff, backoff_level = self.apply_backoff(filters)
        
        # 构建解析结果
        parsed_query = self._clean_query_for_search(query, filters)
        
        return {
            "query": query,
            "constraints": filters_with_backoff,
            "parsed_query": parsed_query,
            "backoff_level": backoff_level,
            "metadata": {
                "parse_method": "rule_based",
                "has_constraints": len(filters_with_backoff) > 0
            }
        }
    
    def extract_filters(self, text: str, enums: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从文本中提取过滤条件
        
        Args:
            text: 用户查询文本
            enums: 枚举值定义，可选
        
        Returns:
            过滤条件字典，包含platform、year、region、order_status等字段，不确定即None
        """
        if not text:
            return {}
        
        filters = {
            'platform': None,
            'year': None,
            'region': None,
            'order_status': None,
        }
        
        # 提取平台信息
        filters['platform'] = self._extract_platform(text)
        
        # 提取年份信息
        filters['year'] = self._extract_year(text)
        
        # 提取地区信息
        filters['region'] = self._extract_region(text)
        
        # 提取订单状态信息
        filters['order_status'] = self._extract_order_status(text)
        
        # 如果提供了enums，验证提取的过滤条件是否在枚举值内
        if enums:
            filters = self._validate_filters_against_enums(filters, enums)
        
        return filters
    
    def apply_backoff(self, filters: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        应用退避策略，放宽或调整过滤条件
        
        Args:
            filters: 原始过滤条件字典
        
        Returns:
            (调整后的过滤条件字典, 退避级别{F1,F2,F3})
        """
        if not filters:
            return {}, 'F1'
        
        # 复制原始过滤条件以避免修改
        adjusted_filters = filters.copy()
        
        # 计算非None过滤条件的数量
        non_none_count = sum(1 for v in filters.values() if v is not None)
        
        # 根据过滤条件的数量和类型确定退避级别
        if non_none_count == 0:
            return adjusted_filters, 'F1'  # 没有过滤条件，使用F1级别的退避
        elif non_none_count == 1:
            # 只有一个过滤条件，使用F1级别的退避
            return self._apply_backoff_level(adjusted_filters, 'F1'), 'F1'
        elif non_none_count == 2:
            # 两个过滤条件，使用F2级别的退避
            return self._apply_backoff_level(adjusted_filters, 'F2'), 'F2'
        else:
            # 三个或更多过滤条件，使用F3级别的退避
            return self._apply_backoff_level(adjusted_filters, 'F3'), 'F3'
    
    def _extract_platform(self, text: str) -> Optional[str]:
        """
        从文本中提取平台信息
        
        Args:
            text: 用户查询文本
        
        Returns:
            平台标识，未识别到则返回None
        """
        for platform, pattern in self.platform_patterns.items():
            if pattern.search(text):
                return platform
        return None
    
    def _extract_year(self, text: str) -> Optional[str]:
        """
        从文本中提取年份信息
        
        Args:
            text: 用户查询文本
        
        Returns:
            年份字符串，未识别到则返回None
        """
        match = self.year_pattern.search(text)
        if match:
            year_str = match.group(1)
            # 处理格式如"2023年"的情况
            if year_str.endswith('年'):
                year_str = year_str[:-1]
            # 确保年份在有效范围内
            try:
                year = int(year_str)
                current_year = datetime.now().year
                if 1900 <= year <= current_year + 1:
                    return str(year)
            except ValueError:
                pass
        return None
    
    def _extract_region(self, text: str) -> Optional[str]:
        """
        从文本中提取地区信息
        
        Args:
            text: 用户查询文本
        
        Returns:
            地区标识，未识别到则返回None
        """
        for region, pattern in self.region_patterns.items():
            if pattern.search(text):
                return region
        return None
    
    def _extract_order_status(self, text: str) -> Optional[str]:
        """
        从文本中提取订单状态信息
        
        Args:
            text: 用户查询文本
        
        Returns:
            订单状态标识，未识别到则返回None
        """
        for status, pattern in self.order_status_patterns.items():
            if pattern.search(text):
                return status
        return None
    
    def _validate_filters_against_enums(self, filters: Dict[str, Any], enums: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据枚举值验证过滤条件
        
        Args:
            filters: 过滤条件字典
            enums: 枚举值定义
        
        Returns:
            验证后的过滤条件字典
        """
        validated_filters = filters.copy()
        
        # 验证平台
        if 'platforms' in enums and filters['platform']:
            platform_values = [item['value'] for item in enums['platforms']]
            if filters['platform'] not in platform_values:
                validated_filters['platform'] = None
        
        # 验证地区
        if 'regions' in enums and filters['region']:
            region_values = [item['value'] for item in enums['regions']]
            if filters['region'] not in region_values:
                validated_filters['region'] = None
        
        return validated_filters
    
    def _apply_backoff_level(self, filters: Dict[str, Any], level: str) -> Dict[str, Any]:
        """
        应用特定级别的退避策略
        
        Args:
            filters: 原始过滤条件字典
            level: 退避级别(F1,F2,F3)
        
        Returns:
            调整后的过滤条件字典
        """
        adjusted_filters = filters.copy()
        
        # F1级别：保留所有条件，但考虑更宽松的匹配
        if level == 'F1':
            # 对于F1级别，我们保留所有条件，但可能在后续处理中放宽匹配标准
            pass
        
        # F2级别：移除最不相关的条件
        elif level == 'F2':
            # 移除最不相关的条件（这里假设order_status是最不相关的）
            if 'order_status' in adjusted_filters:
                adjusted_filters['order_status'] = None
        
        # F3级别：移除两个最不相关的条件
        elif level == 'F3':
            # 移除两个最不相关的条件（这里假设order_status和platform是最不相关的）
            if 'order_status' in adjusted_filters:
                adjusted_filters['order_status'] = None
            if 'platform' in adjusted_filters:
                adjusted_filters['platform'] = None
        
        return adjusted_filters
    
    def _clean_query_for_search(self, query: str, filters: Dict[str, Any]) -> str:
        """
        清理查询文本，移除已提取的过滤条件相关词汇
        
        Args:
            query: 原始查询文本
            filters: 提取的过滤条件
        
        Returns:
            清理后的查询文本，适合用于搜索
        """
        if not query or not filters:
            return query
        
        cleaned_query = query
        
        # 移除平台相关词汇
        if filters.get('platform'):
            for pattern in self.platform_patterns.values():
                cleaned_query = pattern.sub('', cleaned_query)
        
        # 移除年份相关词汇
        if filters.get('year'):
            cleaned_query = self.year_pattern.sub('', cleaned_query)
        
        # 移除地区相关词汇
        if filters.get('region'):
            for pattern in self.region_patterns.values():
                cleaned_query = pattern.sub('', cleaned_query)
        
        # 移除订单状态相关词汇
        if filters.get('order_status'):
            for pattern in self.order_status_patterns.values():
                cleaned_query = pattern.sub('', cleaned_query)
        
        # 清理多余的空白字符
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query or query  # 如果清理后为空，返回原始查询
    
    def apply_constraints(self, documents: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        应用约束条件过滤文档
        
        Args:
            documents: 文档列表
            constraints: 约束条件字典
        
        Returns:
            过滤后的文档列表
        """
        if not documents or not constraints:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            # 检查文档是否满足所有非None的约束条件
            if self._doc_matches_constraints(doc, constraints):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _doc_matches_constraints(self, doc: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """
        检查文档是否满足约束条件
        
        Args:
            doc: 文档
            constraints: 约束条件字典
        
        Returns:
            文档是否满足约束条件
        """
        # 确保文档有metadata字段
        metadata = doc.get('metadata', {})
        
        # 检查每个约束条件
        for key, value in constraints.items():
            if value is None:
                continue  # 跳过为None的约束条件
            
            # 检查文档中是否有对应的字段
            doc_value = metadata.get(key)
            if doc_value != value:
                return False  # 不满足约束条件
        
        return True  # 满足所有非None的约束条件

# 全局实例，方便直接调用
self_query_parser = SelfQueryParser()

def extract_filters(text: str, enums: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    全局函数：从文本中提取过滤条件
    
    Args:
        text: 用户查询文本
        enums: 枚举值定义，可选
    
    Returns:
        过滤条件字典，包含platform、year、region、order_status等字段，不确定即None
    """
    return self_query_parser.extract_filters(text, enums)

def apply_backoff(filters: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    全局函数：应用退避策略，放宽或调整过滤条件
    
    Args:
        filters: 原始过滤条件字典
    
    Returns:
        (调整后的过滤条件字典, 退避级别{F1,F2,F3})
    """
    return self_query_parser.apply_backoff(filters)