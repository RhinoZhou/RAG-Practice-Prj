#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试三个处理模块的功能：
1. preprocess.py - 文本归一化功能
2. rewrite.py - 查询重写功能
3. self_query.py - 自查询解析功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocess import normalize, preprocessor
from app.rewrite import rewrite, query_rewriter, generate_candidates, score_candidates
from app.self_query import extract_filters, apply_backoff, self_query_parser
from typing import Dict, Any, List, Tuple
import json


def test_preprocess_module():
    """测试文本预处理模块"""
    print("\n=== 测试文本预处理模块 ===")
    
    # 测试用例
    test_cases = [
        " 这是 一个  包含 多余空格的文本  ",
        "2023年的数据显示结果很好",
        "大约1000个用户参与了测试",
        "明天下午3点开会讨论项目进展"
    ]
    
    # 测试全局函数
    print("\n测试全局normalize函数:")
    for i, text in enumerate(test_cases):
        normalized = normalize(text)
        print(f"测试 {i+1}: 原始文本='{text}'")
        print(f"      归一化后='{normalized}'")
    
    # 测试类方法
    print("\n测试TextPreprocessor类的方法:")
    sample_text = " 这是一个   示例文本，包含多个空格和特殊字符！！"
    cleaned = preprocessor.clean_text(sample_text)
    print(f"clean_text: '{cleaned}'")
    
    # 测试分块功能
    long_text = "这是一段很长的文本，用于测试分块功能。" * 5
    chunks = preprocessor.chunk_text(long_text, chunk_size=50, overlap=10)
    print(f"分块结果: {len(chunks)}个块")
    for i, chunk in enumerate(chunks[:2]):  # 只显示前两个块
        print(f"块{i+1}: {chunk}")
    
    # 检查函数签名和注释
    print("\n函数签名和注释检查:")
    print(f"- normalize函数注释: {normalize.__doc__[:100]}...")
    


def test_rewrite_module():
    """测试查询重写模块"""
    print("\n=== 测试查询重写模块 ===")
    
    # 测试用例
    test_queries = [
        "如何提高系统性能？",
        "用户登录失败的常见原因",
        "数据备份的最佳实践"
    ]
    
    # 测试generate_candidates函数
    print("\n测试generate_candidates函数:")
    for query in test_queries:
        candidates = generate_candidates(query)
        print(f"查询: '{query}'")
        print(f"生成候选数量: {len(candidates)}")
        for i, candidate in enumerate(candidates[:3]):  # 只显示前3个候选
            print(f"  候选{i+1}: '{candidate}'")
    
    # 测试score_candidates函数
    print("\n测试score_candidates函数:")
    query = "如何提高系统性能？"
    candidates = generate_candidates(query)
    scored_candidates = score_candidates(query, candidates)
    print(f"查询: '{query}'")
    for i, (candidate, score) in enumerate(scored_candidates[:3]):  # 只显示前3个得分最高的候选
        print(f"  候选{i+1}: '{candidate}' (得分: {score:.2f})")
    
    # 测试全局rewrite函数
    print("\n测试全局rewrite函数:")
    query = "用户登录失败的常见原因"
    rewritten = rewrite(query)
    print(f"原始查询: '{query}'")
    print(f"重写结果: '{rewritten}'")
    
    # 检查函数签名和注释
    print("\n函数签名和注释检查:")
    print(f"- generate_candidates函数注释: {generate_candidates.__doc__[:100]}...")
    print(f"- score_candidates函数注释: {score_candidates.__doc__[:100]}...")
    


def test_self_query_module():
    """测试自查询解析模块"""
    print("\n=== 测试自查询解析模块 ===")
    
    # 测试用例
    test_queries = [
        "2023年华北地区的web平台订单处理情况",
        "华东地区移动APP上的待处理订单",
        "华南地区桌面客户端已完成的订单统计",
        "如何提高系统性能？"
    ]
    
    # 测试extract_filters函数
    print("\n测试extract_filters函数:")
    for query in test_queries:
        filters = extract_filters(query)
        print(f"查询: '{query}'")
        print(f"提取的过滤条件: {json.dumps(filters, ensure_ascii=False, indent=2)}")
    
    # 测试apply_backoff函数
    print("\n测试apply_backoff函数:")
    query = "2023年华北地区的web平台订单处理情况"
    filters = extract_filters(query)
    print(f"原始过滤条件: {json.dumps(filters, ensure_ascii=False)}")
    
    # 测试不同过滤条件数量下的退避策略
    test_filters_list = [
        {'platform': 'web', 'year': '2023', 'region': 'north_china', 'order_status': 'completed'},
        {'platform': 'web', 'year': '2023', 'region': None, 'order_status': None},
        {'platform': 'web', 'year': None, 'region': None, 'order_status': None},
        {'platform': None, 'year': None, 'region': None, 'order_status': None}
    ]
    
    for i, test_filters in enumerate(test_filters_list):
        adjusted_filters, backoff_level = apply_backoff(test_filters)
        print(f"\n测试 {i+1} - 过滤条件数量: {sum(1 for v in test_filters.values() if v is not None)}")
        print(f"  原始过滤条件: {json.dumps(test_filters, ensure_ascii=False)}")
        print(f"  调整后过滤条件: {json.dumps(adjusted_filters, ensure_ascii=False)}")
        print(f"  退避级别: {backoff_level}")
    
    # 测试完整的解析功能
    print("\n测试完整的parse_query功能:")
    query = "2023年华北地区的web平台订单处理情况"
    result = self_query_parser.parse_query(query)
    print(f"查询: '{query}'")
    print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 检查函数签名和注释
    print("\n函数签名和注释检查:")
    print(f"- extract_filters函数注释: {extract_filters.__doc__[:100]}...")
    print(f"- apply_backoff函数注释: {apply_backoff.__doc__[:100]}...")
    


def run_all_tests():
    """运行所有测试"""
    print("开始测试三个处理模块...")
    
    try:
        test_preprocess_module()
        print("\n✓ 文本预处理模块测试通过")
    except Exception as e:
        print(f"\n✗ 文本预处理模块测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_rewrite_module()
        print("\n✓ 查询重写模块测试通过")
    except Exception as e:
        print(f"\n✗ 查询重写模块测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_self_query_module()
        print("\n✓ 自查询解析模块测试通过")
    except Exception as e:
        print(f"\n✗ 自查询解析模块测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    run_all_tests()