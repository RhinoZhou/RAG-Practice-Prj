#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端评估诊断优化复盘
作者：Ph.D. Rhino

功能说明：
本程序整合多个评估步骤，展示自动化"评估→诊断→优化→再评估"流程，帮助用户系统地提升RAG系统性能。
程序通过串联Precision/Recall评估、Chunking优化、Top-K调整及重排决策，生成优化前后的性能对比报告。

内容概述：
1. 模拟基础评估流程，计算关键指标（Recall、Precision、NDCG等）
2. 分析评估结果，识别系统瓶颈（如Chunking策略、检索参数等）
3. 执行优化策略（改进Chunking方法、调整动态Top-K参数、应用重排算法）
4. 再次评估优化后的系统性能
5. 生成详细的优化前后对比报告
6. 提供部署建议

执行流程：
1. 运行基础评估脚本并保存初始指标
2. 执行优化策略（Chunk改进+动态K调整+重排优化）
3. 再评估并比较差异
4. 生成before/after报告
5. 输出部署建议

输入说明：
- 程序自动生成模拟评估数据，无需手动输入
- 支持自定义关键参数，如chunk_size、top_k_range、rerank_threshold等

输出展示：
- 优化前后关键指标对比（Recall、NDCG等）
- 延迟变化分析
- 优化建议
- 详细的评估报告CSV文件
- 可视化图表展示优化效果
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def check_dependencies():
    """
    检查并自动安装必要的依赖包
    """
    # 先导入基础库
    import subprocess
    
    required_packages = ["pandas", "numpy", "matplotlib"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"正在安装缺失的依赖包：{', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("依赖包安装成功！")
        except Exception as e:
            print(f"依赖包安装失败：{e}")
            sys.exit(1)
    else:
        print("所有依赖包已安装完成。")


def generate_sample_evaluation_data(num_queries=100, num_docs=500, output_dir="."):
    """
    生成模拟的评估数据
    
    参数：
        num_queries: 查询数量
        num_docs: 文档数量
        output_dir: 输出目录
    
    返回：
        数据文件路径字典
    """
    print("正在生成模拟评估数据...")
    
    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    random.seed(42)
    
    # 生成查询数据
    queries = []
    for i in range(num_queries):
        query = {
            "query_id": f"query_{i+1}",
            "text": f"示例查询 #{i+1} - 关于RAG系统评估的问题",
            "topic": random.choice(["信息检索", "自然语言处理", "机器学习", "数据挖掘", "人工智能"])
        }
        queries.append(query)
    
    # 生成文档数据
    docs = []
    for i in range(num_docs):
        doc = {
            "doc_id": f"doc_{i+1}",
            "title": f"示例文档 #{i+1}",
            "content": f"这是文档 #{i+1} 的内容，包含与查询相关的信息...",
            "length": random.randint(500, 5000),
            "topic": random.choice(["信息检索", "自然语言处理", "机器学习", "数据挖掘", "人工智能"])
        }
        docs.append(doc)
    
    # 生成相关性判断数据（qrels）
    qrels = []
    for query in queries:
        # 每个查询随机有1-5个相关文档
        num_relevant = random.randint(1, 5)
        relevant_docs = random.sample(docs, num_relevant)
        
        for doc in relevant_docs:
            # 相关性评分：1-3（越高越相关）
            relevance = random.randint(1, 3)
            qrel = {
                "query_id": query["query_id"],
                "doc_id": doc["doc_id"],
                "relevance": relevance
            }
            qrels.append(qrel)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据到文件
    data_paths = {
        "queries": os.path.join(output_dir, "queries.json"),
        "docs": os.path.join(output_dir, "docs.json"),
        "qrels": os.path.join(output_dir, "qrels.csv")
    }
    
    with open(data_paths["queries"], "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    with open(data_paths["docs"], "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    
    pd.DataFrame(qrels).to_csv(data_paths["qrels"], index=False, encoding="utf-8-sig")
    
    print(f"已生成模拟评估数据，保存至：{output_dir}")
    print(f"- 查询数量：{num_queries}")
    print(f"- 文档数量：{num_docs}")
    print(f"- 相关性判断数量：{len(qrels)}")
    
    return data_paths


def simulate_basic_evaluation(data_paths, output_file="initial_evaluation.csv"):
    """
    模拟基础评估流程
    
    参数：
        data_paths: 数据文件路径字典
        output_file: 输出文件名
    
    返回：
        evaluation_results: 评估结果字典
    """
    print("\n执行基础评估...")
    
    # 读取相关性判断数据
    qrels_df = pd.read_csv(data_paths["qrels"])
    
    # 模拟检索结果
    # 对于每个查询，随机返回10个文档，其中部分是相关的
    query_ids = qrels_df["query_id"].unique()
    
    # 基础评估指标（模拟值）
    # 这里使用固定值和随机因素来模拟一个中等性能的RAG系统
    recall_basic = 0.68 + np.random.normal(0, 0.02)
    precision_basic = 0.55 + np.random.normal(0, 0.02)
    ndcg_basic = 0.72 + np.random.normal(0, 0.02)
    latency_basic = 350 + np.random.normal(0, 20)  # 毫秒
    
    # 确保指标在合理范围内
    recall_basic = max(0.5, min(0.8, recall_basic))
    precision_basic = max(0.4, min(0.7, precision_basic))
    ndcg_basic = max(0.6, min(0.85, ndcg_basic))
    latency_basic = max(300, min(400, latency_basic))
    
    # 记录详细指标
    evaluation_results = {
        "recall": recall_basic,
        "precision": precision_basic,
        "f1": 2 * (precision_basic * recall_basic) / (precision_basic + recall_basic) if (precision_basic + recall_basic) > 0 else 0,
        "ndcg": ndcg_basic,
        "latency": latency_basic,
        "num_queries": len(query_ids),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存评估结果
    pd.DataFrame([evaluation_results]).to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"基础评估完成，结果保存至：{output_file}")
    print(f"- Recall: {evaluation_results['recall']:.4f}")
    print(f"- Precision: {evaluation_results['precision']:.4f}")
    print(f"- F1 Score: {evaluation_results['f1']:.4f}")
    print(f"- NDCG: {evaluation_results['ndcg']:.4f}")
    print(f"- 平均延迟: {evaluation_results['latency']:.1f}ms")
    
    return evaluation_results


def diagnose_system_bottlenecks(evaluation_results, data_paths):
    """
    分析系统瓶颈
    
    参数：
        evaluation_results: 评估结果字典
        data_paths: 数据文件路径字典
    
    返回：
        bottlenecks: 系统瓶颈分析结果
        optimization_plan: 优化计划
    """
    print("\n分析系统瓶颈...")
    
    # 读取文档数据以分析chunk_size问题
    with open(data_paths["docs"], "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # 计算文档平均长度
    avg_doc_length = np.mean([doc["length"] for doc in docs])
    
    # 分析瓶颈
    bottlenecks = []
    
    # 1. Recall较低
    if evaluation_results["recall"] < 0.75:
        bottlenecks.append({
            "type": "recall",
            "severity": "high",
            "description": "Recall值偏低，可能是因为Chunking策略不合理，导致相关信息未被检索到",
            "suggestion": "优化Chunking策略，调整chunk_size和overlap参数"
        })
    
    # 2. Precision较低
    if evaluation_results["precision"] < 0.6:
        bottlenecks.append({
            "type": "precision",
            "severity": "medium",
            "description": "Precision值偏低，可能是因为检索策略不够精准，返回了较多无关文档",
            "suggestion": "优化检索模型或增加重排环节"
        })
    
    # 3. 文档长度分析
    if avg_doc_length > 2000:
        bottlenecks.append({
            "type": "chunking",
            "severity": "medium",
            "description": "文档平均长度较长，当前Chunking策略可能无法有效捕获关键信息",
            "suggestion": "采用动态Chunking策略，根据文档结构和内容调整chunk_size"
        })
    
    # 生成优化计划
    optimization_plan = {
        "steps": [],
        "expected_improvement": {}
    }
    
    # 1. Chunking优化
    optimization_plan["steps"].append({
        "name": "Chunking优化",
        "description": "将固定chunk_size改为动态chunk_size，根据文档长度和结构调整",
        "expected_effect": "提高Recall约8-12%"
    })
    optimization_plan["expected_improvement"]["recall"] = "+10%"
    
    # 2. 动态Top-K调整
    optimization_plan["steps"].append({
        "name": "动态Top-K调整",
        "description": "根据查询复杂度和历史性能动态调整Top-K参数",
        "expected_effect": "平衡Precision和Recall，提高整体F1分数"
    })
    optimization_plan["expected_improvement"]["f1"] = "+5%"
    
    # 3. 重排优化
    optimization_plan["steps"].append({
        "name": "重排优化",
        "description": "增加重排环节，使用更精确的相关性评分模型",
        "expected_effect": "提高NDCG约0.10-0.15"
    })
    optimization_plan["expected_improvement"]["ndcg"] = "+0.13"
    
    print(f"系统瓶颈分析完成，发现{len(bottlenecks)}个主要问题")
    for i, bottleneck in enumerate(bottlenecks):
        print(f"{i+1}. [{bottleneck['severity']}] {bottleneck['type']}: {bottleneck['description']}")
        print(f"   建议: {bottleneck['suggestion']}")
    
    print("\n优化计划：")
    for step in optimization_plan["steps"]:
        print(f"- {step['name']}: {step['description']}（预期效果：{step['expected_effect']}）")
    
    return bottlenecks, optimization_plan


def implement_optimizations(data_paths, optimization_plan, output_dir="."):
    """
    执行优化策略
    
    参数：
        data_paths: 数据文件路径字典
        optimization_plan: 优化计划
        output_dir: 输出目录
    
    返回：
        optimized_data_paths: 优化后的数据文件路径字典
    """
    print("\n执行优化策略...")
    
    # 创建优化后的数据文件路径
    optimized_data_paths = {
        "chunking_params": os.path.join(output_dir, "optimized_chunking_params.json"),
        "dynamic_k_policy": os.path.join(output_dir, "optimized_dynamic_k_policy.json"),
        "rerank_config": os.path.join(output_dir, "optimized_rerank_config.json")
    }
    
    # 1. 优化Chunking参数
    chunking_params = {
        "base_chunk_size": 500,
        "min_chunk_size": 300,
        "max_chunk_size": 1000,
        "overlap": 50,
        "dynamic_chunking": True,
        "use_semantic_chunking": True,
        "paragraph_detection": True
    }
    
    with open(optimized_data_paths["chunking_params"], "w", encoding="utf-8") as f:
        json.dump(chunking_params, f, ensure_ascii=False, indent=2)
    
    # 2. 优化动态Top-K策略
    dynamic_k_policy = {
        "default_k": 8,
        "min_k": 3,
        "max_k": 15,
        "query_complexity_thresholds": {
            "low": 5,
            "medium": 8,
            "high": 12
        },
        "use_feedback_loop": True,
        "adjustment_factor": 0.2
    }
    
    with open(optimized_data_paths["dynamic_k_policy"], "w", encoding="utf-8") as f:
        json.dump(dynamic_k_policy, f, ensure_ascii=False, indent=2)
    
    # 3. 优化重排配置
    rerank_config = {
        "enabled": True,
        "top_n_for_rerank": 20,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "threshold": 0.5,
        "use_multi_stage_rerank": True
    }
    
    with open(optimized_data_paths["rerank_config"], "w", encoding="utf-8") as f:
        json.dump(rerank_config, f, ensure_ascii=False, indent=2)
    
    print("优化策略执行完成，生成的配置文件：")
    print(f"- Chunking参数: {optimized_data_paths['chunking_params']}")
    print(f"- 动态Top-K策略: {optimized_data_paths['dynamic_k_policy']}")
    print(f"- 重排配置: {optimized_data_paths['rerank_config']}")
    
    return optimized_data_paths


def simulate_optimized_evaluation(initial_results, data_paths, output_file="optimized_evaluation.csv"):
    """
    模拟优化后的评估流程
    
    参数：
        initial_results: 初始评估结果
        data_paths: 数据文件路径字典
        output_file: 输出文件名
    
    返回：
        optimized_results: 优化后的评估结果字典
    """
    print("\n执行优化后的评估...")
    
    # 模拟优化后的指标提升
    # 基于初始结果和预期改进进行调整
    recall_optimized = initial_results["recall"] * 1.12 + np.random.normal(0, 0.01)  # 提升约12%
    precision_optimized = initial_results["precision"] * 1.05 + np.random.normal(0, 0.01)  # 提升约5%
    ndcg_optimized = initial_results["ndcg"] + 0.13 + np.random.normal(0, 0.01)  # 提升约0.13
    latency_optimized = initial_results["latency"] + 120 + np.random.normal(0, 10)  # 增加约120ms延迟
    
    # 确保指标在合理范围内
    recall_optimized = max(0.6, min(0.95, recall_optimized))
    precision_optimized = max(0.5, min(0.85, precision_optimized))
    ndcg_optimized = max(0.7, min(0.95, ndcg_optimized))
    latency_optimized = max(400, min(600, latency_optimized))
    
    # 记录优化后的评估结果
    optimized_results = {
        "recall": recall_optimized,
        "precision": precision_optimized,
        "f1": 2 * (precision_optimized * recall_optimized) / (precision_optimized + recall_optimized) if (precision_optimized + recall_optimized) > 0 else 0,
        "ndcg": ndcg_optimized,
        "latency": latency_optimized,
        "num_queries": initial_results["num_queries"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存优化后的评估结果
    pd.DataFrame([optimized_results]).to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"优化后评估完成，结果保存至：{output_file}")
    print(f"- Recall: {optimized_results['recall']:.4f}")
    print(f"- Precision: {optimized_results['precision']:.4f}")
    print(f"- F1 Score: {optimized_results['f1']:.4f}")
    print(f"- NDCG: {optimized_results['ndcg']:.4f}")
    print(f"- 平均延迟: {optimized_results['latency']:.1f}ms")
    
    return optimized_results


def generate_comparison_report(initial_results, optimized_results, output_file="evaluation_comparison.csv"):
    """
    生成优化前后对比报告
    
    参数：
        initial_results: 初始评估结果
        optimized_results: 优化后评估结果
        output_file: 输出文件名
    
    返回：
        comparison_df: 对比报告数据框
        recommendation: 部署建议
    """
    print("\n生成优化前后对比报告...")
    
    # 计算指标变化
    metrics = ["recall", "precision", "f1", "ndcg", "latency"]
    comparison_data = []
    
    for metric in metrics:
        initial_value = initial_results[metric]
        optimized_value = optimized_results[metric]
        change = optimized_value - initial_value
        
        # 计算变化百分比（延迟指标使用绝对值变化）
        if metric == "latency":
            change_pct = change
            change_str = f"+{change_pct:.1f}ms" if change_pct > 0 else f"{change_pct:.1f}ms"
        else:
            change_pct = (change / initial_value) * 100 if initial_value > 0 else 0
            change_str = f"+{change_pct:.1f}%" if change_pct > 0 else f"{change_pct:.1f}%"
        
        comparison_data.append({
            "指标": metric,
            "优化前": initial_value,
            "优化后": optimized_value,
            "绝对变化": change,
            "相对变化": change_str
        })
    
    # 创建对比报告数据框
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存对比报告
    comparison_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    # 生成部署建议
    # 基于核心指标的改进和延迟增加来判断
    # 1. 检查关键指标是否有显著提升
    recall_improvement = (optimized_results["recall"] - initial_results["recall"]) / initial_results["recall"] * 100
    ndcg_improvement = optimized_results["ndcg"] - initial_results["ndcg"]
    latency_increase = optimized_results["latency"] - initial_results["latency"]
    
    # 判断是否建议部署
    if recall_improvement > 8 and ndcg_improvement > 0.1 and latency_increase < 200:
        recommendation = "Recommend deploy"
        reason = f"性能显著提升（Recall +{recall_improvement:.1f}%, NDCG +{ndcg_improvement:.3f}），延迟增加可控（+{latency_increase:.1f}ms）"
    elif recall_improvement > 5 or ndcg_improvement > 0.05:
        recommendation = "Conditional deploy"
        reason = f"性能有一定提升（Recall +{recall_improvement:.1f}%, NDCG +{ndcg_improvement:.3f}），但延迟增加较多（+{latency_increase:.1f}ms），建议进一步优化性能"
    else:
        recommendation = "Do not deploy"
        reason = f"性能提升不明显（Recall +{recall_improvement:.1f}%, NDCG +{ndcg_improvement:.3f}），但延迟显著增加（+{latency_increase:.1f}ms）"
    
    print(f"对比报告生成完成，保存至：{output_file}")
    print("\n===== 优化前后关键指标对比 ====")
    print(f"Recall: {initial_results['recall']:.2f}→{optimized_results['recall']:.2f} ({recall_improvement:+.1f}%)")
    print(f"NDCG: {initial_results['ndcg']:.2f}→{optimized_results['ndcg']:.2f} ({ndcg_improvement:+.3f})")
    print(f"P95延迟: +{latency_increase:.1f}ms")
    print(f"Result: {recommendation} ({reason})")
    
    return comparison_df, (recommendation, reason)


def visualize_optimization_results(initial_results, optimized_results, output_file="optimization_results.png"):
    """
    可视化优化结果
    
    参数：
        initial_results: 初始评估结果
        optimized_results: 优化后评估结果
        output_file: 输出文件名
    """
    print("\n绘制优化结果可视化图表...")
    
    # 准备可视化数据
    metrics = ["recall", "precision", "f1", "ndcg"]
    initial_values = [initial_results[m] for m in metrics]
    optimized_values = [optimized_results[m] for m in metrics]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 绘制指标对比柱状图
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, initial_values, width, label='优化前')
    ax1.bar(x + width/2, optimized_values, width, label='优化后')
    
    ax1.set_xlabel('评估指标')
    ax1.set_ylabel('值')
    ax1.set_title('优化前后指标对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Recall', 'Precision', 'F1 Score', 'NDCG'])
    ax1.legend()
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(initial_values):
        ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(optimized_values):
        ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    # 2. 绘制延迟和指标提升对比
    latency_labels = ['优化前延迟', '优化后延迟']
    latency_values = [initial_results['latency'], optimized_results['latency']]
    
    ax2.bar(latency_labels, latency_values, color=['blue', 'green'])
    ax2.set_xlabel('延迟比较')
    ax2.set_ylabel('延迟 (ms)')
    ax2.set_title('优化前后延迟对比')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(latency_values):
        ax2.text(i, v + 10, f'{v:.1f}ms', ha='center')
    
    # 添加性能提升注释
    recall_improvement = (optimized_results['recall'] - initial_results['recall']) / initial_results['recall'] * 100
    ndcg_improvement = optimized_results['ndcg'] - initial_results['ndcg']
    
    improvement_text = f"性能提升：\n"
    improvement_text += f"Recall: +{recall_improvement:.1f}%\n"
    improvement_text += f"NDCG: +{ndcg_improvement:.3f}\n"
    improvement_text += f"延迟增加: +{(optimized_results['latency'] - initial_results['latency']):.1f}ms"
    
    ax2.text(0.5, 0.1, improvement_text, transform=ax2.transAxes, 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"可视化图表保存至：{output_file}")


def main():
    """
    主函数，执行端到端评估优化流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='端到端评估诊断优化复盘')
    parser.add_argument('--num_queries', type=int, default=100, help='生成的查询数量')
    parser.add_argument('--num_docs', type=int, default=500, help='生成的文档数量')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录')
    parser.add_argument('--generate_data', action='store_true', help='强制生成新数据')
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查依赖
    check_dependencies()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成或使用已有数据
    data_paths = generate_sample_evaluation_data(args.num_queries, args.num_docs, args.output_dir)
    
    # 1. 执行基础评估
    initial_results = simulate_basic_evaluation(data_paths, os.path.join(args.output_dir, "initial_evaluation.csv"))
    
    # 2. 分析系统瓶颈
    bottlenecks, optimization_plan = diagnose_system_bottlenecks(initial_results, data_paths)
    
    # 3. 执行优化策略
    optimized_configs = implement_optimizations(data_paths, optimization_plan, args.output_dir)
    
    # 4. 执行优化后的评估
    optimized_results = simulate_optimized_evaluation(
        initial_results, 
        {**data_paths, **optimized_configs}, 
        os.path.join(args.output_dir, "optimized_evaluation.csv")
    )
    
    # 5. 生成对比报告
    comparison_df, (recommendation, reason) = generate_comparison_report(
        initial_results, 
        optimized_results, 
        os.path.join(args.output_dir, "evaluation_comparison.csv")
    )
    
    # 6. 可视化优化结果
    visualize_optimization_results(
        initial_results, 
        optimized_results, 
        os.path.join(args.output_dir, "optimization_results.png")
    )
    
    # 记录结束时间并计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n===== 端到端评估优化流程完成 ====")
    print(f"总耗时：{execution_time:.3f}秒")
    print(f"部署建议：{recommendation}")
    print(f"建议理由：{reason}")
    
    # 检查执行效率
    if execution_time > 10:
        print("警告：程序执行时间较长，可能需要优化。")
        print("建议优化方向：1) 减少数据生成量；2) 优化评估模拟逻辑；3) 减少可视化复杂度。")
    else:
        print("程序执行效率良好。")


if __name__ == "__main__":
    main()