#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块策略参数到指标映射器

功能：
- 对策略参数（chunk_size、overlap、top-k、窗口大小）进行实验
- 产出 F1/nDCG vs. 成本（存储增长率/延迟）的映射
- 生成简单决策树建议

主要内容：
- 参数网格评估
- 多任务"最佳窗口"推荐
- 表格/图像信息保留率抽样校验

流程：
1. 设定参数网格与业务任务标签（法律/客服/论文）
2. 运行评估并记录指标与成本
3. 基于规则生成"策略建议树"（例如：法律≥512，客服≈128）
4. 对多模态文档抽样计算图表保留率

输入：标注的问答集、多模态抽样、评估参数
输出：参数-效果映射表、建议树JSON、报告摘要（最佳配置与权衡说明）
"""

import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import f1_score, ndcg_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# -----------------------------
# 数据结构
# -----------------------------

@dataclass
class ChunkingParams:
    """分块策略参数"""
    chunk_size: int  # 分块大小（字符数或token数）
    overlap: int  # 重叠大小
    top_k: int  # 检索时返回的top-k个结果
    window_size: int = 0  # 滑动窗口大小，0表示不使用滑动窗口
    
    def __post_init__(self):
        # 确保参数有效性
        assert self.chunk_size > 0, "分块大小必须大于0"
        assert self.overlap >= 0, "重叠大小必须大于等于0"
        assert self.overlap < self.chunk_size, "重叠大小必须小于分块大小"
        assert self.top_k > 0, "top-k必须大于0"
        assert self.window_size >= 0, "窗口大小必须大于等于0"


@dataclass
class EvaluationMetrics:
    """评估指标"""
    f1: float  # F1分数
    ndcg: float  # nDCG分数
    storage_overhead: float  # 存储开销（增长率）
    latency_ms: float  # 延迟（毫秒）
    table_retention: float = 0.0  # 表格信息保留率
    image_retention: float = 0.0  # 图像信息保留率


@dataclass
class TaskType:
    """业务任务类型"""
    name: str  # 任务名称
    description: str  # 任务描述
    qa_samples: List[Dict[str, str]]  # 问答样本
    weight_f1: float = 0.5  # F1权重
    weight_ndcg: float = 0.3  # nDCG权重
    weight_storage: float = 0.1  # 存储权重
    weight_latency: float = 0.1  # 延迟权重
    multimodal_samples: List[Dict[str, Any]] = field(default_factory=list)  # 多模态样本


@dataclass
class EvaluationResult:
    """评估结果"""
    params: ChunkingParams  # 分块参数
    task: str  # 任务类型
    metrics: EvaluationMetrics  # 评估指标
    weighted_score: float = 0.0  # 加权得分


# -----------------------------
# 模拟评估函数
# -----------------------------

def simulate_chunking_evaluation(params: ChunkingParams, task: TaskType) -> EvaluationMetrics:
    """模拟分块策略评估，返回评估指标
    
    在实际应用中，这里应该实现真实的分块和检索评估逻辑
    
    参数:
        params: 分块策略参数，包括分块大小、重叠大小、top-k和窗口大小
        task: 任务类型，包括任务名称、问答样本和权重设置
        
    返回:
        EvaluationMetrics: 评估指标，包括F1、nDCG、存储开销、延迟等
    """
    # 模拟F1分数 - 分块大小与任务的匹配度
    # 法律文档：较大分块效果好；客服对话：较小分块效果好；学术论文：中等分块效果好
    base_f1 = 0.0
    if task.name == "法律文档":
        # 法律文档偏好较大分块
        base_f1 = min(0.95, 0.5 + params.chunk_size / 1000)
    elif task.name == "客服对话":
        # 客服对话偏好较小分块
        base_f1 = min(0.95, 0.9 - (params.chunk_size - 128) ** 2 / 20000)
    elif task.name == "学术论文":
        # 学术论文偏好中等分块
        base_f1 = min(0.95, 0.7 + (1 - abs(params.chunk_size - 512) / 512) * 0.25)
    
    # 重叠对F1的影响：适当重叠提高F1，过多重叠边际效益递减
    overlap_ratio = params.overlap / params.chunk_size
    if overlap_ratio <= 0.2:
        f1_overlap_factor = overlap_ratio * 0.15  # 重叠率在0-20%时，线性提升
    else:
        f1_overlap_factor = 0.03 + (overlap_ratio - 0.2) * 0.05  # 重叠率超过20%后，边际效益递减
    
    # top-k对F1的影响：top-k越大，召回率越高，但精确率可能下降
    f1_topk_factor = min(0.1, 0.02 * np.log(params.top_k))
    
    # 窗口大小对F1的影响：适当窗口提高上下文理解
    f1_window_factor = 0.0
    if params.window_size > 0:
        f1_window_factor = min(0.05, 0.01 * np.log(params.window_size))
    
    # 计算最终F1，加入随机波动
    f1 = min(0.98, base_f1 + f1_overlap_factor + f1_topk_factor + f1_window_factor)
    f1 = max(0.3, min(0.98, f1 + random.uniform(-0.03, 0.03)))  # 添加随机波动
    
    # 模拟nDCG分数 - 与F1相关但有所不同
    ndcg = max(0.3, min(0.98, f1 * 0.8 + random.uniform(0.0, 0.2)))
    
    # 模拟存储开销 - 主要受重叠率影响
    storage_overhead = 1.0 + overlap_ratio  # 1.0表示无重叠时的基准存储
    
    # 模拟延迟 - 受分块大小、top-k和窗口大小影响
    base_latency = 50  # 基础延迟50ms
    chunk_latency = params.chunk_size / 100  # 分块大小影响
    topk_latency = params.top_k * 2  # top-k影响
    window_latency = params.window_size / 10 if params.window_size > 0 else 0  # 窗口大小影响
    latency_ms = base_latency + chunk_latency + topk_latency + window_latency
    
    # 模拟表格和图像信息保留率
    # 表格保留率：分块越大，保留率越高
    table_retention = min(0.95, 0.3 + 0.7 * (params.chunk_size / 1000))
    # 图像保留率：分块越大，保留率越高，但受窗口大小影响更大
    image_retention = min(0.95, 0.2 + 0.4 * (params.chunk_size / 1000) + 0.4 * (params.window_size / 500 if params.window_size > 0 else 0))
    
    return EvaluationMetrics(
        f1=f1,
        ndcg=ndcg,
        storage_overhead=storage_overhead,
        latency_ms=latency_ms,
        table_retention=table_retention,
        image_retention=image_retention
    )


def calculate_weighted_score(metrics: EvaluationMetrics, task: TaskType) -> float:
    """计算加权得分
    
    根据任务类型的权重设置，计算评估指标的加权得分
    
    参数:
        metrics: 评估指标
        task: 任务类型，包含各指标的权重
        
    返回:
        float: 加权得分，范围为0-1，越高表示效果越好
    """
    # 归一化延迟（越低越好）
    norm_latency = max(0, 1 - metrics.latency_ms / 500)  # 假设500ms是可接受的最大延迟
    
    # 归一化存储开销（越低越好）
    norm_storage = max(0, 1 - (metrics.storage_overhead - 1) / 1)  # 假设最大接受2倍存储开销
    
    # 计算加权得分
    weighted_score = (
        task.weight_f1 * metrics.f1 +
        task.weight_ndcg * metrics.ndcg +
        task.weight_storage * norm_storage +
        task.weight_latency * norm_latency
    )
    
    # 如果有多模态样本，考虑表格和图像保留率
    if task.multimodal_samples:
        multimodal_weight = 0.2  # 多模态因素权重
        multimodal_score = (metrics.table_retention + metrics.image_retention) / 2
        # 调整权重
        adjusted_weight = 1 - multimodal_weight
        weighted_score = adjusted_weight * weighted_score + multimodal_weight * multimodal_score
    
    return weighted_score


# -----------------------------
# 参数网格评估
# -----------------------------

def generate_param_grid() -> List[ChunkingParams]:
    """生成参数网格
    
    生成所有可能的参数组合，用于网格搜索评估
    
    返回:
        List[ChunkingParams]: 参数组合列表
    """
    param_grid = []
    
    # 分块大小选项
    chunk_sizes = [128, 256, 512, 768, 1024]
    
    # 重叠率选项（占分块大小的百分比）
    overlap_ratios = [0.0, 0.1, 0.2, 0.3]
    
    # top-k选项
    top_ks = [3, 5, 10]
    
    # 窗口大小选项
    window_sizes = [0, 256, 512]
    
    # 生成所有参数组合
    for chunk_size in chunk_sizes:
        for overlap_ratio in overlap_ratios:
            overlap = int(chunk_size * overlap_ratio)
            for top_k in top_ks:
                for window_size in window_sizes:
                    param_grid.append(ChunkingParams(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        top_k=top_k,
                        window_size=window_size
                    ))
    
    return param_grid


def evaluate_param_grid(param_grid: List[ChunkingParams], tasks: List[TaskType]) -> Dict[str, List[EvaluationResult]]:
    """评估参数网格
    
    对每个参数组合在每个任务上进行评估
    
    参数:
        param_grid: 参数组合列表
        tasks: 任务类型列表
        
    返回:
        Dict[str, List[EvaluationResult]]: 按任务名称组织的评估结果
    """
    results = {task.name: [] for task in tasks}
    
    total_evals = len(param_grid) * len(tasks)
    print(f"开始评估 {total_evals} 个参数组合...")
    
    for i, params in enumerate(param_grid):
        for task in tasks:
            # 模拟评估
            metrics = simulate_chunking_evaluation(params, task)
            
            # 计算加权得分
            weighted_score = calculate_weighted_score(metrics, task)
            
            # 保存结果
            result = EvaluationResult(
                params=params,
                task=task.name,
                metrics=metrics,
                weighted_score=weighted_score
            )
            results[task.name].append(result)
        
        # 打印进度
        if (i + 1) % 10 == 0 or i == len(param_grid) - 1:
            progress = (i + 1) * len(tasks) / total_evals * 100
            print(f"评估进度: {progress:.1f}% ({(i + 1) * len(tasks)}/{total_evals})")
    
    return results


# -----------------------------
# 结果分析与可视化
# -----------------------------

def results_to_dataframe(results: Dict[str, List[EvaluationResult]]) -> pd.DataFrame:
    """将评估结果转换为DataFrame
    
    将评估结果转换为pandas DataFrame，便于后续分析和可视化
    
    参数:
        results: 按任务名称组织的评估结果
        
    返回:
        pd.DataFrame: 包含所有评估结果的数据框
    """
    data = []
    
    for task_name, task_results in results.items():
        for result in task_results:
            data.append({
                "任务": task_name,
                "分块大小": result.params.chunk_size,
                "重叠大小": result.params.overlap,
                "重叠率": result.params.overlap / result.params.chunk_size,
                "Top-K": result.params.top_k,
                "窗口大小": result.params.window_size,
                "F1": result.metrics.f1,
                "nDCG": result.metrics.ndcg,
                "存储开销": result.metrics.storage_overhead,
                "延迟(ms)": result.metrics.latency_ms,
                "表格保留率": result.metrics.table_retention,
                "图像保留率": result.metrics.image_retention,
                "加权得分": result.weighted_score
            })
    
    return pd.DataFrame(data)


def plot_metrics_vs_cost(df: pd.DataFrame, output_path: str = "results/params_metrics_cost.png"):
    """绘制指标与成本的关系图"""
    plt.figure(figsize=(16, 12))
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1 vs 存储开销
    sns.scatterplot(
        x="存储开销", 
        y="F1", 
        hue="任务",
        size="分块大小",
        sizes=(50, 200),
        alpha=0.7,
        data=df,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("F1 vs 存储开销")
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. nDCG vs 存储开销
    sns.scatterplot(
        x="存储开销", 
        y="nDCG", 
        hue="任务",
        size="分块大小",
        sizes=(50, 200),
        alpha=0.7,
        data=df,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("nDCG vs 存储开销")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. F1 vs 延迟
    sns.scatterplot(
        x="延迟(ms)", 
        y="F1", 
        hue="任务",
        size="分块大小",
        sizes=(50, 200),
        alpha=0.7,
        data=df,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("F1 vs 延迟")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 4. nDCG vs 延迟
    sns.scatterplot(
        x="延迟(ms)", 
        y="nDCG", 
        hue="任务",
        size="分块大小",
        sizes=(50, 200),
        alpha=0.7,
        data=df,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("nDCG vs 延迟")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"指标与成本关系图已保存至: {output_path}")


def plot_best_window_size(df: pd.DataFrame, output_path: str = "results/best_window_size.png"):
    """绘制最佳窗口大小图"""
    plt.figure(figsize=(12, 8))
    
    # 按任务和窗口大小分组，计算平均加权得分
    window_scores = df.groupby(["任务", "窗口大小"])["加权得分"].mean().reset_index()
    
    # 绘制条形图
    sns.barplot(x="任务", y="加权得分", hue="窗口大小", data=window_scores)
    
    plt.title("不同任务下各窗口大小的平均加权得分")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"最佳窗口大小图已保存至: {output_path}")


def plot_multimodal_retention(df: pd.DataFrame, output_path: str = "results/multimodal_retention.png"):
    """绘制多模态信息保留率图"""
    plt.figure(figsize=(14, 8))
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # 1. 表格保留率 vs 分块大小
    sns.boxplot(
        x="分块大小", 
        y="表格保留率", 
        data=df,
        ax=axes[0]
    )
    axes[0].set_title("表格保留率 vs 分块大小")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. 图像保留率 vs 分块大小
    sns.boxplot(
        x="分块大小", 
        y="图像保留率", 
        data=df,
        ax=axes[1]
    )
    axes[1].set_title("图像保留率 vs 分块大小")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"多模态信息保留率图已保存至: {output_path}")


# -----------------------------
# 策略建议树生成
# -----------------------------

def generate_strategy_tree(df: pd.DataFrame) -> Dict[str, Any]:
    """生成策略建议树
    
    基于评估结果生成决策树形式的策略建议
    
    参数:
        df: 包含评估结果的DataFrame
        
    返回:
        Dict[str, Any]: 策略建议树，以嵌套字典形式表示
    """
    # 找出每个任务的最佳参数组合
    best_params = {}
    for task in df["任务"].unique():
        task_df = df[df["任务"] == task]
        best_row = task_df.loc[task_df["加权得分"].idxmax()]
        
        best_params[task] = {
            "分块大小": int(best_row["分块大小"]),
            "重叠率": float(best_row["重叠率"]),
            "Top-K": int(best_row["Top-K"]),
            "窗口大小": int(best_row["窗口大小"]),
            "F1": float(best_row["F1"]),
            "nDCG": float(best_row["nDCG"]),
            "存储开销": float(best_row["存储开销"]),
            "延迟(ms)": float(best_row["延迟(ms)"]),
            "加权得分": float(best_row["加权得分"])
        }
    
    # 基于规则生成策略建议树
    strategy_tree = {
        "root": {
            "question": "文档类型是什么？",
            "options": [
                {
                    "answer": "法律文档",
                    "recommendation": {
                        "分块大小": best_params["法律文档"]["分块大小"],
                        "重叠率": best_params["法律文档"]["重叠率"],
                        "Top-K": best_params["法律文档"]["Top-K"],
                        "窗口大小": best_params["法律文档"]["窗口大小"],
                        "说明": f"法律文档通常需要较大的分块大小({best_params['法律文档']['分块大小']})以保持上下文完整性"
                    }
                },
                {
                    "answer": "客服对话",
                    "recommendation": {
                        "分块大小": best_params["客服对话"]["分块大小"],
                        "重叠率": best_params["客服对话"]["重叠率"],
                        "Top-K": best_params["客服对话"]["Top-K"],
                        "窗口大小": best_params["客服对话"]["窗口大小"],
                        "说明": f"客服对话通常需要较小的分块大小({best_params['客服对话']['分块大小']})以捕捉简短的问答对"
                    }
                },
                {
                    "answer": "学术论文",
                    "recommendation": {
                        "分块大小": best_params["学术论文"]["分块大小"],
                        "重叠率": best_params["学术论文"]["重叠率"],
                        "Top-K": best_params["学术论文"]["Top-K"],
                        "窗口大小": best_params["学术论文"]["窗口大小"],
                        "说明": f"学术论文通常需要中等的分块大小({best_params['学术论文']['分块大小']})以平衡概念完整性和检索精度"
                    }
                },
                {
                    "answer": "其他",
                    "question": "文档是否包含大量表格或图像？",
                    "options": [
                        {
                            "answer": "是",
                            "recommendation": {
                                "分块大小": 512,
                                "重叠率": 0.2,
                                "Top-K": 5,
                                "窗口大小": 512,
                                "说明": "包含表格或图像的文档需要较大的分块大小和窗口大小以保持信息完整性"
                            }
                        },
                        {
                            "answer": "否",
                            "question": "检索延迟要求是否严格？",
                            "options": [
                                {
                                    "answer": "是",
                                    "recommendation": {
                                        "分块大小": 256,
                                        "重叠率": 0.1,
                                        "Top-K": 3,
                                        "窗口大小": 0,
                                        "说明": "对延迟敏感的场景建议使用较小的分块大小、较低的重叠率和较小的Top-K值"
                                    }
                                },
                                {
                                    "answer": "否",
                                    "recommendation": {
                                        "分块大小": 512,
                                        "重叠率": 0.2,
                                        "Top-K": 5,
                                        "窗口大小": 256,
                                        "说明": "通用配置，平衡检索效果和系统开销"
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
    return strategy_tree


def generate_summary_report(df: pd.DataFrame, strategy_tree: Dict[str, Any]) -> Dict[str, Any]:
    """生成摘要报告
    
    基于评估结果和策略建议树生成摘要报告
    
    参数:
        df: 包含评估结果的DataFrame
        strategy_tree: 策略建议树
        
    返回:
        Dict[str, Any]: 摘要报告，包含最佳配置、平均指标、关键发现和权衡说明
    """
    # 找出每个任务的最佳参数组合
    best_params = {}
    for task in df["任务"].unique():
        task_df = df[df["任务"] == task]
        best_row = task_df.loc[task_df["加权得分"].idxmax()]
        
        best_params[task] = {
            "分块大小": int(best_row["分块大小"]),
            "重叠率": float(best_row["重叠率"]),
            "Top-K": int(best_row["Top-K"]),
            "窗口大小": int(best_row["窗口大小"]),
            "F1": float(best_row["F1"]),
            "nDCG": float(best_row["nDCG"]),
            "存储开销": float(best_row["存储开销"]),
            "延迟(ms)": float(best_row["延迟(ms)"]),
            "加权得分": float(best_row["加权得分"])
        }
    
    # 计算平均指标
    avg_metrics = {
        "F1": df["F1"].mean(),
        "nDCG": df["nDCG"].mean(),
        "存储开销": df["存储开销"].mean(),
        "延迟(ms)": df["延迟(ms)"].mean(),
    }
    
    # 生成摘要报告
    summary = {
        "最佳配置": best_params,
        "平均指标": avg_metrics,
        "关键发现": [
            f"法律文档最适合的分块大小为 {best_params['法律文档']['分块大小']}，F1达到 {best_params['法律文档']['F1']:.2f}",
            f"客服对话最适合的分块大小为 {best_params['客服对话']['分块大小']}，F1达到 {best_params['客服对话']['F1']:.2f}",
            f"学术论文最适合的分块大小为 {best_params['学术论文']['分块大小']}，F1达到 {best_params['学术论文']['F1']:.2f}",
            f"重叠率对F1的提升在 {df['重叠率'].min()} 到 0.2 之间最显著，超过0.2后边际效益递减",
            f"增加窗口大小可以提高表格和图像的信息保留率，但会增加延迟"
        ],
        "权衡说明": [
            "分块大小增加会提高信息完整性，但会增加延迟和降低精确匹配能力",
            "重叠率增加会提高召回率，但会增加存储开销和索引大小",
            "Top-K值增加会提高召回率，但会增加延迟和降低精确率",
            "窗口大小增加会提高上下文理解和多模态信息保留，但会增加延迟和内存消耗"
        ]
    }
    
    return summary


# -----------------------------
# 主函数
# -----------------------------

def main():
    # 设置随机种子，确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    print("=== 分块策略参数到指标映射器 ===")
    
    # 1. 创建示例问答集
    legal_qa = [
        {"question": "合同中的不可抗力条款包含哪些内容？", "answer": "不可抗力条款通常包括定义、通知义务、免责范围、合同中止或终止条件等内容。"}
        # 实际应用中应包含更多样本
    ]
    
    customer_qa = [
        {"question": "如何重置我的账户密码？", "answer": "您可以通过点击登录页面的'忘记密码'链接，然后按照邮件中的指示进行操作。"}
        # 实际应用中应包含更多样本
    ]
    
    academic_qa = [
        {"question": "深度学习在自然语言处理中的主要应用是什么？", "answer": "深度学习在NLP中的主要应用包括机器翻译、文本分类、命名实体识别、问答系统等。"}
        # 实际应用中应包含更多样本
    ]
    
    # 2. 创建多模态样本
    multimodal_samples = [
        {"type": "table", "content": "示例表格内容", "size": 500},
        {"type": "image", "content": "示例图像描述", "size": 300}
        # 实际应用中应包含更多样本
    ]
    
    # 3. 创建任务类型
    tasks = [
        TaskType(
            name="法律文档",
            description="法律合同、条款、判决书等",
            qa_samples=legal_qa,
            weight_f1=0.6,
            weight_ndcg=0.2,
            weight_storage=0.1,
            weight_latency=0.1,
            multimodal_samples=multimodal_samples
        ),
        TaskType(
            name="客服对话",
            description="客户服务对话记录",
            qa_samples=customer_qa,
            weight_f1=0.5,
            weight_ndcg=0.3,
            weight_storage=0.1,
            weight_latency=0.1
        ),
        TaskType(
            name="学术论文",
            description="学术研究论文",
            qa_samples=academic_qa,
            weight_f1=0.4,
            weight_ndcg=0.4,
            weight_storage=0.1,
            weight_latency=0.1,
            multimodal_samples=multimodal_samples
        )
    ]
    
    # 4. 生成参数网格
    print("\n生成参数网格...")
    param_grid = generate_param_grid()
    print(f"共生成 {len(param_grid)} 个参数组合")
    
    # 5. 评估参数网格
    print("\n开始评估参数网格...")
    results = evaluate_param_grid(param_grid, tasks)
    
    # 6. 转换为DataFrame
    print("\n转换结果为DataFrame...")
    results_df = results_to_dataframe(results)
    
    # 7. 创建结果目录
    import os
    os.makedirs("results", exist_ok=True)
    
    # 8. 保存结果到CSV
    results_df.to_csv("results/params_metrics_mapping.csv", index=False)
    print("参数-指标映射表已保存至: results/params_metrics_mapping.csv")
    
    # 9. 绘制指标与成本的关系图
    print("\n绘制指标与成本关系图...")
    plot_metrics_vs_cost(results_df)
    
    # 10. 绘制最佳窗口大小图
    print("\n绘制最佳窗口大小图...")
    plot_best_window_size(results_df)
    
    # 11. 绘制多模态信息保留率图
    print("\n绘制多模态信息保留率图...")
    plot_multimodal_retention(results_df)
    
    # 12. 生成策略建议树
    print("\n生成策略建议树...")
    strategy_tree = generate_strategy_tree(results_df)
    
    # 13. 保存策略建议树到JSON
    with open("results/strategy_recommendation_tree.json", "w", encoding="utf-8") as f:
        json.dump(strategy_tree, f, ensure_ascii=False, indent=2)
    print("策略建议树已保存至: results/strategy_recommendation_tree.json")
    
    # 14. 生成摘要报告
    print("\n生成摘要报告...")
    summary = generate_summary_report(results_df, strategy_tree)
    
    # 15. 保存摘要报告到JSON
    with open("results/summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("摘要报告已保存至: results/summary_report.json")
    
    # 16. 打印关键发现
    print("\n=== 关键发现 ===")
    for finding in summary["关键发现"]:
        print(f"- {finding}")
    
    print("\n=== 权衡说明 ===")
    for tradeoff in summary["权衡说明"]:
        print(f"- {tradeoff}")
    
    print("\n=== 分块策略参数到指标映射器执行完成 ===")


if __name__ == "__main__":
    main()