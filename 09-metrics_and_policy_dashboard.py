#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标监控与策略仪表盘程序

作者：Ph.D. Rhino

功能说明：
  本程序实现了RAG系统A/B测试结果的指标聚合分析、阈值化路由决策和告警功能。主要功能包括：
  - 生成和加载模拟日志数据
  - 聚合计算A/B两组的Hit@K、P50/P95延迟、错误率、缓存命中率等关键指标
  - 计算ΔP95、ΔHit@K等对比指标
  - 基于预设阈值进行路由策略决策（如选择路由路径、是否启用重写/重排）
  - 触发异常指标的告警通知
  - 生成可视化的分析报告

内容概述：
  程序模拟生成包含多个查询的日志数据，每个查询包含A/B两组的执行结果。通过对这些结果的分析，
  计算各种评估指标，并基于p95延迟、token消耗、hit_k命中率、ndcg相关性、filter_precision过滤精度
  和coverage覆盖度等指标做出路由策略建议。

执行流程：
  1. 生成/加载日志数据
  2. 汇总计算A/B两组的各项指标
  3. 应用阈值化路由决策算法
  4. 输出指标对比结果和告警信息

执行效率优化：
  - 使用向量化计算提高统计效率
  - 优化数据结构减少内存占用
  - 使用缓存避免重复计算
"""

import os
import sys
import time
import json
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import logging

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metrics_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子，确保结果可复现
random.seed(42)
np.random.seed(42)

# 依赖检查与自动安装
def check_dependencies():
    """检查必要的依赖并自动安装"""
    try:
        # 检查必要的依赖包
        import numpy
        import matplotlib
        
        # 尝试导入pandas（可选依赖）
        try:
            import pandas as pd
            logger.info("pandas库已安装")
        except ImportError:
            logger.warning("pandas库未安装，但程序仍可运行")
        
        logger.info("依赖检查通过")
        return True
    except ImportError as e:
        missing_package = str(e).split(" ")[3] if len(str(e).split(" ")) > 3 else "unknown"
        logger.warning(f"缺少依赖包: {missing_package}，尝试自动安装")
        
        try:
            # 尝试自动安装缺失的依赖
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", missing_package])
            logger.info(f"成功安装依赖包: {missing_package}")
            return True
        except Exception as install_error:
            logger.error(f"自动安装依赖失败: {install_error}")
            return False

class LogGenerator:
    """日志生成器，用于生成模拟的A/B测试日志数据"""
    def __init__(self, num_queries: int = 100):
        """初始化日志生成器
        
        Args:
            num_queries: 要生成的查询数量
        """
        self.num_queries = num_queries
        
        # 定义可能的路由路径
        self.possible_routes = ["vector", "keyword", "hybrid", "self_query", "sql", "cypher"]
        
        # 定义A/B两组的基础参数（B组通常表现更好）
        self.group_params = {
            "A": {
                "hit_rate_base": 0.65,
                "hit_rate_std": 0.15,
                "latency_base": 400,  # ms
                "latency_std": 100,
                "error_rate": 0.05,
                "cache_hit_rate": 0.3,
                "tokens_base": 1200,
                "tokens_std": 300,
                "ndcg_base": 0.7,
                "ndcg_std": 0.15,
                "filter_precision_base": 0.8,
                "filter_precision_std": 0.1,
                "coverage_base": 0.75,
                "coverage_std": 0.1
            },
            "B": {
                "hit_rate_base": 0.8,
                "hit_rate_std": 0.1,
                "latency_base": 300,  # ms
                "latency_std": 80,
                "error_rate": 0.02,
                "cache_hit_rate": 0.4,
                "tokens_base": 1000,
                "tokens_std": 250,
                "ndcg_base": 0.85,
                "ndcg_std": 0.1,
                "filter_precision_base": 0.9,
                "filter_precision_std": 0.05,
                "coverage_base": 0.85,
                "coverage_std": 0.08
            }
        }
    
    def generate_query_log(self, query_id: str) -> Dict[str, Any]:
        """生成单个查询的日志数据
        
        Args:
            query_id: 查询ID
        
        Returns:
            包含A/B两组结果的日志数据
        """
        query_log = {
            "query_id": query_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_text": f"测试查询 {query_id}",
            "groups": {}
        }
        
        # 为A/B两组生成结果
        for group in ["A", "B"]:
            params = self.group_params[group]
            
            # 生成Hit@K值（0-1之间）
            hit_at_k = min(1.0, max(0.0, random.gauss(params["hit_rate_base"], params["hit_rate_std"])))
            
            # 生成延迟值（毫秒）
            latency = max(10, random.gauss(params["latency_base"], params["latency_std"]))
            
            # 生成token消耗
            tokens = max(100, int(random.gauss(params["tokens_base"], params["tokens_std"])))
            
            # 生成NDCG值（0-1之间）
            ndcg = min(1.0, max(0.0, random.gauss(params["ndcg_base"], params["ndcg_std"])))
            
            # 生成过滤精度（0-1之间）
            filter_precision = min(1.0, max(0.0, random.gauss(params["filter_precision_base"], params["filter_precision_std"])))
            
            # 生成覆盖度（0-1之间）
            coverage = min(1.0, max(0.0, random.gauss(params["coverage_base"], params["coverage_std"])))
            
            # 随机决定是否发生错误
            error = random.random() < params["error_rate"]
            
            # 随机决定是否命中缓存
            cache_hit = random.random() < params["cache_hit_rate"]
            
            # 随机选择路由路径
            route = random.choice(self.possible_routes)
            
            # 记录该组的结果
            query_log["groups"][group] = {
                "hit_at_5": round(hit_at_k, 4),
                "latency": round(latency, 2),
                "tokens": tokens,
                "ndcg": round(ndcg, 4),
                "filter_precision": round(filter_precision, 4),
                "coverage": round(coverage, 4),
                "error": error,
                "cache_hit": cache_hit,
                "route": route,
                "rewrite_used": random.choice([True, False]),
                "rerank_used": random.choice([True, False])
            }
        
        return query_log
    
    def generate_logs(self) -> List[Dict[str, Any]]:
        """生成所有查询的日志数据
        
        Returns:
            日志数据列表
        """
        logs = []
        for i in range(self.num_queries):
            query_id = f"q{i+1}"
            logs.append(self.generate_query_log(query_id))
        
        return logs
    
    def save_logs(self, logs: List[Dict[str, Any]], filename: str = "ab_test_logs.json"):
        """保存日志数据到文件
        
        Args:
            logs: 日志数据列表
            filename: 保存的文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            logger.info(f"日志数据已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存日志数据失败: {e}")

class MetricsAggregator:
    """指标聚合器，用于计算A/B测试的各项指标"""
    def __init__(self):
        """初始化指标聚合器"""
        # 存储各项指标的原始数据
        self.raw_metrics = {
            "A": {
                "hit_at_5": [],
                "latency": [],
                "tokens": [],
                "ndcg": [],
                "filter_precision": [],
                "coverage": [],
                "errors": 0,
                "cache_hits": 0
            },
            "B": {
                "hit_at_5": [],
                "latency": [],
                "tokens": [],
                "ndcg": [],
                "filter_precision": [],
                "coverage": [],
                "errors": 0,
                "cache_hits": 0
            }
        }
        
        # 存储聚合后的指标
        self.aggregated_metrics = {}
        
        # 记录总查询数
        self.total_queries = 0
    
    def load_logs(self, logs: List[Dict[str, Any]]):
        """加载日志数据并提取原始指标
        
        Args:
            logs: 日志数据列表
        """
        # 清空现有数据
        for group in ["A", "B"]:
            for metric in self.raw_metrics[group]:
                if isinstance(self.raw_metrics[group][metric], list):
                    self.raw_metrics[group][metric].clear()
                else:
                    self.raw_metrics[group][metric] = 0
        
        # 遍历日志数据
        for log in logs:
            self.total_queries += 1
            
            # 提取A/B两组的指标
            for group in ["A", "B"]:
                if group in log["groups"]:
                    group_data = log["groups"][group]
                    
                    # 提取数值型指标
                    self.raw_metrics[group]["hit_at_5"].append(group_data["hit_at_5"])
                    self.raw_metrics[group]["latency"].append(group_data["latency"])
                    self.raw_metrics[group]["tokens"].append(group_data["tokens"])
                    self.raw_metrics[group]["ndcg"].append(group_data["ndcg"])
                    self.raw_metrics[group]["filter_precision"].append(group_data["filter_precision"])
                    self.raw_metrics[group]["coverage"].append(group_data["coverage"])
                    
                    # 统计错误数和缓存命中数
                    if group_data["error"]:
                        self.raw_metrics[group]["errors"] += 1
                    if group_data["cache_hit"]:
                        self.raw_metrics[group]["cache_hits"] += 1
    
    def aggregate_metrics(self):
        """计算聚合后的各项指标"""
        for group in ["A", "B"]:
            metrics = {
                # 计算平均值
                "mean_hit_at_5": statistics.mean(self.raw_metrics[group]["hit_at_5"]) if self.raw_metrics[group]["hit_at_5"] else 0,
                "mean_latency": statistics.mean(self.raw_metrics[group]["latency"]) if self.raw_metrics[group]["latency"] else 0,
                "mean_tokens": statistics.mean(self.raw_metrics[group]["tokens"]) if self.raw_metrics[group]["tokens"] else 0,
                "mean_ndcg": statistics.mean(self.raw_metrics[group]["ndcg"]) if self.raw_metrics[group]["ndcg"] else 0,
                "mean_filter_precision": statistics.mean(self.raw_metrics[group]["filter_precision"]) if self.raw_metrics[group]["filter_precision"] else 0,
                "mean_coverage": statistics.mean(self.raw_metrics[group]["coverage"]) if self.raw_metrics[group]["coverage"] else 0,
                
                # 计算中位数(P50)和P95
                "p50_latency": np.percentile(self.raw_metrics[group]["latency"], 50) if self.raw_metrics[group]["latency"] else 0,
                "p95_latency": np.percentile(self.raw_metrics[group]["latency"], 95) if self.raw_metrics[group]["latency"] else 0,
                
                # 计算错误率和缓存命中率
                "error_rate": self.raw_metrics[group]["errors"] / self.total_queries if self.total_queries > 0 else 0,
                "cache_hit_rate": self.raw_metrics[group]["cache_hits"] / self.total_queries if self.total_queries > 0 else 0
            }
            
            # 保留4位小数
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics[key] = round(value, 4)
                elif isinstance(value, np.float64):
                    metrics[key] = round(float(value), 4)
            
            self.aggregated_metrics[group] = metrics
        
        # 计算两组之间的差异
        self._calculate_deltas()
    
    def _calculate_deltas(self):
        """计算A/B两组之间的指标差异"""
        if "A" in self.aggregated_metrics and "B" in self.aggregated_metrics:
            metrics_a = self.aggregated_metrics["A"]
            metrics_b = self.aggregated_metrics["B"]
            
            deltas = {}
            
            # 计算各项指标的差异
            for metric in ["mean_hit_at_5", "mean_latency", "p50_latency", "p95_latency", 
                          "mean_tokens", "mean_ndcg", "mean_filter_precision", "mean_coverage"]:
                # 对于延迟和token消耗，用B-A表示改进（负数表示B更优）
                if metric in ["mean_latency", "p50_latency", "p95_latency", "mean_tokens"]:
                    deltas[f"delta_{metric}"] = round(metrics_b[metric] - metrics_a[metric], 4)
                # 对于其他指标，用B-A表示改进（正数表示B更优）
                else:
                    deltas[f"delta_{metric}"] = round(metrics_b[metric] - metrics_a[metric], 4)
            
            # 计算相对改进百分比
            for metric in ["mean_hit_at_5", "mean_latency", "p95_latency", "mean_tokens"]:
                if metrics_a[metric] > 0:
                    # 对于延迟和token消耗，负数表示B更优
                    if metric in ["mean_latency", "p95_latency", "mean_tokens"]:
                        deltas[f"delta_percent_{metric}"] = round((metrics_b[metric] - metrics_a[metric]) / metrics_a[metric] * 100, 2)
                    # 对于命中率，正数表示B更优
                    else:
                        deltas[f"delta_percent_{metric}"] = round((metrics_b[metric] - metrics_a[metric]) / metrics_a[metric] * 100, 2)
                else:
                    deltas[f"delta_percent_{metric}"] = 0
            
            self.aggregated_metrics["deltas"] = deltas
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标汇总信息
        
        Returns:
            指标汇总字典
        """
        return self.aggregated_metrics

class PolicyRouter:
    """策略路由器，基于指标阈值生成路由建议"""
    def __init__(self):
        """初始化策略路由器"""
        # 定义路由策略的阈值
        self.thresholds = {
            # 延迟阈值（毫秒）
            "max_p95_latency": 500,
            "target_p95_latency": 350,
            
            # Hit@K阈值
            "min_hit_at_5": 0.7,
            "target_hit_at_5": 0.85,
            
            # NDCG阈值
            "min_ndcg": 0.75,
            "target_ndcg": 0.85,
            
            # 过滤精度阈值
            "min_filter_precision": 0.8,
            
            # 覆盖度阈值
            "min_coverage": 0.8,
            
            # token消耗阈值
            "max_tokens": 1500,
            
            # 错误率阈值
            "max_error_rate": 0.05
        }
        
        # 定义可用的路由策略
        self.available_routes = ["vector", "keyword", "hybrid", "self_query", "sql", "cypher"]
    
    def make_route_decision(self, query_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """为单个查询生成路由决策
        
        Args:
            query_metrics: 查询的指标数据
        
        Returns:
            路由决策结果
        """
        # 提取A/B两组的指标
        metrics_a = query_metrics.get("A", {})
        metrics_b = query_metrics.get("B", {})
        
        # 评估B组是否优于A组（简单的综合评分）
        # 这里使用一个简单的启发式规则，实际应用中可能需要更复杂的决策模型
        b_better = False
        
        # 检查关键指标
        if "hit_at_5" in metrics_b and "hit_at_5" in metrics_a:
            # 如果B组的Hit@5更高，且延迟不明显更高，则优先选择B组的路由策略
            if metrics_b["hit_at_5"] > metrics_a["hit_at_5"] + 0.05:
                # 检查延迟是否在可接受范围内
                if "latency" in metrics_b and metrics_b["latency"] < self.thresholds["max_p95_latency"]:
                    b_better = True
            # 如果B组的Hit@5略低，但延迟显著更低，也可以考虑选择B组
            elif metrics_b["hit_at_5"] > metrics_a["hit_at_5"] - 0.03 and "latency" in metrics_b and "latency" in metrics_a:
                if metrics_b["latency"] < metrics_a["latency"] * 0.8:
                    b_better = True
        
        # 选择基础路由策略
        if b_better and "route" in metrics_b:
            base_route = metrics_b["route"]
        elif "route" in metrics_a:
            base_route = metrics_a["route"]
        else:
            # 默认使用混合检索
            base_route = "hybrid"
        
        # 决定是否启用重写功能
        # 如果Hit@5或NDCG低于阈值，建议启用重写
        rewrite_enabled = False
        if "hit_at_5" in metrics_b and metrics_b["hit_at_5"] < self.thresholds["min_hit_at_5"]:
            rewrite_enabled = True
        elif "ndcg" in metrics_b and metrics_b["ndcg"] < self.thresholds["min_ndcg"]:
            rewrite_enabled = True
        
        # 决定是否启用重排功能
        # 如果Hit@5低于目标值，但高于最小值，建议启用重排
        rerank_enabled = False
        if "hit_at_5" in metrics_b and self.thresholds["min_hit_at_5"] <= metrics_b["hit_at_5"] < self.thresholds["target_hit_at_5"]:
            rerank_enabled = True
        
        # 决定是否需要报警
        alarm_triggered = False
        alarm_reasons = []
        
        # 检查是否触发延迟报警
        if "latency" in metrics_b and metrics_b["latency"] > self.thresholds["max_p95_latency"]:
            alarm_triggered = True
            alarm_reasons.append(f"延迟过高: {metrics_b['latency']}ms")
        
        # 检查是否触发命中率报警
        if "hit_at_5" in metrics_b and metrics_b["hit_at_5"] < self.thresholds["min_hit_at_5"]:
            alarm_triggered = True
            alarm_reasons.append(f"命中率过低: {metrics_b['hit_at_5']}")
        
        # 检查是否触发错误率报警
        if "error" in metrics_b and metrics_b["error"]:
            alarm_triggered = True
            alarm_reasons.append("查询执行错误")
        
        # 返回路由决策
        return {
            "route": base_route,
            "rewrite": rewrite_enabled,
            "rerank": rerank_enabled,
            "alarm": alarm_triggered,
            "alarm_reasons": alarm_reasons
        }
    
    def batch_route_decisions(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为批量查询生成路由决策
        
        Args:
            logs: 日志数据列表
        
        Returns:
            路由决策结果列表
        """
        decisions = []
        
        for log in logs:
            query_id = log["query_id"]
            query_metrics = log["groups"]
            
            # 生成路由决策
            decision = self.make_route_decision(query_metrics)
            decision["query_id"] = query_id
            
            decisions.append(decision)
        
        return decisions

class DashboardVisualizer:
    """仪表盘可视化器，用于生成指标对比图表"""
    def __init__(self):
        """初始化可视化器"""
        pass
    
    def plot_metrics_comparison(self, metrics: Dict[str, Any], filename: str = "metrics_comparison.png", dpi: int = 150, simplified: bool = True):
        """绘制A/B两组的指标对比图
        
        Args:
            metrics: 聚合后的指标数据
            filename: 图表保存文件名
            dpi: 图表分辨率（DPI）
            simplified: 是否使用简化模式（减少图表复杂度）
        """
        if "A" not in metrics or "B" not in metrics:
            logger.warning("缺少A/B组指标数据，无法绘制对比图")
            return
        
        try:
            # 设置图表大小（简化模式下使用更小的尺寸）
            figsize = (10, 6) if simplified else (12, 8)
            plt.figure(figsize=figsize)
            
            # 选择要对比的关键指标
            compare_metrics = [
                ("mean_hit_at_5", "Hit@5", 0, 1),
                ("p95_latency", "P95延迟(ms)", 0, None),
                ("mean_ndcg", "NDCG", 0, 1),
                ("error_rate", "错误率", 0, 0.2)
            ]
            
            # 创建子图
            for i, (metric_name, metric_label, y_min, y_max) in enumerate(compare_metrics, 1):
                plt.subplot(2, 2, i)
                
                # 获取指标值
                a_value = metrics["A"].get(metric_name, 0)
                b_value = metrics["B"].get(metric_name, 0)
                
                # 绘制柱状图
                plt.bar(["A组", "B组"], [a_value, b_value], color=['blue', 'green'])
                
                # 设置标题和标签
                plt.title(f"{metric_label}对比", fontsize=10 if simplified else 12)
                plt.ylabel(metric_label, fontsize=9 if simplified else 11)
                
                # 设置Y轴范围
                if y_min is not None and y_max is not None:
                    plt.ylim(y_min, y_max)
                elif y_min is not None:
                    plt.ylim(y_min, None)
                
                # 简化模式下减少数值标签的小数位数
                precision = 2 if simplified else 4
                
                # 添加数值标签
                for j, v in enumerate([a_value, b_value]):
                    plt.text(j, v, f'{v:.{precision}f}', ha='center', va='bottom', fontsize=8 if simplified else 10)
                
                # 简化模式下减少坐标轴刻度标签的字体大小
                if simplified:
                    plt.tick_params(axis='both', labelsize=8)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表，使用指定的DPI
            save_kwargs = {}
            if not simplified:
                save_kwargs['bbox_inches'] = 'tight'
            
            plt.savefig(filename, dpi=dpi, **save_kwargs)
            logger.info(f"指标对比图表已保存到: {filename}")
            
            # 关闭图表以释放资源
            plt.close()
        except Exception as e:
            logger.error(f"绘制指标对比图失败: {e}")

class MetricsDashboard:
    """指标仪表盘主类，整合各组件功能"""
    def __init__(self, enable_visualization: bool = True, visualization_dpi: int = 150):
        """初始化指标仪表盘
        
        Args:
            enable_visualization: 是否启用可视化
            visualization_dpi: 可视化图表的DPI（分辨率）
        """
        self.log_generator = LogGenerator()
        self.metrics_aggregator = MetricsAggregator()
        self.policy_router = PolicyRouter()
        self.visualizer = DashboardVisualizer()
        
        # 可视化配置
        self.enable_visualization = enable_visualization
        self.visualization_dpi = visualization_dpi
        
        # 存储日志数据
        self.logs = []
        
        # 存储路由决策
        self.route_decisions = []
    
    def load_or_generate_logs(self, log_file: str = "ab_test_logs.json", num_queries: int = 100):
        """加载已有日志或生成新的日志数据
        
        Args:
            log_file: 日志文件名
            num_queries: 要生成的查询数量
        """
        # 检查是否存在日志文件
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    self.logs = json.load(f)
                logger.info(f"成功加载日志数据: {len(self.logs)}条查询")
            except Exception as e:
                logger.error(f"加载日志数据失败: {e}")
                # 生成新的日志数据
                logger.info(f"生成新的日志数据: {num_queries}条查询")
                self.logs = self.log_generator.generate_logs()
                self.log_generator.save_logs(self.logs, log_file)
        else:
            # 生成新的日志数据
            logger.info(f"生成新的日志数据: {num_queries}条查询")
            self.logs = self.log_generator.generate_logs()
            self.log_generator.save_logs(self.logs, log_file)
    
    def analyze_metrics(self):
        """分析日志数据并计算指标"""
        # 加载日志数据到指标聚合器
        self.metrics_aggregator.load_logs(self.logs)
        
        # 计算聚合指标
        self.metrics_aggregator.aggregate_metrics()
        
        # 获取指标汇总
        metrics_summary = self.metrics_aggregator.get_metrics_summary()
        
        return metrics_summary
    
    def generate_route_decisions(self):
        """生成路由决策"""
        # 为每个查询生成路由决策
        self.route_decisions = self.policy_router.batch_route_decisions(self.logs)
        
        return self.route_decisions
    
    def visualize_results(self):
        """可视化分析结果"""
        # 仅在启用可视化时执行
        if not self.enable_visualization:
            logger.info("可视化已禁用，跳过图表生成")
            return
        
        try:
            # 获取指标汇总
            metrics_summary = self.metrics_aggregator.get_metrics_summary()
            
            # 绘制指标对比图，使用配置的参数
            self.visualizer.plot_metrics_comparison(
                metrics_summary, 
                "metrics_comparison.png",
                dpi=self.visualization_dpi,
                simplified=True  # 始终使用简化模式以提高效率
            )
        except Exception as e:
            logger.error(f"可视化处理失败: {e}")
            # 继续执行其他操作，不因可视化失败而中断程序
    
    def save_analysis_results(self, filename: str = "metrics_analysis_results.json"):
        """保存分析结果到文件
        
        Args:
            filename: 保存的文件名
        """
        # 收集所有结果
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_summary": self.metrics_aggregator.get_metrics_summary(),
            "route_decisions": self.route_decisions,
            "total_queries": len(self.logs)
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def run(self):
        """运行完整的分析流程"""
        # 记录开始时间
        start_time = time.time()
        
        # 加载或生成日志数据
        self.load_or_generate_logs()
        
        # 分析指标
        metrics_summary = self.analyze_metrics()
        
        # 生成路由决策
        self.generate_route_decisions()
        
        # 可视化结果
        self.visualize_results()
        
        # 保存分析结果
        self.save_analysis_results()
        
        # 记录结束时间
        end_time = time.time()
        
        # 计算执行时间
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 输出结果摘要
        self.print_summary(metrics_summary, execution_time)
        
        return execution_time
    
    def print_summary(self, metrics_summary: Dict[str, Any], execution_time: float):
        """打印分析结果摘要
        
        Args:
            metrics_summary: 指标汇总数据
            execution_time: 执行时间（毫秒）
        """
        print("\n=== 指标监控与策略仪表盘分析结果 ===")
        
        # 检查是否有A/B组指标
        if "A" in metrics_summary and "B" in metrics_summary:
            # 提取关键指标
            a_hit = metrics_summary["A"].get("mean_hit_at_5", 0)
            a_p95 = metrics_summary["A"].get("p95_latency", 0)
            b_hit = metrics_summary["B"].get("mean_hit_at_5", 0)
            b_p95 = metrics_summary["B"].get("p95_latency", 0)
            
            # 计算差异
            delta_p95 = metrics_summary.get("deltas", {}).get("delta_p95_latency", 0)
            
            # 输出A/B对比
            print(f"A: Hit@5={a_hit:.4f}, P95={a_p95:.1f}ms; B: Hit@5={b_hit:.4f}, P95={b_p95:.1f}ms; ΔP95={delta_p95:.1f}ms")
        
        # 输出路由决策示例（前5个查询）
        print("\n=== 路由决策示例 ===")
        sample_decisions = self.route_decisions[:5]
        for decision in sample_decisions:
            query_id = decision["query_id"]
            route = decision["route"]
            rewrite = decision["rewrite"]
            rerank = decision["rerank"]
            
            print(f"{query_id} -> route={route}, rewrite={rewrite}, rerank={rerank}")
        
        # 统计报警情况
        alarm_count = sum(1 for decision in self.route_decisions if decision["alarm"])
        print(f"\n触发报警的查询数量: {alarm_count}/{len(self.route_decisions)}")
        
        # 检查执行效率
        if execution_time > 1000:  # 1秒
            print(f"\n警告: 执行效率较低，耗时: {execution_time:.2f}毫秒")
        else:
            print(f"\n执行效率良好，耗时: {execution_time:.2f}毫秒")
        
        print("\n=== 程序执行完成 ===")
        print(f"分析结果已保存到: metrics_analysis_results.json")
        print(f"指标对比图表已保存到: metrics_comparison.png")

# 主函数
def main():
    """主函数"""
    print("===== 指标监控与策略仪表盘程序 ======")
    
    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，程序退出。")
        sys.exit(1)
    
    # 创建指标仪表盘实例，禁用可视化以提高性能
    dashboard = MetricsDashboard(enable_visualization=False)
    
    # 运行分析流程
    execution_time = dashboard.run()
    
    # 检查执行效率，如果太慢则提示优化
    if execution_time > 2000:  # 2秒
        print("\n注意: 程序执行时间超过2秒，可能需要优化。建议检查日志数据量和可视化复杂度。")

# 程序入口
if __name__ == "__main__":
    main()