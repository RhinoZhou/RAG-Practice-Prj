#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示指标与 A/B 对比

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2024-01-20

功能说明:
用两套滑窗参数对比 P@5、Hit@1、冗余率与 p95 延迟。

内容概述:
构造 A/B 两套"窗口/重叠"配置，运行最小评估流程，计算并打印核心指标与推荐结论；
课堂可快速观察参数对准确性与冗余/时延的影响。

使用场景:
- RAG系统参数调优演示
- 课堂教学中展示不同参数配置对检索性能的影响
- 快速评估滑窗参数对检索质量和效率的影响

依赖库:
- numpy: 用于数值计算和统计分析
- pandas: 用于数据处理和表格展示
- scipy: 用于统计函数计算
"""

# 自动安装依赖库
import subprocess
import sys
import time
import json
import random

# 定义所需依赖库
required_dependencies = [
    'numpy',
    'pandas',
    'scipy'
]

def install_dependencies():
    """检查并自动安装缺失的依赖库"""
    for dependency in required_dependencies:
        try:
            # 尝试导入库以检查是否已安装
            __import__(dependency)
            print(f"✅ 依赖库 '{dependency}' 已安装")
        except ImportError:
            print(f"⚠️ 依赖库 '{dependency}' 未安装，正在安装...")
            # 使用pip安装缺失的依赖
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
            print(f"✅ 依赖库 '{dependency}' 安装成功")

# 执行依赖安装
if __name__ == "__main__":
    install_dependencies()

# 导入所需库
import numpy as np
import pandas as pd
from scipy import stats

class ABEvalMetricsPrinter:
    """A/B对比评估指标打印器"""
    
    def __init__(self):
        """初始化A/B对比评估指标打印器"""
        # 定义A/B两套滑窗参数配置
        self.config_a = {
            'name': 'A(win=128,ov=0.3)',
            'window_size': 128,
            'overlap': 0.3
        }
        self.config_b = {
            'name': 'B(win=256,ov=0.5)',
            'window_size': 256,
            'overlap': 0.5
        }
        
        # 初始化评估数据
        self.sample_queries = self._load_sample_queries()
        
        # 设置随机种子，保证结果可复现
        random.seed(42)
        np.random.seed(42)
    
    def _load_sample_queries(self):
        """加载示例查询和金标准数据"""
        # 模拟的查询和金标准数据
        # 在实际应用中，这里可以从文件或数据库中加载真实的查询和金标准
        sample_queries = [
            {
                'query': '混合检索的特点是什么？',
                'gold_relevant_docs': ['docA_2.2', 'docA_3.1']
            },
            {
                'query': '文本分块的作用是什么？',
                'gold_relevant_docs': ['docA_3.1', 'docA_3.2']
            },
            {
                'query': '窗口滑动策略的优势是什么？',
                'gold_relevant_docs': ['docA_3.2']
            },
            {
                'query': 'TF-IDF的原理是什么？',
                'gold_relevant_docs': ['docA_2.2']
            },
            {
                'query': 'RAG系统的核心组件有哪些？',
                'gold_relevant_docs': ['docA_2.2', 'docA_3.1', 'docA_3.2']
            }
        ]
        return sample_queries
    
    def _simulate_retrieval_results(self, config):
        """
        模拟检索结果
        
        参数:
            config: 滑窗参数配置
        
        返回:
            list: 每个查询的检索结果和性能指标
        """
        results = []
        
        for query_info in self.sample_queries:
            query = query_info['query']
            gold_docs = query_info['gold_relevant_docs']
            
            # 根据配置参数调整模拟的检索结果
            # 窗口大小越大，召回率可能越高，但冗余也可能增加
            # 重叠度越高，召回率可能越高，但冗余和延迟也可能增加
            base_precision = 0.5
            base_hit_rate = 0.4
            base_redundancy = 0.2
            base_latency = 80
            
            # 根据窗口大小和重叠度调整性能指标
            # 窗口越大，精度和召回率可能略高，但延迟和冗余也会增加
            precision_factor = 1.0 + (config['window_size'] / 512) * 0.2
            hit_rate_factor = 1.0 + (config['window_size'] / 512) * 0.2
            redundancy_factor = 1.0 + (config['window_size'] / 512) * 0.4 + (config['overlap'] * 0.5)
            latency_factor = 1.0 + (config['window_size'] / 512) * 0.8 + (config['overlap'] * 0.3)
            
            # 添加一些随机波动
            precision = base_precision * precision_factor + random.uniform(-0.05, 0.05)
            hit_rate = base_hit_rate * hit_rate_factor + random.uniform(-0.05, 0.05)
            redundancy = base_redundancy * redundancy_factor + random.uniform(-0.05, 0.05)
            latency = base_latency * latency_factor + random.uniform(-10, 10)
            
            # 确保指标在合理范围内
            precision = max(0.1, min(0.9, precision))
            hit_rate = max(0.1, min(0.9, hit_rate))
            redundancy = max(0.05, min(0.5, redundancy))
            latency = max(50, min(200, latency))
            
            # 记录查询的结果
            results.append({
                'query': query,
                'precision_at_5': precision,
                'hit_at_1': hit_rate,
                'redundancy': redundancy,
                'latency': latency
            })
        
        return results
    
    def _calculate_metrics(self, results):
        """
        计算总体评估指标
        
        参数:
            results: 检索结果列表
        
        返回:
            dict: 总体评估指标
        """
        # 提取各个指标
        precision_at_5_scores = [r['precision_at_5'] for r in results]
        hit_at_1_scores = [r['hit_at_1'] for r in results]
        redundancy_scores = [r['redundancy'] for r in results]
        latency_scores = [r['latency'] for r in results]
        
        # 计算平均指标和p95延迟
        metrics = {
            'precision_at_5': np.mean(precision_at_5_scores),
            'hit_at_1': np.mean(hit_at_1_scores),
            'redundancy': np.mean(redundancy_scores),
            'p95_latency': stats.scoreatpercentile(latency_scores, 95)
        }
        
        return metrics
    
    def _generate_recommendation(self, metrics_a, metrics_b):
        """
        根据评估指标生成推荐方案
        
        参数:
            metrics_a: 方案A的评估指标
            metrics_b: 方案B的评估指标
        
        返回:
            str: 推荐方案
        """
        # 定义各指标的权重
        weights = {
            'precision_at_5': 0.3,
            'hit_at_1': 0.3,
            'redundancy': -0.2,  # 冗余率越低越好，所以权重为负
            'p95_latency': -0.2   # 延迟越低越好，所以权重为负
        }
        
        # 计算加权得分
        score_a = (
            metrics_a['precision_at_5'] * weights['precision_at_5'] +
            metrics_a['hit_at_1'] * weights['hit_at_1'] +
            metrics_a['redundancy'] * weights['redundancy'] +
            metrics_a['p95_latency'] * weights['p95_latency'] / 100  # 归一化延迟
        )
        
        score_b = (
            metrics_b['precision_at_5'] * weights['precision_at_5'] +
            metrics_b['hit_at_1'] * weights['hit_at_1'] +
            metrics_b['redundancy'] * weights['redundancy'] +
            metrics_b['p95_latency'] * weights['p95_latency'] / 100  # 归一化延迟
        )
        
        # 生成推荐结论
        if score_a > score_b:
            recommendation = "方案 A（冗余低、时延更优）"
        else:
            recommendation = "方案 B（精度和召回率更优）"
        
        return recommendation
    
    def _format_metrics_table(self, metrics_a, metrics_b, recommendation):
        """
        格式化指标对比表
        
        参数:
            metrics_a: 方案A的评估指标
            metrics_b: 方案B的评估指标
            recommendation: 推荐方案
        
        返回:
            str: 格式化的Markdown表格
        """
        table = "| 方案 | P@5 | Hit@1 | 冗余率 | p95(ms) |\n"
        table += "|------|-----|-------|--------|---------|\n"
        table += f"| {self.config_a['name']} | {metrics_a['precision_at_5']:.2f} | {metrics_a['hit_at_1']:.2f} | {metrics_a['redundancy']:.2f} | {metrics_a['p95_latency']:.1f} |\n"
        table += f"| {self.config_b['name']} | {metrics_b['precision_at_5']:.2f} | {metrics_b['hit_at_1']:.2f} | {metrics_b['redundancy']:.2f} | {metrics_b['p95_latency']:.1f} |\n"
        table += f"\n推荐：{recommendation}"
        
        return table
    
    def save_results(self, results_a, results_b, metrics_a, metrics_b, recommendation):
        """
        保存评估结果到文件
        
        参数:
            results_a: 方案A的详细结果
            results_b: 方案B的详细结果
            metrics_a: 方案A的评估指标
            metrics_b: 方案B的评估指标
            recommendation: 推荐方案
        """
        # 保存详细结果
        with open("ab_eval_detailed_results.json", "w", encoding="utf-8") as f:
            json.dump({
                'config_a': self.config_a,
                'config_b': self.config_b,
                'results_a': results_a,
                'results_b': results_b,
                'metrics_a': metrics_a,
                'metrics_b': metrics_b,
                'recommendation': recommendation
            }, f, ensure_ascii=False, indent=2)
        print("📝 详细评估结果已保存至: ab_eval_detailed_results.json")
        
        # 保存对比表
        table = self._format_metrics_table(metrics_a, metrics_b, recommendation)
        with open("ab_eval_comparison.md", "w", encoding="utf-8") as f:
            f.write(f"# A/B 评估对比结果\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(table)
        print("📝 评估对比表已保存至: ab_eval_comparison.md")
    
    def run_evaluation(self):
        """运行A/B评估流程"""
        print("🚀 启动A/B评估指标对比工具")
        
        # 记录开始时间
        start_time = time.time()
        
        # 模拟方案A的检索结果
        print(f"\n🔍 运行方案 {self.config_a['name']} 评估...")
        results_a = self._simulate_retrieval_results(self.config_a)
        metrics_a = self._calculate_metrics(results_a)
        print(f"✅ 方案 {self.config_a['name']} 评估完成")
        
        # 模拟方案B的检索结果
        print(f"\n🔍 运行方案 {self.config_b['name']} 评估...")
        results_b = self._simulate_retrieval_results(self.config_b)
        metrics_b = self._calculate_metrics(results_b)
        print(f"✅ 方案 {self.config_b['name']} 评估完成")
        
        # 生成推荐
        recommendation = self._generate_recommendation(metrics_a, metrics_b)
        
        # 格式化并打印对比表
        table = self._format_metrics_table(metrics_a, metrics_b, recommendation)
        print("\n📊 A/B 评估对比结果:")
        print(table)
        
        # 保存结果到文件
        self.save_results(results_a, results_b, metrics_a, metrics_b, recommendation)
        
        # 记录结束时间
        end_time = time.time()
        print(f"\n⏱️  评估执行时间: {end_time - start_time:.4f}秒")
        
        # 检查中文输出
        print("\n🔍 中文输出测试：A/B评估对比工具成功实现不同滑窗参数配置的性能指标对比")
        
        print("\n✅ 程序执行完成")
        
        return metrics_a, metrics_b, recommendation

def main():
    """主函数"""
    # 创建A/B评估实例
    ab_evaluator = ABEvalMetricsPrinter()
    
    # 运行评估
    ab_evaluator.run_evaluation()

if __name__ == "__main__":
    main()