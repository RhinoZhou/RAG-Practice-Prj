#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量重排与行为加权工具

作者：Ph.D. Rhino

功能说明：用可解释特征线性打分，并融合点击/停留等行为信号进行候选重排。

内容概述：构造候选特征（BM25、TF‑IDF、标题命中、来源），线性加权得基准分；再对接历史行为信号（CTR、停留）做后验加权，比较重排前后差异。

执行流程：
1. 为候选构建特征向量
2. 线性权重计算基准分
3. 标准化行为信号并加权融合
4. 输出基准与最终排名对比

输入说明：内置候选特征与行为统计；无需外部输入。
"""

import json
import time
import os
import numpy as np

# 自动安装必要的依赖
def install_dependencies():
    try:
        import numpy as np
        print("依赖已满足，无需安装。")
        return True
    except ImportError:
        print("正在安装必要的依赖包...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "numpy"])
            print("依赖安装成功。")
            return True
        except Exception as e:
            print(f"依赖安装失败：{e}")
            return False

class LightRerankWithBehavior:
    """
    轻量重排与行为加权类
    提供基于可解释特征的线性打分和行为信号融合功能
    """
    def __init__(self):
        """初始化参数和权重配置"""
        # 特征权重配置
        self.feature_weights = {
            'bm25_score': 0.3,
            'tfidf_score': 0.2,
            'title_match': 0.3,
            'source_score': 0.2
        }
        
        # 行为信号权重配置
        self.behavior_weights = {
            'ctr': 0.5,
            'avg_stay_time': 0.5
        }
        
        # 初始化内置的候选集合
        self.initialize_candidates()
        
        # 结果存储
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'feature_weights': self.feature_weights,
                'behavior_weights': self.behavior_weights
            },
            'candidates': [],
            'feature_based_ranking': [],
            'behavior_enhanced_ranking': [],
            'execution_time': 0
        }
    
    def initialize_candidates(self):
        """初始化内置的候选集合及其特征和行为数据"""
        self.candidates = [
            {
                'id': 'd1',
                'content': '大语言模型在自然语言处理领域的最新进展与应用案例分析',
                'features': {
                    'bm25_score': 0.85,
                    'tfidf_score': 0.78,
                    'title_match': 1.0,  # 1表示完全匹配，0表示不匹配
                    'source_score': 0.9   # 来源权威性得分
                },
                'behavior': {
                    'clicks': 150,
                    'impressions': 1000,
                    'avg_stay_time': 120,  # 平均停留时间（秒）
                    'total_users': 500
                }
            },
            {
                'id': 'd2',
                'content': '自然语言处理技术在智能客服系统中的应用实践',
                'features': {
                    'bm25_score': 0.72,
                    'tfidf_score': 0.81,
                    'title_match': 0.6,
                    'source_score': 0.8
                },
                'behavior': {
                    'clicks': 80,
                    'impressions': 1000,
                    'avg_stay_time': 90,
                    'total_users': 500
                }
            },
            {
                'id': 'd3',
                'content': '最新大语言模型技术综述与未来发展趋势',
                'features': {
                    'bm25_score': 0.81,
                    'tfidf_score': 0.85,
                    'title_match': 0.8,
                    'source_score': 0.85
                },
                'behavior': {
                    'clicks': 250,
                    'impressions': 1000,
                    'avg_stay_time': 180,
                    'total_users': 500
                }
            }
        ]
    
    def calculate_feature_based_score(self, candidate):
        """
        计算基于特征的基准分数
        
        Args:
            candidate: 候选文档
        
        Returns:
            基于特征的基准分数
        """
        score = 0
        for feature_name, weight in self.feature_weights.items():
            score += candidate['features'][feature_name] * weight
        return score
    
    def calculate_ctr(self, candidate):
        """
        计算点击率(CTR)
        
        Args:
            candidate: 候选文档
        
        Returns:
            点击率(CTR)值
        """
        behavior = candidate['behavior']
        # 防止除零错误
        if behavior['impressions'] == 0:
            return 0
        return behavior['clicks'] / behavior['impressions']
    
    def normalize_scores(self, scores):
        """
        对分数进行归一化处理
        
        Args:
            scores: 分数列表
        
        Returns:
            归一化后的分数列表
        """
        min_score = min(scores)
        max_score = max(scores)
        
        # 防止除零错误
        if max_score - min_score == 0:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def calculate_behavior_enhanced_score(self, feature_score, behavior_metrics):
        """
        计算融合行为信号的增强分数
        
        Args:
            feature_score: 基于特征的基准分数
            behavior_metrics: 归一化后的行为指标
        
        Returns:
            融合后的增强分数
        """
        behavior_score = 0
        for metric_name, metric_value in behavior_metrics.items():
            behavior_score += metric_value * self.behavior_weights.get(metric_name, 0)
        
        # 将特征分数和行为分数按比例融合，这里使用7:3的比例
        enhanced_score = feature_score * 0.7 + behavior_score * 0.3
        return enhanced_score
    
    def run(self):
        """
        运行完整的轻量重排与行为加权流程
        
        Returns:
            元组(基于特征的排名, 融合行为的排名)
        """
        start_time = time.time()
        
        # 保存候选信息
        self.results['candidates'] = self.candidates
        
        # 计算每个候选的基于特征的分数
        feature_based_scores = []
        for candidate in self.candidates:
            feature_score = self.calculate_feature_based_score(candidate)
            feature_based_scores.append((candidate['id'], feature_score))
        
        # 按基于特征的分数排序
        feature_based_ranking = sorted(feature_based_scores, key=lambda x: x[1], reverse=True)
        self.results['feature_based_ranking'] = feature_based_ranking
        
        # 计算CTR和处理行为数据
        all_ctr = []
        all_stay_time = []
        for candidate in self.candidates:
            ctr = self.calculate_ctr(candidate)
            all_ctr.append(ctr)
            all_stay_time.append(candidate['behavior']['avg_stay_time'])
        
        # 归一化行为指标
        normalized_ctr = self.normalize_scores(all_ctr)
        normalized_stay_time = self.normalize_scores(all_stay_time)
        
        # 计算融合行为信号的增强分数
        behavior_enhanced_scores = []
        for i, candidate in enumerate(self.candidates):
            behavior_metrics = {
                'ctr': normalized_ctr[i],
                'avg_stay_time': normalized_stay_time[i]
            }
            feature_score = dict(feature_based_scores)[candidate['id']]
            enhanced_score = self.calculate_behavior_enhanced_score(feature_score, behavior_metrics)
            behavior_enhanced_scores.append((candidate['id'], enhanced_score))
        
        # 按融合行为的分数排序
        behavior_enhanced_ranking = sorted(behavior_enhanced_scores, key=lambda x: x[1], reverse=True)
        self.results['behavior_enhanced_ranking'] = behavior_enhanced_ranking
        
        # 计算执行时间
        self.results['execution_time'] = time.time() - start_time
        
        return feature_based_ranking, behavior_enhanced_ranking
    
    def save_results(self, output_file='light_rerank_results.json'):
        """
        保存结果到JSON文件
        
        Args:
            output_file: 输出文件名
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_file}")
    
    def display_results(self):
        """显示结果到控制台"""
        # 显示基于特征的排名
        feature_ranked_ids = [item[0] for item in self.results['feature_based_ranking']]
        print(f"基于特征：{' > '.join(feature_ranked_ids)}")
        
        # 显示融合行为的排名
        behavior_ranked_ids = [item[0] for item in self.results['behavior_enhanced_ranking']]
        print(f"加入行为：{' > '.join(behavior_ranked_ids)}")
        
        # 显示执行时间
        print(f"总执行时间：{self.results['execution_time']:.4f}秒")

# 主函数
def main():
    # 安装依赖
    if not install_dependencies():
        print("无法安装必要的依赖，程序退出。")
        return
    
    # 创建轻量重排与行为加权实例
    lrwb = LightRerankWithBehavior()
    
    # 运行处理流程
    print("正在运行轻量重排与行为加权...")
    feature_ranking, behavior_ranking = lrwb.run()
    
    # 显示结果
    lrwb.display_results()
    
    # 保存结果
    lrwb.save_results()
    
    # 检查结果文件是否存在
    if os.path.exists('light_rerank_results.json'):
        print("结果文件生成成功。")
    else:
        print("警告：结果文件未生成。")
    
    # 验证中文输出
    print("验证中文输出：大语言模型与行为信号融合的重排策略")

if __name__ == "__main__":
    main()