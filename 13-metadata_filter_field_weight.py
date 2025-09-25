#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
元数据过滤与字段加权工具

作者：Ph.D. Rhino

功能说明：按元数据过滤候选片段，并对标题/图注等不同字段施加不同权重重排。

内容概述：候选段带year/source/field元数据；先按时间/来源过滤，再对命中字段（title/caption/body）施加权重重排，演示"范围控制+结构证据强化"。

执行流程：
1. 读入候选及元数据
2. 应用时间/来源过滤
3. 依据字段权重重排
4. 输出前k结果与分数

输入说明：脚本内置候选集合与权重表；无需外部输入。
"""

import json
import time
import os

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

class MetadataFilterAndFieldWeight:
    """
    元数据过滤与字段加权类
    提供按元数据过滤候选片段和对不同字段施加不同权重的功能
    """
    def __init__(self):
        """初始化参数和权重配置"""
        # 字段权重配置：标题权重最高，其次是图注，正文权重最低
        self.field_weights = {
            'title': 1.5,
            'caption': 1.2,
            'body': 1.0
        }
        
        # 初始化内置的候选集合
        self.initialize_candidates()
        
        # 结果存储
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'field_weights': self.field_weights,
                'year_filter': {'min': 2020, 'max': 2024},
                'source_filter': ['arxiv', 'conference'],
                'top_k': 3
            },
            'candidates_before_filter': [],
            'candidates_after_filter': [],
            'ranked_results': [],
            'execution_time': 0
        }
    
    def initialize_candidates(self):
        """初始化内置的候选集合及其元数据"""
        self.candidates = [
            {
                'id': 'c1',
                'content': '大语言模型在自然语言处理领域的最新进展表明，参数规模和数据质量是模型性能的关键因素。',
                'metadata': {
                    'year': 2023,
                    'source': 'conference',
                    'field': 'nlp',
                    'field_type': 'title'
                },
                'base_score': 0.85
            },
            {
                'id': 'c2',
                'content': '实验结果显示，在相同参数规模下，高质量训练数据可以带来15%的性能提升。',
                'metadata': {
                    'year': 2019,
                    'source': 'journal',
                    'field': 'ml',
                    'field_type': 'body'
                },
                'base_score': 0.78
            },
            {
                'id': 'c3',
                'content': '图1：不同参数规模模型的性能对比',
                'metadata': {
                    'year': 2022,
                    'source': 'arxiv',
                    'field': 'nlp',
                    'field_type': 'caption'
                },
                'base_score': 0.72
            },
            {
                'id': 'c4',
                'content': '最新的研究提出了一种新的训练策略，可以在保持性能的同时减少计算资源需求。',
                'metadata': {
                    'year': 2024,
                    'source': 'conference',
                    'field': 'ml',
                    'field_type': 'title'
                },
                'base_score': 0.81
            }
        ]
    
    def filter_by_metadata(self, min_year=2020, max_year=2024, allowed_sources=None):
        """
        按元数据过滤候选片段
        
        Args:
            min_year: 最小年份
            max_year: 最大年份
            allowed_sources: 允许的来源列表
        
        Returns:
            过滤后的候选片段列表
        """
        if allowed_sources is None:
            allowed_sources = ['arxiv', 'conference']
        
        filtered = []
        for candidate in self.candidates:
            # 检查年份是否在范围内
            if not (min_year <= candidate['metadata']['year'] <= max_year):
                continue
            
            # 检查来源是否在允许列表中
            if candidate['metadata']['source'] not in allowed_sources:
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def apply_field_weights(self, candidates):
        """
        对候选片段应用字段权重
        
        Args:
            candidates: 候选片段列表
        
        Returns:
            应用权重后的排序结果
        """
        weighted_results = []
        
        for candidate in candidates:
            field_type = candidate['metadata']['field_type']
            # 获取字段权重，如果不存在则使用默认权重1.0
            weight = self.field_weights.get(field_type, 1.0)
            # 计算加权分数
            weighted_score = candidate['base_score'] * weight
            
            weighted_results.append({
                'id': candidate['id'],
                'content': candidate['content'],
                'metadata': candidate['metadata'],
                'base_score': candidate['base_score'],
                'weight': weight,
                'weighted_score': weighted_score
            })
        
        # 按加权分数降序排序
        weighted_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return weighted_results
    
    def run(self, min_year=2020, max_year=2024, allowed_sources=None, top_k=3):
        """
        运行完整的元数据过滤与字段加权流程
        
        Args:
            min_year: 最小年份
            max_year: 最大年份
            allowed_sources: 允许的来源列表
            top_k: 返回的结果数量
        
        Returns:
            排序后的前k个结果
        """
        start_time = time.time()
        
        # 保存过滤前的候选
        self.results['candidates_before_filter'] = self.candidates
        
        # 应用元数据过滤
        filtered_candidates = self.filter_by_metadata(min_year, max_year, allowed_sources)
        self.results['candidates_after_filter'] = filtered_candidates
        
        # 应用字段权重并排序
        ranked_results = self.apply_field_weights(filtered_candidates)
        
        # 取前k个结果
        top_results = ranked_results[:top_k]
        self.results['ranked_results'] = top_results
        
        # 计算执行时间
        self.results['execution_time'] = time.time() - start_time
        
        return top_results
    
    def save_results(self, output_file='metadata_filter_results.json'):
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
        # 显示过滤前后的数量
        before_count = len(self.results['candidates_before_filter'])
        after_count = len(self.results['candidates_after_filter'])
        print(f"过滤前/后：{before_count}/{after_count}")
        
        # 显示排序结果
        if self.results['ranked_results']:
            result_strings = []
            for result in self.results['ranked_results']:
                field_type = result['metadata']['field_type']
                weight = result['weight']
                result_strings.append(f"{result['id']} ({field_type}×{weight:.1f})")
            
            print(' > '.join(result_strings))
        
        # 显示执行时间
        print(f"总执行时间：{self.results['execution_time']:.4f}秒")

# 主函数
def main():
    # 安装依赖
    if not install_dependencies():
        print("无法安装必要的依赖，程序退出。")
        return
    
    # 创建元数据过滤与字段加权实例
    mffw = MetadataFilterAndFieldWeight()
    
    # 运行处理流程
    print("正在运行元数据过滤与字段加权...")
    results = mffw.run()
    
    # 显示结果
    mffw.display_results()
    
    # 保存结果
    mffw.save_results()
    
    # 检查结果文件是否存在
    if os.path.exists('metadata_filter_results.json'):
        print("结果文件生成成功。")
    else:
        print("警告：结果文件未生成。")
    
    # 验证中文输出
    print("验证中文输出：大语言模型在自然语言处理领域的应用")

if __name__ == "__main__":
    main()