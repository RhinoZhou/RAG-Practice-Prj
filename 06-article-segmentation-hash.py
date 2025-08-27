#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""法律条款依存关系分析工具（带哈希值和元数据关联功能）

功能概述：
该工具用于分析法律条款之间的引用关系，并为每个条款生成SHA-256哈希值
及关联元数据，构建完整的证据链以满足审计要求。

主要功能：
- 读取已分割的法律条款JSON文件
- 为每个条款生成SHA-256哈希值
- 关联修订记录、来源机构等元数据
- 提取条款之间的引用关系
- 构建关系图谱
- 生成静态和交互式可视化图表
- 保存完整的关系数据和元数据

使用方法：
1. 准备已分割的法律条款JSON文件
2. 配置文件路径和元数据信息
3. 运行程序，将生成关系图谱和元数据文件

输入输出：
输入：已分割的法律条款JSON文件（如'中华人民共和国民法典_条款分割.json'）
输出：
  - 静态关系图谱图片（'legal_relation_graph.png'）
  - 交互式关系图谱HTML（'legal_relation_graph.html'）
  - 完整关系数据和元数据JSON（'法律条款关系数据及元数据.json'）
"""

import os
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import hashlib  # 哈希库导入，用于生成SHA-256哈希值
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.commons.utils import JsCode
import numpy as np
from typing import List, Dict, Tuple

class LegalRelationGraphAnalyzer:
    """法律条款关系图谱分析器类
    
    提供法律条款关系提取、哈希值生成、元数据管理和可视化功能
    """
    
    def __init__(self):
        """初始化分析器实例，设置引用模式和数据结构"""
        # 中文数字到阿拉伯数字的映射
        self.cn_num_map = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000
        }
        
        # 引用模式正则表达式
        self.reference_patterns = [
            r'参见第(\d+)条',
            r'参照第(\d+)条',
            r'依照第(\d+)条',
            r'依据第(\d+)条',
            r'根据第(\d+)条',
            r'适用第(\d+)条',
            r'违反第(\d+)条',
            r'符合第(\d+)条',
            r'违反第([一二三四五六七八九十百千]+)条'
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_patterns = [re.compile(pattern) for pattern in self.reference_patterns]
        
        # 创建有向图
        self.graph = nx.DiGraph()
        
        # 条款数据
        self.articles = []
        self.article_id_map = {}
        
        # 元数据存储
        self.metadata_store = {}
    
    def _generate_sha256(self, text):
        """为文本生成SHA-256哈希值"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def read_segmented_articles(self, file_path, source_institution="未知机构", revision_history=None):
        """读取已分割的条款文件并添加哈希值和元数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # 处理每条条款
        for article in articles:
            article_id = article.get('id', '')
            if article_id:
                text = article.get('text', '')
                
                # 生成SHA-256哈希值
                text_hash = self._generate_sha256(text)
                article['hash'] = text_hash
                
                # 构建元数据
                metadata = {
                    'hash': text_hash,
                    'source_institution': source_institution,
                    'revision_history': revision_history or [{
                        'timestamp': "当前时间",
                        'author': "系统自动生成",
                        'comment': "初始导入"
                    }],
                    'source_file': os.path.basename(file_path),
                    'import_time': "当前时间",
                    'content_checksum': text_hash
                }
                
                # 存储元数据
                self.metadata_store[article_id] = metadata
                
                # 更新条款映射
                self.article_id_map[article_id] = article
                article_number = self._extract_article_number(text)
                if article_number:
                    self.article_id_map[article_number] = article
                    # 也为条款编号建立元数据映射
                    self.metadata_store[article_number] = metadata
        
        self.articles = articles
        return articles
    
    def _extract_article_number(self, text):
        """从文本中提取条款编号"""
        # 尝试匹配"第X条"格式的条款编号
        match = re.search(r'第\s*(\d+)\s*条', text)
        if match:
            return match.group(1)
        
        # 尝试匹配中文数字格式的条款编号
        match = re.search(r'第\s*([一二三四五六七八九十百千]+)\s*条', text)
        if match:
            return self._cn_to_arabic(match.group(1))
        
        return None
    
    def _cn_to_arabic(self, cn_num):
        """将中文数字转换为阿拉伯数字"""
        # 简单的中文数字转换逻辑
        result = 0
        temp = 0
        
        for char in cn_num:
            if char in self.cn_num_map:
                num = self.cn_num_map[char]
                if num == 10:
                    # 处理"十"的情况
                    if temp == 0:
                        temp = 10
                    else:
                        temp *= 10
                elif num == 100 or num == 1000:
                    # 处理"百"和"千"的情况
                    if temp == 0:
                        result += num
                    else:
                        result += temp * num
                        temp = 0
                else:
                    # 处理个位数字
                    temp = num
            
        # 处理最后剩下的数字
        result += temp
        return str(result)
    
    def build_relation_graph(self):
        """构建条款之间的关系图谱，包含哈希值和元数据"""
        # 为每个条款创建节点
        for article in self.articles:
            article_id = article.get('id', '')
            if not article_id:
                continue
            
            # 添加节点属性，包含哈希值
            text = article.get('text', '')
            text_hash = article.get('hash', '')
            
            self.graph.add_node(article_id, 
                               label=article.get('title', article_id),
                               text=text,
                               hash=text_hash,
                               length=len(text))
            
            # 提取引用关系
            relations = self._extract_relations(text)
            
            # 添加边
            for relation_type, target_article_num in relations:
                # 查找目标条款
                target_article = self.article_id_map.get(target_article_num)
                if target_article:
                    target_id = target_article.get('id', '')
                    if target_id and target_id != article_id:
                        # 添加边，记录关系类型
                        self.graph.add_edge(article_id, target_id, relation_type=relation_type)
        
        return self.graph
    
    def _extract_relations(self, text):
        """提取文本中的引用关系"""
        relations = []
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                relation_type = match.group(0)[:2]  # 获取关系类型（如"参见"、"依照"等）
                article_num_text = match.group(1)
                
                # 转换条款编号为阿拉伯数字
                if article_num_text.isdigit():
                    article_num = article_num_text
                else:
                    article_num = self._cn_to_arabic(article_num_text)
                
                relations.append((relation_type, article_num))
        
        return relations
    
    def analyze_graph(self):
        """分析关系图谱特征"""
        stats = {
            "nodes_count": len(self.graph.nodes()),
            "edges_count": len(self.graph.edges()),
            "density": nx.density(self.graph)
        }
        
        # 计算入度和出度
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        if in_degrees:
            stats["avg_in_degree"] = sum(in_degrees.values()) / len(in_degrees)
            stats["max_in_degree"] = max(in_degrees.values())
            stats["max_in_degree_node"] = max(in_degrees.items(), key=lambda x: x[1])[0]
        
        if out_degrees:
            stats["avg_out_degree"] = sum(out_degrees.values()) / len(out_degrees)
            stats["max_out_degree"] = max(out_degrees.values())
            stats["max_out_degree_node"] = max(out_degrees.items(), key=lambda x: x[1])[0]
        
        # 识别关键节点（中介中心性）
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            if betweenness:
                stats["top_betweenness_nodes"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        except:
            pass
        
        return stats
    
    def save_relation_data(self, output_file):
        """保存关系数据和元数据到JSON文件，构建完整证据链"""
        # 构建完整的关系数据结构
        data = {
            'nodes': [],
            'edges': [],
            'metadata': self.metadata_store,
            'statistics': self.analyze_graph()
        }
        
        # 转换节点数据
        for node_id, node_data in self.graph.nodes(data=True):
            node_info = {
                'id': node_id,
                'label': node_data.get('label', ''),
                'hash': node_data.get('hash', ''),
                'length': node_data.get('length', 0)
            }
            
            # 添加对应的元数据引用
            if node_id in self.metadata_store:
                node_info['metadata_ref'] = node_id
            
            data['nodes'].append(node_info)
        
        # 转换边数据
        for source, target, edge_data in self.graph.edges(data=True):
            edge_info = {
                'source': source,
                'target': target,
                'relation_type': edge_data.get('relation_type', '')
            }
            data['edges'].append(edge_info)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"关系数据和元数据已保存至: {output_file}")
        return output_file
    
    def visualize_with_matplotlib(self, output_file=None):
        """使用Matplotlib可视化关系图谱"""
        plt.figure(figsize=(15, 12))
        
        # 使用spring布局算法
        pos = nx.spring_layout(self.graph, k=0.15, iterations=20)
        
        # 绘制节点和边
        nx.draw_networkx_nodes(self.graph, pos, node_size=300, node_color='lightblue')
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', alpha=0.5)
        
        # 添加标签（只显示部分节点标签以避免拥挤）
        labels = {node: data['label'] for node, data in self.graph.nodes(data=True) 
                 if self.graph.in_degree(node) > 1 or self.graph.out_degree(node) > 1}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title('法律条款引用关系图谱', fontsize=15)
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"关系图谱已保存至: {output_file}")
        else:
            plt.show()
    
    def visualize_with_pyecharts(self, output_file="legal_relation_graph.html"):
        """使用Pyecharts创建交互式关系图，包含哈希值信息"""
        nodes = []
        links = []
        categories = []
        
        # 准备节点数据
        for node_id, data in self.graph.nodes(data=True):
            # 根据节点的入度和出度确定节点大小
            node_size = 10 + self.graph.in_degree(node_id) * 5 + self.graph.out_degree(node_id) * 5
            node_size = min(node_size, 50)  # 限制最大节点大小
            
            # 获取哈希值的前8位作为标识
            hash_preview = data.get('hash', '')[:8] if 'hash' in data else ''
            
            nodes.append({
                "id": str(node_id),
                "name": data['label'],
                "symbolSize": node_size,
                "value": data['length'],
                "category": 0,  # 可以根据需要添加更多类别
                "tooltip": {
                    "formatter": JsCode(f"function(params){{return params.name + '<br/>文本长度: ' + params.value + '<br/>哈希值: {hash_preview}...';}}")
                }
            })
        
        # 准备边数据
        for source, target, data in self.graph.edges(data=True):
            links.append({
                "source": str(source),
                "target": str(target),
                "value": 1,
                "label": {
                    "show": False  # 为了避免图表拥挤，不显示边标签
                },
                "lineStyle": {
                    "curveness": 0.3
                }
            })
        
        # 准备分类数据
        categories.append({"name": "法律条款"})
        
        # 创建图实例
        graph = Graph(init_opts=opts.InitOpts(width="1000px", height="800px"))
        graph.add("法律条款",
                  nodes=nodes,
                  links=links,
                  categories=categories,
                  layout="force",
                  is_rotate_label=True,
                  linestyle_opts=opts.LineStyleOpts(color="source"),
                  label_opts=opts.LabelOpts(formatter="{b}"),
                  tooltip_opts=opts.TooltipOpts(formatter="{b}"),
                  itemstyle_opts=opts.ItemStyleOpts(
                      color=JsCode("function(params) {return '#' + Math.floor(Math.random() * 16777215).toString(16);}")
                  ))
        
        # 设置全局配置
        graph.set_global_opts(
            title_opts=opts.TitleOpts(title="法律条款依存关系图谱", subtitle="交互式可视化"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}")
        )
        
        # 保存为HTML文件
        graph.render(output_file)
        print(f"交互式关系图谱已保存至: {output_file}")
        return output_file

# 修改main函数以使用新功能

def main():
    """主函数"""
    # 创建分析器实例
    relation_graph = LegalRelationGraphAnalyzer()
    
    # 读取分割后的条款文件
    file_path = "中华人民共和国民法典_条款分割.json"
    try:
        # 添加来源机构和修订历史元数据
        revision_history = [
            {
                'timestamp': "2023-01-01",
                'author': "法律专家团队",
                'comment': "民法典条款分割完成"
            },
            {
                'timestamp': "2023-01-15",
                'author': "系统管理员",
                'comment': "导入系统并生成哈希值"
            }
        ]
        
        relation_graph.read_segmented_articles(
            file_path,
            source_institution="中国法律法规数据库",
            revision_history=revision_history
        )
        
        # 构建关系图谱
        relation_graph.build_relation_graph()
        
        # 分析图谱特征
        stats = relation_graph.analyze_graph()
        print("关系图谱分析结果:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 可视化关系图谱
        relation_graph.visualize_with_matplotlib(output_file="legal_relation_graph.png")
        
        # 创建交互式可视化
        html_file = relation_graph.visualize_with_pyecharts()
        
        # 保存关系数据和元数据，构建证据链
        relation_graph.save_relation_data("法律条款关系数据及元数据.json")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()