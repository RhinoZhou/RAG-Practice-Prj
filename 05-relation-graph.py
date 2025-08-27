"""
法律条款依存关系分析工具

功能：
- 读取已分割的法律条款JSON文件
- 识别条款中的引用关系（如"参见第X条"、"参照第X条"等）
- 构建条款之间的逻辑关联图谱
- 分析条款之间的引用网络特征
- 可视化条款关系网络
- 保存关系图谱数据便于后续分析和应用

依赖：
- networkx: 用于构建和分析复杂网络
- matplotlib: 用于可视化网络图
- pyecharts: 用于创建交互式关系图
- numpy: 用于数值计算

使用方法：
1. 安装依赖：pip install networkx matplotlib pyecharts numpy
2. 确保已分割的法律条款JSON文件（如《中华人民共和国民法典_条款分割.json》）在当前目录下
3. 运行程序：python 05-relation-graph.py
4. 查看生成的关系图谱结果和可视化文件

注意事项：
- 程序支持识别多种引用模式，包括"参见第X条"、"参照第X条"、"依照第X条"等
- 中文数字和阿拉伯数字的条款编号都能被正确识别
- 生成的关系数据可以用于后续的法律知识图谱构建
"""
import os
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import hashlib  # 添加哈希库导入
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.commons.utils import JsCode
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

class LegalRelationGraph:
    """
    法律条款依存关系图谱构建器
    用于识别条款间引用关系并构建可视化的关系网络
    """
    def __init__(self):
        """初始化关系图谱构建器"""
        # 定义引用关系的正则表达式模式
        self.reference_patterns = [
            r'参[见照考阅]\s*第\s*([零一二三四五六七八九十百千]+|\d+)\s*条',  # 参见第X条
            r'[依参]照\s*第\s*([零一二三四五六七八九十百千]+|\d+)\s*条',  # 依照第X条/参照第X条
            r'根据\s*第\s*([零一二三四五六七八九十百千]+|\d+)\s*条',    # 根据第X条
            r'依照\s*第\s*([零一二三四五六七八九十百千]+|\d+)\s*条',    # 依照第X条
            r'适用\s*第\s*([零一二三四五六七八九十百千]+|\d+)\s*条',    # 适用第X条
            r'第\s*([零一二三四五六七八九十百千]+|\d+)\s*条\s*规定',    # 第X条规定
            r'第\s*([零一二三四五六七八九十百千]+|\d+)\s*条\s*所规定',  # 第X条所规定
            r'第\s*([零一二三四五六七八九十百千]+|\d+)\s*条\s*的规定',  # 第X条的规定
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_patterns = [re.compile(pattern) for pattern in self.reference_patterns]
        
        # 中文数字到阿拉伯数字的映射
        self.cn_num_map = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000
        }
        
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
        # 注意：这里只处理了简单的中文数字，复杂的组合可能需要更复杂的逻辑
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
                      color=JsCode("function(params) {return '#\\' + Math.floor(Math.random() * 16777215).toString(16);}")
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
    
    def save_relation_data(self, output_file):
        """保存关系数据为JSON格式"""
        relation_data = {
            "nodes": [],
            "edges": []
        }
        
        # 保存节点数据
        for node_id, data in self.graph.nodes(data=True):
            node_info = {
                "id": node_id,
                "label": data['label'],
                "text": data['text'],
                "length": data['length'],
                "in_degree": self.graph.in_degree(node_id),
                "out_degree": self.graph.out_degree(node_id)
            }
            relation_data["nodes"].append(node_info)
        
        # 保存边数据
        for source, target, data in self.graph.edges(data=True):
            edge_info = {
                "source": source,
                "target": target,
                "relation_type": data['relation_type']
            }
            relation_data["edges"].append(edge_info)
        
        # 保存分析结果
        relation_data["analysis"] = self.analyze_graph()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relation_data, f, ensure_ascii=False, indent=2)
        
        print(f"关系数据已保存至: {output_file}")


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