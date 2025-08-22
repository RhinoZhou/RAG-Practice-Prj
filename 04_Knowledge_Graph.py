# d:\rag-project\05-rag-practice\03-Standardization\4_ROI_Knowledge_Graph.py
"""
SVG知识图谱构建系统
SVG Knowledge Graph Construction System

本系统支持用户输入任意主题，生成SVG格式的知识图谱可视化
"""

import os
import json
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# 基础配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicKnowledgeGraph:
    """动态知识图谱构建器"""
    
    def __init__(self):
        self.graph_data = {
            "nodes": [],
            "edges": []
        }
    
    def get_user_input(self) -> Dict[str, Any]:
        """获取用户输入的知识图谱数据"""
        print("\n" + "="*60)
        print("🎯 SVG知识图谱构建器")
        print("="*60)
        print("系统将生成SVG格式的知识图谱可视化文件")
        
        # 获取中心概念
        center_concept = input("请输入核心概念名称: ").strip()
        if not center_concept:
            center_concept = "示例概念"
        
        center_definition = input(f"请输入'{center_concept}'的定义: ").strip()
        if not center_definition:
            center_definition = f"{center_concept}的定义待补充"
        
        # 获取相关术语
        related_terms = {}
        print("\n📚 添加相关术语 (直接回车结束)")
        while True:
            term = input("请输入相关术语名称: ").strip()
            if not term:
                break
            definition = input(f"请输入'{term}'的定义: ").strip()
            if definition:
                related_terms[term] = definition
        
        # 获取应用场景
        scenarios = []
        print("\n🎯 添加应用场景 (直接回车结束)")
        while True:
            scenario = input("请输入应用场景: ").strip()
            if not scenario:
                break
            scenarios.append(scenario)
        
        # 获取同义词
        synonyms = []
        print("\n🔗 添加同义词 (直接回车结束)")
        while True:
            synonym = input("请输入同义词: ").strip()
            if not synonym:
                break
            synonyms.append(synonym)
        
        return {
            "center_concept": center_concept,
            "center_definition": center_definition,
            "related_terms": related_terms,
            "scenarios": scenarios,
            "synonyms": synonyms
        }
    
    def create_dynamic_graph(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """根据用户输入创建知识图谱"""
        logger.info(f"正在创建'{user_data['center_concept']}'的SVG知识图谱...")
        
        graph_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "center_concept": user_data["center_concept"],
                "total_nodes": 0,
                "total_edges": 0
            },
            "nodes": [],
            "edges": []
        }
        
        center_concept = user_data["center_concept"]
        
        # 添加中心节点
        graph_data["nodes"].append({
            "id": "center",
            "name": center_concept,
            "type": "核心概念",
            "definition": user_data["center_definition"]
        })
        
        # 添加相关术语节点
        for i, (term, definition) in enumerate(user_data["related_terms"].items()):
            node_id = f"term_{i}"
            graph_data["nodes"].append({
                "id": node_id,
                "name": term,
                "type": "相关术语",
                "definition": definition
            })
            graph_data["edges"].append({
                "source": "center",
                "target": node_id,
                "relationship": "相关术语"
            })
        
        # 添加应用场景节点
        for i, scenario in enumerate(user_data["scenarios"]):
            node_id = f"scenario_{i}"
            graph_data["nodes"].append({
                "id": node_id,
                "name": scenario,
                "type": "应用场景",
                "definition": f"{center_concept}在{scenario}中的应用"
            })
            graph_data["edges"].append({
                "source": "center",
                "target": node_id,
                "relationship": "应用于"
            })
        
        # 添加同义词节点
        for i, synonym in enumerate(user_data["synonyms"]):
            node_id = f"synonym_{i}"
            graph_data["nodes"].append({
                "id": node_id,
                "name": synonym,
                "type": "同义词",
                "definition": f"{center_concept}的同义词"
            })
            graph_data["edges"].append({
                "source": "center",
                "target": node_id,
                "relationship": "同义词"
            })
        
        # 更新统计信息
        graph_data["metadata"]["total_nodes"] = len(graph_data["nodes"])
        graph_data["metadata"]["total_edges"] = len(graph_data["edges"])
        
        return graph_data
    
    def export_graph(self, graph_data: Dict[str, Any], filename: str = None):
        """导出知识图谱到JSON文件"""
        if not filename:
            center_name = graph_data["metadata"]["center_concept"]
            filename = f"{center_name}_knowledge_graph.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识图谱已导出到: {filename}")
        return filename
    
    def print_graph_summary(self, graph_data: Dict[str, Any]):
        """打印知识图谱摘要"""
        metadata = graph_data["metadata"]
        center_name = metadata["center_concept"]
        
        print("\n" + "="*60)
        print(f"📊 '{center_name}'知识图谱摘要")
        print("="*60)
        
        print(f"\n核心概念: {center_name}")
        print(f"定义: {[node['definition'] for node in graph_data['nodes'] if node['id'] == 'center'][0]}")
        
        # 统计各类节点
        node_types = {}
        for node in graph_data["nodes"]:
            node_type = node["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\n节点统计:")
        for node_type, count in node_types.items():
            print(f"  • {node_type}: {count}个")
        
        print(f"\n总节点数: {metadata['total_nodes']}")
        print(f"总关系数: {metadata['total_edges']}")

class SVGKnowledgeVisualizer:
    """SVG知识图谱可视化生成器"""
    
    def __init__(self):
        self.svg_width = 1200
        self.svg_height = 800
        self.center_x = self.svg_width // 2
        self.center_y = self.svg_height // 2
        self.center_radius = 60
        self.node_radius = 35
        self.orbit_radius = 220
        
        # 颜色配置
        self.color_map = {
            "核心概念": "#ff6b6b",
            "相关术语": "#4ecdc4",
            "应用场景": "#45b7d1",
            "同义词": "#f093fb"
        }
    
    def calculate_positions(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """计算节点位置（圆形布局）"""
        positions = {"center": (self.center_x, self.center_y)}
        outer_nodes = [n for n in nodes if n["id"] != "center"]
        
        if not outer_nodes:
            return positions
        
        # 计算角度步长
        angle_step = 2 * math.pi / len(outer_nodes)
        
        for i, node in enumerate(outer_nodes):
            angle = angle_step * i - math.pi / 2  # 从顶部开始
            x = self.center_x + self.orbit_radius * math.cos(angle)
            y = self.center_y + self.orbit_radius * math.sin(angle)
            positions[node["id"]] = (x, y)
        
        return positions
    
    def create_svg_visualization(self, graph_data: Dict[str, Any]) -> str:
        """创建SVG格式的知识图谱"""
        center_name = graph_data["metadata"]["center_concept"]
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        
        # 计算节点位置
        positions = self.calculate_positions(nodes, edges)
        
        # 生成SVG内容
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{self.svg_width}" height="{self.svg_height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#667eea;stop-opacity:0.05" />
            <stop offset="100%" style="stop-color:#764ba2;stop-opacity:0.05" />
        </linearGradient>
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="2" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.1"/>
        </filter>
        <style>
            .node-text {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 12px; font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; }}
            .title-text {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #333; text-anchor: middle; }}
            .info-text {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 14px; fill: #666; text-anchor: middle; }}
            .edge-text {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 11px; fill: #666; text-anchor: middle; }}
            .legend-text {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 12px; fill: #333; }}
        </style>
    </defs>
    
    <!-- 背景 -->
    <rect width="100%" height="100%" fill="url(#bgGradient)"/>
    
    <!-- 标题和统计信息 -->
    <text x="{self.center_x}" y="30" class="title-text">🎯 {center_name}知识图谱</text>
    <text x="{self.center_x}" y="55" class="info-text">
        节点: {len(nodes)} | 关系: {len(edges)} | 创建: {graph_data["metadata"]["created_at"][:10]}
    </text>'''
        
        # 绘制连接线
        for edge in edges:
            source_pos = positions.get(edge["source"], (self.center_x, self.center_y))
            target_pos = positions.get(edge["target"], (self.center_x, self.center_y))
            
            # 计算连接线的起点和终点（考虑节点半径）
            dx = target_pos[0] - source_pos[0]
            dy = target_pos[1] - source_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                dx_norm = dx / distance
                dy_norm = dy / distance
                
                # 起点和终点偏移
                source_radius = self.center_radius if edge["source"] == "center" else self.node_radius
                target_radius = self.center_radius if edge["target"] == "center" else self.node_radius
                
                start_x = source_pos[0] + dx_norm * source_radius
                start_y = source_pos[1] + dy_norm * source_radius
                end_x = target_pos[0] - dx_norm * target_radius
                end_y = target_pos[1] - dy_norm * target_radius
                
                # 计算关系标签位置
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                svg_content += f'''
    <line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" 
          stroke="#999" stroke-width="2" opacity="0.7"/>
    <text x="{mid_x}" y="{mid_y - 5}" class="edge-text" transform="rotate(0, {mid_x}, {mid_y})">
        {edge["relationship"]}
    </text>'''
        
        # 绘制节点
        for node in nodes:
            pos = positions.get(node["id"], (self.center_x, self.center_y))
            color = self.color_map.get(node["type"], "#999")
            radius = self.center_radius if node["id"] == "center" else self.node_radius
            
            # 节点圆圈
            svg_content += f'''
    <circle cx="{pos[0]}" cy="{pos[1]}" r="{radius}" fill="{color}" 
            stroke="white" stroke-width="3" filter="url(#shadow)" opacity="0.9"/>'''
            
            # 节点文字（名称，限制长度）
            name_text = node["name"]
            if len(name_text) > 6:
                name_text = name_text[:5] + "..."
            
            svg_content += f'''
    <text x="{pos[0]}" y="{pos[1]}" class="node-text">{name_text}</text>'''
        
        # 添加图例
        legend_x = 50
        legend_y = 100
        svg_content += f'''
    <g transform="translate({legend_x}, {legend_y})">
        <text x="0" y="0" class="legend-text" font-weight="bold">图例</text>'''
        
        for i, (node_type, color) in enumerate(self.color_map.items()):
            y = 20 + i * 20
            svg_content += f'''
        <circle cx="10" cy="{y}" r="8" fill="{color}"/>
        <text x="25" y="{y + 4}" class="legend-text">{node_type}</text>'''
        
        svg_content += '''
    </g>'''
        
        # 关闭SVG标签
        svg_content += '''
</svg>'''
        
        return svg_content
    
    def save_svg_files(self, graph_data: Dict[str, Any]) -> Tuple[str, str]:
        """保存SVG文件和HTML查看器"""
        center_name = graph_data["metadata"]["center_concept"]
        
        # 生成SVG内容
        svg_content = self.create_svg_visualization(graph_data)
        
        # 保存SVG文件
        svg_filename = f"{center_name}_knowledge_graph.svg"
        with open(svg_filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        logger.info(f"SVG知识图谱已创建: {svg_filename}")
        
        # 创建HTML查看器
        html_viewer = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{center_name}知识图谱 - SVG</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .svg-container {{
            text-align: center;
            margin: 20px 0;
        }}
        svg {{
            border: 1px solid #ddd;
            border-radius: 5px;
            max-width: 100%;
            height: auto;
        }}
        .download {{
            text-align: center;
            margin-top: 20px;
        }}
        .download a {{
            display: inline-block;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
            margin: 0 10px;
        }}
        .download a:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 {center_name}知识图谱</h1>
        <div class="info">
            节点: {len(graph_data["nodes"])} | 关系: {len(graph_data["edges"])} 
            | 创建时间: {graph_data["metadata"]["created_at"][:10]}
        </div>
        
        <div class="svg-container">
            {svg_content}
        </div>
        
        <div class="download">
            <a href="{svg_filename}" download>下载SVG文件</a>
            <a href="{center_name}_knowledge_graph.json" download>下载JSON数据</a>
        </div>
    </div>
</body>
</html>'''
        
        html_filename = f"{center_name}_svg_viewer.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_viewer)
        
        logger.info(f"SVG查看器已创建: {html_filename}")
        
        return svg_filename, html_filename

def run_interactive_mode():
    """运行交互式模式"""
    print("\n" + "="*60)
    print("🚀 SVG知识图谱构建系统")
    print("="*60)
    print("本系统支持创建任意主题的知识图谱")
    print("将生成SVG格式的可视化文件")
    
    builder = DynamicKnowledgeGraph()
    visualizer = SVGKnowledgeVisualizer()
    
    try:
        while True:
            # 获取用户输入
            user_data = builder.get_user_input()
            
            # 创建知识图谱
            graph_data = builder.create_dynamic_graph(user_data)
            
            # 打印摘要
            builder.print_graph_summary(graph_data)
            
            # 导出JSON文件
            json_filename = builder.export_graph(graph_data)
            
            # 生成SVG文件和HTML查看器
            svg_filename, html_filename = visualizer.save_svg_files(graph_data)
            
            print(f"\n✅ 文件已生成:")
            print(f"  • JSON数据: {json_filename}")
            print(f"  • SVG图形: {svg_filename}")
            print(f"  • HTML查看器: {html_filename}")
            
            # 询问是否继续
            continue_choice = input("\n是否创建另一个知识图谱? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        
        print("\n🎉 知识图谱构建完成！")
        print("请查看生成的SVG和HTML文件")
        
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用！再见！")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        print(f"发生错误: {e}")

def main():
    """主程序入口"""
    print("\n" + "="*60)
    print("🎯 SVG知识图谱构建系统")
    print("="*60)
    print("支持创建任意主题的知识图谱")
    print("将生成SVG格式的可视化文件")
    
    run_interactive_mode()

if __name__ == "__main__":
    main()