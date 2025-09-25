#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
层级索引构建与锚点工具

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2023-12-16

功能说明:
此工具用于从目录与正文构建"章节-段落-句子"节点树，并生成可定位的锚点。
主要功能包括:
1. 解析目录得到章节层级与编号
2. 切分正文为段落和句子
3. 为每个节点生成唯一标识符、父子链接与锚点
4. 构建完整的节点树结构
5. 生成可模拟点击的锚点

使用场景:
用于技术文档、白皮书等长文本的结构化索引构建，方便内容检索和定位。
"""
import re
import hashlib
import os
from typing import List, Dict, Optional, Tuple

class HierarchicalIndexBuilder:
    """层级索引构建器类"""
    def __init__(self):
        """初始化层级索引构建器"""
        self.document = ""  # 文档内容
        self.nodes = []      # 存储所有节点
        self.chapters = []   # 存储章节信息
        self.paragraphs = [] # 存储段落信息
        self.sentences = []  # 存储句子信息
        self.root_nodes = [] # 存储根节点

    def load_sample_document(self) -> str:
        """加载示例文档内容"""
        # 示例文档，包含目录和正文
        self.document = """
# 目录
1. 介绍
1.1 背景
1.2 目的
2. 核心技术
2.1 滑窗机制
2.2 特征提取

# 正文
第1章 介绍

1.1 背景
随着人工智能技术的快速发展，大语言模型在自然语言处理领域取得了显著进展。然而，在处理长文档时，模型的上下文窗口限制成为了一个主要挑战。为了解决这一问题，检索增强生成（RAG）技术应运而生。

1.2 目的
本白皮书旨在介绍RAG技术中的层级索引构建方法，特别是如何通过滑窗参数优化和锚点定位来提高检索精度和效率。通过建立"章节-段落-句子"的三级索引结构，实现对长文档的高效检索和定位。

第2章 核心技术

2.1 滑窗机制
滑窗机制是RAG技术中的关键组成部分。通过设定合适的窗口大小和重叠大小，可以在不丢失重要信息的前提下，将长文档分割成多个可管理的块。窗口大小决定了每个块的最大长度，而重叠大小则确保了相邻块之间的信息连贯性。

2.2 特征提取
特征提取是提高检索精度的重要步骤。在层级索引中，我们不仅提取文本内容特征，还包括位置特征、结构特征等元数据。这些特征有助于更准确地理解文档结构和内容关联，从而提高检索的相关性和准确性。
"""
        return self.document

    def parse_document(self) -> None:
        """解析文档，提取目录和正文内容"""
        # 分割目录和正文
        parts = self.document.split("# 正文", 1)
        toc_part = parts[0].strip() if len(parts) > 0 else ""
        content_part = parts[1].strip() if len(parts) > 1 else ""

        # 解析目录
        self._parse_toc(toc_part)
        
        # 解析正文
        self._parse_content(content_part)

    def _parse_toc(self, toc_text: str) -> None:
        """解析目录，提取章节信息"""
        # 提取目录中的章节
        toc_lines = toc_text.split("\n")
        toc_lines = [line.strip() for line in toc_lines if line.strip() and not line.startswith("#")]
        
        for line in toc_lines:
            # 匹配章节编号和标题
            match = re.match(r'^(\d+(?:\.\d+)*)\s+(.*)$', line)
            if match:
                chapter_num = match.group(1)
                chapter_title = match.group(2)
                self.chapters.append({
                    "number": chapter_num,
                    "title": chapter_title,
                    "level": len(chapter_num.split(".")),
                    "start_pos": -1,  # 将在后续解析中更新
                    "end_pos": -1     # 将在后续解析中更新
                })

    def _parse_content(self, content_text: str) -> None:
        """解析正文，提取段落和句子"""
        # 按章节分割正文
        chapter_pattern = r'第\d+章\s+[^\n]+'
        chapter_matches = list(re.finditer(chapter_pattern, content_text))
        
        if chapter_matches:
            # 处理每个章节
            for i, match in enumerate(chapter_matches):
                chapter_title = match.group()
                chapter_start = match.end()
                chapter_end = content_text.find("第", chapter_start) if i < len(chapter_matches) - 1 else len(content_text)
                chapter_content = content_text[chapter_start:chapter_end].strip()
                
                # 提取章节编号（从标题中）
                chapter_num_match = re.search(r'第(\d+)章', chapter_title)
                if chapter_num_match:
                    chapter_num = chapter_num_match.group(1)
                    
                    # 查找子章节
                    section_pattern = r'(\d+\.\d+)\s+[^\n]+'
                    section_matches = list(re.finditer(section_pattern, chapter_content))
                    
                    if section_matches:
                        # 处理每个子章节
                        for j, section_match in enumerate(section_matches):
                            section_title = section_match.group()
                            section_num = section_match.group(1)
                            section_start = section_match.end()
                            section_end = chapter_content.find("\n\n", section_start) if j < len(section_matches) - 1 else len(chapter_content)
                            section_content = chapter_content[section_start:section_end].strip()
                            
                            # 分割段落
                            paragraphs = self._split_into_paragraphs(section_content)
                            for para_idx, paragraph in enumerate(paragraphs):
                                # 添加段落信息
                                para_id = f"ch{chapter_num}-sec{section_num.split('.')[1]}-p{para_idx+1}"
                                self.paragraphs.append({
                                    "id": para_id,
                                    "content": paragraph,
                                    "chapter_num": chapter_num,
                                    "section_num": section_num,
                                    "paragraph_idx": para_idx + 1
                                })
                                
                                # 分割句子
                                sentences = self._split_into_sentences(paragraph)
                                for sent_idx, sentence in enumerate(sentences):
                                    # 添加句子信息
                                    sent_id = f"ch{chapter_num}-sec{section_num.split('.')[1]}-p{para_idx+1}-s{sent_idx+1}"
                                    self.sentences.append({
                                        "id": sent_id,
                                        "content": sentence,
                                        "paragraph_id": para_id,
                                        "sentence_idx": sent_idx + 1
                                    })

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        return [para.strip() for para in text.split("\n\n") if para.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用简单的句子分割规则（句号、问号、感叹号后接空格或换行）
        sentences = re.split(r'([。！？\.\?!]\s*)', text)
        # 合并句子和分隔符
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i+1])
            else:
                combined_sentences.append(sentences[i])
        # 过滤空句子
        return [sent.strip() for sent in combined_sentences if sent.strip()]

    def build_node_tree(self) -> None:
        """构建节点树结构"""
        # 先添加章节节点
        for chapter in self.chapters:
            node_id = f"chapter-{chapter['number']}"
            anchor = self._generate_anchor(chapter['title'], node_id)
            
            # 查找父节点
            parent_id = None
            if chapter['level'] > 1:
                # 对于子章节，父章节编号是去掉最后一级
                parent_num_parts = chapter['number'].split(".")[:-1]
                parent_num = ".".join(parent_num_parts)
                parent_id = f"chapter-{parent_num}"
            
            node = {
                "node_id": node_id,
                "type": "chapter",
                "content": chapter['title'],
                "number": chapter['number'],
                "parent_id": parent_id,
                "anchor": anchor
            }
            self.nodes.append(node)
            
            # 如果是根节点（一级章节）
            if parent_id is None:
                self.root_nodes.append(node_id)
        
        # 添加段落节点
        for paragraph in self.paragraphs:
            node_id = f"paragraph-{paragraph['id']}"
            anchor = self._generate_anchor(paragraph['content'][:30], node_id)
            
            # 查找父章节节点
            section_num = paragraph['section_num']
            parent_id = f"chapter-{section_num}"
            
            node = {
                "node_id": node_id,
                "type": "paragraph",
                "content": paragraph['content'],
                "number": f"{section_num}-p{paragraph['paragraph_idx']}",
                "parent_id": parent_id,
                "anchor": anchor
            }
            self.nodes.append(node)
        
        # 添加句子节点
        for sentence in self.sentences:
            node_id = f"sentence-{sentence['id']}"
            anchor = self._generate_anchor(sentence['content'][:20], node_id)
            
            # 查找父段落节点
            parent_id = f"paragraph-{sentence['paragraph_id']}"
            
            node = {
                "node_id": node_id,
                "type": "sentence",
                "content": sentence['content'],
                "number": f"{sentence['paragraph_id']}-s{sentence['sentence_idx']}",
                "parent_id": parent_id,
                "anchor": anchor
            }
            self.nodes.append(node)

    def _generate_anchor(self, text: str, identifier: str) -> str:
        """生成锚点字符串"""
        # 对文本和标识符进行哈希，生成唯一锚点
        hash_obj = hashlib.md5(f"{text}{identifier}".encode('utf-8'))
        hash_str = hash_obj.hexdigest()[:4]
        
        # 提取段内序号（从标识符中）
        match = re.search(r'-(p|s)(\d+)', identifier)
        if match:
            type_prefix = match.group(1)
            index = match.group(2)
            return f"anc:{type_prefix}{index}-{hash_str}"
        
        # 如果没有段内序号，使用默认格式
        return f"anc:id-{hash_str}"

    def print_node_statistics(self) -> None:
        """打印节点统计信息"""
        chapter_count = sum(1 for node in self.nodes if node['type'] == 'chapter')
        para_count = sum(1 for node in self.nodes if node['type'] == 'paragraph')
        sent_count = sum(1 for node in self.nodes if node['type'] == 'sentence')
        
        print("=== 节点统计信息 ===")
        print(f"总节点数: {len(self.nodes)}")
        print(f"章节节点数: {chapter_count}")
        print(f"段落节点数: {para_count}")
        print(f"句子节点数: {sent_count}")
        print("==================")

    def print_node_tree(self, max_depth: int = 3) -> None:
        """打印节点树结构"""
        print("\n=== 节点树结构 ===")
        
        # 按类型和编号排序节点
        sorted_nodes = sorted(self.nodes, key=lambda x: (
            x['type'] != 'chapter',  # 先章节，再段落，最后句子
            x['number']
        ))
        
        # 创建节点映射，便于查找父节点
        node_map = {node['node_id']: node for node in sorted_nodes}
        
        # 打印树结构
        for node_id in self.root_nodes:
            self._print_node(node_map[node_id], node_map, 0, max_depth)
        
        print("================")

    def _print_node(self, node: Dict, node_map: Dict, depth: int, max_depth: int) -> None:
        """递归打印节点"""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        node_info = f"{indent}- {node['type']}: {node['number']} - {node['content'][:30]}..."
        print(node_info)
        
        # 查找子节点并打印
        child_nodes = [n for n in node_map.values() if n['parent_id'] == node['node_id']]
        for child in sorted(child_nodes, key=lambda x: x['number']):
            self._print_node(child, node_map, depth + 1, max_depth)

    def print_sample_anchors(self, count: int = 5) -> List[Dict]:
        """打印示例锚点"""
        print("\n=== 示例锚点 ===")
        
        # 随机选择一些节点作为示例
        sample_nodes = []
        for node_type in ['chapter', 'paragraph', 'sentence']:
            type_nodes = [node for node in self.nodes if node['type'] == node_type]
            if type_nodes:
                # 每个类型至少选一个
                sample_count = max(1, count // 3)
                sample_nodes.extend(type_nodes[:sample_count])
        
        # 打印示例锚点
        result_anchors = []
        for node in sample_nodes[:count]:
            anchor_info = {
                "node_id": node['node_id'],
                "type": node['type'],
                "anchor": node['anchor'],
                "preview": node['content'][:50] + ("..." if len(node['content']) > 50 else "")
            }
            result_anchors.append(anchor_info)
            print(f"锚点: {anchor_info['anchor']} | 类型: {anchor_info['type']} | 内容预览: {anchor_info['preview']}")
        
        print("==============")
        return result_anchors

    def save_results(self, output_dir: str = "result") -> None:
        """保存结果到文件"""
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存节点信息
        nodes_file = os.path.join(output_dir, "hierarchical_nodes.txt")
        with open(nodes_file, 'w', encoding='utf-8') as f:
            f.write("=== 层级索引节点信息 ===\n\n")
            for node in sorted(self.nodes, key=lambda x: (x['type'] != 'chapter', x['number'])):
                f.write(f"节点ID: {node['node_id']}\n")
                f.write(f"类型: {node['type']}\n")
                f.write(f"编号: {node['number']}\n")
                f.write(f"父节点ID: {node['parent_id']}\n")
                f.write(f"锚点: {node['anchor']}\n")
                f.write(f"内容: {node['content'][:100]}{'...' if len(node['content']) > 100 else ''}\n")
                f.write("-" * 50 + "\n")
        
        # 保存锚点映射
        anchors_file = os.path.join(output_dir, "anchor_mapping.txt")
        with open(anchors_file, 'w', encoding='utf-8') as f:
            f.write("=== 锚点映射表 ===\n\n")
            for node in self.nodes:
                f.write(f"{node['anchor']} -> {node['node_id']} ({node['type']}: {node['content'][:30]}{'...' if len(node['content']) > 30 else ''})\n")
        
        print(f"\n结果已保存到 {output_dir} 目录")

# 演示代码
if __name__ == "__main__":
    # 创建层级索引构建器实例
    builder = HierarchicalIndexBuilder()
    
    # 加载示例文档
    print("正在加载示例文档...")
    builder.load_sample_document()
    
    # 解析文档
    print("正在解析文档...")
    builder.parse_document()
    
    # 构建节点树
    print("正在构建节点树...")
    builder.build_node_tree()
    
    # 打印节点统计信息
    builder.print_node_statistics()
    
    # 打印节点树结构
    builder.print_node_tree(max_depth=3)
    
    # 打印示例锚点
    builder.print_sample_anchors(count=5)
    
    # 保存结果到文件
    builder.save_results()
    
    print("\n层级索引构建与锚点生成完成！")