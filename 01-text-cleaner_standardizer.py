#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统文本清洗与标准化工具

该脚本展示了RAG系统中常用的文本清洗与标准化技术，用于处理以下问题：
1. UTF-8/BGK混用产生的乱码修复
2. BOM头残留去除
3. 断行错误处理
4. 页眉/页脚/引用干扰过滤
5. 脚注/引用错位修复
6. 表格ASCII乱码处理
"""

import os
import re
import ftfy
import pandas as pd
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import StringIO

class TextCleanerStandardizer:
    """文本清洗与标准化类
    负责对输入文本进行一系列清洗和标准化处理，确保文本质量适合RAG系统使用。
    """
    
    def __init__(self):
        """初始化文本清洗器
        设置各种用于文本处理的正则表达式模式。
        """
        # 定义用于检测和过滤页眉页脚的正则表达式
        self.header_footer_patterns = [
            re.compile(r'^文档标题：.*$'),  # 匹配文档标题
            re.compile(r'^第\d+页，共\d+页$'),  # 匹配页码信息
            re.compile(r'^页脚：.*$'),  # 匹配页脚标记
            re.compile(r'^引用：.*$')  # 匹配引用标记
        ]
        
        # 定义用于检测脚注的正则表达式
        self.footnote_pattern = r'^\[\d+\]\s+.*$'
        
        # 定义用于检测ASCII表格的正则表达式（已编译）
        self.table_pattern = re.compile(r'\+[-+]+\+.*\+[-+]+\+')
    
    def fix_mixed_encoding(self, text):
        """使用ftfy修复UTF-8/BGK混用产生的乱码
        
        参数:
            text: 输入文本
        返回:
            修复编码后的文本
        """
        print(f"  - 修复混合编码问题...")
        return ftfy.fix_text(text)
    
    def remove_bom(self, text):
        """移除UTF-8 BOM头
        
        参数:
            text: 输入文本
        返回:
            移除BOM头后的文本
        """
        if text.startswith('\ufeff'):
            print(f"  - 检测到并移除UTF-8 BOM头")
            return text[1:]
        return text
    
    def fix_line_breaks(self, text):
        """使用pdfminer.six的行拼接方法修复断行错误
        
        参数:
            text: 输入文本
        返回:
            修复断行后的文本
        """
        original_line_count = len(text.split('\n'))
        
        # 使用pdfminer的LAParams进行行拼接
        # 这里我们模拟pdfminer的行拼接逻辑
        lines = text.split('\n')
        fixed_lines = []
        current_line = ""
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                if current_line:
                    fixed_lines.append(current_line)
                    current_line = ""
                fixed_lines.append("")
            else:
                # 简单的行拼接逻辑：如果行不以句号、问号、感叹号结尾，则认为是断行
                if current_line and not current_line.endswith((".", "?", "!", ";", ":", "。", "？", "！")):
                    current_line += " " + stripped_line
                else:
                    if current_line:
                        fixed_lines.append(current_line)
                    current_line = stripped_line
        
        if current_line:
            fixed_lines.append(current_line)
        
        result = '\n'.join(fixed_lines)
        new_line_count = len(result.split('\n'))
        
        # 计算修复的断行数量
        fixed_breaks = original_line_count - new_line_count
        if fixed_breaks > 0:
            print(f"  - 修复断行错误，合并了 {fixed_breaks} 行文本")
        
        return result
    
    def filter_header_footer(self, text):
        """过滤页眉、页脚和引用干扰
        
        参数:
            text: 输入文本
        返回:
            过滤页眉页脚后的文本
        """
        lines = text.split('\n')
        filtered_lines = []
        header_footer_count = 0
        
        for line in lines:
            # 检查是否匹配任何页眉页脚模式
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if pattern.match(line.strip()):
                    is_header_footer = True
                    header_footer_count += 1
                    break
            
            # 如果不是页眉页脚，则保留
            if not is_header_footer:
                filtered_lines.append(line)
        
        if header_footer_count > 0:
            print(f"  - 过滤了 {header_footer_count} 行页眉/页脚/引用信息")
        
        return '\n'.join(filtered_lines)
    
    def fix_footnote_position(self, text):
        """修复脚注/引用错位问题
        
        参数:
            text: 输入文本
        返回:
            修复脚注位置后的文本
        """
        lines = text.split('\n')
        content_lines = []
        footnotes = []
        
        # 分离正文和脚注
        for line in lines:
            stripped_line = line.strip()
            if re.match(self.footnote_pattern, stripped_line):
                footnotes.append(stripped_line)
            else:
                content_lines.append(line)
        
        if footnotes:
            print(f"  - 发现并修复了 {len(footnotes)} 个脚注的位置")
            # 重新组合文本，将脚注重新放置在文档末尾
            result = '\n'.join(content_lines)
            result += '\n\n--- 脚注 ---\n' + '\n'.join(footnotes)
            return result
        
        return text
    
    def fix_table_ascii(self, text):
        """修复表格ASCII乱码，保留并优化表格结构
        
        参数:
            text: 输入文本
        返回:
            修复表格后的文本
        """
        # 检查是否包含表格特征字符
        if '+' not in text or '|' not in text or '-' not in text:
            return text
        
        try:
            lines = text.split('\n')
            table_found = False
            table_lines = []
            in_table = False
            
            # 简单检测表格行
            for line in lines:
                stripped_line = line.strip()
                # 检测表格开始行（包含+和-的行）
                if ('+' in stripped_line and '-' in stripped_line) or (
                   stripped_line.startswith('+') and stripped_line.endswith('+')):
                    in_table = True
                    table_found = True
                # 检测表格结束（当在表格中遇到空行时，认为表格结束）
                elif in_table and not stripped_line:
                    in_table = False
                
                # 收集表格行
                if in_table:
                    table_lines.append(line)
            
            # 如果找到了表格，只修复编码问题，保留原始表格结构
            if table_found and table_lines:
                print(f"  - 发现并修复ASCII表格，共 {len(table_lines)} 行")
                # 修复表格行中的编码问题
                fixed_table_lines = []
                for line in table_lines:
                    # 只修复单元格内容中的编码，保留表格结构字符
                    parts = re.split(r'([|+\-])', line)
                    fixed_parts = []
                    for part in parts:
                        # 保留表格结构字符不变，只修复内容部分
                        if part in ['|', '+', '-']:
                            fixed_parts.append(part)
                        else:
                            fixed_parts.append(ftfy.fix_text(part))
                    fixed_table_lines.append(''.join(fixed_parts))
                
                # 重建文本，将修复后的表格放回原位置
                new_lines = []
                in_table_section = False
                table_index = 0
                
                for line in lines:
                    stripped_line = line.strip()
                    # 检测表格开始
                    if ('+' in stripped_line and '-' in stripped_line) or (
                       stripped_line.startswith('+') and stripped_line.endswith('+')):
                        in_table_section = True
                        # 添加修复后的表格行
                        if table_index < len(fixed_table_lines):
                            new_lines.append(fixed_table_lines[table_index])
                            table_index += 1
                    # 检测表格结束
                    elif in_table_section and not stripped_line:
                        in_table_section = True
                        # 当遇到空行时，继续添加表格行直到表格结束
                        if table_index < len(fixed_table_lines):
                            new_lines.append(fixed_table_lines[table_index])
                            table_index += 1
                    # 如果不在表格中，保留原行
                    elif not in_table_section:
                        new_lines.append(line)
                    # 如果仍在表格中且有更多表格行要添加
                    elif in_table_section and table_index < len(fixed_table_lines):
                        new_lines.append(fixed_table_lines[table_index])
                        table_index += 1
                
                return '\n'.join(new_lines)
        except Exception as e:
            print(f"解析表格时出错: {e}")
        
        # 如果无法处理表格，返回原文本
        return text

    def process_file(self, file_path):
        """处理单个文件
        
        参数:
            file_path: 输入文件路径
        返回:
            处理后的文本内容，失败则返回None
        """
        # 提取文件名
        filename = os.path.basename(file_path)
        print(f"\n===== 开始处理文件: {filename} =====")
        
        # 读取文件内容
        try:
            print(f"  读取文件内容 (UTF-8编码)...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试其他编码
            try:
                print(f"  UTF-8解码失败，尝试GBK编码...")
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                print(f"  ✓ 成功使用GBK编码读取文件")
            except Exception as e:
                print(f"  ✗ 读取文件失败: {e}")
                return None
        
        # 输出文件基本信息
        print(f"  文件字符数: {len(text)}")
        print(f"  文件行数: {len(text.split('\n'))}")
        
        print(f"\n  开始文本清洗处理:")
        
        # 全新的处理顺序：先处理表格，并且表格处理后不再进行任何可能影响表格结构的处理
        # 1. 移除BOM
        text = self.remove_bom(text)
        
        # 2. 处理表格（独立处理，不与其他文本处理步骤交互）
        text = self.fix_table_ascii(text)
        
        # 3. 分离文本为非表格部分和表格部分
        # 简单检测表格部分
        print(f"  分离文本为表格部分和非表格部分...")
        lines = text.split('\n')
        non_table_lines = []
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped_line = line.strip()
            # 检测表格行
            if ('+' in stripped_line and '-' in stripped_line) or (
               stripped_line.startswith('+') and stripped_line.endswith('+')) or (
               '|' in stripped_line and any(c.isdigit() for c in stripped_line)):
                in_table = True
                table_lines.append(line)
            # 如果在表格中且行不为空，继续认为是表格行
            elif in_table and stripped_line:
                table_lines.append(line)
            # 如果在表格中但遇到空行，表格结束
            elif in_table and not stripped_line:
                in_table = False
                non_table_lines.append(line)  # 保留空行
            else:
                non_table_lines.append(line)
        
        print(f"  ✓ 分离完成: 表格部分 {len(table_lines)} 行, 非表格部分 {len(non_table_lines)} 行")
        
        # 4. 只对非表格部分进行其他处理
        print(f"\n  处理非表格部分:")
        non_table_text = '\n'.join(non_table_lines)
        non_table_text = self.fix_mixed_encoding(non_table_text)
        non_table_text = self.filter_header_footer(non_table_text)
        non_table_text = self.fix_line_breaks(non_table_text)
        non_table_text = self.fix_footnote_position(non_table_text)
        
        # 5. 重新组合文本：非表格部分 + 表格部分
        # 由于我们无法准确知道表格在原始文本中的位置，这里采用一个简化的方法
        # 注意：这种方法可能不完全保留原始文本的结构，但确保表格不受其他处理影响
        print(f"\n  重新组合文本内容...")
        result_text = non_table_text
        if table_lines:
            result_text += '\n\n' + '\n'.join(table_lines)
        
        # 输出处理结果统计
        print(f"\n  处理结果统计:")
        print(f"  - 原始字符数: {len(text)}")
        print(f"  - 处理后字符数: {len(result_text)}")
        print(f"  - 原始行数: {len(text.split('\n'))}")
        print(f"  - 处理后行数: {len(result_text.split('\n'))}")
        
        # 保存处理后的文件
        output_file = file_path.replace('.txt', '_cleaned.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"\n===== 处理完成！结果保存至: {os.path.basename(output_file)} =====")
        return result_text

    def batch_process(self, directory):
        """批量处理目录中的所有txt文件
        
        参数:
            directory: 包含txt文件的目录路径
        """
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return
        
        # 获取目录中的所有txt文件（排除已清理的文件）
        txt_files = [f for f in os.listdir(directory) 
                    if f.endswith('.txt') and not f.endswith('_cleaned.txt')]
        
        print(f"\n===== 开始批量处理目录: {directory} =====")
        print(f"  找到 {len(txt_files)} 个txt文件需要处理")
        
        # 逐个处理文件
        for i, filename in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}]")
            file_path = os.path.join(directory, filename)
            self.process_file(file_path)

if __name__ == "__main__":
    # 创建文本清洗器实例
    print("===== RAG系统文本清洗与标准化工具 =====")
    print("该工具用于清洗和标准化文本文件，处理乱码、断行、表格等问题")
    cleaner = TextCleanerStandardizer()
    
    # 批量处理data目录中的所有txt文件
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    cleaner.batch_process(data_dir)
    
    print("\n===== 所有文件处理完成！=====")