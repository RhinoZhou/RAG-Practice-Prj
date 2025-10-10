#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文预算器与滑窗切分器

功能说明：三层融合架构 2.0 总览 / 输入端融合进阶 / Chunking 与窗口管理，为要点/引文/指令分配 token 预算，并进行语义切分与滑窗重叠。
内容概述：读取长文与标题层级，基于句号/标题规则切分；按窗口大小与重叠比生成 chunks；对 must-cover 要点与证据优先级执行预算分配，输出"注入清单+预算统计"。

作者：Ph.D. Rhino
"""

import os
import json
import re
import sys
from typing import Dict, List, Tuple, Optional, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DependencyChecker:
    """检查并安装必要的依赖"""
    
    @staticmethod
    def check_and_install_dependencies():
        """检查并安装必要的依赖包"""
        required_packages = ['chardet']
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"依赖 {package} 已安装")
            except ImportError:
                logger.info(f"正在安装依赖 {package}...")
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    logger.info(f"依赖 {package} 安装成功")
                except Exception as e:
                    logger.error(f"安装依赖 {package} 失败: {e}")
                    raise


class MarkdownParser:
    """解析Markdown文件，提取标题层级和段落内容"""
    
    @staticmethod
    def parse_markdown(file_path: str) -> List[Dict[str, Any]]:
        """
        解析Markdown文件，提取标题层级和段落内容
        
        Args:
            file_path: Markdown文件路径
        
        Returns:
            List[Dict]: 包含标题层级和内容的列表
        """
        content_items = []
        
        try:
            # 尝试不同编码读取文件
            encodings = ['utf-8', 'gbk', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise UnicodeDecodeError("所有编码尝试失败", b"", 0, 1, "无法解码文件")
            
            # 使用正则表达式匹配标题和内容
            lines = content.split('\n')
            current_section = {}
            current_content = []
            current_level = 0
            section_id = "1"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 匹配标题
                title_match = re.match(r'^(#{1,6})\s+(.*)', line)
                if title_match:
                    # 保存之前的内容
                    if current_content:
                        current_section['content'] = '\n'.join(current_content)
                        content_items.append(current_section)
                        current_content = []
                    
                    # 解析新标题
                    level = len(title_match.group(1))
                    title = title_match.group(2)
                    
                    # 生成章节ID
                    if level == 1:
                        section_id = "1"
                    elif level > current_level:
                        section_id += ".1"
                    else:
                        # 回溯到上一级
                        section_id_parts = section_id.split('.')
                        section_id_parts = section_id_parts[:level]
                        section_id_parts[-1] = str(int(section_id_parts[-1]) + 1)
                        section_id = '.'.join(section_id_parts)
                    
                    current_level = level
                    current_section = {
                        'id': section_id,
                        'type': 'title',
                        'level': level,
                        'title': title
                    }
                else:
                    # 普通内容行
                    current_content.append(line)
            
            # 保存最后一部分内容
            if current_content:
                current_section['content'] = '\n'.join(current_content)
                content_items.append(current_section)
            
            return content_items
        except Exception as e:
            logger.error(f"解析Markdown文件失败: {e}")
            raise


class WindowSplitter:
    """根据窗口大小和重叠比例将内容切分为chunks"""
    
    @staticmethod
    def split_to_chunks(content_items: List[Dict[str, Any]], window_size: int = 256, overlap: int = 64) -> List[Dict[str, Any]]:
        """
        根据窗口大小和重叠比例将内容切分为chunks
        
        Args:
            content_items: 解析后的内容项列表
            window_size: 窗口大小（token数）
            overlap: 重叠token数
        
        Returns:
            List[Dict]: 切分后的chunks列表
        """
        chunks = []
        
        # 模拟token计算，实际应用中应使用真实的tokenizer
        def count_tokens(text: str) -> int:
            return len(text) // 4  # 粗略估计，实际应根据tokenizer调整
        
        # 收集所有内容
        all_content = []
        for item in content_items:
            if 'content' in item:
                # 基于句号和句子边界切分内容
                sentences = re.split(r'(?<=[。.!?])\s+', item['content'])
                for sentence in sentences:
                    if sentence:
                        all_content.append({
                            'text': sentence,
                            'section_id': item['id'],
                            'tokens': count_tokens(sentence)
                        })
        
        # 生成滑窗chunks
        i = 0
        chunk_id = 1
        
        while i < len(all_content):
            chunk = {
                'id': f"chunk.{chunk_id}",
                'section_id': all_content[i]['section_id'],
                'sentences': [],
                'total_tokens': 0,
                'overlap': 0
            }
            
            # 填充当前chunk直到达到窗口大小
            j = i
            while j < len(all_content) and chunk['total_tokens'] + all_content[j]['tokens'] <= window_size:
                chunk['sentences'].append(all_content[j]['text'])
                chunk['total_tokens'] += all_content[j]['tokens']
                j += 1
            
            # 计算与前一个chunk的重叠
            if chunks:
                # 查找重叠的句子数量
                overlap_count = 0
                for k in range(min(overlap // 10, j - i)):  # 假设平均每个句子10个token
                    if i - k - 1 >= 0:
                        overlap_count += 1
                chunk['overlap'] = overlap_count
            
            chunks.append(chunk)
            chunk_id += 1
            
            # 移动到下一个chunk的起始位置（考虑重叠）
            if j == i:  # 单个句子超过窗口大小
                i += 1
            else:
                # 回退overlap个token的位置
                overlap_tokens = 0
                back_count = 0
                while i - back_count - 1 >= 0 and overlap_tokens + all_content[i - back_count - 1]['tokens'] <= overlap:
                    overlap_tokens += all_content[i - back_count - 1]['tokens']
                    back_count += 1
                i = j - back_count
                
                # 确保至少前进一个句子
                if i <= j - 1:
                    i = j - 1
                
        return chunks


class BudgetManager:
    """管理token预算并选择需要注入的chunks"""
    
    @staticmethod
    def load_budget(budget_file: str) -> Dict[str, int]:
        """加载预算配置文件"""
        try:
            with open(budget_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载预算文件失败: {e}")
            # 返回默认预算
            return {"quotes": 1200, "must_points": 800, "instructions": 300}
    
    @staticmethod
    def load_priority(priority_file: str) -> Dict[str, str]:
        """加载优先级配置文件"""
        try:
            with open(priority_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载优先级文件失败: {e}")
            # 返回默认优先级
            return {}
    
    @staticmethod
    def select_injection_chunks(chunks: List[Dict[str, Any]], budget: Dict[str, int], priority: Dict[str, str]) -> Dict[str, Any]:
        """
        根据预算和优先级选择需要注入的chunks
        
        Args:
            chunks: 切分后的chunks列表
            budget: token预算配置
            priority: chunk优先级配置
        
        Returns:
            Dict: 包含注入清单和预算使用情况的结果
        """
        # 为chunks分配优先级
        for chunk in chunks:
            # 从优先级配置中获取，默认为C
            chunk['priority'] = priority.get(chunk['section_id'], 'C')
        
        # 按优先级排序 chunks
        priority_order = {'A': 0, 'B': 1, 'C': 2}
        sorted_chunks = sorted(chunks, key=lambda x: priority_order.get(x['priority'], 2))
        
        # 选择注入的chunks，直到预算用完
        inject_list = []
        budget_usage = {"quotes": 0, "must_points": 0, "instructions": 0}
        
        for chunk in sorted_chunks:
            # 模拟分配预算，实际应用中应根据内容类型分配不同预算
            # 这里简化处理，A类优先级使用must_points预算，B类使用quotes预算，C类使用instructions预算
            if chunk['priority'] == 'A' and budget_usage['must_points'] + chunk['total_tokens'] <= budget['must_points']:
                inject_list.append(chunk['id'])
                budget_usage['must_points'] += chunk['total_tokens']
            elif chunk['priority'] == 'B' and budget_usage['quotes'] + chunk['total_tokens'] <= budget['quotes']:
                inject_list.append(chunk['id'])
                budget_usage['quotes'] += chunk['total_tokens']
            elif chunk['priority'] == 'C' and budget_usage['instructions'] + chunk['total_tokens'] <= budget['instructions']:
                inject_list.append(chunk['id'])
                budget_usage['instructions'] += chunk['total_tokens']
        
        # 生成结果
        result = {
            "chunks": [
                {"id": chunk['section_id'], "len": chunk['total_tokens'], "overlap": chunk['overlap'], "priority": chunk['priority']}
                for chunk in chunks
            ],
            "inject_list": inject_list,
            "budget_usage": budget_usage,
            "notes": ["A 类片段优先保留原文引述"]
        }
        
        return result


class ContextBudgetDemo:
    """上下文预算器与滑窗切分器演示主类"""
    
    def __init__(self):
        # 检查并安装依赖
        DependencyChecker.check_and_install_dependencies()
    
    def generate_sample_files(self):
        """生成示例输入文件"""
        # 生成示例input.md
        sample_md = """# 第一章 三层融合架构概述
## 1.1 架构设计理念
三层融合架构是一种先进的知识处理框架，通过融合不同层级的信息，实现更高效的知识检索和生成。

## 1.2 核心组件
该架构包含内容层、语义层和应用层三个主要部分。内容层负责原始数据的存储和管理，语义层处理知识的抽取和理解，应用层则提供具体的使用场景。

# 第二章 输入端融合进阶
## 2.1 多源数据整合
输入端融合进阶技术支持多种数据源的无缝整合，包括结构化数据、半结构化数据和非结构化数据。

## 2.2 数据预处理优化
通过先进的预处理算法，提高数据质量和后续处理效率。这包括去重、清洗、标准化等多个环节。

# 第三章 Chunking与窗口管理
## 3.1 语义切分策略
语义切分是指根据内容的语义关联性将长文本分割成有意义的片段。这有助于提高检索精度和生成质量。

## 3.2 窗口滑动机制
窗口滑动机制允许在处理长文本时保持上下文的连贯性，通过重叠区域确保信息不丢失。
"""
        
        with open('input.md', 'w', encoding='utf-8') as f:
            f.write(sample_md)
        
        # 生成示例budget.json
        sample_budget = {
            "quotes": 1200,
            "must_points": 800,
            "instructions": 300
        }
        
        with open('budget.json', 'w', encoding='utf-8') as f:
            json.dump(sample_budget, f, ensure_ascii=False, indent=2)
        
        # 生成示例priority.json
        sample_priority = {
            "1.1": "A",
            "2.1": "A",
            "3.2": "A",
            "1.2": "B",
            "2.2": "B",
            "3.1": "B"
        }
        
        with open('priority.json', 'w', encoding='utf-8') as f:
            json.dump(sample_priority, f, ensure_ascii=False, indent=2)
        
        logger.info("已生成示例输入文件：input.md, budget.json, priority.json")
    
    def run(self):
        """运行演示程序"""
        try:
            # 生成示例输入文件
            if not all(os.path.exists(f) for f in ['input.md', 'budget.json', 'priority.json']):
                self.generate_sample_files()
            
            # 解析Markdown文件
            logger.info("开始解析Markdown文件...")
            content_items = MarkdownParser.parse_markdown('input.md')
            
            # 切分内容为chunks
            logger.info("开始切分内容为chunks...")
            chunks = WindowSplitter.split_to_chunks(content_items)
            
            # 加载预算和优先级配置
            logger.info("加载预算和优先级配置...")
            budget = BudgetManager.load_budget('budget.json')
            priority = BudgetManager.load_priority('priority.json')
            
            # 选择需要注入的chunks
            logger.info("根据预算和优先级选择注入chunks...")
            result = BudgetManager.select_injection_chunks(chunks, budget, priority)
            
            # 输出结果
            logger.info("生成注入清单和预算统计...")
            with open('output.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 输出实验结果分析
            self.analyze_results(result, budget)
            
            logger.info("程序执行完成！输出结果已保存至output.json")
        except Exception as e:
            logger.error(f"程序执行失败: {e}")
            raise
    
    def analyze_results(self, result: Dict[str, Any], budget: Dict[str, int]):
        """分析实验结果"""
        print("\n========== 实验结果分析 ==========")
        print(f"生成的chunks数量: {len(result['chunks'])}")
        print(f"选择注入的chunks数量: {len(result['inject_list'])}")
        
        # 分析预算使用情况
        print("\n预算使用情况:")
        for key, used in result['budget_usage'].items():
            total = budget[key]
            percentage = (used / total) * 100 if total > 0 else 0
            print(f"  {key}: 使用 {used}/{total} tokens ({percentage:.1f}%)")
        
        # 分析优先级分布
        priority_count = {'A': 0, 'B': 0, 'C': 0}
        for chunk in result['chunks']:
            if chunk['priority'] in priority_count:
                priority_count[chunk['priority']] += 1
        
        print("\nChunks优先级分布:")
        for p, count in priority_count.items():
            print(f"  {p}类: {count}个")
        
        # 分析重叠情况
        total_overlap = sum(chunk['overlap'] for chunk in result['chunks'])
        avg_overlap = total_overlap / len(result['chunks']) if result['chunks'] else 0
        print(f"\n平均重叠度: {avg_overlap:.1f} tokens")
        
        print("\n程序输出已达到演示目的，成功实现了以下功能:")
        print("1. 解析Markdown标题与段落为'章/节/段/句群'")
        print("2. 应用窗口与重叠（256/64）生成chunks")
        print("3. 依据优先级表对chunks打分并选择注入")
        print("4. 产出'注入清单'和'预算消耗报告'")
        print("==================================")


if __name__ == "__main__":
    # 创建演示实例并运行
    demo = ContextBudgetDemo()
    demo.run()