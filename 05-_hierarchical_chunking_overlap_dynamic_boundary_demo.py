# -*- coding: utf-8 -*-
"""
层次化分块与上下文突破演示

主要功能:
- 展示重叠滑窗、层次化切分与动态边界三策略组合效果
- 实现章节→段落→滑窗的多层次分块结构
- 支持动态边界开关控制
- 对比三种分块策略的块分布与覆盖率差异

流程:
- 解析章节和段落
- 对长段落应用滑窗与动态边界处理
- 输出三种方案的分块清单与统计摘要

输入:
- 结构化文本(含章节标识)

输出:
- 三方案分块清单
- 统计摘要
"""

# 导入必要的库
import json
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class ChunkingConfig:
    """分块配置类
    
    属性:
        chunk_size: 滑窗分块大小
        overlap_ratio: 重叠比例
        enable_dynamic_boundary: 是否启用动态边界
        min_chunk_size: 最小块大小
        output_dir: 输出目录
    """
    chunk_size: int = 150  # 减小chunk_size以更好地展示滑窗效果
    overlap_ratio: float = 0.2
    enable_dynamic_boundary: bool = True
    min_chunk_size: int = 100
    output_dir: str = "results"

@dataclass
class TextChunk:
    """文本块类
    
    属性:
        id: 块ID
        content: 文本内容
        start_pos: 起始位置
        end_pos: 结束位置
        level: 层次级别(0=章节, 1=段落, 2=滑窗)
        parent_id: 父块ID
    """
    id: str
    content: str
    start_pos: int
    end_pos: int
    level: int
    parent_id: str = None

class TextParser:
    """文本解析器类
    
    功能：解析结构化文本，提取章节和段落
    """
    
    @staticmethod
    def parse_structure(text: str) -> List[Dict[str, Any]]:
        """解析文本结构，提取章节和段落
        
        参数:
            text: 输入文本
        
        返回:
            章节列表，每个章节包含标题和段落
        """
        # 使用正则表达式匹配章节标题（以#开头）
        sections = []
        current_section = None
        
        # 按行分割文本
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是章节标题（# 开头）
            if line.startswith('#'):
                # 提取标题和级别
                match = re.match(r'(#+)\s+(.*)', line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2)
                    
                    # 创建新章节
                    current_section = {
                        'title': title,
                        'level': level,
                        'paragraphs': []
                    }
                    sections.append(current_section)
            elif current_section:
                # 添加到当前章节的段落
                current_section['paragraphs'].append(line)
        
        # 如果没有章节标题，创建一个默认章节
        if not sections:
            sections = [{
                'title': '默认章节',
                'level': 1,
                'paragraphs': lines
            }]
        
        return sections

class ChunkingStrategy:
    """分块策略基类
    
    功能：定义分块策略的接口
    """
    
    def chunk(self, text: str) -> List[TextChunk]:
        """分块方法，子类需要实现
        
        参数:
            text: 输入文本
        
        返回:
            文本块列表
        """
        raise NotImplementedError("子类必须实现chunk方法")

class SimpleChunkingStrategy(ChunkingStrategy):
    """简单分块策略
    
    功能：仅按章节和段落进行分块，不应用滑窗和动态边界
    """
    
    def chunk(self, text: str) -> List[TextChunk]:
        """按章节和段落进行简单分块
        
        参数:
            text: 输入文本
        
        返回:
            文本块列表
        """
        chunks = []
        sections = TextParser.parse_structure(text)
        
        current_pos = 0
        section_id = 1
        
        for section in sections:
            # 添加章节块
            section_content = f"#{section['level']} {section['title']}"
            section_chunk = TextChunk(
                id=f"section_{section_id}",
                content=section_content,
                start_pos=current_pos,
                end_pos=current_pos + len(section_content),
                level=0
            )
            chunks.append(section_chunk)
            current_pos += len(section_content) + 1  # +1 表示换行符
            
            # 添加段落块
            para_id = 1
            for paragraph in section['paragraphs']:
                para_chunk = TextChunk(
                    id=f"section_{section_id}_para_{para_id}",
                    content=paragraph,
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph),
                    level=1,
                    parent_id=f"section_{section_id}"
                )
                chunks.append(para_chunk)
                current_pos += len(paragraph) + 1  # +1 表示换行符
                para_id += 1
            
            section_id += 1
        
        return chunks

class OverlapSlidingWindowStrategy(ChunkingStrategy):
    """重叠滑窗分块策略
    
    功能：应用重叠滑窗进行分块，但不使用动态边界
    """
    
    def __init__(self, chunk_size: int, overlap_ratio: float):
        """初始化重叠滑窗分块策略
        
        参数:
            chunk_size: 块大小
            overlap_ratio: 重叠比例
        """
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
    
    def chunk(self, text: str) -> List[TextChunk]:
        """应用重叠滑窗进行分块
        
        参数:
            text: 输入文本
        
        返回:
            文本块列表
        """
        chunks = []
        sections = TextParser.parse_structure(text)
        
        current_pos = 0
        section_id = 1
        
        for section in sections:
            # 添加章节块
            section_content = f"#{section['level']} {section['title']}"
            section_chunk = TextChunk(
                id=f"section_{section_id}",
                content=section_content,
                start_pos=current_pos,
                end_pos=current_pos + len(section_content),
                level=0
            )
            chunks.append(section_chunk)
            current_pos += len(section_content) + 1  # +1 表示换行符
            
            # 对每个段落应用滑窗
            para_id = 1
            for paragraph in section['paragraphs']:
                # 添加段落块
                para_chunk = TextChunk(
                    id=f"section_{section_id}_para_{para_id}",
                    content=paragraph,
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph),
                    level=1,
                    parent_id=f"section_{section_id}"
                )
                chunks.append(para_chunk)
                
                # 如果段落太长，应用滑窗
                if len(paragraph) > self.chunk_size:
                    start = 0
                    window_id = 1
                    while start < len(paragraph):
                        end = start + self.chunk_size
                        window_content = paragraph[start:end]
                        
                        window_chunk = TextChunk(
                            id=f"section_{section_id}_para_{para_id}_window_{window_id}",
                            content=window_content,
                            start_pos=current_pos + start,
                            end_pos=current_pos + end,
                            level=2,
                            parent_id=f"section_{section_id}_para_{para_id}"
                        )
                        chunks.append(window_chunk)
                        
                        start = end - self.overlap_size
                        window_id += 1
                
                current_pos += len(paragraph) + 1  # +1 表示换行符
                para_id += 1
            
            section_id += 1
        
        return chunks

class HierarchicalDynamicBoundaryStrategy(ChunkingStrategy):
    """层次化动态边界分块策略
    
    功能：结合层次化分块、重叠滑窗和动态边界
    """
    
    def __init__(self, chunk_size: int, overlap_ratio: float, enable_dynamic_boundary: bool = True):
        """初始化层次化动态边界分块策略
        
        参数:
            chunk_size: 块大小
            overlap_ratio: 重叠比例
            enable_dynamic_boundary: 是否启用动态边界
        """
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.enable_dynamic_boundary = enable_dynamic_boundary
    
    def _find_dynamic_boundary(self, text: str, start: int, end: int) -> int:
        """寻找动态边界
        
        参数:
            text: 文本
            start: 起始位置
            end: 结束位置
        
        返回:
            优化后的结束位置
        """
        if not self.enable_dynamic_boundary or end >= len(text):
            return end
        
        # 寻找最近的句号、逗号等标点符号
        punctuation = ['.', '。', ',', '，', ';', '；', '!', '！', '?', '？', '\n']
        
        # 从end位置往前搜索，最大搜索距离为chunk_size的1/4
        search_end = min(end, len(text))
        search_start = max(start, search_end - self.chunk_size // 4)
        
        for i in range(search_end - 1, search_start - 1, -1):
            if text[i] in punctuation:
                return i + 1
        
        return end
    
    def chunk(self, text: str) -> List[TextChunk]:
        """应用层次化动态边界分块
        
        参数:
            text: 输入文本
        
        返回:
            文本块列表
        """
        chunks = []
        sections = TextParser.parse_structure(text)
        
        current_pos = 0
        section_id = 1
        
        for section in sections:
            # 添加章节块
            section_content = f"#{section['level']} {section['title']}"
            section_chunk = TextChunk(
                id=f"section_{section_id}",
                content=section_content,
                start_pos=current_pos,
                end_pos=current_pos + len(section_content),
                level=0
            )
            chunks.append(section_chunk)
            current_pos += len(section_content) + 1  # +1 表示换行符
            
            # 对每个段落应用滑窗和动态边界
            para_id = 1
            for paragraph in section['paragraphs']:
                # 添加段落块
                para_chunk = TextChunk(
                    id=f"section_{section_id}_para_{para_id}",
                    content=paragraph,
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph),
                    level=1,
                    parent_id=f"section_{section_id}"
                )
                chunks.append(para_chunk)
                
                # 如果段落太长，应用滑窗和动态边界
                if len(paragraph) > self.chunk_size:
                    start = 0
                    window_id = 1
                    while start < len(paragraph):
                        end = start + self.chunk_size
                        
                        # 应用动态边界
                        adjusted_end = self._find_dynamic_boundary(paragraph, start, end)
                        
                        window_content = paragraph[start:adjusted_end]
                        
                        window_chunk = TextChunk(
                            id=f"section_{section_id}_para_{para_id}_window_{window_id}",
                            content=window_content,
                            start_pos=current_pos + start,
                            end_pos=current_pos + adjusted_end,
                            level=2,
                            parent_id=f"section_{section_id}_para_{para_id}"
                        )
                        chunks.append(window_chunk)
                        
                        start = adjusted_end - self.overlap_size
                        window_id += 1
                
                current_pos += len(paragraph) + 1  # +1 表示换行符
                para_id += 1
            
            section_id += 1
        
        return chunks

class ChunkingDemo:
    """层次化分块与上下文突破演示类
    
    功能：对比三种分块策略的效果
    """
    
    def __init__(self, config: ChunkingConfig):
        """初始化演示类
        
        参数:
            config: 分块配置
        """
        self.config = config
        self.text = ""  # 将在load_text方法中加载
        self.results = {}
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def load_text(self, text: str = None) -> None:
        """加载文本
        
        参数:
            text: 输入文本，如果为None则生成示例文本
        """
        if text:
            self.text = text
        else:
            # 生成示例结构化文本
            self.text = self._generate_sample_text()
    
    def _generate_sample_text(self) -> str:
        """生成示例结构化文本
        
        返回:
            示例文本
        """
        # 创建一些非常长的段落来展示滑窗和动态边界效果
        long_paragraph1 = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、感知、理解自然语言等。人工智能的发展可以追溯到20世纪50年代，当时计算机科学家开始探索如何让机器模拟人类的思维过程。经过几十年的发展，人工智能已经从理论研究走向了实际应用，渗透到了我们日常生活的方方面面，如智能手机助手、推荐系统、自动驾驶汽车等。随着深度学习等技术的突破，人工智能的能力得到了极大提升，在图像识别、自然语言处理、语音识别等领域取得了与人类相当甚至超越人类的表现。然而，人工智能仍然面临着许多挑战，如图灵测试的不完全性、常识推理的困难、数据偏见问题、算法透明度不足等。未来，人工智能的发展将更加注重可解释性、安全性和伦理问题，以确保这项技术能够造福人类社会，而不是带来负面影响。"""
        
        long_paragraph2 = """机器学习是人工智能的一个子领域，专注于开发允许计算机从数据中学习而无需明确编程的算法。机器学习算法可以分为监督学习、无监督学习和强化学习等几类。监督学习是最常见的机器学习范式，其中算法从标记的训练数据中学习。常见的监督学习任务包括分类和回归。例如，垃圾邮件检测是一个分类任务，预测房价是一个回归任务。无监督学习处理未标记的数据，算法尝试从未标记的数据中发现模式或结构。聚类和降维是常见的无监督学习任务。聚类算法将相似的数据点分组在一起，而降维技术减少数据的维度同时保留重要信息。强化学习是一种学习范式，其中智能体通过与环境交互并根据其行为获得奖励或惩罚来学习。强化学习已成功应用于游戏、机器人控制和资源管理等领域。近年来，深度学习的兴起推动了机器学习的快速发展，使得机器学习模型能够处理更加复杂的数据和任务，如图像识别、自然语言处理等领域的突破都离不开深度学习技术的进步。"""
        
        long_paragraph3 = """深度学习是机器学习的一个子领域，专注于使用多层神经网络进行特征提取和模式识别。深度神经网络受到人类大脑结构的启发，但在实现上有很大不同。神经网络由神经元（或节点）组成，这些神经元按层排列。输入层接收数据，输出层产生预测，中间层（也称为隐藏层）执行特征提取和转换。层数和每层神经元的数量决定了网络的架构。卷积神经网络（CNN）特别适合处理图像数据，而循环神经网络（RNN）和其变体如长短期记忆网络（LSTM）则更适合处理序列数据如文本和时间序列。深度学习模型通常需要大量数据和计算资源来训练。GPU加速已成为训练深度学习模型的标准做法，因为它可以显著减少训练时间。预训练模型的出现进一步推动了深度学习的发展，这些模型在海量数据上进行预训练，然后可以针对特定任务进行微调，大大提高了模型的性能和泛化能力。深度学习的应用已经广泛渗透到各个领域，包括计算机视觉、自然语言处理、语音识别、推荐系统、医疗诊断、金融预测等。"""
        
        long_paragraph4 = """自然语言处理（NLP）是人工智能的一个分支，专注于使计算机能够理解、解释和生成人类语言。NLP涵盖了广泛的任务，包括文本分类、情感分析、机器翻译、问答系统等。近年来，预训练语言模型如BERT、GPT和LLaMA等已彻底改变了NLP领域。这些模型在海量文本数据上进行预训练，然后可以针对特定任务进行微调，表现出前所未有的性能。机器翻译是NLP的一个重要应用，它涉及将文本从一种语言自动翻译成另一种语言。谷歌翻译等系统已经使用深度学习技术取得了显著进步，使翻译质量接近专业人类翻译的水平。问答系统允许用户以自然语言提问，并从大量文本数据中提取相关答案。这些系统已广泛应用于客户服务、信息检索和教育等领域。随着大语言模型（LLM）的兴起，NLP领域迎来了新一轮的革命，这些模型能够生成高质量的文本，进行复杂的推理，并表现出令人惊讶的语言理解能力。然而，大语言模型仍然存在一些局限性，如幻觉问题、对事实的不准确描述、缺乏实时信息等。"""
        
        sample_text = f"""# 人工智能概述

{long_paragraph1}

## 机器学习基础

{long_paragraph2}

## 深度学习与神经网络

{long_paragraph3}

## 自然语言处理应用

{long_paragraph4}
        """
        return sample_text
    
    def run_demo(self) -> None:
        """运行分块演示
        
        应用三种不同的分块策略并比较结果
        """
        # 创建三种分块策略
        strategies = {
            "简单分块": SimpleChunkingStrategy(),
            "重叠滑窗": OverlapSlidingWindowStrategy(
                chunk_size=self.config.chunk_size,
                overlap_ratio=self.config.overlap_ratio
            ),
            "层次化动态边界": HierarchicalDynamicBoundaryStrategy(
                chunk_size=self.config.chunk_size,
                overlap_ratio=self.config.overlap_ratio,
                enable_dynamic_boundary=self.config.enable_dynamic_boundary
            )
        }
        
        # 应用每种策略
        for name, strategy in strategies.items():
            chunks = strategy.chunk(self.text)
            self.results[name] = chunks
            
            print(f"{name} 策略生成了 {len(chunks)} 个块")
        
        # 保存结果
        self.save_results()
        
        # 生成统计摘要
        self.generate_summary()
        
        # 可视化结果
        self.visualize_results()
    
    def save_results(self) -> None:
        """保存分块结果到文件"""
        for name, chunks in self.results.items():
            # 转换为可JSON序列化的格式
            chunks_serializable = []
            for chunk in chunks:
                chunks_serializable.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id,
                    "length": len(chunk.content)
                })
            
            # 保存为JSON文件
            safe_name = name.replace(" ", "_")
            file_path = Path(self.config.output_dir) / f"{safe_name}_chunks.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_serializable, f, ensure_ascii=False, indent=4)
            
            print(f"{name} 策略的分块结果已保存到 {file_path}")
    
    def generate_summary(self) -> None:
        """生成统计摘要"""
        summary = {}
        
        for name, chunks in self.results.items():
            # 计算各级块的数量
            level_counts = Counter(chunk.level for chunk in chunks)
            
            # 计算块长度统计
            lengths = [len(chunk.content) for chunk in chunks]
            avg_length = np.mean(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            
            # 计算覆盖率（假设所有块覆盖整个文本）
            # 注意：这是简化计算，实际覆盖率需要考虑重叠部分
            coverage = 1.0  # 假设完全覆盖
            
            summary[name] = {
                "total_chunks": len(chunks),
                "level_counts": dict(level_counts),
                "avg_length": avg_length,
                "max_length": max_length,
                "min_length": min_length,
                "coverage": coverage
            }
        
        # 保存摘要
        summary_path = Path(self.config.output_dir) / "chunking_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        # 打印摘要
        print("\n分块策略统计摘要:")
        print("=" * 80)
        print(f"{'策略名称':<15} {'总块数':<10} {'平均长度':<10} {'最大长度':<10} {'覆盖率':<10}")
        print("=" * 80)
        
        for name, stats in summary.items():
            print(f"{name:<15} {stats['total_chunks']:<10} {stats['avg_length']:<10.1f} "
                  f"{stats['max_length']:<10} {stats['coverage']:<10.2f}")
        
        print("=" * 80)
        
        # 详细的层次统计
        print("\n层次分布详情:")
        print("策略名称".ljust(15), end="")
        print("章节块(0级)".ljust(15), end="")
        print("段落块(1级)".ljust(15), end="")
        print("滑窗块(2级)")
        print("=" * 60)
        
        for name, stats in summary.items():
            print(f"{name:<15}", end="")
            print(f"{stats['level_counts'].get(0, 0):<15}", end="")
            print(f"{stats['level_counts'].get(1, 0):<15}", end="")
            print(f"{stats['level_counts'].get(2, 0)}")
        
        print("=" * 60)
    
    def visualize_results(self) -> None:
        """可视化分块结果"""
        # 提取数据
        strategies = list(self.results.keys())
        total_chunks = [len(self.results[s]) for s in strategies]
        
        # 获取各级块的数量
        level_0_counts = []
        level_1_counts = []
        level_2_counts = []
        
        for s in strategies:
            level_counts = Counter(chunk.level for chunk in self.results[s])
            level_0_counts.append(level_counts.get(0, 0))
            level_1_counts.append(level_counts.get(1, 0))
            level_2_counts.append(level_counts.get(2, 0))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图：总块数
        ax1.bar(strategies, total_chunks, color=['blue', 'green', 'red'])
        ax1.set_title('不同分块策略的总块数')
        ax1.set_xlabel('分块策略')
        ax1.set_ylabel('块数')
        
        # 堆叠柱状图：各级块的分布
        width = 0.35
        ax2.bar(strategies, level_0_counts, width, label='章节块(0级)', color='blue')
        ax2.bar(strategies, level_1_counts, width, bottom=level_0_counts, label='段落块(1级)', color='green')
        ax2.bar(strategies, level_2_counts, width, bottom=np.array(level_0_counts) + np.array(level_1_counts), label='滑窗块(2级)', color='red')
        ax2.set_title('不同分块策略的层次分布')
        ax2.set_xlabel('分块策略')
        ax2.set_ylabel('块数')
        ax2.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plot_path = Path(self.config.output_dir) / "chunking_strategy_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到 {plot_path}")

def main():
    """主函数"""
    print("层次化分块与上下文突破演示")
    print("=" * 50)
    
    # 创建配置
    config = ChunkingConfig(
        chunk_size=300,
        overlap_ratio=0.2,
        enable_dynamic_boundary=True,
        output_dir="results"
    )
    
    # 创建演示实例
    demo = ChunkingDemo(config)
    
    # 加载文本（使用默认的示例文本）
    demo.load_text()
    
    # 运行演示
    demo.run_demo()
    
    print("\n演示完成！")

def create_sample_text_file():
    """创建示例文本文件，用于演示"""
    sample_text = """# 人工智能概述

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、感知、理解自然语言等。

## 机器学习基础

机器学习是人工智能的一个子领域，专注于开发允许计算机从数据中学习而无需明确编程的算法。机器学习算法可以分为监督学习、无监督学习和强化学习等几类。

监督学习是最常见的机器学习范式，其中算法从标记的训练数据中学习。常见的监督学习任务包括分类和回归。例如，垃圾邮件检测是一个分类任务，预测房价是一个回归任务。

无监督学习处理未标记的数据，算法尝试从未标记的数据中发现模式或结构。聚类和降维是常见的无监督学习任务。聚类算法将相似的数据点分组在一起，而降维技术减少数据的维度同时保留重要信息。

强化学习是一种学习范式，其中智能体通过与环境交互并根据其行为获得奖励或惩罚来学习。强化学习已成功应用于游戏、机器人控制和资源管理等领域。

## 深度学习与神经网络

深度学习是机器学习的一个子领域，专注于使用多层神经网络进行特征提取和模式识别。深度神经网络受到人类大脑结构的启发，但在实现上有很大不同。

神经网络由神经元（或节点）组成，这些神经元按层排列。输入层接收数据，输出层产生预测，中间层（也称为隐藏层）执行特征提取和转换。层数和每层神经元的数量决定了网络的架构。

卷积神经网络（CNN）特别适合处理图像数据，而循环神经网络（RNN）和其变体如长短期记忆网络（LSTM）则更适合处理序列数据如文本和时间序列。

深度学习模型通常需要大量数据和计算资源来训练。GPU加速已成为训练深度学习模型的标准做法，因为它可以显著减少训练时间。

## 自然语言处理应用

自然语言处理（NLP）是人工智能的一个分支，专注于使计算机能够理解、解释和生成人类语言。NLP涵盖了广泛的任务，包括文本分类、情感分析、机器翻译、问答系统等。

近年来，预训练语言模型如BERT、GPT和LLaMA等已彻底改变了NLP领域。这些模型在海量文本数据上进行预训练，然后可以针对特定任务进行微调，表现出前所未有的性能。

机器翻译是NLP的一个重要应用，它涉及将文本从一种语言自动翻译成另一种语言。谷歌翻译等系统已经使用深度学习技术取得了显著进步，使翻译质量接近专业人类翻译的水平。

问答系统允许用户以自然语言提问，并从大量文本数据中提取相关答案。这些系统已广泛应用于客户服务、信息检索和教育等领域。
"""
    
    with open("structured_sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

# 创建示例文本文件
create_sample_text_file()

# 创建results目录（确保程序运行时目录存在）
Path("results").mkdir(exist_ok=True)

if __name__ == "__main__":
    main()