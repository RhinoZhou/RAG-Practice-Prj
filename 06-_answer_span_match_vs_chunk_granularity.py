# -*- coding: utf-8 -*-
"""
答案跨度匹配度与块粒度关系评估

主要功能:
- 评估问题的答案跨度与块大小匹配度对生成质量的影响
- 基于抽取式proxy（答案片段是否在同一块内）
- 计算同块命中率并与F1/一致性分数对比

流程:
- 为每个问题标注答案片段或关键词集合
- 在不同块方案下计算"同块命中率"
- 与F1/一致性分数进行对比分析

输入:
- 问答标注（答案片段）
- 不同分块方案

输出:
- 匹配度-评分对照表
"""

# 导入必要的库
import json
import re
from typing import List, Dict, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# 定义停用词列表，用于关键词提取时过滤常见但无意义的词汇
STOPWORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class QAPair:
    """问答对数据类
    
    属性:
        question: 问题文本
        answer: 答案文本
        answer_spans: 答案片段列表
        keywords: 关键词集合
    """
    question: str
    answer: str
    answer_spans: List[Dict[str, Any]]
    keywords: Set[str]

@dataclass
class TextChunk:
    """文本块数据类
    
    属性:
        id: 块ID
        content: 文本内容
        start_pos: 起始位置
        end_pos: 结束位置
    """
    id: str
    content: str
    start_pos: int
    end_pos: int

@dataclass
class ChunkingScheme:
    """分块方案数据类
    
    属性:
        name: 方案名称
        chunk_size: 块大小
        overlap_ratio: 重叠比例
    """
    name: str
    chunk_size: int
    overlap_ratio: float = 0.0

@dataclass
class EvaluationResult:
    """评估结果数据类
    
    属性:
        scheme_name: 分块方案名称
        chunk_size: 块大小
        same_chunk_hit_rate: 同块命中率
        f1_score: F1分数
        precision: 精确率
        recall: 召回率
        consistency_score: 一致性分数
        total_qa_pairs: 总问答对数
    """
    scheme_name: str
    chunk_size: int
    same_chunk_hit_rate: float
    f1_score: float
    precision: float
    recall: float
    consistency_score: float
    total_qa_pairs: int

class TextChunker:
    """文本分块器类
    
    功能：将文本分割成不同大小的块
    """
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap_ratio: float = 0.0) -> List[TextChunk]:
        """将文本分割成指定大小的块
        
        参数:
            text: 输入文本
            chunk_size: 块大小
            overlap_ratio: 重叠比例
        
        返回:
            文本块列表
        """
        chunks = []
        if not text:
            return chunks
        
        # 计算重叠大小
        overlap_size = int(chunk_size * overlap_ratio)
        step_size = chunk_size - overlap_size
        
        # 执行分块
        start = 0
        chunk_id = 1
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunk = TextChunk(
                id=f"chunk_{chunk_id}",
                content=chunk_content,
                start_pos=start,
                end_pos=end
            )
            chunks.append(chunk)
            
            # 如果是最后一个块，不需要再移动
            if end >= len(text):
                break
            
            # 移动到下一个块的起始位置
            start += step_size
            chunk_id += 1
        
        return chunks

class AnswerSpanMatcher:
    """答案跨度匹配器类
    
    功能：计算答案片段与文本块的匹配关系
    """
    
    @staticmethod
    def find_answer_spans(question: str, answer: str, text: str) -> List[Dict[str, Any]]:
        """查找答案片段在文本中的位置
        
        参数:
            question: 问题文本
            answer: 答案文本
            text: 完整文本
        
        返回:
            答案片段列表，每个片段包含start和end位置
        """
        spans = []
        
        # 简单实现：基于关键词匹配
        # 提取答案中的主要关键词
        answer_tokens = re.findall(r'\b\w+\b', answer)
        
        # 在文本中查找包含这些关键词的片段
        # 这是一个简化的实现，实际应用中可能需要更复杂的算法
        for token in answer_tokens:
            # 忽略太短的关键词
            if len(token) < 2:
                continue
            
            # 在文本中查找关键词
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            for match in pattern.finditer(text):
                spans.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group()
                })
        
        # 去重和排序
        if spans:
            # 简单去重（基于位置）
            spans = sorted(spans, key=lambda x: (x['start'], x['end']))
            unique_spans = [spans[0]]
            
            for span in spans[1:]:
                last_span = unique_spans[-1]
                if span['start'] > last_span['end']:
                    unique_spans.append(span)
                elif span['end'] > last_span['end']:
                    # 合并重叠或相邻的片段
                    unique_spans[-1]['end'] = span['end']
                    unique_spans[-1]['text'] = text[unique_spans[-1]['start']:span['end']]
            
            return unique_spans
        
        return spans
    
    @staticmethod
    def extract_keywords(answer: str) -> Set[str]:
        """从答案中提取关键词
        
        参数:
            answer: 答案文本
        
        返回:
            关键词集合
        """
        # 使用正则表达式找出所有单词，转换为小写，过滤掉停用词和长度小于等于1的单词
        words = re.findall(r'\b\w+\b', answer)
        
        # 过滤停用词和太短的单词
        keywords = set()
        for word in words:
            if len(word) > 1 and word not in STOPWORDS:
                keywords.add(word.lower())
        
        return keywords
    
    @staticmethod
    def calculate_same_chunk_hit_rate(qa_pairs: List[QAPair], chunks: List[TextChunk]) -> float:
        """计算同块命中率
        
        参数:
            qa_pairs: 问答对列表
            chunks: 文本块列表
        
        返回:
            同块命中率
        """
        if not qa_pairs or not chunks:
            return 0.0
        
        # 计算同块命中率：问答对的所有答案片段完全包含在同一文本块中的比例
        hit_count = 0
        
        for qa in qa_pairs:
            # 找到所有包含答案片段的块
            chunk_ids = set()
            
            for span in qa.answer_spans:
                # 找到包含这个片段的块
                for chunk in chunks:
                    if span['start'] >= chunk.start_pos and span['end'] <= chunk.end_pos:
                        chunk_ids.add(chunk.id)
                        break
            
            # 如果所有答案片段都在同一个块中，或者只有一个片段，则视为命中
            if len(chunk_ids) <= 1:
                hit_count += 1
        
        # 计算命中率
        return hit_count / len(qa_pairs) if qa_pairs else 0.0

class MetricsCalculator:
    """指标计算器类
    
    功能：计算各种评估指标
    """
    
    @staticmethod
    def calculate_f1_score(qa_pairs: List[QAPair], chunks: List[TextChunk]) -> Tuple[float, float, float]:
        """计算F1分数
        
        参数:
            qa_pairs: 问答对列表
            chunks: 文本块列表
        
        返回:
            (F1分数, 精确率, 召回率)
        """
        if not qa_pairs or not chunks:
            return 0.0, 0.0, 0.0
        
        total_precision = 0.0
        total_recall = 0.0
        
        for qa in qa_pairs:
            # 计算这个问答对的精确率和召回率
            # 这里评估的是基于关键词匹配识别相关块的准确性
            
            # 找到所有包含答案关键词的块（系统认为相关的块）
            relevant_chunks = set()
            for chunk in chunks:
                for keyword in qa.keywords:
                    if keyword in chunk.content.lower():
                        relevant_chunks.add(chunk.id)
                        break
            
            # 找出应该相关的块（实际包含答案片段的块）
            ideal_relevant_chunks = set()
            for span in qa.answer_spans:
                for chunk in chunks:
                    if span['start'] >= chunk.start_pos and span['end'] <= chunk.end_pos:
                        ideal_relevant_chunks.add(chunk.id)
                        break
            
            # 计算精确率和召回率
            # 精确率：系统认为相关的块中，实际上相关的比例
            if relevant_chunks:
                precision = len(relevant_chunks.intersection(ideal_relevant_chunks)) / len(relevant_chunks)
            else:
                precision = 0.0
            
            # 召回率：实际上相关的块中，被系统正确识别的比例
            if ideal_relevant_chunks:
                recall = len(relevant_chunks.intersection(ideal_relevant_chunks)) / len(ideal_relevant_chunks)
            else:
                recall = 0.0
            
            total_precision += precision
            total_recall += recall
        
        # 计算平均精确率和召回率
        avg_precision = total_precision / len(qa_pairs)
        avg_recall = total_recall / len(qa_pairs)
        
        # 计算F1分数（精确率和召回率的调和平均值）
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        return f1_score, avg_precision, avg_recall
    
    @staticmethod
    def calculate_consistency_score(qa_pairs: List[QAPair], chunks: List[TextChunk]) -> float:
        """计算一致性分数
        
        参数:
            qa_pairs: 问答对列表
            chunks: 文本块列表
        
        返回:
            一致性分数
        """
        if not qa_pairs or not chunks:
            return 0.0
        
        # 计算一致性分数：衡量答案片段在文本块中的分布集中度
        # 分数越高表示所有答案片段越集中在少数几个块中
        total_score = 0.0
        
        for qa in qa_pairs:
            # 找到所有包含答案片段的块
            chunk_ids = []
            
            for span in qa.answer_spans:
                for chunk in chunks:
                    if span['start'] >= chunk.start_pos and span['end'] <= chunk.end_pos:
                        chunk_ids.append(chunk.id)
                        break
            
            # 计算一致性分数（块ID出现频率越高，一致性越高）
            if chunk_ids:
                # 计算块ID的集中度：使用出现次数最多的块ID的频率作为一致性指标
                chunk_counter = Counter(chunk_ids)
                max_count = max(chunk_counter.values())
                consistency = max_count / len(chunk_ids)
                total_score += consistency
        
        return total_score / len(qa_pairs) if qa_pairs else 0.0

class AnswerSpanMatchEvaluator:
    """答案跨度匹配评估器类
    
    功能：评估答案跨度匹配度与块粒度的关系
    """
    
    def __init__(self, output_dir: str = "results"):
        """初始化评估器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.qa_pairs = []
        self.text = ""
        self.chunking_schemes = []
        self.results = []
        
        # 创建输出目录
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def load_data(self, text: str = None, qa_pairs_data: List[Dict[str, Any]] = None) -> None:
        """加载数据
        
        参数:
            text: 完整文本
            qa_pairs_data: 问答对数据
        """
        # 如果没有提供文本，生成示例文本
        if not text:
            self.text = self._generate_sample_text()
        else:
            self.text = text
        
        # 如果没有提供问答对数据，生成示例数据
        if not qa_pairs_data:
            self.qa_pairs = self._generate_sample_qa_pairs()
        else:
            self.qa_pairs = self._parse_qa_pairs(qa_pairs_data)
    
    def _generate_sample_text(self) -> str:
        """生成示例文本
        
        返回:
            示例文本
        """
        sample_text = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、感知、理解自然语言等。

机器学习是人工智能的一个子领域，专注于开发允许计算机从数据中学习而无需明确编程的算法。机器学习算法可以分为监督学习、无监督学习和强化学习等几类。

深度学习是机器学习的一个子领域，专注于使用多层神经网络进行特征提取和模式识别。深度神经网络受到人类大脑结构的启发，但在实现上有很大不同。

自然语言处理（NLP）是人工智能的一个分支，专注于使计算机能够理解、解释和生成人类语言。NLP涵盖了广泛的任务，包括文本分类、情感分析、机器翻译、问答系统等。

近年来，预训练语言模型如BERT、GPT和LLaMA等已彻底改变了NLP领域。这些模型在海量文本数据上进行预训练，然后可以针对特定任务进行微调，表现出前所未有的性能。
        """
        return sample_text.strip()
    
    def _generate_sample_qa_pairs(self) -> List[QAPair]:
        """生成示例问答对
        
        返回:
            问答对列表
        """
        # 手动创建一些示例问答对
        sample_qa_data = [
            {
                "question": "什么是人工智能？",
                "answer": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
            },
            {
                "question": "机器学习有哪些主要类型？",
                "answer": "机器学习算法可以分为监督学习、无监督学习和强化学习等几类。"
            },
            {
                "question": "深度学习的主要特点是什么？",
                "answer": "深度学习是机器学习的一个子领域，专注于使用多层神经网络进行特征提取和模式识别。"
            },
            {
                "question": "自然语言处理涵盖哪些任务？",
                "answer": "NLP涵盖了广泛的任务，包括文本分类、情感分析、机器翻译、问答系统等。"
            },
            {
                "question": "近年来哪些模型改变了NLP领域？",
                "answer": "近年来，预训练语言模型如BERT、GPT和LLaMA等已彻底改变了NLP领域。"
            }
        ]
        
        return self._parse_qa_pairs(sample_qa_data)
    
    def _parse_qa_pairs(self, qa_pairs_data: List[Dict[str, Any]]) -> List[QAPair]:
        """解析问答对数据
        
        参数:
            qa_pairs_data: 问答对数据
        
        返回:
            问答对列表
        """
        qa_pairs = []
        
        for qa_data in qa_pairs_data:
            question = qa_data.get("question", "")
            answer = qa_data.get("answer", "")
            
            # 查找答案片段
            answer_spans = AnswerSpanMatcher.find_answer_spans(question, answer, self.text)
            
            # 提取关键词
            keywords = AnswerSpanMatcher.extract_keywords(answer)
            
            qa_pair = QAPair(
                question=question,
                answer=answer,
                answer_spans=answer_spans,
                keywords=keywords
            )
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def define_chunking_schemes(self, chunk_sizes: List[int] = None, overlap_ratio: float = 0.0) -> None:
        """定义分块方案
        
        参数:
            chunk_sizes: 块大小列表
            overlap_ratio: 重叠比例
        """
        if not chunk_sizes:
            # 默认使用一系列块大小
            chunk_sizes = [100, 200, 300, 500, 800, 1300, 2100]
        
        self.chunking_schemes = []
        for size in chunk_sizes:
            scheme = ChunkingScheme(
                name=f"块大小_{size}",
                chunk_size=size,
                overlap_ratio=overlap_ratio
            )
            self.chunking_schemes.append(scheme)
    
    def evaluate(self) -> None:
        """执行评估
        
        应用不同的分块方案，计算各项指标
        """
        if not self.qa_pairs or not self.chunking_schemes:
            print("没有加载数据或定义分块方案，无法执行评估。")
            return
        
        self.results = []
        
        for scheme in self.chunking_schemes:
            print(f"评估分块方案: {scheme.name}")
            
            # 应用分块方案
            chunks = TextChunker.chunk_text(self.text, scheme.chunk_size, scheme.overlap_ratio)
            
            # 计算同块命中率
            same_chunk_hit_rate = AnswerSpanMatcher.calculate_same_chunk_hit_rate(self.qa_pairs, chunks)
            
            # 计算F1分数、精确率和召回率
            f1_score, precision, recall = MetricsCalculator.calculate_f1_score(self.qa_pairs, chunks)
            
            # 计算一致性分数
            consistency_score = MetricsCalculator.calculate_consistency_score(self.qa_pairs, chunks)
            
            # 创建评估结果
            result = EvaluationResult(
                scheme_name=scheme.name,
                chunk_size=scheme.chunk_size,
                same_chunk_hit_rate=same_chunk_hit_rate,
                f1_score=f1_score,
                precision=precision,
                recall=recall,
                consistency_score=consistency_score,
                total_qa_pairs=len(self.qa_pairs)
            )
            self.results.append(result)
            
            print(f"  同块命中率: {same_chunk_hit_rate:.4f}")
            print(f"  F1分数: {f1_score:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  一致性分数: {consistency_score:.4f}")
    
    def save_results(self) -> None:
        """保存评估结果
        
        将评估结果保存为JSON和CSV文件
        """
        if not self.results:
            print("没有评估结果可保存。")
            return
        
        # 转换结果为可序列化的格式
        results_serializable = []
        for result in self.results:
            results_serializable.append({
                "scheme_name": result.scheme_name,
                "chunk_size": result.chunk_size,
                "same_chunk_hit_rate": result.same_chunk_hit_rate,
                "f1_score": result.f1_score,
                "precision": result.precision,
                "recall": result.recall,
                "consistency_score": result.consistency_score,
                "total_qa_pairs": result.total_qa_pairs
            })
        
        # 保存为JSON文件
        json_path = Path(self.output_dir) / "answer_span_match_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=4)
        
        # 保存为CSV文件
        df = pd.DataFrame(results_serializable)
        csv_path = Path(self.output_dir) / "answer_span_match_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"评估结果已保存到 {json_path} 和 {csv_path}")
    
    def visualize_results(self) -> None:
        """可视化评估结果
        
        创建图表展示匹配度与评分的关系
        """
        if not self.results:
            print("没有评估结果可可视化。")
            return
        
        # 提取数据
        chunk_sizes = [result.chunk_size for result in self.results]
        same_chunk_hit_rates = [result.same_chunk_hit_rate for result in self.results]
        f1_scores = [result.f1_score for result in self.results]
        consistency_scores = [result.consistency_score for result in self.results]
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 绘制同块命中率和F1分数
        ax1.set_xlabel('块大小')
        ax1.set_ylabel('匹配度/评分', color='tab:blue')
        line1 = ax1.plot(chunk_sizes, same_chunk_hit_rates, 'o-', color='tab:blue', label='同块命中率')
        line2 = ax1.plot(chunk_sizes, f1_scores, 's-', color='tab:red', label='F1分数')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # 创建第二个y轴，绘制一致性分数
        ax2 = ax1.twinx()
        ax2.set_ylabel('一致性分数', color='tab:green')
        line3 = ax2.plot(chunk_sizes, consistency_scores, '^-', color='tab:green', label='一致性分数')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        # 合并图例
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 设置标题
        plt.title('答案跨度匹配度与块粒度关系评估')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plot_path = Path(self.output_dir) / "answer_span_match_vs_chunk_granularity.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到 {plot_path}")
    
    def generate_matching_table(self) -> None:
        """生成匹配度-评分对照表
        
        创建一个表格，展示不同块大小下的匹配度和评分
        """
        if not self.results:
            print("没有评估结果可生成对照表。")
            return
        
        # 提取数据并创建对照表
        print("\n答案跨度匹配度与块粒度关系对照表:")
        print("=" * 90)
        print(f"{'块大小':<10} {'同块命中率':<15} {'F1分数':<10} {'精确率':<10} {'召回率':<10} {'一致性分数':<15}")
        print("=" * 90)
        
        for result in self.results:
            print(f"{result.chunk_size:<10} {result.same_chunk_hit_rate:<15.4f} {result.f1_score:<10.4f} "
                  f"{result.precision:<10.4f} {result.recall:<10.4f} {result.consistency_score:<15.4f}")
        
        print("=" * 90)
        
        # 保存对照表为文本文件
        table_path = Path(self.output_dir) / "answer_span_match_table.txt"
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("答案跨度匹配度与块粒度关系对照表:\n")
            f.write("=" * 90 + "\n")
            f.write(f"{'块大小':<10} {'同块命中率':<15} {'F1分数':<10} {'精确率':<10} {'召回率':<10} {'一致性分数':<15}\n")
            f.write("=" * 90 + "\n")
            
            for result in self.results:
                f.write(f"{result.chunk_size:<10} {result.same_chunk_hit_rate:<15.4f} {result.f1_score:<10.4f} "
                      f"{result.precision:<10.4f} {result.recall:<10.4f} {result.consistency_score:<15.4f}\n")
            
            f.write("=" * 90 + "\n")
        
        print(f"对照表已保存到 {table_path}")

    def run(self) -> None:
        """运行完整的评估流程
        
        包括加载数据、定义分块方案、执行评估、保存结果和可视化
        """
        print("答案跨度匹配度与块粒度关系评估")
        print("=" * 50)
        
        # 加载数据
        print("加载数据...")
        self.load_data()
        print(f"已加载 {len(self.qa_pairs)} 个问答对")
        
        # 定义分块方案
        print("定义分块方案...")
        self.define_chunking_schemes()
        print(f"已定义 {len(self.chunking_schemes)} 个分块方案")
        
        # 执行评估
        print("执行评估...")
        self.evaluate()
        
        # 保存结果
        print("保存结果...")
        self.save_results()
        
        # 生成对照表
        self.generate_matching_table()
        
        # 可视化结果
        print("可视化结果...")
        self.visualize_results()
        
        print("\n评估完成！")

def main():
    """主函数"""
    # 创建评估器实例
    evaluator = AnswerSpanMatchEvaluator(output_dir="results")
    
    # 运行完整的评估流程
    evaluator.run()

if __name__ == "__main__":
    main()