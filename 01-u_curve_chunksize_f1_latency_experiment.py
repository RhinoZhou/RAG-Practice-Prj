#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统中的分块粒度-准确率-延迟倒U曲线实验
验证不同 chunk_size 对 F1 与平均延迟的影响，展示倒U型趋势

主要功能:
- 固定语料与小样本问答
- 测试不同chunk_size（512,768,1024,1536）的表现
- 计算F1分数和平均延迟
- 生成倒U曲线图表和结果数据

"""

# 重新组织整个文件结构，确保所有类都在main函数之前定义
# 1. 首先是所有的导入和配置
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 然后是所有的类定义
@dataclass
class ExperimentConfig:
    """实验配置类
    
    属性:
        chunk_sizes: 要测试的分块大小列表
        overlap_ratio: 分块重叠比例
        corpus_file: 语料文件路径
        qa_file: 问答对文件路径
        output_dir: 结果输出目录
    """
    chunk_sizes: List[int] = None
    overlap_ratio: float = 0.1
    corpus_file: str = "corpus.txt"
    qa_file: str = "qa_pairs.json"
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = [512, 768, 1024, 1536]


@dataclass
class ExperimentResult:
    """实验结果类
    
    属性:
        chunk_size: 分块大小
        num_chunks: 分块数量
        f1_score: F1分数
        avg_latency: 平均延迟
        precision: 精确率
        recall: 召回率
    """
    chunk_size: int
    num_chunks: int
    f1_score: float
    avg_latency: float
    precision: float
    recall: float


class TextChunker:
    """文本分块器类
    
    功能：将长文本按照指定大小和重叠比例分割成多个文本块
    实现细节：优先在句子边界处分割，保证语义完整性
    """
    
    def __init__(self, chunk_size: int, overlap_ratio: float = 0.1):
        """初始化分块器
        
        参数:
            chunk_size: 每个文本块的最大字符数
            overlap_ratio: 相邻块之间的重叠比例
        """
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)  # 计算实际重叠字符数
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分割成块
        
        参数:
            text: 要分块的文本
        
        返回:
            文本块列表
        
        实现说明：
        1. 如果文本长度小于等于chunk_size，直接返回整个文本
        2. 否则，按照chunk_size进行分割，同时尝试在句号处分割以保证语义完整性
        3. 支持相邻块之间的重叠，以防止重要信息被分割在两个块之间
        
        优化提示：当前实现在中文句号处分割，未来可以扩展到更多标点符号
        和更智能的语义分割算法，以进一步提高分块质量
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在句号处分割
            if end < len(text):
                # 寻找最近的句号 - 可以优化：支持更多标点符号和更智能的句子边界检测
                period_pos = text.rfind('。', start, end)
                if period_pos > start + self.chunk_size // 2:  # 确保不会太短
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个块的起始位置（考虑重叠）
            start = end - self.overlap_size
            if start >= len(text):
                break
        
        return chunks


# 对程序进行进一步优化，增强实验的真实度和可观察性
# 1. 改进检索器的相似度计算方法
class SimpleRetriever:
    """简单检索器类
    
    功能：基于词频向量的简单检索器，用于从文本块集合中检索与查询最相关的块
    实现细节：使用词频向量和余弦相似度计算来衡量文本相关性
    
    优化提示：当前实现使用简单的词频向量和余弦相似度，可以改进为使用更先进的
    向量表示方法（如预训练语言模型生成的嵌入向量）以提高检索准确性
    """
    
    def __init__(self, chunks: List[str]):
        """初始化检索器
        
        参数:
            chunks: 要检索的文本块列表
        
        说明：初始化时预先计算所有文本块的词频向量，以提高检索效率
        """
        self.chunks = chunks
        # 预先计算每个块的词频向量，避免重复计算
        self.chunk_vectors = [self._text_to_vector(chunk) for chunk in chunks]
    
    def _text_to_vector(self, text: str) -> Dict[str, int]:
        """将文本转换为词频向量
        
        参数:
            text: 输入文本
        
        返回:
            词频向量（词汇到频率的映射）
        
        实现说明：
        1. 使用正则表达式提取文本中的词汇
        2. 过滤停用词和长度小于2的词
        3. 统计每个词的出现频率
        
        优化提示：可以使用更先进的分词工具（如jieba）来提高中文分词准确性
        """
        # 中文停用词列表
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 分词并过滤停用词
        words = re.findall(r'\w+', text.lower())
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return Counter(filtered_words)
    
    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """计算两个词频向量之间的余弦相似度
        
        参数:
            vec1: 第一个词频向量
            vec2: 第二个词频向量
        
        返回:
            余弦相似度值（范围0-1，值越大表示相似度越高）
        
        实现说明：
        1. 计算两个向量的点积
        2. 计算每个向量的模长
        3. 点积除以模长的乘积得到余弦相似度
        """
        # 获取所有词汇
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        if not all_words:
            return 0.0
        
        # 计算点积和模长
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        norm1 = sum(vec1.get(word, 0) ** 2 for word in all_words) ** 0.5
        norm2 = sum(vec2.get(word, 0) ** 2 for word in all_words) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """检索与查询最相关的文档块
        
        参数:
            query: 用户查询
            top_k: 返回的最大文档块数量
        
        返回:
            (文档块, 相似度)元组列表，按相似度降序排列
        
        实现说明：
        1. 将查询转换为词频向量
        2. 计算查询向量与所有文本块向量的余弦相似度
        3. 添加少量随机噪声，避免完全相同相似度值导致排序不稳定
        4. 按相似度降序排序并返回前top_k个结果
        
        优化提示：对于大规模文本集合，可以使用近似最近邻算法（如FAISS）
        来加速相似度计算和检索过程
        """
        query_vector = self._text_to_vector(query)
        
        # 计算相似度
        similarities = []
        for i, chunk_vector in enumerate(self.chunk_vectors):
            similarity = self._cosine_similarity(query_vector, chunk_vector)
            # 添加一个小的随机噪声，避免完全相同的相似度值导致排序不稳定
            similarity += np.random.normal(0, 0.001)
            similarities.append((self.chunks[i], similarity))
        
        # 按相似度排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# 2. 改进F1计算方法，增加更合理的评估机制
class F1Calculator:
    """F1分数计算器类
    
    功能：计算检索结果的F1分数、精确率和召回率，用于评估检索质量
    实现细节：基于关键词重叠的评估方法，支持停用词过滤
    
    优化提示：当前实现基于简单的关键词重叠，可以改进为使用更复杂的评估指标，
    如BLEU、ROUGE或基于语义理解的评估方法
    """
    
    @staticmethod
    def calculate_f1(predicted_chunks: List[str], ground_truth_chunks: List[str]) -> Tuple[float, float, float]:
        """计算F1分数、精确率和召回率
        
        参数:
            predicted_chunks: 预测的文本块列表（检索结果）
            ground_truth_chunks: 真实的文本块列表（标准答案）
        
        返回:
            (F1分数, 精确率, 召回率)元组，三个值的范围均为0-1
        
        实现说明：
        1. 将预测块和真实块分别拼接成文本
        2. 提取文本中的关键词，并过滤停用词和长度小于2的词
        3. 计算预测关键词集合与真实关键词集合的交集
        4. 精确率 = 交集大小 / 预测关键词集合大小
        5. 召回率 = 交集大小 / 真实关键词集合大小
        6. F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)，作为综合评估指标
        
        优化提示：可以考虑关键词的权重、位置信息或语义相关性，
        以获得更准确的评估结果
        """
        if not ground_truth_chunks:
            return 0.0, 0.0, 0.0
        
        # 简单的基于文本重叠的评估
        predicted_text = " ".join(predicted_chunks)
        ground_truth_text = " ".join(ground_truth_chunks)
        
        # 提取关键词，增加停用词过滤
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        pred_words = set([word for word in re.findall(r'\w+', predicted_text.lower()) 
                         if word not in stop_words and len(word) > 1])
        gt_words = set([word for word in re.findall(r'\w+', ground_truth_text.lower()) 
                       if word not in stop_words and len(word) > 1])
        
        if not gt_words or not pred_words:
            return 0.0, 0.0, 0.0
        
        # 计算交集
        intersection = pred_words & gt_words
        
        precision = len(intersection) / len(pred_words) if pred_words else 0.0
        recall = len(intersection) / len(gt_words) if gt_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall


# 3. 优化主函数，增加更多chunk_size选项以更好地观察倒U曲线
# 修改config定义部分
def main():
    """主函数"""
    print("RAG系统分块粒度-准确率-延迟倒U曲线实验")
    print("="*50)
    
    # 创建实验配置 - 增加更多chunk_size选项以更好地观察倒U曲线
    # 选择更多中等大小的chunk_size值来更清晰地观察峰值
    config = ExperimentConfig(
        chunk_sizes=[128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560],
        overlap_ratio=0.1,
        corpus_file="corpus.txt",
        qa_file="qa_pairs.json",
        output_dir="results"
    )
    
    # 创建实验实例
    experiment = UCurveExperiment(config)
    
    try:
        # 运行实验
        results = experiment.run_experiment()
        
        # 保存结果
        experiment.save_results()
        
        # 绘制图表
        experiment.plot_results()
        
        # 打印总结
        experiment.print_summary()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保语料文件和问答对文件存在")
    except Exception as e:
        print(f"实验过程中出现错误: {e}")


class UCurveExperiment:
    """倒U曲线实验主类
    
    功能：执行分块粒度-准确率-延迟倒U曲线实验，用于评估不同分块大小对检索性能的影响
    实现细节：测试多个分块大小，计算每个大小对应的F1分数、延迟、精确率和召回率
    
    实验目的：验证分块大小与检索性能之间的倒U曲线关系，找出最优分块大小
    """
    
    def __init__(self, config: ExperimentConfig):
        """初始化实验
        
        参数:
            config: 实验配置对象，包含分块大小列表、重叠比例等参数
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        
        # 创建输出目录（如果不存在）
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def load_data(self) -> Tuple[str, List[Dict[str, str]]]:
        """加载语料和问答对数据
        
        返回:
            (语料文本, 问答对列表)元组
        
        说明：从配置文件中指定的路径加载文本语料和问答对数据
        """
        # 加载语料
        with open(self.config.corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read()
        
        # 加载问答对
        with open(self.config.qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        return corpus, qa_pairs
    
    def run_experiment(self) -> List[ExperimentResult]:
        """运行实验的核心方法
        
        返回:
            实验结果列表，每个结果对应一个分块大小的测试结果
        
        实现说明：
        1. 加载语料和问答对数据
        2. 对每个分块大小，执行以下步骤：
           a. 创建TextChunker对象进行文本分块
           b. 创建SimpleRetriever对象用于检索
           c. 对每个问答对，记录检索延迟并计算F1分数
           d. 计算所有问答对的平均指标
        3. 收集每个分块大小的实验结果
        
        实验流程详解：
        - 对于每个分块大小，首先对整个语料进行分块
        - 然后对每个问答对，使用检索器检索相关文本块
        - 记录检索延迟并计算F1分数、精确率和召回率
        - 最后计算所有问答对的平均指标作为该分块大小的实验结果
        
        优化提示：可以考虑添加并行处理来加速大规模语料的实验，
        或者使用更复杂的评估方法来获得更全面的实验结果
        """
        corpus, qa_pairs = self.load_data()
        
        print(f"加载语料: {len(corpus)} 字符")
        print(f"加载问答对: {len(qa_pairs)} 对")
        print("开始实验...")
        print("=" * 80)
        print(f"{'分块大小':<10} {'块数':<10} {'F1分数':<10} {'平均延迟(s)':<12} {'精确率':<10} {'召回率':<10}")
        print("=" * 80)
        
        # 对每个分块大小进行实验
        for chunk_size in self.config.chunk_sizes:
            # 创建分块器，使用当前分块大小和配置的重叠比例
            chunker = TextChunker(chunk_size, self.config.overlap_ratio)
            
            # 分块：将语料分割成指定大小的文本块
            chunks = chunker.chunk_text(corpus)
            num_chunks = len(chunks)
            
            # 创建检索器，使用分好的文本块初始化
            retriever = SimpleRetriever(chunks)
            
            # 用于存储所有问答对的评估指标
            all_f1_scores = []
            all_latencies = []
            all_precisions = []
            all_recalls = []
            
            # 测试每个问答对
            for qa_pair in qa_pairs:
                query = qa_pair['question']
                ground_truth = qa_pair['answer']
                
                # 记录检索开始时间，用于计算延迟
                start_time = time.time()
                
                # 检索相关文本块 - 动态调整top_k值，避免超过块数
                retrieved_chunks_with_scores = retriever.retrieve(query, top_k=min(3, len(chunks)))
                retrieved_chunks = [chunk for chunk, _ in retrieved_chunks_with_scores]
                
                # 计算检索延迟
                end_time = time.time()
                latency = end_time - start_time
                
                # 计算F1分数、精确率和召回率
                f1, precision, recall = F1Calculator.calculate_f1(retrieved_chunks, [ground_truth])
                
                # 存储结果
                all_f1_scores.append(f1)
                all_latencies.append(latency)
                all_precisions.append(precision)
                all_recalls.append(recall)
            
            # 计算平均指标
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
            avg_latency = sum(all_latencies) / len(all_latencies)
            avg_precision = sum(all_precisions) / len(all_precisions)
            avg_recall = sum(all_recalls) / len(all_recalls)
            
            # 创建实验结果对象
            result = ExperimentResult(
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                f1_score=avg_f1,
                avg_latency=avg_latency,
                precision=avg_precision,
                recall=avg_recall
            )
            
            # 添加到结果列表
            self.results.append(result)
            
            # 打印当前分块大小的实验结果
            print(f"{result.chunk_size:<10} {result.num_chunks:<10} {result.f1_score:<10.4f} "
                  f"{result.avg_latency:<12.4f} {result.precision:<10.4f} {result.recall:<10.4f}")
        
        print("=" * 80)
        
        return self.results
    
    def save_results(self) -> None:
        """保存实验结果到文件"""
        # 保存为JSON文件
        results_dir = Path(self.config.output_dir)
        json_path = results_dir / "experiment_results.json"
        
        results_dict = []
        for result in self.results:
            results_dict.append({
                "chunk_size": result.chunk_size,
                "num_chunks": result.num_chunks,
                "f1_score": result.f1_score,
                "avg_latency": result.avg_latency,
                "precision": result.precision,
                "recall": result.recall
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        # 保存为CSV文件
        csv_path = results_dir / "experiment_results.csv"
        df = pd.DataFrame(results_dict)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"实验结果已保存到 {json_path} 和 {csv_path}")
    
    def plot_results(self) -> None:
        """绘制实验结果图表"""
        # 创建图表目录
        results_dir = Path(self.config.output_dir)
        plot_path = results_dir / "experiment_plots.png"
        
        # 提取数据
        chunk_sizes = [result.chunk_size for result in self.results]
        f1_scores = [result.f1_score for result in self.results]
        latencies = [result.avg_latency for result in self.results]
        num_chunks = [result.num_chunks for result in self.results]
        precisions = [result.precision for result in self.results]
        recalls = [result.recall for result in self.results]
        
        # 创建一个包含多个子图的图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制F1分数与分块大小的关系
        axs[0, 0].plot(chunk_sizes, f1_scores, 'o-', color='blue')
        axs[0, 0].set_title('F1分数与分块大小的关系')
        axs[0, 0].set_xlabel('分块大小(chunk_size)')
        axs[0, 0].set_ylabel('F1分数')
        axs[0, 0].grid(True)
        
        # 绘制延迟与分块大小的关系
        axs[0, 1].plot(chunk_sizes, latencies, 'o-', color='red')
        axs[0, 1].set_title('平均延迟与分块大小的关系')
        axs[0, 1].set_xlabel('分块大小(chunk_size)')
        axs[0, 1].set_ylabel('平均延迟(s)')
        axs[0, 1].grid(True)
        
        # 绘制块数与分块大小的关系
        axs[1, 0].plot(chunk_sizes, num_chunks, 'o-', color='green')
        axs[1, 0].set_title('块数与分块大小的关系')
        axs[1, 0].set_xlabel('分块大小(chunk_size)')
        axs[1, 0].set_ylabel('块数')
        axs[1, 0].grid(True)
        
        # 绘制精确率和召回率与分块大小的关系
        axs[1, 1].plot(chunk_sizes, precisions, 'o-', color='purple', label='精确率')
        axs[1, 1].plot(chunk_sizes, recalls, 'o-', color='orange', label='召回率')
        axs[1, 1].set_title('精确率和召回率与分块大小的关系')
        axs[1, 1].set_xlabel('分块大小(chunk_size)')
        axs[1, 1].set_ylabel('比率')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"实验图表已保存到 {plot_path}")
    
    def print_summary(self) -> None:
        """打印实验总结"""
        if not self.results:
            print("没有实验结果可供总结")
            return
        
        # 找到最佳F1分数的结果
        best_f1_result = max(self.results, key=lambda x: x.f1_score)
        
        # 找到最低延迟的结果
        lowest_latency_result = min(self.results, key=lambda x: x.avg_latency)
        
        print("实验总结:")
        print(f"- 最佳F1分数: {best_f1_result.f1_score:.4f} (分块大小: {best_f1_result.chunk_size}, 块数: {best_f1_result.num_chunks})")
        print(f"- 最低延迟: {lowest_latency_result.avg_latency:.4f}s (分块大小: {lowest_latency_result.chunk_size}, 块数: {lowest_latency_result.num_chunks})")


# 3. 最后是主函数和相关初始化
# 创建results目录（确保程序运行时目录存在）
Path("results").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
