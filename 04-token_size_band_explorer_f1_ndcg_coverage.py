# -*- coding: utf-8 -*-
"""
RAG系统中的Token尺寸区间探索与任务适配分析
在512-2048 tokens区间进行快速探索，比较F1/nDCG与覆盖率

主要功能:
- 在512-2048 tokens区间内设置小范围网格
- 统一top-k检索参数，固定overlap=15%
- 构建索引、检索评估
- 计算F1分数、nDCG指标和覆盖率
- 输出推荐的token尺寸区间

输入:
- 文本语料(corpus.txt)
- 问答对(qa_pairs.json)

输出:
- 表格/图表数据(size→F1/nDCG/coverage)
- 推荐的token尺寸区间
"""

# 导入必要的库
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter
import matplotlib.pyplot as plt

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class ExplorerConfig:
    """探索器配置类
    
    属性:
        token_sizes: 要测试的token大小列表
        overlap_ratio: 分块重叠比例
        corpus_file: 语料文件路径
        qa_file: 问答对文件路径
        output_dir: 结果输出目录
        top_k: 检索返回的最大文档块数量
    """
    token_sizes: List[int] = None
    overlap_ratio: float = 0.15
    corpus_file: str = "corpus.txt"
    qa_file: str = "qa_pairs.json"
    output_dir: str = "results"
    top_k: int = 3
    
    def __post_init__(self):
        if self.token_sizes is None:
            # 在512-2048区间内设置小范围网格
            self.token_sizes = [512, 768, 1024, 1280, 1536, 1792, 2048]

@dataclass
class ExplorerResult:
    """探索结果类
    
    属性:
        token_size: token大小
        num_chunks: 分块数量
        f1_score: F1分数
        ndcg_score: nDCG分数
        coverage: 覆盖率
        avg_latency: 平均延迟
    """
    token_size: int
    num_chunks: int
    f1_score: float
    ndcg_score: float
    coverage: float
    avg_latency: float

class TextChunker:
    """文本分块器类
    
    功能：将长文本按照指定大小和重叠比例分割成多个文本块
    实现细节：优先在句子边界处分割，保证语义完整性
    """
    
    def __init__(self, chunk_size: int, overlap_ratio: float = 0.15):
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
        
        处理逻辑:
        1. 对于短文本（长度小于等于chunk_size），直接返回原始文本
        2. 对于长文本，使用滑动窗口策略，优先在句号处分割以保持语义完整性
        3. 相邻块之间保持配置的重叠比例
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在句号处分割
            if end < len(text):
                # 寻找最近的句号
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
    


class SimpleRetriever:
    """简单检索器类
    
    功能：基于词频向量的简单检索器，用于从文本块集合中检索与查询最相关的块
    实现细节：使用词频向量和余弦相似度计算来衡量文本相关性
    """
    
    def __init__(self, chunks: List[str]):
        """初始化检索器
        
        参数:
            chunks: 要检索的文本块列表
        """
        self.chunks = chunks
        # 预先计算每个块的词频向量，以提高检索效率
        self.chunk_vectors = [self._text_to_vector(chunk) for chunk in chunks]
        
    def _text_to_vector(self, text: str) -> Dict[str, int]:
        """将文本转换为词频向量
        
        参数:
            text: 输入文本
        
        返回:
            词频向量（词汇到频率的映射）
        
        处理逻辑:
        1. 提取文本中的所有词汇
        2. 过滤停用词和长度小于等于1的词
        3. 统计每个词的频率
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
        
        处理逻辑:
        1. 计算两个向量的点积
        2. 计算两个向量的模长
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
    
    def _text_to_vector(self, text: str) -> Dict[str, int]:
        """将文本转换为词频向量
        
        参数:
            text: 输入文本
        
        返回:
            词频向量（词汇到频率的映射）
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

class MetricsCalculator:
    """评估指标计算器类
    
    功能：计算F1分数、nDCG和覆盖率等评估指标
    """
    
    @staticmethod
    def calculate_f1(predicted_chunks: List[str], ground_truth_chunks: List[str]) -> float:
        """计算F1分数
        
        参数:
            predicted_chunks: 预测的文本块列表（检索结果）
            ground_truth_chunks: 真实的文本块列表（标准答案）
        
        返回:
            F1分数（范围0-1）
        
        处理逻辑:
        1. 将预测文本块和真实文本块合并为字符串
        2. 提取关键词，过滤停用词和长度小于等于1的词
        3. 计算精确率和召回率
        4. 通过精确率和召回率计算F1分数
        """
        if not ground_truth_chunks:
            return 0.0
        
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
            return 0.0
        
        # 计算交集
        intersection = pred_words & gt_words
        
        precision = len(intersection) / len(pred_words) if pred_words else 0.0
        recall = len(intersection) / len(gt_words) if gt_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
        
    @staticmethod
    def calculate_ndcg(retrieved_ranking: List[float], ideal_ranking: List[float]) -> float:
        """计算归一化折损累积增益(nDCG)
        
        参数:
            retrieved_ranking: 检索结果的相关性分数列表
            ideal_ranking: 理想情况下的相关性分数列表
        
        返回:
            nDCG分数（范围0-1）
        
        处理逻辑:
        1. 计算DCG（折损累积增益）
        2. 计算IDCG（理想折损累积增益）
        3. DCG除以IDCG得到nDCG
        """
        # 计算DCG
        dcg = retrieved_ranking[0]  # DCG@1 = rel_1
        for i in range(1, len(retrieved_ranking)):
            dcg += retrieved_ranking[i] / np.log2(i + 1)  # DCG@k = sum_{i=1 to k} rel_i / log2(i+1)
        
        # 计算IDCG
        ideal_sorted = sorted(ideal_ranking, reverse=True)
        idcg = ideal_sorted[0]  # IDCG@1 = rel_1
        for i in range(1, len(ideal_sorted)):
            idcg += ideal_sorted[i] / np.log2(i + 1)  # IDCG@k = sum_{i=1 to k} rel_i / log2(i+1)
        
        # 避免除以零
        if idcg == 0:
            return 0.0
        
        # 计算nDCG
        ndcg = dcg / idcg
        
        return ndcg
    
    @staticmethod
    def calculate_ndcg(retrieved_ranking: List[float], ideal_ranking: List[float]) -> float:
        """计算归一化折损累积增益(nDCG)
        
        参数:
            retrieved_ranking: 检索结果的相关性分数列表
            ideal_ranking: 理想情况下的相关性分数列表
        
        返回:
            nDCG分数（范围0-1）
        """
        # 计算DCG
        dcg = retrieved_ranking[0]  # DCG@1 = rel_1
        for i in range(1, len(retrieved_ranking)):
            dcg += retrieved_ranking[i] / np.log2(i + 1)  # DCG@k = sum_{i=1 to k} rel_i / log2(i+1)
        
        # 计算IDCG
        ideal_sorted = sorted(ideal_ranking, reverse=True)
        idcg = ideal_sorted[0]  # IDCG@1 = rel_1
        for i in range(1, len(ideal_sorted)):
            idcg += ideal_sorted[i] / np.log2(i + 1)  # IDCG@k = sum_{i=1 to k} rel_i / log2(i+1)
        
        # 避免除以零
        if idcg == 0:
            return 0.0
        
        # 计算nDCG
        ndcg = dcg / idcg
        
        return ndcg
    
    @staticmethod
    def calculate_coverage(retrieved_chunks: List[str], all_chunks: List[str]) -> float:
        """计算覆盖率
        
        参数:
            retrieved_chunks: 检索结果的文本块列表
            all_chunks: 所有文本块列表
        
        返回:
            覆盖率（范围0-1）
        
        处理逻辑:
        1. 将检索到的文本块转换为集合以去除重复
        2. 计算检索到的唯一块数量与所有块数量的比值
        """
        if not all_chunks:
            return 0.0
        
        # 计算检索到的唯一块数量
        retrieved_set = set(retrieved_chunks)
        coverage = len(retrieved_set) / len(all_chunks)
        
        return coverage

class TokenSizeExplorer:
    """Token尺寸探索器主类
    
    功能：执行Token尺寸区间探索与任务适配分析，评估不同Token大小对检索性能的影响
    """
    
    def __init__(self, config: ExplorerConfig):
        """初始化探索器
        
        参数:
            config: 探索器配置对象
        """
        self.config = config
        self.results: List[ExplorerResult] = []
        
        # 创建输出目录（如果不存在）
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
    def load_data(self) -> Tuple[str, List[Dict[str, str]]]:
        """加载语料和问答对数据
        
        返回:
            (语料文本, 问答对列表)元组
        
        处理逻辑:
        1. 从配置的文件路径加载语料文本
        2. 从配置的文件路径加载问答对数据
        """
        # 加载语料
        with open(self.config.corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read()
        
        # 加载问答对
        with open(self.config.qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        return corpus, qa_pairs
        
    def run_exploration(self) -> List[ExplorerResult]:
        """运行探索的核心方法
        
        返回:
            探索结果列表
        
        处理逻辑:
        1. 加载语料和问答对数据
        2. 对每个Token大小执行以下步骤:
           a. 创建分块器并对语料进行分块
           b. 创建检索器并检索相关文本块
           c. 计算评估指标(F1分数、nDCG、覆盖率、延迟)
           d. 记录结果
        3. 汇总所有结果并返回
        """
        corpus, qa_pairs = self.load_data()
        
        print(f"加载语料: {len(corpus)} 字符")
        print(f"加载问答对: {len(qa_pairs)} 对")
        print("开始探索Token尺寸区间...")
        print("=" * 100)
        print(f"{'Token尺寸':<12} {'块数':<10} {'F1分数':<10} {'nDCG分数':<10} {'覆盖率':<10} {'平均延迟(s)':<12}")
        print("=" * 100)
        
        # 对每个Token大小进行探索
        for token_size in self.config.token_sizes:
            # 创建分块器，使用当前Token大小和配置的重叠比例
            chunker = TextChunker(token_size, self.config.overlap_ratio)
            
            # 分块：将语料分割成指定大小的文本块
            chunks = chunker.chunk_text(corpus)
            num_chunks = len(chunks)
            
            # 创建检索器，使用分好的文本块初始化
            retriever = SimpleRetriever(chunks)
            
            # 用于存储所有问答对的评估指标
            all_f1_scores = []
            all_ndcg_scores = []
            all_coverages = []
            all_latencies = []
            all_retrieved_chunks = []
            
            # 测试每个问答对
            for qa_pair in qa_pairs:
                query = qa_pair['question']
                ground_truth = qa_pair['answer']
                
                # 记录检索开始时间，用于计算延迟
                start_time = time.time()
                
                # 检索相关文本块
                retrieved_chunks_with_scores = retriever.retrieve(query, top_k=min(self.config.top_k, len(chunks)))
                retrieved_chunks = [chunk for chunk, _ in retrieved_chunks_with_scores]
                retrieved_scores = [score for _, score in retrieved_chunks_with_scores]
                
                # 计算检索延迟
                end_time = time.time()
                latency = end_time - start_time
                
                # 计算F1分数
                f1 = MetricsCalculator.calculate_f1(retrieved_chunks, [ground_truth])
                
                # 计算nDCG分数
                # 简化处理：假设前k个结果都是相关的
                ideal_scores = [1.0] * len(retrieved_scores)
                ndcg = MetricsCalculator.calculate_ndcg(retrieved_scores, ideal_scores)
                
                # 记录检索到的块，用于计算覆盖率
                all_retrieved_chunks.extend(retrieved_chunks)
                
                # 存储结果
                all_f1_scores.append(f1)
                all_ndcg_scores.append(ndcg)
                all_latencies.append(latency)
            
            # 计算平均指标
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
            avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores)
            avg_latency = sum(all_latencies) / len(all_latencies)
            
            # 计算总体覆盖率
            coverage = MetricsCalculator.calculate_coverage(all_retrieved_chunks, chunks)
            
            # 创建探索结果对象
            result = ExplorerResult(
                token_size=token_size,
                num_chunks=num_chunks,
                f1_score=avg_f1,
                ndcg_score=avg_ndcg,
                coverage=coverage,
                avg_latency=avg_latency
            )
            
            # 添加到结果列表
            self.results.append(result)
            
            # 打印当前Token大小的探索结果
            print(f"{result.token_size:<12} {result.num_chunks:<10} {result.f1_score:<10.4f} "
                  f"{result.ndcg_score:<10.4f} {result.coverage:<10.4f} {result.avg_latency:<12.4f}")
        
        print("=" * 100)
        
        return self.results
    
    def save_results(self) -> None:
        """保存探索结果到文件
        
        处理逻辑:
        1. 将探索结果转换为字典列表格式
        2. 保存为JSON文件，便于后续分析和读取
        3. 保存为CSV文件，便于在Excel等工具中查看
        """
        # 保存为JSON文件
        results_dir = Path(self.config.output_dir)
        json_path = results_dir / "token_size_exploration_results.json"
        
        results_dict = []
        for result in self.results:
            results_dict.append({
                "token_size": result.token_size,
                "num_chunks": result.num_chunks,
                "f1_score": result.f1_score,
                "ndcg_score": result.ndcg_score,
                "coverage": result.coverage,
                "avg_latency": result.avg_latency
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        # 保存为CSV文件
        csv_path = results_dir / "token_size_exploration_results.csv"
        df = pd.DataFrame(results_dict)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig确保Excel能正确显示中文
        
        print(f"探索结果已保存到 {json_path} 和 {csv_path}")
    
    def plot_results(self) -> None:
        """绘制探索结果图表
        
        处理逻辑:
        1. 从探索结果中提取需要的数据（Token尺寸、F1分数、nDCG分数、覆盖率、延迟、块数）
        2. 创建2x2网格布局的图表
        3. 绘制四个子图：
           - F1分数与Token大小的关系图
           - nDCG分数与Token大小的关系图
           - 覆盖率与Token大小的关系图
           - 平均延迟与Token大小的关系图
        4. 保存图表为PNG文件
        """
        # 创建图表目录
        results_dir = Path(self.config.output_dir)
        plot_path = results_dir / "token_size_exploration_plots.png"
        
        # 提取数据
        token_sizes = [result.token_size for result in self.results]
        f1_scores = [result.f1_score for result in self.results]
        ndcg_scores = [result.ndcg_score for result in self.results]
        coverages = [result.coverage for result in self.results]
        latencies = [result.avg_latency for result in self.results]
        num_chunks = [result.num_chunks for result in self.results]
        
        # 创建一个包含多个子图的图表
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制F1分数与Token大小的关系
        axs[0, 0].plot(token_sizes, f1_scores, 'o-', color='blue')
        axs[0, 0].set_title('F1分数与Token大小的关系')
        axs[0, 0].set_xlabel('Token大小')
        axs[0, 0].set_ylabel('F1分数')
        axs[0, 0].grid(True)
        
        # 绘制nDCG分数与Token大小的关系
        axs[0, 1].plot(token_sizes, ndcg_scores, 'o-', color='red')
        axs[0, 1].set_title('nDCG分数与Token大小的关系')
        axs[0, 1].set_xlabel('Token大小')
        axs[0, 1].set_ylabel('nDCG分数')
        axs[0, 1].grid(True)
        
        # 绘制覆盖率与Token大小的关系
        axs[1, 0].plot(token_sizes, coverages, 'o-', color='green')
        axs[1, 0].set_title('覆盖率与Token大小的关系')
        axs[1, 0].set_xlabel('Token大小')
        axs[1, 0].set_ylabel('覆盖率')
        axs[1, 0].grid(True)
        
        # 绘制延迟与Token大小的关系
        axs[1, 1].plot(token_sizes, latencies, 'o-', color='purple')
        axs[1, 1].set_title('平均延迟与Token大小的关系')
        axs[1, 1].set_xlabel('Token大小')
        axs[1, 1].set_ylabel('平均延迟(s)')
        axs[1, 1].grid(True)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"探索图表已保存到 {plot_path}")
    
    def recommend_token_size(self) -> Dict[str, Any]:
        """推荐最佳Token尺寸区间
        
        返回:
            推荐的Token尺寸区间信息，包含：
            - 最佳F1分数对应的Token尺寸
            - 最佳nDCG分数对应的Token尺寸
            - 最佳覆盖率对应的Token尺寸
            - 推荐的Token尺寸区间范围
        
        处理逻辑:
        1. 找出F1分数最高的结果
        2. 找出nDCG分数最高的结果
        3. 找出覆盖率最高的结果
        4. 筛选出F1分数在最佳F1的90%以上的所有结果
        5. 确定推荐的Token尺寸区间（从高性能结果中最小到最大的Token尺寸）
        """
        if not self.results:
            return {}
        
        # 找出F1分数最高的结果
        best_f1_result = max(self.results, key=lambda x: x.f1_score)
        
        # 找出nDCG分数最高的结果
        best_ndcg_result = max(self.results, key=lambda x: x.ndcg_score)
        
        # 找出覆盖率最高的结果
        best_coverage_result = max(self.results, key=lambda x: x.coverage)
        
        # 综合考虑，推荐一个平衡的区间
        # 寻找F1分数在最佳F1的90%以上的所有结果
        high_performance_results = [r for r in self.results if r.f1_score >= 0.9 * best_f1_result.f1_score]
        
        # 从这些高性能结果中，找到覆盖范围的最小值和最大值
        if high_performance_results:
            min_token_size = min(high_performance_results, key=lambda x: x.token_size).token_size
            max_token_size = max(high_performance_results, key=lambda x: x.token_size).token_size
        else:
            min_token_size = best_f1_result.token_size
            max_token_size = best_f1_result.token_size
        
        recommendation = {
            "best_f1_token_size": best_f1_result.token_size,
            "best_f1_score": best_f1_result.f1_score,
            "best_ndcg_token_size": best_ndcg_result.token_size,
            "best_ndcg_score": best_ndcg_result.ndcg_score,
            "best_coverage_token_size": best_coverage_result.token_size,
            "best_coverage_score": best_coverage_result.coverage,
            "recommended_token_size_range": f"{min_token_size}-{max_token_size}",
            "min_recommended_token_size": min_token_size,
            "max_recommended_token_size": max_token_size
        }
        
        # 打印推荐结果
        print("Token尺寸推荐结果:")
        print(f"- F1分数最佳Token尺寸: {recommendation['best_f1_token_size']} (F1分数: {recommendation['best_f1_score']:.4f})")
        print(f"- nDCG分数最佳Token尺寸: {recommendation['best_ndcg_token_size']} (nDCG分数: {recommendation['best_ndcg_score']:.4f})")
        print(f"- 覆盖率最佳Token尺寸: {recommendation['best_coverage_token_size']} (覆盖率: {recommendation['best_coverage_score']:.4f})")
        print(f"- 推荐的Token尺寸区间: {recommendation['recommended_token_size_range']}")
        
        return recommendation

    def print_summary(self) -> None:
        """打印探索总结并保存推荐结果
        
        处理逻辑:
        1. 如果没有探索结果，打印提示信息并返回
        2. 获取推荐的Token尺寸区间信息
        3. 将推荐结果保存为JSON文件，便于后续使用
        """
        if not self.results:
            print("没有探索结果可供总结")
            return
        
        # 获取推荐的Token尺寸区间
        recommendation = self.recommend_token_size()
        
        # 保存推荐结果到文件
        results_dir = Path(self.config.output_dir)
        recommendation_path = results_dir / "token_size_recommendation.json"
        
        with open(recommendation_path, 'w', encoding='utf-8') as f:
            json.dump(recommendation, f, ensure_ascii=False, indent=4)
        
        print(f"推荐结果已保存到 {recommendation_path}")

def main():
    """主函数
    
    处理逻辑:
    1. 打印程序标题
    2. 创建探索器配置（设置Token尺寸范围、重叠比例、文件路径等）
    3. 创建TokenSizeExplorer实例
    4. 尝试执行探索流程：
       a. 运行探索
       b. 保存结果
       c. 绘制图表
       d. 打印总结
    5. 处理可能的异常情况（文件不存在等）
    """
    print("Token尺寸区间探索与任务适配分析")
    print("="*50)
    
    # 创建探索器配置
    config = ExplorerConfig(
        token_sizes=[512, 768, 1024, 1280, 1536, 1792, 2048],
        overlap_ratio=0.15,
        corpus_file="corpus.txt",
        qa_file="qa_pairs.json",
        output_dir="results",
        top_k=3
    )
    
    # 创建探索器实例
    explorer = TokenSizeExplorer(config)
    
    try:
        # 运行探索
        results = explorer.run_exploration()
        
        # 保存结果
        explorer.save_results()
        
        # 绘制图表
        explorer.plot_results()
        
        # 打印总结
        explorer.print_summary()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保语料文件和问答对文件存在")
    except Exception as e:
        print(f"探索过程中出现错误: {e}")

# 创建results目录（确保程序运行时目录存在）
Path("results").mkdir(exist_ok=True)

if __name__ == "__main__":
    main()