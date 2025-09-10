#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""嵌入与聚类驱动的边界发现演示

该模块实现了基于向量嵌入和聚类的文本边界发现方法，能够融合文本语义特征和版面特征，
自动识别文本中的语义段落边界。主要应用于RAG系统中的文本分块任务，以提高检索质量和生成效果。

核心功能:
- 句子分割与向量化
- 版面特征提取与模拟
- 文本向量与版面特征向量的加权融合
- 基于聚类的边界检测
- 语义段块生成与保存
- 结果可视化

依赖项:
- numpy, pandas: 数据处理
- scikit-learn: 聚类算法
- matplotlib, seaborn: 可视化
- sentence-transformers (可选): 生成高质量句子向量
"""

import os
import json
import time
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 进度条显示函数
def show_progress(current: int, total: int, desc: str = "Processing", bar_length: int = 30):
    """显示进度条
    
    Args:
        current: 当前进度
        total: 总进度
        desc: 进度描述
        bar_length: 进度条长度
    """
    if total == 0:
        return
    
    progress = current / total
    block = int(round(bar_length * progress))
    text = f"\r{desc}: [{'█' * block}{'-' * (bar_length - block)}] {progress * 100:.1f}% {current}/{total}"
    sys.stdout.write(text)
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# 数据类定义
class SentenceVector:
    """句子向量数据类
    
    存储单个句子及其向量表示和版面特征
    """
    def __init__(self, text: str, vector: np.ndarray, index: int, layout_features: Optional[Dict[str, float]] = None):
        self.text = text          # 原始句子文本
        self.vector = vector      # 句子向量表示
        self.index = index        # 句子索引
        self.layout_features = layout_features or {}  # 版面特征

class FusedVector:
    """融合后的向量数据类
    
    存储文本向量和版面特征向量的融合结果
    """
    def __init__(self, text_vector: np.ndarray, layout_vector: np.ndarray, weight: float = 0.5):
        self.text_vector = text_vector  # 原始文本向量
        self.layout_vector = layout_vector  # 原始版面向量
        self.weight = weight            # 文本向量权重
        self.fused_vector = self._fuse_vectors()  # 融合后的向量
        
    def _fuse_vectors(self) -> np.ndarray:
        """融合文本向量和版面向量
        
        Returns:
            融合后的向量
        """
        # 归一化
        text_norm = self._normalize_vector(self.text_vector)
        layout_norm = self._normalize_vector(self.layout_vector)
        # 加权融合
        return self.weight * text_norm + (1 - self.weight) * layout_norm
        
    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """向量归一化
        
        Args:
            vector: 输入向量
        
        Returns:
            归一化后的单位向量
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

class SemanticSegment:
    """语义段块数据类
    
    存储一个语义段块的完整信息
    """
    def __init__(self, text: str, start_index: int, end_index: int, cluster_id: int,
                 features_used: Dict[str, Any], original_sentences: List[str]):
        self.text = text                      # 段块文本
        self.start_index = start_index        # 起始句子索引
        self.end_index = end_index            # 结束句子索引
        self.cluster_id = cluster_id          # 聚类ID
        self.features_used = features_used    # 使用的特征信息
        self.original_sentences = original_sentences  # 原始句子列表

class SentenceEmbedder:
    """句子嵌入器，用于生成句子向量
    
    负责将文本句子转换为向量表示，支持sentence-transformers模型或随机向量
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """初始化句子嵌入器
        
        Args:
            model_name: 使用的预训练模型名称
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化嵌入模型
        
        尝试加载sentence-transformers模型，如果失败则使用随机向量替代
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("sentence-transformers 库未安装，使用随机向量替代")
            
    def get_sentence_vectors(self, sentences: List[str]) -> List[np.ndarray]:
        """获取句子向量
        
        Args:
            sentences: 句子列表
        
        Returns:
            句子向量列表
        """
        if self.model is not None:
            print(f"正在使用 {self.model_name} 生成句子向量...")
            vectors = []
            for i, sentence in enumerate(sentences):
                show_progress(i + 1, len(sentences), "生成句子向量")
                vectors.append(self.model.encode(sentence))
            return vectors
        else:
            # 如果没有安装sentence-transformers，返回随机向量
            print("使用随机向量替代句子嵌入...")
            return [np.random.rand(384) for _ in sentences]

class LayoutFeatureExtractor:
    """版面特征提取器
    
    负责从文本中提取或生成版面特征，如长度、位置、行高、字体大小等
    """
    def __init__(self):
        """初始化版面特征提取器"""
        pass
        
    def extract_layout_features(self, sentences: List[str], layout_data: Optional[List[Dict[str, float]]] = None) -> List[Dict[str, float]]:
        """提取版面特征
        
        Args:
            sentences: 句子列表
            layout_data: 可选的预定义版面数据
        
        Returns:
            版面特征列表
        """
        if layout_data is not None and len(layout_data) == len(sentences):
            print("使用提供的版面数据...")
            return layout_data
        
        # 如果没有提供版面数据，生成模拟的版面特征
        print("生成模拟版面特征...")
        layout_features = []
        for i, sentence in enumerate(sentences):
            show_progress(i + 1, len(sentences), "提取版面特征")
            # 模拟版面特征：长度、位置、行高、字体大小等
            feature = {
                "length": len(sentence),               # 句子长度
                "position": i / max(1, len(sentences) - 1),  # 相对位置
                "line_height": np.random.uniform(0.8, 1.2),  # 行高
                "font_size": np.random.uniform(0.9, 1.1),   # 字体大小
                "is_title": 1 if i < 2 or (i > 0 and len(sentence) < 50 and sentence.endswith("。")) else 0  # 是否为标题
            }
            layout_features.append(feature)
        
        return layout_features

class VectorFusion:
    """向量融合器
    
    负责融合文本向量和版面特征向量，通过加权策略生成综合特征向量
    """
    def __init__(self):
        """初始化向量融合器
        
        创建StandardScaler用于版面特征归一化
        """
        self.scaler = StandardScaler()
        
    def fuse_vectors(self, text_vectors: List[np.ndarray], layout_features_list: List[Dict[str, float]], 
                    text_weight: float = 0.7) -> List[np.ndarray]:
        """融合文本向量和版面特征向量
        
        Args:
            text_vectors: 文本向量列表
            layout_features_list: 版面特征列表
            text_weight: 文本向量的权重（0-1之间）
        
        Returns:
            融合后的向量列表
        """
        # 确保text_weight在有效范围内
        text_weight = max(0, min(1, text_weight))
        
        # 处理版面特征
        if layout_features_list:
            # 将字典列表转换为DataFrame
            layout_df = pd.DataFrame(layout_features_list)
            # 填充缺失值
            layout_df = layout_df.fillna(0)
            # 标准化
            layout_vectors = self.scaler.fit_transform(layout_df.values)
        else:
            # 如果没有版面特征，返回文本向量
            return text_vectors
        
        # 确保向量数量匹配
        if len(text_vectors) != len(layout_vectors):
            raise ValueError("文本向量和版面特征向量数量不匹配")
        
        # 融合向量
        print(f"融合向量 (文本权重: {text_weight})...")
        fused_vectors = []
        for i, (text_vec, layout_vec) in enumerate(zip(text_vectors, layout_vectors)):
            show_progress(i + 1, len(text_vectors), "融合向量")
            # 归一化
            text_norm = self._normalize_vector(text_vec)
            layout_norm = self._normalize_vector(layout_vec)
            
            # 解决维度不匹配问题：扩展版面特征向量到文本向量维度
            # 通过创建一个与文本向量相同维度的零向量，然后将版面特征向量的值复制到前面的位置
            extended_layout = np.zeros_like(text_norm)
            min_dim = min(len(layout_norm), len(extended_layout))
            extended_layout[:min_dim] = layout_norm[:min_dim]
            
            # 加权融合
            fused = text_weight * text_norm + (1 - text_weight) * extended_layout
            fused_vectors.append(fused)
        
        return fused_vectors
        
    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """向量归一化
        
        Args:
            vector: 输入向量
        
        Returns:
            归一化后的单位向量
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

class ClusterBoundaryDetector:
    """聚类边界检测器
    
    负责对融合向量进行聚类，并基于聚类结果检测文本边界
    """
    def __init__(self):
        """初始化聚类边界检测器"""
        pass
        
    def detect_boundaries(self, vectors: List[np.ndarray], sentences: List[str], 
                         n_clusters: Optional[int] = None, method: str = "kmeans") -> Tuple[List[int], np.ndarray]:
        """检测聚类边界
        
        Args:
            vectors: 向量列表
            sentences: 句子列表（用于确定最优聚类数量）
            n_clusters: 聚类数量，None表示自动确定
            method: 聚类方法，支持"kmeans"和"agglomerative"
        
        Returns:
            边界索引列表和聚类标签数组
        """
        # 确定聚类数量
        if n_clusters is None:
            print("正在确定最优聚类数量...")
            n_clusters = self._determine_optimal_clusters(vectors)
        print(f"使用 {n_clusters} 个聚类")
        
        # 执行聚类
        print(f"使用 {method} 方法进行聚类...")
        if method.lower() == "kmeans":
            cluster_labels = self._kmeans_clustering(vectors, n_clusters)
        elif method.lower() == "agglomerative":
            cluster_labels = self._agglomerative_clustering(vectors, n_clusters)
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        # 检测边界
        boundaries = self._find_boundaries(cluster_labels)
        
        return boundaries, cluster_labels
        
    def _determine_optimal_clusters(self, vectors: List[np.ndarray], max_clusters: int = 10) -> int:
        """使用轮廓系数确定最优聚类数量
        
        轮廓系数取值范围为[-1,1]，值越大表示聚类效果越好
        
        Args:
            vectors: 向量列表
            max_clusters: 最大尝试的聚类数量
        
        Returns:
            最优聚类数量
        """
        if len(vectors) <= 2:
            return 1
        
        # 限制最大聚类数量不超过向量数量的一半
        max_clusters = min(max_clusters, len(vectors) // 2)
        
        if max_clusters <= 1:
            return 1
        
        best_score = -1
        best_n = 2
        
        print(f"尝试 2-{max_clusters} 个聚类...")
        for n in range(2, max_clusters + 1):
            try:
                show_progress(n - 1, max_clusters - 1, "确定最优聚类数量")
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(vectors)
                score = silhouette_score(vectors, labels)
                
                if score > best_score:
                    best_score = score
                    best_n = n
            except:
                continue
        
        print(f"最优聚类数量: {best_n} (轮廓系数: {best_score:.3f})")
        return best_n
        
    @staticmethod
    def _kmeans_clustering(vectors: List[np.ndarray], n_clusters: int) -> np.ndarray:
        """KMeans聚类
        
        Args:
            vectors: 向量列表
            n_clusters: 聚类数量
        
        Returns:
            聚类标签数组
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(vectors)
        
    @staticmethod
    def _agglomerative_clustering(vectors: List[np.ndarray], n_clusters: int) -> np.ndarray:
        """层次聚类
        
        Args:
            vectors: 向量列表
            n_clusters: 聚类数量
        
        Returns:
            聚类标签数组
        """
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return agg_clustering.fit_predict(vectors)
        
    @staticmethod
    def _find_boundaries(cluster_labels: np.ndarray) -> List[int]:
        """查找聚类边界
        
        通过检测相邻句子的聚类标签变化来确定边界位置
        
        Args:
            cluster_labels: 聚类标签数组
        
        Returns:
            边界索引列表
        """
        boundaries = [0]  # 总是从0开始
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                boundaries.append(i)
        
        boundaries.append(len(cluster_labels))  # 总是以最后一个索引结束
        
        return boundaries

class EmbedLayoutFusionClustering:
    """嵌入与版面融合聚类主类
    
    集成所有组件，提供完整的文本处理流程
    """
    def __init__(self):
        """初始化嵌入与版面融合聚类
        
        创建各个组件的实例
        """
        self.embedder = SentenceEmbedder()
        self.layout_extractor = LayoutFeatureExtractor()
        self.vector_fuser = VectorFusion()
        self.boundary_detector = ClusterBoundaryDetector()
        
    def process_text(self, text: str, layout_data: Optional[List[Dict[str, float]]] = None, 
                    text_weight: float = 0.7, n_clusters: Optional[int] = None,
                    cluster_method: str = "kmeans", visualize: bool = False) -> List[Dict[str, Any]]:
        """处理文本，进行嵌入、融合和聚类
        
        完整流程：
        1. 分割句子
        2. 获取句向量
        3. 提取版面特征
        4. 融合向量
        5. 检测边界和聚类
        6. 生成语义段块
        7. 可视化（可选）
        8. 返回结果
        
        Args:
            text: 输入文本
            layout_data: 可选的版面数据
            text_weight: 文本向量权重
            n_clusters: 聚类数量，None表示自动确定
            cluster_method: 聚类方法
            visualize: 是否可视化结果
        
        Returns:
            语义段块列表
        """
        start_time = time.time()
        
        # 1. 分割句子
        sentences = self._split_sentences(text)
        print(f"分割得到 {len(sentences)} 个句子")
        
        if not sentences:
            return []
        
        # 2. 获取句向量
        text_vectors = self.embedder.get_sentence_vectors(sentences)
        print(f"获取了 {len(text_vectors)} 个句子向量")
        
        # 3. 提取版面特征
        layout_features = self.layout_extractor.extract_layout_features(sentences, layout_data)
        print(f"提取了 {len(layout_features)} 个版面特征")
        
        # 4. 融合向量
        fused_vectors = self.vector_fuser.fuse_vectors(text_vectors, layout_features, text_weight)
        print(f"融合得到 {len(fused_vectors)} 个向量")
        
        # 5. 检测边界和聚类
        boundaries, cluster_labels = self.boundary_detector.detect_boundaries(
            fused_vectors, sentences, n_clusters, cluster_method
        )
        print(f"检测到 {len(boundaries) - 1} 个边界")
        
        # 6. 生成语义段块
        segments = self._generate_segments(sentences, boundaries, cluster_labels, text_weight, cluster_method)
        print(f"生成了 {len(segments)} 个语义段块")
        
        # 7. 可视化
        if visualize:
            self._visualize_results(sentences, cluster_labels, boundaries)
        
        # 8. 统计信息
        end_time = time.time()
        print(f"总执行时间: {end_time - start_time:.3f} 秒")
        
        return segments
        
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """分割句子
        
        基于中文标点符号和换行符分割句子
        
        Args:
            text: 输入文本
        
        Returns:
            句子列表
        """
        # 简单的句子分割，实际应用中可以使用更复杂的分割方法
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in ["。", "！", "？", "\n"] and len(current_sentence.strip()) > 0:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if len(current_sentence.strip()) > 0:
            sentences.append(current_sentence.strip())
        
        return sentences
        
    def _generate_segments(self, sentences: List[str], boundaries: List[int], cluster_labels: np.ndarray,
                          text_weight: float, cluster_method: str) -> List[Dict[str, Any]]:
        """生成语义段块
        
        根据边界位置，将相邻且属于同一聚类的句子合并为语义段块
        
        Args:
            sentences: 句子列表
            boundaries: 边界索引列表
            cluster_labels: 聚类标签数组
            text_weight: 文本向量权重
            cluster_method: 聚类方法
        
        Returns:
            语义段块信息列表
        """
        segments = []
        
        print("正在生成语义段块...")
        for i in range(len(boundaries) - 1):
            show_progress(i + 1, len(boundaries) - 1, "生成段块")
            
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            # 获取该段的句子
            segment_sentences = sentences[start_idx:end_idx]
            
            # 合并句子为段文本
            segment_text = "".join(segment_sentences)
            
            # 获取该段的聚类ID（使用第一个句子的聚类ID）
            cluster_id = int(cluster_labels[start_idx]) if start_idx < len(cluster_labels) else 0
            
            # 构建段块信息
            segment_info = {
                "text": segment_text,
                "start_index": start_idx,
                "end_index": end_idx - 1,  # 转换为闭区间
                "cluster_id": cluster_id,
                "features_used": {
                    "text_weight": text_weight,
                    "cluster_method": cluster_method,
                    "sentence_count": len(segment_sentences),
                    "char_count": len(segment_text)
                },
                "original_sentences": segment_sentences
            }
            
            segments.append(segment_info)
        
        return segments
        
    def _visualize_results(self, sentences: List[str], cluster_labels: np.ndarray, boundaries: List[int]):
        """可视化结果
        
        绘制聚类标签变化和边界位置的图表
        
        Args:
            sentences: 句子列表
            cluster_labels: 聚类标签数组
            boundaries: 边界索引列表
        """
        try:
            # 创建画布
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制聚类标签
            ax.plot(range(len(cluster_labels)), cluster_labels, 'o-', label='聚类标签')
            
            # 标记边界
            for boundary in boundaries[1:-1]:  # 排除开始和结束边界
                ax.axvline(x=boundary, color='r', linestyle='--', alpha=0.5, label='边界' if boundary == boundaries[1] else "")
            
            # 设置标题和标签
            ax.set_title('嵌入与聚类驱动的边界发现结果')
            ax.set_xlabel('句子索引')
            ax.set_ylabel('聚类标签')
            
            # 添加图例
            ax.legend()
            
            # 保存图像
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "embed_layout_fusion_clustering_results.png"))
            print("结果可视化已保存到 results/embed_layout_fusion_clustering_results.png")
            
            # 关闭图像
            plt.close()
        except Exception as e:
            print(f"可视化失败: {e}")

class ResultsSaver:
    """结果保存器
    
    负责将处理结果保存为JSON文件
    """
    @staticmethod
    def save_to_json(segments: List[Dict[str, Any]], output_file: str = "results/embed_layout_fusion_clustering_segments.json"):
        """将结果保存为JSON文件
        
        Args:
            segments: 语义段块信息列表
            output_file: 输出文件路径
        
        Returns:
            保存文件的路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 准备保存数据
        save_data = {
            "segments": segments,
            "metadata": {
                "total_segments": len(segments),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0"
            }
        }
        
        # 保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到 {output_file}")
        
        # 返回保存路径
        return output_file

# 主函数
if __name__ == "__main__":
    """主函数，演示嵌入与聚类驱动的边界发现流程
    
    1. 加载示例文本
    2. 初始化处理主类
    3. 处理文本，进行嵌入、融合和聚类
    4. 保存结果到JSON文件
    5. 打印部分结果信息
    """
    # 示例文本
    sample_text = """随着人工智能技术的快速发展，自然语言处理领域取得了显著进展。
深度学习模型，特别是基于Transformer架构的模型，在各种NLP任务中表现出色。

文本分块是RAG（检索增强生成）系统中的关键步骤，直接影响检索质量和生成效果。
传统的分块方法通常基于固定大小或简单的规则，无法充分考虑文本的语义结构。

嵌入与聚类驱动的边界发现方法结合了语义信息和结构特征，能够更准确地识别自然段落边界。
这种方法不仅考虑了文本内容，还可以融合版面特征，如标题、段落间距等信息。

实验结果表明，融合多种特征的分块方法在问答系统和文档摘要任务中均取得了更好的性能。
未来的研究方向包括探索更多类型的特征融合和更高效的聚类算法。"""
    
    # 初始化处理主类
    processor = EmbedLayoutFusionClustering()
    
    # 处理文本
    segments = processor.process_text(
        text=sample_text,
        text_weight=0.7,  # 文本向量权重
        n_clusters=None,  # 自动确定聚类数量
        cluster_method="kmeans",  # 使用KMeans聚类
        visualize=True  # 可视化结果
    )
    
    # 保存结果
    saver = ResultsSaver()
    output_path = saver.save_to_json(segments)
    
    # 打印前3个段块的信息
    print("\n前3个段块信息:")
    for i, segment in enumerate(segments[:3]):
        print(f"\n段块 {i+1}:")
        print(f"聚类ID: {segment['cluster_id']}")
        print(f"范围: 句子 {segment['start_index']}-{segment['end_index']}")
        print(f"特征: {segment['features_used']}")
        print(f"文本预览: {segment['text'][:100]}...")