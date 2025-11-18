#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
文档相关性重排序模块，提供多种排序算法。
本模块实现了文档相关性排序功能，用于优化检索结果，
通过精排阶段的语义匹配提升系统回答质量。主要功能：
1. 标准化的排序器接口设计
2. 多模型支持
3. 批量处理机制
4. 自动依赖管理
5. 异常处理与性能优化
'''

import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from abc import ABC, abstractmethod


# 依赖管理
def setup_dependencies():
    """
    自动检查并安装所需依赖
    """
    try:
        import pip
        
        required_libs = [
            'transformers>=4.36.0',
            'torch>=2.0.0',
            'numpy>=1.24.0',
            'tqdm>=4.65.0'
        ]
        
        print("检查并安装必要的依赖库...")
        for lib in required_libs:
            try:
                # 尝试导入库
                if lib.split('>=')[0].replace('-', '_') == 'transformers':
                    import transformers
                elif lib.split('>=')[0].replace('-', '_') == 'torch':
                    import torch
                elif lib.split('>=')[0].replace('-', '_') == 'numpy':
                    import numpy
                elif lib.split('>=')[0].replace('-', '_') == 'tqdm':
                    import tqdm
            except ImportError:
                print(f"正在安装 {lib}...")
                pip.main(['install', lib])
        
        print("✓ 依赖库安装完成")
    except Exception as e:
        print(f"警告: 依赖安装过程中出现错误: {str(e)}")
        print("请手动安装所需依赖库")


# 初始化依赖
setup_dependencies()


class RelevanceReranker(ABC):
    """
    文档相关性排序器抽象基类，定义了排序器的通用接口
    """
    
    def __init__(self):
        """初始化相关性排序器"""
        self.ranker_name = "Base Relevance Ranker"
        self.is_initialized = False
    
    @abstractmethod
    def reorder_by_relevance(self, query: str, docs: List[str], top_k: int) -> List[str]:
        """
        根据查询文本对文档列表进行相关性排序
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的最相关结果数量
            
        Returns:
            排序后的文档列表（前top_k个最相关的结果）
        """
        pass
    
    def batch_reorder(self, queries: List[str], doc_lists: List[List[str]], top_k: int) -> List[List[str]]:
        """
        批量处理文档相关性排序
        
        Args:
            queries: 查询文本列表
            doc_lists: 每个查询对应的文档列表
            top_k: 返回的最相关结果数量
            
        Returns:
            批量排序后的文档列表
        """
        results = []
        for q, docs in zip(queries, doc_lists):
            try:
                result = self.reorder_by_relevance(q, docs, top_k)
                results.append(result)
            except Exception as e:
                print(f"警告: 批量排序处理出错: {str(e)}")
                # 出错时返回原始文档的前top_k个
                results.append(docs[:min(top_k, len(docs))])
        return results
    
    def check_health(self) -> bool:
        """
        检查排序器的运行状态
        
        Returns:
            排序器是否正常工作
        """
        try:
            test_q = "测试查询"
            test_docs = ["相关内容", "不相关内容", "测试内容"]
            result = self.reorder_by_relevance(test_q, test_docs, 1)
            return len(result) == 1
        except Exception as e:
            print(f"错误: 排序器健康检查失败: {str(e)}")
            return False


class BgeRelevanceReranker(RelevanceReranker):
    """
    BGE相关性排序器实现
    基于BAAI的BGE模型，使用预训练序列分类模型计算查询与文档相关性
    """
    
    def __init__(self, model_path: str = 'BAAI/bge-reranker-base') -> None:
        """
        初始化BGE相关性排序器
        
        Args:
            model_path: 模型路径或Hugging Face模型标识符
        """
        super().__init__()
        self.ranker_name = "BGE Relevance Ranker"
        self.model_path = model_path
        
        # 默认模型路径列表（按优先级排序）
        self.model_paths_candidates = [
            model_path,
            "./models/bge-reranker-base",
            "./model_hub/Xorbits/bge-reranker-base",
            "/root/sunyd/model_hub/Xorbits/bge-reranker-base",
            "BAAI/bge-reranker-base"
        ]
        
        # 模型组件初始化
        self._model = None
        self._tokenizer = None
        self.device = None
        self.initialize_model()
    
    def initialize_model(self) -> bool:
        """
        初始化并加载相关性排序模型，尝试多个路径
        
        Returns:
            是否成功初始化
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            print(f"正在初始化{BgeRelevanceReranker.__name__}...")
            
            # 确定运行设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  计算设备: {self.device}")
            
            # 尝试加载模型
            model_ready = False
            for path in self.model_paths_candidates:
                if not path:  # 跳过空路径
                    continue
                    
                try:
                    print(f"  尝试从路径加载: {path}")
                    
                    # 加载分词器
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # 加载模型
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                        trust_remote_code=True
                    ).to(self.device)
                    
                    # 设置为评估模式
                    self._model.eval()
                    
                    model_ready = True
                    self.is_initialized = True
                    print(f"✓ 模型初始化成功: {path}")
                    break
                    
                except Exception as e:
                    print(f"  初始化失败: {str(e)}")
                    continue
            
            if not model_ready:
                raise Exception("所有候选路径模型加载失败")
                
            return True
            
        except Exception as e:
            print(f"错误: 初始化{BgeRelevanceReranker.__name__}失败: {str(e)}")
            print("请确保模型路径正确且已下载相应文件")
            self.is_initialized = False
            return False
    
    def reorder_by_relevance(self, query: str, docs: List[str], top_k: int) -> List[str]:
        """
        使用BGE模型对文档列表进行相关性排序
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的最相关结果数量
            
        Returns:
            排序后的文档列表（前top_k个最相关的结果）
        """
        if not self.is_initialized:
            print("警告: 排序器未初始化，尝试重新初始化")
            if not self.initialize_model():
                # 初始化失败时返回原始文档的前top_k个
                return docs[:min(top_k, len(docs))]
        
        try:
            # 验证输入
            if not docs:
                return []
                
            top_k = min(top_k, len(docs))  # 确保top_k不超过文档列表长度
            
            # 创建查询-文档对
            import torch
            doc_pairs = [(query, doc) for doc in docs]
            
            # 计时开始
            start_time = time.time()
            
            # 禁用梯度计算以节省内存
            with torch.no_grad():
                # 批量处理，避免内存溢出
                batch_size = 32  # 可根据可用内存调整
                all_scores = []
                
                for i in range(0, len(doc_pairs), batch_size):
                    batch = doc_pairs[i:i + batch_size]
                    
                    # 编码输入
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=512
                    )
                    
                    # 移至计算设备
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    
                    # 获取相关性分数
                    scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
                    all_scores.extend(scores.tolist())
            
            # 计算处理时间
            end_time = time.time()
            print(f"  相关性排序耗时: {end_time - start_time:.4f}秒 (处理 {len(docs)} 个文档)")
            
            # 排序并返回前top_k个最相关的文档
            scores_array = np.array(all_scores)
            top_doc_indices = np.argsort(-scores_array)[:top_k]  # 降序排序
            
            # 返回排序后的文档
            return [docs[i] for i in top_doc_indices]
            
        except Exception as e:
            print(f"错误: 相关性排序处理出错: {str(e)}")
            # 出错时返回原始文档的前top_k个
            return docs[:min(top_k, len(docs))]
    
    def get_relevance_scores(self, query: str, docs: List[str], top_k: int) -> List[Tuple[str, float]]:
        """
        排序文档并返回相关性分数
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的最相关结果数量
            
        Returns:
            包含(文档, 相关性分数)元组的列表，按相关性降序排列
        """
        if not self.is_initialized:
            print("警告: 排序器未初始化，尝试重新初始化")
            if not self.initialize_model():
                # 初始化失败时返回原始文档的前top_k个，分数设为0
                return [(doc, 0.0) for doc in docs[:min(top_k, len(docs))]]
        
        try:
            # 验证输入
            if not docs:
                return []
                
            top_k = min(top_k, len(docs))  # 确保top_k不超过文档列表长度
            
            # 创建查询-文档对
            import torch
            doc_pairs = [(query, doc) for doc in docs]
            
            # 禁用梯度计算以节省内存
            with torch.no_grad():
                # 编码输入
                inputs = self._tokenizer(
                    doc_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                # 移至计算设备
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # 获取相关性分数
                scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores_list = scores.tolist()
            
            # 创建(文档, 分数)元组并排序
            scored_docs = [(docs[i], scores_list[i]) for i in range(len(docs))]
            # 按相关性分数降序排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个结果
            return scored_docs[:top_k]
            
        except Exception as e:
            print(f"错误: 获取相关性分数时出错: {str(e)}")
            # 出错时返回原始文档的前top_k个，分数设为0
            return [(doc, 0.0) for doc in docs[:min(top_k, len(docs))]]


class TFIDFRelevanceReranker(RelevanceReranker):
    """
    TF-IDF相关性排序器
    轻量级排序选项，适用于无法加载深度学习模型的场景
    """
    
    def __init__(self):
        """
        初始化TF-IDF相关性排序器
        """
        super().__init__()
        self.ranker_name = "TF-IDF Relevance Ranker"
        self._vectorizer = None
        self._setup_components()
    
    def _setup_components(self):
        """
        设置TF-IDF向量化组件
        """
        try:
            # 尝试导入scikit-learn
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.TfidfVectorizer = TfidfVectorizer
            except ImportError:
                print("正在安装scikit-learn库...")
                import pip
                pip.main(['install', 'scikit-learn'])
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.TfidfVectorizer = TfidfVectorizer
            
            # 初始化向量器
            self._vectorizer = self.TfidfVectorizer(
                token_pattern=r'(?u)\\b\\w+\\b',
                lowercase=True
            )
            
            self.is_initialized = True
            print(f"✓ {self.ranker_name}组件设置完成")
            
        except Exception as e:
            print(f"错误: 设置{self.ranker_name}组件失败: {str(e)}")
            self.is_initialized = False
    
    def reorder_by_relevance(self, query: str, docs: List[str], top_k: int) -> List[str]:
        """
        使用TF-IDF对文档列表进行相关性排序
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的最相关结果数量
            
        Returns:
            排序后的文档列表（前top_k个最相关的结果）
        """
        if not self.is_initialized:
            self._setup_components()
            
        try:
            # 验证输入
            if not docs:
                return []
                
            top_k = min(top_k, len(docs))  # 确保top_k不超过文档列表长度
            
            # 如果查询文本为空，返回原始文档
            if not query:
                return docs[:top_k]
            
            # 计时开始
            start_time = time.time()
            
            # 合并查询和文档进行TF-IDF计算
            all_texts = [query] + docs
            
            # 计算TF-IDF矩阵
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
            
            # 提取查询向量和文档向量
            query_vector = tfidf_matrix[0]
            doc_vectors = tfidf_matrix[1:]
            
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # 计算处理时间
            end_time = time.time()
            print(f"  TF-IDF排序耗时: {end_time - start_time:.4f}秒 (处理 {len(docs)} 个文档)")
            
            # 排序并返回前top_k个最相关的文档
            top_doc_indices = np.argsort(-similarities)[:top_k]  # 降序排序
            
            # 返回排序后的文档
            return [docs[i] for i in top_doc_indices]
            
        except Exception as e:
            print(f"错误: TF-IDF排序处理出错: {str(e)}")
            # 出错时返回原始文档的前top_k个
            return docs[:min(top_k, len(docs))]


# 创建相关性排序器工厂函数
def create_relevance_reranker(ranker_type: str = "bge", **kwargs) -> RelevanceReranker:
    """
    创建文档相关性排序器实例的工厂函数
    
    Args:
        ranker_type: 排序器类型，支持 "bge", "tfidf"
        **kwargs: 排序器初始化参数
        
    Returns:
        相关性排序器实例
    """
    try:
        if ranker_type.lower() == "bge":
            return BgeRelevanceReranker(**kwargs)
        elif ranker_type.lower() == "tfidf":
            return TFIDFRelevanceReranker()
        else:
            print(f"警告: 不支持的排序器类型: {ranker_type}，使用默认排序器")
            return BgeRelevanceReranker(**kwargs)
    except Exception as e:
        print(f"错误: 创建排序器实例时出错: {str(e)}")
        # 返回TF-IDF作为备用选项
        return TFIDFRelevanceReranker()


if __name__ == '__main__':
    # 示例用法
    try:
        print("=== 测试相关性排序器模块 ===")
        
        # 测试BGE相关性排序器
        print("\n1. 测试BGE相关性排序器:")
        ranker = create_relevance_reranker("bge")
        
        # 健康检查
        if ranker.check_health():
            print("排序器健康检查通过")
            
            # 测试排序功能
            test_query = "这个电影很不错"
            test_docs = [
                "这个电影我看的都快睡着了",
                "什么垃圾电影",
                "明天吃什么",
                "你一定要看这个电影",
                "这是一部非常精彩的电影，情节跌宕起伏",
                "这个电影画面很美，但剧情一般",
                "我不喜欢这个类型的电影"
            ]
            
            print(f"\n查询: {test_query}")
            print(f"文档数量: {len(test_docs)}")
            
            # 获取前3个最相关的文档
            top_docs = ranker.reorder_by_relevance(test_query, test_docs, top_k=3)
            
            print("\n相关性排序后的前3个文档:")
            for i, doc in enumerate(top_docs, 1):
                print(f"  {i}. {doc}")
            
            # 测试带分数的排序功能
            if hasattr(ranker, 'get_relevance_scores'):
                scored_docs = ranker.get_relevance_scores(test_query, test_docs, top_k=3)
                print("\n带相关性分数的前3个文档:")
                for i, (doc, score) in enumerate(scored_docs, 1):
                    print(f"  {i}. {doc} (分数: {score:.4f})")
        else:
            print("排序器健康检查失败")
            
        # 测试TF-IDF相关性排序器作为备用选项
        print("\n2. 测试TF-IDF相关性排序器:")
        tfidf_ranker = create_relevance_reranker("tfidf")
        
        if tfidf_ranker.check_health():
            print("TF-IDF排序器健康检查通过")
            
            # 使用相同的查询和文档
            tfidf_results = tfidf_ranker.reorder_by_relevance(test_query, test_docs, top_k=3)
            print("\nTF-IDF排序后的前3个文档:")
            for i, doc in enumerate(tfidf_results, 1):
                print(f"  {i}. {doc}")
        else:
            print("TF-IDF排序器健康检查失败")
            
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
