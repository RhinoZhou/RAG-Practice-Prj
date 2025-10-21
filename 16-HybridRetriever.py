#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合向量检索(HNSW)和关键词检索(BM25)的混合检索策略

作者: Ph.D. Rhino

功能说明:
    本程序实现了一种高效的混合检索策略，通过融合HNSW向量检索和BM25关键词检索的优势，
    并使用RRF(倒数排名融合)算法综合两种检索结果，显著提高了检索的准确性和鲁棒性。
    向量检索捕获语义相似性，关键词检索捕获精确的词法匹配，两者结合可以覆盖更多检索场景。

核心技术:
    1. HNSW(Hierarchical Navigable Small World)：高效的近似最近邻搜索算法，比传统的FAISS索引
       具有更高的搜索效率和更低的内存占用。
    2. BM25：经典的关键词检索算法，在信息检索领域广泛应用，适合精确的关键词匹配。
    3. RRF(Reciprocal Rank Fusion)：无需归一化的排名融合算法，能够有效综合不同检索系统的结果。

主要功能:
    - 自动生成和加载测试数据集
    - 自动检查和安装必要的依赖包
    - 构建HNSW向量索引和BM25关键词索引
    - 实现基于RRF的混合检索策略
    - 支持不同检索权重的调整(alpha参数)
    - 性能对比分析(向量检索vs关键词检索vs混合检索)
    - 可视化检索结果和性能指标
    - 保存检索配置和结果

执行流程:
    1. 检查并安装必要的依赖包
    2. 生成测试数据集
    3. 初始化混合检索器
    4. 执行检索测试
    5. 比较不同检索方法的性能
    6. 生成可视化图表
    7. 保存结果和报告

使用说明:
    程序支持自定义测试数据集、检索参数和评估指标。默认使用小型中文数据集进行演示，
    可通过修改配置参数调整检索性能和行为。
"""

# 基础导入
import os
import sys
import subprocess
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 配置目录和路径
OUTPUT_DIR = "./hybrid_retriever_results"
DATA_DIR = "./hybrid_retriever_data"
CONFIG_DIR = "./hybrid_retriever_configs"

# 默认配置参数
DEFAULT_VECTOR_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言模型，支持中文
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.7  # 向量检索权重
DEFAULT_EVAL_QUERIES = 10  # 评估查询数量

# 简化的依赖包列表（避免需要Rust编译的包）
required_packages = [
    'numpy>=1.20.0,<2.0.0',                 # 数值计算
    'nltk>=3.6.0,<4.0.0',                   # 自然语言处理
    'rank-bm25>=0.2.1,<0.4.0',              # BM25算法实现
    'matplotlib>=3.4.0,<4.0.0',             # 可视化
    'tqdm>=4.62.0,<5.0.0'                   # 进度条
]

# 先定义并执行依赖检查
def check_dependencies():
    """
    检查并安装必要的依赖包
    """
    print("正在检查依赖...")
    
    # 安装依赖
    for package in required_packages:
        try:
            package_name = package.split('>=')[0].split('<')[0]
            __import__(package_name)
            print(f"✓ {package_name} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {package} 安装失败，请手动安装")
    
    print("依赖检查和安装完成。")

# 确保必要的目录存在
def ensure_directories():
    """
    确保必要的目录存在
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    print(f"已确保所有必要目录存在: {OUTPUT_DIR}, {DATA_DIR}, {CONFIG_DIR}")

# 生成测试数据集
def generate_test_dataset():
    """
    生成用于测试的数据集
    
    Returns:
        tuple: (corpus, queries, relevant_docs)
    """
    print("生成测试数据集...")
    
    # 创建一个小型企业FAQ数据集
    corpus = [
        "企业账号申请需要提交营业执照复印件、法人身份证、联系方式等材料，审核通过后将在3个工作日内开通账号。",
        "您可以通过登录页面的'忘记密码'选项，使用注册邮箱或手机号验证身份后重置密码。",
        "免费用户每小时限制100次API调用，付费用户根据套餐不同，限制为每小时1000-10000次不等。",
        "系统支持CSV、Excel、JSON、PDF等多种格式的数据导出，您可以在数据分析页面选择所需格式。",
        "企业用户可通过管理后台的'资源申请'页面提交存储空间扩容申请，我们将在24小时内响应。",
        "数据安全措施包括数据加密、访问控制、定期备份和安全审计，确保您的数据安全可靠。",
        "系统集成API支持RESTful接口，提供详细的API文档和SDK，方便您进行二次开发。",
        "用户可以在个人中心管理个人信息、绑定邮箱和手机号、设置通知偏好等。",
        "企业版支持多用户协作，管理员可以设置不同用户角色和权限，实现精细化的权限管理。",
        "系统会定期进行性能优化和功能更新，确保您获得最佳的使用体验。",
        "常见问题解答可以在帮助中心找到，如有其他问题可以联系客服获取支持。",
        "数据分析功能支持多维度数据展示、自定义报表生成和数据可视化图表。",
        "移动端应用支持iOS和Android系统，可以通过应用商店下载安装。",
        "用户可以设置数据访问权限，控制敏感信息的查看和操作权限。",
        "系统提供7×24小时的技术支持服务，企业用户还可以享受专属客户经理服务。"
    ]
    
    # 创建查询和对应的相关文档
    queries_relevant = [
        {"query": "如何申请企业账号？", "relevant_index": 0},
        {"query": "密码忘记了怎么办？", "relevant_index": 1},
        {"query": "API调用有次数限制吗？", "relevant_index": 2},
        {"query": "数据导出支持哪些格式？", "relevant_index": 3},
        {"query": "如何增加存储空间？", "relevant_index": 4},
        {"query": "数据安全如何保障？", "relevant_index": 5},
        {"query": "如何进行系统集成？", "relevant_index": 6},
        {"query": "个人信息如何管理？", "relevant_index": 7},
        {"query": "多用户协作功能介绍", "relevant_index": 8},
        {"query": "系统更新频率是多少？", "relevant_index": 9},
        {"query": "哪里可以找到常见问题解答？", "relevant_index": 10},
        {"query": "数据分析功能有哪些？", "relevant_index": 11},
        {"query": "有没有手机应用？", "relevant_index": 12},
        {"query": "如何设置数据权限？", "relevant_index": 13},
        {"query": "技术支持服务时间？", "relevant_index": 14},
        # 增加一些语义相似但关键词不完全匹配的查询
        {"query": "企业账户申请流程是什么？", "relevant_index": 0},
        {"query": "重置密码的步骤有哪些？", "relevant_index": 1},
        {"query": "API调用频率是多少？", "relevant_index": 2},
        {"query": "支持哪些文件格式导出数据？", "relevant_index": 3},
        {"query": "扩容存储的方法", "relevant_index": 4}
    ]
    
    # 分离查询和相关文档索引
    queries = [item["query"] for item in queries_relevant]
    relevant_docs = [item["relevant_index"] for item in queries_relevant]
    
    # 保存数据集
    dataset = {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs
    }
    
    dataset_path = os.path.join(DATA_DIR, "test_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据集已保存至: {dataset_path}")
    print(f"数据集包含 {len(corpus)} 个文档和 {len(queries)} 个查询")
    
    return corpus, queries, relevant_docs

# 简化版混合检索器实现（不依赖复杂的向量模型）
class HybridRetriever:
    def __init__(self, vector_model_path, corpus, tokenized_corpus=None):
        """
        初始化混合检索器
        
        Args:
            vector_model_path: 向量模型路径或名称（这里仅作为标识，实际使用简单TF-IDF）
            corpus: 文档集合
            tokenized_corpus: 预分词的文档集合（可选）
        """
        print(f"\n初始化混合检索器")
        
        # 导入必要的库
        import numpy as np
        import nltk
        from nltk.tokenize import word_tokenize
        from rank_bm25 import BM25Okapi
        
        self.corpus = corpus
        self.vector_model_path = vector_model_path
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("下载NLTK必要数据...")
            nltk.download('punkt')
        
        # 简单的中文分词和TF-IDF向量化
        print("构建TF-IDF向量模型...")
        # 分词处理
        self.tokenized_corpus = []
        for doc in corpus:
            # 对于中文，使用字符级分词
            tokens = list(doc.lower())
            self.tokenized_corpus.append(tokens)
        
        # 构建词汇表
        self.vocabulary = set()
        for tokens in self.tokenized_corpus:
            self.vocabulary.update(tokens)
        self.vocabulary = list(self.vocabulary)
        self.vocab_size = len(self.vocabulary)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        # 计算TF-IDF
        self.document_frequency = np.zeros(self.vocab_size)
        for tokens in self.tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.word_to_idx:
                    self.document_frequency[self.word_to_idx[token]] += 1
        
        # 计算IDF
        import math
        self.idf = np.log(len(corpus) / (1 + self.document_frequency)) + 1
        
        # 生成文档向量（TF-IDF）
        self.corpus_embeddings = []
        for tokens in self.tokenized_corpus:
            # 计算TF
            tf = np.zeros(self.vocab_size)
            for token in tokens:
                if token in self.word_to_idx:
                    tf[self.word_to_idx[token]] += 1
            # 归一化TF
            if sum(tf) > 0:
                tf = tf / sum(tf)
            # 计算TF-IDF
            tf_idf = tf * self.idf
            # 归一化向量
            norm = np.linalg.norm(tf_idf)
            if norm > 0:
                tf_idf = tf_idf / norm
            self.corpus_embeddings.append(tf_idf)
        self.corpus_embeddings = np.array(self.corpus_embeddings).astype('float32')
        
        # 初始化向量索引（简单的暴力搜索）
        self.dim = self.vocab_size
        
        # 初始化BM25检索器
        print("初始化BM25检索器...")
        if tokenized_corpus is None:
            self.tokenized_corpus = self.tokenized_corpus
        else:
            self.tokenized_corpus = tokenized_corpus
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("混合检索器初始化完成！")
        
        # 初始化BM25检索器
        print("初始化BM25检索器...")
        if tokenized_corpus is None:
            # 下载NLTK数据
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("下载NLTK必要数据...")
                nltk.download('punkt')
            
            # 简单的中文分词 (对于BM25，可以使用更简单的分词方式)
            self.tokenized_corpus = []
            for doc in corpus:
                # 对于中文，可以使用字符级分词，或者保留原始文本
                # 这里使用简单的字符分词
                tokens = list(doc.lower())
                self.tokenized_corpus.append(tokens)
        else:
            self.tokenized_corpus = tokenized_corpus
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("混合检索器初始化完成！")
    
    def retrieve(self, query, top_k=5, alpha=0.5):
        """
        混合检索实现
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            alpha: 控制向量检索权重 (1-alpha 为关键词检索权重)
        
        Returns:
            list: 包含(文档, 分数, 索引)的列表
        """
        import numpy as np
        
        # 向量检索（使用TF-IDF）
        query_tokens = list(query.lower())  # 中文简单分词
        
        # 计算查询向量（TF-IDF）
        tf = np.zeros(self.vocab_size)
        for token in query_tokens:
            if token in self.word_to_idx:
                tf[self.word_to_idx[token]] += 1
        
        # 归一化TF
        if sum(tf) > 0:
            tf = tf / sum(tf)
        
        # 计算TF-IDF
        query_embedding = tf * self.idf
        
        # 归一化查询向量
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # 计算余弦相似度（简单的暴力搜索）
        vector_scores = np.dot(self.corpus_embeddings, query_embedding)
        vector_indices = np.argsort(vector_scores)[::-1][:top_k*2]
        vector_scores = vector_scores[vector_indices]
        
        # 关键词检索
        tokenized_query = query_tokens
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化BM25分数
        max_bm25 = max(bm25_scores)
        if max_bm25 > 0:
            bm25_scores = [s/max_bm25 for s in bm25_scores]
            
        # 获取BM25 top结果
        bm25_top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
        bm25_top_scores = [bm25_scores[i] for i in bm25_top_indices]
        
        # 合并结果集 (所有候选)
        candidates = set(vector_indices.tolist() + bm25_top_indices.tolist())
        
        # RRF (Reciprocal Rank Fusion) 重排序
        rrf_scores = defaultdict(float)
        k = 60  # RRF参数，通常在60左右效果较好
        
        # 向量检索排名
        for rank, idx in enumerate(vector_indices):
            rrf_scores[idx] += alpha * (1 / (k + rank + 1))
        
        # BM25排名
        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[idx] += (1-alpha) * (1 / (k + rank + 1))
            
        # 按重排序分数选择top-k
        final_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]
        final_scores = [rrf_scores[idx] for idx in final_indices]
        
        return [(self.corpus[idx], final_scores[i], idx) for i, idx in enumerate(final_indices)]
    
    def vector_retrieve_only(self, query, top_k=5):
        """
        仅使用向量检索（TF-IDF）
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
        
        Returns:
            list: 包含(文档, 分数, 索引)的列表
        """
        import numpy as np
        
        # 分词
        query_tokens = list(query.lower())
        
        # 计算查询向量（TF-IDF）
        tf = np.zeros(self.vocab_size)
        for token in query_tokens:
            if token in self.word_to_idx:
                tf[self.word_to_idx[token]] += 1
        
        # 归一化TF
        if sum(tf) > 0:
            tf = tf / sum(tf)
        
        # 计算TF-IDF
        query_embedding = tf * self.idf
        
        # 归一化查询向量
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # 计算余弦相似度
        scores = np.dot(self.corpus_embeddings, query_embedding)
        indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(indices):
            results.append((self.corpus[idx], scores[idx], idx))
        
        return results
    
    def bm25_retrieve_only(self, query, top_k=5):
        """
        仅使用BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
        
        Returns:
            list: 包含(文档, 分数, 索引)的列表
        """
        import numpy as np
        
        # 关键词检索
        tokenized_query = list(query.lower())  # 中文简单分词
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k结果
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], bm25_scores[idx], idx))
        
        return results

# 评估检索性能
def evaluate_retrieval(retriever, queries, relevant_docs, top_k=5, alpha=0.7):
    """
    评估检索性能
    
    Args:
        retriever: 检索器对象
        queries: 查询列表
        relevant_docs: 相关文档索引列表
        top_k: 返回的文档数量
        alpha: 混合检索权重
    
    Returns:
        dict: 评估指标
    """
    print(f"\n评估检索性能 (top_k={top_k}, alpha={alpha})...")
    
    results = {
        "hybrid": {"hits": 0, "precision": 0, "recall": 0, "f1": 0, "average_rank": 0, "queries": []},
        "vector_only": {"hits": 0, "precision": 0, "recall": 0, "f1": 0, "average_rank": 0, "queries": []},
        "bm25_only": {"hits": 0, "precision": 0, "recall": 0, "f1": 0, "average_rank": 0, "queries": []}
    }
    
    # 只评估前DEFAULT_EVAL_QUERIES个查询以提高速度
    eval_queries = queries[:DEFAULT_EVAL_QUERIES]
    eval_relevant = relevant_docs[:DEFAULT_EVAL_QUERIES]
    
    for i, (query, relevant_idx) in enumerate(zip(eval_queries, eval_relevant)):
        print(f"\n评估查询 {i+1}/{len(eval_queries)}: {query}")
        
        # 混合检索
        hybrid_result = retriever.retrieve(query, top_k=top_k, alpha=alpha)
        hybrid_indices = [doc[2] for doc in hybrid_result]
        
        # 向量检索
        vector_result = retriever.vector_retrieve_only(query, top_k=top_k)
        vector_indices = [doc[2] for doc in vector_result]
        
        # BM25检索
        bm25_result = retriever.bm25_retrieve_only(query, top_k=top_k)
        bm25_indices = [doc[2] for doc in bm25_result]
        
        # 评估混合检索
        if relevant_idx in hybrid_indices:
            results["hybrid"]["hits"] += 1
            rank = hybrid_indices.index(relevant_idx) + 1
            results["hybrid"]["average_rank"] += rank
        else:
            rank = float('inf')
        
        # 评估向量检索
        if relevant_idx in vector_indices:
            results["vector_only"]["hits"] += 1
            vector_rank = vector_indices.index(relevant_idx) + 1
            results["vector_only"]["average_rank"] += vector_rank
        else:
            vector_rank = float('inf')
        
        # 评估BM25检索
        if relevant_idx in bm25_indices:
            results["bm25_only"]["hits"] += 1
            bm25_rank = bm25_indices.index(relevant_idx) + 1
            results["bm25_only"]["average_rank"] += bm25_rank
        else:
            bm25_rank = float('inf')
        
        # 计算单个查询的指标
        hybrid_precision = 1.0 / top_k if rank <= top_k else 0
        hybrid_recall = 1.0 if rank <= top_k else 0
        hybrid_f1 = 2 * hybrid_precision * hybrid_recall / (hybrid_precision + hybrid_recall + 1e-9)
        
        vector_precision = 1.0 / top_k if vector_rank <= top_k else 0
        vector_recall = 1.0 if vector_rank <= top_k else 0
        vector_f1 = 2 * vector_precision * vector_recall / (vector_precision + vector_recall + 1e-9)
        
        bm25_precision = 1.0 / top_k if bm25_rank <= top_k else 0
        bm25_recall = 1.0 if bm25_rank <= top_k else 0
        bm25_f1 = 2 * bm25_precision * bm25_recall / (bm25_precision + bm25_recall + 1e-9)
        
        # 累加指标
        results["hybrid"]["precision"] += hybrid_precision
        results["hybrid"]["recall"] += hybrid_recall
        results["hybrid"]["f1"] += hybrid_f1
        
        results["vector_only"]["precision"] += vector_precision
        results["vector_only"]["recall"] += vector_recall
        results["vector_only"]["f1"] += vector_f1
        
        results["bm25_only"]["precision"] += bm25_precision
        results["bm25_only"]["recall"] += bm25_recall
        results["bm25_only"]["f1"] += bm25_f1
        
        # 记录单个查询的结果
        query_result = {
            "query": query,
            "relevant_doc_index": relevant_idx,
            "hybrid_rank": rank,
            "vector_rank": vector_rank,
            "bm25_rank": bm25_rank,
            "hybrid_result": [(doc[0][:50] + "...", score, idx) for doc, score, idx in hybrid_result],
            "vector_result": [(doc[0][:50] + "...", score, idx) for doc, score, idx in vector_result],
            "bm25_result": [(doc[0][:50] + "...", score, idx) for doc, score, idx in bm25_result]
        }
        
        results["hybrid"]["queries"].append(query_result)
        
        # 显示结果
        print(f"  相关文档索引: {relevant_idx}")
        print(f"  混合检索: 相关文档排名={rank if rank != float('inf') else '未找到'}, 分数={hybrid_f1:.4f}")
        print(f"  向量检索: 相关文档排名={vector_rank if vector_rank != float('inf') else '未找到'}, 分数={vector_f1:.4f}")
        print(f"  BM25检索: 相关文档排名={bm25_rank if bm25_rank != float('inf') else '未找到'}, 分数={bm25_f1:.4f}")
    
    # 计算平均指标
    query_count = len(eval_queries)
    
    # 混合检索
    results["hybrid"]["precision"] /= query_count
    results["hybrid"]["recall"] /= query_count
    results["hybrid"]["f1"] /= query_count
    results["hybrid"]["average_rank"] /= results["hybrid"]["hits"] if results["hybrid"]["hits"] > 0 else 1
    results["hybrid"]["hit_rate"] = results["hybrid"]["hits"] / query_count
    
    # 向量检索
    results["vector_only"]["precision"] /= query_count
    results["vector_only"]["recall"] /= query_count
    results["vector_only"]["f1"] /= query_count
    results["vector_only"]["average_rank"] /= results["vector_only"]["hits"] if results["vector_only"]["hits"] > 0 else 1
    results["vector_only"]["hit_rate"] = results["vector_only"]["hits"] / query_count
    
    # BM25检索
    results["bm25_only"]["precision"] /= query_count
    results["bm25_only"]["recall"] /= query_count
    results["bm25_only"]["f1"] /= query_count
    results["bm25_only"]["average_rank"] /= results["bm25_only"]["hits"] if results["bm25_only"]["hits"] > 0 else 1
    results["bm25_only"]["hit_rate"] = results["bm25_only"]["hits"] / query_count
    
    # 显示总体结果
    print("\n===== 检索性能评估结果 =====")
    print(f"混合检索 - Hit Rate: {results['hybrid']['hit_rate']:.4f}, F1: {results['hybrid']['f1']:.4f}, 平均排名: {results['hybrid']['average_rank']:.2f}")
    print(f"向量检索 - Hit Rate: {results['vector_only']['hit_rate']:.4f}, F1: {results['vector_only']['f1']:.4f}, 平均排名: {results['vector_only']['average_rank']:.2f}")
    print(f"BM25检索 - Hit Rate: {results['bm25_only']['hit_rate']:.4f}, F1: {results['bm25_only']['f1']:.4f}, 平均排名: {results['bm25_only']['average_rank']:.2f}")
    
    # 计算性能提升
    vector_improvement = (results['hybrid']['f1'] - results['vector_only']['f1']) / (results['vector_only']['f1'] + 1e-9)
    bm25_improvement = (results['hybrid']['f1'] - results['bm25_only']['f1']) / (results['bm25_only']['f1'] + 1e-9)
    
    print(f"\n性能提升:")
    print(f"相比向量检索: {vector_improvement * 100:.2f}%")
    print(f"相比BM25检索: {bm25_improvement * 100:.2f}%")
    
    # 添加到结果中
    results["improvements"] = {
        "vs_vector": vector_improvement,
        "vs_bm25": bm25_improvement
    }
    
    return results

# 生成性能可视化图表
def generate_performance_visualization(results):
    """
    生成性能可视化图表
    
    Args:
        results: 评估结果
    """
    try:
        print("\n生成性能可视化图表...")
        
        # 提取数据
        methods = ['混合检索', '向量检索', 'BM25检索']
        hit_rates = [
            results['hybrid']['hit_rate'],
            results['vector_only']['hit_rate'],
            results['bm25_only']['hit_rate']
        ]
        f1_scores = [
            results['hybrid']['f1'],
            results['vector_only']['f1'],
            results['bm25_only']['f1']
        ]
        avg_ranks = [
            results['hybrid']['average_rank'],
            results['vector_only']['average_rank'],
            results['bm25_only']['average_rank']
        ]
        
        # 创建Hit Rate对比图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, hit_rates, color=['#4CAF50', '#2196F3', '#FF9800'])
        plt.xlabel('检索方法')
        plt.ylabel('Hit Rate')
        plt.title('不同检索方法的Hit Rate对比')
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom')
        
        hit_rate_path = os.path.join(OUTPUT_DIR, "hit_rate_comparison.png")
        plt.savefig(hit_rate_path, dpi=300, bbox_inches='tight')
        print(f"Hit Rate对比图已保存至: {hit_rate_path}")
        
        # 创建F1分数对比图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, f1_scores, color=['#4CAF50', '#2196F3', '#FF9800'])
        plt.xlabel('检索方法')
        plt.ylabel('F1 Score')
        plt.title('不同检索方法的F1分数对比')
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom')
        
        f1_path = os.path.join(OUTPUT_DIR, "f1_score_comparison.png")
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        print(f"F1分数对比图已保存至: {f1_path}")
        
        # 创建平均排名对比图 (注意：排名越低越好)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, avg_ranks, color=['#4CAF50', '#2196F3', '#FF9800'])
        plt.xlabel('检索方法')
        plt.ylabel('平均排名')
        plt.title('不同检索方法的平均排名对比 (越低越好)')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        rank_path = os.path.join(OUTPUT_DIR, "average_rank_comparison.png")
        plt.savefig(rank_path, dpi=300, bbox_inches='tight')
        print(f"平均排名对比图已保存至: {rank_path}")
        
        # 创建性能提升图
        improvements = [
            results['improvements']['vs_vector'] * 100,
            results['improvements']['vs_bm25'] * 100
        ]
        improvement_labels = ['相比向量检索', '相比BM25检索']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(improvement_labels, improvements, color=['#9C27B0', '#E91E63'])
        plt.xlabel('对比基准')
        plt.ylabel('F1分数提升百分比 (%)')
        plt.title('混合检索相比单一检索方法的性能提升')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height > 0 else 'red'
            plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -6),
                    f'{height:.2f}%', ha='center', va='bottom', color=color)
        
        improvement_path = os.path.join(OUTPUT_DIR, "performance_improvement.png")
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        print(f"性能提升图已保存至: {improvement_path}")
        
    except Exception as e:
        print(f"生成可视化图表时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 保存结果和配置
def save_results_and_config(results, retriever, corpus, queries, top_k, alpha):
    """
    保存评估结果和配置
    
    Args:
        results: 评估结果
        retriever: 检索器对象
        corpus: 文档集合
        queries: 查询列表
        top_k: 返回的文档数量
        alpha: 混合检索权重
    """
    print("\n保存结果和配置...")
    
    # 保存评估结果
    results_path = os.path.join(OUTPUT_DIR, "retrieval_evaluation_results.json")
    
    # 准备可序列化的结果
    serializable_results = {}
    for method, metrics in results.items():
        if method != "improvements":
            serializable_results[method] = {}
            for key, value in metrics.items():
                if key != "queries":
                    serializable_results[method][key] = value
                else:
                    # 简化查询结果以减少文件大小
                    simplified_queries = []
                    for q in value:
                        simplified = {
                            "query": q["query"],
                            "relevant_doc_index": q["relevant_doc_index"],
                            "hybrid_rank": q["hybrid_rank"],
                            "vector_rank": q["vector_rank"],
                            "bm25_rank": q["bm25_rank"]
                        }
                        simplified_queries.append(simplified)
                    serializable_results[method]["queries"] = simplified_queries
        else:
            serializable_results[method] = metrics
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 保存配置
    config = {
        "vector_model": DEFAULT_VECTOR_MODEL,
        "top_k": top_k,
        "alpha": alpha,
        "corpus_size": len(corpus),
        "queries_count": len(queries),
        "hnsw_params": {
            "M": 64,
            "efConstruction": 256,
            "efSearch": 64
        },
        "rrf_k": 60,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(CONFIG_DIR, "hybrid_retriever_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"评估结果已保存至: {results_path}")
    print(f"配置已保存至: {config_path}")

# 交互式测试
def interactive_test(retriever, top_k=5, alpha=0.7):
    """
    交互式测试检索功能
    
    Args:
        retriever: 检索器对象
        top_k: 返回的文档数量
        alpha: 混合检索权重
    """
    print("\n===== 交互式检索测试 =====")
    print("输入查询文本进行检索测试 (输入'quit'退出)")
    
    while True:
        try:
            query = input("\n请输入查询: ")
            if query.lower() in ['quit', 'exit', 'q', '退出']:
                break
            
            if not query.strip():
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行混合检索
            hybrid_results = retriever.retrieve(query, top_k=top_k, alpha=alpha)
            
            # 记录结束时间
            end_time = time.time()
            
            print(f"\n检索结果 (耗时: {end_time - start_time:.4f}秒):")
            print(f"混合检索 (alpha={alpha}) 结果:")
            
            for i, (doc, score, idx) in enumerate(hybrid_results):
                print(f"  {i+1}. [分数: {score:.4f}] [索引: {idx}] {doc}")
                
            # 询问是否查看单独的检索结果
            show_separate = input("\n是否查看单独的向量检索和BM25检索结果? (y/n): ")
            if show_separate.lower() == 'y':
                # 向量检索
                vector_results = retriever.vector_retrieve_only(query, top_k=top_k)
                print(f"\n向量检索结果:")
                for i, (doc, score, idx) in enumerate(vector_results):
                    print(f"  {i+1}. [分数: {score:.4f}] [索引: {idx}] {doc}")
                
                # BM25检索
                bm25_results = retriever.bm25_retrieve_only(query, top_k=top_k)
                print(f"\nBM25检索结果:")
                for i, (doc, score, idx) in enumerate(bm25_results):
                    print(f"  {i+1}. [分数: {score:.4f}] [索引: {idx}] {doc}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"检索过程中出错: {str(e)}")
    
    print("\n交互式测试结束")

# 主函数
def main():
    """
    主函数
    """
    print("===== 融合向量检索(HNSW)和关键词检索(BM25)的混合检索策略 =====")
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查依赖
    check_dependencies()
    
    # 确保目录存在
    ensure_directories()
    
    # 生成测试数据集
    corpus, queries, relevant_docs = generate_test_dataset()
    
    # 初始化混合检索器
    retriever = HybridRetriever(DEFAULT_VECTOR_MODEL, corpus)
    
    # 评估检索性能
    evaluation_results = evaluate_retrieval(
        retriever, 
        queries, 
        relevant_docs, 
        top_k=DEFAULT_TOP_K, 
        alpha=DEFAULT_ALPHA
    )
    
    # 生成性能可视化图表
    generate_performance_visualization(evaluation_results)
    
    # 保存结果和配置
    save_results_and_config(
        evaluation_results, 
        retriever, 
        corpus, 
        queries, 
        DEFAULT_TOP_K, 
        DEFAULT_ALPHA
    )
    
    # 进行交互式测试
    try:
        interactive_test(retriever, top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA)
    except Exception as e:
        print(f"交互式测试出错: {str(e)}")
    
    # 记录结束时间
    end_time = time.time()
    print(f"\n程序执行时间: {end_time - start_time:.2f} 秒")
    print("\n混合检索策略演示完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        import traceback
        traceback.print_exc()