#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索与重排系统 - 分块策略影响分析

功能说明:
- 实现Dense/Sparse/混合检索的完整闭环
- 评估不同chunk_overlap参数对关键信息覆盖率和索引规模的影响
- 在小型FAQ语料上构建TF-IDF与词向量两路检索
- 支持并行召回、z-score归一化与线性融合
- 实现α网格搜索以优化nDCG@10性能指标
- 输出最终Top-M检索结果

输入:
- 自动生成示例数据: 模拟FAQ语料、查询和相关性判断
- 可选配置文件: configs/hybrid.json (包含检索参数、分块设置等)

输出:
- outputs/hybrid_candidates.jsonl: 混合检索的候选结果
- outputs/chunk_analysis.png: 分块策略影响分析图表
- 控制台输出: 融合α参数与nDCG@10性能指标

依赖包:
- numpy: 用于数值计算
- scikit-learn: 用于TF-IDF实现和评估指标
- matplotlib: 用于可视化分析结果

作者: Trae AI
版本: 1.0
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# 设置中文字体，确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 检查并安装依赖包 (优化版本)
def check_and_install_dependencies():
    """检查并自动安装必要的依赖包 - 优化版：减少不必要的导入检查"""
    required_packages = ["numpy>=1.21.0", "scikit-learn>=1.0.0", "matplotlib>=3.5.0"]
    installed = False
    
    try:
        # 仅检查一个核心包来验证安装状态
        import numpy as np
        installed = True
    except ImportError:
        print("正在安装必要的依赖包...")
        # 并行安装所有包（使用单个pip命令）
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
            print("✓ 成功安装所有依赖包")
        except Exception as e:
            print(f"✗ 安装依赖包失败: {str(e)}")
            raise RuntimeError(f"无法安装必要的依赖包，请手动安装") from e
    
    if installed:
        print("✓ 所有依赖包已安装完成")

# 创建必要的目录 (优化版本)
def create_directories():
    """创建数据、配置和输出目录 - 优化版：使用列表推导式减少循环开销"""
    directories = ["data", "configs", "outputs"]
    created = [os.makedirs(d, exist_ok=True) for d in directories if not os.path.exists(d)]
    if created:
        print(f"✓ 创建了 {len(created)} 个必要目录")

# 加载或生成示例数据 (优化版本)
@lru_cache(maxsize=1)  # 缓存结果，避免重复加载
def load_or_generate_sample_data():
    """加载已有数据或生成示例数据 - 优化版：优先读取已有数据，避免重复生成"""
    corpus_path = "data/corpus.jsonl"
    queries_path = "data/queries.jsonl"
    qrels_path = "data/qrels.json"
    
    # 检查文件是否存在
    if all(os.path.exists(p) for p in [corpus_path, queries_path, qrels_path]):
        try:
            # 读取已有的语料数据
            corpus = []
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    corpus.append(json.loads(line.strip()))
            
            # 读取已有的查询数据
            queries = []
            with open(queries_path, "r", encoding="utf-8") as f:
                for line in f:
                    queries.append(json.loads(line.strip()))
            
            # 读取已有的相关性判断数据
            with open(qrels_path, "r", encoding="utf-8") as f:
                qrels = json.load(f)
            
            print("✓ 成功加载已有数据文件")
            return corpus, queries, qrels
        except Exception as e:
            print(f"✗ 读取数据文件失败: {str(e)}，将重新生成数据")
    
    # 生成新的示例数据
    # 示例FAQ语料
    corpus = [
        {
            "id": "doc1",
            "title": "什么是人工智能？",
            "content": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在开发能够模拟和执行通常需要人类智能的任务的系统。这些任务包括学习、推理、解决问题、感知、理解自然语言等。"
        },
        {
            "id": "doc2",
            "title": "人工智能有哪些应用领域？",
            "content": "人工智能的应用领域非常广泛，包括但不限于：自然语言处理（如语音识别、机器翻译）、计算机视觉（如图像识别、人脸识别）、推荐系统、自动驾驶、医疗诊断、金融分析、智能制造等。"
        },
        {
            "id": "doc3",
            "title": "机器学习和深度学习有什么区别？",
            "content": "机器学习是人工智能的一个分支，专注于开发能够从数据中学习的算法。深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的工作方式，特别适合处理大量复杂数据，如图像、音频和自然语言。"
        },
        {
            "id": "doc4",
            "title": "什么是自然语言处理？",
            "content": "自然语言处理（NLP）是人工智能的一个分支，致力于使计算机能够理解、解释和生成人类语言。NLP技术包括文本分析、情感分析、命名实体识别、机器翻译、问答系统等。"
        },
        {
            "id": "doc5",
            "title": "计算机视觉技术有哪些应用？",
            "content": "计算机视觉技术的应用包括：图像识别、人脸识别、物体检测、视频分析、医学图像分析、自动驾驶中的环境感知、安防监控、增强现实等。"
        },
        {
            "id": "doc6",
            "title": "推荐系统的工作原理是什么？",
            "content": "推荐系统通过分析用户的历史行为、偏好和相似用户的行为，预测用户可能感兴趣的内容。主要方法包括基于内容的推荐、协同过滤、矩阵分解、深度学习推荐等。"
        },
        {
            "id": "doc7",
            "title": "什么是深度学习框架？",
            "content": "深度学习框架是用于构建和训练深度学习模型的软件库。常见的深度学习框架包括TensorFlow、PyTorch、Keras、MXNet等，它们提供了丰富的API和工具，简化了神经网络的实现和训练过程。"
        },
        {
            "id": "doc8",
            "title": "人工智能的伦理问题有哪些？",
            "content": "人工智能引发的伦理问题包括：隐私保护、算法偏见、就业影响、安全风险、责任归属、军事应用、社会不平等加剧等。随着AI技术的发展，这些问题越来越受到关注。"
        }
    ]
    
    # 示例查询
    queries = [
        {"id": "q1", "text": "人工智能的定义是什么？"},
        {"id": "q2", "text": "AI可以应用在哪些地方？"},
        {"id": "q3", "text": "机器学习和深度学习的不同之处？"},
        {"id": "q4", "text": "自然语言处理技术包括什么？"},
        {"id": "q5", "text": "计算机视觉有哪些实际应用？"}
    ]
    
    # 示例相关性判断
    qrels = {
        "q1": {"doc1": 2, "doc2": 1},
        "q2": {"doc2": 2, "doc8": 1},
        "q3": {"doc3": 2, "doc7": 1},
        "q4": {"doc4": 2, "doc2": 1},
        "q5": {"doc5": 2, "doc2": 1}
    }
    
    # 批量保存数据到文件
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    with open(queries_path, "w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
    
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False, indent=2)
    
    print("✓ 生成了示例数据文件")
    return corpus, queries, qrels

# 加载配置文件
def load_config(config_path="configs/hybrid.json"):
    """加载配置文件，如果不存在则使用默认配置"""
    # 默认配置
    default_config = {
        "top_k": 10,  # 检索返回的top-k结果
        "top_m": 5,   # 最终返回的top-m结果
        "alpha_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 融合系数α的候选值
        "chunk_sizes": [100, 200, 300],  # 测试的分块大小
        "chunk_overlaps": [0, 20, 50, 100],  # 测试的重叠大小
        "use_tfidf_as_bm25": True  # 是否使用TF-IDF近似BM25
    }
    
    # 检查是否有自定义配置文件
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 合并默认配置和自定义配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            print(f"✓ 从文件加载配置: {config_path}")
            return config
        except Exception as e:
            print(f"✗ 加载自定义配置失败: {str(e)}，使用默认配置")
    
    # 保存默认配置
    os.makedirs("configs", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    print("✓ 使用默认配置并保存到文件")
    return default_config

# 分块处理文档
def chunk_documents(corpus_tuple, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    将文档分块处理
    corpus_tuple: 原始文档元组（用于缓存）
    chunk_size: 每个块的最大长度
    chunk_overlap: 块之间的重叠长度
    返回: 分块后的文档列表
    """
    # 将元组转换回列表
    corpus = list(corpus_tuple)
    chunked_corpus = []
    
    # 并行处理文档分块（对于大量文档更高效）
    if len(corpus) > 10:  # 只有当文档数量较多时才使用并行
        with ThreadPoolExecutor(max_workers=min(len(corpus), 4)) as executor:
            futures = [executor.submit(_process_single_doc_chunking, doc, chunk_size, chunk_overlap)
                     for doc in corpus]
            for future in futures:
                chunks = future.result()
                chunked_corpus.extend(chunks)
    else:
        # 串行处理
        for doc in corpus:
            chunks = _process_single_doc_chunking(doc, chunk_size, chunk_overlap)
            chunked_corpus.extend(chunks)
    
    return chunked_corpus

def _process_single_doc_chunking(doc: Dict, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """处理单个文档的分块，用于并行处理"""
    # 合并标题和内容作为完整文本
    full_text = doc["title"] + ": " + doc["content"]
    chunks = []
    
    # 如果文本长度小于chunk_size，不进行分块
    if len(full_text) <= chunk_size:
        chunks.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "full_text": full_text,
            "original_id": doc["id"]
        })
        return chunks
    
    # 进行分块
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end]
        
        # 为每个块创建一个新的文档对象
        chunk_id = f"{doc['id']}_chunk_{len(chunks)}"
        chunks.append({
            "id": chunk_id,
            "title": doc["title"],
            "content": chunk,
            "full_text": chunk,
            "original_id": doc["id"]
        })
        
        # 更新起始位置，考虑重叠
        start += chunk_size - chunk_overlap
        
        # 防止无限循环
        if start >= end and start < len(full_text):
            start = end
    
    return chunks

# 构建TF-IDF模型（近似BM25）
@lru_cache(maxsize=4)  # 缓存不同参数下的模型结果
# 使用元组作为缓存键，因为列表不可哈希
# 传入文档ID和文本内容的元组而不是整个文档列表
# 这里的参数进行了调整，只传入真正影响模型构建的核心信息
def build_tfidf_model(doc_texts_tuple):
    """构建TF-IDF模型用于文本检索 - 优化版：添加缓存机制"""
    # 从元组中解压文档文本和ID
    doc_texts = [item[0] for item in doc_texts_tuple]
    doc_ids = [item[1] for item in doc_texts_tuple]
    
    # 使用更高效的参数配置
    vectorizer = TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        stop_words=None,  # 不使用停用词
        lowercase=True,
        use_idf=True,     # 使用IDF权重
        smooth_idf=True,  # 平滑IDF，避免零分母
        sublinear_tf=True # 亚线性TF缩放，改善长文档处理
    )
    
    # 拟合模型并转换文本
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    return vectorizer, tfidf_matrix, doc_ids

# 构建词向量模型（基于TF-IDF的词频表示）
@lru_cache(maxsize=4)  # 缓存不同参数下的模型结果
def build_vector_model(doc_texts_tuple):
    """构建基于TF-IDF的词向量模型用于文本检索 - 优化版：添加缓存机制"""
    # 从元组中解压文档文本和ID
    doc_texts = [item[0] for item in doc_texts_tuple]
    doc_ids = [item[1] for item in doc_texts_tuple]
    
    # 初始化TF-IDF向量器 - 使用与build_tfidf_model相同的高效配置
    vectorizer = TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        stop_words=None,  # 不使用停用词
        lowercase=True,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    # 拟合模型并转换文本
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    # 转换为密集矩阵以用于相似度计算
    dense_matrix = tfidf_matrix.toarray()
    
    # 归一化向量以用于余弦相似度计算
    norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
    # 避免除零错误
    norms[norms == 0] = 1
    dense_matrix = dense_matrix / norms
    
    return vectorizer, dense_matrix, doc_ids

# 辅助函数：准备用于模型构建的文本元组
def prepare_texts_for_model(corpus: List[Dict]) -> tuple:
    """准备用于模型构建的文本和ID元组，便于缓存"""
    # 转换为元组，使其可哈希（用于缓存）
    return tuple((doc["full_text"], doc["id"]) for doc in corpus)

# 执行TF-IDF检索
def tfidf_search(query: str, vectorizer, tfidf_matrix, doc_ids: List[str], top_k: int = 10):
    """使用TF-IDF模型执行文本检索"""
    # 将查询转换为向量
    query_vector = vectorizer.transform([query])
    
    # 计算相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # 获取top-k结果
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # 构建结果列表
    results = []
    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "score": similarities[idx],
            "rank": len(results) + 1
        })
    
    return results

# 执行向量检索（使用基于TF-IDF的词向量）
def vector_search(query: str, vectorizer, dense_matrix, doc_ids: List[str], top_k: int = 10):
    """使用词向量模型执行检索并返回top-k结果"""
    # 将查询转换为向量
    query_vector = vectorizer.transform([query]).toarray()[0]
    
    # 归一化查询向量
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # 计算余 cosine 相似度
    similarities = np.dot(dense_matrix, query_vector)
    
    # 获取top-k结果
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 构建结果列表
    results = [
        {
            "doc_id": doc_ids[idx],
            "score": float(similarities[idx]),
            "rank": i + 1
        }
        for i, idx in enumerate(top_indices)
    ]
    
    return results

# 归一化分数（z-score）
def normalize_scores(results: List[Dict]) -> List[Dict]:
    """
    对检索结果的分数进行z-score归一化
    
    目的：将不同检索方法（TF-IDF和词向量）的分数标准化到相同的分布，
    使得融合时两种方法的贡献更加平衡。
    
    算法：z-score = (原始分数 - 均值) / 标准差
    这样可以将分数转换为均值为0、标准差为1的标准正态分布
    """
    if not results:
        return results
    
    # 提取分数
    scores = np.array([r["score"] for r in results])
    
    # 计算均值和标准差
    mean = np.mean(scores)
    std = np.std(scores) if len(scores) > 1 else 1.0  # 避免除零错误
    
    # 归一化分数
    normalized_results = []
    for r in results:
        normalized_score = (r["score"] - mean) / std if std > 0 else 0.0
        normalized_results.append({
            "doc_id": r["doc_id"],
            "score": normalized_score,
            "original_score": r["score"]
        })
    
    return normalized_results

# 融合检索结果
def fuse_results(sparse_results: List[Dict], dense_results: List[Dict], alpha: float = 0.5) -> List[Dict]:
    """
    融合sparse和dense检索结果
    
    混合检索的核心算法：通过线性组合两种检索方法的分数来获得更好的检索效果
    
    参数：
    - sparse_results: TF-IDF检索结果（擅长精确匹配）
    - dense_results: 词向量检索结果（擅长语义相似性）
    - alpha: dense结果的权重，(1-alpha)为sparse结果的权重
    
    融合公式：final_score = (1-α) * sparse_score + α * dense_score
    - α=0: 纯sparse检索
    - α=1: 纯dense检索  
    - α=0.5: 两种方法等权重融合
    """
    # 创建文档ID到分数的映射
    doc_scores = {}
    
    # 添加sparse结果
    for r in sparse_results:
        doc_id = r["doc_id"]
        doc_scores[doc_id] = {
            "sparse_score": r["score"],
            "dense_score": 0.0,
            "fused_score": 0.0
        }
    
    # 添加dense结果
    for r in dense_results:
        doc_id = r["doc_id"]
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                "sparse_score": 0.0,
                "dense_score": r["score"]
            }
        else:
            doc_scores[doc_id]["dense_score"] = r["score"]
    
    # 计算融合分数
    fused_results = []
    for doc_id, scores in doc_scores.items():
        fused_score = (1 - alpha) * scores["sparse_score"] + alpha * scores["dense_score"]
        fused_results.append({
            "doc_id": doc_id,
            "sparse_score": scores["sparse_score"],
            "dense_score": scores["dense_score"],
            "fused_score": fused_score
        })
    
    # 按融合分数排序
    fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
    
    # 添加排名
    for i, r in enumerate(fused_results):
        r["rank"] = i + 1
    
    return fused_results

# 计算nDCG@k
def compute_ndcg(ranked_list: List[Dict], qrels: Dict, k: int = 10) -> float:
    """
    计算归一化折损累积增益(nDCG)@k
    
    nDCG是信息检索中常用的评估指标，考虑了排序位置对相关性的影响
    
    算法步骤：
    1. 计算DCG (Discounted Cumulative Gain): DCG@k = Σ(rel_i / log2(i+1))
    2. 计算IDCG (Ideal DCG): 理想排序下的DCG
    3. 计算nDCG: nDCG = DCG / IDCG
    
    参数：
    - ranked_list: 排序后的文档列表
    - qrels: 相关性判断 {doc_id: relevance_score}
    - k: 评估位置（只考虑前k个结果）
    
    返回值：nDCG@k分数，范围[0,1]，越高表示检索效果越好
    """
    if not ranked_list or k <= 0:
        return 0.0
    
    # 限制到前k个结果
    ranked_list = ranked_list[:k]
    
    # 计算DCG
    dcg = 0.0
    for i, doc in enumerate(ranked_list):
        doc_id = doc["doc_id"]
        # 获取原始文档ID（如果是分块的文档）
        original_id = doc_id.split("_")[:1][0] if "_chunk_" in doc_id else doc_id
        rel = qrels.get(original_id, 0)
        dcg += rel / np.log2(i + 2)  # i从0开始，所以+2
    
    # 计算IDCG（理想DCG）
    # 获取所有相关文档的相关性分数并排序
    rel_scores = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(rel_scores):
        idcg += rel / np.log2(i + 2)
    
    # 避免除零错误
    if idcg == 0:
        return 0.0
    
    # 计算nDCG
    ndcg = dcg / idcg
    
    return ndcg

# 执行α网格搜索以找到最佳融合权重
def grid_search_alpha(
    queries: List[Dict],
    vectorizer,
    tfidf_matrix,
    tfidf_doc_ids,
    vec_vectorizer,
    dense_matrix,
    vector_doc_ids,
    qrels: Dict,
    alpha_values: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    top_k: int = 10
) -> Tuple[float, List[Dict]]:
    """
    网格搜索找到最佳融合权重α
    返回: (最佳α值, 每个α值对应的平均nDCG@k)
    """
    print("✓ 开始α网格搜索...")
    
    alpha_ndcg = []
    
    for alpha in alpha_values:
        total_ndcg = 0.0
        
        for query in queries:
            # 执行sparse检索
            sparse_results = tfidf_search(query["text"], vectorizer, tfidf_matrix, tfidf_doc_ids, top_k)
            sparse_results = normalize_scores(sparse_results)
            
            # 执行dense检索
            dense_results = vector_search(query["text"], vec_vectorizer, dense_matrix, vector_doc_ids, top_k)
            dense_results = normalize_scores(dense_results)
            
            # 融合结果
            fused_results = fuse_results(sparse_results, dense_results, alpha)
            
            # 计算nDCG@k
            query_id = query["id"]
            query_qrels = qrels.get(query_id, {})
            ndcg = compute_ndcg(fused_results, query_qrels, top_k)
            total_ndcg += ndcg
        
        # 计算平均nDCG@k
        avg_ndcg = total_ndcg / len(queries) if queries else 0.0
        alpha_ndcg.append({
            "alpha": alpha,
            "avg_ndcg": avg_ndcg
        })
        
        print(f"  α = {alpha:.1f}: nDCG@{top_k} = {avg_ndcg:.4f}")
    
    # 找到最佳α值
    best_result = max(alpha_ndcg, key=lambda x: x["avg_ndcg"])
    best_alpha = best_result["alpha"]
    
    print(f"✓ 网格搜索完成，最佳α值: {best_alpha:.1f}, 对应的nDCG@{top_k}: {best_result['avg_ndcg']:.4f}")
    
    return best_alpha, alpha_ndcg

# 使用Cross-Encoder进行重排
def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    corpus: List[Dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    top_m: int = 5
) -> List[Dict]:
    """使用Cross-Encoder对检索结果进行重排"""
    # 注意：此功能需要安装sentence-transformers包
    # pip install sentence-transformers
    try:
        from sentence_transformers import CrossEncoder
        # 加载Cross-Encoder模型
        model = CrossEncoder(model_name)
    except ImportError:
        print("警告：未安装sentence-transformers包，跳过Cross-Encoder重排")
        return candidates[:top_m]
    
    # 创建文档ID到文档内容的映射
    doc_id_to_content = {doc["id"]: doc["full_text"] for doc in corpus}
    
    # 准备用于重排的查询-文档对
    pairs = []
    for candidate in candidates:
        doc_id = candidate["doc_id"]
        content = doc_id_to_content.get(doc_id, "")
        pairs.append([query, content])
    
    # 如果没有候选文档，直接返回空列表
    if not pairs:
        return []
    
    # 预测相关性分数
    scores = model.predict(pairs)
    
    # 更新候选文档的分数
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = scores[i]
    
    # 按重排分数排序
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    # 限制到前top_m个结果并添加排名
    reranked = reranked[:top_m]
    for i, doc in enumerate(reranked):
        doc["rerank_rank"] = i + 1
    
    return reranked

# 评估不同分块策略对检索性能的影响
def evaluate_chunking_strategies(
    corpus: List[Dict],
    queries: List[Dict],
    qrels: Dict,
    config: Dict
) -> List[Dict]:
    """
    评估不同分块策略对检索性能的影响
    """
    print("\n=== 评估不同分块策略的影响 ===")
    
    results = []
    # 修复：正确计算原始文档大小，因为原始文档没有full_text字段
    original_size = sum(len(doc["title"] + ": " + doc["content"]) for doc in corpus)
    
    # 遍历所有分块策略组合
    for chunk_size in config["chunk_sizes"]:
        for chunk_overlap in config["chunk_overlaps"]:
            print(f"\n评估分块策略: 块大小={chunk_size}, 块重叠={chunk_overlap}")
            
            # 对文档进行分块 - 转换为元组以支持缓存
            chunked_corpus = chunk_documents(tuple(corpus), chunk_size, chunk_overlap)
            
            # 计算索引规模增长
            chunked_size = sum(len(doc["full_text"]) for doc in chunked_corpus)
            size_increase = ((chunked_size - original_size) / original_size) * 100
            print(f"  索引规模增长: {size_increase:.1f}%")
            
            # 构建检索模型 - 更新为使用prepare_texts_for_model辅助函数
            doc_texts_tuple = prepare_texts_for_model(chunked_corpus)
            vectorizer, tfidf_matrix, tfidf_doc_ids = build_tfidf_model(doc_texts_tuple)
            # 复用相同的文本元组，因为两个模型使用相同的文本预处理
            vec_vectorizer, dense_matrix, vector_doc_ids = build_vector_model(doc_texts_tuple)
            
            # 计算平均nDCG
            ndcg_scores = []
            
            for query in queries:
                # 执行sparse检索
                sparse_results = tfidf_search(query["text"], vectorizer, tfidf_matrix, tfidf_doc_ids, config["top_k"])
                sparse_results = normalize_scores(sparse_results)
                
                # 执行dense检索
                dense_results = vector_search(query["text"], vec_vectorizer, dense_matrix, vector_doc_ids, config["top_k"])
                dense_results = normalize_scores(dense_results)
                
                # 使用α=0.5进行融合
                fused_results = fuse_results(sparse_results, dense_results, 0.5)
                
                # 修复：qrels的格式问题
                query_id = query["id"]
                query_qrels = qrels.get(query_id, {})
                ndcg = compute_ndcg(fused_results, query_qrels, config["top_k"])
                ndcg_scores.append(ndcg)
            
            # 计算平均nDCG
            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            print(f"  nDCG@{config['top_k']}: {avg_ndcg:.4f}")
            
            # 保存结果
            results.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "size_increase": size_increase,
                "avg_ndcg": avg_ndcg
            })
    
    return results

# 绘制分块策略分析图表
def plot_chunking_analysis(results: List[Dict], output_path: str = "outputs/chunk_analysis.png", top_k: int = 10):
    """
    绘制分块策略对系统性能的影响图表
    """
    plt.figure(figsize=(15, 10))
    
    # 创建子图1：nDCG随分块大小和重叠的变化
    plt.subplot(2, 1, 1)
    
    # 按chunk_size分组
    chunk_sizes = sorted(set(r["chunk_size"] for r in results))
    colors = plt.cm.get_cmap("tab10", len(chunk_sizes))
    
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_overlaps = []
        ndcgs = []
        
        # 获取当前chunk_size的所有结果
        chunk_results = [r for r in results if r["chunk_size"] == chunk_size]
        chunk_results.sort(key=lambda x: x["chunk_overlap"])
        
        for r in chunk_results:
            chunk_overlaps.append(r["chunk_overlap"])
            ndcgs.append(r["avg_ndcg"])
        
        plt.plot(chunk_overlaps, ndcgs, marker='o', label=f'块大小={chunk_size}', color=colors(i))
    
    plt.title("不同分块策略对nDCG性能的影响", fontsize=16)
    plt.xlabel("块重叠大小 (Chunk Overlap)", fontsize=14)
    plt.ylabel(f"平均nDCG@{top_k}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    # 创建子图2：索引规模增长随分块大小和重叠的变化
    plt.subplot(2, 1, 2)
    
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_overlaps = []
        size_increases = []
        
        # 获取当前chunk_size的所有结果
        chunk_results = [r for r in results if r["chunk_size"] == chunk_size]
        chunk_results.sort(key=lambda x: x["chunk_overlap"])
        
        for r in chunk_results:
            chunk_overlaps.append(r["chunk_overlap"])
            size_increases.append(r["size_increase"])
        
        plt.plot(chunk_overlaps, size_increases, marker='s', label=f'块大小={chunk_size}', color=colors(i))
    
    plt.title("不同分块策略对索引规模的影响", fontsize=16)
    plt.xlabel("块重叠大小 (Chunk Overlap)", fontsize=14)
    plt.ylabel("索引规模增长百分比 (%)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ 分块策略分析图表已保存至: {output_path}")
    
    plt.close()

# 主函数
def main():
    """主函数，执行完整流程"""
    print("\n===== 混合检索系统 - 分块策略影响分析 =====")
    
    # 1. 检查并安装依赖
    check_and_install_dependencies()
    
    # 2. 创建必要的目录
    create_directories()
    
    # 3. 加载或生成示例数据 (优化版)
    corpus, queries, qrels = load_or_generate_sample_data()
    
    # 4. 加载配置
    config = load_config()
    
    # 5. 评估分块策略的影响
    chunking_results = evaluate_chunking_strategies(corpus, queries, qrels, config)
    
    # 6. 绘制分块策略分析图表
    chunk_analysis_path = os.path.join("outputs", "chunk_analysis.png")
    plot_chunking_analysis(chunking_results, chunk_analysis_path, config["top_k"])
    
    # 7. 使用原始文档（无分块）执行完整的混合检索流程
    print("\n=== 执行完整的混合检索流程（无分块）===")
    
    # 为原始文档添加full_text字段（与分块后的文档格式保持一致）
    full_corpus = []
    for doc in corpus:
        full_text = doc["title"] + ": " + doc["content"]
        full_corpus.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "full_text": full_text,
            "original_id": doc["id"]
        })
    
    # 构建检索模型 - 更新为使用prepare_texts_for_model辅助函数
    doc_texts_tuple = prepare_texts_for_model(full_corpus)
    vectorizer, tfidf_matrix, tfidf_doc_ids = build_tfidf_model(doc_texts_tuple)
    # 复用相同的文本元组，因为两个模型使用相同的文本预处理
    vec_vectorizer, dense_matrix, vector_doc_ids = build_vector_model(doc_texts_tuple)
    
    # 执行α网格搜索
    best_alpha, alpha_ndcg = grid_search_alpha(
        queries,
        vectorizer,
        tfidf_matrix,
        tfidf_doc_ids,
        vec_vectorizer,
        dense_matrix,
        vector_doc_ids,
        qrels,
        config["alpha_values"],
        config["top_k"]
    )
    
    # 执行混合检索
    hybrid_candidates = []
    
    for query in queries:
        # 执行sparse检索
        sparse_results = tfidf_search(query["text"], vectorizer, tfidf_matrix, tfidf_doc_ids, config["top_k"])
        sparse_results = normalize_scores(sparse_results)
        
        # 执行dense检索
        dense_results = vector_search(query["text"], vec_vectorizer, dense_matrix, vector_doc_ids, config["top_k"])
        dense_results = normalize_scores(dense_results)
        
        # 融合结果
        fused_results = fuse_results(sparse_results, dense_results, best_alpha)
        
        # 保存混合检索结果
        for r in fused_results:
            hybrid_candidates.append({
                "query_id": query["id"],
                "query_text": query["text"],
                **r
            })
    
    # 保存结果到文件
    hybrid_candidates_path = os.path.join("outputs", "hybrid_candidates.jsonl")
    with open(hybrid_candidates_path, "w", encoding="utf-8") as f:
        for candidate in hybrid_candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    
    print(f"✓ 混合检索结果已保存至: {hybrid_candidates_path}")
    
    # 8. 打印分块策略分析摘要
    print("\n=== 分块策略影响分析摘要 ===")
    for result in chunking_results:
        print(f"块大小={result['chunk_size']}, 块重叠={result['chunk_overlap']}: ")
        print(f"  索引规模增长: {result['size_increase']:.1f}%, nDCG@{config['top_k']}: {result['avg_ndcg']:.4f}")
    
    print("\n分析说明:")
    print("1. 分块可以提高关键信息的覆盖率，但会增加索引规模")
    print("2. 较大的块重叠可以改善信息连续性，但会进一步增加索引规模")
    print("3. 最佳分块策略需要在性能提升和资源消耗之间找到平衡")
    print("4. 在实际应用中，应根据文档特点和系统资源选择合适的分块参数")
    
    print("\n===== 程序执行完成 =====")

# 程序入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)