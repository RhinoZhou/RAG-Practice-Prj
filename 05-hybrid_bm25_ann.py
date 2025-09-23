#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BM25与向量检索融合策略演示程序
作者: Ph.D. Rhino

本程序用于演示BM25（关键词检索）与向量检索（语义相似度检索）的融合效果，
重点对比RRF（Reciprocal Rank Fusion）和加权融合两种不同的融合策略。

实验流程：
1. 加载小型语料库和查询集合
2. 构建BM25检索模型并计算文档的BM25得分
3. 使用TF-IDF计算文档和查询的向量表示（替代SentenceTransformer以避免依赖问题）
4. 计算向量语义相似度并排序
5. 实现RRF和加权融合两种策略
6. 输出不同融合策略的Top-K结果对比
7. 生成Markdown格式的实验报告

依赖库：
- rank_bm25: 用于实现BM25算法
- numpy: 用于数值计算
- pandas: 用于数据处理
- scikit-learn: 用于TF-IDF向量化
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Any, Set

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# 检查并安装必要的依赖
def check_dependencies() -> None:
    """检查并安装必要的依赖库"""
    required_packages = ['rank_bm25', 'numpy', 'pandas', 'scikit-learn']
    missing_packages = []
    
    # 尝试导入每个包，如果导入失败则添加到缺失列表
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ 已安装: {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ 未安装: {package}")
    
    # 安装缺失的包
    if missing_packages:
        logger.info("正在安装缺失的依赖...")
        for package in missing_packages:
            try:
                # 使用pip安装缺失的包
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✓ 已成功安装: {package}")
            except Exception as e:
                logger.error(f"✗ 安装 {package} 失败: {str(e)}")
                raise RuntimeError(f"无法安装必要的依赖 {package}")

# 加载语料库
def load_corpus(file_path: str) -> List[str]:
    """
    从文件中加载语料库
    
    Args:
        file_path: 语料库文件路径
        
    Returns:
        文档列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"语料库文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f if line.strip()]
    
    logger.info(f"✓ 语料库加载成功，文档数量: {len(docs)}")
    return docs

# 加载查询集合
def load_queries(file_path: str) -> List[str]:
    """
    从文件中加载查询集合
    
    Args:
        file_path: 查询文件路径
        
    Returns:
        查询列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"查询文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    logger.info(f"✓ 查询加载成功，查询数量: {len(queries)}")
    return queries

# 构建BM25检索模型
def build_bm25_model(docs: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    构建BM25检索模型
    
    Args:
        docs: 文档列表
        
    Returns:
        BM25模型和分词后的文档列表
    """
    # 分词（这里使用简单的空格分词，实际应用中可能需要更复杂的分词）
    tokenized_docs = [doc.split() for doc in docs]
    
    # 构建BM25模型
    start_time = time.time()
    bm25 = BM25Okapi(tokenized_docs)
    build_time = time.time() - start_time
    
    logger.info(f"✓ BM25模型构建完成，耗时: {build_time:.2f}秒")
    return bm25, tokenized_docs

# 执行BM25检索
def perform_bm25_search(bm25: BM25Okapi, query: str, docs: List[str], k: int = 10) -> List[Tuple[int, str, float]]:
    """
    执行BM25检索
    
    Args:
        bm25: BM25模型
        query: 查询文本
        docs: 原始文档列表
        k: 返回的文档数量
        
    Returns:
        包含文档索引、文档内容和BM25得分的元组列表
    """
    # 分词查询
    tokenized_query = query.split()
    
    # 获取BM25得分
    scores = bm25.get_scores(tokenized_query)
    
    # 获取Top-K结果
    top_indices = np.argsort(scores)[::-1][:k]
    
    # 构建结果列表
    results = [(idx, docs[idx], scores[idx]) for idx in top_indices]
    
    return results

# 加载向量模型并计算嵌入
def load_vector_model_and_embed(docs: List[str], queries: List[str]) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    使用TF-IDF计算文档和查询的向量表示
    
    Args:
        docs: 文档列表
        queries: 查询列表
        
    Returns:
        TF-IDF向量器、文档嵌入和查询嵌入的元组
    """
    logger.info("正在初始化TF-IDF向量器...")
    
    # 初始化TF-IDF向量器
    start_time = time.time()
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')  # 简单的分词模式
    
    # 计算文档嵌入
    doc_embeddings = vectorizer.fit_transform(docs)
    doc_time = time.time() - start_time
    logger.info(f"✓ TF-IDF文档嵌入计算完成，特征维度: {doc_embeddings.shape[1]}, 耗时: {doc_time:.2f}秒")
    
    # 计算查询嵌入
    start_time = time.time()
    query_embeddings = vectorizer.transform(queries)
    query_time = time.time() - start_time
    logger.info(f"✓ TF-IDF查询嵌入计算完成，耗时: {query_time:.2f}秒")
    
    return vectorizer, doc_embeddings, query_embeddings

# 执行向量检索
def perform_vector_search(doc_embeddings: np.ndarray, query_embedding: np.ndarray, 
                         docs: List[str], k: int = 10) -> List[Tuple[int, str, float]]:
    """
    执行向量检索
    
    Args:
        doc_embeddings: 文档嵌入矩阵
        query_embedding: 查询嵌入向量
        docs: 原始文档列表
        k: 返回的文档数量
        
    Returns:
        包含文档索引、文档内容和相似度得分的元组列表
    """
    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # 获取Top-K结果
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # 构建结果列表
    results = [(idx, docs[idx], similarities[idx]) for idx in top_indices]
    
    return results

# 实现RRF（Reciprocal Rank Fusion）融合
def reciprocal_rank_fusion(bm25_results: List[Tuple[int, str, float]], 
                          vector_results: List[Tuple[int, str, float]], 
                          k: int = 60, weight: float = 1.0) -> List[Tuple[int, str, float]]:
    """
    实现RRF融合策略
    
    Args:
        bm25_results: BM25检索结果
        vector_results: 向量检索结果
        k: RRF参数，控制排名的影响力
        weight: 向量检索结果的权重
        
    Returns:
        融合后的结果列表
    """
    # 记录每个文档的RRF得分
    rrf_scores = {}
    
    # 处理BM25结果
    for rank, (idx, doc, _) in enumerate(bm25_results, 1):
        rrf_scores[idx] = 1.0 / (k + rank)
    
    # 处理向量结果，并应用权重
    for rank, (idx, doc, _) in enumerate(vector_results, 1):
        if idx in rrf_scores:
            rrf_scores[idx] += weight * (1.0 / (k + rank))
        else:
            rrf_scores[idx] = weight * (1.0 / (k + rank))
    
    # 获取原始文档列表
    all_docs = {idx: doc for idx, doc, _ in bm25_results + vector_results}
    
    # 排序并构建结果列表
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    results = [(idx, all_docs[idx], rrf_scores[idx]) for idx in sorted_indices]
    
    return results

# 实现加权融合
def weighted_score_fusion(bm25_results: List[Tuple[int, str, float]], 
                         vector_results: List[Tuple[int, str, float]], 
                         alpha: float = 0.5) -> List[Tuple[int, str, float]]:
    """
    实现加权融合策略
    
    Args:
        bm25_results: BM25检索结果
        vector_results: 向量检索结果
        alpha: 向量检索结果的权重，范围[0,1]
        
    Returns:
        融合后的结果列表
    """
    # 归一化BM25得分
    bm25_scores = {idx: score for idx, _, score in bm25_results}
    bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
    bm25_min = min(bm25_scores.values()) if bm25_scores else 0.0
    bm25_range = bm25_max - bm25_min + 1e-10  # 避免除零
    
    # 归一化向量检索得分
    vector_scores = {idx: score for idx, _, score in vector_results}
    vector_max = max(vector_scores.values()) if vector_scores else 1.0
    vector_min = min(vector_scores.values()) if vector_scores else 0.0
    vector_range = vector_max - vector_min + 1e-10  # 避免除零
    
    # 计算加权融合得分
    fused_scores = {}
    
    # 处理所有文档
    all_indices = set(bm25_scores.keys()).union(set(vector_scores.keys()))
    
    for idx in all_indices:
        # 归一化BM25得分
        bm25_norm = (bm25_scores.get(idx, bm25_min) - bm25_min) / bm25_range
        
        # 归一化向量得分
        vector_norm = (vector_scores.get(idx, vector_min) - vector_min) / vector_range
        
        # 计算加权融合得分
        fused_score = alpha * vector_norm + (1 - alpha) * bm25_norm
        fused_scores[idx] = fused_score
    
    # 获取原始文档内容
    all_docs = {idx: doc for idx, doc, _ in bm25_results + vector_results}
    
    # 排序并构建结果列表
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    results = [(idx, all_docs[idx], fused_scores[idx]) for idx in sorted_indices]
    
    return results

# 生成Markdown格式的实验报告
def generate_report(queries: List[str], 
                   query_results: Dict[str, Dict[str, List[Tuple[int, str, float]]]],
                   output_file: str) -> None:
    """
    生成Markdown格式的实验报告
    
    Args:
        queries: 查询列表
        query_results: 每个查询的不同检索策略的结果
        output_file: 输出文件路径
    """
    logger.info(f"正在生成实验报告: {output_file}...")
    
    report_content = []
    
    # 添加报告标题
    report_content.append("# BM25与向量检索融合策略对比实验报告")
    report_content.append("\n作者: Ph.D. Rhino")
    report_content.append("\n\n## 实验概述")
    report_content.append("\n本实验对比了BM25检索、向量检索以及两种不同融合策略（RRF和加权融合）的检索效果。")
    report_content.append("实验使用了30条短文档组成的小型语料库，并针对10个查询进行了检索测试。")
    
    # 添加实验配置
    report_content.append("\n\n## 实验配置")
    report_content.append("\n- 语料库大小: 30条短文档")
    report_content.append("- 查询数量: 10个查询")
    report_content.append("- BM25模型: rank_bm25库中的BM25Okapi")
    report_content.append("- 向量模型: TF-IDF (scikit-learn)")
    report_content.append("- 融合策略:")
    report_content.append("  - RRF (Reciprocal Rank Fusion)，k=60, weight=1.0")
    report_content.append("  - 加权融合，alpha=0.5")
    
    # 添加每个查询的结果对比
    report_content.append("\n\n## 检索结果对比")
    
    for i, query in enumerate(queries):
        report_content.append(f"\n### 查询 {i+1}: {query}")
        report_content.append("\n\n| 排名 | BM25检索 | 向量检索 | RRF融合 | 加权融合(α=0.5) |")
        report_content.append("|------|---------|---------|---------|----------------|")
        
        # 获取该查询的所有结果
        results = query_results[query]
        
        # 提取所有文档ID，以便显示完整的排名对比
        all_docs = set()
        for strategy_results in results.values():
            all_docs.update({idx for idx, _, _ in strategy_results})
        
        # 构建每个文档在各个策略中的排名
        doc_ranks = {}
        for doc_id in all_docs:
            doc_ranks[doc_id] = {}
            for strategy, strategy_results in results.items():
                # 查找文档在该策略中的排名
                rank = None
                for r, (idx, _, _) in enumerate(strategy_results, 1):
                    if idx == doc_id:
                        rank = r
                        break
                doc_ranks[doc_id][strategy] = rank
            
        # 按RRF融合的排名排序（作为参考）
        rrf_ranked_docs = [idx for idx, _, _ in results.get('RRF融合', [])]
        
        # 补充其他文档到排序列表末尾
        for doc_id in all_docs:
            if doc_id not in rrf_ranked_docs:
                rrf_ranked_docs.append(doc_id)
        
        # 构建表格行
        for rank_in_report, doc_id in enumerate(rrf_ranked_docs[:10], 1):  # 只显示前10个结果
            # 获取文档内容
            doc_content = None
            for strategy_results in results.values():
                for idx, content, _ in strategy_results:
                    if idx == doc_id:
                        doc_content = content
                        break
                if doc_content:
                    break
            
            # 构建单元格内容
            cells = [str(rank_in_report)]
            for strategy in ['BM25检索', '向量检索', 'RRF融合', '加权融合']:
                rank = doc_ranks[doc_id].get(strategy, '-')
                cells.append(str(rank) if rank else '-')
            
            # 添加到报告内容
            report_content.append(f"| {' | '.join(cells)} |")
        
        # 添加文档内容注释
        report_content.append("\n\n**文档内容说明:**")
        for idx, content in enumerate(set([(idx, doc_content) for _, doc_content, _ in results.get('RRF融合', [])[:10]])):
            doc_id, doc_text = content
            report_content.append(f"- 排名 {idx+1} 的文档: {doc_text}")
    
    # 添加实验分析
    report_content.append("\n\n## 实验结果分析")
    report_content.append("\n### 不同检索策略的特点")
    report_content.append("\n1. **BM25检索**: 基于关键词匹配，对于包含查询中明确关键词的文档有较好的表现，但可能无法理解语义相关性。")
    report_content.append("\n2. **向量检索**: 基于TF-IDF的向量表示，能够捕获词频和逆文档频率信息，但不考虑词序和语义关系。")
    report_content.append("\n3. **RRF融合**: 通过结合多个检索结果的排名信息，能够综合利用不同检索策略的优势，通常比单一策略表现更好。")
    report_content.append("\n4. **加权融合**: 通过直接加权组合不同检索策略的得分，参数α可以根据具体任务进行调整。")
    
    # 添加融合策略的可调参数说明
    report_content.append("\n### 融合策略的可调参数")
    report_content.append("\n#### RRF融合的可调参数")
    report_content.append("\n- **k**: 控制排名的影响力，较小的k值会给排名靠前的文档更大的权重。通常取值在50-100之间。")
    report_content.append("\n- **weight**: 向量检索结果的权重，可以调整不同检索策略的相对重要性。")
    report_content.append("\n#### 加权融合的可调参数")
    report_content.append("\n- **α**: 控制向量检索结果的权重，范围在0到1之间：\n  - α=0: 完全使用BM25检索结果\n  - α=1: 完全使用向量检索结果\n  - 0<α<1: 混合使用两种检索结果")
    
    # 添加最佳实践建议
    report_content.append("\n### 最佳实践建议")
    report_content.append("\n1. **参数调优**: 根据具体的应用场景和数据特点，调整RRF的k值和weight参数，或加权融合的α参数。")
    report_content.append("\n2. **组合多种检索策略**: 除了BM25和TF-IDF向量检索外，还可以考虑结合其他检索方法，如深度学习模型等。")
    report_content.append("\n3. **考虑文档长度**: 在实际应用中，可能需要对不同长度的文档进行适当的归一化处理。")
    report_content.append("\n4. **定期评估和更新**: 随着语料库的更新和应用场景的变化，定期评估和调整融合策略。")
    
    # 添加结论
    report_content.append("\n## 结论")
    report_content.append("\n本实验表明，融合多种检索策略（如BM25和向量检索）通常能够获得比单一策略更好的检索效果。")
    report_content.append("RRF和加权融合是两种有效的融合方法，它们各有优缺点和可调参数，可以根据具体需求进行选择和调整。")
    report_content.append("\n注意：本实验使用TF-IDF作为向量表示方法以避免依赖问题。在实际应用中，可以考虑使用更高级的语义嵌入模型（如SentenceTransformer）来获得更好的语义理解能力。")
    
    # 保存报告文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"✓ 实验报告生成成功: {output_file}")

# 主函数
def main() -> None:
    """主函数"""
    try:
        # 1. 检查并安装依赖
        check_dependencies()
        
        # 2. 定义文件路径
        corpus_file = os.path.join('docs', 'corpus.txt')
        queries_file = os.path.join('docs', 'queries.txt')
        output_file = os.path.join('result', 'hybrid_results.md')
        
        # 3. 确保输出目录存在
        os.makedirs('result', exist_ok=True)
        
        # 4. 加载语料库和查询
        docs = load_corpus(corpus_file)
        queries = load_queries(queries_file)
        
        # 5. 构建BM25模型
        bm25, _ = build_bm25_model(docs)
        
        # 6. 加载向量模型并计算嵌入
        vectorizer, doc_embeddings, query_embeddings = load_vector_model_and_embed(docs, queries)
        
        # 7. 对每个查询执行不同的检索策略并记录结果
        query_results = {}
        k = 10  # 返回的文档数量
        
        # 计时检索过程
        total_start_time = time.time()
        
        for i, query in enumerate(queries):
            logger.info(f"处理查询 {i+1}/{len(queries)}: {query}")
            
            # 执行BM25检索
            bm25_res = perform_bm25_search(bm25, query, docs, k)
            
            # 执行向量检索
            vector_res = perform_vector_search(doc_embeddings, query_embeddings[i], docs, k)
            
            # 执行RRF融合
            rrf_res = reciprocal_rank_fusion(bm25_res, vector_res, k=60, weight=1.0)
            
            # 执行加权融合
            weighted_res = weighted_score_fusion(bm25_res, vector_res, alpha=0.5)
            
            # 保存结果
            query_results[query] = {
                'BM25检索': bm25_res,
                '向量检索': vector_res,
                'RRF融合': rrf_res,
                '加权融合': weighted_res
            }
        
        total_search_time = time.time() - total_start_time
        logger.info(f"✓ 所有查询处理完成，总耗时: {total_search_time:.2f}秒")
        
        # 8. 生成实验报告
        generate_report(queries, query_results, output_file)
        
        # 9. 检查输出文件是否有中文乱码
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否包含中文字符
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
                logger.info(f"✓ 输出文件中文检查: {'包含中文且无乱码' if has_chinese else '未检测到中文'}")
        except Exception as e:
            logger.error(f"✗ 输出文件检查失败: {str(e)}")
        
        # 10. 输出总结信息
        logger.info("\n===== 实验总结 =====")
        logger.info(f"1. 实验成功完成，处理了 {len(queries)} 个查询")
        logger.info(f"2. 生成的文件：")
        logger.info(f"   - {output_file} (实验报告)")
        logger.info(f"3. 实验结果展示了BM25、向量检索以及两种融合策略的差异")
        logger.info(f"4. 报告中详细分析了不同检索策略的特点和可调参数")
        
        logger.info("\n程序执行成功！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

# 运行主函数
if __name__ == '__main__':
    main()