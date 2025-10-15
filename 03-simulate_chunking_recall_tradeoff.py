#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
程序名称：文本分块与重叠效果评估
作者：Ph.D. Rhino
功能说明：模拟不同窗口与重叠对召回与前列覆盖的影响（基于关键词近似）
内容概述：使用标准库将长文档按不同窗口与重叠切分；用简单关键词匹配模拟检索召回；
          统计命中片段在前列的比例，估算Recall近似值。
执行流程：
1. 读取文档集合（自动生成示例数据），接受窗口与重叠参数
2. 对每个query的关键词集进行匹配，记录命中chunk与位置
3. 统计不同配置下的"命中chunk比例"和"前K片段命中率"
4. 计算成本近似（片段数、总字符数）
5. 输出配置-指标-成本的对照表
输入说明：--win 256 --overlap 0.15 --k 5 --queries queries.csv
输出展示：
Config(win=256,ov=0.15): hit_chunks=0.78 topK_hit=0.69 cost_units=1.32x
Config(win=384,ov=0.10): hit_chunks=0.82 topK_hit=0.73 cost_units=1.18x
Summary saved: chunking_grid_summary.csv
"""

import os
import sys
import argparse
import random
import json
import csv
import time
from collections import defaultdict

# 检查并安装必要的依赖
def check_and_install_dependencies():
    """检查并自动安装必要的依赖包"""
    try:
        # 导入可能需要的第三方库
        import numpy as np
    except ImportError:
        print("正在安装必要的依赖包...")
        try:
            # 使用pip安装依赖
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            print("依赖包安装成功！")
        except Exception as e:
            print(f"安装依赖包时出错: {e}")
            sys.exit(1)

# 生成示例数据集
def generate_sample_data(num_docs=10, max_tokens=5000):
    """
    生成示例文档数据集
    参数:
        num_docs: 生成的文档数量
        max_tokens: 每个文档的最大字符数
    返回:
        文档集合字典，格式: {doc_id: {'text': 文档内容, 'keywords': 关键词列表}}
    """
    # 示例主题和关键词
    topics = {
        "人工智能": ["深度学习", "神经网络", "自然语言处理", "计算机视觉", "机器学习", "模型训练", "数据标注", "特征提取"],
        "软件工程": ["敏捷开发", "代码重构", "测试驱动", "持续集成", "微服务", "架构设计", "版本控制", "DevOps"],
        "医疗健康": ["基因编辑", "精准医疗", "疫苗研发", "药物发现", "临床试验", "医学影像", "患者护理", "健康管理"],
        "金融科技": ["区块链", "智能合约", "量化交易", "风险评估", "支付系统", "数字货币", "监管科技", "财富管理"],
        "环境科学": ["气候变化", "可再生能源", "碳排放", "生态保护", "可持续发展", "污染治理", "资源循环", "绿色技术"]
    }
    
    # 示例句子模板
    templates = [
        "{topic}领域的最新发展表明，{keyword1}和{keyword2}正在成为行业的焦点。",
        "近年来，{keyword1}技术在{topic}领域取得了突破性进展，特别是在{keyword2}方面的应用。",
        "专家指出，{topic}的未来发展将很大程度上依赖于{keyword1}和{keyword2}的创新应用。",
        "随着{keyword1}技术的成熟，{topic}行业正面临前所未有的变革机遇，尤其是在{keyword2}领域。",
        "在当前{topic}研究中，{keyword1}和{keyword2}的结合已成为解决复杂问题的重要途径。"
    ]
    
    documents = {}
    all_queries = []
    
    for i in range(num_docs):
        topic = random.choice(list(topics.keys()))
        topic_keywords = topics[topic]
        doc_keywords = random.sample(topic_keywords, k=random.randint(3, 5))
        
        # 生成文档内容
        doc_content = []
        current_length = 0
        
        while current_length < max_tokens * 0.7:
            template = random.choice(templates)
            keyword1, keyword2 = random.sample(doc_keywords, 2)
            sentence = template.format(topic=topic, keyword1=keyword1, keyword2=keyword2)
            doc_content.append(sentence)
            current_length += len(sentence)
        
        doc_text = "\n".join(doc_content)
        documents[f"doc_{i+1}"] = {
            "text": doc_text,
            "keywords": doc_keywords
        }
        
        # 为每个文档生成一个查询
        query_keywords = random.sample(doc_keywords, k=random.randint(1, 2))
        query_text = f"关于{topic}中的{'和'.join(query_keywords)}的信息"
        all_queries.append({
            "query_id": f"query_{i+1}",
            "query_text": query_text,
            "relevant_keywords": query_keywords,
            "relevant_docs": [f"doc_{i+1}"]  # 假设每个查询只与一个文档相关
        })
    
    # 保存文档数据
    with open("sample_docs.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    # 保存查询数据
    with open("queries.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text", "relevant_keywords", "relevant_docs"])
        writer.writeheader()
        for query in all_queries:
            # 将列表转换为字符串以便CSV存储
            query["relevant_keywords"] = ",".join(query["relevant_keywords"])
            query["relevant_docs"] = ",".join(query["relevant_docs"])
            writer.writerow(query)
    
    print(f"生成示例数据完成：{num_docs}个文档和{len(all_queries)}个查询")
    return documents, all_queries

# 文档分块函数
def chunk_document(doc_text, window_size, overlap_ratio):
    """
    将文档按指定窗口大小和重叠比例进行分块
    参数:
        doc_text: 文档文本内容
        window_size: 分块窗口大小（字符数）
        overlap_ratio: 重叠比例（0-1之间）
    返回:
        chunks: 分块后的文档片段列表
    """
    overlap = int(window_size * overlap_ratio)
    step = window_size - overlap
    
    chunks = []
    for i in range(0, len(doc_text), step):
        end = min(i + window_size, len(doc_text))
        chunk = doc_text[i:end]
        chunks.append(chunk)
    
    return chunks

# 模拟检索函数
def simulate_retrieval(doc_chunks, query_keywords, k=5):
    """
    模拟基于关键词的检索过程
    参数:
        doc_chunks: 文档的分块列表
        query_keywords: 查询的关键词列表
        k: 返回的前k个结果
    返回:
        results: 排序后的结果列表[(chunk_index, score)]
    """
    chunk_scores = []
    
    for i, chunk in enumerate(doc_chunks):
        # 计算关键词匹配得分
        score = sum(1 for keyword in query_keywords if keyword in chunk)
        if score > 0:
            chunk_scores.append((i, score))
    
    # 按得分降序排序，如果得分相同则按块索引升序排序
    chunk_scores.sort(key=lambda x: (-x[1], x[0]))
    
    # 返回前k个结果
    return chunk_scores[:k]

# 评估分块配置
def evaluate_chunking_config(documents, queries, window_size, overlap_ratio, k=5):
    """
    评估指定分块配置的性能
    参数:
        documents: 文档集合
        queries: 查询集合
        window_size: 窗口大小
        overlap_ratio: 重叠比例
        k: 前k评估值
    返回:
        metrics: 评估指标字典
    """
    total_hit_chunks = 0  # 命中的chunk总数
    total_possible_chunks = 0  # 可能命中的chunk总数
    total_topK_hits = 0  # 在前k个结果中命中的查询数
    total_chunks = 0  # 所有文档的总块数
    total_characters = 0  # 所有块的总字符数
    
    for query in queries:
        query_keywords = query["relevant_keywords"].split(",")
        relevant_docs = query["relevant_docs"].split(",")
        
        for doc_id in relevant_docs:
            if doc_id not in documents:
                continue
            
            doc_text = documents[doc_id]["text"]
            
            # 分块文档
            chunks = chunk_document(doc_text, window_size, overlap_ratio)
            total_chunks += len(chunks)
            total_characters += sum(len(chunk) for chunk in chunks)
            
            # 找出包含关键词的块（作为理想情况）
            relevant_chunks = []
            for i, chunk in enumerate(chunks):
                if any(keyword in chunk for keyword in query_keywords):
                    relevant_chunks.append(i)
            
            total_possible_chunks += len(relevant_chunks)
            
            # 模拟检索
            retrieval_results = simulate_retrieval(chunks, query_keywords, k)
            retrieved_chunk_indices = [idx for idx, _ in retrieval_results]
            
            # 计算命中的块数
            hits = len(set(relevant_chunks) & set(retrieved_chunk_indices))
            total_hit_chunks += hits
            
            # 检查前k个结果中是否有命中
            if hits > 0:
                total_topK_hits += 1
    
    # 计算指标
    hit_chunks_ratio = total_hit_chunks / total_possible_chunks if total_possible_chunks > 0 else 0
    topK_hit_rate = total_topK_hits / len(queries) if len(queries) > 0 else 0
    
    metrics = {
        "window_size": window_size,
        "overlap_ratio": overlap_ratio,
        "hit_chunks_ratio": hit_chunks_ratio,
        "topK_hit_rate": topK_hit_rate,
        "total_chunks": total_chunks,
        "total_characters": total_characters
    }
    
    return metrics

# 计算成本单位（相对于基准配置）
def calculate_cost_units(metrics, base_metrics):
    """
    计算相对于基准配置的成本单位
    参数:
        metrics: 当前配置的指标
        base_metrics: 基准配置的指标
    返回:
        cost_units: 成本单位
    """
    # 简单平均成本（块数和字符数）
    chunk_cost = metrics["total_chunks"] / base_metrics["total_chunks"] if base_metrics["total_chunks"] > 0 else 0
    char_cost = metrics["total_characters"] / base_metrics["total_characters"] if base_metrics["total_characters"] > 0 else 0
    
    cost_units = (chunk_cost + char_cost) / 2
    return cost_units

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="模拟不同窗口与重叠对召回与前列覆盖的影响")
    parser.add_argument("--win", type=int, default=256, help="窗口大小")
    parser.add_argument("--overlap", type=float, default=0.15, help="重叠比例")
    parser.add_argument("--k", type=int, default=5, help="前K评估值")
    parser.add_argument("--queries", type=str, default="queries.csv", help="查询文件路径")
    parser.add_argument("--docs", type=str, default="sample_docs.json", help="文档文件路径")
    parser.add_argument("--grid", action="store_true", help="是否运行网格搜索评估多种配置")
    
    args = parser.parse_args()
    
    # 检查并安装依赖
    check_and_install_dependencies()
    
    start_time = time.time()
    
    # 检查数据文件是否存在，不存在则生成
    if not os.path.exists(args.docs) or not os.path.exists(args.queries):
        print("数据文件不存在，正在生成示例数据...")
        documents, queries = generate_sample_data()
    else:
        # 读取文档数据
        with open(args.docs, "r", encoding="utf-8") as f:
            documents = json.load(f)
        
        # 读取查询数据
        queries = []
        with open(args.queries, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row)
        
        print(f"读取数据完成：{len(documents)}个文档和{len(queries)}个查询")
    
    # 评估分块配置
    if args.grid:
        # 定义多种分块配置进行网格搜索
        window_sizes = [128, 256, 384, 512]
        overlap_ratios = [0.10, 0.15, 0.20]
        all_metrics = []
        
        # 首先评估一个基准配置用于计算相对成本
        base_window = window_sizes[0]
        base_overlap = overlap_ratios[0]
        base_metrics = evaluate_chunking_config(documents, queries, base_window, base_overlap, args.k)
        
        # 添加基准配置到结果列表
        base_metrics["cost_units"] = 1.0  # 基准配置的成本单位为1.0
        all_metrics.append(base_metrics)
        
        # 评估其他配置
        for win in window_sizes:
            for ov in overlap_ratios:
                if win == base_window and ov == base_overlap:
                    continue  # 跳过基准配置，已经评估过了
                
                metrics = evaluate_chunking_config(documents, queries, win, ov, args.k)
                metrics["cost_units"] = calculate_cost_units(metrics, base_metrics)
                all_metrics.append(metrics)
                
                # 打印当前配置的结果
                print(f"Config(win={win},ov={ov:.2f}): hit_chunks={metrics['hit_chunks_ratio']:.2f} topK_hit={metrics['topK_hit_rate']:.2f} cost_units={metrics['cost_units']:.2f}x")
    else:
        # 只评估指定的单个配置
        all_metrics = [evaluate_chunking_config(documents, queries, args.win, args.overlap, args.k)]
        print(f"Config(win={args.win},ov={args.overlap:.2f}): hit_chunks={all_metrics[0]['hit_chunks_ratio']:.2f} topK_hit={all_metrics[0]['topK_hit_rate']:.2f}")
    
    # 保存结果到CSV文件
    output_file = "chunking_grid_summary.csv"
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["window_size", "overlap_ratio", "hit_chunks_ratio", "topK_hit_rate", "total_chunks", "total_characters"]
        if "cost_units" in all_metrics[0]:
            fieldnames.append("cost_units")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow(metrics)
    
    print(f"Summary saved: {output_file}")
    
    # 检查执行效率
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time:.2f}秒")
    
    # 检查输出文件的中文是否有乱码
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        print("输出文件的中文显示正常，没有乱码问题")
    except UnicodeDecodeError:
        print("警告：输出文件可能存在中文乱码问题")
    
    # 实验结果分析
    print("\n实验结果分析：")
    if args.grid and len(all_metrics) > 1:
        # 找出最佳性能的配置（平衡hit_chunks_ratio和cost_units）
        best_config = None
        best_score = -float('inf')
        
        for metrics in all_metrics:
            # 简单评分：hit_chunks_ratio * (2 - cost_units)，惩罚高成本
            score = metrics['hit_chunks_ratio'] * (2 - min(metrics.get('cost_units', 1), 2))
            if score > best_score:
                best_score = score
                best_config = metrics
        
        print(f"最佳平衡配置: window_size={best_config['window_size']}, overlap_ratio={best_config['overlap_ratio']}")
        print(f"  命中块比例: {best_config['hit_chunks_ratio']:.2f}")
        print(f"  前{args.k}命中率: {best_config['topK_hit_rate']:.2f}")
        if 'cost_units' in best_config:
            print(f"  相对成本: {best_config['cost_units']:.2f}x")
        
        # 分析趋势
        print("\n趋势分析：")
        # 按窗口大小分组分析
        win_groups = defaultdict(list)
        for metrics in all_metrics:
            win_groups[metrics['window_size']].append(metrics)
        
        for win, group in sorted(win_groups.items()):
            avg_hit = sum(m['hit_chunks_ratio'] for m in group) / len(group)
            avg_cost = sum(m.get('cost_units', 1) for m in group) / len(group)
            print(f"  窗口大小 {win}: 平均命中比例={avg_hit:.2f}, 平均相对成本={avg_cost:.2f}x")
    
    print("\n程序完成！")

if __name__ == "__main__":
    main()