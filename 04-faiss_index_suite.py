#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS索引类型与参数扫描套件

功能说明：
- 在同一嵌入集合上构建多种FAISS索引类型（Flat、IVF、HNSW、IVF-PQ）
- 对比IP/L2距离度量与单位化设置的召回率/延迟表现
- 扫描关键参数（nlist/nprobe、M/efSearch）生成延迟-召回曲线
- 模拟分片查询与全局Top-K合并延迟分析

输入：
- data/embeddings.npz（可选自动生成）
- configs/index_sweep.json（索引参数配置）

输出：
- outputs/index_compare.csv（索引性能对比表格）
- outputs/latency_recall_curves.png（延迟-召回率曲线图）
- outputs/shard_merge_stats.json（分片合并统计数据）

作者：Ph.D. Rhino
版本：1.0
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

# 设置中文字体，确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 检查并安装依赖包
def check_and_install_dependencies():
    """检查并自动安装必要的依赖包"""
    required_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "faiss-cpu>=1.7.0"
    ]
    
    try:
        # 检查是否已安装所有必要的包
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import faiss
        print("✓ 所有依赖包已安装完成")
    except ImportError:
        print("正在安装必要的依赖包...")
        # 使用pip安装缺失的包
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ 成功安装 {package}")
            except Exception as e:
                print(f"✗ 安装 {package} 失败: {str(e)}")
                raise RuntimeError(f"无法安装必要的依赖包，请手动安装: {package}") from e

# 创建必要的目录
def create_directories():
    """创建数据、配置和输出目录"""
    directories = ["data", "configs", "outputs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 创建目录: {directory}")

# 生成示例嵌入数据
def generate_sample_embeddings(embedding_dim: int = 768, db_size: int = 10000, query_size: int = 1000):
    """
    生成示例嵌入数据用于测试
    embedding_dim: 嵌入维度
    db_size: 数据库嵌入数量
    query_size: 查询嵌入数量
    """
    print(f"✓ 生成示例嵌入数据 (维度: {embedding_dim}, 数据库大小: {db_size}, 查询大小: {query_size})")
    
    # 生成随机嵌入向量
    np.random.seed(42)  # 设置随机种子以确保结果可复现
    db_vectors = np.random.random((db_size, embedding_dim)).astype('float32')
    query_vectors = np.random.random((query_size, embedding_dim)).astype('float32')
    
    # 为了使测试更有意义，确保每个查询向量在数据库中有一定数量的相似向量
    ground_truth = {}
    for i in range(query_size):
        # 每个查询选择5个数据库向量作为相似向量
        similar_indices = np.random.choice(db_size, size=5, replace=False)
        # 稍微调整这些向量以使其更相似
        for idx in similar_indices:
            db_vectors[idx] = query_vectors[i] + 0.1 * np.random.randn(embedding_dim)
        ground_truth[i] = similar_indices.tolist()
    
    # 保存数据
    embeddings_path = "data/embeddings.npz"
    np.savez_compressed(embeddings_path, 
                        db_vectors=db_vectors,
                        query_vectors=query_vectors,
                        ground_truth=ground_truth,
                        embedding_dim=embedding_dim)
    
    print(f"✓ 示例嵌入数据已保存至: {embeddings_path}")
    return db_vectors, query_vectors, ground_truth

# 加载嵌入数据
def load_embeddings(embeddings_path: str = "data/embeddings.npz"):
    """加载嵌入数据，如果不存在则生成示例数据"""
    if not os.path.exists(embeddings_path):
        print(f"✗ 未找到嵌入数据文件: {embeddings_path}，将生成示例数据")
        return generate_sample_embeddings()
    
    print(f"✓ 加载嵌入数据: {embeddings_path}")
    with np.load(embeddings_path, allow_pickle=True) as data:
        db_vectors = data["db_vectors"]
        query_vectors = data["query_vectors"]
        ground_truth = data["ground_truth"].item()
        embedding_dim = data["embedding_dim"]
    
    print(f"  ✓ 数据库向量数量: {len(db_vectors)}, 查询向量数量: {len(query_vectors)}, 嵌入维度: {embedding_dim}")
    return db_vectors, query_vectors, ground_truth

# 加载配置文件
def load_config(config_path: str = "configs/index_sweep.json"):
    """加载配置文件，如果不存在则使用默认配置"""
    # 默认配置
    default_config = {
        "top_k": 10,  # 检索返回的top-k结果
        "nlist_values": [10, 30, 50, 100, 200],  # IVF索引的聚类中心数量
        "nprobe_values": [1, 3, 5, 10, 20],  # IVF索引的探测数量
        "M_values": [16, 32, 64],  # HNSW的M参数
        "ef_search_values": [16, 32, 64, 128],  # HNSW的efSearch参数
        "nbits_values": [4, 8],  # PQ的nbits参数
        "n_shards": [1, 2, 4, 8],  # 模拟分片数量
        "normalize_vectors": True,  # 是否归一化向量
        "distance_types": ["IP", "L2"]  # 距离度量类型
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

# 构建FAISS索引
def build_index(db_vectors: np.ndarray, index_type: str, config: Dict, distance_type: str = "IP") -> faiss.Index:
    """
    构建指定类型的FAISS索引
    db_vectors: 数据库向量
    index_type: 索引类型 (Flat, IVF, HNSW, IVF-PQ)
    config: 配置参数
    distance_type: 距离度量类型 (IP, L2)
    """
    embedding_dim = db_vectors.shape[1]
    
    # 根据距离类型选择索引类型
    if distance_type == "IP":
        # 内积 (余弦相似度，假设向量已归一化)
        quantizer = faiss.IndexFlatIP(embedding_dim)
        base_index = faiss.IndexFlatIP(embedding_dim)
    else:
        # L2距离 (欧几里得距离)
        quantizer = faiss.IndexFlatL2(embedding_dim)
        base_index = faiss.IndexFlatL2(embedding_dim)
    
    # 根据索引类型创建索引
    if index_type == "Flat":
        index = base_index
    elif index_type == "IVF":
        # 倒排文件索引
        nlist = config.get("nlist", 100)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        # IVF索引需要训练
        index.train(db_vectors)
    elif index_type == "HNSW":
        # 分层导航小世界图索引
        M = config.get("M", 16)
        ef_construction = config.get("ef_construction", 64)
        index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT if distance_type == "IP" else faiss.METRIC_L2)
        index.hnsw.efConstruction = ef_construction
    elif index_type == "IVF-PQ":
        # 倒排文件+乘积量化索引
        nlist = config.get("nlist", 100)
        nbits = config.get("nbits", 8)
        # 计算合适的PQ段数
        m = min(64, embedding_dim // 4)  # 确保每个段至少4维
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
        # IVF-PQ索引需要训练
        index.train(db_vectors)
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
    
    # 添加向量到索引
    index.add(db_vectors)
    
    # 设置查询参数
    if hasattr(index, "nprobe"):
        index.nprobe = config.get("nprobe", 10)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = config.get("ef_search", 64)
    
    return index

# 评估索引性能
def evaluate_index(
    index: faiss.Index,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    top_k: int = 10,
    batch_size: int = 100
) -> Tuple[float, float, np.ndarray]:
    """
    评估索引的召回率和查询延迟
    index: FAISS索引
    query_vectors: 查询向量
    ground_truth: 真实相关结果
    top_k: 评估的top-k值
    batch_size: 批处理大小
    """
    # 计时查询过程
    start_time = time.time()
    
    # 分批次执行查询以避免内存问题
    all_distances = []
    all_indices = []
    
    for i in range(0, len(query_vectors), batch_size):
        end = min(i + batch_size, len(query_vectors))
        batch_queries = query_vectors[i:end]
        
        # 执行查询
        distances, indices = index.search(batch_queries, top_k)
        
        all_distances.append(distances)
        all_indices.append(indices)
    
    # 合并结果
    distances = np.vstack(all_distances)
    indices = np.vstack(all_indices)
    
    # 计算总查询时间和平均延迟
    total_time = time.time() - start_time
    avg_latency = total_time / len(query_vectors) * 1000  # 转换为毫秒
    
    # 计算召回率
    recall_at_k = 0.0
    for i, (dist, ind) in enumerate(zip(distances, indices)):
        # 获取该查询的真实相关文档
        true_positives = set(ground_truth.get(i, []))
        if not true_positives:
            continue  # 如果没有真实相关文档，跳过该查询
        
        # 计算检索结果中的相关文档数量
        retrieved_relevant = len(set(ind) & true_positives)
        
        # 更新召回率
        recall_at_k += retrieved_relevant / min(len(true_positives), top_k)
    
    # 计算平均召回率
    recall_at_k = recall_at_k / len(query_vectors) if query_vectors else 0.0
    
    return recall_at_k, avg_latency, indices

# 对比不同索引类型
def compare_index_types(
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    config: Dict
) -> pd.DataFrame:
    """
    对比不同索引类型的性能
    """
    print("\n=== 对比不同索引类型的性能 ===")
    
    results = []
    index_types = ["Flat", "IVF", "HNSW", "IVF-PQ"]
    
    # 对每种距离类型进行测试
    for distance_type in config["distance_types"]:
        # 根据距离类型准备向量
        processed_db_vectors = db_vectors.copy()
        processed_query_vectors = query_vectors.copy()
        
        # 如果使用IP距离，需要归一化向量
        if distance_type == "IP" or (distance_type == "L2" and config["normalize_vectors"]):
            # 归一化向量
            faiss.normalize_L2(processed_db_vectors)
            faiss.normalize_L2(processed_query_vectors)
        
        for index_type in index_types:
            print(f"\n测试索引类型: {index_type}, 距离类型: {distance_type}")
            
            # 构建索引
            try:
                index = build_index(processed_db_vectors, index_type, config, distance_type)
                
                # 评估索引性能
                recall_at_k, avg_latency, _ = evaluate_index(
                    index, processed_query_vectors, ground_truth, config["top_k"]
                )
                
                # 记录结果
                results.append({
                    "index_type": index_type,
                    "distance_type": distance_type,
                    "normalized": (distance_type == "IP" or config["normalize_vectors"]),
                    "recall_at_k": recall_at_k,
                    "avg_latency_ms": avg_latency,
                    "index_size_mb": index.ntotal * db_vectors.shape[1] * 4 / 1024 / 1024  # 估算索引大小
                })
                
                print(f"  ✓ 召回率@{config['top_k']}: {recall_at_k:.4f}")
                print(f"  ✓ 平均延迟: {avg_latency:.4f} ms")
            except Exception as e:
                print(f"  ✗ 测试失败: {str(e)}")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果到CSV文件
    output_path = os.path.join("outputs", "index_compare.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 索引性能对比结果已保存至: {output_path}")
    
    return df

# 扫描IVF索引参数
def sweep_ivf_params(
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    config: Dict,
    distance_type: str = "IP"
) -> pd.DataFrame:
    """
    扫描IVF索引的nlist和nprobe参数
    """
    print(f"\n=== 扫描IVF索引参数 (距离类型: {distance_type}) ===")
    
    # 准备向量
    processed_db_vectors = db_vectors.copy()
    processed_query_vectors = query_vectors.copy()
    
    # 如果使用IP距离，需要归一化向量
    if distance_type == "IP" or config["normalize_vectors"]:
        faiss.normalize_L2(processed_db_vectors)
        faiss.normalize_L2(processed_query_vectors)
    
    results = []
    
    # 扫描nlist和nprobe参数
    for nlist in config["nlist_values"]:
        print(f"\n测试nlist = {nlist}")
        for nprobe in config["nprobe_values"]:
            try:
                # 构建索引
                index = build_index(processed_db_vectors, "IVF", {"nlist": nlist, "nprobe": nprobe}, distance_type)
                
                # 评估索引性能
                recall_at_k, avg_latency, _ = evaluate_index(
                    index, processed_query_vectors, ground_truth, config["top_k"]
                )
                
                # 记录结果
                results.append({
                    "param_type": "IVF",
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "recall_at_k": recall_at_k,
                    "avg_latency_ms": avg_latency
                })
                
                print(f"  ✓ nprobe = {nprobe}: 召回率@{config['top_k']} = {recall_at_k:.4f}, 延迟 = {avg_latency:.4f} ms")
            except Exception as e:
                print(f"  ✗ nprobe = {nprobe}: 测试失败: {str(e)}")
    
    return pd.DataFrame(results)

# 扫描HNSW索引参数
def sweep_hnsw_params(
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    config: Dict,
    distance_type: str = "IP"
) -> pd.DataFrame:
    """
    扫描HNSW索引的M和efSearch参数
    """
    print(f"\n=== 扫描HNSW索引参数 (距离类型: {distance_type}) ===")
    
    # 准备向量
    processed_db_vectors = db_vectors.copy()
    processed_query_vectors = query_vectors.copy()
    
    # 如果使用IP距离，需要归一化向量
    if distance_type == "IP" or config["normalize_vectors"]:
        faiss.normalize_L2(processed_db_vectors)
        faiss.normalize_L2(processed_query_vectors)
    
    results = []
    
    # 扫描M和efSearch参数
    for M in config["M_values"]:
        print(f"\n测试M = {M}")
        for ef_search in config["ef_search_values"]:
            try:
                # 构建索引
                index = build_index(processed_db_vectors, "HNSW", {"M": M, "ef_search": ef_search}, distance_type)
                
                # 评估索引性能
                recall_at_k, avg_latency, _ = evaluate_index(
                    index, processed_query_vectors, ground_truth, config["top_k"]
                )
                
                # 记录结果
                results.append({
                    "param_type": "HNSW",
                    "M": M,
                    "ef_search": ef_search,
                    "recall_at_k": recall_at_k,
                    "avg_latency_ms": avg_latency
                })
                
                print(f"  ✓ ef_search = {ef_search}: 召回率@{config['top_k']} = {recall_at_k:.4f}, 延迟 = {avg_latency:.4f} ms")
            except Exception as e:
                print(f"  ✗ ef_search = {ef_search}: 测试失败: {str(e)}")
    
    return pd.DataFrame(results)

# 参数扫描主函数
def parameter_sweep(
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    config: Dict
) -> pd.DataFrame:
    """
    扫描不同索引类型的参数，生成延迟-召回曲线数据
    """
    all_results = []
    
    # 对每种距离类型进行参数扫描
    for distance_type in config["distance_types"]:
        # 扫描IVF参数
        ivf_results = sweep_ivf_params(db_vectors, query_vectors, ground_truth, config, distance_type)
        ivf_results["distance_type"] = distance_type
        all_results.append(ivf_results)
        
        # 扫描HNSW参数
        hnsw_results = sweep_hnsw_params(db_vectors, query_vectors, ground_truth, config, distance_type)
        hnsw_results["distance_type"] = distance_type
        all_results.append(hnsw_results)
    
    # 合并结果
    df = pd.concat(all_results, ignore_index=True)
    
    # 保存结果到CSV文件
    output_path = os.path.join("outputs", "parameter_sweep.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 参数扫描结果已保存至: {output_path}")
    
    return df

# 绘制延迟-召回曲线
def plot_latency_recall_curve(sweep_results: pd.DataFrame, output_path: str = "outputs/latency_recall_curves.png"):
    """
    绘制延迟-召回率曲线
    """
    print("\n=== 绘制延迟-召回率曲线 ===")
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 按距离类型分组
    distance_types = sweep_results["distance_type"].unique()
    
    # 对每种距离类型创建一个子图
    for i, distance_type in enumerate(distance_types, 1):
        plt.subplot(len(distance_types), 1, i)
        
        # 获取该距离类型的结果
        dt_results = sweep_results[sweep_results["distance_type"] == distance_type]
        
        # 按参数类型分组绘制曲线
        param_types = dt_results["param_type"].unique()
        colors = plt.cm.get_cmap("tab10", len(param_types))
        
        for j, param_type in enumerate(param_types):
            # 获取该参数类型的结果
            pt_results = dt_results[dt_results["param_type"] == param_type]
            
            # 对IVF和HNSW使用不同的参数标记方式
            if param_type == "IVF":
                # 为每个nlist值绘制一条曲线
                nlist_values = pt_results["nlist"].unique()
                for nlist in nlist_values:
                    nlist_results = pt_results[pt_results["nlist"] == nlist]
                    # 按nprobe排序
                    nlist_results = nlist_results.sort_values(by="nprobe")
                    
                    # 绘制曲线
                    plt.scatter(nlist_results["avg_latency_ms"], nlist_results["recall_at_k"], 
                                label=f'IVF (nlist={nlist})', color=colors(j), alpha=0.7)
                    plt.plot(nlist_results["avg_latency_ms"], nlist_results["recall_at_k"], 
                             color=colors(j), alpha=0.5)
            else:  # HNSW
                # 为每个M值绘制一条曲线
                M_values = pt_results["M"].unique()
                for M in M_values:
                    M_results = pt_results[pt_results["M"] == M]
                    # 按ef_search排序
                    M_results = M_results.sort_values(by="ef_search")
                    
                    # 绘制曲线
                    plt.scatter(M_results["avg_latency_ms"], M_results["recall_at_k"], 
                                label=f'HNSW (M={M})', color=colors(j), alpha=0.7, marker='^')
                    plt.plot(M_results["avg_latency_ms"], M_results["recall_at_k"], 
                             color=colors(j), alpha=0.5)
        
        plt.title(f"延迟-召回率曲线 (距离类型: {distance_type})")
        plt.xlabel("平均延迟 (ms)")
        plt.ylabel("召回率@k")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=10, loc="best")
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=1.05)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ 延迟-召回率曲线已保存至: {output_path}")
    
    plt.close()

# 模拟分片查询与合并
def simulate_shard_query(
    db_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: Dict[int, List[int]],
    config: Dict,
    index_type: str = "IVF",
    distance_type: str = "IP"
) -> Dict:
    """
    模拟分片查询与全局Top-K合并
    """
    print(f"\n=== 模拟分片查询与合并 (索引类型: {index_type}, 距离类型: {distance_type}) ===")
    
    # 准备向量
    processed_db_vectors = db_vectors.copy()
    processed_query_vectors = query_vectors.copy()
    
    # 如果使用IP距离，需要归一化向量
    if distance_type == "IP" or config["normalize_vectors"]:
        faiss.normalize_L2(processed_db_vectors)
        faiss.normalize_L2(processed_query_vectors)
    
    results = {}
    
    # 对每种分片数量进行测试
    for n_shards in config["n_shards"]:
        print(f"\n测试分片数量: {n_shards}")
        
        # 计算每个分片的大小
        shard_size = len(processed_db_vectors) // n_shards
        
        # 构建每个分片的索引
        shard_indices = []
        shard_ranges = []
        
        for i in range(n_shards):
            start = i * shard_size
            end = len(processed_db_vectors) if i == n_shards - 1 else (i + 1) * shard_size
            shard_ranges.append((start, end))
            
            # 构建分片索引
            shard_vector = processed_db_vectors[start:end]
            shard_index = build_index(shard_vector, index_type, config, distance_type)
            shard_indices.append(shard_index)
        
        # 执行分片查询并合并结果
        start_time = time.time()
        
        total_queries = 0
        for query_idx, query in enumerate(processed_query_vectors):
            # 限制查询数量以加快测试
            if query_idx >= 100:  # 只测试前100个查询
                break
            total_queries += 1
            
            # 对每个分片执行查询
            all_shard_results = []
            for shard_idx, (start, end) in enumerate(shard_ranges):
                # 查询单个分片
                distances, indices = shard_indices[shard_idx].search(query.reshape(1, -1), config["top_k"])
                
                # 转换为全局索引
                global_indices = indices + start
                
                # 保存结果
                for dist, idx in zip(distances[0], global_indices[0]):
                    all_shard_results.append((-dist, idx))  # 使用负距离以便升序排序
            
            # 全局排序并选择Top-K
            all_shard_results.sort()
            top_k_results = all_shard_results[:config["top_k"]]
        
        # 计算总查询时间和平均延迟
        total_time = time.time() - start_time
        avg_latency = total_time / total_queries * 1000  # 转换为毫秒
        
        # 记录结果
        results[n_shards] = {
            "n_shards": n_shards,
            "avg_latency_ms": avg_latency,
            "total_queries": total_queries,
            "total_time_s": total_time
        }
        
        print(f"  ✓ 平均延迟: {avg_latency:.4f} ms")
    
    # 保存结果
    output_path = os.path.join("outputs", "shard_merge_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 分片合并统计结果已保存至: {output_path}")
    
    # 绘制分片延迟图表
    plot_shard_latency(results)
    
    return results

# 绘制分片延迟图表
def plot_shard_latency(shard_results: Dict, output_path: str = "outputs/shard_latency.png"):
    """
    绘制分片数量与延迟的关系图表
    """
    # 提取数据
    n_shards = list(shard_results.keys())
    latencies = [result["avg_latency_ms"] for result in shard_results.values()]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(n_shards, latencies, marker='o', linestyle='-', color='b', linewidth=2)
    
    # 添加数据标签
    for x, y in zip(n_shards, latencies):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
    
    plt.title("分片数量对查询延迟的影响")
    plt.xlabel("分片数量")
    plt.ylabel("平均查询延迟 (ms)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(min(n_shards) - 1, max(n_shards) + 1)
    plt.xticks(n_shards)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ 分片延迟图表已保存至: {output_path}")
    
    plt.close()

# 主函数
def main():
    """主函数，执行完整的FAISS索引对比与参数调优流程"""
    print("\n===== FAISS索引类型与参数扫描套件 =====")
    
    # 1. 检查并安装依赖
    check_and_install_dependencies()
    
    # 2. 创建必要的目录
    create_directories()
    
    # 3. 加载嵌入数据
    db_vectors, query_vectors, ground_truth = load_embeddings()
    
    # 4. 加载配置
    config = load_config()
    
    # 5. 对比不同索引类型的性能
    index_compare_results = compare_index_types(db_vectors, query_vectors, ground_truth, config)
    
    # 6. 执行参数扫描
    sweep_results = parameter_sweep(db_vectors, query_vectors, ground_truth, config)
    
    # 7. 绘制延迟-召回曲线
    plot_latency_recall_curve(sweep_results)
    
    # 8. 模拟分片查询与合并
    simulate_shard_query(db_vectors, query_vectors, ground_truth, config)
    
    print("\n===== 程序执行完成 =====")
    
    # 9. 输出实验结果分析
    print("\n=== 实验结果分析 ===")
    print("1. 索引类型对比:")
    print(f"   - Flat索引: 提供最高精度但速度最慢")
    print(f"   - IVF索引: 通过聚类加速查询，精度和速度平衡")
    print(f"   - HNSW索引: 提供最佳的延迟-召回平衡")
    print(f"   - IVF-PQ索引: 内存效率最高，但精度略有损失")
    print("2. 参数调优建议:")
    print(f"   - IVF索引: 增加nlist可提高精度但增加内存使用；增加nprobe可提高精度但降低速度")
    print(f"   - HNSW索引: 增加M和efSearch可提高精度但降低速度")
    print("3. 分片策略:")
    print(f"   - 分片可以提高系统的可扩展性，但会增加查询延迟")
    print(f"   - 实际应用中需要根据数据规模和查询负载选择合适的分片数量")
    print("4. 距离度量选择:")
    print(f"   - 对于语义检索任务，IP距离(内积)配合向量归一化通常提供更好的结果")
    print(f"   - L2距离在某些特定场景下可能更合适")

# 程序入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)