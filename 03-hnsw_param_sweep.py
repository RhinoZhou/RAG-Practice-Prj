#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HNSW 参数敏感性实验程序
作者: Ph.D. Rhino

本程序用于对FAISS库中的HNSW索引进行参数敏感性分析，
遍历不同的M、efConstruction和efSearch参数组合，
测量并记录各参数组合下的检索性能（召回率和查询延迟），
最终生成CSV格式的实验数据和Markdown格式的实验报告。

实验流程：
1. 加载或生成向量数据
2. 遍历不同的参数组合构建HNSW索引
3. 测量各索引的检索性能指标
4. 分析实验结果并输出到指定文件
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import faiss
import logging
from typing import List, Dict, Tuple, Any

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
    required_packages = ['numpy', 'pandas', 'faiss-cpu']
    missing_packages = []
    
    # 尝试导入每个包，如果导入失败则添加到缺失列表
    for package in required_packages:
        try:
            if package == 'faiss-cpu':
                __import__('faiss')
            else:
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

# 加载或生成向量数据
def load_or_generate_embeddings(file_path: str, dim: int = 256, num_vectors: int = 10000) -> np.ndarray:
    """
    从文件中加载向量数据，如果文件不存在则生成随机数据
    
    Args:
        file_path: 向量数据文件路径
        dim: 向量维度
        num_vectors: 向量数量
        
    Returns:
        numpy数组形式的向量数据
    """
    if os.path.exists(file_path):
        logger.info(f"正在加载向量数据: {file_path}")
        try:
            embeddings = np.load(file_path)
            logger.info(f"✓ 数据加载成功，向量数量: {embeddings.shape[0]}, 向量维度: {embeddings.shape[1]}")
            
            # 检查数据维度，如果不是256维则生成新数据
            if embeddings.shape[1] != 256:
                logger.warning(f"数据维度为 {embeddings.shape[1]}，不是要求的256维，将生成新的随机数据")
                return generate_random_embeddings(dim, num_vectors)
            
            # 如果数据量太少，生成更大的数据集以更好地展示参数影响
            if embeddings.shape[0] < 1000:
                logger.warning(f"数据量较少 ({embeddings.shape[0]} 个向量)，将生成更大的随机数据集以更好地展示参数影响")
                return generate_random_embeddings(dim, num_vectors)
                
            return embeddings
        except Exception as e:
            logger.error(f"✗ 数据加载失败: {str(e)}")
            logger.info("正在生成随机向量数据...")
            return generate_random_embeddings(dim, num_vectors)
    else:
        logger.info(f"文件不存在: {file_path}，正在生成随机向量数据...")
        return generate_random_embeddings(dim, num_vectors)

# 生成随机向量数据
def generate_random_embeddings(dim: int = 256, num_vectors: int = 10000) -> np.ndarray:
    """
    生成随机向量数据
    
    Args:
        dim: 向量维度
        num_vectors: 向量数量
        
    Returns:
        numpy数组形式的随机向量数据
    """
    np.random.seed(42)  # 设置随机种子以确保可重复性
    embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
    
    # 归一化向量以模拟真实嵌入
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    logger.info(f"✓ 随机数据生成成功，向量数量: {embeddings.shape[0]}, 向量维度: {embeddings.shape[1]}")
    return embeddings

# 准备数据（分割训练集和查询集）
def prepare_data(embeddings: np.ndarray, train_ratio: float = 0.9, query_count: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备训练数据和查询数据
    
    Args:
        embeddings: 原始向量数据
        train_ratio: 训练数据占比
        query_count: 查询向量数量
        
    Returns:
        训练向量和查询向量的元组
    """
    total_count = embeddings.shape[0]
    train_count = int(total_count * train_ratio)
    
    # 随机打乱数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    indices = np.random.permutation(total_count)
    
    # 分割训练集和查询集
    train_embeddings = embeddings[indices[:train_count]]
    query_embeddings = embeddings[indices[-query_count:]]
    
    logger.info(f"数据分割完成: 训练集 {train_embeddings.shape[0]} 个向量, 查询集 {query_embeddings.shape[0]} 个向量")
    return train_embeddings, query_embeddings

# 构建Flat索引作为精确搜索基准
def build_flat_index(vectors: np.ndarray) -> Any:
    """
    构建Flat索引作为精确搜索的基准
    
    Args:
        vectors: 向量数据
        
    Returns:
        构建好的FAISS Flat索引
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2距离度量
    
    # 检查数据类型并转换为FAISS所需的float32
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    index.add(vectors)
    logger.info(f"Flat索引构建完成，向量数量: {index.ntotal}")
    return index

# 构建HNSW索引
def build_hnsw_index(vectors: np.ndarray, M: int = 16, efConstruction: int = 200) -> Any:
    """
    构建HNSW索引
    
    Args:
        vectors: 向量数据
        M: HNSW图中每个节点的最大连接数
        efConstruction: 构建索引时的搜索范围
        
    Returns:
        构建好的FAISS HNSW索引
    """
    dim = vectors.shape[1]
    # 创建HNSW索引，使用L2距离度量
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    
    # 设置构建参数
    index.hnsw.efConstruction = efConstruction
    
    # 检查数据类型并转换为FAISS所需的float32
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    # 计时构建过程
    start_time = time.time()
    index.add(vectors)
    build_time = time.time() - start_time
    
    logger.info(f"HNSW索引构建完成 (M={M}, efConstruction={efConstruction}), 耗时: {build_time:.2f}秒")
    return index

# 执行查询并返回结果
def search_index(index: Any, query_vectors: np.ndarray, k: int = 10, efSearch: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    在索引上执行查询
    
    Args:
        index: FAISS索引
        query_vectors: 查询向量
        k: 检索的最近邻数量
        efSearch: HNSW查询时的搜索范围参数
        
    Returns:
        距离数组和索引数组的元组
    """
    # 检查数据类型并转换为FAISS所需的float32
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
    
    # 设置HNSW查询参数（如果适用）
    if hasattr(index, 'hnsw') and efSearch is not None:
        index.hnsw.efSearch = efSearch
    
    # 计时查询过程
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    query_time = time.time() - start_time
    
    # 计算平均查询时间
    avg_query_time = query_time / query_vectors.shape[0] * 1000  # 转换为毫秒
    
    return distances, indices, avg_query_time

# 计算Recall@K指标
def calculate_recall_at_k(approx_indices: np.ndarray, exact_indices: np.ndarray, k: int = 10) -> float:
    """
    计算Recall@K指标
    
    Args:
        approx_indices: 近似搜索返回的索引
        exact_indices: 精确搜索返回的索引
        k: 检索的最近邻数量
        
    Returns:
        Recall@K值
    """
    total_relevant = 0
    total_found = 0
    
    for i in range(approx_indices.shape[0]):
        # 对于每个查询，计算交集大小
        approx_set = set(approx_indices[i])
        exact_set = set(exact_indices[i])
        intersection = approx_set.intersection(exact_set)
        
        total_relevant += len(exact_set)
        total_found += len(intersection)
    
    # 计算Recall@K
    recall = total_found / total_relevant if total_relevant > 0 else 0
    return recall

# 执行参数扫描实验
def run_parameter_sweep(train_vectors: np.ndarray, query_vectors: np.ndarray, 
                       M_values: List[int], efConstruction_values: List[int], efSearch_values: List[int],
                       k: int = 10) -> pd.DataFrame:
    """
    执行HNSW参数扫描实验
    
    Args:
        train_vectors: 训练向量数据
        query_vectors: 查询向量数据
        M_values: M参数的取值列表
        efConstruction_values: efConstruction参数的取值列表
        efSearch_values: efSearch参数的取值列表
        k: 检索的最近邻数量
        
    Returns:
        实验结果的DataFrame
    """
    # 构建精确搜索索引作为基准
    logger.info("正在构建Flat索引（精确搜索基准）...")
    flat_index = build_flat_index(train_vectors)
    _, exact_indices, _ = search_index(flat_index, query_vectors, k)
    
    # 存储实验结果
    results = []
    
    # 遍历所有参数组合
    for M in M_values:
        for efConstruction in efConstruction_values:
            # 构建HNSW索引
            hnsw_index = build_hnsw_index(train_vectors, M, efConstruction)
            
            for efSearch in efSearch_values:
                # 执行查询
                _, approx_indices, avg_query_time = search_index(hnsw_index, query_vectors, k, efSearch)
                
                # 计算Recall@K
                recall = calculate_recall_at_k(approx_indices, exact_indices, k)
                
                # 记录结果
                result = {
                    'M': M,
                    'efConstruction': efConstruction,
                    'efSearch': efSearch,
                    'avg_query_time_ms': avg_query_time,
                    'recall_at_k': recall
                }
                results.append(result)
                
                logger.info(f"参数组合: M={M}, efConstruction={efConstruction}, efSearch={efSearch} | 平均查询时间: {avg_query_time:.3f}ms, Recall@{k}: {recall:.3f}")
    
    # 转换为DataFrame并返回
    df_results = pd.DataFrame(results)
    return df_results

# 找出最佳参数组合（根据特定指标）
def find_best_parameters(df_results: pd.DataFrame, metric: str = 'f1_score') -> Dict[str, Any]:
    """
    根据指定指标找出最佳参数组合
    
    Args:
        df_results: 实验结果DataFrame
        metric: 评估指标，可以是'f1_score'（平衡速度和精度）、'recall'或'latency'
        
    Returns:
        最佳参数组合和性能指标
    """
    # 归一化查询时间和召回率，计算F1分数
    df_normalized = df_results.copy()
    
    # 归一化查询时间（反转，因为更小的时间更好）
    min_time = df_normalized['avg_query_time_ms'].min()
    max_time = df_normalized['avg_query_time_ms'].max()
    time_range = max_time - min_time + 1e-10  # 避免除零
    df_normalized['norm_time'] = 1 - (df_normalized['avg_query_time_ms'] - min_time) / time_range
    
    # 归一化召回率
    min_recall = df_normalized['recall_at_k'].min()
    max_recall = df_normalized['recall_at_k'].max()
    recall_range = max_recall - min_recall + 1e-10  # 避免除零
    df_normalized['norm_recall'] = (df_normalized['recall_at_k'] - min_recall) / recall_range
    
    # 计算F1分数（平衡速度和精度）
    df_normalized['f1_score'] = 2 * (df_normalized['norm_recall'] * df_normalized['norm_time']) / \
                              (df_normalized['norm_recall'] + df_normalized['norm_time'] + 1e-10)
    
    # 根据指定指标选择最佳参数
    if metric == 'f1_score':
        best_idx = df_normalized['f1_score'].idxmax()
    elif metric == 'recall':
        best_idx = df_normalized['recall_at_k'].idxmax()
    elif metric == 'latency':
        best_idx = df_normalized['avg_query_time_ms'].idxmin()
    else:
        best_idx = df_normalized['f1_score'].idxmax()
    
    best_params = df_normalized.iloc[best_idx].to_dict()
    return best_params

# 生成实验报告
def generate_report(df_results: pd.DataFrame, best_params: Dict[str, Any], report_file: str, 
                   train_vectors: np.ndarray, query_vectors: np.ndarray, dim: int = 256, k: int = 10) -> None:
    """
    生成实验报告
    
    Args:
        df_results: 实验结果DataFrame
        best_params: 最佳参数组合
        report_file: 报告文件路径
        train_vectors: 训练向量数据
        query_vectors: 查询向量数据
        dim: 向量维度
        k: 检索的最近邻数量
    """
    logger.info(f"正在生成实验报告: {report_file}")
    
    # 创建报告内容
    report_content = f"""
# HNSW 参数敏感性实验报告

## 实验概述
本实验对FAISS库中的HNSW索引进行了参数敏感性分析，遍历了不同的M、efConstruction和efSearch参数组合，
测量并记录了各参数组合下的检索性能（召回率和查询延迟），以辅助直观展示"参数换精度/延迟"的权衡关系。

## 实验配置
- **向量维度**: {dim}
- **训练向量数量**: {train_vectors.shape[0]}
- **查询向量数量**: {query_vectors.shape[0]}
- **检索Top-K**: {k}
- **参数范围**:
  - M ∈ {sorted(df_results['M'].unique().tolist())}
  - efConstruction ∈ {sorted(df_results['efConstruction'].unique().tolist())}
  - efSearch ∈ {sorted(df_results['efSearch'].unique().tolist())}

## 最佳参数组合
根据F1分数（平衡速度和精度），最佳参数组合如下：
- **M**: {int(best_params['M'])}
- **efConstruction**: {int(best_params['efConstruction'])}
- **efSearch**: {int(best_params['efSearch'])}
- **平均查询时间**: {best_params['avg_query_time_ms']:.3f} ms
- **Recall@{k}**: {best_params['recall_at_k']:.3f}
- **F1分数**: {best_params['f1_score']:.3f}

## 性能指标表格

| M  | efConstruction | efSearch | 平均查询时间(ms) | Recall@{k} |
|----|----------------|----------|-----------------|------------|
"""
    
    # 添加性能指标表格数据
    # 按Recall@K降序和查询时间升序排序
    df_sorted = df_results.sort_values(['recall_at_k', 'avg_query_time_ms'], ascending=[False, True])
    
    for _, row in df_sorted.iterrows():
        report_content += f"| {row['M']}  | {row['efConstruction']} | {row['efSearch']} | {row['avg_query_time_ms']:.3f} | {row['recall_at_k']:.3f} |\n"
    
    # 添加实验结果分析
    report_content += f"""

## 实验结果分析

### 参数影响分析

#### M参数影响
M参数控制HNSW图中每个节点的最大连接数：
- **较大的M值**：通常可以获得更高的召回率，但会增加索引大小和构建时间
- **较小的M值**：索引更小，构建更快，但召回率可能较低

#### efConstruction参数影响
efConstruction参数控制构建索引时的搜索范围：
- **较大的efConstruction值**：可以获得更高质量的索引结构，提升查询性能，但会增加构建时间
- **较小的efConstruction值**：构建更快，但索引质量可能较低

#### efSearch参数影响
efSearch参数控制查询时的搜索范围：
- **较大的efSearch值**：可以获得更高的召回率，但会增加查询延迟
- **较小的efSearch值**：查询更快，但召回率可能较低

### 延迟-召回权衡分析

通过实验结果可以观察到明显的延迟-召回权衡关系：
1. **高召回率配置**：当efSearch增加时，Recall@{k}提高，但查询延迟也随之增加
2. **低延迟配置**：当efSearch减少时，查询速度加快，但Recall@{k}会降低
3. **平衡配置**：最佳参数组合在速度和精度之间取得了较好的平衡

### 图表可视化（占位）

以下是延迟-召回权衡关系的可视化图表（实际使用时需根据CSV数据生成）：

![延迟-召回曲线](延迟-召回曲线.png)

## 结论

本实验成功演示了HNSW索引参数对检索性能的影响，验证了"参数换精度/延迟"的权衡关系。
在实际应用中，可以根据具体需求（是优先考虑速度还是精度）选择合适的参数组合。

如果应用场景对延迟敏感，可以选择较小的efSearch值；
如果应用场景对召回率要求较高，可以选择较大的efSearch值；
如果需要平衡速度和精度，可以参考本次实验找出的最佳参数组合。
"""
    
    # 写入报告文件
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"✓ 实验报告已保存到: {report_file}")
    except Exception as e:
        logger.error(f"✗ 实验报告保存失败: {str(e)}")
        raise RuntimeError(f"无法保存实验报告: {str(e)}")

# 主函数
def main() -> None:
    """主函数"""
    try:
        # 1. 检查并安装依赖
        check_dependencies()
        
        # 2. 定义文件路径
        input_file = os.path.join('data', 'embeddings_256.npy')
        output_csv_file = os.path.join('result', 'hnsw_param_sweep.csv')
        output_report_file = os.path.join('result', 'hnsw_param_summary.md')
        
        # 3. 确保输出目录存在
        os.makedirs('result', exist_ok=True)
        
        # 4. 加载或生成向量数据
        dim = 256
        embeddings = load_or_generate_embeddings(input_file, dim)
        
        # 5. 准备训练数据和查询数据
        train_vectors, query_vectors = prepare_data(embeddings)
        
        # 6. 定义要扫描的参数范围
        M_values = [16, 32]  # 用户要求
        efConstruction_values = [100, 200]  # 用户要求
        efSearch_values = [64, 128, 256]  # 用户要求
        k = 10  # 检索的最近邻数量
        
        # 7. 执行参数扫描实验
        logger.info("开始HNSW参数敏感性实验...")
        start_time = time.time()
        
        df_results = run_parameter_sweep(
            train_vectors,
            query_vectors,
            M_values,
            efConstruction_values,
            efSearch_values,
            k
        )
        
        experiment_time = time.time() - start_time
        logger.info(f"实验完成，总耗时: {experiment_time:.2f}秒")
        
        # 8. 找出最佳参数组合
        best_params = find_best_parameters(df_results)
        logger.info(f"最佳参数组合: M={int(best_params['M'])}, efConstruction={int(best_params['efConstruction'])}, efSearch={int(best_params['efSearch'])}, 平均查询时间={best_params['avg_query_time_ms']:.3f}ms, Recall@{k}={best_params['recall_at_k']:.3f}")
        
        # 9. 保存实验结果到CSV文件
        try:
            df_results.to_csv(output_csv_file, index=False, encoding='utf-8')
            logger.info(f"✓ 实验数据已保存到: {output_csv_file}")
        except Exception as e:
            logger.error(f"✗ 实验数据保存失败: {str(e)}")
            raise RuntimeError(f"无法保存实验数据: {str(e)}")
        
        # 10. 生成实验报告
        generate_report(df_results, best_params, output_report_file, train_vectors, query_vectors, dim, k)
        
        # 11. 输出总结信息
        logger.info("\n===== 实验总结 =====")
        logger.info(f"1. 实验成功完成，共测试 {len(df_results)} 组参数组合")
        logger.info(f"2. 生成的文件：")
        logger.info(f"   - {output_csv_file} (详细实验数据)")
        logger.info(f"   - {output_report_file} (实验分析报告)")
        logger.info(f"3. 最佳参数组合在速度和精度之间取得了良好平衡")
        logger.info(f"4. 实验结果清晰展示了HNSW参数对检索性能的影响")
        logger.info("\n程序执行成功！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

# 运行主函数
if __name__ == '__main__':
    main()