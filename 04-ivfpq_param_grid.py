#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IVFPQ 参数敏感性实验程序
作者: Ph.D. Rhino

本程序用于对FAISS库中的IVFPQ索引进行参数敏感性分析，
遍历不同的nlist（粗排桶数）、nprobe（探测数）、m（量化强度）和nbits参数组合，
测量并记录各参数组合下的检索性能（召回率和查询延迟），
最终生成CSV格式的实验数据，以帮助理解"粗排桶数、探测数、量化强度"的关系。

实验流程：
1. 加载或生成向量数据
2. 构建Flat索引作为精确搜索的真值基准
3. 遍历不同的参数组合构建IVFPQ索引
4. 测量各索引的检索性能指标
5. 分析实验结果并输出到指定文件
"""

import os
import sys
import time
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
    从文件中加载向量数据，如果文件不存在或数据不足则生成随机数据
    
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
                
            return embeddings.astype(np.float32)
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

# 构建Flat索引作为精确搜索基准（使用内积相似度）
def build_flat_index(vectors: np.ndarray) -> Any:
    """
    构建Flat索引作为精确搜索的基准（使用内积相似度）
    
    Args:
        vectors: 向量数据
        
    Returns:
        构建好的FAISS Flat索引
    """
    dim = vectors.shape[1]
    # 注意：IVFPQ通常搭配Flat IP使用，所以这里使用内积相似度
    index = faiss.IndexFlatIP(dim)  # 内积相似度
    
    # 检查数据类型并转换为FAISS所需的float32
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    index.add(vectors)
    logger.info(f"Flat索引构建完成，向量数量: {index.ntotal}")
    return index

# 构建IVFPQ索引
def build_ivfpq_index(train_vectors: np.ndarray, query_vectors: np.ndarray, 
                     nlist: int = 128, m: int = 8, nbits: int = 8) -> Any:
    """
    构建IVFPQ索引
    
    Args:
        train_vectors: 训练向量数据
        query_vectors: 查询向量数据，用于训练PQ编码器
        nlist: 粗排桶数
        m: 乘积量化的子空间数量（量化强度）
        nbits: 每个子空间的比特数
        
    Returns:
        构建好的FAISS IVFPQ索引
    """
    dim = train_vectors.shape[1]
    
    # 创建IVFPQ索引，使用内积相似度
    quantizer = faiss.IndexFlatIP(dim)  # 量化器使用Flat IP
    # 创建IVFPQ索引，参数为：量化器、维度、nlist、m、nbits
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    
    # 检查数据类型并转换为FAISS所需的float32
    if train_vectors.dtype != np.float32:
        train_vectors = train_vectors.astype(np.float32)
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
    
    # 计时构建过程
    start_time = time.time()
    # 训练IVFPQ索引
    index.train(train_vectors)
    # 添加向量到索引
    index.add(train_vectors)
    build_time = time.time() - start_time
    
    logger.info(f"IVFPQ索引构建完成 (nlist={nlist}, m={m}, nbits={nbits}), 耗时: {build_time:.2f}秒")
    return index

# 执行查询并返回结果
def search_index(index: Any, query_vectors: np.ndarray, k: int = 10, nprobe: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    在索引上执行查询
    
    Args:
        index: FAISS索引
        query_vectors: 查询向量
        k: 检索的最近邻数量
        nprobe: IVFPQ查询时的探测桶数
        
    Returns:
        距离数组、索引数组和平均查询时间的元组
    """
    # 检查数据类型并转换为FAISS所需的float32
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
    
    # 设置IVFPQ查询参数（如果适用）
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe
    
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

# 执行参数网格扫描实验
def run_parameter_grid(train_vectors: np.ndarray, query_vectors: np.ndarray, 
                      nlist_values: List[int], nprobe_values: List[int], 
                      m_values: List[int], nbits_values: List[int],
                      k: int = 10) -> pd.DataFrame:
    """
    执行IVFPQ参数网格扫描实验
    
    Args:
        train_vectors: 训练向量数据
        query_vectors: 查询向量数据
        nlist_values: nlist参数的取值列表
        nprobe_values: nprobe参数的取值列表
        m_values: m参数的取值列表
        nbits_values: nbits参数的取值列表
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
    for nlist in nlist_values:
        for m in m_values:
            for nbits in nbits_values:
                # 构建IVFPQ索引
                ivfpq_index = build_ivfpq_index(train_vectors, query_vectors, nlist, m, nbits)
                
                for nprobe in nprobe_values:
                    # 执行查询
                    _, approx_indices, avg_query_time = search_index(ivfpq_index, query_vectors, k, nprobe)
                    
                    # 计算Recall@K
                    recall = calculate_recall_at_k(approx_indices, exact_indices, k)
                    
                    # 计算量化后的数据大小压缩率
                    # IVFPQ的存储大小约为：nlist*(8+dim) + ntotal*(2*log2(nlist) + m*nbits/8)
                    # 这里简化计算，使用估计值
                    original_size = train_vectors.shape[0] * train_vectors.shape[1] * 4  # 假设float32
                    pq_size = train_vectors.shape[0] * (m * nbits / 8)  # 只计算向量的PQ部分
                    compression_ratio = pq_size / original_size
                    
                    # 记录结果
                    result = {
                        'nlist': nlist,
                        'nprobe': nprobe,
                        'm': m,
                        'nbits': nbits,
                        'avg_query_time_ms': avg_query_time,
                        'recall_at_k': recall,
                        'compression_ratio': compression_ratio
                    }
                    results.append(result)
                    
                    logger.info(f"参数组合: nlist={nlist}, nprobe={nprobe}, m={m}, nbits={nbits} | 平均查询时间: {avg_query_time:.3f}ms, Recall@{k}: {recall:.3f}")
    
    # 转换为DataFrame并返回
    df_results = pd.DataFrame(results)
    return df_results

# 主函数
def main() -> None:
    """主函数"""
    try:
        # 1. 检查并安装依赖
        check_dependencies()
        
        # 2. 定义文件路径
        input_file = os.path.join('data', 'embeddings_256.npy')
        output_csv_file = os.path.join('result', 'ivfpq_grid.csv')
        
        # 3. 确保输出目录存在
        os.makedirs('result', exist_ok=True)
        
        # 4. 加载或生成向量数据
        dim = 256
        embeddings = load_or_generate_embeddings(input_file, dim)
        
        # 5. 准备训练数据和查询数据
        train_vectors, query_vectors = prepare_data(embeddings)
        
        # 6. 定义要扫描的参数范围（根据用户要求）
        nlist_values = [128, 256]  # 粗排桶数
        nprobe_values = [16, 64]   # 探测数
        m_values = [8, 16]         # 量化强度（子空间数量）
        nbits_values = [8]         # 每个子空间的比特数
        k = 10                     # 检索的最近邻数量
        
        # 7. 执行参数网格扫描实验
        logger.info("开始IVFPQ参数网格扫描实验...")
        start_time = time.time()
        
        df_results = run_parameter_grid(
            train_vectors,
            query_vectors,
            nlist_values,
            nprobe_values,
            m_values,
            nbits_values,
            k
        )
        
        experiment_time = time.time() - start_time
        logger.info(f"实验完成，总耗时: {experiment_time:.2f}秒")
        
        # 8. 保存实验结果到CSV文件
        try:
            df_results.to_csv(output_csv_file, index=False, encoding='utf-8')
            logger.info(f"✓ 实验数据已保存到: {output_csv_file}")
        except Exception as e:
            logger.error(f"✗ 实验数据保存失败: {str(e)}")
            raise RuntimeError(f"无法保存实验数据: {str(e)}")
        
        # 9. 输出总结信息
        logger.info("\n===== 实验总结 =====")
        logger.info(f"1. 实验成功完成，共测试 {len(df_results)} 组参数组合")
        logger.info(f"2. 生成的文件：")
        logger.info(f"   - {output_csv_file} (详细实验数据)")
        logger.info(f"3. 实验结果展示了IVFPQ参数对检索性能的影响")
        logger.info(f"4. 可通过分析CSV数据来理解'粗排桶数、探测数、量化强度'的关系")
        
        # 10. 对实验结果进行简单分析
        analyze_results(df_results)
        
        logger.info("\n程序执行成功！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

# 分析实验结果
def analyze_results(df_results: pd.DataFrame) -> None:
    """
    简单分析实验结果，提取一些关键洞察
    
    Args:
        df_results: 实验结果DataFrame
    """
    logger.info("\n===== 实验结果简要分析 =====")
    
    # 找到召回率最高的参数组合
    best_recall_idx = df_results['recall_at_k'].idxmax()
    best_recall_params = df_results.iloc[best_recall_idx]
    
    # 找到查询速度最快的参数组合
    best_speed_idx = df_results['avg_query_time_ms'].idxmin()
    best_speed_params = df_results.iloc[best_speed_idx]
    
    # 找到综合性能最好的参数组合（F1分数）
    # 归一化查询时间和召回率
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
    
    # 计算F1分数
    df_normalized['f1_score'] = 2 * (df_normalized['norm_recall'] * df_normalized['norm_time']) / \
                              (df_normalized['norm_recall'] + df_normalized['norm_time'] + 1e-10)
    
    best_f1_idx = df_normalized['f1_score'].idxmax()
    best_f1_params = df_normalized.iloc[best_f1_idx]
    
    # 输出分析结果
    logger.info(f"最高召回率参数组合: nlist={int(best_recall_params['nlist'])}, nprobe={int(best_recall_params['nprobe'])}, m={int(best_recall_params['m'])}")
    logger.info(f"  Recall@10: {best_recall_params['recall_at_k']:.3f}, 平均查询时间: {best_recall_params['avg_query_time_ms']:.3f}ms")
    
    logger.info(f"最快查询速度参数组合: nlist={int(best_speed_params['nlist'])}, nprobe={int(best_speed_params['nprobe'])}, m={int(best_speed_params['m'])}")
    logger.info(f"  Recall@10: {best_speed_params['recall_at_k']:.3f}, 平均查询时间: {best_speed_params['avg_query_time_ms']:.3f}ms")
    
    logger.info(f"最佳综合性能参数组合: nlist={int(best_f1_params['nlist'])}, nprobe={int(best_f1_params['nprobe'])}, m={int(best_f1_params['m'])}")
    logger.info(f"  Recall@10: {best_f1_params['recall_at_k']:.3f}, 平均查询时间: {best_f1_params['avg_query_time_ms']:.3f}ms, F1分数: {best_f1_params['f1_score']:.3f}")
    
    # 参数影响洞察
    logger.info("\n参数影响洞察:")
    logger.info("1. nprobe（探测数）: 增加nprobe通常会提高召回率，但会增加查询延迟")
    logger.info("2. nlist（粗排桶数）: 增加nlist可能提高召回率，但需要平衡构建时间和内存消耗")
    logger.info("3. m（量化强度）: 增加m会提高精度但降低压缩率，减少m会提高压缩率但降低精度")

# 运行主函数
if __name__ == '__main__':
    main()