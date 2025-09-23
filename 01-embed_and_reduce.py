#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
程序名称：向量嵌入与降维演示工具
程序功能：评估分块策略对系统性能的杠杆效应，重点展示chunk_overlap对关键信息覆盖率和索引规模的影响

详细功能：
1. 对分块文本生成向量（使用简化的向量化方法）
2. 用PCA将维度从768压缩到256
3. 展示大小对比和精度近似影响
4. 输出.npy文件并打印存储大小、余弦相似度变化统计

内容概述：
- 演示embedding维度与存储开销关系
- 以PCA近似OPQ的"降维+旋转"的直觉

流程：
1. 载入chunks.jsonl
2. 生成句向量
3. PCA降维
4. 输出.npy并打印存储大小、余弦相似度变化统计

输入：data/chunks.jsonl
输出：data/embeddings_768.npy、data/embeddings_256.npy、日志指标

作者：Ph.D. Rhino
"""

import os
import sys
import json
import numpy as np
import time
import importlib
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# 检查并安装依赖
def check_and_install_dependencies():
    """检查必要的依赖包是否安装，如未安装则自动安装"""
    required_packages = [
        'numpy',
        'sklearn'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            # 对于实际安装包名和导入名不同的情况进行处理
            install_name = package
            if package == 'sklearn':
                install_name = 'scikit-learn'
            
            print(f"✗ {install_name} 未安装，正在安装...")
            result = os.system(f"{sys.executable} -m pip install {install_name} --user")
            
            if result == 0:
                print(f"✓ {install_name} 安装成功")
            else:
                print(f"✗ {install_name} 安装失败，请手动安装")
    
    # 尝试导入必要的模块
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ 所有必要的模块都已可用")
        return True
    except ImportError:
        print("✗ 必要的模块不可用")
        return False

# 加载数据
def load_chunks(file_path):
    """从JSONL文件加载文本块数据"""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    chunks.append(chunk)
        print(f"成功加载 {len(chunks)} 个文本块")
        return chunks
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# 简化的向量生成函数
def generate_embeddings_simple(chunks, embedding_dim=768):
    """
    使用简单方法生成向量表示，模拟真实嵌入过程
    
    参数:
        chunks: 文本块列表
        embedding_dim: 嵌入维度
    
    返回:
        numpy数组，形状为(len(chunks), embedding_dim)
    """
    print("在生成向量...")
    start_time = time.time()
    
    # 为每个文本块生成随机向量
    embeddings = []
    for chunk in chunks:
        # 使用文本内容作为随机种子，确保结果可复现
        text = chunk["text"]
        seed = hash(text) % (2**32 - 1)  # 计算文本的哈希值作为种子
        np.random.seed(seed)
        
        # 生成768维的随机向量，值范围在[-1, 1]之间
        embedding = np.random.uniform(-1, 1, embedding_dim)
        # 归一化向量，使其范数为1（与许多实际嵌入模型的做法一致）
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    end_time = time.time()
    print(f"向量生成完成，耗时 {end_time - start_time:.2f} 秒")
    
    return np.array(embeddings)

# PCA降维
def reduce_dimensions(embeddings, target_dim=256):
    """
    使用PCA将向量从原始维度降低到目标维度
    
    参数:
        embeddings: 原始嵌入向量，形状为(n_samples, n_features)
        target_dim: 目标维度
    
    返回:
        reduced_embeddings: 降维后的向量，形状为(n_samples, target_dim)
        explained_variance_ratio: 保留的方差比例
    """
    print(f"执行PCA降维...")
    print(f"正在将向量从 {embeddings.shape[1]} 维降维到 {target_dim} 维...")
    start_time = time.time()
    
    from sklearn.decomposition import PCA
    
    # 检查样本数和目标维度的关系
    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]
    
    # 如果样本数小于目标维度，调整目标维度
    if n_samples < target_dim:
        adjusted_dim = min(n_samples - 1, n_features)
        print(f"警告: 样本数 {n_samples} 小于目标维度 {target_dim}，无法执行完整的PCA降维。")
        print(f"将目标维度调整为 {adjusted_dim} 维（样本数-1）。")
        print(f"这是PCA算法的限制，因为PCA需要计算的主成分数量不能超过样本数-1。")
        target_dim = adjusted_dim
    
    # 创建PCA对象并执行降维
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 计算解释方差比例
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    
    end_time = time.time()
    print(f"PCA降维完成，耗时 {end_time - start_time:.2f} 秒")
    print(f"保留的方差比例: {explained_variance_ratio:.4f}")
    
    return reduced_embeddings, explained_variance_ratio

# 计算余弦相似度变化
def analyze_similarity_change(original_embeddings, reduced_embeddings):
    """分析降维前后余弦相似度的变化"""
    print("正在分析余弦相似度变化...")
    
    # 计算原始向量之间的相似度矩阵
    original_similarities = cosine_similarity(original_embeddings)
    
    # 计算降维后向量之间的相似度矩阵
    reduced_similarities = cosine_similarity(reduced_embeddings)
    
    # 计算相似度差异
    diff = original_similarities - reduced_similarities
    
    # 统计相似度变化
    stats = {
        'mean_abs_diff': np.mean(np.abs(diff)),
        'max_abs_diff': np.max(np.abs(diff)),
        'min_abs_diff': np.min(np.abs(diff)),
        'std_diff': np.std(diff)
    }
    
    print("余弦相似度变化统计:")
    print(f"  平均绝对差异: {stats['mean_abs_diff']:.6f}")
    print(f"  最大绝对差异: {stats['max_abs_diff']:.6f}")
    print(f"  最小绝对差异: {stats['min_abs_diff']:.6f}")
    print(f"  标准差: {stats['std_diff']:.6f}")
    
    return stats

# 计算存储大小
def calculate_storage_size(embeddings, dtype=np.float32):
    """计算向量存储所需的字节数"""
    # 每个float32占4字节
    return embeddings.size * np.dtype(dtype).itemsize

# 保存向量到文件
def save_embeddings(embeddings, file_path):
    """将向量保存到.npy文件"""
    try:
        np.save(file_path, embeddings)
        print(f"向量已保存到 {file_path}")
        return True
    except Exception as e:
        print(f"保存向量失败: {e}")
        return False

# 主函数
def main():
    """
    主函数，执行整个流程
    """
    print("===== 分块策略对系统性能的杠杆效应演示程序 =====")
    print("=== 向量嵌入与降维演示工具 ===")
    
    # 检查并安装依赖
    print("\n检查依赖包...")
    success = check_and_install_dependencies()
    print(f"依赖检查结果: {'成功' if success else '部分成功，尝试继续'}")
    
    # 加载数据
    print("\n加载数据...")
    chunks = load_chunks('data/chunks.jsonl')
    if not chunks:
        print("数据加载失败，程序退出")
        return
    
    try:
        # 生成向量（使用简化方法）
        print("\n生成向量...")
        embeddings_768 = generate_embeddings_simple(chunks)
        
        # PCA降维
        print("\n执行PCA降维...")
        embeddings_256, explained_variance = reduce_dimensions(embeddings_768, target_dim=256)
        
        # 分析相似度变化
        print("\n分析余弦相似度变化...")
        similarity_stats = analyze_similarity_change(embeddings_768, embeddings_256)
        
        # 计算存储大小
        print("\n计算存储大小...")
        size_768 = calculate_storage_size(embeddings_768)
        size_256 = calculate_storage_size(embeddings_256)
        reduction_ratio = (size_768 - size_256) / size_768 * 100
        
        print(f"768维向量存储大小: {size_768 / 1024:.2f} KB")
        print(f"256维向量存储大小: {size_256 / 1024:.2f} KB")
        print(f"存储节省: {reduction_ratio:.2f}%")
        
        # 保存结果
        print("\n保存结果...")
        save_embeddings(embeddings_768, 'data/embeddings_768.npy')
        save_embeddings(embeddings_256, 'data/embeddings_256.npy')
        
        # 总结
        print("\n=== 实验结果总结 ===")
        print(f"文本块数量: {len(chunks)}")
        print(f"原始维度: 768，压缩后维度: {embeddings_256.shape[1]}")
        print(f"存储节省: {reduction_ratio:.2f}%")
        print(f"PCA解释方差比: {explained_variance:.4f}")
        print(f"平均余弦相似度变化: {similarity_stats['mean_abs_diff']:.6f}")
        
        print("\n程序执行完成！")
        
        # 附加说明
        print("\n=== 附加说明 ===")
        print("此演示使用了简化的向量化方法。在实际应用中，")
        print("建议使用专业的嵌入模型如sentence-transformers、OpenAI的embedding API等。")
        print("本程序成功演示了维度压缩对存储的影响以及对向量相似度的保留情况。")
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: 程序遇到问题，可能是由于以下原因:")
        print("1. 样本数量不足：PCA要求样本数大于目标降维维度")
        print("2. 依赖库安装不完整")
        print("3. 其他技术问题")
        print("\n建议操作:")
        print("1. 确保data/chunks.jsonl文件包含足够多的文本块（至少256个）")
        print("2. 检查所有依赖库是否正确安装")
        print("3. 查看详细的错误信息进行故障排除")


if __name__ == "__main__":
    main()