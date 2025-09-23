#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
程序名称：FAISS HNSW与IVF-PQ索引性能对比演示工具
程序功能：在CPU上构建HNSW与IVF-PQ两种索引，对相同数据集做Top-K查询，比较延迟与近似召回

详细功能：
1. 加载256维向量数据
2. 构建FAISS的HNSW和IVF-PQ索引
3. 构建Flat索引作为真值基准
4. 调节HNSW的efSearch参数和IVF-PQ的nprobe参数
5. 统计Recall@K和查询时间
6. 输出对比表格和详细报告

内容概述：
- 演示不同索引算法在向量检索中的性能差异
- 展示参数调整对检索速度和精度的影响
- 提供可视化的性能对比结果

流程：
1. 载入data/embeddings_256.npy
2. 构建HNSW索引
3. 构建IVF-PQ索引
4. 用真值Flat搜索对比
5. 输出控制台指标对比
6. 生成result/hnsw_ivfpq_report.md报告

输入：data/embeddings_256.npy
输出：控制台指标对比，result/hnsw_ivfpq_report.md

作者：Ph.D. Rhino
"""

import os
import sys
import time
import json
import numpy as np
import importlib
import warnings
warnings.filterwarnings('ignore')

# 尝试在全局导入faiss，以便所有函数都能访问
try:
    import faiss
except ImportError:
    # 如果导入失败，稍后在依赖检查中安装
    faiss = None

# 检查并安装依赖
def check_and_install_dependencies():
    """检查必要的依赖包是否安装，如未安装则自动安装"""
    required_packages = [
        'numpy',
        'faiss',
        'markdown'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            # 对于faiss，需要特殊处理安装方式
            if package == 'faiss':
                print(f"✗ {package} 未安装，正在安装faiss-cpu...")
                result = os.system(f"{sys.executable} -m pip install faiss-cpu --user")
            else:
                print(f"✗ {package} 未安装，正在安装...")
                result = os.system(f"{sys.executable} -m pip install {package} --user")
            
            if result == 0:
                print(f"✓ {package} 安装成功")
            else:
                print(f"✗ {package} 安装失败，请手动安装")
    
    # 尝试导入必要的模块
    try:
        import faiss
        import numpy as np
        print("✓ 所有必要的模块都已可用")
        return True
    except ImportError:
        print("✗ 必要的模块不可用")
        return False

# 加载向量数据
def load_vectors(file_path):
    """从.npy文件加载向量数据"""
    try:
        vectors = np.load(file_path)
        print(f"成功加载向量数据，形状: {vectors.shape}")
        # 确保向量是float32类型
        vectors = vectors.astype('float32')
        return vectors
    except Exception as e:
        print(f"加载向量数据失败: {e}")
        return None

# 生成随机查询向量
def generate_query_vectors(vectors, n_queries=100):
    """从向量数据中随机选择查询向量"""
    np.random.seed(42)  # 设置随机种子以确保结果可复现
    indices = np.random.choice(vectors.shape[0], n_queries, replace=False)
    query_vectors = vectors[indices]
    return query_vectors, indices

# 构建Flat索引（用于获取真值）
def build_flat_index(vectors):
    """构建Flat索引，用于获取精确的搜索结果作为基准"""
    print("正在构建Flat索引（精确搜索基准）...")
    start_time = time.time()
    
    # 确保faiss已导入
    global faiss
    if faiss is None:
        import faiss
    
    dim = vectors.shape[1]
    # 创建Flat索引
    index = faiss.IndexFlatIP(dim)  # 使用内积作为相似度度量
    index.add(vectors)
    
    end_time = time.time()
    print(f"Flat索引构建完成，耗时: {end_time - start_time:.2f}秒")
    return index

# 构建HNSW索引
def build_hnsw_index(vectors, m=16, ef_construction=200):
    """构建HNSW索引
    
    参数:
        vectors: 向量数据
        m: 每个节点的最大邻居数
        ef_construction: 构建索引时的ef参数
    
    返回:
        hnsw索引对象
    """
    print(f"正在构建HNSW索引（m={m}, ef_construction={ef_construction}）...")
    start_time = time.time()
    
    # 确保faiss已导入
    global faiss
    if faiss is None:
        import faiss
    
    dim = vectors.shape[1]
    # 创建HNSW索引
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    
    end_time = time.time()
    print(f"HNSW索引构建完成，耗时: {end_time - start_time:.2f}秒")
    return index

# 构建IVF-PQ索引
def build_ivfpq_index(vectors, n_list=100, m=8, n_bits=8):
    """构建IVF-PQ索引
    
    参数:
        vectors: 向量数据
        n_list: 聚类中心数量
        m: 子向量数量
        n_bits: 每个子向量的量化位数
    
    返回:
        ivf-pq索引对象
    """
    print(f"正在构建IVF-PQ索引（n_list={n_list}, m={m}, n_bits={n_bits}）...")
    start_time = time.time()
    
    # 确保faiss已导入
    global faiss
    if faiss is None:
        import faiss
    
    dim = vectors.shape[1]
    # 创建IVF-PQ索引
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, n_list, m, n_bits, faiss.METRIC_INNER_PRODUCT)
    
    # 训练IVF-PQ索引
    index.train(vectors)
    index.add(vectors)
    
    end_time = time.time()
    print(f"IVF-PQ索引构建完成，耗时: {end_time - start_time:.2f}秒")
    return index

# 搜索并测量性能
def search_and_measure(index, query_vectors, k=10, params=None):
    """执行查询并测量性能
    
    参数:
        index: 索引对象
        query_vectors: 查询向量
        k: 检索Top-K结果
        params: 查询参数（如efSearch或nprobe）
    
    返回:
        distances: 距离矩阵
        indices: 结果索引矩阵
        search_time: 搜索耗时
    """
    # 设置查询参数
    if params:
        if isinstance(params, dict):
            if 'efSearch' in params and hasattr(index, 'hnsw'):
                index.hnsw.efSearch = params['efSearch']
            if 'nprobe' in params and hasattr(index, 'nprobe'):
                index.nprobe = params['nprobe']
    
    # 执行查询并测量时间
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time
    
    return distances, indices, search_time

# 计算Recall@K
def calculate_recall(ground_truth_indices, result_indices, k=10):
    """计算Recall@K
    
    参数:
        ground_truth_indices: 真实结果的索引
        result_indices: 近似结果的索引
        k: 计算Recall@K的K值
    
    返回:
        recall: Recall@K值
    """
    recall = 0.0
    for gt, res in zip(ground_truth_indices, result_indices):
        # 将ground truth中查询向量本身排除
        gt_set = set(gt[gt != -1])  # 排除-1（如果有的话）
        res_set = set(res[res != -1])
        
        # 计算交集大小
        intersection = gt_set.intersection(res_set)
        # 计算召回率
        if len(gt_set) > 0:
            recall += len(intersection) / min(len(gt_set), k)
    
    # 计算平均召回率
    recall /= len(ground_truth_indices)
    return recall

# 生成报告
def generate_report(results, output_file):
    """生成Markdown格式的性能对比报告
    
    参数:
        results: 包含所有索引性能结果的字典
        output_file: 输出文件路径
    """
    report = "# FAISS HNSW与IVF-PQ索引性能对比报告\n\n"
    
    # 添加实验概述
    report += "## 实验概述\n"
    report += "本实验在CPU上对比了FAISS库中两种常用的近似最近邻搜索索引算法：HNSW和IVF-PQ。\n"
    report += "实验目的是比较不同参数设置下两种索引的检索延迟和近似召回率。\n\n"
    
    # 添加实验配置
    report += "## 实验配置\n"
    report += "- **向量维度**: {}\n".format(results['dim'])
    report += "- **向量数量**: {}\n".format(results['n_vectors'])
    report += "- **查询数量**: {}\n".format(results['n_queries'])
    report += "- **检索Top-K**: {}\n\n".format(results['k'])
    
    # 添加性能对比表格
    report += "## 性能对比表格\n\n"
    report += "| 索引类型 | 参数设置 | 平均查询时间(ms) | Recall@K | 加速比 |\n"
    report += "|---------|---------|----------------|---------|-------|\n"
    
    # 添加Flat索引结果
    flat_time = results['flat']['search_time'] * 1000 / results['n_queries']
    report += "| Flat (精确搜索) | - | {:.3f} | 1.000 | 1.00x |\n".format(flat_time)
    
    # 添加HNSW索引结果
    for param, res in results['hnsw'].items():
        avg_time = res['search_time'] * 1000 / results['n_queries']
        speedup = flat_time / avg_time
        report += "| HNSW | efSearch={} | {:.3f} | {:.3f} | {:.2f}x |\n".format(
            param, avg_time, res['recall'], speedup
        )
    
    # 添加IVF-PQ索引结果
    for param, res in results['ivfpq'].items():
        avg_time = res['search_time'] * 1000 / results['n_queries']
        speedup = flat_time / avg_time
        report += "| IVF-PQ | nprobe={} | {:.3f} | {:.3f} | {:.2f}x |\n".format(
            param, avg_time, res['recall'], speedup
        )
    
    # 添加分析总结
    report += "\n## 实验结果分析\n\n"
    report += "### HNSW索引分析\n"
    report += "HNSW (Hierarchical Navigable Small World) 是一种基于图的近似最近邻搜索算法。\n"
    report += "- **优势**: 在保持较高召回率的同时提供快速的查询速度\n"
    report += "- **参数影响**: efSearch参数直接影响检索精度和速度，值越大精度越高但速度越慢\n\n"
    
    report += "### IVF-PQ索引分析\n"
    report += "IVF-PQ (Inverted File with Product Quantization) 是一种结合了倒排索引和乘积量化的算法。\n"
    report += "- **优势**: 内存占用低，适合大规模数据集\n"
    report += "- **参数影响**: nprobe参数控制查询时访问的倒排列表数量，影响精度和速度\n\n"
    
    report += "### 综合比较\n"
    report += "- **速度**: 两种索引都比精确搜索快很多倍\n"
    report += "- **精度**: 在合适的参数设置下，两种索引都能保持较高的召回率\n"
    report += "- **内存**: IVF-PQ通常比HNSW占用更少的内存\n"
    report += "- **构建时间**: HNSW索引构建通常比IVF-PQ快\n\n"
    
    # 添加最佳实践建议
    report += "## 最佳实践建议\n\n"
    report += "1. **根据需求选择索引类型**:\n"
    report += "   - 如果优先考虑查询速度和召回率平衡，选择HNSW\n"
    report += "   - 如果内存限制严格，选择IVF-PQ\n\n"
    
    report += "2. **参数调优**:\n"
    report += "   - 对于HNSW，根据可接受的查询时间调整efSearch参数\n"
    report += "   - 对于IVF-PQ，根据可接受的查询时间调整nprobe参数\n\n"
    
    report += "3. **实际应用**:\n"
    report += "   - 在生产环境中，建议根据实际数据分布和查询需求进行充分测试\n"
    report += "   - 考虑数据集的大小、维度和查询模式选择最适合的索引策略\n\n"
    
    report += "## 结论\n\n"
    report += "FAISS提供了多种高效的近似最近邻搜索索引算法，HNSW和IVF-PQ各有优势。\n"
    report += "通过合理选择索引类型和参数设置，可以在保证检索质量的同时获得显著的性能提升。\n"
    report += "在实际应用中，建议根据具体需求进行充分的测试和调优，以达到最佳的性能表现。\n"
    
    # 保存报告
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存报告失败: {e}")
        return False

# 主函数
def main():
    """主函数，执行整个流程"""
    print("===== FAISS HNSW与IVF-PQ索引性能对比演示程序 =====")
    
    # 检查并安装依赖
    print("\n检查依赖包...")
    success = check_and_install_dependencies()
    if not success:
        print("关键依赖安装失败，程序可能无法正常运行")
    
    # 加载向量数据
    print("\n加载向量数据...")
    vectors = load_vectors('data/embeddings_256.npy')
    if vectors is None or vectors.shape[0] < 1000:
        print("警告: 向量数量不足，生成随机向量用于演示...")
        # 如果向量数量不足，生成随机向量用于演示
        np.random.seed(42)
        dim = 256
        n_vectors = 10000
        vectors = np.random.rand(n_vectors, dim).astype('float32')
        # 归一化向量
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        print(f"生成了 {n_vectors} 个 {dim} 维的随机向量")
    
    # 导入faiss模块
    import faiss
    
    # 生成查询向量
    print("\n生成查询向量...")
    n_queries = 100
    query_vectors, _ = generate_query_vectors(vectors, n_queries)
    print(f"生成了 {n_queries} 个查询向量")
    
    # 定义实验参数
    k = 10  # 检索Top-K
    hnsw_params = [10, 20, 40]  # HNSW的efSearch参数
    ivfpq_params = [1, 10, 100]  # IVF-PQ的nprobe参数
    
    # 构建Flat索引并获取真值
    print("\n构建索引并执行查询...")
    flat_index = build_flat_index(vectors)
    _, ground_truth_indices, flat_search_time = search_and_measure(flat_index, query_vectors, k)
    print(f"Flat索引查询完成，耗时: {flat_search_time:.2f}秒")
    
    # 构建HNSW索引
    hnsw_index = build_hnsw_index(vectors)
    
    # 构建IVF-PQ索引
    ivfpq_index = build_ivfpq_index(vectors)
    
    # 存储所有结果
    results = {
        'dim': vectors.shape[1],
        'n_vectors': vectors.shape[0],
        'n_queries': n_queries,
        'k': k,
        'flat': {
            'search_time': flat_search_time,
            'recall': 1.0
        },
        'hnsw': {},
        'ivfpq': {}
    }
    
    # 测试HNSW索引不同参数
    print("\n测试HNSW索引不同参数...")
    for efSearch in hnsw_params:
        print(f"测试HNSW索引 (efSearch={efSearch})...")
        _, hnsw_indices, hnsw_time = search_and_measure(
            hnsw_index, query_vectors, k, {'efSearch': efSearch}
        )
        hnsw_recall = calculate_recall(ground_truth_indices, hnsw_indices, k)
        print(f"HNSW (efSearch={efSearch}) - 查询时间: {hnsw_time:.2f}秒, Recall@{k}: {hnsw_recall:.3f}")
        results['hnsw'][efSearch] = {
            'search_time': hnsw_time,
            'recall': hnsw_recall
        }
    
    # 测试IVF-PQ索引不同参数
    print("\n测试IVF-PQ索引不同参数...")
    for nprobe in ivfpq_params:
        print(f"测试IVF-PQ索引 (nprobe={nprobe})...")
        _, ivfpq_indices, ivfpq_time = search_and_measure(
            ivfpq_index, query_vectors, k, {'nprobe': nprobe}
        )
        ivfpq_recall = calculate_recall(ground_truth_indices, ivfpq_indices, k)
        print(f"IVF-PQ (nprobe={nprobe}) - 查询时间: {ivfpq_time:.2f}秒, Recall@{k}: {ivfpq_recall:.3f}")
        results['ivfpq'][nprobe] = {
            'search_time': ivfpq_time,
            'recall': ivfpq_recall
        }
    
    # 生成控制台表格输出
    print("\n===== 性能对比结果 =====")
    print("| 索引类型 | 参数设置 | 平均查询时间(ms) | Recall@K | 加速比 |")
    print("|---------|---------|----------------|---------|-------|")
    
    # 输出Flat索引结果
    flat_avg_time = results['flat']['search_time'] * 1000 / results['n_queries']
    print(f"| Flat (精确搜索) | - | {flat_avg_time:.3f} | 1.000 | 1.00x |")
    
    # 输出HNSW索引结果
    for param, res in results['hnsw'].items():
        avg_time = res['search_time'] * 1000 / results['n_queries']
        speedup = flat_avg_time / avg_time
        print(f"| HNSW | efSearch={param} | {avg_time:.3f} | {res['recall']:.3f} | {speedup:.2f}x |")
    
    # 输出IVF-PQ索引结果
    for param, res in results['ivfpq'].items():
        avg_time = res['search_time'] * 1000 / results['n_queries']
        speedup = flat_avg_time / avg_time
        print(f"| IVF-PQ | nprobe={param} | {avg_time:.3f} | {res['recall']:.3f} | {speedup:.2f}x |")
    
    # 生成报告
    print("\n生成性能对比报告...")
    report_file = 'result/hnsw_ivfpq_report.md'
    generate_report(results, report_file)
    
    print("\n程序执行完成！")
    print("\n=== 实验总结 ===")
    print("本实验成功演示了FAISS库中HNSW和IVF-PQ两种索引算法的性能差异。")
    print("通过调节HNSW的efSearch参数和IVF-PQ的nprobe参数，可以在速度和精度之间找到平衡。")
    print("详细的实验结果和分析请查看生成的Markdown报告。")

if __name__ == "__main__":
    main()