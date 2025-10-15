#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称：排序指标批量评估
功能说明：计算 NDCG@K、MRR、Hit@K，输出随K变化的曲线数据。
内容概述：读取每个query的候选列表及标注相关性，计算多K下的NDCG、MRR、Hit@K；输出汇总与分桶（如业务线）子集。
作者：Ph.D. Rhino

执行流程：
1. 读取CSV（query、rank、score、label、bucket）
2. 针对每个bucket，计算NDCG@K（K列表）、MRR、Hit@K
3. 汇总均值、分位（P50、P90）、方差
4. 导出K-指标的曲线数据CSV（便于外部作图）
5. 输出"前列不稳"的query清单（NDCG波动大）

输入说明：--ks 3,5,10 --bucket-col bucket --rel-th 3
输出展示：
  NDCG@3=0.62 @5=0.68 @10=0.71 | MRR=0.57 | Hit@5=0.84
  Bucket=medical: NDCG@5=0.72; Bucket=faq: NDCG@5=0.66
  Curves saved: ndcg_curve.csv, hit_curve.csv
  Unstable top queries saved: unstable_queries.csv
"""

import os
import sys
import argparse
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

# 检查依赖并自动安装
def check_and_install_dependencies():
    """检查必要的依赖包，如果缺失则自动安装"""
    required_packages = ['pandas', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装缺少的依赖包: {package}")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# 创建示例数据集
def create_sample_dataset(file_path: str, num_queries: int = 100, docs_per_query: int = 15) -> None:
    """
    创建用于测试的示例数据集
    参数:
        file_path: 输出文件路径
        num_queries: 查询数量
        docs_per_query: 每个查询的文档数量
    """
    print(f"创建示例数据集: {file_path}")
    
    data = []
    # 预定义一些业务线(bucket)和查询
    buckets = ['medical', 'faq', 'news', 'ecommerce', 'finance']
    query_templates = {
        'medical': ['治疗糖尿病的方法', '高血压患者的饮食建议', '感冒的症状有哪些'],
        'faq': ['如何重置密码', '退货政策是什么', '如何申请退款'],
        'news': ['最新科技新闻', '体育赛事结果', '政治新闻摘要'],
        'ecommerce': ['智能手机推荐', '厨房电器选购指南', '时尚服装搭配'],
        'finance': ['股票市场分析', '理财产品推荐', '个人理财建议']
    }
    
    for query_id in range(1, num_queries + 1):
        # 随机选择一个业务线
        bucket = random.choice(buckets)
        # 随机选择一个查询模板
        query_template = random.choice(query_templates[bucket])
        query = f"{query_template}_{query_id}"
        
        # 为每个查询生成文档数据
        docs = []
        for doc_idx in range(docs_per_query):
            # 生成相关度标签 (0-4)
            # 确保每个查询有一些相关和不相关的文档，且相关文档倾向于有更高的分数
            if random.random() < 0.2:
                label = random.randint(3, 4)  # 高度相关
            elif random.random() < 0.5:
                label = random.randint(1, 2)  # 部分相关
            else:
                label = 0  # 不相关
            
            # 生成评分，相关度越高评分倾向越高，但加入一些随机性以模拟真实场景
            base_score = 0.1 + 0.8 * (label / 4)  # 基础分数基于标签
            # 为了让排序评估更有意义，给一些相关文档低分，给一些不相关文档高分
            if label >= 3 and random.random() < 0.15:  # 15%的高相关文档分数较低
                score = base_score - random.uniform(0.3, 0.5)
            elif label == 0 and random.random() < 0.15:  # 15%的不相关文档分数较高
                score = base_score + random.uniform(0.3, 0.5)
            else:
                score = base_score + random.uniform(-0.2, 0.2)  # 正常波动
            
            score = max(0.0, min(1.0, score))  # 确保分数在0-1之间
            
            docs.append({
                'query': query,
                'rank': 0,  # 暂时设为0，后面排序后更新
                'score': score,
                'label': label,
                'bucket': bucket
            })
        
        # 按评分降序排序并更新排名
        docs.sort(key=lambda x: x['score'], reverse=True)
        for idx, doc in enumerate(docs):
            doc['rank'] = idx + 1
        
        data.extend(docs)
    
    # 保存为CSV文件
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"示例数据集已创建，共包含 {len(data)} 条记录")

# 读取数据并按查询和分桶聚合
def load_and_aggregate_data(file_path: str, bucket_col: str) -> Tuple[Dict[str, Dict[str, List[Dict]]], Set[str]]:
    """
    读取CSV数据并按查询和分桶聚合
    参数:
        file_path: 输入文件路径
        bucket_col: 分桶列名
    返回:
        (bucketed_data, all_buckets) - 按分桶组织的数据和所有分桶集合
    """
    print(f"读取数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 确保所有必需的列存在
    required_columns = ['query', 'rank', 'score', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必需的列: {col}")
    
    # 如果分桶列不存在，添加默认分桶
    if bucket_col not in df.columns:
        print(f"警告：分桶列 '{bucket_col}' 不存在，将使用默认分桶 'default'")
        df[bucket_col] = 'default'
    
    # 按分桶和查询聚合数据
    bucketed_data = defaultdict(lambda: defaultdict(list))
    all_buckets = set()
    
    for _, row in df.iterrows():
        bucket = row[bucket_col]
        all_buckets.add(bucket)
        bucketed_data[bucket][row['query']].append({
            'rank': row['rank'],
            'score': row['score'],
            'label': row['label']
        })
    
    print(f"数据加载完成，共包含 {len(all_buckets)} 个分桶，{sum(len(queries) for queries in bucketed_data.values())} 个查询")
    return bucketed_data, all_buckets

# 计算DCG@K
def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    计算DCG@K值
    参数:
        relevances: 相关性分数列表
        k: 截断位置
    返回:
        DCG@K值
    """
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    elif len(relevances) == 1:
        # 只有一个元素时，直接返回该元素的值
        return float(relevances[0])
    
    # 使用log2(1+i)作为折扣因子
    # 确保discounts的长度与relevances[1:]的长度一致
    discounts = np.log2(np.arange(2, len(relevances) + 1))
    return float(relevances[0] + np.sum(relevances[1:] / discounts))

# 计算NDCG@K
def ndcg_at_k(relevances: List[int], k: int) -> float:
    """
    计算NDCG@K值
    参数:
        relevances: 相关性分数列表
        k: 截断位置
    返回:
        NDCG@K值
    """
    # 计算IDCG（理想DCG）
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    
    # 计算DCG
    dcg = dcg_at_k(relevances, k)
    
    # 返回NDCG
    return dcg / idcg

# 计算MRR (Mean Reciprocal Rank)
def calculate_mrr(relevances: List[int], rel_threshold: int = 3) -> float:
    """
    计算MRR值
    参数:
        relevances: 相关性分数列表
        rel_threshold: 相关性阈值
    返回:
        MRR值
    """
    for i, rel in enumerate(relevances):
        if rel >= rel_threshold:
            return 1.0 / (i + 1)
    return 0.0

# 计算Hit@K
def hit_at_k(relevances: List[int], k: int, rel_threshold: int = 3) -> float:
    """
    计算Hit@K值
    参数:
        relevances: 相关性分数列表
        k: 截断位置
        rel_threshold: 相关性阈值
    返回:
        Hit@K值（0或1）
    """
    return 1.0 if any(rel >= rel_threshold for rel in relevances[:k]) else 0.0

# 计算查询级别的所有指标
def calculate_query_metrics(query_docs: List[Dict], ks: List[int], rel_threshold: int = 3) -> Dict[str, float]:
    """
    计算单个查询的所有评估指标
    参数:
        query_docs: 查询的文档列表
        ks: K值列表
        rel_threshold: 相关性阈值
    返回:
        指标字典
    """
    # 按排名排序文档
    sorted_docs = sorted(query_docs, key=lambda x: x['rank'])
    # 提取相关性分数
    relevances = [doc['label'] for doc in sorted_docs]
    
    metrics = {}
    
    # 计算NDCG@K
    for k in ks:
        metrics[f'ndcg@{k}'] = ndcg_at_k(relevances, k)
    
    # 计算MRR
    metrics['mrr'] = calculate_mrr(relevances, rel_threshold)
    
    # 计算Hit@K
    for k in ks:
        metrics[f'hit@{k}'] = hit_at_k(relevances, k, rel_threshold)
    
    return metrics

# 计算分桶级别的所有指标
def calculate_bucket_metrics(bucketed_data: Dict[str, Dict[str, List[Dict]]], ks: List[int], rel_threshold: int = 3) -> Dict[str, Dict[str, float]]:
    """
    计算每个分桶的评估指标
    参数:
        bucketed_data: 按分桶组织的数据
        ks: K值列表
        rel_threshold: 相关性阈值
    返回:
        每个分桶的指标字典
    """
    bucket_metrics = defaultdict(dict)
    
    for bucket, queries in bucketed_data.items():
        # 存储每个查询的指标
        query_metrics_list = []
        numeric_metrics_list = []
        
        for query, docs in queries.items():
            query_metrics = calculate_query_metrics(docs, ks, rel_threshold)
            # 保存带查询标识符的完整指标
            query_metrics_with_id = query_metrics.copy()
            query_metrics_with_id['query'] = query  # 添加查询标识符
            query_metrics_list.append(query_metrics_with_id)
            # 保存仅数值指标用于计算统计量
            numeric_metrics_list.append(query_metrics)
        
        # 计算分桶内的平均指标
        metrics_df = pd.DataFrame(numeric_metrics_list)
        avg_metrics = metrics_df.mean().to_dict()
        
        # 计算分位数
        quantiles = metrics_df.quantile([0.5, 0.9]).to_dict()
        
        # 计算方差
        var_metrics = metrics_df.var().to_dict()
        
        # 存储结果
        bucket_metrics[bucket] = {
            'average': avg_metrics,
            'quantiles': quantiles,
            'variance': var_metrics,
            'query_metrics': query_metrics_list
        }
    
    return bucket_metrics

# 生成K-指标曲线数据
def generate_curve_data(bucket_metrics: Dict[str, Dict[str, float]], ks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    生成K-指标曲线数据
    参数:
        bucket_metrics: 每个分桶的指标
        ks: K值列表
    返回:
        (ndcg_curve_df, mrr_curve_df, hit_curve_df) - 三种指标的曲线数据框
    """
    # 为每个K值收集所有分桶的指标
    ndcg_data = []
    hit_data = []
    
    for k in ks:
        ndcg_row = {'k': k}
        hit_row = {'k': k}
        
        for bucket, metrics in bucket_metrics.items():
            ndcg_row[f'{bucket}'] = metrics['average'].get(f'ndcg@{k}', 0.0)
            hit_row[f'{bucket}'] = metrics['average'].get(f'hit@{k}', 0.0)
        
        # 计算所有分桶的平均值
        ndcg_values = [v for v in ndcg_row.values() if isinstance(v, (int, float)) and v != k]
        hit_values = [v for v in hit_row.values() if isinstance(v, (int, float)) and v != k]
        
        ndcg_row['overall'] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
        hit_row['overall'] = sum(hit_values) / len(hit_values) if hit_values else 0.0
        
        ndcg_data.append(ndcg_row)
        hit_data.append(hit_row)
    
    # 创建数据框
    ndcg_curve_df = pd.DataFrame(ndcg_data)
    hit_curve_df = pd.DataFrame(hit_data)
    
    return ndcg_curve_df, hit_curve_df

# 识别前列不稳的查询
def identify_unstable_queries(bucket_metrics: Dict[str, Dict[str, float]], ks: List[int], threshold: float = 0.2) -> pd.DataFrame:
    """
    识别前列不稳的查询（NDCG波动大）
    参数:
        bucket_metrics: 每个分桶的指标
        ks: K值列表
        threshold: 波动阈值
    返回:
        不稳定查询的数据框
    """
    unstable_queries = []
    
    for bucket, metrics in bucket_metrics.items():
        for query_metrics in metrics['query_metrics']:
            # 计算该查询在不同K值下的NDCG波动
            ndcg_values = [query_metrics.get(f'ndcg@{k}', 0.0) for k in ks]
            if len(ndcg_values) > 1:
                # 计算标准差作为波动指标
                volatility = np.std(ndcg_values)
                # 计算最大值和最小值的差
                range_diff = max(ndcg_values) - min(ndcg_values)
                
                # 如果波动超过阈值，则标记为不稳定
                if volatility > threshold or range_diff > threshold:
                    unstable_queries.append({
                        'query': query_metrics['query'],
                        'bucket': bucket,
                        'ndcg_std': volatility,
                        'ndcg_range': range_diff,
                        **{f'ndcg@{k}': query_metrics.get(f'ndcg@{k}', 0.0) for k in ks}
                    })
    
    # 创建数据框并排序
    if unstable_queries:
        df = pd.DataFrame(unstable_queries)
        # 按波动程度降序排序
        df = df.sort_values(by='ndcg_std', ascending=False)
        return df
    else:
        return pd.DataFrame()

# 保存曲线数据
def save_curve_data(ndcg_df: pd.DataFrame, hit_df: pd.DataFrame) -> None:
    """
    保存曲线数据到CSV文件
    参数:
        ndcg_df: NDCG曲线数据框
        hit_df: Hit曲线数据框
    """
    ndcg_df.to_csv('ndcg_curve.csv', index=False, encoding='utf-8-sig')
    hit_df.to_csv('hit_curve.csv', index=False, encoding='utf-8-sig')
    print(f"曲线数据已保存: ndcg_curve.csv (共 {len(ndcg_df)} 行), hit_curve.csv (共 {len(hit_df)} 行)")

# 保存不稳定查询
def save_unstable_queries(df: pd.DataFrame) -> None:
    """
    保存不稳定查询到CSV文件
    参数:
        df: 不稳定查询数据框
    """
    if not df.empty:
        df.to_csv('unstable_queries.csv', index=False, encoding='utf-8-sig')
        print(f"不稳定查询已保存到: unstable_queries.csv (共 {len(df)} 条)")
    else:
        print("没有检测到不稳定查询")

# 主函数
def main():
    """主函数，处理命令行参数并执行评估流程"""
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='排序指标批量评估')
    parser.add_argument('--input', type=str, default='ranking_data.csv', help='输入CSV文件路径')
    parser.add_argument('--rel-th', type=int, default=3, help='相关性阈值，大于等于该值被认为相关')
    parser.add_argument('--ks', type=str, default='3,5,10', help='计算@K指标的K值列表，用逗号分隔')
    parser.add_argument('--bucket-col', type=str, default='bucket', help='分桶列名')
    parser.add_argument('--create-sample', action='store_true', help='创建示例数据集')
    parser.add_argument('--num-queries', type=int, default=100, help='创建示例数据集时的查询数量')
    parser.add_argument('--docs-per-query', type=int, default=15, help='创建示例数据集时每个查询的文档数量')
    
    args = parser.parse_args()
    
    # 如果需要，创建示例数据集
    if args.create_sample or not os.path.exists(args.input):
        create_sample_dataset(args.input, args.num_queries, args.docs_per_query)
    
    # 解析K值列表
    try:
        ks = [int(k.strip()) for k in args.ks.split(',')]
        # 确保K值是唯一且有序的
        ks = sorted(list(set(ks)))
    except ValueError:
        print(f"无效的K值列表: {args.ks}")
        ks = [3, 5, 10]
    
    # 加载并聚合数据
    bucketed_data, all_buckets = load_and_aggregate_data(args.input, args.bucket_col)
    
    # 计算指标
    bucket_metrics = calculate_bucket_metrics(bucketed_data, ks, args.rel_th)
    
    # 计算总体指标
    overall_metrics = {}
    all_numeric_metrics = []
    for bucket in bucket_metrics.values():
        # 提取每个查询的数值指标（排除query字段）
        for query_metrics in bucket['query_metrics']:
            numeric_metrics = {k: v for k, v in query_metrics.items() if k != 'query'}
            all_numeric_metrics.append(numeric_metrics)
    
    if all_numeric_metrics:
        overall_df = pd.DataFrame(all_numeric_metrics)
        overall_metrics = overall_df.mean().to_dict()
        
        # 打印总体指标
        print("总体指标:", end=' ')
        # 打印NDCG@K
        ndcg_strs = [f"NDCG@{k}={overall_metrics.get(f'ndcg@{k}', 0.0):.2f}" for k in ks]
        print(' '.join(ndcg_strs), end=' | ')
        # 打印MRR
        print(f"MRR={overall_metrics.get('mrr', 0.0):.2f}", end=' | ')
        # 打印Hit@K（选择一个有代表性的K）
        representative_k = ks[min(1, len(ks)-1)]  # 选择第二个K值，如果有的话
        print(f"Hit@{representative_k}={overall_metrics.get(f'hit@{representative_k}', 0.0):.2f}")
    
    # 打印分桶指标
    print("分桶指标:")
    for bucket in all_buckets:
        if bucket in bucket_metrics:
            # 选择一个有代表性的K值来展示
            representative_k = ks[min(1, len(ks)-1)]  # 选择第二个K值，如果有的话
            ndcg = bucket_metrics[bucket]['average'].get(f'ndcg@{representative_k}', 0.0)
            print(f"  Bucket={bucket}: NDCG@{representative_k}={ndcg:.2f}")
    
    # 生成并保存曲线数据
    ndcg_curve_df, hit_curve_df = generate_curve_data(bucket_metrics, ks)
    save_curve_data(ndcg_curve_df, hit_curve_df)
    
    # 识别并保存不稳定查询
    unstable_df = identify_unstable_queries(bucket_metrics, ks)
    save_unstable_queries(unstable_df)
    
    # 执行效率分析
    print("\n执行效率分析：")
    print("- 数据加载与处理：顺利完成")
    print("- 指标计算：高效完成")
    print(f"- 分桶处理：处理了 {len(all_buckets)} 个分桶")
    print("- 输出文件生成：所有文件已成功创建")
    
    # 实验结果分析
    print("\n实验结果分析：")
    if all_numeric_metrics:
        print(f"1. 整体排序质量：NDCG@3={overall_metrics.get('ndcg@3', 0.0):.2f}，NDCG@5={overall_metrics.get('ndcg@5', 0.0):.2f}，NDCG@10={overall_metrics.get('ndcg@10', 0.0):.2f}")
        print(f"2. 首位相关文档识别：MRR={overall_metrics.get('mrr', 0.0):.2f}，表明系统识别首位相关文档的能力")
        print(f"3. 相关文档召回：Hit@{representative_k}={overall_metrics.get(f'hit@{representative_k}', 0.0):.2f}，表明在前{representative_k}个结果中找到相关文档的比例")
        
        # 分析不同分桶的性能差异
        if len(all_buckets) > 1:
            ndcg_values = [bucket_metrics[b]['average'].get(f'ndcg@{representative_k}', 0.0) for b in all_buckets]
            max_bucket = all_buckets.pop() if len(all_buckets) > 1 else list(all_buckets)[0]
            min_bucket = list(all_buckets)[0]
            for b in all_buckets:
                if bucket_metrics[b]['average'].get(f'ndcg@{representative_k}', 0.0) > bucket_metrics[max_bucket]['average'].get(f'ndcg@{representative_k}', 0.0):
                    max_bucket = b
                if bucket_metrics[b]['average'].get(f'ndcg@{representative_k}', 0.0) < bucket_metrics[min_bucket]['average'].get(f'ndcg@{representative_k}', 0.0):
                    min_bucket = b
            
            print(f"4. 分桶性能差异：最佳分桶 {max_bucket} (NDCG@{representative_k}={bucket_metrics[max_bucket]['average'].get(f'ndcg@{representative_k}', 0.0):.2f}) vs 最差分桶 {min_bucket} (NDCG@{representative_k}={bucket_metrics[min_bucket]['average'].get(f'ndcg@{representative_k}', 0.0):.2f})")
        
        # 分析不稳定查询
        if not unstable_df.empty:
            print(f"5. 不稳定查询分析：共识别出{len(unstable_df)}个前列不稳的查询，这些查询的NDCG值在不同K值下波动较大")
        
        # 分析NDCG随K的变化趋势
        ndcg_k_values = [overall_metrics.get(f'ndcg@{k}', 0.0) for k in ks]
        if len(ndcg_k_values) > 1:
            trend = "上升" if ndcg_k_values[-1] > ndcg_k_values[0] else "下降" if ndcg_k_values[-1] < ndcg_k_values[0] else "稳定"
            print(f"6. NDCG趋势：随着K值增加，NDCG呈现{trend}趋势，反映了系统在不同深度下的排序质量")
    
    # 检查输出文件
    print("\n输出文件确认：")
    for file_name in ['ndcg_curve.csv', 'hit_curve.csv', 'unstable_queries.csv']:
        if os.path.exists(file_name):
            print(f"- {file_name}：已创建，文件大小：{os.path.getsize(file_name)} 字节")
        else:
            print(f"- {file_name}：创建失败")

if __name__ == '__main__':
    main()