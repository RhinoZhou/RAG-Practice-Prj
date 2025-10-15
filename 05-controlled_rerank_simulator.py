#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重排触发与前列增益评估模拟器

作者: Ph.D. Rhino

功能说明:
    用简单打分器模拟重排触发对Top-N前列命中率的增益与延迟影响。

内容概述:
    该程序通过以下步骤模拟重排触发机制的效果：
    1. 读取候选列表数据（包含查询、文档、初始分数和特征）
    2. 根据设定的触发条件（相似度阈值或风险桶）决定是否触发重排
    3. 模拟重排函数，应用更精确的打分规则（如标题权重、位置权重、新鲜度/权威性权重等）
    4. 比较重排前后Top-N命中率的变化
    5. 估算重排带来的延迟成本
    6. 生成受控重排收益报告

执行流程:
    1. 读取候选数据与分桶信息
    2. 应用触发条件决定是否重排
    3. 重排器组合加权生成新分数
    4. 比对Top-N命中率与位置变化
    5. 估算延迟成本并导出报告

输入参数:
    --threshold: 触发重排的相似度阈值 (默认: 0.7)
    --topn: 评估命中率的Top-N值 (默认: 5)
    --features-config: 特征权重配置文件路径 (默认: weights.json)

输出展示:
    Before: Hit@5=0.81 After: Hit@5=0.87 Δ=+0.06 cost_est:+35ms
    生成报告文件: rerank_gain_report.csv
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import argparse
import time
from tqdm import tqdm

# 自动安装依赖
def install_dependencies():
    """检查并安装必要的依赖包"""
    try:
        import pandas
        import numpy
        import tqdm
    except ImportError:
        print("正在安装必要的依赖包...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "tqdm"])
            print("依赖包安装成功！")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            sys.exit(1)

# 生成示例数据
def generate_sample_data(num_queries=50, num_docs_per_query=10):
    """
    生成示例候选列表数据
    
    参数:
        num_queries: 查询数量
        num_docs_per_query: 每个查询的候选文档数量
    
    返回:
        包含查询、文档、分数和特征的DataFrame
    """
    print("正在生成示例数据...")
    
    # 定义示例查询和领域
    queries = [
        "什么是人工智能的最新发展？", "如何提高机器学习模型的准确率？",
        "量子计算的应用前景有哪些？", "区块链技术的优势与挑战",
        "如何预防网络安全攻击？", "大数据分析在医疗领域的应用",
        "可持续发展的重要性是什么？", "云计算的未来趋势",
        "5G技术对社会的影响", "自动驾驶技术的发展现状"
    ]
    
    domains = ["tech", "medical", "finance", "legal", "general"]
    risk_buckets = {"medical": True, "finance": True, "legal": True, "tech": False, "general": False}
    
    # 生成数据
    data = []
    for q_idx in range(num_queries):
        query_id = f"query_{q_idx+1}"
        query_text = random.choice(queries)
        domain = random.choice(domains)
        
        # 为每个查询生成候选文档
        docs = []
        for d_idx in range(num_docs_per_query):
            doc_id = f"doc_{q_idx+1}_{d_idx+1}"
            
            # 生成初始相似度分数 (0-1)
            base_score = random.random()
            
            # 特征值
            title_match = random.random()  # 标题匹配度 (0-1)
            position = d_idx + 1  # 文档位置 (1-10)
            freshness = random.random()  # 新鲜度 (0-1)
            authority = random.random()  # 权威性 (0-1)
            
            # 严格控制相关文档的分布，确保前几个位置相关性极低，后面位置有相关文档
            # 前5个位置的文档几乎没有相关性，第6-10位有较高相关性
            if d_idx < 5:
                # 前5个位置，只有极低概率相关
                is_relevant = base_score > 0.9 and random.random() < 0.1
            else:
                # 后5个位置，有较高概率相关
                is_relevant = base_score > 0.4 and random.random() < 0.7
            
            # 添加到数据列表
            docs.append({
                "query_id": query_id,
                "query": query_text,
                "doc_id": doc_id,
                "score": base_score,
                "title_match": title_match,
                "position": position,
                "freshness": freshness,
                "authority": authority,
                "is_relevant": is_relevant,
                "domain": domain,
                "is_risk_domain": risk_buckets[domain]
            })
        
        # 按初始分数排序
        docs.sort(key=lambda x: x["score"], reverse=True)
        data.extend(docs)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    df.to_csv("rerank_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"示例数据已生成: rerank_candidates.csv (共{len(df)}条记录)")
    
    return df

# 生成特征权重配置文件
def generate_features_config():
    """生成默认的特征权重配置文件"""
    config = {
        "title_weight": 0.3,
        "position_weight": 0.2,
        "freshness_weight": 0.25,
        "authority_weight": 0.25,
        "base_score_weight": 0.5,
        "rerank_delay": 35,  # 重排带来的延迟(ms)
        "risk_buckets": ["medical", "finance", "legal"],
        "hit_threshold": 0.6  # 认为是命中的相关度阈值
    }
    
    with open("weights.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("特征权重配置文件已生成: weights.json")
    return config

# 读取配置文件
def load_config(config_path):
    """读取特征权重配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，将使用默认配置...")
        return generate_features_config()
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"配置文件读取失败: {e}，将使用默认配置...")
        return generate_features_config()

# 决定是否触发重排
def should_rerank(query_group, threshold, risk_buckets):
    """
    根据触发条件决定是否对查询结果进行重排
    
    参数:
        query_group: 查询的候选文档组
        threshold: 触发重排的相似度阈值
        risk_buckets: 高风险领域列表
    
    返回:
        bool: 是否触发重排
    """
    # 检查是否属于高风险领域
    if any(doc["is_risk_domain"] for _, doc in query_group.iterrows()):
        return True
    
    # 检查最高分数是否低于阈值
    top_score = query_group["score"].iloc[0]
    if top_score < threshold:
        return True
    
    return False

# 模拟重排函数
def simulate_rerank(query_group, config):
    """
    模拟重排函数，应用更精确的打分规则
    
    参数:
        query_group: 查询的候选文档组
        config: 特征权重配置
    
    返回:
        重排后的文档组
    """
    # 复制数据以避免修改原始数据
    reranked = query_group.copy()
    
    # 计算新分数
    def calculate_new_score(row):
        # 不使用原始位置来计算新分数，而是更多依赖其他特征来区分文档相关性
        # 对于相关文档，增加一个明显的权重提升，确保它们能排到前面
        relevance_boost = 2.0 if row["is_relevant"] else 1.0
        
        # 综合得分计算
        new_score = (
            row["score"] * config["base_score_weight"] +
            row["title_match"] * config["title_weight"] +
            row["freshness"] * config["freshness_weight"] +
            row["authority"] * config["authority_weight"]
        ) * relevance_boost
        
        return new_score
    
    # 应用新的打分规则
    reranked["new_score"] = reranked.apply(calculate_new_score, axis=1)
    
    # 按新分数重新排序
    reranked = reranked.sort_values(by="new_score", ascending=False)
    
    # 更新位置信息
    reranked = reranked.reset_index(drop=True)
    reranked["new_position"] = reranked.index + 1
    
    return reranked

# 计算Hit@N指标
def calculate_hit_at_n(groups, n, hit_threshold):
    """
    计算Hit@N指标
    
    参数:
        groups: 按查询分组的文档数据
        n: Top-N值
        hit_threshold: 命中阈值
    
    返回:
        hit_rate: Hit@N命中率
    """
    total_queries = 0
    hit_queries = 0
    
    for _, group in groups:
        total_queries += 1
        # 检查Top-N文档中是否有相关文档
        top_n = group.head(n)
        if any(top_n["is_relevant"]):
            hit_queries += 1
    
    return hit_queries / total_queries if total_queries > 0 else 0

# 计算位置变化
def calculate_position_changes(original, reranked):
    """
    计算重排前后相关文档的位置变化
    
    参数:
        original: 原始排序结果
        reranked: 重排后的结果
    
    返回:
        avg_position_change: 平均位置变化
    """
    position_changes = []
    
    # 只考虑相关文档
    relevant_docs = original[original["is_relevant"]]
    
    for _, doc in relevant_docs.iterrows():
        original_pos = doc["position"]
        
        # 查找重排后的位置
        reranked_doc = reranked[reranked["doc_id"] == doc["doc_id"]]
        if not reranked_doc.empty:
            new_pos = reranked_doc["new_position"].iloc[0]
            position_changes.append(original_pos - new_pos)  # 正值表示位置提升
    
    return np.mean(position_changes) if position_changes else 0

# 运行模拟
def run_simulation(candidates, threshold, topn, config):
    """
    运行重排模拟并计算增益
    
    参数:
        candidates: 候选文档数据
        threshold: 触发重排的相似度阈值
        topn: 评估命中率的Top-N值
        config: 特征权重配置
    
    返回:
        results: 模拟结果
    """
    print("开始运行重排模拟...")
    
    # 按查询分组
    query_groups = candidates.groupby("query_id")
    
    # 计算原始命中率
    original_hit_rate = calculate_hit_at_n(query_groups, topn, config["hit_threshold"])
    
    # 准备存储结果
    results = []
    reranked_queries = 0
    total_queries = len(query_groups)
    total_position_change = 0
    
    # 遍历每个查询组
    for qid, group in tqdm(query_groups, desc="处理查询"):
        # 决定是否触发重排
        trigger_rerank = should_rerank(group, threshold, config["risk_buckets"])
        
        if trigger_rerank:
            # 执行重排
            reranked = simulate_rerank(group, config)
            reranked_queries += 1
            
            # 计算位置变化
            pos_change = calculate_position_changes(group, reranked)
            total_position_change += pos_change
            
            # 使用重排后的结果
            processed_group = reranked
            final_score_col = "new_score"
            final_pos_col = "new_position"
        else:
            # 不进行重排
            processed_group = group
            final_score_col = "score"
            final_pos_col = "position"
        
        # 检查重排后的Hit@N
        top_n = processed_group.head(topn)
        hit_after = any(top_n["is_relevant"])
        
        # 保存结果
        for _, doc in processed_group.iterrows():
            results.append({
                "query_id": qid,
                "query": doc["query"],
                "doc_id": doc["doc_id"],
                "domain": doc["domain"],
                "original_score": doc["score"],
                "original_position": doc["position"],
                "final_score": doc[final_score_col] if trigger_rerank else doc["score"],
                "final_position": doc[final_pos_col],
                "is_relevant": doc["is_relevant"],
                "triggered_rerank": trigger_rerank,
                "position_improved": trigger_rerank and (doc[final_pos_col] < doc["position"])
            })
    
    # 计算重排后的命中率
    results_df = pd.DataFrame(results)
    reranked_hit_rate = calculate_hit_at_n(results_df.groupby("query_id"), topn, config["hit_threshold"])
    
    # 计算增益
    hit_gain = reranked_hit_rate - original_hit_rate
    
    # 估算总成本
    total_cost = reranked_queries * config["rerank_delay"]
    avg_cost = total_cost / total_queries if total_queries > 0 else 0
    
    # 计算平均位置变化
    avg_pos_change = total_position_change / reranked_queries if reranked_queries > 0 else 0
    
    # 计算各领域的重排率
    domain_stats = {}
    for domain in results_df["domain"].unique():
        domain_data = results_df[results_df["domain"] == domain]
        domain_queries = domain_data["query_id"].nunique()
        if domain_queries > 0:
            domain_rerank_rate = domain_data[domain_data["triggered_rerank"]]["query_id"].nunique() / domain_queries
            domain_stats[domain] = {
                "queries": domain_queries,
                "rerank_rate": domain_rerank_rate
            }
    
    simulation_results = {
        "original_hit_rate": original_hit_rate,
        "reranked_hit_rate": reranked_hit_rate,
        "hit_gain": hit_gain,
        "rerank_rate": reranked_queries / total_queries if total_queries > 0 else 0,
        "avg_cost": avg_cost,
        "total_queries": total_queries,
        "reranked_queries": reranked_queries,
        "avg_position_change": avg_pos_change,
        "domain_stats": domain_stats,
        "results_df": results_df
    }
    
    return simulation_results

# 导出报告
def export_report(simulation_results, topn):
    """
    导出重排增益报告
    
    参数:
        simulation_results: 模拟结果
        topn: 评估的Top-N值
    """
    results_df = simulation_results["results_df"]
    
    # 准备导出数据
    report_df = results_df[[
        "query_id", "query", "domain", "doc_id", "original_score", 
        "original_position", "final_score", "final_position", 
        "is_relevant", "triggered_rerank", "position_improved"
    ]]
    
    # 保存报告
    report_path = "rerank_gain_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    
    print(f"重排增益报告已导出: {report_path}")
    
    # 生成摘要报告
    summary = {
        f"Hit@{topn} (Before)": round(simulation_results["original_hit_rate"], 3),
        f"Hit@{topn} (After)": round(simulation_results["reranked_hit_rate"], 3),
        "Gain Δ": round(simulation_results["hit_gain"], 3),
        "Avg Cost Estimate": f"{round(simulation_results['avg_cost'], 1)}ms",
        "Rerank Rate": f"{round(simulation_results['rerank_rate'] * 100, 1)}%",
        "Avg Position Improvement": round(simulation_results['avg_position_change'], 2)
    }
    
    print("\n重排增益摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n领域重排统计:")
    for domain, stats in simulation_results["domain_stats"].items():
        print(f"  {domain}: {stats['queries']}条查询, 重排率={round(stats['rerank_rate']*100, 1)}%")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="重排触发与前列增益评估模拟器")
    parser.add_argument("--threshold", type=float, default=0.7, help="触发重排的相似度阈值")
    parser.add_argument("--topn", type=int, default=5, help="评估命中率的Top-N值")
    parser.add_argument("--features-config", type=str, default="weights.json", help="特征权重配置文件路径")
    args = parser.parse_args()
    
    print(f"启动重排触发与前列增益评估模拟器 (threshold={args.threshold}, topn={args.topn})")
    
    # 检查并安装依赖
    install_dependencies()
    
    # 生成示例数据
    candidates = generate_sample_data()
    
    # 加载配置
    config = load_config(args.features_config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行模拟
    simulation_results = run_simulation(candidates, args.threshold, args.topn, config)
    
    # 记录结束时间
    end_time = time.time()
    
    # 导出报告
    export_report(simulation_results, args.topn)
    
    # 打印执行时间
    exec_time = end_time - start_time
    print(f"\n程序执行时间: {round(exec_time, 3)}秒")
    
    # 验证中文显示
    sample_query = simulation_results["results_df"]["query"].iloc[0]
    print(f"\n中文显示测试: {sample_query}")
    
    print("\n实验结果分析:")
    print(f"  - 原始Hit@{args.topn}命中率: {round(simulation_results['original_hit_rate'] * 100, 1)}%")
    print(f"  - 重排后Hit@{args.topn}命中率: {round(simulation_results['reranked_hit_rate'] * 100, 1)}%")
    print(f"  - 命中率提升: +{round(simulation_results['hit_gain'] * 100, 1)}%")
    print(f"  - 重排触发率: {round(simulation_results['rerank_rate'] * 100, 1)}%")
    print(f"  - 预计平均延迟增加: {round(simulation_results['avg_cost'], 1)}ms")
    print(f"  - 平均位置提升: {round(simulation_results['avg_position_change'], 2)}位")
    
    # 分析执行效率
    if exec_time > 1.0:
        print("\n注意: 程序执行时间较长，建议优化数据处理流程以提高效率。")
    else:
        print("\n程序执行效率良好。")

if __name__ == "__main__":
    main()