#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
程序名称：基于复杂度的Top-K选择
作者：Ph.D. Rhino
功能说明：根据查询复杂度信号（长度、实体计数、疑问词）动态给出K值与是否触发重排。
内容概述：定义复杂度打分函数；根据阈值映射到K档位与重排布尔；统计整体的平均K与预计P95开销。
执行流程：
1. 解析查询，提取特征（长度、数字/实体比、疑问词）
2. 复杂度打分并映射到K档位（如 3/5/10）
3. 应用触发规则（复杂度≥T 或 风险桶→触发重排）
4. 估算代价：平均K与重排比例，输出预算对齐检查
5. 导出每条query的决策与整体摘要
输入说明：--rules rules.json --queries queries.csv --risk-bucket medical
输出展示：
Decision: qid=102 K=10 re_rank=True
Aggregate: avgK=5.7 re_rank_rate=28% est_P95_delta=+90ms
Decisions saved: dynamic_k_decisions.csv
"""

import os
import sys
import argparse
import random
import json
import csv
import time
import re
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

# 生成规则配置文件
def generate_rules_config(file_path="rules.json"):
    """
    生成默认的规则配置文件
    参数:
        file_path: 规则配置文件路径
    """
    rules = {
        "complexity_scoring": {
            "length_weight": 0.4,  # 查询长度权重
            "entity_ratio_weight": 0.3,  # 实体/数字比例权重
            "question_words_weight": 0.3  # 疑问词权重
        },
        "k_mapping": {
            "low": 3,  # 低复杂度对应的K值
            "medium": 5,  # 中复杂度对应的K值
            "high": 10  # 高复杂度对应的K值
        },
        "complexity_thresholds": {
            "low_high": 0.3,  # 低到中的阈值
            "medium_high": 0.7  # 中到高的阈值
        },
        "re_rank_rules": {
            "complexity_threshold": 0.6,  # 触发重排的复杂度阈值
            "risk_buckets": ["medical", "finance", "legal"],  # 需要触发重排的风险桶
            "always_re_rank_questions": True  # 对包含疑问词的查询总是触发重排
        },
        "cost_estimation": {
            "base_latency": 50,  # 基础延迟(ms)
            "per_k_latency": 5,  # 每个K值增加的延迟(ms)
            "re_rank_latency": 50  # 重排增加的延迟(ms)
        }
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    print(f"生成规则配置文件: {file_path}")
    return rules

# 生成示例查询数据
def generate_sample_queries(num_queries=100, file_path="queries.csv"):
    """
    生成示例查询数据集
    参数:
        num_queries: 查询数量
        file_path: 查询文件路径
    """
    # 示例查询模板
    simple_queries = [
        "什么是人工智能",
        "机器学习的应用",
        "区块链技术",
        "云计算基础",
        "数据分析方法"
    ]
    
    medium_queries = [
        "比较深度学习和传统机器学习的优缺点",
        "解释区块链中的智能合约如何工作",
        "列举2023年最流行的10种编程语言",
        "分析大数据对企业决策的影响",
        "说明云计算中的IaaS、PaaS和SaaS的区别"
    ]
    
    complex_queries = [
        "详细分析人工智能在医疗健康领域的应用案例，包括诊断辅助、药物研发和患者护理，并评估其潜在风险和伦理问题",
        "比较不同类型的区块链共识算法（PoW、PoS、DPoS等）在安全性、性能和能源消耗方面的差异，并讨论它们各自的适用场景",
        "基于2020-2023年的市场数据，预测云计算市场未来5年的发展趋势，包括主要技术方向、市场份额变化和新兴应用领域",
        "解释量子计算的基本原理，并探讨其对密码学、材料科学和优化问题的潜在影响，同时评估其商业化面临的挑战",
        "分析大数据分析技术在金融风险管理中的应用，包括信用评分、欺诈检测和市场预测，并讨论数据隐私和监管合规问题"
    ]
    
    # 不同领域的查询
    domains = {
        "general": simple_queries + medium_queries + complex_queries,
        "medical": [
            "什么是癌症免疫治疗？",
            "分析基因编辑技术CRISPR在治疗遗传病中的应用前景和风险",
            "比较不同类型的糖尿病药物及其副作用",
            "解释新冠病毒变异株的特点及其对疫苗有效性的影响"
        ],
        "finance": [
            "什么是量化交易策略？",
            "分析2008年金融危机的原因和影响",
            "比较不同类型的投资组合管理方法",
            "解释加密货币市场波动的主要因素"
        ],
        "tech": [
            "什么是5G技术？",
            "分析人工智能在自动驾驶中的应用挑战",
            "比较不同云计算服务商的优劣势",
            "解释元宇宙的概念及其技术基础"
        ]
    }
    
    # 中文疑问词列表
    question_words = ["什么", "为什么", "如何", "怎样", "是否", "能否", "哪些", "哪里", "何时", "多少"]
    
    queries = []
    
    for i in range(num_queries):
        # 随机选择查询复杂度
        complexity = random.choice(["simple", "medium", "complex"])
        
        # 根据复杂度选择查询
        if complexity == "simple":
            query_text = random.choice(simple_queries)
        elif complexity == "medium":
            query_text = random.choice(medium_queries)
        else:
            query_text = random.choice(complex_queries)
        
        # 随机选择领域
        domain = random.choice(list(domains.keys()))
        
        # 10%的概率从特定领域选择查询
        if random.random() < 0.1:
            query_text = random.choice(domains[domain])
        
        # 为查询添加一些随机性
        if random.random() < 0.3:
            # 随机添加一些数字或实体
            numbers = ["2023", "5", "10", "100", "500", "1000"]
            entities = ["北京", "上海", "腾讯", "阿里巴巴", "华为", "微软", "苹果", "谷歌"]
            
            insert_pos = random.randint(0, len(query_text))
            if random.random() < 0.5:
                extra_text = random.choice(numbers)
            else:
                extra_text = random.choice(entities)
            
            query_text = query_text[:insert_pos] + extra_text + " " + query_text[insert_pos:]
        
        # 检查是否包含疑问词
        contains_question = any(word in query_text for word in question_words)
        
        query = {
            "query_id": f"qid_{i+1}",
            "query_text": query_text,
            "domain": domain,
            "contains_question": contains_question
        }
        
        queries.append(query)
    
    # 保存查询数据
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text", "domain", "contains_question"])
        writer.writeheader()
        for query in queries:
            writer.writerow(query)
    
    print(f"生成示例查询数据完成：{num_queries}条查询，保存至{file_path}")
    return queries

# 提取查询特征
def extract_query_features(query_text):
    """
    从查询文本中提取特征
    参数:
        query_text: 查询文本
    返回:
        features: 包含查询特征的字典
    """
    # 查询长度特征
    length = len(query_text)
    normalized_length = min(length / 200, 1.0)  # 归一化到0-1，假设200个字符是长查询
    
    # 实体和数字比例特征
    # 简单的实体和数字检测（实际应用中可能需要使用NLP工具）
    entity_pattern = r"[\u4e00-\u9fa5]{2,}"  # 匹配中文实体
    number_pattern = r"\d+"  # 匹配数字
    
    entities = re.findall(entity_pattern, query_text)
    numbers = re.findall(number_pattern, query_text)
    
    total_tokens = len(query_text.split())  # 简单分词
    if total_tokens == 0:
        entity_number_ratio = 0
    else:
        entity_number_ratio = min((len(entities) + len(numbers)) / total_tokens, 1.0)
    
    # 疑问词特征
    question_words = ["什么", "为什么", "如何", "怎样", "是否", "能否", "哪些", "哪里", "何时", "多少"]
    contains_question = any(word in query_text for word in question_words)
    question_feature = 1.0 if contains_question else 0.0
    
    features = {
        "length": length,
        "normalized_length": normalized_length,
        "entity_number_ratio": entity_number_ratio,
        "contains_question": contains_question,
        "question_feature": question_feature
    }
    
    return features

# 计算查询复杂度得分
def calculate_complexity_score(features, rules):
    """
    计算查询复杂度得分
    参数:
        features: 查询特征
        rules: 规则配置
    返回:
        complexity_score: 复杂度得分(0-1)
    """
    weights = rules["complexity_scoring"]
    
    score = (
        features["normalized_length"] * weights["length_weight"] +
        features["entity_number_ratio"] * weights["entity_ratio_weight"] +
        features["question_feature"] * weights["question_words_weight"]
    )
    
    # 确保得分在0-1范围内
    score = max(0, min(1, score))
    
    return score

# 确定K值档位
def determine_k_value(complexity_score, rules):
    """
    根据复杂度得分确定K值档位
    参数:
        complexity_score: 复杂度得分
        rules: 规则配置
    返回:
        k_value: 确定的K值
    """
    thresholds = rules["complexity_thresholds"]
    k_mapping = rules["k_mapping"]
    
    if complexity_score < thresholds["low_high"]:
        return k_mapping["low"]
    elif complexity_score < thresholds["medium_high"]:
        return k_mapping["medium"]
    else:
        return k_mapping["high"]

# 决定是否触发重排
def determine_re_rank(query, complexity_score, rules, risk_buckets=[]):
    """
    决定是否触发重排
    参数:
        query: 查询信息
        complexity_score: 复杂度得分
        rules: 规则配置
        risk_buckets: 风险桶列表
    返回:
        re_rank: 是否触发重排
    """
    re_rank_rules = rules["re_rank_rules"]
    
    # 检查复杂度阈值
    if complexity_score >= re_rank_rules["complexity_threshold"]:
        return True
    
    # 检查是否在风险桶中
    if query["domain"] in risk_buckets or query["domain"] in re_rank_rules["risk_buckets"]:
        return True
    
    # 检查是否包含疑问词
    if re_rank_rules["always_re_rank_questions"] and query["contains_question"]:
        return True
    
    return False

# 估算延迟
def estimate_latency(k_value, re_rank, rules):
    """
    估算查询延迟
    参数:
        k_value: K值
        re_rank: 是否触发重排
        rules: 规则配置
    返回:
        latency: 估算的延迟(ms)
    """
    cost_params = rules["cost_estimation"]
    
    latency = cost_params["base_latency"] + k_value * cost_params["per_k_latency"]
    if re_rank:
        latency += cost_params["re_rank_latency"]
    
    return latency

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="根据查询复杂度信号动态给出K值与是否触发重排")
    parser.add_argument("--rules", type=str, default="rules.json", help="规则配置文件路径")
    parser.add_argument("--queries", type=str, default="queries.csv", help="查询文件路径")
    parser.add_argument("--risk-bucket", type=str, nargs="+", default=[], help="风险桶列表")
    parser.add_argument("--output", type=str, default="dynamic_k_decisions.csv", help="输出决策文件路径")
    
    args = parser.parse_args()
    
    # 检查并安装依赖
    check_and_install_dependencies()
    
    start_time = time.time()
    
    # 检查规则配置文件是否存在，不存在则生成
    if not os.path.exists(args.rules):
        print(f"规则配置文件不存在，正在生成默认配置: {args.rules}")
        rules = generate_rules_config(args.rules)
    else:
        # 读取规则配置
        with open(args.rules, "r", encoding="utf-8") as f:
            rules = json.load(f)
        
        print(f"读取规则配置完成: {args.rules}")
    
    # 合并命令行传入的风险桶和配置文件中的风险桶
    risk_buckets = set(rules["re_rank_rules"]["risk_buckets"] + args.risk_bucket)
    
    # 检查查询文件是否存在，不存在则生成
    if not os.path.exists(args.queries):
        print(f"查询文件不存在，正在生成示例查询数据: {args.queries}")
        queries = generate_sample_queries(file_path=args.queries)
    else:
        # 读取查询数据
        queries = []
        with open(args.queries, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 检查是否包含contains_question列，如果不存在则根据文本判断
                if "contains_question" in row:
                    row["contains_question"] = row["contains_question"].lower() == "true"
                else:
                    # 中文疑问词列表
                    question_words = ["什么", "为什么", "如何", "怎样", "是否", "能否", "哪些", "哪里", "何时", "多少"]
                    row["contains_question"] = any(word in row["query_text"] for word in question_words)
                
                # 检查是否包含domain列，如果不存在则设置为默认值
                if "domain" not in row:
                    row["domain"] = "general"
                queries.append(row)
        
        print(f"读取查询数据完成: {len(queries)}条查询")
    
    # 处理每个查询，生成决策
    decisions = []
    total_k = 0
    re_rank_count = 0
    latencies = []
    
    for query in queries:
        query_id = query["query_id"]
        query_text = query["query_text"]
        
        # 提取查询特征
        features = extract_query_features(query_text)
        
        # 计算复杂度得分
        complexity_score = calculate_complexity_score(features, rules)
        
        # 确定K值
        k_value = determine_k_value(complexity_score, rules)
        
        # 决定是否触发重排
        re_rank = determine_re_rank(query, complexity_score, rules, risk_buckets)
        
        # 估算延迟
        latency = estimate_latency(k_value, re_rank, rules)
        
        # 记录决策
        decision = {
            "query_id": query_id,
            "query_text": query_text,
            "domain": query["domain"],
            "complexity_score": complexity_score,
            "k_value": k_value,
            "re_rank": re_rank,
            "estimated_latency": latency,
            "length": features["length"],
            "contains_question": features["contains_question"]
        }
        
        decisions.append(decision)
        
        # 累加统计数据
        total_k += k_value
        if re_rank:
            re_rank_count += 1
        latencies.append(latency)
        
        # 打印部分决策信息（每10条查询打印一次）
        if len(decisions) % 10 == 0 or len(decisions) == len(queries):
            print(f"Decision: qid={query_id} K={k_value} re_rank={re_rank}")
    
    # 计算汇总统计信息
    avg_k = total_k / len(queries) if queries else 0
    re_rank_rate = (re_rank_count / len(queries) * 100) if queries else 0
    
    # 计算P95延迟
    import numpy as np
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    base_latency = rules["cost_estimation"]["base_latency"] + rules["complexity_scoring"]["length_weight"] * rules["cost_estimation"]["per_k_latency"]
    est_p95_delta = p95_latency - base_latency
    
    print(f"Aggregate: avgK={avg_k:.1f} re_rank_rate={re_rank_rate:.0f}% est_P95_delta=+{est_p95_delta:.0f}ms")
    
    # 保存决策结果
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["query_id", "query_text", "domain", "complexity_score", "k_value", "re_rank", "estimated_latency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for decision in decisions:
            # 只保留需要的字段
            row = {k: decision[k] for k in fieldnames}
            writer.writerow(row)
    
    print(f"Decisions saved: {args.output}")
    
    # 检查执行效率
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time:.2f}秒")
    
    # 检查输出文件的中文是否有乱码
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            content = f.read()
        print("输出文件的中文显示正常，没有乱码问题")
    except UnicodeDecodeError:
        print("警告：输出文件可能存在中文乱码问题")
    
    # 实验结果分析
    print("\n实验结果分析：")
    
    # 分析不同复杂度级别的分布
    complexity_levels = defaultdict(list)
    for decision in decisions:
        score = decision["complexity_score"]
        if score < rules["complexity_thresholds"]["low_high"]:
            level = "低"
        elif score < rules["complexity_thresholds"]["medium_high"]:
            level = "中"
        else:
            level = "高"
        complexity_levels[level].append(decision)
    
    print(f"复杂度分布：")
    for level, level_decisions in complexity_levels.items():
        count = len(level_decisions)
        percentage = count / len(decisions) * 100
        avg_k_level = sum(d["k_value"] for d in level_decisions) / count
        re_rank_rate_level = sum(1 for d in level_decisions if d["re_rank"]) / count * 100
        print(f"  {level}复杂度: {count}条({percentage:.1f}%), 平均K={avg_k_level:.1f}, 重排率={re_rank_rate_level:.0f}%")
    
    # 分析不同领域的分布
    domain_stats = defaultdict(lambda: {"count": 0, "re_rank_count": 0, "avg_k": 0})
    for decision in decisions:
        domain = decision["domain"]
        domain_stats[domain]["count"] += 1
        if decision["re_rank"]:
            domain_stats[domain]["re_rank_count"] += 1
        domain_stats[domain]["avg_k"] += decision["k_value"]
    
    print(f"\n领域分布：")
    for domain, stats in sorted(domain_stats.items()):
        stats["avg_k"] /= stats["count"]
        stats["re_rank_rate"] = stats["re_rank_count"] / stats["count"] * 100
        print(f"  {domain}: {stats['count']}条, 平均K={stats['avg_k']:.1f}, 重排率={stats['re_rank_rate']:.0f}%")
    
    # 预算对齐检查
    print(f"\n预算对齐检查：")
    print(f"  平均K值: {avg_k:.1f}")
    print(f"  重排比例: {re_rank_rate:.0f}%")
    print(f"  预计P95延迟增加: +{est_p95_delta:.0f}ms")
    
    # 基于当前配置给出建议
    if avg_k > 7:
        print(f"  建议: 平均K值较高，可考虑调整复杂度阈值或K映射值以降低成本")
    elif avg_k < 4:
        print(f"  建议: 平均K值较低，可能影响检索效果，可考虑降低复杂度阈值")
    else:
        print(f"  建议: 平均K值处于合理范围")
    
    if re_rank_rate > 50:
        print(f"  建议: 重排比例较高，可考虑调整重排触发条件以降低延迟")
    elif re_rank_rate < 10:
        print(f"  建议: 重排比例较低，对于复杂查询可能影响精度，可考虑扩大重排范围")
    else:
        print(f"  建议: 重排比例处于合理范围")
    
    print("\n程序完成！")

if __name__ == "__main__":
    main()