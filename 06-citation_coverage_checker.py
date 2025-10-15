#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
must-cover命中与引用三元校验

作者: Ph.D. Rhino

功能说明:
    校验回答是否覆盖 must-cover 要点，检查引用三元是否完整并可回放。

内容概述:
    该程序通过以下步骤进行校验：
    1. 读取回答文本、must-cover要点清单和引用列表
    2. 逐条判断“要点是否被回答命中且引用有效”
    3. 统计覆盖率与回放成功率
    4. 列出缺口清单

执行流程:
    1. 读取answers.csv（包含引用信息：source/fragment/offset/conf）
    2. 读取must_cover.csv（包含要点短语/正则表达式）
    3. 匹配要点→检查回答是否命中→验证对应引用存在且完整
    4. 统计覆盖率与回放率，输出缺失项和无效引用
    5. 生成“增检触发建议”列表

输入参数:
    --answers: 回答数据文件路径 (默认: answers.csv)
    --must: must-cover要点文件路径 (默认: must_cover.csv)

输出展示:
    Coverage=0.78 Replay=0.83
    Missing points: [MC-03, MC-07]
    Invalid citations: 12 rows (missing offset/conf<0.6)
    Actions saved: cover_gap_actions.csv
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import argparse
import re
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

# 生成示例回答数据
def generate_sample_answers(num_answers=30):
    """
    生成示例回答数据，包含引用信息
    
    参数:
        num_answers: 回答数量
    
    返回:
        包含回答文本和引用信息的DataFrame
    """
    print("正在生成示例回答数据...")
    
    # 示例回答模板
    answer_templates = [
        "人工智能（AI）是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。{citations}AI的应用领域广泛，包括自然语言处理、计算机视觉、机器学习等。",
        "气候变化是当前全球面临的重大挑战之一。{citations}温室气体排放导致全球平均气温上升，引发极端天气事件增加。",
        "区块链技术是一种分布式账本技术，具有去中心化、不可篡改等特点。{citations}它在金融、供应链管理等领域有广泛应用前景。",
        "量子计算利用量子力学原理进行计算，相比传统计算机在某些问题上具有指数级优势。{citations}量子比特是量子计算的基本单位。",
        "可持续发展目标（SDGs）是联合国制定的全球发展框架。{citations}它包括消除贫困、保护地球、促进繁荣等17项目标。"
    ]
    
    # 示例来源和片段
    sources = ["doc_tech_01.pdf", "doc_climate_02.txt", "doc_blockchain_03.md", "doc_quantum_04.pdf", "doc_sustain_05.docx"]
    
    data = []
    
    for i in range(num_answers):
        answer_id = f"answer_{i+1}"
        
        # 随机选择回答模板
        template = random.choice(answer_templates)
        
        # 生成引用信息
        num_citations = random.randint(1, 4)
        citations = []
        citation_texts = []
        
        for j in range(num_citations):
            source = random.choice(sources)
            fragment = f"{source}中的相关内容段落"
            offset = random.randint(1, 200)
            # 80%的概率生成有效置信度
            conf = random.uniform(0.6, 1.0) if random.random() < 0.8 else random.uniform(0.3, 0.59)
            
            # 90%的概率包含所有引用字段
            if random.random() < 0.9:
                citations.append({
                    "answer_id": answer_id,
                    "source": source,
                    "fragment": fragment,
                    "offset": offset,
                    "conf": round(conf, 2)
                })
            else:
                # 缺少某些字段的情况
                partial_citation = {"answer_id": answer_id, "source": source}
                if random.random() < 0.5: partial_citation["fragment"] = fragment
                if random.random() < 0.6: partial_citation["offset"] = offset
                if random.random() < 0.7: partial_citation["conf"] = round(conf, 2)
                citations.append(partial_citation)
            
            # 生成引用文本标记
            citation_texts.append(f"[引用{citations[-1]['source']}]")
        
        # 生成完整回答文本
        answer_text = template.format(citations=" ".join(citation_texts))
        
        # 添加主回答记录
        data.append({
            "answer_id": answer_id,
            "answer_text": answer_text,
            "source": None,
            "fragment": None,
            "offset": None,
            "conf": None
        })
        
        # 添加引用记录
        data.extend(citations)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    df.to_csv("answers.csv", index=False, encoding="utf-8-sig")
    print(f"示例回答数据已生成: answers.csv (共{len(df)}条记录)")
    
    return df

# 生成示例must-cover要点数据
def generate_sample_must_cover():
    """
    生成示例must-cover要点数据
    
    返回:
        包含要点ID、描述和匹配模式的DataFrame
    """
    print("正在生成示例must-cover要点数据...")
    
    # 示例must-cover要点
    must_cover_points = [
        {"mc_id": "MC-01", "description": "AI的定义和主要分支", "pattern": "人工智能.*(计算机科学|分支).*模拟人类智能"},
        {"mc_id": "MC-02", "description": "AI的应用领域", "pattern": "应用领域.*(自然语言处理|计算机视觉|机器学习)"},
        {"mc_id": "MC-03", "description": "气候变化的主要原因", "pattern": "气候变化.*(温室气体|排放|气温上升)"},
        {"mc_id": "MC-04", "description": "区块链的核心特点", "pattern": "区块链.*(分布式|账本|去中心化|不可篡改)"},
        {"mc_id": "MC-05", "description": "量子计算的基本原理", "pattern": "量子计算.*(量子力学|量子比特)"},
        {"mc_id": "MC-06", "description": "可持续发展目标的来源和数量", "pattern": "可持续发展目标.*(联合国|17项|十七项)"},
        {"mc_id": "MC-07", "description": "全球变暖的影响", "pattern": "全球变暖.*(极端天气|海平面上升|生态系统)"},
        {"mc_id": "MC-08", "description": "区块链的应用场景", "pattern": "区块链.*(金融|供应链|应用前景)"}
    ]
    
    # 转换为DataFrame
    df = pd.DataFrame(must_cover_points)
    
    # 保存数据
    df.to_csv("must_cover.csv", index=False, encoding="utf-8-sig")
    print(f"示例must-cover要点数据已生成: must_cover.csv (共{len(df)}条记录)")
    
    return df

# 读取回答数据
def load_answers(file_path):
    """
    读取回答数据
    
    参数:
        file_path: 回答数据文件路径
    
    返回:
        包含回答和引用信息的字典
    """
    if not os.path.exists(file_path):
        print(f"回答文件 {file_path} 不存在，将生成示例数据...")
        return generate_sample_answers()
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"已加载回答数据: {file_path} (共{len(df)}条记录)")
        return df
    except Exception as e:
        print(f"回答数据读取失败: {e}，将生成示例数据...")
        return generate_sample_answers()

# 读取must-cover要点数据
def load_must_cover(file_path):
    """
    读取must-cover要点数据
    
    参数:
        file_path: must-cover要点文件路径
    
    返回:
        must-cover要点DataFrame
    """
    if not os.path.exists(file_path):
        print(f"must-cover文件 {file_path} 不存在，将生成示例数据...")
        return generate_sample_must_cover()
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"已加载must-cover要点: {file_path} (共{len(df)}条记录)")
        return df
    except Exception as e:
        print(f"must-cover数据读取失败: {e}，将生成示例数据...")
        return generate_sample_must_cover()

# 检查回答是否覆盖must-cover要点
def check_coverage(answers_df, must_cover_df, min_confidence=0.6):
    """
    检查回答是否覆盖must-cover要点并验证引用有效性
    
    参数:
        answers_df: 回答数据
        must_cover_df: must-cover要点数据
        min_confidence: 引用有效最低置信度
    
    返回:
        包含覆盖检查结果的字典
    """
    print("开始检查must-cover覆盖情况...")
    
    # 分离主回答和引用记录
    main_answers = answers_df[answers_df["source"].isna()].copy()
    citations = answers_df[answers_df["source"].notna()].copy()
    
    # 初始化结果
    results = {
        "total_points": len(must_cover_df),
        "covered_points": 0,
        "replayable_points": 0,
        "missing_points": [],
        "invalid_citations": [],
        "point_details": [],
        "action_suggestions": []
    }
    
    # 按回答ID分组检查
    for _, answer_row in tqdm(main_answers.iterrows(), desc="检查回答", total=len(main_answers)):
        answer_id = answer_row["answer_id"]
        answer_text = answer_row["answer_text"]
        
        # 获取该回答的所有引用
        answer_citations = citations[citations["answer_id"] == answer_id].to_dict('records')
        
        # 检查每个must-cover要点
        for _, mc_row in must_cover_df.iterrows():
            mc_id = mc_row["mc_id"]
            mc_description = mc_row["description"]
            mc_pattern = mc_row["pattern"]
            
            # 检查回答是否命中要点
            is_hit = bool(re.search(mc_pattern, answer_text, re.IGNORECASE))
            
            # 检查是否有对应的有效引用
            has_valid_citation = False
            invalid_reasons = []
            
            if is_hit:
                # 检查引用是否存在且有效
                if not answer_citations:
                    invalid_reasons.append("无引用")
                else:
                    # 检查每个引用是否完整有效
                    for citation in answer_citations:
                        # 检查必要字段是否存在
                        missing_fields = []
                        for field in ["source", "fragment", "offset", "conf"]:
                            if field not in citation or pd.isna(citation[field]):
                                missing_fields.append(field)
                        
                        if missing_fields:
                            invalid_reasons.append(f"缺少字段: {', '.join(missing_fields)}")
                        elif citation["conf"] < min_confidence:
                            invalid_reasons.append(f"置信度低 ({citation['conf']} < {min_confidence})")
                        else:
                            has_valid_citation = True
                            break  # 只要有一个有效引用即可
            
            # 记录详细结果
            point_result = {
                "answer_id": answer_id,
                "mc_id": mc_id,
                "mc_description": mc_description,
                "is_hit": is_hit,
                "has_valid_citation": has_valid_citation,
                "invalid_reasons": invalid_reasons
            }
            results["point_details"].append(point_result)
            
            # 统计覆盖率
            if is_hit and has_valid_citation:
                results["replayable_points"] += 1
            
            # 如果该回答中没有命中任何must-cover要点，记录缺失
            if not is_hit:
                if mc_id not in results["missing_points"]:
                    results["missing_points"].append(mc_id)
    
    # 重新计算覆盖的要点总数 (不重复计算)
    covered_mc_ids = set()
    for detail in results["point_details"]:
        if detail["is_hit"]:
            covered_mc_ids.add(detail["mc_id"])
    results["covered_points"] = len(covered_mc_ids)
    
    # 计算覆盖率和回放率
    results["coverage_rate"] = results["covered_points"] / results["total_points"] if results["total_points"] > 0 else 0
    # 回放率 = 可回放的命中次数 / 所有命中次数
    total_hits = sum(1 for detail in results["point_details"] if detail["is_hit"])
    results["replay_rate"] = results["replayable_points"] / total_hits if total_hits > 0 else 0
    
    # 收集无效引用信息
    for detail in results["point_details"]:
        if detail["is_hit"] and not detail["has_valid_citation"] and detail["invalid_reasons"]:
            results["invalid_citations"].append({
                "answer_id": detail["answer_id"],
                "mc_id": detail["mc_id"],
                "reasons": detail["invalid_reasons"]
            })
    
    # 生成增检触发建议
    generate_action_suggestions(results, min_confidence)
    
    return results

# 生成增检触发建议
def generate_action_suggestions(results, min_confidence):
    """
    生成增检触发建议
    
    参数:
        results: 覆盖检查结果
        min_confidence: 引用有效最低置信度
    """
    # 按mc_id统计每个要点的命中情况
    mc_stats = {}
    for detail in results["point_details"]:
        mc_id = detail["mc_id"]
        if mc_id not in mc_stats:
            mc_stats[mc_id] = {
                "total_answers": 0,
                "hit_answers": 0,
                "replayable_answers": 0
            }
        
        mc_stats[mc_id]["total_answers"] += 1
        if detail["is_hit"]:
            mc_stats[mc_id]["hit_answers"] += 1
        if detail["is_hit"] and detail["has_valid_citation"]:
            mc_stats[mc_id]["replayable_answers"] += 1
    
    # 生成建议
    for mc_id, stats in mc_stats.items():
        # 计算该要点的命中率和回放率
        hit_rate = stats["hit_answers"] / stats["total_answers"]
        replay_rate = stats["replayable_answers"] / stats["hit_answers"] if stats["hit_answers"] > 0 else 0
        
        # 根据不同情况生成建议
        if hit_rate < 0.5:
            results["action_suggestions"].append({
                "type": "ADD_SOURCE",
                "priority": "HIGH",
                "mc_id": mc_id,
                "reason": f"该要点命中率低 ({hit_rate:.2f})，建议增加相关源文档"
            })
        elif replay_rate < 0.7:
            results["action_suggestions"].append({
                "type": "IMPROVE_CITATION",
                "priority": "MEDIUM",
                "mc_id": mc_id,
                "reason": f"该要点引用质量差 ({replay_rate:.2f})，建议优化引用生成"
            })
        
    # 检查整体情况
    if results["coverage_rate"] < 0.7:
        results["action_suggestions"].append({
            "type": "REVIEW_PROMPT",
            "priority": "HIGH",
            "reason": f"整体覆盖率低 ({results['coverage_rate']:.2f})，建议优化提示词工程"
        })
    
    if results["replay_rate"] < 0.7:
        results["action_suggestions"].append({
            "type": "ENHANCE_CITATION_ENGINE",
            "priority": "MEDIUM",
            "reason": f"整体引用回放率低 ({results['replay_rate']:.2f})，建议增强引用生成引擎"
        })

# 导出检查报告
def export_report(results):
    """
    导出覆盖检查报告
    
    参数:
        results: 覆盖检查结果
    """
    # 准备导出数据
    action_df = pd.DataFrame(results["action_suggestions"])
    
    # 保存报告
    report_path = "cover_gap_actions.csv"
    action_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    
    print(f"增检触发建议报告已导出: {report_path}")
    
    # 生成摘要报告
    print("\n校验结果摘要:")
    print(f"  Coverage={results['coverage_rate']:.2f} Replay={results['replay_rate']:.2f}")
    
    if results["missing_points"]:
        print(f"  Missing points: [{', '.join(results['missing_points'])}]")
    else:
        print("  Missing points: None")
    
    invalid_count = len(results["invalid_citations"])
    print(f"  Invalid citations: {invalid_count} rows")
    
    # 统计无效原因分布
    reason_counts = {}
    for cite in results["invalid_citations"]:
        for reason in cite["reasons"]:
            if reason.startswith("缺少字段"):
                reason_counts["missing fields"] = reason_counts.get("missing fields", 0) + 1
            elif reason.startswith("置信度低"):
                reason_counts["low confidence"] = reason_counts.get("low confidence", 0) + 1
            elif reason == "无引用":
                reason_counts["no citation"] = reason_counts.get("no citation", 0) + 1
    
    if reason_counts:
        reason_text = ", ".join([f"{k}: {v}" for k, v in reason_counts.items()])
        print(f"  Invalid reasons: {reason_text}")
    
    print("\n增检触发建议:")
    for action in results["action_suggestions"]:
        priority_text = "[高]" if action["priority"] == "HIGH" else "[中]" if action["priority"] == "MEDIUM" else "[低]"
        print(f"  {priority_text} {action['type']}: {action['reason']}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="must-cover命中与引用三元校验")
    parser.add_argument("--answers", type=str, default="answers.csv", help="回答数据文件路径")
    parser.add_argument("--must", type=str, default="must_cover.csv", help="must-cover要点文件路径")
    args = parser.parse_args()
    
    print(f"启动must-cover命中与引用三元校验")
    
    # 检查并安装依赖
    install_dependencies()
    
    # 加载数据
    answers_df = load_answers(args.answers)
    must_cover_df = load_must_cover(args.must)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行检查
    results = check_coverage(answers_df, must_cover_df)
    
    # 记录结束时间
    end_time = time.time()
    
    # 导出报告
    export_report(results)
    
    # 打印执行时间
    exec_time = end_time - start_time
    print(f"\n程序执行时间: {round(exec_time, 3)}秒")
    
    # 验证中文显示
    sample_answer = answers_df[answers_df["source"].isna()]["answer_text"].iloc[0] if not answers_df.empty else "中文测试"
    print(f"\n中文显示测试: {sample_answer[:50]}...")
    
    print("\n实验结果分析:")
    print(f"  - 总must-cover要点数量: {results['total_points']}")
    print(f"  - 命中的要点数量: {results['covered_points']}")
    print(f"  - 可回放的要点数量: {results['replayable_points']}")
    print(f"  - 覆盖率: {round(results['coverage_rate'] * 100, 1)}%")
    print(f"  - 回放率: {round(results['replay_rate'] * 100, 1)}%")
    print(f"  - 缺失要点数量: {len(results['missing_points'])}")
    print(f"  - 无效引用数量: {len(results['invalid_citations'])}")
    print(f"  - 生成建议数量: {len(results['action_suggestions'])}")
    
    # 分析执行效率
    if exec_time > 1.0:
        print("\n注意: 程序执行时间较长，建议优化数据处理流程以提高效率。")
    else:
        print("\n程序执行效率良好。")

if __name__ == "__main__":
    main()