#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
规则化一致性与矛盾率估计

作者: Ph.D. Rhino

功能说明:
    用规则/字典近似检测“内部矛盾”“与证据矛盾”，估算矛盾率。

内容概述:
    该程序通过以下步骤进行一致性检测：
    1. 定义一组冲突词典与数值/单位对齐规则
    2. 对回答段落两两比对，检测否定冲突与数值口径冲突
    3. 与引用内容对照，输出矛盾样例
    4. 计算矛盾率并给出修订建议

执行流程:
    1. 载入反义/否定词表、单位换算映射
    2. 对回答句子两两比对，检测否定与数值冲突
    3. 与引用片段进行关键词与数字对齐校验
    4. 汇总矛盾条数/总句对，计算矛盾率
    5. 导出矛盾样例与修订建议

输入说明:
    --answers: 回答数据文件路径 (默认: answers.csv)
    --evidence: 证据数据文件路径 (默认: evidence.csv)

输出展示:
    NLI-approx contradiction_rate=2.8%
    Conflicts saved: contradictions_examples.csv
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
import nltk
from nltk.tokenize import sent_tokenize

# 自动安装依赖
def install_dependencies():
    """检查并安装必要的依赖包"""
    try:
        import pandas
        import numpy
        import tqdm
        import nltk
    except ImportError:
        print("正在安装必要的依赖包...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "tqdm", "nltk"])
            print("依赖包安装成功！")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            sys.exit(1)
    
    # 下载NLTK必要的资源
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载NLTK必要的分词资源...")
        nltk.download('punkt')

# 生成示例回答数据
def generate_sample_answers(num_answers=20):
    """
    生成示例回答数据，包含可能的内部矛盾
    
    参数:
        num_answers: 回答数量
    
    返回:
        包含回答ID和文本的DataFrame
    """
    print("正在生成示例回答数据...")
    
    # 示例回答模板（包含一些故意的矛盾）
    answer_templates = [
        "人工智能（AI）的发展速度非常快，近年来取得了突破性进展。然而，AI技术的发展速度实际上非常缓慢，进步有限。",
        "全球气候变化的显著性，主要由人类活动引起。温室气体排放是气候变化的主要原因。",
        "健康饮食应该包含丰富的水果和蔬菜，每天建议摄入5种以上。然而，专家建议每天只需要摄入2种水果和蔬菜就足够了。",
        "区块链技术具有去中心化的特点，没有中央控制机构。但实际上，大多数区块链系统都有核心开发团队进行维护和控制。",
        "学习编程需要长期坚持，没有捷径可走。不过，通过参加速成课程，你可以在短短几周内掌握编程技能。",
        "充足的睡眠对健康至关重要，成年人每天需要7-9小时的睡眠。然而，研究表明成年人每天只需要5小时睡眠就足够了。",
        "电动汽车是未来的发展趋势，具有环保、低噪音等优点。但是，电动汽车的电池寿命短、充电时间长，不适合长途旅行。",
        "阅读纸质书籍可以提高阅读效率和理解力，比电子阅读更有优势。不过，最新研究表明电子阅读和纸质阅读的效果没有显著差异。"
    ]
    
    data = []
    
    for i in range(num_answers):
        answer_id = f"answer_{i+1}"
        
        # 随机选择回答模板
        if random.random() < 0.3:  # 30%的概率生成包含矛盾的回答
            template = random.choice(answer_templates)
        else:
            # 生成不包含矛盾的回答
            template_parts = random.choice(answer_templates).split("然而，")
            template = template_parts[0]
        
        data.append({
            "answer_id": answer_id,
            "answer_text": template
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    df.to_csv("answers.csv", index=False, encoding="utf-8-sig")
    print(f"示例回答数据已生成: answers.csv (共{len(df)}条记录)")
    
    return df

# 生成示例证据数据
def generate_sample_evidence(answers_df):
    """
    生成示例证据数据，包含与回答相关的证据片段
    
    参数:
        answers_df: 回答数据
    
    返回:
        包含证据ID、内容和相关回答ID的DataFrame
    """
    print("正在生成示例证据数据...")
    
    # 示例证据片段
    evidence_templates = [
        "人工智能的发展速度近年来显著加快，特别是在自然语言处理和计算机视觉领域。",
        "全球气候变化的显著性，主要由人类活动引起。温室气体排放是气候变化的主要原因。",
        "世界卫生组织建议成年人每天摄入至少5种不同的水果和蔬菜。",
        "区块链技术的核心特点是去中心化、透明和不可篡改。",
        "学习编程需要持续的实践和时间投入，没有快速掌握的捷径。",
        "美国国家睡眠基金会建议成年人每天睡眠7-9小时以保持健康。",
        "电动汽车相比传统燃油车具有更低的碳排放和运营成本。",
        "研究表明纸质阅读和电子阅读在理解和记忆效果上没有显著差异。"
    ]
    
    data = []
    evidence_id = 1
    
    for _, answer_row in answers_df.iterrows():
        answer_id = answer_row["answer_id"]
        
        # 每个回答关联1-3个证据
        num_evidences = random.randint(1, 3)
        
        for _ in range(num_evidences):
            # 80%的概率生成相关证据，20%的概率生成矛盾证据
            if random.random() < 0.8:
                evidence_text = random.choice(evidence_templates)
            else:
                # 生成与某些回答矛盾的证据
                if random.random() < 0.5:
                    evidence_text = "人工智能的发展速度近年来并没有明显加快，进步有限。"
                else:
                    evidence_text = "成年人每天只需要5-6小时的睡眠就足够保持健康。"
            
            data.append({
                "evidence_id": f"evidence_{evidence_id}",
                "evidence_text": evidence_text,
                "answer_id": answer_id
            })
            evidence_id += 1
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据
    df.to_csv("evidence.csv", index=False, encoding="utf-8-sig")
    print(f"示例证据数据已生成: evidence.csv (共{len(df)}条记录)")
    
    return df

# 读取回答数据
def load_answers(file_path):
    """
    读取回答数据
    
    参数:
        file_path: 回答数据文件路径
    
    返回:
        回答数据DataFrame
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

# 读取证据数据
def load_evidence(file_path, answers_df):
    """
    读取证据数据
    
    参数:
        file_path: 证据数据文件路径
        answers_df: 回答数据，用于生成示例证据
    
    返回:
        证据数据DataFrame
    """
    if not os.path.exists(file_path):
        print(f"证据文件 {file_path} 不存在，将生成示例数据...")
        return generate_sample_evidence(answers_df)
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"已加载证据数据: {file_path} (共{len(df)}条记录)")
        return df
    except Exception as e:
        print(f"证据数据读取失败: {e}，将生成示例数据...")
        return generate_sample_evidence(answers_df)

# 初始化矛盾检测规则
def init_conflict_rules():
    """
    初始化矛盾检测规则和词典
    
    返回:
        包含反义词典、否定词表和单位换算映射的字典
    """
    # 反义词典（部分示例）
    antonym_dict = {
        "快": ["慢", "缓慢"],
        "慢": ["快", "迅速"],
        "多": ["少", "有限"],
        "少": ["多", "丰富"],
        "高": ["低", "矮小"],
        "低": ["高", "高大"],
        "增加": ["减少", "降低"],
        "减少": ["增加", "提高"],
        "重要": ["不重要", "无关紧要"],
        "必要": ["不必要", "不需要"],
        "有效": ["无效", "没有效果"],
        "有限": ["无限", "丰富"],
        "足够": ["不足", "不够"],
        "显著": ["不显著", "微小"],
        "主要": ["次要", "无关"],
        "必须": ["不必", "无需"]
    }
    
    # 否定词表
    negation_words = ["不", "非", "无", "没", "未", "否", "不是", "没有", "不应该", "不可能", "不会", "不需要"]
    
    # 单位换算映射（部分示例）
    unit_conversion = {
        "小时": {"分钟": 60, "秒": 3600},
        "分钟": {"小时": 1/60, "秒": 60},
        "秒": {"小时": 1/3600, "分钟": 1/60},
        "天": {"小时": 24, "周": 1/7},
        "周": {"天": 7, "月": 1/4},
        "月": {"周": 4, "年": 1/12},
        "年": {"月": 12, "天": 365},
        "千克": {"克": 1000, "吨": 1/1000},
        "克": {"千克": 1/1000, "吨": 1/1000000},
        "吨": {"千克": 1000, "克": 1000000},
        "米": {"厘米": 100, "毫米": 1000},
        "厘米": {"米": 1/100, "毫米": 10},
        "毫米": {"米": 1/1000, "厘米": 1/10}
    }
    
    return {
        "antonyms": antonym_dict,
        "negations": negation_words,
        "units": unit_conversion
    }

# 检测句子中的数字和单位
def extract_numbers_with_units(text):
    """
    从文本中提取数字和相关单位
    
    参数:
        text: 输入文本
    
    返回:
        包含数字和单位的列表
    """
    # 匹配数字（整数、小数）和可能的单位
    number_pattern = r'(\d+\.?\d*)\s*([年月日时分秒千克克吨米厘米毫米]+)'  # 简化版，实际应用需扩展
    matches = re.findall(number_pattern, text)
    
    results = []
    for match in matches:
        number = float(match[0])
        unit = match[1]
        results.append({"number": number, "unit": unit})
    
    return results

# 检测两个句子之间的矛盾
def detect_sentence_conflict(sent1, sent2, rules):
    """
    检测两个句子之间是否存在矛盾
    
    参数:
        sent1: 第一个句子
        sent2: 第二个句子
        rules: 矛盾检测规则
    
    返回:
        矛盾检测结果（包含是否矛盾、矛盾类型和矛盾内容）
    """
    # 初始化结果
    result = {
        "has_conflict": False,
        "conflict_type": None,
        "conflict_details": None
    }
    
    # 1. 检测否定词矛盾
    for neg_word in rules["negations"]:
        if neg_word in sent1 and neg_word in sent2:
            continue  # 两个句子都有否定词，暂时无法判断
        
        # 检查是否存在否定词+反义词的情况
        for word, antonyms in rules["antonyms"].items():
            for antonym in antonyms:
                if (neg_word in sent1 and antonym in sent1 and word in sent2) or \
                   (neg_word in sent2 and antonym in sent2 and word in sent1) or \
                   (word in sent1 and antonym in sent2) or (antonym in sent1 and word in sent2):
                    result["has_conflict"] = True
                    result["conflict_type"] = "antonym_conflict"
                    result["conflict_details"] = f"检测到反义词矛盾: '{word}' vs '{antonym}'"
                    return result
    
    # 2. 检测数字和单位矛盾
    numbers1 = extract_numbers_with_units(sent1)
    numbers2 = extract_numbers_with_units(sent2)
    
    if numbers1 and numbers2:
        # 简化版检测：检查是否有相同或可换算的单位，且数值差异较大
        for num1 in numbers1:
            for num2 in numbers2:
                if num1["unit"] == num2["unit"]:
                    # 相同单位，检查数值是否差异超过50%
                    ratio = max(num1["number"], num2["number"]) / min(num1["number"], num2["number"])
                    if ratio > 2:  # 差异超过2倍
                        result["has_conflict"] = True
                        result["conflict_type"] = "numeric_conflict"
                        result["conflict_details"] = f"检测到数值矛盾: {num1['number']}{num1['unit']} vs {num2['number']}{num2['unit']}"
                        return result
                elif num1["unit"] in rules["units"] and num2["unit"] in rules["units"][num1["unit"]]:
                    # 可换算单位，进行换算后比较
                    conversion_factor = rules["units"][num1["unit"]][num2["unit"]]
                    converted_num1 = num1["number"] * conversion_factor
                    ratio = max(converted_num1, num2["number"]) / min(converted_num1, num2["number"])
                    if ratio > 2:  # 差异超过2倍
                        result["has_conflict"] = True
                        result["conflict_type"] = "numeric_conflict"
                        result["conflict_details"] = f"检测到数值矛盾: {num1['number']}{num1['unit']} ({converted_num1:.2f}{num2['unit']}) vs {num2['number']}{num2['unit']}"
                        return result
    
    return result

# 检测回答内部矛盾
def detect_internal_conflicts(answer_text, rules):
    """
    检测回答文本内部是否存在矛盾
    
    参数:
        answer_text: 回答文本
        rules: 矛盾检测规则
    
    返回:
        内部矛盾检测结果列表
    """
    conflicts = []
    
    # 确保answer_text是字符串类型
    if not isinstance(answer_text, str):
        answer_text = str(answer_text) if answer_text is not None else ""
    
    # 使用NLTK进行句子分割
    sentences = sent_tokenize(answer_text)
    
    # 如果句子数量少于2，不可能有内部矛盾
    if len(sentences) < 2:
        return conflicts
    
    # 对句子进行两两比对
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sent1 = sentences[i]
            sent2 = sentences[j]
            
            # 检测矛盾
            conflict_result = detect_sentence_conflict(sent1, sent2, rules)
            
            if conflict_result["has_conflict"]:
                conflicts.append({
                    "type": "internal",
                    "sentence1": sent1,
                    "sentence2": sent2,
                    "conflict_type": conflict_result["conflict_type"],
                    "details": conflict_result["conflict_details"]
                })
    
    return conflicts

# 检测回答与证据之间的矛盾
def detect_evidence_conflicts(answer_text, evidence_texts, rules):
    """
    检测回答与证据之间是否存在矛盾
    
    参数:
        answer_text: 回答文本
        evidence_texts: 证据文本列表
        rules: 矛盾检测规则
    
    返回:
        与证据矛盾的检测结果列表
    """
    conflicts = []
    
    # 确保answer_text是字符串类型
    if not isinstance(answer_text, str):
        answer_text = str(answer_text) if answer_text is not None else ""
    
    # 分割回答句子
    answer_sentences = sent_tokenize(answer_text)
    
    # 对每个证据进行比对
    for evidence_id, evidence_text in evidence_texts.items():
        # 确保evidence_text是字符串类型
        if not isinstance(evidence_text, str):
            evidence_text = str(evidence_text) if evidence_text is not None else ""
        
        # 分割证据句子
        evidence_sentences = sent_tokenize(evidence_text)
        
        # 回答句子与证据句子两两比对
        for ans_sent in answer_sentences:
            for evi_sent in evidence_sentences:
                # 检测矛盾
                conflict_result = detect_sentence_conflict(ans_sent, evi_sent, rules)
                
                if conflict_result["has_conflict"]:
                    conflicts.append({
                        "type": "evidence",
                        "evidence_id": evidence_id,
                        "answer_sentence": ans_sent,
                        "evidence_sentence": evi_sent,
                        "conflict_type": conflict_result["conflict_type"],
                        "details": conflict_result["conflict_details"]
                    })
    
    return conflicts

# 生成修订建议
def generate_revision_suggestions(conflicts):
    """
    根据检测到的矛盾生成修订建议
    
    参数:
        conflicts: 矛盾检测结果列表
    
    返回:
        修订建议列表
    """
    suggestions = []
    
    for conflict in conflicts:
        if conflict["type"] == "internal":
            suggestions.append({
                "suggestion_type": "resolve_internal_conflict",
                "priority": "high",
                "description": f"回答内部存在矛盾：'{conflict['details']}'。建议修改其中一个句子以保持一致性。",
                "original_text": f"句子1: {conflict['sentence1']}\n句子2: {conflict['sentence2']}"
            })
        elif conflict["type"] == "evidence":
            suggestions.append({
                "suggestion_type": "align_with_evidence",
                "priority": "medium",
                "description": f"回答与证据{conflict['evidence_id']}存在矛盾：'{conflict['details']}'。建议修改回答以与证据保持一致。",
                "original_text": f"回答: {conflict['answer_sentence']}\n证据: {conflict['evidence_sentence']}"
            })
    
    return suggestions

# 计算矛盾率
def calculate_contradiction_rate(conflicts, total_pairs):
    """
    计算矛盾率
    
    参数:
        conflicts: 矛盾检测结果列表
        total_pairs: 总比对句子对数量
    
    返回:
        矛盾率（百分比）
    """
    if total_pairs == 0:
        return 0.0
    
    return (len(conflicts) / total_pairs) * 100

# 导出矛盾样例报告
def export_conflicts_report(conflicts, suggestions):
    """
    导出矛盾样例报告
    
    参数:
        conflicts: 矛盾检测结果列表
        suggestions: 修订建议列表
    """
    # 准备导出数据
    export_data = []
    
    for i, (conflict, suggestion) in enumerate(zip(conflicts, suggestions)):
        export_item = {
            "id": f"conflict_{i+1}",
            "conflict_type": conflict["type"],
            "details": conflict["details"],
            "suggestion": suggestion["description"],
            "priority": suggestion["priority"]
        }
        
        # 根据矛盾类型添加不同的原始文本
        if conflict["type"] == "internal":
            export_item["original_text"] = f"句子1: {conflict['sentence1']}\n句子2: {conflict['sentence2']}"
        elif conflict["type"] == "evidence":
            export_item["original_text"] = f"回答: {conflict['answer_sentence']}\n证据: {conflict['evidence_sentence']}"
            export_item["evidence_id"] = conflict["evidence_id"]
        
        export_data.append(export_item)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(export_data)
    report_path = "contradictions_examples.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    
    print(f"矛盾样例报告已导出: {report_path} (共{len(df)}条记录)")
    
    return report_path

def main():
    """
    主函数，协调整个矛盾检测流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="规则化一致性与矛盾率估计")
    parser.add_argument("--answers", type=str, default="answers.csv", help="回答数据文件路径")
    parser.add_argument("--evidence", type=str, default="evidence.csv", help="证据数据文件路径")
    args = parser.parse_args()
    
    print(f"启动规则化一致性与矛盾率估计")
    
    # 检查并安装依赖
    install_dependencies()
    
    # 加载数据
    answers_df = load_answers(args.answers)
    evidence_df = load_evidence(args.evidence, answers_df)
    
    # 初始化矛盾检测规则
    rules = init_conflict_rules()
    
    # 记录开始时间
    start_time = time.time()
    
    # 汇总所有矛盾和计算总比对对数
    all_conflicts = []
    total_sentence_pairs = 0
    
    # 按回答ID分组检查
    for _, answer_row in tqdm(answers_df.iterrows(), desc="检查回答", total=len(answers_df)):
        answer_id = answer_row["answer_id"]
        answer_text = answer_row["answer_text"]
        
        # 确保answer_text是字符串类型
        if not isinstance(answer_text, str):
            answer_text = str(answer_text) if answer_text is not None else ""
        
        # 获取该回答的所有证据
        answer_evidences = evidence_df[evidence_df["answer_id"] == answer_id]
        evidence_texts = {row["evidence_id"]: row["evidence_text"] for _, row in answer_evidences.iterrows()}
        
        # 检测内部矛盾
        internal_conflicts = detect_internal_conflicts(answer_text, rules)
        all_conflicts.extend(internal_conflicts)
        
        # 检测与证据的矛盾
        evidence_conflicts = detect_evidence_conflicts(answer_text, evidence_texts, rules)
        all_conflicts.extend(evidence_conflicts)
        
        # 计算该回答的总比对对数
        answer_sentences = sent_tokenize(answer_text)
        num_answer_sents = len(answer_sentences)
        # 内部比对对数: C(n, 2) = n*(n-1)/2
        internal_pairs = num_answer_sents * (num_answer_sents - 1) // 2
        
        # 与证据比对对数: 回答句子数 × 证据句子数
        evidence_pairs = 0
        for evidence_text in evidence_texts.values():
            # 确保evidence_text是字符串类型
            if not isinstance(evidence_text, str):
                evidence_text = str(evidence_text) if evidence_text is not None else ""
            
            evidence_sentences = sent_tokenize(evidence_text)
            evidence_pairs += num_answer_sents * len(evidence_sentences)
        
        # 累加总比对对数
        total_sentence_pairs += internal_pairs + evidence_pairs
    
    # 生成修订建议
    revision_suggestions = generate_revision_suggestions(all_conflicts)
    
    # 计算矛盾率
    contradiction_rate = calculate_contradiction_rate(all_conflicts, total_sentence_pairs)
    
    # 记录结束时间
    end_time = time.time()
    
    # 导出矛盾样例报告
    if all_conflicts and revision_suggestions:
        export_conflicts_report(all_conflicts, revision_suggestions)
    
    # 打印执行时间
    exec_time = end_time - start_time
    
    # 输出结果摘要
    print(f"\nNLI-approx contradiction_rate={contradiction_rate:.1f}%")
    print(f"检测到矛盾总数: {len(all_conflicts)}")
    print(f"总比对句子对数: {total_sentence_pairs}")
    
    # 统计矛盾类型分布
    conflict_types = {}
    for conflict in all_conflicts:
        c_type = conflict["conflict_type"]
        conflict_types[c_type] = conflict_types.get(c_type, 0) + 1
    
    if conflict_types:
        print("矛盾类型分布:")
        for c_type, count in conflict_types.items():
            print(f"  {c_type}: {count}条")
    
    # 验证中文显示
    sample_answer = answers_df["answer_text"].iloc[0] if not answers_df.empty else "中文测试"
    # 确保sample_answer是字符串类型
    if not isinstance(sample_answer, str):
        sample_answer = str(sample_answer) if sample_answer is not None else "中文测试"
    print(f"\n中文显示测试: {sample_answer[:50]}...")
    
    # 检查执行效率
    if exec_time > 1.0:
        print("\n注意: 程序执行时间较长，建议优化数据处理流程以提高效率。")
    else:
        print(f"\n程序执行时间: {round(exec_time, 3)}秒")
        print("程序执行效率良好。")

if __name__ == "__main__":
    main()