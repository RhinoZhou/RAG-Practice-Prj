#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集分层划分示例

功能说明：
    模拟生成训练集/验证集/测试集并保持主题分布一致
    基于常见的多类别样本，执行分层抽样（stratified sampling），
    保证各类别比例均匀，输出三类数据文件。

作者：Ph.D. Rhino

执行流程：
    1. 生成模拟数据集（包含多类别标注的JSON格式数据）
    2. 检查并安装必要的依赖
    3. 加载数据并提取类别标签
    4. 使用分层抽样方法划分数据集
    5. 保存划分后的train/valid/test文件
    6. 分析并输出各子集的类别分布比例对比

输入说明：
    程序会自动生成模拟的数据集文件 dataset_cleaned.json

输出展示：
    - train.json (70%)
    - valid.json (15%)
    - test.json (15%)
    - 控制台输出各子集的类别分布统计信息
"""

import os
import json
import random
import shutil
from collections import Counter
from typing import List, Dict, Any, Tuple

# 设置随机种子，确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    使用Python标准库实现，无需额外依赖
    """
    print("正在检查依赖...")
    # 此程序仅使用Python标准库，无需额外安装依赖
    print("依赖检查完成，所有必要模块已可用。")


def generate_sample_dataset(output_file: str, num_samples: int = 1000) -> None:
    """
    生成模拟的多类别数据集
    
    Args:
        output_file: 输出文件路径
        num_samples: 样本数量
    """
    print(f"正在生成模拟数据集，共{num_samples}个样本...")
    
    # 定义示例类别主题
    categories = [
        "技术", "教育", "医疗", "金融", "娱乐",
        "体育", "旅游", "美食", "科技", "政治"
    ]
    
    # 定义示例问题模板
    question_templates = [
        "什么是{topic}？",
        "解释一下{topic}的概念。",
        "{topic}有哪些应用场景？",
        "如何学习{topic}？",
        "{topic}的未来发展趋势是什么？"
    ]
    
    dataset = []
    # 为每个类别生成一定数量的样本，确保类别分布相对均匀
    samples_per_category = num_samples // len(categories)
    
    for category in categories:
        for _ in range(samples_per_category):
            # 随机选择问题模板
            question_template = random.choice(question_templates)
            # 根据主题填充问题模板
            question = question_template.format(topic=category)
            # 生成简单的回答内容
            answer = f"这是关于{category}主题的详细回答。{question}的答案涉及多个方面..."
            
            # 创建样本字典
            sample = {
                "instruction": question,
                "response": answer,
                "category": category,  # 添加类别标签
                "id": f"{category}_{len(dataset) + 1}"  # 生成唯一ID
            }
            dataset.append(sample)
    
    # 添加一些随机样本，使类别分布不完全均匀
    remaining_samples = num_samples - len(dataset)
    for i in range(remaining_samples):
        category = random.choice(categories)
        question_template = random.choice(question_templates)
        question = question_template.format(topic=category)
        answer = f"这是关于{category}主题的详细回答。{question}的答案涉及多个方面..."
        
        sample = {
            "instruction": question,
            "response": answer,
            "category": category,
            "id": f"{category}_{len(dataset) + 1}"
        }
        dataset.append(sample)
    
    # 打乱数据集顺序
    random.shuffle(dataset)
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"模拟数据集已生成: {output_file}")
    # 统计各类别数量
    category_counts = Counter([sample["category"] for sample in dataset])
    print("数据集类别分布:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}个样本 ({count/num_samples:.1%})")


def load_dataset(input_file: str) -> List[Dict[str, Any]]:
    """
    加载数据集文件
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        数据样本列表
    """
    print(f"正在加载数据集: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载{len(data)}个样本")
    return data


def stratified_split(data: List[Dict[str, Any]], 
                     train_ratio: float = 0.7, 
                     valid_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层抽样划分数据集
    
    Args:
        data: 原始数据列表
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
    
    Returns:
        (train_data, valid_data, test_data) 元组
    """
    print("开始执行分层抽样...")
    
    # 按类别分组
    category_to_samples = {}
    for sample in data:
        category = sample.get("category", "unknown")
        if category not in category_to_samples:
            category_to_samples[category] = []
        category_to_samples[category].append(sample)
    
    train_data, valid_data, test_data = [], [], []
    
    # 对每个类别分别进行分层抽样
    for category, samples in category_to_samples.items():
        # 打乱当前类别的样本顺序
        random.shuffle(samples)
        
        # 计算各类别在子集中的样本数量
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_valid = int(n_samples * valid_ratio)
        
        # 分配样本到各个子集
        train_data.extend(samples[:n_train])
        valid_data.extend(samples[n_train:n_train + n_valid])
        test_data.extend(samples[n_train + n_valid:])
    
    # 再次打乱各个子集，使类别混合
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    print(f"分层抽样完成:")
    print(f"  训练集: {len(train_data)}个样本 ({len(train_data)/len(data):.1%})")
    print(f"  验证集: {len(valid_data)}个样本 ({len(valid_data)/len(data):.1%})")
    print(f"  测试集: {len(test_data)}个样本 ({len(test_data)/len(data):.1%})")
    
    return train_data, valid_data, test_data


def save_dataset(data: List[Dict[str, Any]], output_file: str) -> None:
    """
    保存数据集到文件
    
    Args:
        data: 数据样本列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据集已保存: {output_file}")


def analyze_category_distribution(data: List[Dict[str, Any]], dataset_name: str) -> Dict[str, float]:
    """
    分析数据集的类别分布
    
    Args:
        data: 数据样本列表
        dataset_name: 数据集名称
    
    Returns:
        类别分布字典 {category: percentage}
    """
    if not data:
        return {}
    
    # 统计各类别数量
    category_counts = Counter([sample.get("category", "unknown") for sample in data])
    total = sum(category_counts.values())
    
    # 计算各类别占比
    distribution = {cat: count/total for cat, count in category_counts.items()}
    
    print(f"\n{dataset_name}类别分布:")
    for category, percentage in sorted(distribution.items()):
        count = category_counts[category]
        print(f"  {category}: {count}个样本 ({percentage:.1%})")
    
    return distribution


def compare_distributions(train_dist: Dict[str, float], 
                          valid_dist: Dict[str, float], 
                          test_dist: Dict[str, float]) -> None:
    """
    比较不同数据集的类别分布差异
    
    Args:
        train_dist: 训练集分布
        valid_dist: 验证集分布
        test_dist: 测试集分布
    """
    print("\n类别分布一致性分析:")
    print("类别       训练集     验证集     测试集     最大差异")
    print("-" * 60)
    
    # 获取所有类别
    all_categories = set(train_dist.keys()) | set(valid_dist.keys()) | set(test_dist.keys())
    
    max_overall_diff = 0.0
    
    for category in sorted(all_categories):
        train_pct = train_dist.get(category, 0.0)
        valid_pct = valid_dist.get(category, 0.0)
        test_pct = test_dist.get(category, 0.0)
        
        # 计算该类别在三个子集中的最大差异
        values = [train_pct, valid_pct, test_pct]
        max_diff = max(values) - min(values)
        max_overall_diff = max(max_overall_diff, max_diff)
        
        print(f"{category: <10}{train_pct: >9.1%}{valid_pct: >9.1%}{test_pct: >9.1%}{max_diff: >10.1%}")
    
    print("-" * 60)
    print(f"总体类别分布最大差异: {max_overall_diff:.2%}")
    print(f"分层抽样质量评估: {'良好' if max_overall_diff < 0.02 else '一般' if max_overall_diff < 0.05 else '需要改进'}")


def main():
    """
    主函数：协调整个数据集划分流程
    """
    print("===== 数据集分层划分示例程序 =====")
    
    # 数据集文件路径
    input_file = "dataset_cleaned.json"
    train_file = "train.json"
    valid_file = "valid.json"
    test_file = "test.json"
    
    # 1. 检查依赖
    check_and_install_dependencies()
    
    # 2. 如果输入文件不存在，生成模拟数据集
    if not os.path.exists(input_file):
        print(f"未找到输入文件 {input_file}，开始生成模拟数据集...")
        generate_sample_dataset(input_file, num_samples=1000)
    else:
        print(f"输入文件 {input_file} 已存在，将直接使用。")
    
    # 3. 加载数据集
    data = load_dataset(input_file)
    
    # 4. 分层抽样划分数据集
    train_data, valid_data, test_data = stratified_split(data, train_ratio=0.7, valid_ratio=0.15)
    
    # 5. 保存划分后的数据集
    save_dataset(train_data, train_file)
    save_dataset(valid_data, valid_file)
    save_dataset(test_data, test_file)
    
    # 6. 分析各类别分布
    train_dist = analyze_category_distribution(train_data, "训练集")
    valid_dist = analyze_category_distribution(valid_data, "验证集")
    test_dist = analyze_category_distribution(test_data, "测试集")
    
    # 7. 比较不同数据集的类别分布一致性
    compare_distributions(train_dist, valid_dist, test_dist)
    
    print("\n===== 数据集分层划分完成 =====")
    print(f"训练集文件: {train_file} ({len(train_data)}个样本)")
    print(f"验证集文件: {valid_file} ({len(valid_data)}个样本)")
    print(f"测试集文件: {test_file} ({len(test_data)}个样本)")
    print("\n程序执行完毕！")


if __name__ == "__main__":
    main()