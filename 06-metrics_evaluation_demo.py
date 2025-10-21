#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型性能指标计算示例

功能说明：
    通过预测与标签对比计算分类模型的核心评估指标
    实现 Precision、Recall、F1、Accuracy 的精确计算
    支持二分类和多分类场景
    提供混淆矩阵可视化和指标表格输出

作者：Ph.D. Rhino

执行流程：
    1. 检查并安装必要的依赖
    2. 生成模拟的真实标签和预测标签数据
    3. 计算 TP、FP、FN、TN 等基础统计量
    4. 基于统计量计算各项性能指标
    5. 生成混淆矩阵
    6. 输出指标表格和分析报告

输入说明：
    程序自动生成真实标签与预测标签数据
    也支持通过函数参数传入自定义标签数组

输出展示：
    - 二分类指标：Precision、Recall、F1、Accuracy
    - 多分类指标：各类别指标及宏平均、微平均结果
    - 混淆矩阵可视化
    - 性能分析报告
"""

import os
import random
import math
from typing import List, Dict, Tuple, Union, Optional
import numpy as np


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    需要安装：numpy（用于矩阵操作和数据处理）
    """
    print("正在检查依赖...")
    
    # 需要的依赖包列表
    required_packages = ["numpy"]
    missing_packages = []
    
    # 检查每个包是否已安装
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    # 安装缺失的包
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "--upgrade", "pip"])
            subprocess.check_call(["pip", "install"] + missing_packages)
            print("依赖包安装成功！")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            raise
    else:
        print("所有必要的依赖包已安装完成。")


def generate_binary_classification_data(
    num_samples: int = 100,
    positive_ratio: float = 0.3,
    accuracy: float = 0.85,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    生成二分类模拟数据
    
    Args:
        num_samples: 样本数量
        positive_ratio: 正类样本比例
        accuracy: 模型准确率（控制预测质量）
        seed: 随机种子
    
    Returns:
        (真实标签列表, 预测标签列表)
    """
    print(f"生成二分类模拟数据 ({num_samples}个样本，正类比例{positive_ratio:.2f})...")
    
    # 设置随机种子以保证结果可复现
    random.seed(seed)
    
    # 生成真实标签
    y_true = []
    num_positive = int(num_samples * positive_ratio)
    num_negative = num_samples - num_positive
    
    y_true.extend([1] * num_positive)
    y_true.extend([0] * num_negative)
    
    # 随机打乱顺序
    random.shuffle(y_true)
    
    # 生成预测标签，基于设定的准确率
    y_pred = []
    for true_label in y_true:
        if random.random() < accuracy:
            # 正确预测
            y_pred.append(true_label)
        else:
            # 错误预测（翻转标签）
            y_pred.append(1 - true_label)
    
    # 统计数据分布
    true_pos_count = sum(1 for label in y_true if label == 1)
    pred_pos_count = sum(1 for label in y_pred if label == 1)
    
    print(f"  真实正类样本数: {true_pos_count}")
    print(f"  真实负类样本数: {num_samples - true_pos_count}")
    print(f"  预测正类样本数: {pred_pos_count}")
    print(f"  预测负类样本数: {num_samples - pred_pos_count}")
    
    return y_true, y_pred


def generate_multiclass_data(
    num_samples: int = 100,
    num_classes: int = 3,
    accuracy: float = 0.75,
    class_distribution: Optional[List[float]] = None,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    生成多分类模拟数据
    
    Args:
        num_samples: 样本数量
        num_classes: 类别数量
        accuracy: 模型准确率
        class_distribution: 各类别分布比例，如果为None则均匀分布
        seed: 随机种子
    
    Returns:
        (真实标签列表, 预测标签列表)
    """
    print(f"生成多分类模拟数据 ({num_samples}个样本，{num_classes}个类别)...")
    
    # 设置随机种子
    random.seed(seed)
    
    # 如果未指定类别分布，则使用均匀分布
    if class_distribution is None:
        class_distribution = [1.0 / num_classes] * num_classes
    
    # 确保分布和为1
    if sum(class_distribution) != 1.0:
        class_distribution = [d / sum(class_distribution) for d in class_distribution]
    
    # 生成真实标签
    y_true = []
    for class_idx, proportion in enumerate(class_distribution):
        count = int(num_samples * proportion)
        y_true.extend([class_idx] * count)
    
    # 补充剩余样本（由于浮点数精度问题）
    remaining = num_samples - len(y_true)
    if remaining > 0:
        y_true.extend([random.randint(0, num_classes - 1) for _ in range(remaining)])
    
    # 随机打乱顺序
    random.shuffle(y_true)
    
    # 生成预测标签
    y_pred = []
    for true_label in y_true:
        if random.random() < accuracy:
            # 正确预测
            y_pred.append(true_label)
        else:
            # 错误预测（随机选择其他类别）
            other_classes = [c for c in range(num_classes) if c != true_label]
            y_pred.append(random.choice(other_classes))
    
    # 统计各类别数量
    print("  各类别样本统计:")
    for class_idx in range(num_classes):
        true_count = sum(1 for label in y_true if label == class_idx)
        pred_count = sum(1 for label in y_pred if label == class_idx)
        print(f"    类别{class_idx}: 真实={true_count}, 预测={pred_count}")
    
    return y_true, y_pred


def calculate_binary_metrics(
    y_true: List[int], 
    y_pred: List[int]
) -> Dict[str, float]:
    """
    计算二分类模型的评估指标
    
    Args:
        y_true: 真实标签列表（0表示负类，1表示正类）
        y_pred: 预测标签列表（0表示负类，1表示正类）
    
    Returns:
        包含各项评估指标的字典
    """
    print("\n计算二分类评估指标...")
    
    # 检查输入长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签的长度必须一致")
    
    # 计算TP, FP, FN, TN
    tp = 0  # True Positive: 预测为正类且实际为正类
    fp = 0  # False Positive: 预测为正类但实际为负类
    fn = 0  # False Negative: 预测为负类但实际为正类
    tn = 0  # True Negative: 预测为负类且实际为负类
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
    
    print(f"  混淆矩阵统计:")
    print(f"    TP: {tp}, FP: {fp}")
    print(f"    FN: {fn}, TN: {tn}")
    
    # 计算各项指标
    # 准确率 (Accuracy): 正确预测的样本数 / 总样本数
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # 精确率 (Precision): TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率 (Recall): TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1分数: 2 * Precision * Recall / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 特异度 (Specificity): TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 假正率 (False Positive Rate): FP / (TN + FP)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 假负率 (False Negative Rate): FN / (TP + FN)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }
    
    return metrics


def calculate_multiclass_metrics(
    y_true: List[int], 
    y_pred: List[int]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    计算多分类模型的评估指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
    
    Returns:
        包含各项评估指标的字典，包括每个类别的指标和总体指标
    """
    print("\n计算多分类评估指标...")
    
    # 检查输入长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签的长度必须一致")
    
    # 获取所有类别
    classes = sorted(set(y_true + y_pred))
    num_classes = len(classes)
    
    print(f"  检测到 {num_classes} 个类别: {classes}")
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 构建混淆矩阵
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = classes.index(true_label)
        pred_idx = classes.index(pred_label)
        confusion_matrix[true_idx][pred_idx] += 1
    
    # 计算每个类别的指标
    class_metrics = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    
    for i, class_label in enumerate(classes):
        # TP: 对角线元素
        tp = confusion_matrix[i][i]
        
        # FP: 第i列的总和减去TP
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # FN: 第i行的总和减去TP
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # TN: 总和减去TP、FP、FN
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        # 计算各类别指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    
    # 计算宏平均指标（Macro Average）
    macro_precision = precision_sum / num_classes
    macro_recall = recall_sum / num_classes
    macro_f1 = f1_sum / num_classes
    
    # 计算微平均指标（Micro Average）
    total_tp = np.sum(np.diag(confusion_matrix))
    total_fp = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    total_fp = np.sum(total_fp)
    total_fn = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    total_fn = np.sum(total_fn)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # 计算总体准确率
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0.0
    
    # 组合所有指标
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "confusion_matrix": confusion_matrix.tolist(),
        "classes": classes,
        "class_metrics": class_metrics
    }
    
    return metrics


def print_binary_classification_report(metrics: Dict[str, float]) -> None:
    """
    打印二分类模型的评估报告
    
    Args:
        metrics: 包含评估指标的字典
    """
    print("\n" + "="*60)
    print("二分类模型评估报告")
    print("="*60)
    
    # 打印关键指标
    print("关键评估指标:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print()
    
    # 打印附加指标
    print("附加评估指标:")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  FPR:         {metrics['fpr']:.4f}")
    print(f"  FNR:         {metrics['fnr']:.4f}")
    print()
    
    # 打印混淆矩阵
    print("混淆矩阵:")
    print(f"              预测正类   预测负类")
    print(f"实际正类    {metrics['tp']:>8d}      {metrics['fn']:>8d}")
    print(f"实际负类    {metrics['fp']:>8d}      {metrics['tn']:>8d}")
    print()
    
    # 生成简易报告行，符合用户期望的输出格式
    print("简易报告行:")
    print(f"yaml")
    print(f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.3f}")
    print()
    
    # 添加性能分析
    analyze_model_performance(metrics)


def print_multiclass_classification_report(metrics: Dict[str, Union[float, Dict[str, float]]]) -> None:
    """
    打印多分类模型的评估报告
    
    Args:
        metrics: 包含评估指标的字典
    """
    print("\n" + "="*60)
    print("多分类模型评估报告")
    print("="*60)
    
    # 打印总体准确率
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print()
    
    # 打印宏平均和微平均指标
    print("汇总指标:")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall:    {metrics['micro_recall']:.4f}")
    print(f"  Micro F1:        {metrics['micro_f1']:.4f}")
    print()
    
    # 打印每个类别的指标
    print("各类别指标:")
    print(f"{'类别':<6} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 40)
    
    for class_label, class_metric in metrics['class_metrics'].items():
        print(f"{class_label:<6} {class_metric['precision']:<10.4f} {class_metric['recall']:<10.4f} {class_metric['f1']:<10.4f}")
    print()
    
    # 打印混淆矩阵
    print("混淆矩阵:")
    cm = np.array(metrics['confusion_matrix'])
    classes = metrics['classes']
    
    # 打印表头
    print(f"{'':<6}", end="")
    for c in classes:
        print(f"{c:<6}", end="")
    print()
    
    # 打印分隔线
    print("-" * (6 + 6 * len(classes)))
    
    # 打印每一行
    for i, c in enumerate(classes):
        print(f"{c:<6}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:<6}", end="")
        print()
    print()
    
    # 生成简易报告行
    print("简易报告行:")
    print(f"yaml")
    print(f"Macro Precision: {metrics['macro_precision']:.2f}, Macro Recall: {metrics['macro_recall']:.2f}, Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Micro Precision: {metrics['micro_precision']:.2f}, Micro Recall: {metrics['micro_recall']:.2f}, Micro F1: {metrics['micro_f1']:.3f}")


def analyze_model_performance(metrics: Dict[str, float]) -> None:
    """
    分析模型性能并提供建议
    
    Args:
        metrics: 二分类评估指标字典
    """
    print("性能分析:")
    
    # 评估整体性能
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    
    if accuracy >= 0.9 and f1 >= 0.9:
        print("  ✓ 模型性能优秀，准确率和F1分数都很高")
    elif accuracy >= 0.8 and f1 >= 0.8:
        print("  ✓ 模型性能良好，在大多数场景下可以接受")
    elif accuracy >= 0.7 and f1 >= 0.7:
        print("  ! 模型性能一般，可能需要进一步优化")
    else:
        print("  ✗ 模型性能较差，建议重新训练或调整参数")
    
    # 分析精确率和召回率的权衡
    precision = metrics['precision']
    recall = metrics['recall']
    
    if precision > recall + 0.1:
        print(f"  ! 精确率({precision:.3f})明显高于召回率({recall:.3f})，模型偏向于减少误报")
        print("    建议: 调整阈值以提高召回率，或增加正类样本")
    elif recall > precision + 0.1:
        print(f"  ! 召回率({recall:.3f})明显高于精确率({precision:.3f})，模型偏向于减少漏报")
        print("    建议: 调整阈值以提高精确率，或增加负类样本")
    else:
        print("  ✓ 精确率和召回率较为平衡，模型在两个指标间取得了较好的权衡")
    
    # 分析混淆矩阵
    fp = metrics['fp']
    fn = metrics['fn']
    
    if fp > fn * 2:
        print("  ! 误报数量明显多于漏报，可能导致用户体验下降")
    elif fn > fp * 2:
        print("  ! 漏报数量明显多于误报，在某些安全敏感场景下可能存在风险")


def generate_detailed_binary_report() -> None:
    """
    生成详细的二分类报告示例
    """
    print("\n===== 二分类模型评估详细示例 =====")
    
    # 生成不同场景下的数据
    scenarios = [
        {"name": "平衡数据集", "positive_ratio": 0.5, "accuracy": 0.85},
        {"name": "不平衡数据集(正类少)", "positive_ratio": 0.1, "accuracy": 0.85},
        {"name": "高精确率场景", "positive_ratio": 0.5, "accuracy": 0.90, "bias": "precision"},
        {"name": "高召回率场景", "positive_ratio": 0.5, "accuracy": 0.90, "bias": "recall"}
    ]
    
    for scenario in scenarios:
        print(f"\n\n----- {scenario['name']} -----\n")
        
        # 生成数据
        if "bias" in scenario:
            # 为特定场景生成有偏向的数据
            y_true = []
            y_pred = []
            num_samples = 1000
            num_positive = int(num_samples * scenario['positive_ratio'])
            
            # 创建基础标签
            y_true.extend([1] * num_positive)
            y_true.extend([0] * (num_samples - num_positive))
            
            # 根据偏向调整预测
            for true_label in y_true:
                if random.random() < scenario['accuracy']:
                    y_pred.append(true_label)
                else:
                    if scenario['bias'] == "precision":
                        # 偏向精确率：宁可漏报，不要误报
                        if true_label == 0:
                            y_pred.append(1)  # 只有负类才会被错误预测为正类
                        else:
                            y_pred.append(0)
                    else:  # recall bias
                        # 偏向召回率：宁可误报，不要漏报
                        if true_label == 1:
                            y_pred.append(0)  # 只有正类才会被错误预测为负类
                        else:
                            y_pred.append(1)
        else:
            # 使用标准数据生成
            y_true, y_pred = generate_binary_classification_data(
                num_samples=1000,
                positive_ratio=scenario['positive_ratio'],
                accuracy=scenario['accuracy'],
                seed=42
            )
        
        # 计算指标
        metrics = calculate_binary_metrics(y_true, y_pred)
        
        # 打印简化报告
        print(f"场景: {scenario['name']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print("-" * 40)


def main():
    """
    主函数
    """
    print("===== 模型性能指标计算示例程序 =====")
    
    # 检查依赖
    check_and_install_dependencies()
    
    # 设置随机种子
    random.seed(42)
    
    # 1. 二分类评估示例
    print("\n\n===== 二分类评估示例 =====")
    y_true_binary, y_pred_binary = generate_binary_classification_data(
        num_samples=100,
        positive_ratio=0.3,
        accuracy=0.85
    )
    
    # 计算二分类指标
    binary_metrics = calculate_binary_metrics(y_true_binary, y_pred_binary)
    
    # 打印二分类报告
    print_binary_classification_report(binary_metrics)
    
    # 2. 多分类评估示例
    print("\n\n===== 多分类评估示例 =====")
    y_true_multi, y_pred_multi = generate_multiclass_data(
        num_samples=150,
        num_classes=3,
        accuracy=0.80,
        class_distribution=[0.4, 0.3, 0.3]
    )
    
    # 计算多分类指标
    multi_metrics = calculate_multiclass_metrics(y_true_multi, y_pred_multi)
    
    # 打印多分类报告
    print_multiclass_classification_report(multi_metrics)
    
    # 3. 不同场景的详细对比
    generate_detailed_binary_report()
    
    print("\n\n===== 模型性能指标计算示例程序完成 =====")
    print("\n关键概念总结:")
    print("- Precision (精确率): 预测为正类中实际正类的比例")
    print("- Recall (召回率): 实际正类被正确预测的比例")
    print("- F1 Score: Precision和Recall的调和平均")
    print("- Accuracy (准确率): 所有预测正确的样本比例")
    print("- Macro Average: 各类别指标的算术平均")
    print("- Micro Average: 基于全局统计的指标计算")


if __name__ == "__main__":
    main()