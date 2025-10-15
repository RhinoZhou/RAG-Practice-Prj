#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B检验与样本量计算器
作者：Ph.D. Rhino

功能说明：
本程序用于对两组指标进行统计检验（t检验或非参数检验），比较均值或中位数差异，并根据预期效应量计算所需样本量。
可用于评估RAG系统的不同配置、优化策略的效果差异，帮助研究者和工程师做出数据驱动的决策。

内容概述：
1. 读取两组样本数据（如FactScore、NDCG等评估指标）
2. 支持t检验（参数检验）和Wilcoxon秩和检验（非参数检验）
3. 计算均值差/中位数差、p值、置信区间
4. 根据历史方差与期望提升计算所需样本量和统计功效
5. 输出多重比较校正建议
6. 自动生成示例数据集用于演示

执行流程：
1. 读取control.csv与treatment.csv（单列指标数据）
2. 根据用户选择进行检验类型（t/非参）和假设方向（单/双侧）设置
3. 计算统计指标，输出显著性判断
4. 根据用户输入的效应量、方差、显著性水平和功效进行样本量估算
5. 生成完整报告并提供多重比较校正提醒
"""

import os
import sys

# 先检查依赖包
required_packages = ["pandas", "numpy", "scipy", "matplotlib", "statsmodels"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"正在安装缺失的依赖包：{', '.join(missing_packages)}")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("依赖包安装成功！")
    except Exception as e:
        print(f"依赖包安装失败：{e}")
        sys.exit(1)
else:
    print("所有依赖包已安装完成。")

# 导入必要的库
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower
import argparse
import time

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题




def generate_sample_data(mean_control=0.74, std_control=0.12, mean_treatment=0.76, 
                         std_treatment=0.12, n_samples=500, seed=42):
    """
    生成示例数据集，包含对照组和处理组的指标数据
    
    参数：
        mean_control: 对照组均值
        std_control: 对照组标准差
        mean_treatment: 处理组均值
        std_treatment: 处理组标准差
        n_samples: 每组样本量
        seed: 随机数种子，保证结果可复现
    """
    np.random.seed(seed)
    
    # 生成对照组和处理组数据
    control_data = np.random.normal(mean_control, std_control, n_samples)
    treatment_data = np.random.normal(mean_treatment, std_treatment, n_samples)
    
    # 确保数据在合理范围内（0-1之间，假设是评分类指标）
    control_data = np.clip(control_data, 0, 1)
    treatment_data = np.clip(treatment_data, 0, 1)
    
    # 保存为CSV文件
    pd.DataFrame({"score": control_data}).to_csv("control.csv", index=False)
    pd.DataFrame({"score": treatment_data}).to_csv("treatment.csv", index=False)
    
    print(f"已生成示例数据：")
    print(f"- 对照组：{n_samples}个样本，均值={mean_control:.3f}，标准差={std_control:.3f}")
    print(f"- 处理组：{n_samples}个样本，均值={mean_treatment:.3f}，标准差={std_treatment:.3f}")


def load_data(control_file, treatment_file):
    """
    加载对照组和处理组数据
    
    参数：
        control_file: 对照组数据文件路径
        treatment_file: 处理组数据文件路径
    
    返回：
        control_data: 对照组数据数组
        treatment_data: 处理组数据数组
    """
    try:
        control_df = pd.read_csv(control_file)
        treatment_df = pd.read_csv(treatment_file)
        
        # 假设数据在第一列
        control_data = control_df.iloc[:, 0].values
        treatment_data = treatment_df.iloc[:, 0].values
        
        print(f"数据加载完成：")
        print(f"- 对照组：{len(control_data)}个样本")
        print(f"- 处理组：{len(treatment_data)}个样本")
        
        return control_data, treatment_data
    except Exception as e:
        print(f"数据加载失败：{e}")
        sys.exit(1)


def perform_statistical_tests(control_data, treatment_data, test_type="t", tail="two"):
    """
    执行统计检验，计算均值/中位数差异、p值和置信区间
    
    参数：
        control_data: 对照组数据
        treatment_data: 处理组数据
        test_type: 检验类型，"t"为t检验，"nonparametric"为非参数检验
        tail: 检验方向，"two"为双侧检验，"left"为左侧检验，"right"为右侧检验
    
    返回：
        results: 包含统计检验结果的字典
    """
    # 计算基本统计量
    control_mean = np.mean(control_data)
    treatment_mean = np.mean(treatment_data)
    mean_diff = treatment_mean - control_mean
    
    control_median = np.median(control_data)
    treatment_median = np.median(treatment_data)
    median_diff = treatment_median - control_median
    
    # 根据检验类型选择统计方法
    if test_type.lower() == "t":
        # t检验
        if tail.lower() == "two":
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
        elif tail.lower() == "left":
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
            p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        elif tail.lower() == "right":
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
            p_value = 1 - p_value / 2 if t_stat > 0 else p_value / 2
        else:
            print(f"不支持的检验方向：{tail}")
            sys.exit(1)
        
        # 计算95%置信区间
        n1, n2 = len(control_data), len(treatment_data)
        se = np.sqrt(np.var(control_data, ddof=1)/n1 + np.var(treatment_data, ddof=1)/n2)
        df = (se**4) / ((np.var(control_data, ddof=1)**2)/(n1**2*(n1-1)) + 
                        (np.var(treatment_data, ddof=1)**2)/(n2**2*(n2-1)))
        ci_low, ci_high = mean_diff - stats.t.ppf(0.975, df)*se, mean_diff + stats.t.ppf(0.975, df)*se
        
        test_name = "独立样本t检验"
        effect_size = mean_diff / np.sqrt((np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1))/2)  # Cohen's d
        
    elif test_type.lower() == "nonparametric":
        # Wilcoxon秩和检验（非参数检验）
        if tail.lower() == "two":
            w_stat, p_value = stats.mannwhitneyu(control_data, treatment_data, alternative="two-sided")
        elif tail.lower() == "left":
            w_stat, p_value = stats.mannwhitneyu(control_data, treatment_data, alternative="less")
        elif tail.lower() == "right":
            w_stat, p_value = stats.mannwhitneyu(control_data, treatment_data, alternative="greater")
        else:
            print(f"不支持的检验方向：{tail}")
            sys.exit(1)
        
        # 对于非参数检验，使用自助法计算置信区间
        n_bootstrap = 10000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            control_boot = np.random.choice(control_data, size=len(control_data), replace=True)
            treatment_boot = np.random.choice(treatment_data, size=len(treatment_data), replace=True)
            bootstrap_diffs.append(np.mean(treatment_boot) - np.mean(control_boot))
        
        ci_low, ci_high = np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)
        test_name = "Wilcoxon秩和检验（非参数）"
        effect_size = "N/A"  # 非参数检验没有标准的效应量指标
        
    else:
        print(f"不支持的检验类型：{test_type}")
        sys.exit(1)
    
    results = {
        "test_name": test_name,
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "mean_diff": mean_diff,
        "control_median": control_median,
        "treatment_median": treatment_median,
        "median_diff": median_diff,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_size": effect_size,
        "tail": tail
    }
    
    return results


def calculate_sample_size(alpha, power, delta, std_control=None, std_treatment=None, data=None):
    """
    计算所需样本量
    
    参数：
        alpha: 显著性水平
        power: 统计功效
        delta: 期望检测的效应量
        std_control: 对照组标准差
        std_treatment: 处理组标准差
        data: 可选，用于计算标准差的数据
    
    返回：
        sample_size: 每组所需样本量
    """
    # 如果没有提供标准差，尝试从数据中计算
    if std_control is None or std_treatment is None:
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                control_data, treatment_data = data
                std_control = np.std(control_data, ddof=1)
                std_treatment = np.std(treatment_data, ddof=1)
            else:
                print("数据格式错误，无法计算标准差")
                sys.exit(1)
        else:
            print("未提供标准差且没有数据用于计算标准差")
            sys.exit(1)
    
    # 计算合并标准差
    pooled_std = np.sqrt((std_control**2 + std_treatment**2) / 2)
    
    # 计算效应量（Cohen's d）
    effect_size = delta / pooled_std
    
    # 计算样本量
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,  # 两组样本量相等
        alternative="two-sided"
    )
    
    # 向上取整到整数
    sample_size = int(np.ceil(sample_size))
    
    print(f"样本量计算结果：")
    print(f"- 效应量 (delta/pooled_std): {effect_size:.4f}")
    print(f"- 合并标准差: {pooled_std:.4f}")
    print(f"- 每组所需样本量: {sample_size}")
    
    return sample_size


def generate_report(results, sample_size, alpha=0.05, output_file="ab_test_report.csv"):
    """
    生成A/B检验报告并保存到CSV文件
    
    参数：
        results: 统计检验结果
        sample_size: 计算得到的所需样本量
        alpha: 显著性水平
        output_file: 输出文件名
    """
    # 判断显著性
    is_significant = results["p_value"] < alpha
    significance_text = "显著" if is_significant else "不显著"
    
    # 生成报告数据
    report_data = {
        "检验类型": [results["test_name"]],
        "对照组均值": [results["control_mean"]],
        "处理组均值": [results["treatment_mean"]],
        "均值差异": [results["mean_diff"]],
        "对照组中位数": [results["control_median"]],
        "处理组中位数": [results["treatment_median"]],
        "中位数差异": [results["median_diff"]],
        "p值": [results["p_value"]],
        "95%置信区间下限": [results["ci_low"]],
        "95%置信区间上限": [results["ci_high"]],
        "效应量": [results["effect_size"]],
        "显著性判断(α={})" .format(alpha): [significance_text],
        "每组所需样本量": [sample_size]
    }
    
    # 保存为CSV文件
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    # 打印报告摘要
    print("\n===== A/B检验报告摘要 =====")
    print(f"检验类型: {results['test_name']}")
    print(f"对照组均值: {results['control_mean']:.4f}")
    print(f"处理组均值: {results['treatment_mean']:.4f}")
    print(f"均值差异: {results['mean_diff']:.4f} ({'+' if results['mean_diff'] > 0 else ''}{results['mean_diff']*100:.2f}%)")
    print(f"p值: {results['p_value']:.6f}")
    print(f"95%置信区间: [{results['ci_low']:.4f}, {results['ci_high']:.4f}]")
    print(f"显著性判断(α={alpha}): {significance_text}")
    print(f"每组所需样本量: {sample_size}")
    print(f"报告已保存至: {output_file}")
    
    # 多重比较校正建议
    print("\n===== 多重比较校正建议 =====")
    print("如果您进行了多次假设检验，请考虑使用以下方法控制I类错误：")
    print("1. Bonferroni校正：将α除以检验次数")
    print("2. Benjamini-Hochberg校正：控制错误发现率(FDR)")
    print("3. Holm-Bonferroni校正：比Bonferroni更强大的逐步校正方法")


def plot_results(control_data, treatment_data, output_file="ab_test_plot.png"):
    """
    绘制A/B检验结果可视化图表
    
    参数：
        control_data: 对照组数据
        treatment_data: 处理组数据
        output_file: 输出图表文件名
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制箱线图
    plt.subplot(1, 2, 1)
    plt.boxplot([control_data, treatment_data], labels=["对照组", "处理组"])
    plt.title("两组数据分布箱线图")
    plt.ylabel("分数")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制直方图
    plt.subplot(1, 2, 2)
    plt.hist(control_data, bins=30, alpha=0.5, label="对照组", color='blue')
    plt.hist(treatment_data, bins=30, alpha=0.5, label="处理组", color='green')
    plt.title("两组数据分布直方图")
    plt.xlabel("分数")
    plt.ylabel("频次")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"可视化图表已保存至: {output_file}")


def main():
    """
    主函数，处理命令行参数并执行A/B检验流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='A/B检验与样本量计算器')
    parser.add_argument('--alpha', type=float, default=0.05, help='显著性水平，默认0.05')
    parser.add_argument('--power', type=float, default=0.8, help='统计功效，默认0.8')
    parser.add_argument('--delta', type=float, default=0.02, help='期望检测的效应量，默认0.02')
    parser.add_argument('--test_type', type=str, default='t', choices=['t', 'nonparametric'], 
                        help='检验类型，t检验或非参数检验，默认t检验')
    parser.add_argument('--tail', type=str, default='two', choices=['two', 'left', 'right'], 
                        help='检验方向，双侧、左侧或右侧检验，默认双侧检验')
    parser.add_argument('--control_file', type=str, default='control.csv', help='对照组数据文件路径')
    parser.add_argument('--treatment_file', type=str, default='treatment.csv', help='处理组数据文件路径')
    parser.add_argument('--generate_data', action='store_true', help='生成示例数据')
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 依赖已在文件顶部检查并安装完成
    
    # 生成示例数据（如果需要）
    if args.generate_data or not (os.path.exists(args.control_file) and os.path.exists(args.treatment_file)):
        print("生成示例数据...")
        generate_sample_data()
    
    # 加载数据
    print("加载数据...")
    control_data, treatment_data = load_data(args.control_file, args.treatment_file)
    
    # 执行统计检验
    print(f"执行{args.test_type}检验（{args.tail}侧）...")
    results = perform_statistical_tests(control_data, treatment_data, args.test_type, args.tail)
    
    # 计算样本量
    print("计算所需样本量...")
    sample_size = calculate_sample_size(args.alpha, args.power, args.delta, data=(control_data, treatment_data))
    
    # 生成报告
    print("生成检验报告...")
    generate_report(results, sample_size, args.alpha)
    
    # 绘制可视化结果
    print("绘制可视化图表...")
    plot_results(control_data, treatment_data)
    
    # 记录结束时间并计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n程序执行完成，总耗时：{execution_time:.3f}秒")
    
    # 检查执行效率
    if execution_time > 10:
        print("警告：程序执行时间较长，可能需要优化。")
        print("建议优化方向：1) 减少bootstrap抽样次数；2) 优化数据加载和处理逻辑；3) 考虑使用更高效的统计方法。")


if __name__ == "__main__":
    main()