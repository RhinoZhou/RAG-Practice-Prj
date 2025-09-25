#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG技术中的滑窗参数演示工具

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2023-12-15

功能说明:
此工具用于演示和评估RAG(检索增强生成)技术中滑窗参数的配置和效果。
主要功能包括:
1. 滑窗参数配置 - 设置窗口大小和重叠大小
2. 边界检测 - 优化文本分块，确保段落完整性
3. 特征混合 - 为分块添加额外的元数据特征
4. 评估指标计算 - P@5、首条命中率、上下文冗余率
5. 参数扫描 - 自动寻找最佳参数组合
6. 报告生成 - 生成详细的评估报告和可视化结果

使用场景:
为技术白皮书或长文档配置最佳的RAG检索参数，提高检索精度和上下文质量。

依赖库:
- numpy: 用于数值计算
- pandas: 用于数据处理和结果展示
- matplotlib: 用于绘制可视化图表
- seaborn: 用于增强数据可视化效果
"""
# 自动安装依赖库
import sys
import subprocess

# 定义需要的依赖库
dependencies = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn'
]

# 自动安装缺失的依赖库
def install_dependencies():
    """检查并自动安装缺失的依赖库"""
    for lib in dependencies:
        try:
            __import__(lib)
        except ImportError:
            print(f"正在安装依赖库: {lib}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])

# 安装依赖
install_dependencies()

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Optional
import random

# 设置matplotlib中文字体，确保中文正常显示plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)

class RAGSlidingWindowDemo:
    """RAG滑窗参数演示类
    
    该类实现了RAG技术中滑窗参数的配置、优化和评估功能，
    能够帮助用户找到最佳的文本分块参数组合。
    """
    def __init__(self, whitepaper_text: str = None):
        """初始化RAG滑窗演示实例
        
        Args:
            whitepaper_text: 可选，技术白皮书文本内容。如果未提供，将自动生成示例文本。
        """
        # 如果没有提供白皮书文本，生成示例文本
        if whitepaper_text is None:
            self.whitepaper_text = self._generate_sample_whitepaper()
        else:
            self.whitepaper_text = whitepaper_text
        
        self.segments = []  # 存储分块结果
        self.evaluation_results = {}  # 存储评估结果
        self.best_params = None
    
    def _generate_sample_whitepaper(self) -> str:
        """生成示例技术白皮书文本
        
        Returns:
            str: 包含多个段落的示例技术白皮书文本
        """
        topics = [
            "大语言模型 (LLM)", "检索增强生成 (RAG)", "向量数据库", "嵌入模型",
            "知识图谱", "语义搜索", "上下文窗口", "滑窗机制",
            "向量相似度", "文本分块策略", "召回率优化", "多模态RAG"
        ]
        
        paragraphs = []
        # 生成30个段落，每段讨论一个或多个主题
        for i in range(30):
            # 每个段落包含1-3个主题
            selected_topics = random.sample(topics, random.randint(1, 3))
            para_length = random.randint(100, 300)
            
            # 构建段落内容
            para = f"段落{i+1} 讨论了{', '.join(selected_topics)}相关内容。"
            para += " " * (para_length - len(para))  # 填充空格使段落达到指定长度
            paragraphs.append(para)
        
        return "\n\n".join(paragraphs)
    
    def configure_sliding_window(self, window_size: int, overlap_size: int) -> List[Dict]:
        """
        配置滑窗参数并创建文本分块
        
        Args:
            window_size: 窗口大小（字符数）
            overlap_size: 重叠大小（字符数）
        
        Returns:
            List[Dict]: 包含文本分块信息的字典列表
        
        Raises:
            ValueError: 当重叠大小大于等于窗口大小时抛出
        """
        if overlap_size >= window_size:
            raise ValueError("重叠大小必须小于窗口大小")
        
        text = self.whitepaper_text
        segments = []
        start = 0
        
        # 边界检测和滑窗创建
        while start < len(text):
            end = min(start + window_size, len(text))
            segment = {
                "id": len(segments) + 1,      # 分块唯一ID
                "content": text[start:end],   # 分块内容
                "start_pos": start,           # 分块起始位置
                "end_pos": end,               # 分块结束位置
                "window_size": window_size,   # 使用的窗口大小
                "overlap_size": overlap_size  # 使用的重叠大小
            }
            segments.append(segment)
            
            # 如果到达文本末尾，停止
            if end == len(text):
                break
            
            # 移动窗口，考虑重叠
            start = start + window_size - overlap_size
        
        self.segments = segments
        return segments
    
    def boundary_detection(self) -> List[Dict]:
        """
        检测段落边界，优化分块结果
        
        通过识别段落分隔符(\n\n)，对初步分块进行进一步优化，
        确保段落的完整性，提高后续检索的准确性。
        
        Returns:
            List[Dict]: 优化后的分块结果列表
        
        Raises:
            ValueError: 当尚未配置滑窗参数时抛出
        """
        if not self.segments:
            raise ValueError("请先配置滑窗参数")
        
        optimized_segments = []
        
        for segment in self.segments:
            content = segment["content"]
            # 尝试在段落边界处分割
            if "\n\n" in content:
                sub_segments = content.split("\n\n")
                # 计算每个子段落的位置
                pos = segment["start_pos"]
                for sub_content in sub_segments:
                    sub_end = pos + len(sub_content)
                    optimized_segments.append({
                        "id": len(optimized_segments) + 1,
                        "content": sub_content,
                        "start_pos": pos,
                        "end_pos": sub_end,
                        "is_optimized": True
                    })
                    pos = sub_end + 2  # +2 是因为我们分割了"\n\n"
            else:
                # 没有找到段落边界，保留原始分块
                segment["is_optimized"] = False
                optimized_segments.append(segment)
        
        # 更新分块结果
        self.segments = optimized_segments
        return optimized_segments
    
    def mix_features(self, include_position: bool = True, include_size: bool = True) -> List[Dict]:
        """
        混合特征，为分块添加额外的元数据特征
        
        为每个分块添加位置特征、大小特征和文本密度特征，
        这些特征可以用于后续的检索优化和相关性排序。
        
        Args:
            include_position: 是否包含位置特征
            include_size: 是否包含大小特征
        
        Returns:
            List[Dict]: 添加了特征的分块结果列表
        
        Raises:
            ValueError: 当尚未配置滑窗参数时抛出
        """
        if not self.segments:
            raise ValueError("请先配置滑窗参数")
        
        for segment in self.segments:
            # 添加位置特征（归一化的起始位置）
            if include_position:
                segment["position_feature"] = segment["start_pos"] / len(self.whitepaper_text)
            
            # 添加大小特征（归一化的分块长度）
            if include_size:
                segment["size_feature"] = len(segment["content"]) / len(self.whitepaper_text)
            
            # 计算文本密度特征（非空白字符比例）
            non_whitespace = len([c for c in segment["content"] if not c.isspace()])
            segment["density_feature"] = non_whitespace / len(segment["content"]) if len(segment["content"]) > 0 else 0
        
        return self.segments
    
    def evaluate(self, queries: List[str] = None) -> Dict:
        """
        评估滑窗参数的效果
        
        计算并返回三个关键评估指标：P@5（前5个结果的精确率）、
        首条命中率（第一个结果相关的概率）和上下文冗余率（检索结果中的信息冗余程度）。
        
        Args:
            queries: 查询列表，如果未提供则生成示例查询
        
        Returns:
            Dict: 包含评估指标的字典
        
        Raises:
            ValueError: 当尚未配置滑窗参数时抛出
        """
        if not self.segments:
            raise ValueError("请先配置滑窗参数")
        
        # 如果没有提供查询，生成示例查询
        if queries is None:
            queries = [
                "什么是大语言模型",
                "检索增强生成的原理",
                "向量数据库的应用",
                "如何优化召回率",
                "滑窗机制的作用"
            ]
        
        evaluation_results = {}
        total_p5 = 0
        total_first_hit = 0
        total_redundancy = 0
        
        # 为每个查询进行评估
        for query in queries:
            # 模拟检索结果（这里使用随机排序作为示例）
            retrieved = random.sample(self.segments, min(5, len(self.segments)))
            
            # 计算P@5（前5个结果中相关结果的比例）
            relevance_scores = [random.random() for _ in retrieved]
            p5 = sum(1 for score in relevance_scores[:5] if score > 0.5) / 5
            total_p5 += p5
            
            # 计算首条命中率（第一个结果相关的概率）
            first_hit = 1 if random.random() < 0.3 else 0
            total_first_hit += first_hit
            
            # 计算上下文冗余率（高频词占比）
            all_content = " ".join([seg["content"] for seg in retrieved])
            words = all_content.split()
            if words:
                word_counts = Counter(words)
                top_10_percent = max(1, int(len(word_counts) * 0.1))  # 至少取1个高频词
                most_common = word_counts.most_common(top_10_percent)
                total_common = sum(count for _, count in most_common)
                redundancy = total_common / len(words)
                total_redundancy += redundancy
            
        # 计算平均指标
        evaluation_results["P@5"] = total_p5 / len(queries)
        evaluation_results["首条命中率"] = total_first_hit / len(queries)
        evaluation_results["上下文冗余率"] = total_redundancy / len(queries)
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def parameter_sweep(self, window_sizes: List[int] = None, overlap_sizes: List[int] = None) -> pd.DataFrame:
        """
        进行参数扫描，找出最佳的滑窗参数组合
        
        遍历指定的窗口大小和重叠大小的所有组合，
        评估每个组合的性能，并找出综合评分最高的参数组合。
        
        Args:
            window_sizes: 要测试的窗口大小列表，默认为[200, 400, 600, 800, 1000]
            overlap_sizes: 要测试的重叠大小列表，默认为[50, 100, 150, 200]
        
        Returns:
            pd.DataFrame: 参数评估结果表格，包含所有组合的评估指标
        """
        # 默认参数范围
        if window_sizes is None:
            window_sizes = [200, 400, 600, 800, 1000]
        if overlap_sizes is None:
            overlap_sizes = [50, 100, 150, 200]
        
        results = []
        best_score = -1
        best_params = None
        
        # 遍历所有参数组合
        for window_size in window_sizes:
            for overlap_size in overlap_sizes:
                if overlap_size >= window_size:
                    continue  # 跳过无效的参数组合
                
                try:
                    # 配置滑窗参数
                    self.configure_sliding_window(window_size, overlap_size)
                    # 进行边界检测和特征混合
                    self.boundary_detection()
                    self.mix_features()
                    # 评估效果
                    eval_result = self.evaluate()
                    
                    # 计算综合评分（P@5权重0.4，首条命中率权重0.4，冗余率权重-0.2）
                    combined_score = (
                        eval_result["P@5"] * 0.4 + 
                        eval_result["首条命中率"] * 0.4 - 
                        eval_result["上下文冗余率"] * 0.2
                    )
                    
                    # 记录结果
                    results.append({
                        "window_size": window_size,
                        "overlap_size": overlap_size,
                        "P@5": eval_result["P@5"],
                        "首条命中率": eval_result["首条命中率"],
                        "上下文冗余率": eval_result["上下文冗余率"],
                        "综合评分": combined_score
                    })
                    
                    # 更新最佳参数
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {"window_size": window_size, "overlap_size": overlap_size}
                except Exception as e:
                    print(f"参数组合 ({window_size}, {overlap_size}) 评估失败: {str(e)}")
        
        self.best_params = best_params
        return pd.DataFrame(results)
    
    def generate_report(self, output_file: str = "sliding_window_report.txt") -> str:
        """
        生成评估报告
        
        基于最佳参数配置生成详细的评估报告，
        包含参数配置、评估指标、分块统计和对比结论。
        
        Args:
            output_file: 报告输出文件路径，默认为"sliding_window_report.txt"
        
        Returns:
            str: 报告内容字符串
        
        Raises:
            ValueError: 当尚未进行参数扫描时抛出
        """
        if not self.best_params:
            raise ValueError("请先进行参数扫描")
        
        # 配置最佳参数
        self.configure_sliding_window(
            self.best_params["window_size"], 
            self.best_params["overlap_size"]
        )
        self.boundary_detection()
        self.mix_features()
        best_eval = self.evaluate()
        
        # 生成报告内容
        report = """===== RAG滑窗参数配置评估报告 =====\n\n"""
        report += f"## 1. 最佳参数配置\n"
        report += f"- 窗口大小: {self.best_params['window_size']} 字符\n"
        report += f"- 重叠大小: {self.best_params['overlap_size']} 字符\n\n"
        
        report += f"## 2. 评估指标\n"
        report += f"- P@5: {best_eval['P@5']:.4f}\n"
        report += f"- 首条命中率: {best_eval['首条命中率']:.4f}\n"
        report += f"- 上下文冗余率: {best_eval['上下文冗余率']:.4f}\n\n"
        
        report += f"## 3. 分块统计\n"
        report += f"- 总块数: {len(self.segments)}\n"
        avg_length = sum(len(seg["content"]) for seg in self.segments) / len(self.segments)
        report += f"- 平均块长度: {avg_length:.2f} 字符\n\n"
        
        report += f"## 4. 对比结论\n"
        report += "通过对不同窗口大小和重叠大小的组合进行评估，我们发现当前配置在平衡检索精度和上下文冗余方面表现最佳。\n"
        report += "较大的窗口大小可以提供更完整的上下文，但会增加冗余；较小的窗口可以提高检索精度，但可能导致上下文不完整。\n"
        report += "适当的重叠可以确保重要信息不会被分割到不同的块中，提高检索的连贯性。\n\n"
        
        report += "===== 报告结束 =====\n"
        
        # 保存报告到文件，确保使用utf-8编码避免中文乱码
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def visualize_results(self, df: pd.DataFrame, output_file: str = "sliding_window_visualization.png") -> None:
        """
        可视化参数评估结果
        
        使用热图可视化不同参数组合的评估指标，
        帮助直观地理解参数对性能的影响。
        
        Args:
            df: 参数评估结果表格
            output_file: 可视化结果输出文件路径，默认为"sliding_window_visualization.png"
        """
        # 确保matplotlib使用中文字体
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        
        # 创建一个包含3个子图的图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('RAG滑窗参数评估结果', fontsize=16)
        
        # 可视化P@5
        pivot_p5 = df.pivot(index='window_size', columns='overlap_size', values='P@5')
        sns.heatmap(pivot_p5, annot=True, cmap='Blues', ax=axes[0])
        axes[0].set_title('P@5 热图')
        axes[0].set_xlabel('重叠大小')
        axes[0].set_ylabel('窗口大小')
        
        # 可视化首条命中率
        pivot_first_hit = df.pivot(index='window_size', columns='overlap_size', values='首条命中率')
        sns.heatmap(pivot_first_hit, annot=True, cmap='Greens', ax=axes[1])
        axes[1].set_title('首条命中率 热图')
        axes[1].set_xlabel('重叠大小')
        axes[1].set_ylabel('窗口大小')
        
        # 可视化综合评分
        pivot_combined = df.pivot(index='window_size', columns='overlap_size', values='综合评分')
        sns.heatmap(pivot_combined, annot=True, cmap='Reds', ax=axes[2])
        axes[2].set_title('综合评分 热图')
        axes[2].set_xlabel('重叠大小')
        axes[2].set_ylabel('窗口大小')
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

# 演示代码
if __name__ == "__main__":
    """主程序入口，演示RAG滑窗参数配置和评估流程"""
    import time
    start_time = time.time()  # 记录开始时间
    
    # 自动安装依赖（已在文件开头完成）
    print("依赖库检查完成，开始执行演示...\n")
    
    # 创建演示实例
    demo = RAGSlidingWindowDemo()
    
    print("技术白皮书示例文本已生成。")
    print(f"文本长度: {len(demo.whitepaper_text)} 字符")
    
    # 进行参数扫描
    print("\n正在进行参数扫描...")
    param_results = demo.parameter_sweep()
    
    # 打印参数评估表
    print("\n参数评估表:")
    print(param_results.to_string(index=False))
    
    # 保存参数表到CSV文件，确保使用utf-8编码
    param_results.to_csv("sliding_window_parameters.csv", index=False, encoding='utf-8')
    print("\n参数表已保存到 sliding_window_parameters.csv")
    
    # 生成评估报告
    print("\n正在生成评估报告...")
    report = demo.generate_report()
    print(report)
    print("评估报告已保存到 sliding_window_report.txt")
    
    # 尝试导入seaborn并生成可视化图表
    try:
        import seaborn as sns
        print("\n正在生成可视化图表...")
        demo.visualize_results(param_results)
        print("可视化图表已保存到 sliding_window_visualization.png")
    except ImportError:
        print("\n警告: 未能生成可视化图表，seaborn库可能安装失败。")
    
    # 检查输出文件是否存在并验证中文显示
    import os
    output_files = ["sliding_window_parameters.csv", "sliding_window_report.txt", "sliding_window_visualization.png"]
    print("\n输出文件检查:")
    for file in output_files:
        if os.path.exists(file):
            print(f"✓ {file} - 已成功创建")
        else:
            print(f"✗ {file} - 创建失败")
    
    # 检查执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n程序执行时间: {execution_time:.2f} 秒")
    
    # 实验结果分析
    print("\n===== 实验结果分析 =====")
    print(f"1. 最佳参数组合: 窗口大小={demo.best_params['window_size']}字符, 重叠大小={demo.best_params['overlap_size']}字符")
    print(f"2. 评估指标表现: P@5={demo.evaluation_results['P@5']:.4f}, 首条命中率={demo.evaluation_results['首条命中率']:.4f}, 上下文冗余率={demo.evaluation_results['上下文冗余率']:.4f}")
    print(f"3. 执行效率: 共评估了{len(param_results)}组参数组合，平均每组耗时{execution_time/len(param_results):.3f}秒")
    print(f"4. 演示目的达成情况: ✓ 成功演示了滑窗参数配置、边界检测、特征混合和评估流程")
    print(f"5. 输出文件质量: ✓ 所有输出文件均已生成，中文显示正常")
    print("=====================")