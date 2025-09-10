#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指代链与SRL守护边界分块器结果分析工具

该脚本用于分析14-coref_srl_boundary_guard.py生成的分块结果，评估边界守护机制的有效性。
主要功能包括：
1. 加载分块结果和边界调整记录
2. 分析分块的基本统计信息
3. 评估边界调整的效果
4. 检查分块的语义连贯性
5. 生成可视化报告
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class CorefSRLChunkAnalyzer:
    """指代链与SRL守护边界分块器结果分析器"""
    def __init__(self, results_dir: str = "results"):
        """初始化分析器"""
        self.results_dir = results_dir
        self.chunks_data: Optional[Dict] = None
        self.chunks: List[Dict] = []
        self.adjustments: List[Dict] = []
        self.output_dir = os.path.join(results_dir, "analysis")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self, file_name: str = "coref_srl_guard_chunks.json") -> bool:
        """加载分块结果文件"""
        file_path = os.path.join(self.results_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.chunks_data = json.load(f)
                
                # 提取分块和调整记录
                if "chunks" in self.chunks_data:
                    self.chunks = self.chunks_data["chunks"]
                if "adjustments" in self.chunks_data:
                    self.adjustments = self.chunks_data["adjustments"]
                
            print(f"成功加载分块结果: {file_path}")
            print(f"总共有 {len(self.chunks)} 个分块")
            print(f"总共有 {len(self.adjustments)} 个边界调整记录")
            return True
        except Exception as e:
            print(f"加载分块结果时出错: {e}")
            return False
    
    def load_original_chunks(self, file_name: str = "topic_cluster_chunks.json") -> Optional[List[Dict]]:
        """加载原始分块结果用于比较"""
        file_path = os.path.join(self.results_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "chunks" in data:
                    return data["chunks"]
                elif isinstance(data, list):
                    return data
                else:
                    print(f"原始分块文件格式不符合预期: {file_path}")
                    return None
        except Exception as e:
            print(f"加载原始分块结果时出错: {e}")
            return None
    
    def compute_basic_statistics(self) -> Dict[str, Any]:
        """计算分块的基本统计信息"""
        if not self.chunks:
            return {}
        
        # 计算token数量统计
        tokens_counts = [chunk.get('tokens_count', 0) for chunk in self.chunks]
        total_tokens = sum(tokens_counts)
        avg_tokens = total_tokens / len(tokens_counts) if tokens_counts else 0
        min_tokens = min(tokens_counts) if tokens_counts else 0
        max_tokens = max(tokens_counts) if tokens_counts else 0
        
        # 计算重叠统计
        overlap_with_prev = [chunk.get('overlap_with_prev', 0) for chunk in self.chunks]
        overlap_with_next = [chunk.get('overlap_with_next', 0) for chunk in self.chunks]
        total_overlap = sum(overlap_with_prev) + sum(overlap_with_next)
        avg_overlap_prev = sum(overlap_with_prev) / len(overlap_with_prev) if overlap_with_prev else 0
        avg_overlap_next = sum(overlap_with_next) / len(overlap_with_next) if overlap_with_next else 0
        
        # 计算调整统计
        adjustment_count = len(self.adjustments)
        move_adjustments = sum(1 for adj in self.adjustments if adj.get('adjustment_type') == 'move')
        overlap_adjustments = sum(1 for adj in self.adjustments if adj.get('adjustment_type') == 'overlap')
        
        stats = {
            "total_chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "median_tokens": np.median(tokens_counts) if tokens_counts else 0,
            "std_tokens": np.std(tokens_counts) if tokens_counts else 0,
            "total_overlap": total_overlap,
            "avg_overlap_prev": avg_overlap_prev,
            "avg_overlap_next": avg_overlap_next,
            "adjustment_count": adjustment_count,
            "move_adjustments": move_adjustments,
            "overlap_adjustments": overlap_adjustments
        }
        
        return stats
    
    def analyze_token_distribution(self) -> None:
        """分析分块的token数量分布"""
        if not self.chunks:
            return
        
        tokens_counts = [chunk.get('tokens_count', 0) for chunk in self.chunks]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(tokens_counts, bins=50, alpha=0.7, color='blue')
        plt.title('分块Token数量分布')
        plt.xlabel('Token数量')
        plt.ylabel('分块数量')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(tokens_counts, vert=False)
        plt.title('分块Token数量箱线图')
        plt.xlabel('Token数量')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "token_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"分块Token数量分布图已保存至: {output_path}")
        
        plt.close()
    
    def analyze_overlap_distribution(self) -> None:
        """分析分块重叠分布"""
        if not self.chunks:
            return
        
        overlap_with_prev = [chunk.get('overlap_with_prev', 0) for chunk in self.chunks]
        overlap_with_next = [chunk.get('overlap_with_next', 0) for chunk in self.chunks]
        
        # 过滤掉0重叠的分块以更好地显示分布
        non_zero_prev = [o for o in overlap_with_prev if o > 0]
        non_zero_next = [o for o in overlap_with_next if o > 0]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 前向重叠直方图
        plt.subplot(1, 2, 1)
        if non_zero_prev:
            plt.hist(non_zero_prev, bins=20, alpha=0.7, color='green')
        plt.title('前向重叠分布（非零值）')
        plt.xlabel('重叠字符数')
        plt.ylabel('分块数量')
        plt.grid(True, alpha=0.3)
        
        # 后向重叠直方图
        plt.subplot(1, 2, 2)
        if non_zero_next:
            plt.hist(non_zero_next, bins=20, alpha=0.7, color='orange')
        plt.title('后向重叠分布（非零值）')
        plt.xlabel('重叠字符数')
        plt.ylabel('分块数量')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "overlap_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"分块重叠分布图已保存至: {output_path}")
        
        plt.close()
    
    def analyze_adjustments(self) -> None:
        """分析边界调整情况"""
        if not self.adjustments:
            print("没有边界调整记录可供分析")
            return
        
        # 统计调整类型
        adjustment_types = [adj.get('adjustment_type') for adj in self.adjustments]
        type_counts = Counter(adjustment_types)
        
        # 统计调整原因
        reasons = [adj.get('reason') for adj in self.adjustments]
        reason_counts = Counter(reasons)
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 调整类型饼图
        plt.subplot(2, 2, 1)
        if type_counts:
            plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('边界调整类型分布')
        
        # 调整位置分布
        plt.subplot(2, 2, 2)
        boundary_indices = [adj.get('boundary_index') for adj in self.adjustments]
        if boundary_indices:
            plt.hist(boundary_indices, bins=min(20, len(boundary_indices)), alpha=0.7, color='purple')
            plt.title('边界调整位置分布')
            plt.xlabel('边界索引')
            plt.ylabel('调整数量')
            plt.grid(True, alpha=0.3)
        
        # 调整距离分布（仅移动类型）
        plt.subplot(2, 2, 3)
        move_adjustments = [adj for adj in self.adjustments if adj.get('adjustment_type') == 'move']
        if move_adjustments:
            distances = [abs(adj.get('new_position', 0) - adj.get('original_position', 0)) for adj in move_adjustments]
            plt.hist(distances, bins=15, alpha=0.7, color='red')
            plt.title('边界移动距离分布')
            plt.xlabel('移动距离（字符）')
            plt.ylabel('调整数量')
            plt.grid(True, alpha=0.3)
        
        # 涉及的指代和SRL论元数量
        plt.subplot(2, 2, 4)
        coref_counts = [adj.get('coref_count', 0) for adj in self.adjustments]
        srl_counts = [adj.get('srl_arg_count', 0) for adj in self.adjustments]
        
        if coref_counts or srl_counts:
            x = np.arange(len(self.adjustments))
            width = 0.35
            
            plt.bar(x - width/2, coref_counts, width, label='指代数量')
            plt.bar(x + width/2, srl_counts, width, label='SRL论元数量')
            plt.xlabel('调整索引')
            plt.ylabel('数量')
            plt.title('每次调整涉及的指代和SRL论元数量')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "adjustments_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"边界调整分析图已保存至: {output_path}")
        
        plt.close()
        
        # 打印调整原因统计
        print("\n边界调整原因统计:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}次")
    
    def compare_with_original(self) -> None:
        """与原始分块结果进行比较"""
        original_chunks = self.load_original_chunks()
        if not original_chunks or not self.chunks:
            print("无法进行分块比较")
            return
        
        # 计算原始分块的统计信息
        original_tokens = [chunk.get('tokens_count', 0) for chunk in original_chunks]
        original_total = sum(original_tokens)
        original_avg = original_total / len(original_tokens) if original_tokens else 0
        
        # 计算当前分块的统计信息
        current_tokens = [chunk.get('tokens_count', 0) for chunk in self.chunks]
        current_total = sum(current_tokens)
        current_avg = current_total / len(current_tokens) if current_tokens else 0
        
        # 创建比较图表
        plt.figure(figsize=(12, 6))
        
        # Token数量比较
        plt.subplot(1, 2, 1)
        labels = ['原始分块', '边界守护后']
        totals = [original_total, current_total]
        averages = [original_avg, current_avg]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, totals, width, label='总Token数')
        plt.bar(x + width/2, averages, width, label='平均分块Token数')
        plt.xlabel('分块类型')
        plt.ylabel('数量')
        plt.title('分块Token数量比较')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 分块数量比较
        plt.subplot(1, 2, 2)
        chunk_counts = [len(original_chunks), len(self.chunks)]
        
        plt.bar(labels, chunk_counts, alpha=0.7, color=['blue', 'green'])
        plt.xlabel('分块类型')
        plt.ylabel('分块数量')
        plt.title('分块数量比较')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "comparison_with_original.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"与原始分块比较图已保存至: {output_path}")
        
        plt.close()
        
        # 打印比较结果
        print("\n与原始分块比较结果:")
        print(f"  原始分块数量: {len(original_chunks)}, 守护后分块数量: {len(self.chunks)}")
        print(f"  原始总Token数: {original_total}, 守护后总Token数: {current_total}")
        print(f"  原始平均分块Token数: {original_avg:.2f}, 守护后平均分块Token数: {current_avg:.2f}")
    
    def check_chunk_quality(self) -> List[Dict[str, Any]]:
        """检查分块质量，返回可能存在问题的分块"""
        if not self.chunks:
            return []
        
        problematic_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            issues = []
            
            # 检查Token数量是否合理
            tokens_count = chunk.get('tokens_count', 0)
            if tokens_count < 10:
                issues.append(f'Token数量过少 ({tokens_count})')
            elif tokens_count > 500:
                issues.append(f'Token数量过多 ({tokens_count})')
            
            # 检查重叠是否合理
            overlap_prev = chunk.get('overlap_with_prev', 0)
            overlap_next = chunk.get('overlap_with_next', 0)
            if overlap_prev > 200:
                issues.append(f'与前一分块重叠过大 ({overlap_prev}字符)')
            if overlap_next > 200:
                issues.append(f'与后一分块重叠过大 ({overlap_next}字符)')
            
            # 检查文本内容（简单检查）
            text = chunk.get('text', '')
            if len(text.strip()) == 0:
                issues.append('分块内容为空')
            elif len(text) < 10:
                issues.append(f'分块内容过短 ({len(text)}字符)')
            
            # 检查边界位置是否合理（如果有前后分块）
            start_idx = chunk.get('start_index', 0)
            end_idx = chunk.get('end_index', 0)
            
            if i > 0:
                prev_chunk = self.chunks[i-1]
                prev_end = prev_chunk.get('end_index', 0)
                if start_idx < prev_end - overlap_prev:
                    issues.append('与前一分块边界不连续')
            
            if i < len(self.chunks) - 1:
                next_chunk = self.chunks[i+1]
                next_start = next_chunk.get('start_index', 0)
                if end_idx > next_start + overlap_next:
                    issues.append('与后一分块边界不连续')
            
            # 如果有问题，记录该分块
            if issues:
                problematic_chunks.append({
                    'chunk_id': i,
                    'tokens_count': tokens_count,
                    'issues': issues
                })
        
        # 打印问题分块统计
        print(f"\n质量检查结果:")
        print(f"  总检查分块数: {len(self.chunks)}")
        print(f"  发现问题分块数: {len(problematic_chunks)}")
        
        if problematic_chunks:
            print("  问题类型统计:")
            all_issues = []
            for chunk in problematic_chunks:
                all_issues.extend(chunk['issues'])
            issue_counts = Counter(all_issues)
            for issue, count in issue_counts.most_common():
                print(f"    {issue}: {count}个分块")
        
        return problematic_chunks
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成完整的分析报告"""
        # 计算基本统计信息
        basic_stats = self.compute_basic_statistics()
        
        # 分析Token分布
        self.analyze_token_distribution()
        
        # 分析重叠分布
        self.analyze_overlap_distribution()
        
        # 分析边界调整
        self.analyze_adjustments()
        
        # 与原始分块比较
        self.compare_with_original()
        
        # 检查分块质量
        problematic_chunks = self.check_chunk_quality()
        
        # 生成报告
        report = {
            "basic_statistics": basic_stats,
            "adjustment_analysis": {
                "total_adjustments": len(self.adjustments),
                "has_adjustments": len(self.adjustments) > 0
            },
            "quality_check": {
                "problematic_chunks_count": len(problematic_chunks),
                "problematic_chunks_percentage": (len(problematic_chunks) / len(self.chunks) * 100) if self.chunks else 0
            },
            "visualizations": {
                "token_distribution": os.path.join(self.output_dir, "token_distribution.png"),
                "overlap_distribution": os.path.join(self.output_dir, "overlap_distribution.png"),
                "adjustments_analysis": os.path.join(self.output_dir, "adjustments_analysis.png"),
                "comparison_with_original": os.path.join(self.output_dir, "comparison_with_original.png")
            }
        }
        
        # 保存报告为JSON
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析报告已保存至: {report_path}")
        
        # 打印关键发现
        print("\n关键发现:")
        print(f"1. 分块总数: {basic_stats.get('total_chunks')}")
        print(f"2. 总Token数: {basic_stats.get('total_tokens')}")
        print(f"3. 平均分块Token数: {basic_stats.get('avg_tokens_per_chunk', 0):.2f}")
        print(f"4. 边界调整数量: {basic_stats.get('adjustment_count')}")
        print(f"5. 问题分块比例: {report['quality_check']['problematic_chunks_percentage']:.2f}%")
        
        # 生成评估结论
        self._generate_evaluation_conclusion(report)
        
        return report
    
    def _generate_evaluation_conclusion(self, report: Dict[str, Any]) -> None:
        """生成评估结论"""
        print("\n评估结论:")
        
        # 基于基本统计的结论
        avg_tokens = report['basic_statistics'].get('avg_tokens_per_chunk', 0)
        adjustment_count = report['basic_statistics'].get('adjustment_count', 0)
        problematic_percentage = report['quality_check']['problematic_chunks_percentage']
        
        if avg_tokens >= 30 and avg_tokens <= 150:
            print("- 分块大小整体合理，平均Token数在建议范围内")
        elif avg_tokens < 30:
            print("- 分块大小普遍偏小，可能影响上下文完整性")
        else:
            print("- 分块大小普遍偏大，可能影响检索精度")
        
        if adjustment_count > 0:
            print(f"- 成功识别并调整了 {adjustment_count} 个可能存在语义断裂的边界")
            print("  指代链与SRL守护机制发挥了作用")
        else:
            print("- 未发现需要调整的边界，可能原因是:")
            print("  1. 初始分块质量较高，已经很好地保持了语义连贯性")
            print("  2. 指代与SRL检测算法在当前文本中表现不够敏感")
        
        if problematic_percentage < 5:
            print("- 分块质量良好，问题分块比例较低")
        elif problematic_percentage < 15:
            print("- 分块质量一般，存在少量需要关注的问题分块")
        else:
            print("- 分块质量需要改进，存在较多问题分块")
        
        # 生成使用建议
        print("\n使用建议:")
        
        if adjustment_count == 0:
            print("1. 考虑调整指代链和SRL检测的阈值参数，以提高敏感度")
            print("2. 对于复杂文本，可能需要更高级的指代消解和SRL模型")
        
        if avg_tokens < 30 or avg_tokens > 150:
            print("3. 根据具体应用场景，调整初始分块的大小参数")
            print("  - 对于需要更多上下文的任务，适当增大分块大小")
            print("  - 对于需要更高检索精度的任务，适当减小分块大小")
        
        if problematic_percentage > 5:
            print("4. 检查并优化分块边界的确定算法，特别是对于特殊格式文本")
        
        print("5. 在实际应用中，建议结合人工评估和下游任务表现进一步优化参数")

# 主函数
def main():
    """主函数，用于运行指代链与SRL守护边界分块器结果分析"""
    # 创建分析器实例
    analyzer = CorefSRLChunkAnalyzer()
    
    # 加载分块结果
    if not analyzer.load_results():
        print("无法加载分块结果，程序退出")
        return
    
    # 生成分析报告
    print("\n开始分析指代链与SRL守护边界分块结果...")
    report = analyzer.generate_summary_report()
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()