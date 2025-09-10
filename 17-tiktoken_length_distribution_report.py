#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tiktoken 精准计数与分布报告生成器

功能：
- 逐段统计文本的token长度分布
- 输出P50/P90/P99等百分位数统计
- 识别并报告长尾样本（超长段落）
- 处理异常字符和表情符号
- 生成JSON和CSV格式的统计报告

用法示例：
```python
from tiktoken_length_distribution_report import TiktokenLengthDistributionReport

# 创建报告生成器实例
reporter = TiktokenLengthDistributionReport(tokenizer_name="cl100k_base")

# 处理文本列表
texts = ["段落1内容...", "段落2内容...", ...]
results = reporter.analyze_texts(texts)

# 或者处理JSON格式的块列表
chunks = [{"text": "块1内容"}, {"text": "块2内容"}, ...]
results = reporter.analyze_chunks(chunks)
```
"""

import json
import csv
import re
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import time

# 尝试导入tiktoken库
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    print("警告: tiktoken库未安装，将使用简单的字符计数作为替代。")
    print("建议安装tiktoken以获得更准确的token计数: pip install tiktoken")


@dataclass
class TokenStatistics:
    """存储单个文本段的token统计信息"""
    text: str  # 原始文本
    token_count: int  # token数量
    char_count: int  # 字符数量
    has_emoji: bool = False  # 是否包含表情符号
    has_special_chars: bool = False  # 是否包含特殊字符
    issues: List[str] = field(default_factory=list)  # 编码问题列表


@dataclass
class DistributionReport:
    """存储整体token长度分布报告"""
    total_segments: int  # 总段数
    total_tokens: int  # 总token数
    total_chars: int  # 总字符数
    avg_tokens_per_segment: float  # 每段平均token数
    p50: int  # 50%分位数
    p90: int  # 90%分位数
    p99: int  # 99%分位数
    max_tokens: int  # 最大token数
    min_tokens: int  # 最小token数
    long_tail_segments: List[Dict[str, Any]] = field(default_factory=list)  # 长尾样本列表
    distribution_data: Dict[str, List[float]] = field(default_factory=dict)  # 分布数据
    processing_time: float = 0.0  # 处理时间（秒）
    tokenizer_used: str = ""  # 使用的tokenizer
    timestamp: str = ""  # 处理时间戳


class SimpleCharacterTokenizer:
    """简单的字符tokenizer实现，用于在tiktoken不可用时作为替代"""
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为字符级别token"""
        return list(text)
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenize(text))


class TiktokenLengthDistributionReport:
    """Tiktoken长度分布报告生成器"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base", use_simple_tokenizer: bool = False, 
                 long_tail_threshold: float = 0.95, max_long_tail_samples: int = 20):
        """初始化报告生成器
        
        参数:
            tokenizer_name: 使用的tiktoken编码名称，默认为cl100k_base（GPT-4使用的编码）
            use_simple_tokenizer: 是否强制使用简单的字符tokenizer
            long_tail_threshold: 长尾样本的阈值百分比（默认95%，即只保留长度在前5%的样本）
            max_long_tail_samples: 最多保留的长尾样本数量
        """
        self.tokenizer_name = tokenizer_name
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        self.long_tail_threshold = long_tail_threshold
        self.max_long_tail_samples = max_long_tail_samples
        self.tokenizer = self._load_tokenizer()
        
        # 用于检测表情符号和特殊字符的正则表达式
        self.emoji_pattern = re.compile(r'[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF\\U0001F1E0-\\U0001F1FF]', 
                                       re.UNICODE)
        self.special_char_pattern = re.compile(r'[^\w\s\u4e00-\u9fa5，。！？；：,.!?;:"\'\-]')
    
    def _load_tokenizer(self):
        """加载指定的tokenizer"""
        if self.use_simple_tokenizer:
            print("使用简单的字符tokenizer进行计数")
            return SimpleCharacterTokenizer()
        
        if not HAS_TIKTOKEN:
            print("tiktoken库不可用，自动使用简单的字符tokenizer")
            return SimpleCharacterTokenizer()
        
        try:
            print(f"加载tiktoken编码: {self.tokenizer_name}")
            return tiktoken.get_encoding(self.tokenizer_name)
        except Exception as e:
            print(f"加载tiktoken失败: {e}")
            print("切换到简单的字符tokenizer")
            return SimpleCharacterTokenizer()
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        参数:
            text: 输入文本
            
        返回:
            token数量
        """
        if isinstance(self.tokenizer, SimpleCharacterTokenizer):
            return self.tokenizer.count_tokens(text)
        else:
            try:
                # 使用tiktoken的encode方法计算token数量，忽略不允许的特殊字符
                return len(self.tokenizer.encode(text, disallowed_special=()))
            except Exception as e:
                print(f"编码错误: {e}，使用字符计数作为替代")
                return len(text)
    
    def _detect_issues(self, text: str) -> Tuple[bool, bool, List[str]]:
        """检测文本中的表情符号、特殊字符和编码问题
        
        参数:
            text: 输入文本
            
        返回:
            (是否包含表情符号, 是否包含特殊字符, 问题列表)
        """
        has_emoji = bool(self.emoji_pattern.search(text))
        has_special_chars = bool(self.special_char_pattern.search(text))
        issues = []
        
        if has_emoji:
            issues.append("包含表情符号")
        
        if has_special_chars:
            issues.append("包含特殊字符")
        
        # 检测可能的编码问题
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            issues.append("存在编码问题")
        
        return has_emoji, has_special_chars, issues
    
    def _analyze_single_text(self, text: str, segment_id: int = 0) -> TokenStatistics:
        """分析单个文本段的token统计信息
        
        参数:
            text: 输入文本段
            segment_id: 段ID
            
        返回:
            TokenStatistics对象
        """
        # 清理文本（去除多余的空白字符）
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # 计算token数量和字符数量
        token_count = self._count_tokens(cleaned_text)
        char_count = len(cleaned_text)
        
        # 检测问题
        has_emoji, has_special_chars, issues = self._detect_issues(cleaned_text)
        
        return TokenStatistics(
            text=cleaned_text,
            token_count=token_count,
            char_count=char_count,
            has_emoji=has_emoji,
            has_special_chars=has_special_chars,
            issues=issues
        )
    
    def analyze_texts(self, texts: List[str]) -> Dict[str, Any]:
        """分析文本列表的token长度分布
        
        参数:
            texts: 文本段列表
            
        返回:
            包含统计报告的字典
        """
        start_time = time.time()
        
        # 分析每个文本段
        all_statistics = []
        token_counts = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue  # 跳过空文本
            
            stats = self._analyze_single_text(text, i)
            all_statistics.append(stats)
            token_counts.append(stats.token_count)
        
        # 生成分布报告
        report = self._generate_report(all_statistics, token_counts)
        
        # 计算处理时间
        report.processing_time = time.time() - start_time
        report.tokenizer_used = "tiktoken (" + self.tokenizer_name + ")" if HAS_TIKTOKEN and not self.use_simple_tokenizer else "简单字符计数"
        report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "metadata": {
                "total_segments": report.total_segments,
                "total_tokens": report.total_tokens,
                "total_chars": report.total_chars,
                "avg_tokens_per_segment": report.avg_tokens_per_segment,
                "p50": report.p50,
                "p90": report.p90,
                "p99": report.p99,
                "max_tokens": report.max_tokens,
                "min_tokens": report.min_tokens,
                "tokenizer_used": report.tokenizer_used,
                "processing_time": report.processing_time,
                "timestamp": report.timestamp
            },
            "distribution_data": report.distribution_data,
            "long_tail_segments": report.long_tail_segments,
            "detailed_statistics": [{
                "text": stat.text,
                "token_count": stat.token_count,
                "char_count": stat.char_count,
                "has_emoji": stat.has_emoji,
                "has_special_chars": stat.has_special_chars,
                "issues": stat.issues
            } for stat in all_statistics]
        }
    
    def analyze_chunks(self, chunks: List[Dict[str, Any]], text_key: str = "text") -> Dict[str, Any]:
        """分析块列表的token长度分布
        
        参数:
            chunks: 块列表，每个块是包含text字段的字典
            text_key: 块中包含文本的键名
            
        返回:
            包含统计报告的字典
        """
        # 提取文本列表
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict) and text_key in chunk and chunk[text_key]:
                texts.append(chunk[text_key])
        
        # 调用analyze_texts进行分析
        return self.analyze_texts(texts)
    
    def _generate_report(self, all_statistics: List[TokenStatistics], token_counts: List[int]) -> DistributionReport:
        """生成token长度分布报告
        
        参数:
            all_statistics: 所有文本段的统计信息列表
            token_counts: 所有文本段的token数量列表
            
        返回:
            DistributionReport对象
        """
        if not token_counts:
            return DistributionReport(
                total_segments=0,
                total_tokens=0,
                total_chars=0,
                avg_tokens_per_segment=0,
                p50=0,
                p90=0,
                p99=0,
                max_tokens=0,
                min_tokens=0
            )
        
        # 计算基本统计信息
        total_segments = len(all_statistics)
        total_tokens = sum(token_counts)
        total_chars = sum(stat.char_count for stat in all_statistics)
        avg_tokens_per_segment = total_tokens / total_segments if total_segments > 0 else 0
        
        # 计算百分位数
        p50 = int(np.percentile(token_counts, 50))
        p90 = int(np.percentile(token_counts, 90))
        p99 = int(np.percentile(token_counts, 99))
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        
        # 生成分布数据（按token数量分组）
        distribution_data = self._generate_distribution_data(token_counts)
        
        # 识别长尾样本
        long_tail_segments = self._identify_long_tail_segments(all_statistics, token_counts)
        
        return DistributionReport(
            total_segments=total_segments,
            total_tokens=total_tokens,
            total_chars=total_chars,
            avg_tokens_per_segment=avg_tokens_per_segment,
            p50=p50,
            p90=p90,
            p99=p99,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            distribution_data=distribution_data,
            long_tail_segments=long_tail_segments
        )
    
    def _generate_distribution_data(self, token_counts: List[int]) -> Dict[str, List[float]]:
        """生成token数量分布数据
        
        参数:
            token_counts: token数量列表
            
        返回:
            包含分布数据的字典，包括区间、计数、百分比和累积百分比
        """
        # 创建等宽的bin：根据最大token数，最多创建20个bin，确保bin宽度至少为1
        max_count = max(token_counts)
        bin_width = max(1, int(max_count / 20))  # 最多20个bin
        bins = list(range(0, max_count + bin_width, bin_width))
        
        # 使用numpy的histogram函数计算直方图
        # hist: 每个bin中的样本数量
        # bin_edges: 各个bin的边界值
        hist, bin_edges = np.histogram(token_counts, bins=bins)
        
        # 将计数转换为百分比，便于直观理解分布情况
        total = len(token_counts)
        hist_percent = [count / total * 100 for count in hist]
        
        # 计算累积分布，用于确定长尾样本的位置
        cum_hist = np.cumsum(hist_percent)
        
        # 返回格式化的分布数据，便于后续展示和保存
        return {
            "bins": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],  # token区间
            "counts": hist.tolist(),  # 每个区间的样本数量
            "percentages": hist_percent,  # 每个区间的样本百分比
            "cumulative_percentages": cum_hist.tolist()  # 累积百分比
        }
    
    def _identify_long_tail_segments(self, all_statistics: List[TokenStatistics], 
                                   token_counts: List[int]) -> List[Dict[str, Any]]:
        """识别长尾样本（超长段落）
        
        参数:
            all_statistics: 所有文本段的统计信息列表
            token_counts: 所有文本段的token数量列表
            
        返回:
            长尾样本列表，包含索引、token数、特征和文本预览
        """
        if not token_counts:
            return []
        
        # 计算长尾阈值：根据配置的阈值百分比确定
        threshold = np.percentile(token_counts, self.long_tail_threshold * 100)
        
        # 筛选出超过阈值的样本，这些样本属于分布的长尾部分
        long_tail_indices = [i for i, count in enumerate(token_counts) if count >= threshold]
        
        # 按token数量降序排序并取前N个，便于后续分析和展示
        
        # 构建结果列表，包含每个长尾样本的关键信息
        long_tail_segments = []
        for i in long_tail_indices:
            stat = all_statistics[i]
            # 为每个长尾样本创建详细信息字典，包含索引、长度、特征和文本预览
            long_tail_segments.append({
                "segment_index": i,  # 样本在原列表中的索引
                "token_count": stat.token_count,  # token数量
                "char_count": stat.char_count,  # 字符数量
                "has_emoji": stat.has_emoji,  # 是否包含表情符号
                "has_special_chars": stat.has_special_chars,  # 是否包含特殊字符
                "issues": stat.issues,  # 检测到的问题
                "text_preview": stat.text[:200] + ("..." if len(stat.text) > 200 else "")  # 文本预览（最多200字符）
            })
        
        return long_tail_segments
    
    def save_report_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        """将统计报告保存为JSON文件
        
        参数:
            results: 分析结果字典
            output_path: 输出文件路径
        """
        # 确保输出目录存在，如果不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存JSON文件，使用ensure_ascii=False保留中文字符，indent=2格式化输出
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"统计报告已保存至: {output_path}")
    
    def save_report_to_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """将统计报告保存为CSV文件
        
        参数:
            results: 分析结果字典
            output_path: 输出文件路径
        """
        # 确保输出目录存在，如果不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 检查是否有详细统计数据，这是生成CSV报告的必要条件
        if "detailed_statistics" not in results:
            print("错误: 结果中没有详细统计数据")
            return
        
        # 保存CSV文件，使用带有BOM的UTF-8编码以解决Excel中文乱码问题
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头，定义每一列的名称
            writer.writerow(["segment_index", "token_count", "char_count", "has_emoji", 
                            "has_special_chars", "issues", "text_preview"])
            
            # 写入数据，为每个文本段创建一行数据
            for i, stat in enumerate(results["detailed_statistics"]):
                writer.writerow([
                    i,  # 段索引
                    stat["token_count"],  # token数量
                    stat["char_count"],  # 字符数量
                    "是" if stat["has_emoji"] else "否",  # 转换为中文输出
                    "是" if stat["has_special_chars"] else "否",  # 转换为中文输出
                    ",".join(stat["issues"]) if stat["issues"] else "无",  # 问题列表
                    stat["text"][:200] + ("..." if len(stat["text"]) > 200 else "")  # 文本预览
                ])
        
        print(f"CSV报告已保存至: {output_path}")
    
    def print_summary_report(self, results: Dict[str, Any]) -> None:
        """打印统计报告摘要，在控制台输出关键统计信息
        
        参数:
            results: 分析结果字典
        """
        metadata = results.get("metadata", {})
        
        print("=" * 80)
        print("Tiktoken 长度分布统计报告摘要")
        print("=" * 80)
        print(f"总段数: {metadata.get('total_segments', 0)}")
        print(f"总token数: {metadata.get('total_tokens', 0)}")
        print(f"总字符数: {metadata.get('total_chars', 0)}")
        print(f"每段平均token数: {metadata.get('avg_tokens_per_segment', 0):.2f}")
        print(f"\n百分位数统计:")
        print(f"  P50 (中位数): {metadata.get('p50', 0)} tokens")
        print(f"  P90: {metadata.get('p90', 0)} tokens")
        print(f"  P99: {metadata.get('p99', 0)} tokens")
        print(f"\n极值统计:")
        print(f"  最大token数: {metadata.get('max_tokens', 0)} tokens")
        print(f"  最小token数: {metadata.get('min_tokens', 0)} tokens")
        print(f"\n处理信息:")
        print(f"  Tokenizer: {metadata.get('tokenizer_used', '')}")
        print(f"  处理时间: {metadata.get('processing_time', 0):.3f} 秒")
        print(f"  处理时间戳: {metadata.get('timestamp', '')}")
        
        # 打印长尾样本信息：只显示前5个，其余用省略表示
        long_tail_segments = results.get("long_tail_segments", [])
        if long_tail_segments:
            print(f"\n长尾样本 ({len(long_tail_segments)} 个):")
            print("=" * 80)
            # 只打印前5个长尾样本，避免输出过多
            for i, segment in enumerate(long_tail_segments[:5]):
                print(f"[{i+1}] Token数: {segment['token_count']}, 字符数: {segment['char_count']}")
                print(f"  问题: {', '.join(segment.get('issues', [])) or '无'}")
                print(f"  文本预览: {segment['text_preview']}")
                print("-" * 80)
            
            # 如果有更多长尾样本，提示用户
            if len(long_tail_segments) > 5:
                print(f"... 还有 {len(long_tail_segments) - 5} 个长尾样本未显示")
        
        # 打印分布数据：显示token长度在不同区间的分布情况
        distribution_data = results.get("distribution_data", {})
        if distribution_data:
            print(f"\nToken长度分布:")
            print("区间        数量    百分比    累积百分比")
            print("-" * 80)
            bins = distribution_data.get("bins", [])
            counts = distribution_data.get("counts", [])
            percentages = distribution_data.get("percentages", [])
            cum_percentages = distribution_data.get("cumulative_percentages", [])
            
            # 只打印前10个区间，避免输出过多
            for i in range(min(10, len(bins))):
                print(f"{bins[i]:<10} {counts[i]:<6} {percentages[i]:<8.2f}% {cum_percentages[i]:<8.2f}%")
            
            # 如果有更多区间，提示用户
            if len(bins) > 10:
                print(f"... 还有 {len(bins) - 10} 个区间未显示")
        
        print("=" * 80)


# 主函数示例
if __name__ == "__main__":
    """主函数，展示如何使用TiktokenLengthDistributionReport类"""
    
    # 创建报告生成器实例
    reporter = TiktokenLengthDistributionReport(
        tokenizer_name="cl100k_base",  # GPT-4使用的tokenizer
        long_tail_threshold=0.9,  # 只保留长度在前10%的样本
        max_long_tail_samples=10  # 最多保留10个长尾样本
    )
    
    # 示例1: 从corpus.txt文件加载文本
    corpus_path = "corpus.txt"
    if os.path.exists(corpus_path):
        print(f"从文件加载文本: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按段落分割文本（假设段落由两个或多个换行符分隔）
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        print(f"分割得到 {len(paragraphs)} 个段落")
        
        # 分析段落
        results = reporter.analyze_texts(paragraphs)
        
        # 打印摘要报告
        reporter.print_summary_report(results)
        
        # 保存报告
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        reporter.save_report_to_json(results, os.path.join(output_dir, "tiktoken_length_distribution_report.json"))
        reporter.save_report_to_csv(results, os.path.join(output_dir, "tiktoken_length_distribution_report.csv"))
        
    else:
        print(f"未找到文件: {corpus_path}")
        
        # 示例2: 使用示例文本
        sample_texts = [
            "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的智能机器。",
            "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习而不需要明确编程。",
            "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人类大脑的某些功能。",
            "自然语言处理（NLP）是人工智能和语言学的交叉领域，旨在让计算机理解、解释和生成人类语言。😍",
            "计算机视觉是人工智能的另一个重要分支，它使计算机能够从图像和视频中获取信息。#计算机视觉#",
        ]
        
        print("使用示例文本进行分析")
        results = reporter.analyze_texts(sample_texts)
        reporter.print_summary_report(results)
        
        # 保存示例报告
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        reporter.save_report_to_json(results, os.path.join(output_dir, "tiktoken_length_distribution_report_example.json"))
        reporter.save_report_to_csv(results, os.path.join(output_dir, "tiktoken_length_distribution_report_example.csv"))
        
    print("分析完成！")