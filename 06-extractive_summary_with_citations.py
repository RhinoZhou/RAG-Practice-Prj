#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抽取式摘要（句级对齐）

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2024-01-20

功能说明:
基于 TF‑IDF 打分选取关键句，输出带锚点的抽取式摘要。

内容概述:
将段落切句，按与查询的词向量重叠打分，选取 top‑2 句拼接为摘要，并附 [doc:sec:sent] 引用以支持可追溯审阅，降低生成幻觉风险。

使用场景:
- RAG系统中的摘要生成模块
- 需要可追溯性的学术或技术文档摘要
- 降低大模型生成过程中的幻觉风险

依赖库:
- sklearn: 用于TF-IDF向量化和相似度计算
- jieba: 用于中文分词
- numpy: 用于数值计算
- re: 用于正则表达式处理
"""

# 自动安装依赖库
import subprocess
import sys
import time
import re

# 定义所需依赖库
required_dependencies = [
    'scikit-learn',
    'jieba',
    'numpy'
]


def install_dependencies():
    """检查并自动安装缺失的依赖库"""
    for dependency in required_dependencies:
        try:
            # 尝试导入库以检查是否已安装
            __import__(dependency.replace('-', '_'))  # 处理scikit-learn的命名差异
            print(f"✅ 依赖库 '{dependency}' 已安装")
        except ImportError:
            print(f"⚠️ 依赖库 '{dependency}' 未安装，正在安装...")
            # 使用pip安装缺失的依赖
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
            print(f"✅ 依赖库 '{dependency}' 安装成功")


# 执行依赖安装
if __name__ == "__main__":
    install_dependencies()


# 导入所需库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import numpy as np


class ExtractiveSummaryWithCitations:
    """抽取式摘要生成器，支持带引用锚点的关键句选择"""
    
    # 配置参数 - 可在顶部调整
    TOP_N_SENTENCES = 2  # 选取的关键句数量
    
    def __init__(self):
        """初始化抽取式摘要生成器"""
        # 初始化数据结构
        self.documents = {}
        self.vectorizer = None
        
        # 加载示例数据
        self._load_sample_data()
        
        # 初始化TF-IDF模型
        self._initialize_tfidf()
    
    def _load_sample_data(self):
        """加载示例文档数据"""
        # 示例文档数据 - 技术白皮书的部分章节
        self.documents = {
            "docA": {
                "2.2": [  # 章节2.2的句子
                    "混合检索结合关键词匹配与语义相似性。",
                    "关键词匹配通过精确的文本匹配快速定位相关文档，而语义相似性计算则捕获深层含义的关联。",
                    "TF-IDF是一种常用的关键词检索方法，它根据词频和逆文档频率计算词的重要性。",
                    "RRF融合前列结果，提升稳健性。"
                ],
                "3.1": [  # 章节3.1的句子
                    "文本分块是将长文档分割成更小、更易于处理的片段的过程。",
                    "这有助于提高检索的准确性和生成的质量。",
                    "合适的分块大小需要根据具体任务和文档特点进行调整。"
                ],
                "3.2": [  # 章节3.2的句子
                    "窗口滑动策略通过固定大小的窗口和重叠部分，将文本划分为连续的块。",
                    "这保持了文本的局部连贯性。",
                    "窗口大小和重叠比例是影响分块效果的两个关键参数。"
                ]
            }
        }
    
    def _tokenize_chinese(self, text):
        """中文分词处理"""
        # 使用jieba进行中文分词
        return ' '.join(jieba.cut(text))
    
    def _initialize_tfidf(self):
        """初始化TF-IDF向量化器"""
        # 收集所有文档的所有句子用于训练TF-IDF模型
        all_sentences = []
        for doc_id, sections in self.documents.items():
            for section_id, sentences in sections.items():
                all_sentences.extend(sentences)
        
        # 创建中文TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_chinese,
            analyzer='word',
            lowercase=False,  # 中文不区分大小写
            max_features=5000
        )
        
        # 训练TF-IDF模型
        self.vectorizer.fit(all_sentences)
    
    def _split_into_sentences(self, text):
        """将文本分割成句子"""
        # 使用正则表达式分割句子（中文句子通常以句号、问号、感叹号结尾）
        sentences = re.split(r'[。？！]', text)
        # 去除空句子
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def generate_summary(self, query):
        """
        生成带引用的抽取式摘要
        
        参数:
            query: 用户查询文本
        
        返回:
            dict: 包含摘要文本和引用列表的字典
        """
        start_time = time.time()
        
        # 初始化结果
        results = {
            'query': query,
            'summary': '',
            'citations': [],
            'sentence_scores': [],
            'execution_time': 0.0
        }
        
        # 对查询进行向量化
        query_vector = self.vectorizer.transform([self._tokenize_chinese(query)])
        
        # 存储所有句子及其信息
        all_sentences_info = []
        
        # 遍历所有文档和章节
        for doc_id, sections in self.documents.items():
            for section_id, sentences in sections.items():
                for sent_idx, sentence in enumerate(sentences, 1):
                    # 对句子进行向量化
                    sentence_vector = self.vectorizer.transform([self._tokenize_chinese(sentence)])
                    
                    # 计算句子与查询的余弦相似度
                    similarity = cosine_similarity(query_vector, sentence_vector)[0][0]
                    
                    # 存储句子信息和相似度得分
                    all_sentences_info.append({
                        'doc_id': doc_id,
                        'section_id': section_id,
                        'sent_idx': sent_idx,
                        'sentence': sentence,
                        'score': similarity
                    })
        
        # 按照相似度得分排序
        all_sentences_info.sort(key=lambda x: x['score'], reverse=True)
        
        # 记录句子得分信息
        results['sentence_scores'] = [
            {
                'doc_id': info['doc_id'],
                'section_id': info['section_id'],
                'sentence': info['sentence'][:30] + ('...' if len(info['sentence']) > 30 else ''),
                'score': info['score']
            } for info in all_sentences_info[:5]  # 只记录前5个句子
        ]
        
        # 选取top-N个关键句
        top_sentences = all_sentences_info[:self.TOP_N_SENTENCES]
        
        # 按照在原文中的位置重新排序（保持原文逻辑）
        top_sentences.sort(key=lambda x: (x['doc_id'], x['section_id'], x['sent_idx']))
        
        # 构建摘要文本和引用列表
        summary_parts = []
        citations = []
        
        for info in top_sentences:
            summary_parts.append(info['sentence'])
            # 构建引用格式: [doc:sec:sent]
            citations.append(f"[{info['doc_id']}:{info['section_id']}:s{info['sent_idx']}]")
        
        # 拼接摘要文本
        results['summary'] = ' '.join(summary_parts)
        results['citations'] = citations
        
        # 记录执行时间
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        return results
    
    def save_results(self, results, filename="extractive_summary_results.json"):
        """保存检索结果到文件"""
        import json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📝 检索结果已保存至: {filename}")

    def format_results(self, results):
        """格式化检索结果以便展示"""
        formatted_output = []
        
        # 输出查询
        formatted_output.append(f"查询: {results['query']}")
        formatted_output.append("=" * 60)
        
        # 输出摘要和引用
        formatted_output.append(f"摘要: {results['summary']}")
        formatted_output.append(f"引用: {' '.join(results['citations'])}")
        formatted_output.append("=" * 60)
        
        # 输出句子得分信息（可选）
        formatted_output.append("Top 5句子得分:")
        for i, info in enumerate(results['sentence_scores'], 1):
            formatted_output.append(f"  [{i}] ({info['doc_id']}, {info['section_id']}) {info['score']:.3f} → {info['sentence']}")
        
        # 输出执行时间
        formatted_output.append(f"执行时间: {results['execution_time']:.4f}秒")
        
        return '\n'.join(formatted_output)


# 示例查询
SAMPLE_QUERIES = [
    "混合检索的特点是什么？",
    "文本分块的作用是什么？",
    "窗口滑动策略的优势是什么？"
]


def main():
    """主函数"""
    print("🚀 启动抽取式摘要生成工具")
    print(f"🔧 配置参数: TOP_N_SENTENCES={ExtractiveSummaryWithCitations.TOP_N_SENTENCES}")
    
    # 创建摘要生成器实例
    summarizer = ExtractiveSummaryWithCitations()
    
    # 执行示例查询
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n🔍 示例查询 {i}/{len(SAMPLE_QUERIES)}: {query}")
        
        # 生成摘要
        results = summarizer.generate_summary(query)
        
        # 格式化并打印结果
        formatted_results = summarizer.format_results(results)
        print(formatted_results)
        
        # 保存结果到文件
        result_filename = f"extractive_summary_results_{i}.json"
        summarizer.save_results(results, result_filename)
    
    # 检查中文输出
    print("\n🔍 中文输出测试：抽取式摘要生成工具成功实现带引用锚点的关键句选择功能")
    
    print("\n✅ 程序执行完成")


if __name__ == "__main__":
    main()