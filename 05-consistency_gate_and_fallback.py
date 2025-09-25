#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一致性阈值守门与回退

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2024-01-19

功能说明:
计算摘要与细节一致性，低于阈值回退到全文重排。

内容概述:
在双库检索基础上，计算"摘要句 ↔ 细节段"一致性均分；若低于阈值τ，触发回退，对全库段落重排保障召回稳定，输出一致性分、回退标志与最终候选。

使用场景:
- RAG系统中的检索质量保障
- 防止因摘要误匹配导致的细节检索偏差
- 通过回退机制提高检索的稳健性

依赖库:
- sklearn: 用于TF-IDF向量化和相似度计算
- jieba: 用于中文分词
- numpy: 用于数值计算
"""

# 自动安装依赖库
import subprocess
import sys
import time

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
import json


class ConsistencyGateAndFallback:
    """一致性阈值守门与回退实现类"""
    
    # 配置参数 - 可在顶部调整
    TAU = 0.85  # 提高阈值以更容易触发回退演示
    TOP_K_SECTIONS = 1  # 召回的章节数量
    TOP_K_DETAILS = 3   # 每个章节召回的细节段落数量
    TOP_K_FINAL = 5     # 最终输出的段落数量
    
    def __init__(self):
        """初始化一致性阈值守门与回退系统"""
        # 初始化数据结构
        self.summary_corpus = []  # 摘要句库
        self.detail_corpus = {}   # 细节段落库 {section_id: [paragraphs]}
        self.global_paragraphs = []  # 全局段落库，用于回退时的全文重排
        self.global_paragraph_info = []  # 全局段落信息
        self.section_info = []    # 章节信息 [(doc_id, section_id, anchor, summary_text)]
        self.vectorizer = None    # TF-IDF向量化器
        self.summary_matrix = None  # 摘要向量矩阵
        self.global_matrix = None   # 全局段落向量矩阵
        
        # 加载示例数据
        self._load_sample_data()
        # 初始化TF-IDF模型
        self._initialize_tfidf()
    
    def _load_sample_data(self):
        """加载示例数据：构建摘要句库、细节段落库和全局段落库"""
        # 示例文档数据 - 技术白皮书的部分章节
        self.section_info = [
            # (文档ID, 章节ID, 锚点, 摘要文本)
            ("docA", "1.1", "anc:sec-1-1", "RAG系统是检索增强生成技术的核心实现方案。"),
            ("docA", "1.2", "anc:sec-1-2", "检索模块负责从外部知识库获取相关文档片段。"),
            ("docA", "2.1", "anc:sec-2-1", "向量检索是基于语义相似度的检索方法。"),
            ("docA", "2.2", "anc:sec-2-2", "混合检索结合关键词匹配与语义相似性。"),
            ("docA", "3.1", "anc:sec-3-1", "文本分块是RAG系统中的重要预处理步骤。"),
            ("docA", "3.2", "anc:sec-3-2", "窗口滑动是常用的文本分块策略之一。"),
        ]
        
        # 构建摘要句库
        for _, _, _, summary in self.section_info:
            self.summary_corpus.append(summary)
        
        # 构建细节段落库和全局段落库
        self.detail_corpus = {
            "1.1": [  # 章节1.1的细节段落
                {"id": "p1", "anchor": "anc:p1-1-1", "content": "RAG系统通过检索外部知识库中的相关信息，并将这些信息与用户查询一起输入到生成模型中，生成更加准确、全面的回答。"},
                {"id": "p2", "anchor": "anc:p2-1-1", "content": "检索增强生成技术有效地弥补了大语言模型在特定领域知识和实时信息方面的不足。"},
                {"id": "p3", "anchor": "anc:p3-1-1", "content": "RAG系统的核心组件包括检索模块、知识库和生成模块。"}
            ],
            "1.2": [  # 章节1.2的细节段落
                {"id": "p1", "anchor": "anc:p1-1-2", "content": "检索模块根据用户查询从知识库中检索相关文档片段，这些片段包含回答问题所需的事实信息。"},
                {"id": "p2", "anchor": "anc:p2-1-2", "content": "检索的准确性直接影响生成回答的质量，因此检索策略的选择至关重要。"}
            ],
            "2.1": [  # 章节2.1的细节段落
                {"id": "p1", "anchor": "anc:p1-2-1", "content": "向量检索通过将文本转换为高维向量表示，利用余弦相似度等度量计算文本之间的语义相似性。"},
                {"id": "p2", "anchor": "anc:p2-2-1", "content": "常用的向量表示方法包括Word2Vec、GloVe和基于Transformer的预训练语言模型。"}
            ],
            "2.2": [  # 章节2.2的细节段落
                {"id": "p1", "anchor": "anc:p1-2-2", "content": "混合检索策略结合了关键词检索的精确性和向量检索的语义理解能力，能够更全面地捕获相关信息。"},
                {"id": "p2", "anchor": "anc:p2-2-2", "content": "关键词匹配通过精确的文本匹配快速定位相关文档，而语义相似性计算则捕获深层含义的关联。"},
                {"id": "p3", "anchor": "anc:p3-2-2", "content": "TF-IDF是一种常用的关键词检索方法，它根据词频和逆文档频率计算词的重要性。"},
                {"id": "p4", "anchor": "anc:p4-2-2", "content": "混合检索系统通常会结合多种检索结果的分数，生成最终的排序结果。"}
            ],
            "3.1": [  # 章节3.1的细节段落
                {"id": "p1", "anchor": "anc:p1-3-1", "content": "文本分块是将长文档分割成更小、更易于处理的片段的过程，这有助于提高检索的准确性和生成的质量。"},
                {"id": "p2", "anchor": "anc:p2-3-1", "content": "合适的分块大小需要根据具体任务和文档特点进行调整，过大或过小的块都可能影响性能。"}
            ],
            "3.2": [  # 章节3.2的细节段落
                {"id": "p1", "anchor": "anc:p1-3-2", "content": "窗口滑动策略通过固定大小的窗口和重叠部分，将文本划分为连续的块，保持了文本的局部连贯性。"},
                {"id": "p2", "anchor": "anc:p2-3-2", "content": "窗口大小和重叠比例是影响分块效果的两个关键参数，需要根据具体应用场景进行优化。"}
            ]
        }
        
        # 构建全局段落库和对应的信息
        for section_id, paragraphs in self.detail_corpus.items():
            for para in paragraphs:
                # 查找对应的文档ID
                doc_id = "docA"  # 示例中所有段落都来自同一文档
                
                # 添加到全局段落库
                self.global_paragraphs.append(para['content'])
                self.global_paragraph_info.append({
                    'doc_id': doc_id,
                    'section_id': section_id,
                    'paragraph_id': para['id'],
                    'anchor': para['anchor'],
                    'content': para['content']
                })
    
    def _tokenize_chinese(self, text):
        """中文分词处理"""
        # 使用jieba进行中文分词
        return ' '.join(jieba.cut(text))
    
    def _initialize_tfidf(self):
        """初始化TF-IDF向量化器"""
        # 创建中文TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_chinese,
            analyzer='word',
            lowercase=False,  # 中文不区分大小写
            max_features=5000
        )
        
        # 对摘要句库进行向量化
        self.summary_matrix = self.vectorizer.fit_transform(self.summary_corpus)
        
        # 对全局段落库进行向量化
        self.global_matrix = self.vectorizer.transform(self.global_paragraphs)
    
    def _search_summary(self, query):
        """在摘要库中检索相关章节"""
        # 对查询进行向量化
        query_vec = self.vectorizer.transform([query])
        
        # 计算查询与所有摘要的余弦相似度
        similarities = cosine_similarity(query_vec, self.summary_matrix).flatten()
        
        # 获取排序后的索引
        sorted_indices = similarities.argsort()[::-1][:self.TOP_K_SECTIONS]
        
        # 返回检索结果
        results = []
        for idx in sorted_indices:
            doc_id, section_id, anchor, summary = self.section_info[idx]
            results.append({
                'doc_id': doc_id,
                'section_id': section_id,
                'anchor': anchor,
                'summary': summary,
                'score': similarities[idx]
            })
        
        return results
    
    def _search_details(self, query, section_id):
        """在指定章节的细节库中检索相关段落"""
        if section_id not in self.detail_corpus:
            return []
        
        # 获取该章节的所有段落
        paragraphs = self.detail_corpus[section_id]
        
        # 提取段落文本
        paragraph_texts = [para['content'] for para in paragraphs]
        
        # 对段落进行向量化
        try:
            # 尝试使用已有的vectorizer进行转换
            paragraph_matrix = self.vectorizer.transform(paragraph_texts)
        except ValueError:
            # 如果有未见过的词汇，创建临时vectorizer
            temp_vectorizer = TfidfVectorizer(
                tokenizer=self._tokenize_chinese,
                analyzer='word',
                lowercase=False,
                vocabulary=self.vectorizer.vocabulary_  # 使用已有词汇表
            )
            paragraph_matrix = temp_vectorizer.fit_transform(paragraph_texts)
        
        # 对查询进行向量化
        query_vec = self.vectorizer.transform([query])
        
        # 计算查询与所有段落的余弦相似度
        similarities = cosine_similarity(query_vec, paragraph_matrix).flatten()
        
        # 获取排序后的索引
        sorted_indices = similarities.argsort()[::-1][:self.TOP_K_DETAILS]
        
        # 返回检索结果
        results = []
        for idx in sorted_indices:
            para = paragraphs[idx]
            results.append({
                'doc_id': "docA",  # 示例中所有段落都来自同一文档
                'section_id': section_id,
                'paragraph_id': para['id'],
                'anchor': para['anchor'],
                'content': para['content'],
                'score': similarities[idx]
            })
        
        return results
    
    def _compute_consistency(self, summary_text, detail_paragraphs):
        """计算摘要与细节段落之间的一致性得分"""
        if not detail_paragraphs:
            return 0.0
        
        # 对摘要进行向量化
        summary_vec = self.vectorizer.transform([summary_text])
        
        # 提取细节段落文本并向量化
        detail_texts = [para['content'] for para in detail_paragraphs]
        detail_matrix = self.vectorizer.transform(detail_texts)
        
        # 计算摘要与每个细节段落的余弦相似度
        similarities = cosine_similarity(summary_vec, detail_matrix).flatten()
        
        # 返回平均一致性得分
        return np.mean(similarities)
    
    def _fallback_to_full_text(self, query):
        """回退策略：对全局段落库进行重排"""
        # 对查询进行向量化
        query_vec = self.vectorizer.transform([query])
        
        # 计算查询与所有全局段落的余弦相似度
        similarities = cosine_similarity(query_vec, self.global_matrix).flatten()
        
        # 获取排序后的索引
        sorted_indices = similarities.argsort()[::-1][:self.TOP_K_FINAL]
        
        # 返回重排结果
        results = []
        for idx in sorted_indices:
            para_info = self.global_paragraph_info[idx].copy()
            para_info['score'] = similarities[idx]
            results.append(para_info)
        
        return results
    
    def consistency_gate_retrieval(self, query):
        """
        执行带一致性阈值守门的检索流程
        1. 在摘要库中检索相关章节
        2. 在检索到的章节的细节库中检索相关段落
        3. 计算摘要与细节的一致性得分
        4. 如果一致性得分低于阈值，触发回退策略
        5. 返回最终检索结果
        """
        start_time = time.time()
        
        # 第一阶段：摘要检索
        section_results = self._search_summary(query)
        
        # 初始化结果
        results = {
            'query': query,
            'sections': section_results,
            'details': [],
            'consistency_score': 0.0,
            'fallback_triggered': False,
            'final_results': [],
            'execution_time': 0.0
        }
        
        # 如果没有检索到章节，直接回退
        if not section_results:
            results['fallback_triggered'] = True
            results['final_results'] = self._fallback_to_full_text(query)
        else:
            # 第二阶段：细节检索
            section = section_results[0]  # 只处理top-1章节
            section_id = section['section_id']
            detail_results = self._search_details(query, section_id)
            results['details'] = detail_results
            
            # 计算一致性得分
            consistency_score = self._compute_consistency(section['summary'], detail_results)
            results['consistency_score'] = consistency_score
            
            # 检查是否需要回退
            if consistency_score < self.TAU:
                results['fallback_triggered'] = True
                results['final_results'] = self._fallback_to_full_text(query)
            else:
                # 不需要回退，使用细节检索结果作为最终结果
                results['final_results'] = detail_results[:self.TOP_K_FINAL]  # 限制最终结果数量
        
        end_time = time.time()
        results['execution_time'] = end_time - start_time
        
        return results
    
    def format_results(self, results):
        """格式化检索结果以便展示"""
        formatted_output = []
        
        # 输出查询
        formatted_output.append(f"查询: {results['query']}")
        formatted_output.append("=" * 60)
        
        # 输出章节检索结果
        if results['sections']:
            section = results['sections'][0]  # 只展示top-1章节
            consistency_str = f"Consistency={results['consistency_score']:.2f} < {self.TAU:.2f} → 回退" if results['fallback_triggered'] else f"Consistency={results['consistency_score']:.2f} ≥ {self.TAU:.2f} → 保留"
            formatted_output.append(f"Top section: ({section['doc_id']}, {section['section_id']}), {consistency_str}")
        
        # 输出最终结果
        formatted_output.append("Final:")
        for i, result in enumerate(results['final_results'], 1):
            # 截断长文本以便展示
            content_preview = result['content'][:50] + ('...' if len(result['content']) > 50 else '')
            formatted_output.append(f"  [{i}] ({result['doc_id']}, {result['section_id']}, {result['anchor']}) {result['score']:.2f} → \"{content_preview}\"")
        
        # 输出执行时间
        formatted_output.append(f"执行时间: {results['execution_time']:.4f}秒")
        
        return '\n'.join(formatted_output)
    
    def save_results(self, results, filename="consistency_gate_results.json"):
        """保存检索结果到文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📝 检索结果已保存至: {filename}")


# 示例查询 - 包含可能触发回退和不触发回退的情况
SAMPLE_QUERIES = [
    "混合检索策略的优势是什么？",  # 可能触发回退的查询
    "文本分块的方法有哪些？",       # 可能保留的查询
    "RAG系统的核心组件包括什么？",    # 可能保留的查询
    "什么是混合检索的最佳实践？"      # 新的查询，可能触发回退
]


def main():
    """主函数"""
    print("🚀 启动一致性阈值守门与回退工具")
    print(f"🔧 配置参数: TAU={ConsistencyGateAndFallback.TAU}, TOP_K_SECTIONS={ConsistencyGateAndFallback.TOP_K_SECTIONS}, ")
    print(f"              TOP_K_DETAILS={ConsistencyGateAndFallback.TOP_K_DETAILS}, TOP_K_FINAL={ConsistencyGateAndFallback.TOP_K_FINAL}")
    
    # 创建检索器实例
    retriever = ConsistencyGateAndFallback()
    
    # 执行示例查询
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n🔍 示例查询 {i}/{len(SAMPLE_QUERIES)}: {query}")
        
        # 执行带一致性阈值守门的检索
        results = retriever.consistency_gate_retrieval(query)
        
        # 格式化并打印结果
        formatted_results = retriever.format_results(results)
        print(formatted_results)
        
        # 保存结果到文件
        retriever.save_results(results, f"consistency_gate_results_{i}.json")
    
    # 检查中文输出
    print("\n🔍 中文输出测试：一致性阈值守门与回退系统成功实现摘要与细节一致性校验功能")
    
    print("\n✅ 程序执行完成")


if __name__ == "__main__":
    main()