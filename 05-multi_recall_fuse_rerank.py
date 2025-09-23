#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF两阶段检索与融合重排演示工具

功能说明：
此程序实现了一种简化版的两阶段检索策略，首先使用倒排索引进行初步检索，
然后对结果进行融合重排，融合公式为w1*BM25 + w2*SEM，其中w1和w2是可调节的权重参数。
程序将展示不同权重组合下的排序结果差异，帮助理解权重对最终排序的影响。

执行流程：
1. 初始化依赖检查与安装
2. 创建示例文档集合
3. 构建倒排索引和文档长度信息
4. 模拟查询处理
5. 计算BM25分数和模拟语义分数
6. 使用不同权重组合进行线性融合排序
7. 输出不同权重下的Top-K结果对比

作者：Ph.D. Rhino
"""

import re
import sys
import math
import subprocess
import importlib.util
import time
from collections import defaultdict, Counter


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    本程序主要使用Python标准库，但为了演示依赖检查功能，我们仍保留此函数
    """
    required_packages = []  # 目前此程序仅使用标准库，无需外部依赖
    missing_packages = []
    
    # 检查所需的包是否已安装
    for package in required_packages:
        try:
            # 使用importlib代替废弃的pkg_resources
            importlib.util.find_spec(package)
        except ModuleNotFoundError:
            missing_packages.append(package)
    
    # 安装缺失的包
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            print("依赖包安装成功!")
        except subprocess.CalledProcessError as e:
            print(f"依赖包安装失败: {e}")
            sys.exit(1)
    else:
        print("所有依赖包已安装完成。")


class IVFTwoStageDemo:
    """IVF两阶段检索与融合重排演示主类"""
    
    def __init__(self):
        # 初始化示例文档集合
        self.documents = self._initialize_documents()
        
        # 构建倒排索引和文档长度信息
        self.inverted_index = defaultdict(set)
        self.doc_lengths = {}
        self._build_inverted_index()
        
        # BM25参数
        self.k1 = 1.2  # 词频饱和参数
        self.b = 0.75  # 文档长度归一化参数
        self.avg_doc_len = self._calculate_avg_doc_length()
        
        # 文档总数
        self.total_docs = len(self.documents)
    
    def _initialize_documents(self):
        """初始化示例文档集合"""
        return [
            {"id": "d1", "title": "人工智能在医疗领域的应用", "content": "人工智能技术正在改变医疗行业的诊断和治疗方式，提高医疗效率和准确性。"},
            {"id": "d2", "title": "机器学习算法详解", "content": "本文详细介绍了各种机器学习算法的原理、应用场景和优缺点。"},
            {"id": "d3", "title": "医疗人工智能的伦理问题", "content": "随着人工智能在医疗领域的广泛应用，相关的伦理问题也日益凸显。"},
            {"id": "d4", "title": "深度学习在自然语言处理中的应用", "content": "深度学习技术极大地推动了自然语言处理领域的发展和进步。"},
            {"id": "d5", "title": "人工智能与医疗数据隐私保护", "content": "如何在利用人工智能技术的同时，保护患者的医疗数据隐私是一个重要挑战。"}
        ]
    
    def _tokenize(self, text):
        """简单的文本分词函数"""
        # 移除标点符号，转换为小写，然后按空格分割
        # 使用字符类来匹配常见标点符号，因为Python的re模块不支持\p{P}
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\s]+', ' ', text.lower())
        return text.split()
    
    def _build_inverted_index(self):
        """构建倒排索引和计算文档长度"""
        for doc in self.documents:
            doc_id = doc['id']
            # 合并标题和内容进行索引
            text = doc['title'] + ' ' + doc['content']
            tokens = self._tokenize(text)
            
            # 记录文档长度
            self.doc_lengths[doc_id] = len(tokens)
            
            # 更新倒排索引
            for token in set(tokens):  # 使用set去重，只记录词是否出现
                self.inverted_index[token].add(doc_id)
    
    def _calculate_avg_doc_length(self):
        """计算平均文档长度"""
        if not self.doc_lengths:
            return 0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def _calculate_bm25_score(self, query, doc_id):
        """
        计算简化的BM25分数
        参数: query - 查询文本
              doc_id - 文档ID
        返回: float - BM25分数
        """
        score = 0.0
        tokens = self._tokenize(query)
        
        # 计算每个查询词的BM25分数并累加
        for token in tokens:
            # 如果查询词不在倒排索引中，仍然计算分数但权重较低
            # 这是为了演示目的，使结果更有意义
            
            # 计算文档频率 (DF) - 改进：即使词不在索引中也给出一个基础DF值
            if token in self.inverted_index:
                df = len(self.inverted_index[token])
            else:
                df = self.total_docs  # 不在索引中的词，假设文档频率为总文档数
            
            # 计算逆文档频率 (IDF)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # 计算词频 (TF) - 改进：更精确地计算词频
            doc = next(d for d in self.documents if d['id'] == doc_id)
            text = doc['title'] + ' ' + doc['content']
            token_list = self._tokenize(text)
            token_count = Counter(token_list)
            tf = token_count.get(token, 0)
            
            # 为了演示目的，即使词不在文档中也给出一个基础TF值
            # 在实际应用中这可能不适合，但这里有助于展示融合效果
            if tf == 0 and (token in doc['title'].lower() or token in doc['content'].lower()):
                tf = 0.5  # 给部分匹配一个低的词频
            
            # 计算BM25分数
            doc_len = self.doc_lengths.get(doc_id, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator
        
        # 为了演示目的，确保分数不为0
        if score == 0.0:
            score = 0.1  # 给一个最小基础分
        
        return round(score, 2)
    
    def _simulate_semantic_score(self, query, doc_id):
        """
        模拟语义相似度分数
        在实际应用中，这里应该使用预训练的语言模型计算真实的语义相似度
        参数: query - 查询文本
              doc_id - 文档ID
        返回: float - 语义相似度分数
        """
        # 简单的模拟逻辑：基于预定义的语义相关性
        semantic_relations = {
            "d1": 0.9,  # 与"人工智能"和"医疗"高度相关
            "d2": 0.6,  # 与"人工智能"相关但主题不同
            "d3": 0.85, # 与"人工智能"和"医疗"高度相关
            "d4": 0.5,  # 与"人工智能"相关但主题不同
            "d5": 0.8   # 与"人工智能"和"医疗"相关
        }
        
        # 添加一些随机噪声来模拟真实世界的不确定性
        import random
        noise = random.uniform(-0.05, 0.05)
        base_score = semantic_relations.get(doc_id, 0.0)
        
        # 确保分数在0-1范围内
        return round(max(0.0, min(1.0, base_score + noise)), 2)
    
    def _fuse_scores(self, bm25_scores, semantic_scores, w1=0.7, w2=0.3):
        """
        融合BM25分数和语义分数
        参数: bm25_scores - BM25分数字典
              semantic_scores - 语义分数字典
              w1 - BM25分数的权重
              w2 - 语义分数的权重
        返回: list - 包含(文档ID, 融合分数)的列表，按分数降序排列
        """
        fused_scores = []
        
        # 确保权重归一化
        total_weight = w1 + w2
        w1_normalized = w1 / total_weight
        w2_normalized = w2 / total_weight
        
        # 对每个文档计算融合分数
        for doc_id in bm25_scores:
            # 线性融合
            fused_score = w1_normalized * bm25_scores[doc_id] + w2_normalized * semantic_scores[doc_id]
            fused_scores.append((doc_id, round(fused_score, 2)))
        
        # 按分数降序排序
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        return fused_scores
    
    def search_and_rerank(self, query, top_k=3, weight_pairs=None):
        """
        执行搜索和重排操作
        参数: query - 查询文本
              top_k - 返回的结果数量
              weight_pairs - 权重组合列表
        返回: dict - 不同权重下的排序结果
        """
        if weight_pairs is None:
            # 默认权重组合
            weight_pairs = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
        
        # 初步检索：使用倒排索引找出包含查询词的文档
        query_tokens = self._tokenize(query)
        relevant_docs = set()
        
        for token in query_tokens:
            if token in self.inverted_index:
                relevant_docs.update(self.inverted_index[token])
        
        # 如果没有相关文档，返回所有文档
        if not relevant_docs:
            relevant_docs = {doc['id'] for doc in self.documents}
        
        # 计算BM25分数和语义分数
        bm25_scores = {}
        semantic_scores = {}
        
        for doc_id in relevant_docs:
            bm25_scores[doc_id] = self._calculate_bm25_score(query, doc_id)
            semantic_scores[doc_id] = self._simulate_semantic_score(query, doc_id)
        
        # 使用不同权重组合进行融合重排
        results = {}
        
        for w1, w2 in weight_pairs:
            fused_scores = self._fuse_scores(bm25_scores, semantic_scores, w1, w2)
            results[(w1, w2)] = fused_scores[:top_k]  # 取Top-K结果
        
        return results, bm25_scores, semantic_scores
    
    def run_demo(self, query="人工智能医疗应用", top_k=3):
        """
        运行演示程序
        参数: query - 查询文本
              top_k - 返回的结果数量
        """
        # 记录开始时间，用于性能分析
        start_time = time.time()
        
        print(f"查询: {query}")
        
        # 执行搜索和重排
        results, bm25_scores, semantic_scores = self.search_and_rerank(query, top_k)
        
        # 打印原始分数（用于参考）
        print("\n原始分数:")
        print(f"BM25分数: {sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)}")
        print(f"语义分数: {sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)}")
        
        # 打印不同权重下的融合结果
        print("\n不同权重下的融合排序结果:")
        for (w1, w2), ranked_docs in results.items():
            print(f"w=({w1:.1f},{w2:.1f}): {ranked_docs}")
        
        # 计算执行时间
        execution_time = time.time() - start_time
        print(f"\n执行时间: {execution_time:.4f}秒")
        
        # 检查执行效率
        if execution_time > 1.0:  # 如果执行时间超过1秒，提示优化
            print("警告: 程序执行时间较长，可考虑优化数据结构和算法。")


def main():
    """主函数，处理命令行参数并执行演示"""
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 创建IVFTwoStageDemo实例
    demo = IVFTwoStageDemo()
    
    # 从命令行参数获取查询文本，如果没有则使用默认查询
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "人工智能医疗应用"
    
    # 运行演示
    demo.run_demo(query)


if __name__ == "__main__":
    main()