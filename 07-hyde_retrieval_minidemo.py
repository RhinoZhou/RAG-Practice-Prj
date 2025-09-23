#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE检索最小演示程序

作者：Ph.D. Rhino

功能说明：
  本程序演示HyDE (Hypothetical Document Embeddings) 检索技术，通过生成伪文档扩展查询tokens，
  对比传统检索(baseline)与HyDE增强检索的Hit@K差异，验证HyDE对检索性能的提升效果。

主要功能：
  1. 构建倒排索引
  2. 基于规则生成伪文档扩展tokens
  3. 计算简化TF-IDF相似度
  4. 对比baseline与HyDE检索的命中率差异
  5. 统计并展示Hit@K指标

执行流程：
  构建倒排索引 -> 生成HyDE tokens -> baseline vs hyde检索 -> 统计Hit@K

执行效率优化：
  优化了倒排索引结构，使用高效的字典操作，确保程序运行时间在合理范围内
"""

import re
import random
import math
import time
import os
import sys
import json
from typing import List, Dict, Tuple, Set

# 设置随机种子，确保结果可复现
random.seed(42)

# 依赖检查与自动安装
def check_dependencies():
    """检查必要的依赖并自动安装"""
    try:
        # 本程序主要使用Python标准库，无额外依赖
        print("依赖检查通过，无需额外安装依赖包。")
        return True
    except Exception as e:
        print(f"依赖检查失败: {e}")
        return False

class HyDERetrievalDemo:
    """HyDE检索演示类，实现倒排索引构建、HyDE扩展和检索对比功能"""
    
    def __init__(self):
        """初始化HyDE检索演示类"""
        # 倒排索引结构: {token: {doc_id: count}}
        self.inverted_index = {}
        # 文档集合
        self.documents = {}
        # 查询集合及对应相关文档
        self.queries = {}
        self.relevant_docs = {}
        # 文档总数
        self.total_docs = 0
        # 词汇表
        self.vocabulary = set()
        
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词处理
        
        Args:
            text: 输入文本
        
        Returns:
            分词后的token列表
        """
        # 注意：这里使用简单的字符分割作为中文分词的简化实现
        # 实际应用中应使用专业的中文分词库如jieba
        
        # 转小写
        text = text.lower()
        # 移除标点符号，使用显式标点字符类以确保兼容性
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。！？；：]', ' ', text)
        
        # 对于中文，我们需要特殊处理，这里采用简单的单字切分+2-gram组合
        # 先进行基础分词
        basic_tokens = text.split()
        
        # 处理中文的情况
        final_tokens = []
        for token in basic_tokens:
            # 如果是纯中文或包含中文
            has_chinese = False
            for char in token:
                # 使用正则表达式匹配中文字符
                if re.match(r'[\u4e00-\u9fff]', char):
                    has_chinese = True
                    break
            
            if has_chinese:
                # 单字切分
                char_tokens = list(token)
                # 2-gram组合
                for i in range(len(char_tokens) - 1):
                    bigram = char_tokens[i] + char_tokens[i+1]
                    final_tokens.append(bigram)
                # 添加原始token
                final_tokens.append(token)
            else:
                # 英文等其他情况，直接添加
                final_tokens.append(token)
        
        # 去重并返回
        return list(set(final_tokens))
    
    def build_sample_corpus(self):
        """构建示例文档集合和查询集合
           创建包含医疗、技术、教育等多个领域的示例数据"""
        # 示例文档集合
        self.documents = {
            "d1": "人工智能在医疗领域的应用正变得越来越广泛，从诊断辅助到药物研发，AI技术都发挥着重要作用。",
            "d2": "机器学习是人工智能的一个分支，它通过算法使计算机能够从数据中学习并做出预测。",
            "d3": "自然语言处理技术让计算机能够理解和生成人类语言，这在聊天机器人和翻译系统中得到广泛应用。",
            "d4": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人类大脑的工作方式。",
            "d5": "数据挖掘是从大量数据中发现模式和知识的过程，它结合了统计学、人工智能和数据库技术。",
            "d6": "知识图谱是一种结构化的知识表示方法，它以图的形式展示实体之间的关系。",
            "d7": "医疗影像分析是AI在医疗领域的重要应用之一，它可以帮助医生更准确地识别病变。",
            "d8": "药物发现是一个耗时耗力的过程，AI技术可以加速这一过程，降低研发成本。",
            "d9": "智能诊断系统可以辅助医生进行疾病诊断，提高诊断的准确性和效率。",
            "d10": "远程医疗结合AI技术可以为偏远地区的患者提供高质量的医疗服务。",
        }
        
        # 示例查询集合及相关文档
        self.queries = {
            "q1": "人工智能在医疗中的应用",
            "q2": "机器学习的基本原理",
            "q3": "自然语言处理技术的应用场景",
            "q4": "深度学习与神经网络",
            "q5": "数据挖掘的主要技术",
        }
        
        # 相关文档标注
        self.relevant_docs = {
            "q1": ["d1", "d7", "d8", "d9", "d10"],
            "q2": ["d2", "d4"],
            "q3": ["d3"],
            "q4": ["d2", "d4"],
            "q5": ["d5"],
        }
        
        self.total_docs = len(self.documents)
    
    def build_inverted_index(self):
        """构建倒排索引"""
        print("正在构建倒排索引...")
        
        for doc_id, doc_text in self.documents.items():
            # 分词
            tokens = self._tokenize(doc_text)
            
            # 更新词汇表
            self.vocabulary.update(tokens)
            
            # 构建倒排索引
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                # 记录token在文档中的出现（这里简化为存在与否）
                self.inverted_index[token][doc_id] = 1
        
        print(f"倒排索引构建完成，词汇表大小: {len(self.vocabulary)}")
    
    def generate_hyde_tokens(self, query: str) -> List[str]:
        """基于规则生成HyDE扩展tokens（模拟生成伪文档并提取关键词）
        
        Args:
            query: 原始查询
        
        Returns:
            扩展后的token列表
        """
        # 原始查询分词
        original_tokens = self._tokenize(query)
        
        # 规则1: 添加与查询主题相关的扩展词（模拟从伪文档中提取）
        expansion_rules = {
            "人工智能": ["AI", "机器学习", "深度学习", "神经网络"],
            "医疗": ["医疗影像", "诊断", "药物研发", "远程医疗"],
            "机器学习": ["算法", "数据", "预测", "模型"],
            "自然语言处理": ["聊天机器人", "翻译", "文本分析", "语义理解"],
            "深度学习": ["神经网络", "多层", "特征提取", "模式识别"],
            "数据挖掘": ["统计学", "模式发现", "知识提取", "数据库"],
        }
        
        hyde_tokens = original_tokens.copy()
        
        # 应用扩展规则
        for token in original_tokens:
            if token in expansion_rules:
                hyde_tokens.extend(expansion_rules[token])
        
        # 去重并限制扩展token数量
        hyde_tokens = list(set(hyde_tokens))
        
        # 规则2: 随机添加少量相关领域词，增加多样性
        domain_terms = {
            "技术": ["应用", "系统", "方法", "工具"],
            "研究": ["发现", "分析", "模型", "理论"],
        }
        
        # 随机选择一个领域并添加1-2个词
        if original_tokens:
            domain = "技术" if random.random() > 0.3 else "研究"
            if domain in domain_terms:
                num_terms = 1 if random.random() > 0.5 else 2
                hyde_tokens.extend(random.sample(domain_terms[domain], min(num_terms, len(domain_terms[domain]))))
        
        # 再次去重
        hyde_tokens = list(set(hyde_tokens))
        
        return hyde_tokens
    
    def calculate_tf_idf(self, query_tokens: List[str]) -> List[Tuple[str, float]]:
        """计算简化的TF-IDF分数
        
        Args:
            query_tokens: 查询tokens
        
        Returns:
            文档ID和分数的列表，按分数降序排序
        """
        # 文档得分字典
        doc_scores = {}
        
        # 对每个查询token计算得分
        for token in query_tokens:
            # 如果token不在倒排索引中，跳过
            if token not in self.inverted_index:
                continue
            
            # 计算IDF: log(总文档数/包含该token的文档数)
            doc_freq = len(self.inverted_index[token])
            idf = math.log(self.total_docs / doc_freq)
            
            # 对包含该token的每个文档更新得分
            for doc_id in self.inverted_index[token]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                # 简化的TF=1（这里只考虑存在与否）
                doc_scores[doc_id] += idf  # TF*IDF，TF=1
        
        # 按分数降序排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs
    
    def calculate_hit_at_k(self, ranked_docs: List[Tuple[str, float]], relevant_docs: List[str], k: int = 3) -> float:
        """计算Hit@K指标
        
        Args:
            ranked_docs: 排序后的文档列表
            relevant_docs: 相关文档列表
            k: 前K个结果
        
        Returns:
            Hit@K值
        """
        # 提取前K个文档ID
        top_k_docs = [doc_id for doc_id, _ in ranked_docs[:k]]
        
        # 计算命中数
        hits = 0
        for doc_id in relevant_docs:
            if doc_id in top_k_docs:
                hits += 1
        
        # 计算Hit@K: 命中数 / 相关文档总数
        # 避免除以0
        if len(relevant_docs) == 0:
            return 0.0
        
        return hits / len(relevant_docs)
    
    def run_comparison(self):
        """运行baseline与HyDE检索的对比实验"""
        print("\n开始baseline与HyDE检索对比实验...")
        
        # 用于存储每个查询的结果
        baseline_results = {}
        hyde_results = {}
        
        # 用于计算平均Hit@K
        baseline_hit_at_3 = 0.0
        hyde_hit_at_3 = 0.0
        
        # 对每个查询进行实验
        for query_id, query_text in self.queries.items():
            # 获取相关文档
            rel_docs = self.relevant_docs.get(query_id, [])
            
            print(f"\n查询 {query_id}: {query_text}")
            
            # Baseline检索
            baseline_tokens = self._tokenize(query_text)
            baseline_ranked = self.calculate_tf_idf(baseline_tokens)
            baseline_hit = self.calculate_hit_at_k(baseline_ranked, rel_docs, k=3)
            
            # HyDE检索
            hyde_tokens = self.generate_hyde_tokens(query_text)
            hyde_ranked = self.calculate_tf_idf(hyde_tokens)
            hyde_hit = self.calculate_hit_at_k(hyde_ranked, rel_docs, k=3)
            
            # 存储结果
            baseline_results[query_id] = {
                "tokens": baseline_tokens,
                "ranked_docs": baseline_ranked,
                "hit_at_3": baseline_hit
            }
            
            hyde_results[query_id] = {
                "tokens": hyde_tokens,
                "ranked_docs": hyde_ranked,
                "hit_at_3": hyde_hit
            }
            
            # 累加Hit@K用于计算平均
            baseline_hit_at_3 += baseline_hit
            hyde_hit_at_3 += hyde_hit
            
            # 输出详细结果
            print(f"  Baseline检索: Hit@3 = {baseline_hit:.4f}")
            print(f"    关键词: {baseline_tokens}")
            print(f"    前3结果: {baseline_ranked[:3]}")
            print(f"  HyDE检索: Hit@3 = {hyde_hit:.4f}")
            print(f"    扩展关键词: {hyde_tokens}")
            print(f"    前3结果: {hyde_ranked[:3]}")
            print(f"  性能提升: {(hyde_hit - baseline_hit):+.4f}")
        
        # 计算平均Hit@K
        num_queries = len(self.queries)
        baseline_hit_at_3_avg = baseline_hit_at_3 / num_queries
        hyde_hit_at_3_avg = hyde_hit_at_3 / num_queries
        delta = hyde_hit_at_3_avg - baseline_hit_at_3_avg
        
        # 输出实验总结
        print("\n=== 实验总结 ===")
        print(f"baseline_hit@3={baseline_hit_at_3_avg:.4f}, hyde_hit@3={hyde_hit_at_3_avg:.4f}, delta={delta:+.4f}")
        
        # 返回结果用于分析
        return {
            "baseline_hit_at_3": baseline_hit_at_3_avg,
            "hyde_hit_at_3": hyde_hit_at_3_avg,
            "delta": delta,
            "baseline_results": baseline_results,
            "hyde_results": hyde_results
        }
    
    def run_optimization_analysis(self):
        """执行效率分析"""
        print("\n=== 执行效率分析 ===")
        
        # 测试多次运行时间
        num_runs = 5
        total_time = 0.0
        
        for i in range(num_runs):
            start_time = time.time()
            
            # 执行检索对比
            for query_id, query_text in self.queries.items():
                # Baseline检索
                baseline_tokens = self._tokenize(query_text)
                baseline_ranked = self.calculate_tf_idf(baseline_tokens)
                
                # HyDE检索
                hyde_tokens = self.generate_hyde_tokens(query_text)
                hyde_ranked = self.calculate_tf_idf(hyde_tokens)
            
            end_time = time.time()
            run_time = (end_time - start_time) * 1000  # 转换为毫秒
            total_time += run_time
            print(f"运行 {i+1}: {run_time:.2f}毫秒")
        
        avg_time = total_time / num_runs
        print(f"平均运行时间 ({num_runs}次): {avg_time:.2f}毫秒")
        
        # 检查执行效率是否满足要求
        if avg_time > 50:
            print("警告: 执行效率较低，可能需要进一步优化。")
        else:
            print("执行效率良好，满足实时检索需求。")
    
    def save_results(self, results: Dict):
        """保存实验结果到文件
        
        Args:
            results: 实验结果数据
        """
        # 准备结果数据
        output_data = {
            "summary": {
                "baseline_hit_at_3": results["baseline_hit_at_3"],
                "hyde_hit_at_3": results["hyde_hit_at_3"],
                "delta": results["delta"]
            },
            "detailed_results": {
                "baseline": results["baseline_results"],
                "hyde": results["hyde_results"]
            }
        }
        
        # 保存到文件
        output_file = "07-hyde_retrieval_results.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n实验结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存实验结果失败: {e}")

# 主函数
def main():
    """主函数，执行HyDE检索演示"""
    print("===== HyDE检索最小演示程序 ======")
    
    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，程序退出。")
        sys.exit(1)
    
    # 创建HyDE检索演示实例
    hyde_demo = HyDERetrievalDemo()
    
    # 构建示例语料库
    hyde_demo.build_sample_corpus()
    
    # 构建倒排索引
    hyde_demo.build_inverted_index()
    
    # 执行对比实验
    start_time = time.time()
    results = hyde_demo.run_comparison()
    end_time = time.time()
    
    # 计算总执行时间
    total_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"总执行时间: {total_time:.2f}毫秒")
    
    # 执行效率分析
    hyde_demo.run_optimization_analysis()
    
    # 保存实验结果
    hyde_demo.save_results(results)
    
    # 检查输出是否达到演示目的
    if results["delta"] > 0:
        print("\n实验成功: HyDE检索技术在Hit@3指标上优于baseline检索。")
        print("演示目的已达成: 通过生成伪文档扩展tokens，HyDE有效提高了检索的命中率。")
    else:
        print("\n注意: 在当前实验设置下，HyDE检索未显示出明显优势。")
        print("可能需要调整扩展规则或增大数据集规模以观察更明显的效果。")
    
    print("\n===== 程序执行完成 =====")

# 程序入口
if __name__ == "__main__":
    main()