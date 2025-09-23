#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
程序名称：重写、缓存与去重组合系统
功能说明：实现规范化、同义词替换与编辑距=1纠错；L1缓存命中短路；
         对返回文本生成MinHash风格签名进行阈值去重并统计命中率。

主要功能点：
- 小词典重写：通过预定义词典实现查询规范化和同义词替换
- 编辑距候选与替换映射：支持编辑距离为1的单词纠错
- 字典缓存与命中计数：实现快速查询缓存机制，提高响应速度
- shingles+多哈希的签名：生成文档的MinHash风格签名
- 签名一致比例近似Jaccard去重：基于签名相似度进行去重

执行流程：
1. 输入NL → 规范化与同义扩展+纠错
2. L1缓存检查
3. 检索结果内容签名与去重
4. 写回缓存与统计

作者：Ph.D. Rhino
"""

import re
import time
import hashlib
import json
import os
import sys
import logging
from collections import defaultdict, deque
import heapq
import random
import math

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rewrite_cache_dedup.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def check_and_install_dependencies():
    """检查并自动安装必要的依赖包"""
    try:
        # 导入可能需要安装的包
        import numpy as np
        logger.info("所有依赖包已安装")
        return True
    except ImportError:
        logger.warning("检测到缺失依赖包，尝试自动安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            logger.info("依赖包安装成功")
            return True
        except Exception as e:
            logger.error(f"依赖包安装失败: {e}")
            print("错误: 无法安装必要的依赖包，请手动安装numpy")
            return False


class QueryRewriter:
    """查询重写器，实现规范化、同义词替换与编辑距纠错"""
    def __init__(self):
        # 初始化重写词典
        self.normalization_dict = {
            "ai": "人工智能",
            "nlp": "自然语言处理",
            "ml": "机器学习",
            "dl": "深度学习",
            "rag": "检索增强生成",
            "faq": "常见问题",
            "gpu": "图形处理器",
        }
        
        # 同义词词典
        self.synonym_dict = {
            "人工智能": ["AI", "artificial intelligence"],
            "机器学习": ["machine learning"],
            "自然语言处理": ["NLP"],
            "深度学习": ["deep learning"],
            "检索": ["搜索", "查找", "检索"],
            "文档": ["文件", "文本", "资料"],
        }
        
        # 常见拼写错误词典
        self.spell_check_dict = {
            "人工智嫩": "人工智能",
            "机器学系": "机器学习",
            "自然语言除理": "自然语言处理",
            "深度学系": "深度学习",
            "检索增强生层": "检索增强生成",
        }
        
        # 编辑距1的单词替换映射
        self.edit_distance_map = self._build_edit_distance_map()
    
    def _build_edit_distance_map(self):
        """构建编辑距映射表"""
        # 这里简化实现，实际项目中可以基于更大的词典构建
        common_words = ["人工智能", "机器学习", "自然语言处理", "深度学习", "检索", "文档"]
        map = {}
        for word in common_words:
            # 这里我们简单地将一些可能的错别字映射到正确的词
            # 实际应用中可以使用更复杂的编辑距离算法
            if word == "人工智能":
                map["人工智嫩"] = word
                map["人工知能"] = word
            elif word == "机器学习":
                map["机器学系"] = word
                map["机器学习"] = word
            elif word == "自然语言处理":
                map["自然语言除理"] = word
                map["自然语方处理"] = word
        return map
    
    def normalize(self, query):
        """执行查询规范化"""
        for key, value in self.normalization_dict.items():
            # 确保大小写不敏感
            query = re.sub(rf'\b{key}\b', value, query, flags=re.IGNORECASE)
        return query
    
    def replace_synonyms(self, query):
        """执行同义词替换"""
        # 简单实现：将查询中的词替换为其同义词列表中的第一个词
        for target, synonyms in self.synonym_dict.items():
            for synonym in synonyms:
                if synonym.lower() in query.lower():
                    # 替换为规范表述
                    query = re.sub(rf'\b{synonym}\b', target, query, flags=re.IGNORECASE)
        return query
    
    def correct_spelling(self, query):
        """执行拼写纠错"""
        # 首先检查已知的拼写错误
        for misspelled, correct in self.spell_check_dict.items():
            if misspelled in query:
                query = query.replace(misspelled, correct)
                logger.info(f"纠正拼写错误: '{misspelled}' -> '{correct}'")
                
        # 然后检查编辑距为1的可能错误
        for error, correct in self.edit_distance_map.items():
            if error in query:
                query = query.replace(error, correct)
                logger.info(f"基于编辑距纠正: '{error}' -> '{correct}'")
                
        return query
    
    def rewrite(self, query):
        """执行完整的查询重写流程"""
        original_query = query
        
        # 规范化
        query = self.normalize(query)
        
        # 同义词替换
        query = self.replace_synonyms(query)
        
        # 拼写纠错
        query = self.correct_spelling(query)
        
        if original_query != query:
            logger.info(f"查询重写: '{original_query}' -> '{query}'")
            
        return query


class L1Cache:
    """L1缓存实现，支持快速查询命中短路"""
    def __init__(self, capacity=1000, ttl=60):
        """
        初始化缓存
        
        Args:
            capacity: 缓存容量
            ttl: 缓存项的生存时间(秒)
        """
        self.cache = {}
        self.capacity = capacity
        self.ttl = ttl
        self.access_times = deque()  # 用于LRU淘汰
        self.hit_count = 0
        self.miss_count = 0
        
    def _is_expired(self, timestamp):
        """检查缓存项是否过期"""
        return time.time() - timestamp > self.ttl
    
    def _evict(self):
        """执行缓存淘汰策略"""
        if len(self.cache) >= self.capacity:
            # 简单LRU淘汰：移除最早访问的项
            if self.access_times:
                oldest_key = self.access_times.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
                    logger.debug(f"缓存项被淘汰: {oldest_key}")
    
    def get(self, key):
        """从缓存获取项"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if self._is_expired(timestamp):
                # 缓存过期
                del self.cache[key]
                self.miss_count += 1
                logger.debug(f"缓存过期: {key}")
                return None
            
            # 缓存命中
            self.hit_count += 1
            self.access_times.append(key)
            logger.debug(f"缓存命中: {key}")
            return value
        
        # 缓存未命中
        self.miss_count += 1
        logger.debug(f"缓存未命中: {key}")
        return None
    
    def put(self, key, value):
        """向缓存添加项"""
        self._evict()
        self.cache[key] = (value, time.time())
        self.access_times.append(key)
        logger.debug(f"缓存项添加: {key}")
    
    def get_hit_rate(self):
        """计算缓存命中率"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("缓存已清空")


class TextDeduplicator:
    """文本去重器，使用MinHash风格签名"""
    def __init__(self, num_hashes=10, threshold=0.7, shingle_size=3):
        """
        初始化文本去重器
        
        Args:
            num_hashes: 哈希函数数量
            threshold: 去重阈值
            shingle_size: shingle的大小(字符数)
        """
        self.num_hashes = num_hashes
        self.threshold = threshold
        self.shingle_size = shingle_size
        
        # 生成多个哈希函数的随机种子
        self.hash_seeds = [random.randint(1, 1000000) for _ in range(num_hashes)]
        
        # 存储已处理文本的签名
        self.signatures = []
        
        # 统计信息
        self.total_docs = 0
        self.removed_docs = 0
    
    def _get_shingles(self, text):
        """提取文本的shingles"""
        # 简单实现：提取字符级别的shingles
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i+self.shingle_size]
            shingles.add(shingle)
        return shingles
    
    def _compute_min_hash(self, shingles):
        """计算MinHash签名"""
        signature = []
        for seed in self.hash_seeds:
            min_hash = float('inf')
            for shingle in shingles:
                # 使用哈希种子计算shingle的哈希值
                hash_val = int(hashlib.md5(f"{seed}{shingle}".encode()).hexdigest(), 16) % (2**32)
                if hash_val < min_hash:
                    min_hash = hash_val
            signature.append(min_hash)
        return signature
    
    def _jaccard_similarity(self, sig1, sig2):
        """计算两个签名的Jaccard相似度"""
        if len(sig1) != len(sig2):
            return 0.0
        # 使用签名一致比例近似Jaccard相似度
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)
    
    def is_duplicate(self, text):
        """检查文本是否为重复项"""
        self.total_docs += 1
        
        # 提取shingles
        shingles = self._get_shingles(text)
        if not shingles:
            return False  # 空文本不视为重复
        
        # 计算签名
        signature = self._compute_min_hash(shingles)
        
        # 检查与现有签名的相似度
        for existing_sig in self.signatures:
            similarity = self._jaccard_similarity(signature, existing_sig)
            if similarity >= self.threshold:
                self.removed_docs += 1
                return True
        
        # 不是重复项，保存签名
        self.signatures.append(signature)
        return False
    
    def get_duplicate_rate(self):
        """获取重复率"""
        if self.total_docs == 0:
            return 0.0
        return self.removed_docs / self.total_docs


class MockRetriever:
    """模拟检索器，用于演示"""
    def __init__(self):
        # 模拟文档库
        self.documents = {
            "人工智能": [
                "人工智能是计算机科学的一个分支，研究如何使机器能够模拟人类智能行为。",
                "AI技术正在各个领域得到广泛应用，包括医疗、金融和自动驾驶等。",
                "机器学习是实现人工智能的重要方法之一。"
            ],
            "机器学习": [
                "机器学习是人工智能的一个子集，专注于开发能从数据中学习的算法。",
                "监督学习、无监督学习和强化学习是机器学习的主要范式。",
                "深度学习是机器学习的一个分支，基于人工神经网络。"
            ],
            "自然语言处理": [
                "自然语言处理是人工智能的一个分支，研究计算机如何理解和处理人类语言。",
                "NLP技术包括文本分类、情感分析和机器翻译等。",
                "预训练语言模型如BERT和GPT在自然语言处理领域取得了突破性进展。"
            ]
        }
    
    def retrieve(self, query):
        """模拟检索过程"""
        # 简单的关键词匹配
        results = []
        for topic, docs in self.documents.items():
            if topic in query:
                results.extend(docs)
        
        # 如果没有匹配到，返回默认结果
        if not results:
            results = ["抱歉，没有找到相关文档。"]
        
        return results


class RewriteCacheDedupSystem:
    """重写、缓存与去重组合系统"""
    def __init__(self):
        # 初始化各个组件
        self.rewriter = QueryRewriter()
        self.cache = L1Cache()
        self.deduplicator = TextDeduplicator()
        self.retriever = MockRetriever()
        
        # 系统级统计
        self.total_queries = 0
        self.start_time = 0
    
    def process_query(self, query):
        """处理单个查询"""
        self.total_queries += 1
        logger.info(f"处理查询: '{query}'")
        
        # 1. 查询重写
        rewritten_query = self.rewriter.rewrite(query)
        
        # 2. L1缓存检查
        cache_key = hashlib.md5(rewritten_query.encode()).hexdigest()
        cached_results = self.cache.get(cache_key)
        
        if cached_results:
            logger.info("缓存命中，跳过检索")
            return cached_results
        
        # 3. 执行检索
        raw_results = self.retriever.retrieve(rewritten_query)
        
        # 4. 去重处理
        dedup_results = []
        for result in raw_results:
            if not self.deduplicator.is_duplicate(result):
                dedup_results.append(result)
        
        # 5. 写回缓存
        self.cache.put(cache_key, dedup_results)
        
        logger.info(f"检索完成，原始结果数: {len(raw_results)}, 去重后结果数: {len(dedup_results)}")
        return dedup_results
    
    def run_demo(self):
        """运行演示"""
        # 内置多轮相似查询调用
        demo_queries = [
            "查找关于人工智能的文档",
            "我想了解人工智能的相关内容",
            "查询AI技术的资料",
            "人工智嫩的应用场景有哪些？",
            "机器学习和深度学习的区别是什么？",
            "什么是自然语言处理？",
            "NLP技术在哪些领域有应用？",
            "机器学系的主要方法有哪些？",
            "查找深度学习相关文档",
            "人工智能和机器学习的关系是什么？"
        ]
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 处理所有查询
        for i, query in enumerate(demo_queries, 1):
            logger.info(f"===== 处理查询 {i}/{len(demo_queries)} =====")
            results = self.process_query(query)
            logger.info(f"查询结果 ({len(results)}条): {results}")
        
        # 计算总执行时间
        total_time = time.time() - self.start_time
        
        # 收集统计信息
        stats = {
            "total_queries": self.total_queries,
            "hit_rate": self.cache.get_hit_rate(),
            "dedup_removed": self.deduplicator.removed_docs,
            "total_docs_processed": self.deduplicator.total_docs,
            "execution_time": total_time,
            "avg_query_time": total_time / self.total_queries if self.total_queries > 0 else 0
        }
        
        # 保存统计结果
        with open("rewrite_cache_dedup_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        self.print_results(stats)
        
        return stats
    
    def print_results(self, stats):
        """打印结果摘要"""
        print("\n===== 重写、缓存与去重组合系统 - 执行结果 =====")
        print(f"总查询数: {stats['total_queries']}")
        print(f"缓存命中率: {stats['hit_rate']:.2f}")
        print(f"去重移除文档数: {stats['dedup_removed']}")
        print(f"处理文档总数: {stats['total_docs_processed']}")
        print(f"总执行时间: {stats['execution_time']:.4f}秒")
        print(f"平均查询时间: {stats['avg_query_time']*1000:.2f}毫秒")
        print(f"\nhit_rate={stats['hit_rate']:.2f}, dedup_removed={stats['dedup_removed']}")
        print("=============================================")


def main():
    """主函数"""
    # 检查依赖
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # 初始化系统
    system = RewriteCacheDedupSystem()
    
    # 运行演示
    stats = system.run_demo()
    
    # 检查执行效率
    if stats['avg_query_time'] > 0.1:  # 如果平均查询时间超过100毫秒，认为执行较慢
        logger.warning("警告：程序执行效率较低，平均查询时间超过100毫秒")
        print("警告：程序执行效率较低，建议进一步优化")
    
    # 检查结果文件
    if os.path.exists("rewrite_cache_dedup_stats.json"):
        logger.info("结果文件已生成：rewrite_cache_dedup_stats.json")
        # 验证中文是否有乱码
        try:
            with open("rewrite_cache_dedup_stats.json", "r", encoding="utf-8") as f:
                content = f.read()
                # 检查是否包含中文字符
                if any('\u4e00' <= char <= '\u9fff' for char in content):
                    logger.info("结果文件中的中文显示正常")
        except Exception as e:
            logger.error(f"验证结果文件时出错: {e}")
    
    logger.info("程序执行完成")


if __name__ == "__main__":
    main()