#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""批量处理、缓存与降级策略演示程序

该脚本演示了在检索增强生成(RAG)系统中实现高性能和高可用性的关键技术：
1. 批量处理：基于最大等待时间和最大批大小的智能请求聚合
2. 查询缓存：使用LRU算法实现查询结果的高效缓存
3. 多级降级：基于延迟和错误率的自适应策略降级机制

作者：Ph.D. Rhino

使用方法：
    python 08-batch_cache_degrade_demo.py

输入文件：
    data/queries.txt - 包含查询文本的文件
    data/metrics_stream.csv - 模拟的系统指标数据流
    configs/degrade.json - 降级策略配置文件

输出文件：
    outputs/strategy_decisions.log - 记录降级策略决策日志

依赖项：
    numpy - 用于数值计算
    pandas - 用于数据处理
    tqdm - 用于显示进度条

功能特点：
    - 自动检查和安装所需依赖
    - 实现智能批量处理器，平衡延迟与吞吐量
    - 实现LRU缓存，提高查询响应速度
    - 基于系统指标实现多级策略降级
    - 提供详细的策略决策日志
"""

import os
import sys
import json
import time
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置日志
def setup_logger():
    """配置日志记录器"""
    # 确保outputs目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 配置日志记录器
    logger = logging.getLogger('strategy_decision_logger')
    logger.setLevel(logging.INFO)
    
    # 检查是否已有处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(os.path.join('outputs', 'strategy_decisions.log'), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 依赖检查与安装
def install_dependencies():
    """检查并安装必要的依赖包"""
    try:
        import numpy
        import pandas
        import tqdm
        print("所有必要的依赖已安装。")
    except ImportError:
        print("正在安装必要的依赖...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "tqdm"])
            print("依赖安装成功。")
        except Exception as e:
            print(f"依赖安装失败: {e}")
            sys.exit(1)

# LRU缓存实现
class LRUCache:
    """实现LRU缓存机制"""
    def __init__(self, capacity, default_ttl=3600):
        """
        初始化LRU缓存
        
        Args:
            capacity: 缓存容量
            default_ttl: 默认缓存过期时间(秒)
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.logger = setup_logger()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """获取缓存中的值，如果过期则返回None"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, expiry_time = self.cache[key]
        current_time = time.time()
        
        # 检查是否过期
        if current_time > expiry_time:
            # 过期则移除
            del self.cache[key]
            self.misses += 1
            return None
        
        # 更新访问顺序（移到末尾）
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    def put(self, key, value, ttl=None):
        """将值放入缓存"""
        # 如果键已存在，先删除
        if key in self.cache:
            del self.cache[key]
        
        # 如果缓存已满，删除最少使用的项（最前面的）
        elif len(self.cache) >= self.capacity:
            # 优先删除已过期的项
            current_time = time.time()
            expired_keys = [k for k, (_, expiry) in self.cache.items() if current_time > expiry]
            if expired_keys:
                self.logger.info(f"清理{len(expired_keys)}个过期缓存项")
                for k in expired_keys:
                    del self.cache[k]
            else:
                # 如果没有过期项，删除最少使用的项
                self.cache.popitem(last=False)
        
        # 设置过期时间
        expiry_time = time.time() + (ttl if ttl is not None else self.default_ttl)
        self.cache[key] = (value, expiry_time)
    
    def get_stats(self):
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache),
            "capacity": self.capacity
        }

# 批量处理器实现
class BatchProcessor:
    """实现基于最大等待时间和最大批大小的批量处理器"""
    def __init__(self, max_batch_size=20, max_wait_time_ms=50, cooldown_period_ms=1000):
        """
        初始化批量处理器
        
        Args:
            max_batch_size: 最大批大小
            max_wait_time_ms: 最大等待时间(毫秒)
            cooldown_period_ms: 冷却期(毫秒)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.cooldown_period_ms = cooldown_period_ms
        self.queue = []
        self.last_process_time = 0
        self.logger = setup_logger()
    
    def add_request(self, request):
        """添加请求到队列"""
        self.queue.append(request)
        
        # 检查是否满足批处理条件
        if len(self.queue) >= self.max_batch_size:
            self.logger.info(f"达到最大批大小({self.max_batch_size})，触发批量处理")
            return self.process_batch()
        
        current_time = time.time() * 1000  # 转换为毫秒
        
        # 检查是否超过最大等待时间
        if (self.last_process_time > 0 and 
            current_time - self.last_process_time > self.max_wait_time_ms):
            self.logger.info(f"超过最大等待时间({self.max_wait_time_ms}ms)，触发批量处理")
            return self.process_batch()
        
        return None  # 未触发批量处理
    
    def process_batch(self):
        """处理当前队列中的请求"""
        if not self.queue:
            return []
        
        # 记录处理时间
        self.last_process_time = time.time() * 1000
        
        # 获取当前批次
        batch_size = min(len(self.queue), self.max_batch_size)
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        
        self.logger.info(f"处理批次: 大小={batch_size}, 剩余队列长度={len(self.queue)}")
        
        # 模拟处理时间
        processing_time = np.random.uniform(10, 50)  # 10-50ms
        time.sleep(processing_time / 1000)  # 转换为秒
        
        # 模拟处理结果
        results = [{
            'id': req['id'],
            'result': f"处理结果 for {req['query']}",
            'processing_time_ms': round(processing_time, 2),
            'timestamp': time.time(),
            'normalized_query': req.get('normalized_query', '')
        } for req in batch]
        
        return results
    
    def flush(self):
        """强制刷新队列"""
        if self.queue:
            self.logger.info("强制刷新队列")
            return self.process_batch()
        return []

# 降级策略管理器
class DegradeStrategyManager:
    """实现基于延迟和错误率的多级降级策略"""
    def __init__(self, config_path):
        """
        初始化降级策略管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.degrade_levels = self.config['degrade_levels']
        self.current_level = self.degrade_levels[0]  # 初始为正常模式
        self.logger = setup_logger()
        self.state_change_count = 0
        
        # 记录历史状态
        self.history = []
    
    def evaluate_state(self, metrics):
        """
        根据系统指标评估当前状态并确定降级级别
        
        Args:
            metrics: 包含latency_ms和error_rate的指标字典
        
        Returns:
            更新后的当前降级级别
        """
        latency = metrics.get('latency_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        request_count = metrics.get('request_count', 0)
        
        # 找出合适的降级级别
        new_level = None
        for level in reversed(self.degrade_levels):  # 从最严重的开始检查
            if latency >= level['latency_threshold'] or error_rate >= level['error_threshold']:
                new_level = level
                break
        
        # 如果没有达到任何降级条件，使用正常模式
        if new_level is None:
            new_level = self.degrade_levels[0]
        
        # 检查状态是否发生变化
        if new_level['name'] != self.current_level['name']:
            self.state_change_count += 1
            self.logger.info(
                f"状态变更: 从{self.current_level['name']}({self.current_level['description']}) -> "
                f"{new_level['name']}({new_level['description']}) [延迟={latency}ms, 错误率={error_rate:.2%}, 请求数={request_count}]")
            self.current_level = new_level
        
        # 记录历史
        self.history.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'level': self.current_level['name']
        })
        
        return self.current_level
    
    def get_current_params(self):
        """获取当前降级级别的参数"""
        return self.current_level['params']
    
    def get_stats(self):
        """获取降级策略统计信息"""
        return {
            "current_level": self.current_level['name'],
            "state_changes": self.state_change_count,
            "history_length": len(self.history)
        }

# 查询归一化处理器
class QueryNormalizer:
    """实现查询归一化功能"""
    def __init__(self):
        """初始化查询归一化器"""
        self.logger = setup_logger()
    
    def normalize(self, query):
        """
        对查询进行归一化处理
        
        Args:
            query: 原始查询文本
        
        Returns:
            归一化后的查询文本
        """
        # 基本归一化处理
        normalized = query.strip().lower()
        
        # 去除多余空格
        normalized = ' '.join(normalized.split())
        
        # 移除常见标点符号
        punctuation = '.,?!:;"\'()[]{}\/|'
        
        # 记录归一化日志
        if normalized != query.lower():
            self.logger.debug(f"查询归一化: '{query}' -> '{normalized}'")
        
        return normalized

# 模拟检索器
class MockRetriever:
    """模拟检索器，用于演示"""
    def __init__(self):
        """初始化模拟检索器"""
        self.logger = setup_logger()
    
    def retrieve(self, queries, params):
        """
        模拟检索过程
        
        Args:
            queries: 查询列表
            params: 检索参数
        
        Returns:
            检索结果列表
        """
        search_method = params.get('search_method', 'vector')
        top_k = params.get('top_k', 10)
        use_rerank = params.get('use_rerank', True)
        
        self.logger.info(f"执行检索: 方法={search_method}, top_k={top_k}, 重排={use_rerank}, 查询数={len(queries)}")
        
        # 模拟检索延迟
        latency = {
            'vector': np.random.uniform(50, 150),
            'hybrid': np.random.uniform(100, 200),
            'bm25': np.random.uniform(20, 80)
        }.get(search_method, np.random.uniform(50, 150))
        
        # 模拟错误率
        error_prob = {
            'vector': np.random.uniform(0.01, 0.05),
            'hybrid': np.random.uniform(0.02, 0.08),
            'bm25': np.random.uniform(0.005, 0.03)
        }.get(search_method, np.random.uniform(0.01, 0.05))
        
        # 根据top_k调整
        if top_k < 5:
            latency *= 0.7
        elif top_k > 8:
            latency *= 1.3
        
        # 模拟处理时间
        time.sleep(latency / 1000)  # 转换为秒
        
        # 生成结果
        results = []
        for query in queries:
            # 模拟是否发生错误
            if np.random.random() < error_prob:
                results.append({
                    'query': query,
                    'success': False,
                    'error': f"模拟错误: {search_method}检索失败",
                    'latency_ms': latency
                })
            else:
                # 生成模拟结果
                items = []
                for i in range(top_k):
                    score = 1.0 - (i * 0.1)  # 分数递减
                    items.append({
                        'id': f"doc_{query[:3]}_{i}",
                        'score': round(score, 2),
                        'text': f"与'{query}'相关的文档内容摘要 #{i+1}"
                    })
                
                # 模拟重排
                if use_rerank:
                    # 简单随机扰动模拟重排
                    np.random.shuffle(items)
                    items = sorted(items, key=lambda x: x['score'], reverse=True)
                    
                results.append({
                    'query': query,
                    'success': True,
                    'results': items,
                    'latency_ms': latency,
                    'search_method': search_method,
                    'top_k': top_k,
                    'used_rerank': use_rerank
                })
        
        return results, {
            'avg_latency_ms': latency,
            'error_rate': error_prob,
            'total_queries': len(queries)
        }

# 主程序逻辑
def main():
    """主函数，演示批量处理、缓存和降级策略"""
    start_time = time.time()
    
    # 设置日志
    logger = setup_logger()
    logger.info("=== 批量处理、缓存与降级策略演示程序启动 ===")
    
    # 检查并安装依赖
    install_dependencies()
    
    # 定义文件路径
    queries_path = os.path.join('data', 'queries.txt')
    metrics_path = os.path.join('data', 'metrics_stream.csv')
    config_path = os.path.join('configs', 'degrade.json')
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化组件
    batch_config = config.get('batch_processor', {})
    cache_config = config.get('cache', {})
    
    batch_processor = BatchProcessor(
        max_batch_size=batch_config.get('max_batch_size', 20),
        max_wait_time_ms=batch_config.get('max_wait_time_ms', 50),
        cooldown_period_ms=batch_config.get('cooldown_period_ms', 1000)
    )
    
    cache = LRUCache(
        capacity=cache_config.get('max_size', 1000),
        default_ttl=cache_config.get('default_ttl', 3600)
    )
    
    strategy_manager = DegradeStrategyManager(config_path)
    query_normalizer = QueryNormalizer()
    retriever = MockRetriever()
    
    # 加载查询数据
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    # 加载指标流数据
    metrics_df = pd.read_csv(metrics_path)
    
    logger.info(f"加载完成: 查询{len(queries)}条, 指标数据{len(metrics_df)}条")
    
    # 处理查询并演示批量、缓存和降级功能
    processed_count = 0
    total_processed = 0
    
    # 创建进度条
    pbar = tqdm(total=len(queries) * 5, desc="处理查询")  # 模拟5次重复查询以演示缓存
    
    # 模拟多个查询批次
    for cycle in range(5):
        for i, query in enumerate(queries):
            # 查询归一化
            normalized_query = query_normalizer.normalize(query)
            
            # 生成查询ID
            query_id = f"q_{cycle}_{i}"
            
            # 尝试从缓存获取
            cached_result = cache.get(normalized_query)
            
            if cached_result:
                logger.info(f"缓存命中: 查询'{query_id}' (缓存键: '{normalized_query}')")
                processed_count += 1
                pbar.update(1)
                continue
            
            logger.info(f"缓存未命中: 查询'{query_id}' (缓存键: '{normalized_query}')")
            
            # 模拟系统指标
            metric_idx = min(i, len(metrics_df) - 1)
            metrics = {
                'latency_ms': metrics_df.iloc[metric_idx]['latency_ms'],
                'error_rate': metrics_df.iloc[metric_idx]['error_rate'],
                'request_count': metrics_df.iloc[metric_idx]['request_count']
            }
            
            # 评估降级状态
            current_level = strategy_manager.evaluate_state(metrics)
            current_params = strategy_manager.get_current_params()
            
            # 添加到批量处理器
            request = {
                'id': query_id,
                'query': query,
                'normalized_query': normalized_query,
                'timestamp': time.time(),
                'metrics': metrics
            }
            
            batch_results = batch_processor.add_request(request)
            
            # 处理批量结果
            if batch_results:
                for result in batch_results:
                    # 执行检索
                    retrieved_results, retrieval_metrics = retriever.retrieve(
                        [result['id']],  # 这里简化处理，实际应处理整个批次
                        current_params
                    )
                    
                    # 更新缓存
                    cache_ttl = current_params.get('cache_ttl', cache_config.get('default_ttl', 3600))
                    cache.put(result['normalized_query'], retrieved_results, ttl=cache_ttl)
                    
                    processed_count += 1
                    total_processed += 1
                    
            pbar.update(1)
            
            # 模拟请求间隔
            time.sleep(0.05)  # 50ms
        
        # 每个周期结束时刷新批量处理器
        remaining_results = batch_processor.flush()
        for result in remaining_results:
            processed_count += 1
            total_processed += 1
        
        # 模拟周期间隔
        time.sleep(0.5)
    
    pbar.close()
    
    # 输出统计信息
    cache_stats = cache.get_stats()
    strategy_stats = strategy_manager.get_stats()
    
    logger.info("\n=== 执行统计信息 ===")
    logger.info(f"总处理查询数: {total_processed}")
    logger.info(f"缓存统计: 命中={cache_stats['hits']}, 未命中={cache_stats['misses']}, 命中率={cache_stats['hit_rate']}%")
    logger.info(f"降级策略统计: 当前级别={strategy_stats['current_level']}, 状态变更次数={strategy_stats['state_changes']}")
    logger.info(f"总执行时间: {time.time() - start_time:.2f}秒")
    logger.info("=== 演示程序执行完成 ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)