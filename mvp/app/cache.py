#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存模块
负责查询结果和向量的缓存管理
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import json
import os
from app.config import config

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = cache_dir or os.path.join(config.DATA_DIR, "cache")
        self._init_cache_dir()
        
        # 内存缓存
        self.memory_cache = {}
        self.cache_ttl = 3600  # 默认缓存过期时间（秒）
    
    def _init_cache_dir(self) -> None:
        """\初始化缓存目录"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        从缓存中获取数据
        
        Args:
            key: 缓存键
            namespace: 缓存命名空间
        
        Returns:
            缓存的数据，如果不存在则返回None
        """
        # TODO: 实现缓存获取功能
        full_key = self._get_full_key(key, namespace)
        
        # 1. 首先检查内存缓存
        if full_key in self.memory_cache:
            cache_entry = self.memory_cache[full_key]
            if not self._is_expired(cache_entry):
                return cache_entry["data"]
            else:
                # 缓存已过期，删除
                del self.memory_cache[full_key]
        
        # 2. 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{full_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_entry = json.load(f)
                
                if not self._is_expired(cache_entry):
                    # 加载到内存缓存
                    self.memory_cache[full_key] = cache_entry
                    return cache_entry["data"]
                else:
                    # 缓存已过期，删除文件
                    os.remove(cache_file)
            except Exception as e:
                print(f"读取缓存文件失败: {e}")
        
        return None
    
    def set(self, key: str, data: Any, namespace: str = "default", ttl: Optional[int] = None) -> None:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            namespace: 缓存命名空间
            ttl: 缓存过期时间（秒），None表示使用默认值
        """
        # TODO: 实现缓存设置功能
        full_key = self._get_full_key(key, namespace)
        expire_time = time.time() + (ttl or self.cache_ttl)
        
        # 构建缓存条目
        cache_entry = {
            "data": data,
            "expire_time": expire_time,
            "created_at": time.time()
        }
        
        # 1. 更新内存缓存
        self.memory_cache[full_key] = cache_entry
        
        # 2. 更新磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{full_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入缓存文件失败: {e}")
    
    def delete(self, key: str, namespace: str = "default") -> None:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            namespace: 缓存命名空间
        """
        # TODO: 实现缓存删除功能
        full_key = self._get_full_key(key, namespace)
        
        # 1. 从内存缓存中删除
        if full_key in self.memory_cache:
            del self.memory_cache[full_key]
        
        # 2. 从磁盘缓存中删除
        cache_file = os.path.join(self.cache_dir, f"{full_key}.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception as e:
                print(f"删除缓存文件失败: {e}")
    
    def clear(self, namespace: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            namespace: 要清除的命名空间，None表示清除所有缓存
        """
        # TODO: 实现缓存清理功能
        if namespace:
            # 清除特定命名空间的缓存
            # 1. 内存缓存
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{namespace}:")]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # 2. 磁盘缓存
            pattern = f"{namespace}:*.json"
            # 使用glob查找匹配的文件
            import glob
            for file_path in glob.glob(os.path.join(self.cache_dir, pattern)):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")
        else:
            # 清除所有缓存
            # 1. 内存缓存
            self.memory_cache.clear()
            
            # 2. 磁盘缓存
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file_name))
                    except Exception as e:
                        print(f"删除缓存文件失败: {e}")
    
    def _get_full_key(self, key: str, namespace: str) -> str:
        """
        获取完整的缓存键
        
        Args:
            key: 原始键
            namespace: 命名空间
        
        Returns:
            完整的缓存键
        """
        return f"{namespace}:{key}"
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """
        检查缓存是否已过期
        
        Args:
            cache_entry: 缓存条目
        
        Returns:
            如果缓存已过期则返回True，否则返回False
        """
        expire_time = cache_entry.get("expire_time", 0)
        return time.time() > expire_time

# 创建全局缓存实例
cache_manager = CacheManager()

# 缓存装饰器
def cache_result(ttl: Optional[int] = None, namespace: str = "function_results"):
    """
    缓存函数结果的装饰器
    
    Args:
        ttl: 缓存过期时间（秒）
        namespace: 缓存命名空间
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [func.__name__]
            # 添加参数作为键的一部分
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            cache_key = "_".join(key_parts)
            
            # 尝试从缓存获取结果
            cached_result = cache_manager.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache_manager.set(cache_key, result, namespace, ttl)
            
            return result
        
        return wrapper
    
    return decorator

# TODO: 实现更多缓存功能
# 1. 分布式缓存支持
# 2. 缓存统计和监控
# 3. LRU缓存策略
# 4. 缓存预热机制