#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载模块
负责加载项目路径、规则文件等配置
"""

import os
from dotenv import load_dotenv
from typing import Optional

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据和索引路径
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(PROJECT_ROOT, "index"))
    
    # 配置文件路径
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    ROUTING_RULES_PATH = os.path.join(CONFIG_DIR, "routing_rules.yaml")
    ENUMS_PATH = os.path.join(CONFIG_DIR, "enums.yaml")
    
    # API配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
    
    # 检索配置
    TOP_K = int(os.getenv("TOP_K", "10"))
    
    # 开关配置 - 用于本地降级
    USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"
    USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
    USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
    USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "true").lower() == "true"
    DEGRADATION_MODE = os.getenv("DEGRADATION_MODE", "false").lower() == "true"
    
    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        for dir_path in [
            cls.DATA_DIR,
            os.path.join(cls.DATA_DIR, "policy"),
            os.path.join(cls.DATA_DIR, "faq"),
            os.path.join(cls.DATA_DIR, "sop"),
            cls.INDEX_DIR,
            cls.CONFIG_DIR,
            cls.LOG_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)

# 全局配置实例
config = Config()

# 确保必要的目录存在
config.ensure_directories()

import yaml
from typing import Dict, Any, Optional, List
import time
import threading

class YAMLConfigLoader:
    """YAML配置加载器"""
    def __init__(self):
        self._config_cache = {}
        self._last_modified = {}
        self._lock = threading.Lock()
    
    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载YAML文件并返回配置字典"""
        with self._lock:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"配置文件不存在: {file_path}")
            
            # 检查文件是否有更新
            current_modified = os.path.getmtime(file_path)
            if file_path in self._last_modified and current_modified <= self._last_modified[file_path]:
                return self._config_cache[file_path]
            
            # 加载YAML文件
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # 更新缓存和最后修改时间
            self._config_cache[file_path] = config
            self._last_modified[file_path] = current_modified
            
            return config

# 全局YAML配置加载器实例
yaml_loader = YAMLConfigLoader()

class RoutingRules:
    """路由规则配置类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policy_priority = config.get("policy_priority", False)
        self.safe_view_only = config.get("safe_view_only", False)
        self.degrade_mode = config.get("degrade_mode", False)
        
    def get_strategy_for_query_type(self, query_type: str) -> Optional[Dict[str, Any]]:
        """获取特定查询类型的策略"""
        return self.config.get("query_type_strategies", {}).get(query_type)
    
    def get_domain_rules(self, domain: str) -> Optional[Dict[str, Any]]:
        """获取特定领域的规则"""
        return self.config.get("domain_rules", {}).get(domain)

class EnumsConfig:
    """枚举配置类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.platforms = self._load_enum("platforms")
        self.regions = self._load_enum("regions")
        self.retrieval_strategies = self._load_enum("retrieval_strategies")
        self.document_types = self._load_enum("document_types")
        self.query_types = self._load_enum("query_types")
    
    def _load_enum(self, enum_name: str) -> Dict[str, Dict[str, Any]]:
        """加载枚举配置"""
        enum_list = self.config.get(enum_name, [])
        return {item.get("value"): item for item in enum_list if "value" in item}
    
    def get_enum_value(self, enum_name: str, enum_key: str) -> Optional[str]:
        """获取枚举值"""
        enum_dict = getattr(self, enum_name, {})
        return enum_dict.get(enum_key, {}).get("value")

# 加载路由规则配置
def load_routing_rules() -> RoutingRules:
    """加载路由规则配置"""
    routing_rules_config = yaml_loader.load_yaml(config.ROUTING_RULES_PATH)
    return RoutingRules(routing_rules_config)

# 加载枚举配置
def load_enums() -> EnumsConfig:
    """加载枚举配置"""
    enums_config = yaml_loader.load_yaml(config.ENUMS_PATH)
    return EnumsConfig(enums_config)

# 全局配置实例
routing_rules = load_routing_rules()
enums = load_enums()

# 配置热更新功能
class ConfigMonitor:
    """配置监控器"""
    def __init__(self, check_interval: float = 30.0):  # 30秒检查一次
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 重新加载配置
                global routing_rules, enums
                routing_rules = load_routing_rules()
                enums = load_enums()
            except Exception as e:
                print(f"配置热更新失败: {e}")
            
            # 等待下一次检查
            time.sleep(self.check_interval)

# 创建配置监控器
config_monitor = ConfigMonitor()

# 启动配置监控
def start_config_monitor():
    """启动配置监控"""
    config_monitor.start()

# 停止配置监控
def stop_config_monitor():
    """停止配置监控"""
    config_monitor.stop()