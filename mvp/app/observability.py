#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可观测性模块
负责日志、指标和跟踪功能
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
import json
import os
from app.config import config
import threading

# 请求上下文本地存储
_local = threading.local()

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_demo")

class RequestContext:
    """
    请求上下文管理器
    用于存储和访问请求相关的上下文信息
    """
    def __init__(self, request_id: str):
        """
        初始化请求上下文
        
        Args:
            request_id: 请求ID
        """
        self.request_id = request_id
        self.old_context = None
    
    def __enter__(self):
        """进入上下文管理器"""
        # 保存旧的上下文（如果有）
        if hasattr(_local, 'context'):
            self.old_context = _local.context
        
        # 设置新的上下文
        _local.context = {
            'request_id': self.request_id,
            'start_time': time.time(),
            'attributes': {}
        }
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        # 恢复旧的上下文
        if self.old_context:
            _local.context = self.old_context
        else:
            # 如果没有旧上下文，则删除当前上下文
            if hasattr(_local, 'context'):
                delattr(_local, 'context')
    
    @staticmethod
    def get_request_id() -> Optional[str]:
        """
        获取当前请求ID
        
        Returns:
            请求ID或None
        """
        if hasattr(_local, 'context'):
            return _local.context.get('request_id')
        return None
    
    @staticmethod
    def get_attribute(key: str, default: Any = None) -> Any:
        """
        获取上下文属性
        
        Args:
            key: 属性键
            default: 默认值
        
        Returns:
            属性值或默认值
        """
        if hasattr(_local, 'context'):
            return _local.context.get('attributes', {}).get(key, default)
        return default
    
    @staticmethod
    def set_attribute(key: str, value: Any) -> None:
        """
        设置上下文属性
        
        Args:
            key: 属性键
            value: 属性值
        """
        if hasattr(_local, 'context'):
            _local.context.setdefault('attributes', {})[key] = value
    
    @staticmethod
    def get_context() -> Optional[Dict[str, Any]]:
        """
        获取完整上下文
        
        Returns:
            上下文字典或None
        """
        if hasattr(_local, 'context'):
            return _local.context.copy()
        return None

# 日志配置函数
def setup_logging():
    """
    设置日志系统
    配置日志级别、格式和输出位置
    """
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 设置日志级别
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    
    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    
    # 创建文件处理器（如果配置了日志目录）
    file_handler = None
    if config.LOG_DIR:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        log_file = os.path.join(config.LOG_DIR, f"rag_demo_{time.strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    
    # 设置日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(log_format)
    if file_handler:
        file_handler.setFormatter(log_format)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # 配置第三方库的日志级别
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)

# 请求日志记录函数
def log_request(event_name: str, data: Dict[str, Any]) -> None:
    """
    记录请求相关的日志，包括每步输入长度与输出大小
    
    Args:
        event_name: 事件名称
        data: 日志数据
    """
    # 获取当前请求上下文
    context = RequestContext.get_context()
    
    # 构建日志条目
    log_entry = {
        'event': event_name,
        'timestamp': time.time(),
        'data': data
    }
    
    # 添加请求ID（如果有）
    if context:
        log_entry['request_id'] = context.get('request_id')
        
        # 计算处理时间（如果有开始时间）
        if 'start_time' in context:
            log_entry['processing_time'] = time.time() - context['start_time']
    
    # 记录输入输出统计信息
    if isinstance(data, dict):
        stats = {}
        
        # 记录输入文本长度
        if 'query' in data and isinstance(data['query'], str):
            stats['input_length'] = len(data['query'])
            stats['input_lines'] = data['query'].count('\n') + 1
        
        # 记录输出大小或长度
        if 'results' in data or 'evidences' in data:
            results = data.get('results') or data.get('evidences')
            if isinstance(results, list):
                stats['output_count'] = len(results)
                # 计算输出总字符数和总行数
                total_chars = 0
                total_lines = 0
                for item in results:
                    if isinstance(item, dict):
                        content = str(item.get('content', ''))
                        total_chars += len(content)
                        total_lines += content.count('\n') + 1
                    elif hasattr(item, 'content'):
                        content = str(item.content)
                        total_chars += len(content)
                        total_lines += content.count('\n') + 1
                stats['output_total_chars'] = total_chars
                stats['output_total_lines'] = total_lines
        
        # 记录生成的回答信息
        if 'answer' in data and isinstance(data['answer'], str):
            stats['answer_length'] = len(data['answer'])
            stats['answer_lines'] = data['answer'].count('\n') + 1
        
        if stats:
            log_entry['stats'] = stats
    
    # 记录日志
    logger.info(json.dumps(log_entry, ensure_ascii=False))

class Tracer:
    """分布式追踪器"""
    
    def __init__(self):
        """初始化追踪器"""
        # TODO: 集成分布式追踪系统
        self.traces = {}
    
    def start_span(self, name: str, trace_id: Optional[str] = None) -> str:
        """
        开始一个新的追踪 span
        
        Args:
            name: span名称
            trace_id: 可选的跟踪ID
        
        Returns:
            span ID
        """
        # TODO: 实现分布式追踪
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        span = {
            "name": name,
            "span_id": span_id,
            "trace_id": trace_id,
            "start_time": time.time(),
            "events": [],
            "attributes": {},
            "children": []
        }
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        
        return span_id
    
    def end_span(self, span_id: str, trace_id: str) -> None:
        """
        结束一个追踪 span
        
        Args:
            span_id: span ID
            trace_id: 跟踪 ID
        """
        # TODO: 实现分布式追踪
        if trace_id in self.traces:
            for span in self.traces[trace_id]:
                if span["span_id"] == span_id:
                    span["end_time"] = time.time()
                    span["duration"] = span["end_time"] - span["start_time"]
                    break
    
    def add_event(self, span_id: str, trace_id: str, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        添加事件到追踪 span
        
        Args:
            span_id: span ID
            trace_id: 跟踪 ID
            name: 事件名称
            attributes: 事件属性
        """
        # TODO: 实现事件添加
        if trace_id in self.traces:
            for span in self.traces[trace_id]:
                if span["span_id"] == span_id:
                    event = {
                        "name": name,
                        "timestamp": time.time(),
                        "attributes": attributes or {}
                    }
                    span["events"].append(event)
                    break
    
    def add_attribute(self, span_id: str, trace_id: str, key: str, value: Any) -> None:
        """
        添加属性到追踪 span
        
        Args:
            span_id: span ID
            trace_id: 跟踪 ID
            key: 属性键
            value: 属性值
        """
        # TODO: 实现属性添加
        if trace_id in self.traces:
            for span in self.traces[trace_id]:
                if span["span_id"] == span_id:
                    span["attributes"][key] = value
                    break

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        # TODO: 集成指标系统
        self.metrics = {}
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        增加计数器指标
        
        Args:
            name: 指标名称
            value: 增加的值
            labels: 指标标签
        """
        # TODO: 实现指标收集
        key = self._get_metric_key(name, labels)
        if key not in self.metrics:
            self.metrics[key] = 0.0
        self.metrics[key] += value
        
        # 记录日志
        logger.debug(f"Counter increment: {name} by {value}, labels: {labels}")
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        观察直方图指标
        
        Args:
            name: 指标名称
            value: 观察的值
            labels: 指标标签
        """
        # TODO: 实现直方图指标
        logger.debug(f"Histogram observation: {name} = {value}, labels: {labels}")
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """
        获取指标的唯一键
        
        Args:
            name: 指标名称
            labels: 指标标签
        
        Returns:
            指标键
        """
        if labels:
            label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
            return f"{name}[{label_str}]"
        return name

class Logger:
    """日志记录器"""
    
    @staticmethod
    def log_query(query: str, query_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        记录查询日志
        
        Args:
            query: 查询文本
            query_id: 查询 ID
            metadata: 元数据
        """
        # TODO: 实现结构化日志记录
        log_data = {
            "query_id": query_id,
            "query": query,
            "metadata": metadata or {}
        }
        logger.info(f"Query received: {json.dumps(log_data, ensure_ascii=False)}")
    
    @staticmethod
    def log_response(query_id: str, response: Dict[str, Any], duration: float) -> None:
        """
        记录响应日志
        
        Args:
            query_id: 查询 ID
            response: 响应数据
            duration: 处理时间（秒）
        """
        # TODO: 实现结构化日志记录
        log_data = {
            "query_id": query_id,
            "response": response,
            "duration": duration
        }
        logger.info(f"Query completed: {json.dumps(log_data, ensure_ascii=False)}")
    
    @staticmethod
    def log_error(query_id: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        记录错误日志
        
        Args:
            query_id: 查询 ID
            error: 异常对象
            context: 上下文信息
        """
        # TODO: 实现结构化错误日志
        log_data = {
            "query_id": query_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        logger.error(f"Error processing query: {json.dumps(log_data, ensure_ascii=False)}")

# 创建全局实例
tracer = Tracer()
metrics_collector = MetricsCollector()
app_logger = Logger()


class Obs:
    """
    观测类，用于记录和追踪请求处理过程中的各个步骤、耗时和事件
    """
    
    def __init__(self, request_id: str = None, session_id: str = None):
        """
        初始化观测对象
        
        Args:
            request_id: 请求ID
            session_id: 会话ID
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.session_id = session_id
        self.start_time = time.time()
        self.steps = []  # 步骤列表
        self.events = []  # 事件列表
        self._current_step = None
        self._step_start_time = None
    
    def start_step(self, name: str):
        """
        开始一个新的处理步骤
        
        Args:
            name: 步骤名称
        """
        # 如果有未结束的步骤，先结束它
        if self._current_step:
            self.end_step()
        
        # 开始新步骤
        self._current_step = name
        self._step_start_time = time.time()
    
    def end_step(self):
        """
        结束当前处理步骤，并记录耗时
        """
        if self._current_step:
            duration = (time.time() - self._step_start_time) * 1000  # 转换为毫秒
            self.steps.append({
                "name": self._current_step,
                "dt_ms": duration,
                "start_time": self._step_start_time,
                "end_time": time.time()
            })
            
            # 重置当前步骤
            self._current_step = None
            self._step_start_time = None
    
    def add_event(self, name: str, data: Dict[str, Any] = None):
        """
        添加一个事件
        
        Args:
            name: 事件名称
            data: 事件数据
        """
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "data": data or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将观测数据转换为字典格式
        
        Returns:
            包含所有观测数据的字典
        """
        # 确保当前步骤已结束
        if self._current_step:
            self.end_step()
        
        total_duration = (time.time() - self.start_time) * 1000  # 总耗时（毫秒）
        
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_dt_ms": total_duration,
            "steps": self.steps,
            "events": self.events
        }
    
    def timings(self) -> Dict[str, float]:
        """
        返回各个步骤的耗时统计
        
        Returns:
            步骤名称到耗时（毫秒）的映射
        """
        # 确保当前步骤已结束
        if self._current_step:
            self.end_step()
        
        return {step["name"]: step["dt_ms"] for step in self.steps}
    
    def write_to_log(self):
        """
        将观测数据写入日志文件（JSONL格式）
        """
        try:
            # 确保日志目录存在
            if config.LOG_DIR:
                os.makedirs(config.LOG_DIR, exist_ok=True)
                
                # 生成日志文件名（YYYYMMDD格式）
                log_file = os.path.join(config.LOG_DIR, f"rag_demo_{time.strftime('%Y%m%d')}.jsonl")
                
                # 将观测数据转换为字典并写入文件
                log_entry = self.to_dict()
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to write observability data to log: {str(e)}")
            return False


# 全局观测实例
obs = Obs()

# 工具函数：追踪装饰器
def trace_function(name: Optional[str] = None):
    """
    用于追踪函数执行的装饰器
    
    Args:
        name: 追踪名称
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            # 获取或生成 trace_id
            trace_id = kwargs.get("trace_id", str(uuid.uuid4()))
            
            # 开始追踪
            span_id = tracer.start_span(func_name, trace_id)
            
            try:
                # 执行函数
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 更新指标
                metrics_collector.increment_counter(f"function_calls", labels={"function": func_name})
                metrics_collector.observe_histogram(f"function_duration", duration, labels={"function": func_name})
                
                # 添加属性
                tracer.add_attribute(span_id, trace_id, "duration", duration)
                tracer.add_attribute(span_id, trace_id, "success", True)
                
                return result
            except Exception as e:
                # 记录错误
                tracer.add_attribute(span_id, trace_id, "success", False)
                tracer.add_attribute(span_id, trace_id, "error", str(e))
                metrics_collector.increment_counter(f"function_errors", labels={"function": func_name, "error_type": type(e).__name__})
                raise
            finally:
                # 结束追踪
                tracer.end_span(span_id, trace_id)
        
        return wrapper
    
    return decorator

# TODO: 实现更多可观测性功能
# 1. 集成 Prometheus
# 2. 集成 OpenTelemetry
# 3. 健康检查端点
# 4. 性能监控面板