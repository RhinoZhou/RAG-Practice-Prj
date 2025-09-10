#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海量文档处理流水线模拟器（带断点续传）

功能：
- 模拟海量文档的解析→分块→向量化→入索引的流水线
- K8s弹性伸缩策略（worker数量随队列深度调节）
- 断点续传机制（检查点快照）
- 任务队列管理（消息中间件模拟）

主要内容：
- 文档队列入库（模拟2.3TB/日速率）
- Dispatcher划分任务，Workers并发执行
- 队列深度监控，动态调节并发（扩/缩容）
- 失败任务重试与断点恢复
- 索引批量刷新与合并

输入：文档元数据与内容、并发上限、SLA目标
输出：吞吐曲线、P50/P95延迟、失败重试日志、检查点快照
"""

import json
import time
import random
import threading
import queue
import uuid
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from enum import Enum

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 数据结构定义
# -----------------------------

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 待处理
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败
    RETRYING = "retrying"     # 重试中

class WorkerStatus(Enum):
    """Worker状态枚举"""
    IDLE = "idle"            # 空闲
    BUSY = "busy"            # 忙碌
    SCALING_UP = "scaling_up" # 扩容中
    SCALING_DOWN = "scaling_down" # 缩容中

@dataclass
class DocumentMetadata:
    """文档元数据"""
    doc_id: str
    filename: str
    size_mb: float
    doc_type: str  # pdf, txt, docx等
    priority: int = 1  # 优先级，1-5
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class ProcessingTask:
    """处理任务"""
    task_id: str
    doc_metadata: DocumentMetadata
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> Optional[float]:
        """处理时间（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

@dataclass
class WorkerMetrics:
    """Worker指标"""
    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    current_task_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    queue_depth: int
    active_workers: int
    total_workers: int
    throughput_per_minute: float
    p50_latency: float
    p95_latency: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float

@dataclass
class CheckpointSnapshot:
    """检查点快照"""
    snapshot_id: str
    timestamp: datetime
    completed_tasks: List[str]  # 已完成任务ID列表
    failed_tasks: List[str]     # 失败任务ID列表
    queue_state: Dict[str, Any] # 队列状态
    worker_states: Dict[str, Dict] # Worker状态
    system_metrics: SystemMetrics
    
# -----------------------------
# 核心组件
# -----------------------------

class DocumentQueue:
    """文档队列（模拟消息中间件）
    
    负责管理待处理的文档任务，支持优先级队列和任务状态跟踪。
    使用线程安全的队列实现，支持高并发场景下的任务调度。
    """
    
    def __init__(self, max_size: int = 10000):
        """初始化文档队列
        
        Args:
            max_size: 队列最大容量，防止内存溢出
        """
        self.queue = queue.PriorityQueue(maxsize=max_size)  # 优先级队列，按优先级处理
        self.pending_tasks = {}    # 待处理任务映射
        self.completed_tasks = {}  # 已完成任务缓存
        self.failed_tasks = {}     # 失败任务记录
        self.lock = threading.Lock()  # 线程锁，保证并发安全
        
    def enqueue(self, task: ProcessingTask) -> bool:
        """入队任务"""
        try:
            # 使用负优先级实现高优先级优先处理
            priority = -task.doc_metadata.priority
            self.queue.put((priority, task.task_id, task), timeout=1)
            
            with self.lock:
                self.pending_tasks[task.task_id] = task
            return True
        except queue.Full:
            return False
    
    def dequeue(self, timeout: float = 1.0) -> Optional[ProcessingTask]:
        """出队任务"""
        try:
            priority, task_id, task = self.queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None
    
    def mark_completed(self, task_id: str, task: ProcessingTask):
        """标记任务完成"""
        with self.lock:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            self.completed_tasks[task_id] = task
    
    def mark_failed(self, task_id: str, task: ProcessingTask):
        """标记任务失败"""
        with self.lock:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            self.failed_tasks[task_id] = task
    
    def get_depth(self) -> int:
        """获取队列深度"""
        return self.queue.qsize()
    
    def get_stats(self) -> Dict[str, int]:
        """获取队列统计信息"""
        with self.lock:
            return {
                "pending": len(self.pending_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "queue_depth": self.get_depth()
            }

class DocumentProcessor:
    """文档处理器（模拟解析→分块→向量化流程）"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.processing_times = {
            "pdf": (2.0, 8.0),    # PDF处理时间范围（秒）
            "txt": (0.5, 2.0),    # 文本处理时间范围
            "docx": (1.0, 4.0),   # Word文档处理时间范围
            "default": (1.0, 5.0) # 默认处理时间范围
        }
    
    def process_document(self, task: ProcessingTask) -> Tuple[bool, Optional[str]]:
        """处理文档
        
        返回:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息)
        """
        doc = task.doc_metadata
        
        # 模拟处理时间（基于文档类型和大小）
        base_time_range = self.processing_times.get(doc.doc_type, self.processing_times["default"])
        base_time = random.uniform(*base_time_range)
        
        # 文档大小影响处理时间
        size_factor = max(0.1, doc.size_mb / 10.0)  # 10MB为基准
        processing_time = base_time * size_factor
        
        # 模拟处理步骤
        steps = [
            ("解析文档", 0.2),
            ("文本分块", 0.3),
            ("向量化", 0.4),
            ("索引入库", 0.1)
        ]
        
        try:
            for step_name, time_ratio in steps:
                step_time = processing_time * time_ratio
                time.sleep(step_time)
                
                # 模拟随机失败（5%失败率）
                if random.random() < 0.05:
                    error_msg = f"处理失败在步骤: {step_name}"
                    return False, error_msg
                
                # 更新检查点
                step_name_en = {
                    "解析文档": "parse_document",
                    "文本分块": "text_chunking", 
                    "向量化": "vectorization",
                    "索引入库": "index_storage"
                }.get(step_name, step_name)
                
                task.checkpoint_data[step_name_en] = {
                    "completed_at": datetime.now().isoformat(),
                    "processing_time": step_time
                }
            
            return True, None
            
        except Exception as e:
            return False, str(e)

class K8sScaler:
    """K8s弹性伸缩器
    
    模拟Kubernetes HPA（Horizontal Pod Autoscaler）功能，
    根据队列深度和worker负载自动调整worker数量。
    实现智能的扩缩容策略，确保系统性能和资源利用率的平衡。
    """
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20):
        """初始化弹性伸缩器
        
        Args:
            min_workers: 最小worker数量，保证基础处理能力
            max_workers: 最大worker数量，防止资源过度消耗
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers  # 当前worker数量
        self.scaling_history = []  # 伸缩历史记录，用于分析和调优
        
    def should_scale_up(self, queue_depth: int, active_workers: int) -> bool:
        """判断是否需要扩容"""
        # 队列深度超过worker数量的2倍时扩容
        if queue_depth > active_workers * 2 and self.current_workers < self.max_workers:
            return True
        return False
    
    def should_scale_down(self, queue_depth: int, active_workers: int) -> bool:
        """判断是否需要缩容"""
        # 队列深度小于worker数量的0.5倍且worker数量大于最小值时缩容
        if queue_depth < active_workers * 0.5 and self.current_workers > self.min_workers:
            return True
        return False
    
    def scale_up(self, target_workers: Optional[int] = None) -> int:
        """扩容"""
        if target_workers is None:
            target_workers = min(self.current_workers + 2, self.max_workers)
        else:
            target_workers = min(target_workers, self.max_workers)
        
        old_count = self.current_workers
        self.current_workers = target_workers
        
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "action": "scale_up",
            "from": old_count,
            "to": self.current_workers
        })
        
        return self.current_workers - old_count
    
    def scale_down(self, target_workers: Optional[int] = None) -> int:
        """缩容"""
        if target_workers is None:
            target_workers = max(self.current_workers - 1, self.min_workers)
        else:
            target_workers = max(target_workers, self.min_workers)
        
        old_count = self.current_workers
        self.current_workers = target_workers
        
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "action": "scale_down",
            "from": old_count,
            "to": self.current_workers
        })
        
        return old_count - self.current_workers

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.snapshots = {}
    
    def create_snapshot(self, doc_queue: DocumentQueue, workers: Dict[str, WorkerMetrics], 
                       system_metrics: SystemMetrics) -> str:
        """创建检查点快照"""
        snapshot_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 收集队列状态
        queue_stats = doc_queue.get_stats()
        
        # 创建快照
        snapshot = CheckpointSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            completed_tasks=list(doc_queue.completed_tasks.keys()),
            failed_tasks=list(doc_queue.failed_tasks.keys()),
            queue_state=queue_stats,
            worker_states={wid: asdict(worker) for wid, worker in workers.items()},
            system_metrics=system_metrics
        )
        
        # 保存到文件
        snapshot_file = os.path.join(self.checkpoint_dir, f"{snapshot_id}.pkl")
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot, f)
        
        self.snapshots[snapshot_id] = snapshot
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> Optional[CheckpointSnapshot]:
        """加载检查点快照"""
        snapshot_file = os.path.join(self.checkpoint_dir, f"{snapshot_id}.pkl")
        if os.path.exists(snapshot_file):
            with open(snapshot_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def list_snapshots(self) -> List[str]:
        """列出所有快照"""
        snapshots = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pkl'):
                snapshots.append(filename[:-4])  # 移除.pkl扩展名
        return sorted(snapshots)

# -----------------------------
# 主要流水线系统
# -----------------------------

class MassivePipelineSimulator:
    """海量文档处理流水线模拟器
    
    这是一个企业级文档处理系统的核心模拟器，具备以下特性：
    1. 高并发文档处理：支持多worker并行处理
    2. 智能弹性伸缩：基于负载自动调整资源
    3. 断点续传机制：支持任务失败恢复和检查点
    4. 实时监控告警：提供详细的性能指标和可视化
    5. 容错和重试：自动处理失败任务和异常恢复
    
    适用于处理TB级别的文档数据，满足企业级SLA要求。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化流水线模拟器
        
        Args:
            config: 配置参数字典，包含队列大小、worker数量等设置
        """
        self.config = config
        # 初始化核心组件
        self.doc_queue = DocumentQueue(max_size=config.get("max_queue_size", 10000))
        self.scaler = K8sScaler(
            min_workers=config.get("min_workers", 2),
            max_workers=config.get("max_workers", 20)
        )
        self.checkpoint_manager = CheckpointManager()
        
        # Worker管理
        self.workers = {}  # worker_id -> WorkerMetrics
        self.worker_pool = None  # 线程池执行器
        self.running = False  # 运行状态标志
        
        # 指标收集和监控
        self.metrics_history = []  # 系统指标历史记录
        self.throughput_history = deque(maxlen=60)  # 保留最近60分钟的吞吐量
        self.latency_history = deque(maxlen=1000)   # 保留最近1000个任务的延迟
        
        # 初始化workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """初始化workers"""
        for i in range(self.scaler.current_workers):
            worker_id = f"worker_{i:03d}"
            self.workers[worker_id] = WorkerMetrics(worker_id=worker_id)
    
    def _generate_documents(self, daily_volume_tb: float = 2.3) -> None:
        """生成文档任务（模拟2.3TB/日的速率）"""
        # 计算每分钟应该生成的文档数量
        daily_volume_mb = daily_volume_tb * 1024 * 1024  # 转换为MB
        avg_doc_size_mb = 50  # 假设平均文档大小50MB
        docs_per_day = daily_volume_mb / avg_doc_size_mb
        docs_per_minute = docs_per_day / (24 * 60)
        
        doc_types = ["pdf", "txt", "docx"]
        doc_type_weights = [0.6, 0.3, 0.1]  # PDF占60%，文本30%，Word 10%
        
        while self.running:
            try:
                # 每分钟生成相应数量的文档
                for _ in range(int(docs_per_minute)):
                    if not self.running:
                        break
                    
                    # 生成文档元数据
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                    doc_type = np.random.choice(doc_types, p=doc_type_weights)
                    size_mb = max(1.0, np.random.lognormal(mean=3.5, sigma=1.0))  # 对数正态分布
                    priority = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                    
                    doc_metadata = DocumentMetadata(
                        doc_id=doc_id,
                        filename=f"{doc_id}.{doc_type}",
                        size_mb=size_mb,
                        doc_type=doc_type,
                        priority=priority
                    )
                    
                    # 创建处理任务
                    task = ProcessingTask(
                        task_id=f"task_{uuid.uuid4().hex[:12]}",
                        doc_metadata=doc_metadata
                    )
                    
                    # 入队
                    if not self.doc_queue.enqueue(task):
                        print(f"队列已满，跳过任务: {task.task_id}")
                
                # 等待下一分钟
                time.sleep(60)
                
            except Exception as e:
                print(f"文档生成器错误: {e}")
                time.sleep(1)
    
    def _process_tasks(self):
        """处理任务的主循环"""
        while self.running:
            try:
                # 获取任务
                task = self.doc_queue.dequeue(timeout=1.0)
                if task is None:
                    continue
                
                # 分配worker
                available_worker = self._get_available_worker()
                if available_worker is None:
                    # 没有可用worker，重新入队
                    self.doc_queue.enqueue(task)
                    time.sleep(0.1)
                    continue
                
                # 提交任务到线程池
                future = self.worker_pool.submit(self._execute_task, task, available_worker)
                
            except Exception as e:
                print(f"任务处理错误: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: ProcessingTask, worker_id: str) -> None:
        """执行单个任务"""
        try:
            # 更新任务状态
            task.status = TaskStatus.PROCESSING
            task.worker_id = worker_id
            task.start_time = datetime.now()
            
            # 更新worker状态
            worker = self.workers[worker_id]
            worker.status = WorkerStatus.BUSY
            worker.current_task_id = task.task_id
            worker.last_heartbeat = datetime.now()
            
            # 创建处理器并执行任务
            processor = DocumentProcessor(worker_id)
            success, error_msg = processor.process_document(task)
            
            # 更新任务结束状态
            task.end_time = datetime.now()
            
            if success:
                task.status = TaskStatus.COMPLETED
                self.doc_queue.mark_completed(task.task_id, task)
                worker.tasks_completed += 1
                
                # 记录延迟
                if task.processing_time:
                    self.latency_history.append(task.processing_time)
                
            else:
                task.status = TaskStatus.FAILED
                task.error_message = error_msg
                task.retry_count += 1
                
                # 重试逻辑
                if task.retry_count < self.config.get("max_retries", 3):
                    task.status = TaskStatus.RETRYING
                    # 延迟重试
                    time.sleep(task.retry_count * 2)  # 指数退避
                    self.doc_queue.enqueue(task)
                else:
                    self.doc_queue.mark_failed(task.task_id, task)
                    worker.tasks_failed += 1
            
            # 更新worker状态
            worker.status = WorkerStatus.IDLE
            worker.current_task_id = None
            worker.total_processing_time += task.processing_time or 0
            worker.last_heartbeat = datetime.now()
            
        except Exception as e:
            print(f"任务执行错误 {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.doc_queue.mark_failed(task.task_id, task)
    
    def _get_available_worker(self) -> Optional[str]:
        """获取可用的worker"""
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.IDLE:
                return worker_id
        return None
    
    def _monitor_and_scale(self):
        """监控系统并执行弹性伸缩"""
        while self.running:
            try:
                # 收集指标
                queue_depth = self.doc_queue.get_depth()
                active_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
                total_workers = len(self.workers)
                
                # 计算吞吐量（每分钟完成的任务数）
                current_time = datetime.now()
                completed_in_last_minute = len([
                    task for task in self.doc_queue.completed_tasks.values()
                    if task.end_time and (current_time - task.end_time).total_seconds() <= 60
                ])
                self.throughput_history.append(completed_in_last_minute)
                
                # 计算延迟指标
                if self.latency_history:
                    latencies = list(self.latency_history)
                    p50_latency = np.percentile(latencies, 50)
                    p95_latency = np.percentile(latencies, 95)
                else:
                    p50_latency = p95_latency = 0.0
                
                # 计算错误率
                total_tasks = len(self.doc_queue.completed_tasks) + len(self.doc_queue.failed_tasks)
                error_rate = len(self.doc_queue.failed_tasks) / max(1, total_tasks)
                
                # 模拟系统资源使用
                memory_usage_mb = total_workers * 512 + queue_depth * 10  # 简化计算
                cpu_usage_percent = min(95, active_workers / total_workers * 100)
                
                # 创建系统指标
                metrics = SystemMetrics(
                    timestamp=current_time,
                    queue_depth=queue_depth,
                    active_workers=active_workers,
                    total_workers=total_workers,
                    throughput_per_minute=np.mean(self.throughput_history) if self.throughput_history else 0,
                    p50_latency=p50_latency,
                    p95_latency=p95_latency,
                    error_rate=error_rate,
                    memory_usage_mb=memory_usage_mb,
                    cpu_usage_percent=cpu_usage_percent
                )
                
                self.metrics_history.append(metrics)
                
                # 弹性伸缩决策
                if self.scaler.should_scale_up(queue_depth, active_workers):
                    added_workers = self.scaler.scale_up()
                    self._add_workers(added_workers)
                    print(f"扩容: 添加 {added_workers} 个worker，当前总数: {self.scaler.current_workers}")
                
                elif self.scaler.should_scale_down(queue_depth, active_workers):
                    removed_workers = self.scaler.scale_down()
                    self._remove_workers(removed_workers)
                    print(f"缩容: 移除 {removed_workers} 个worker，当前总数: {self.scaler.current_workers}")
                
                # 定期创建检查点
                if len(self.metrics_history) % 10 == 0:  # 每10次监控创建一次检查点
                    snapshot_id = self.checkpoint_manager.create_snapshot(
                        self.doc_queue, self.workers, metrics
                    )
                    print(f"创建检查点: {snapshot_id}")
                
                # 定期保存报告（每4次监控，即2分钟）
                if len(self.metrics_history) % 4 == 0:
                    self._save_interim_report()
                
                # 打印监控信息
                print(f"监控 - 队列深度: {queue_depth}, 活跃Worker: {active_workers}/{total_workers}, "
                      f"吞吐量: {metrics.throughput_per_minute:.1f}/min, "
                      f"P50延迟: {p50_latency:.2f}s, P95延迟: {p95_latency:.2f}s")
                
                time.sleep(30)  # 每30秒监控一次
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(5)
    
    def _add_workers(self, count: int):
        """添加workers"""
        current_count = len(self.workers)
        for i in range(count):
            worker_id = f"worker_{current_count + i:03d}"
            self.workers[worker_id] = WorkerMetrics(worker_id=worker_id)
    
    def _remove_workers(self, count: int):
        """移除workers（只移除空闲的）"""
        idle_workers = [wid for wid, w in self.workers.items() if w.status == WorkerStatus.IDLE]
        to_remove = idle_workers[:count]
        
        for worker_id in to_remove:
            del self.workers[worker_id]
    
    def start(self):
        """启动流水线"""
        print("启动海量文档处理流水线...")
        self.running = True
        
        # 创建线程池
        self.worker_pool = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 20))
        
        # 启动各个组件线程
        threads = [
            threading.Thread(target=self._generate_documents, daemon=True),
            threading.Thread(target=self._process_tasks, daemon=True),
            threading.Thread(target=self._monitor_and_scale, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        return threads
    
    def stop(self):
        """停止流水线"""
        print("停止海量文档处理流水线...")
        self.running = False
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
    
    def _save_interim_report(self):
        """保存中间报告"""
        try:
            report = self.get_summary_report()
            if "error" not in report:
                os.makedirs("results", exist_ok=True)
                report_path = "results/pipeline_summary_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                print(f"已保存中间报告到: {report_path}")
        except Exception as e:
            print(f"保存中间报告失败: {e}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """获取摘要报告"""
        if not self.metrics_history:
            return {"error": "没有可用的指标数据"}
        
        latest_metrics = self.metrics_history[-1]
        queue_stats = self.doc_queue.get_stats()
        
        # 计算平均指标
        avg_throughput = np.mean([m.throughput_per_minute for m in self.metrics_history])
        avg_p50_latency = np.mean([m.p50_latency for m in self.metrics_history])
        avg_p95_latency = np.mean([m.p95_latency for m in self.metrics_history])
        avg_error_rate = np.mean([m.error_rate for m in self.metrics_history])
        
        return {
            "runtime_minutes": len(self.metrics_history) * 0.5,  # 每30秒一次监控
            "current_status": {
                "queue_depth": latest_metrics.queue_depth,
                "active_workers": latest_metrics.active_workers,
                "total_workers": latest_metrics.total_workers,
                "current_throughput": latest_metrics.throughput_per_minute
            },
            "task_statistics": queue_stats,
            "average_performance_metrics": {
                "avg_throughput_per_minute": avg_throughput,
                "avg_p50_latency_seconds": avg_p50_latency,
                "avg_p95_latency_seconds": avg_p95_latency,
                "avg_error_rate": avg_error_rate
            },
            "scaling_history": self.scaler.scaling_history,
            "checkpoint_snapshots": self.checkpoint_manager.list_snapshots()
        }

# -----------------------------
# 可视化和报告
# -----------------------------

def plot_throughput_curve(metrics_history: List[SystemMetrics], output_path: str = "results/throughput_curve.png"):
    """绘制吞吐量曲线"""
    if not metrics_history:
        print("没有指标数据，无法绘制吞吐量曲线")
        return
    
    plt.figure(figsize=(12, 8))
    
    timestamps = [m.timestamp for m in metrics_history]
    throughputs = [m.throughput_per_minute for m in metrics_history]
    queue_depths = [m.queue_depth for m in metrics_history]
    worker_counts = [m.total_workers for m in metrics_history]
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 吞吐量曲线
    ax1.plot(timestamps, throughputs, 'b-', linewidth=2, label='吞吐量')
    ax1.set_ylabel('吞吐量 (任务/分钟)')
    ax1.set_title('系统性能监控')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 队列深度
    ax2.plot(timestamps, queue_depths, 'r-', linewidth=2, label='队列深度')
    ax2.set_ylabel('队列深度')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Worker数量
    ax3.plot(timestamps, worker_counts, 'g-', linewidth=2, label='Worker数量')
    ax3.set_ylabel('Worker数量')
    ax3.set_xlabel('时间')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"吞吐量曲线已保存至: {output_path}")

def plot_latency_distribution(metrics_history: List[SystemMetrics], output_path: str = "results/latency_distribution.png"):
    """绘制延迟分布图"""
    if not metrics_history:
        print("没有指标数据，无法绘制延迟分布")
        return
    
    plt.figure(figsize=(12, 6))
    
    p50_latencies = [m.p50_latency for m in metrics_history]
    p95_latencies = [m.p95_latency for m in metrics_history]
    timestamps = [m.timestamp for m in metrics_history]
    
    plt.plot(timestamps, p50_latencies, 'b-', linewidth=2, label='P50延迟')
    plt.plot(timestamps, p95_latencies, 'r-', linewidth=2, label='P95延迟')
    
    plt.xlabel('时间')
    plt.ylabel('延迟 (秒)')
    plt.title('任务处理延迟分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"延迟分布图已保存至: {output_path}")

def generate_failure_retry_log(doc_queue: DocumentQueue, output_path: str = "results/failure_retry_log.json"):
    """生成失败重试日志"""
    failed_tasks = []
    
    for task_id, task in doc_queue.failed_tasks.items():
        failed_tasks.append({
            "task_id": task_id,
            "doc_id": task.doc_metadata.doc_id,
            "filename": task.doc_metadata.filename,
            "doc_type": task.doc_metadata.doc_type,
            "size_mb": task.doc_metadata.size_mb,
            "retry_count": task.retry_count,
            "error_message": task.error_message,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "worker_id": task.worker_id,
            "checkpoint_data": task.checkpoint_data
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_failed_tasks": len(failed_tasks),
            "failed_tasks": failed_tasks
        }, f, ensure_ascii=False, indent=2)
    
    print(f"失败重试日志已保存至: {output_path}")

# -----------------------------
# 主函数
# -----------------------------

def main():
    print("=== 海量文档处理流水线模拟器 ===")
    
    # 配置参数
    config = {
        "max_queue_size": 5000,
        "min_workers": 3,
        "max_workers": 15,
        "max_retries": 3,
        "daily_volume_tb": 2.3,  # 每日处理量2.3TB
        "sla_target_latency": 300,  # SLA目标延迟300秒
        "sla_target_throughput": 100  # SLA目标吞吐量100任务/分钟
    }
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 创建模拟器
    simulator = MassivePipelineSimulator(config)
    
    try:
        # 启动模拟器
        threads = simulator.start()
        
        # 运行指定时间（例如10分钟用于演示）
        print(f"模拟器运行中，将运行 {config.get('simulation_duration_minutes', 10)} 分钟...")
        print("按 Ctrl+C 停止模拟")
        
        # 模拟运行时间
        simulation_duration = config.get('simulation_duration_minutes', 10) * 60
        time.sleep(simulation_duration)
        
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    
    finally:
        # 停止模拟器
        simulator.stop()
        
        # 等待线程结束
        time.sleep(2)
        
        print("\n=== 生成报告 ===")
        
        # 生成摘要报告
        summary = simulator.get_summary_report()
        with open("results/pipeline_summary_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print("摘要报告已保存至: results/pipeline_summary_report.json")
        
        # 生成可视化图表
        if simulator.metrics_history:
            plot_throughput_curve(simulator.metrics_history)
            plot_latency_distribution(simulator.metrics_history)
        
        # 生成失败重试日志
        generate_failure_retry_log(simulator.doc_queue)
        
        # 保存指标历史
        metrics_data = []
        for metrics in simulator.metrics_history:
            metrics_data.append({
                "timestamp": metrics.timestamp.isoformat(),
                "queue_depth": metrics.queue_depth,
                "active_workers": metrics.active_workers,
                "total_workers": metrics.total_workers,
                "throughput_per_minute": metrics.throughput_per_minute,
                "p50_latency": metrics.p50_latency,
                "p95_latency": metrics.p95_latency,
                "error_rate": metrics.error_rate,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent
            })
        
        with open("results/metrics_history.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        print("指标历史已保存至: results/metrics_history.json")
        
        # 打印最终统计
        print("\n=== 最终统计 ===")
        for key, value in summary.items():
            if key != "scaling_history" and key != "checkpoint_snapshots":
                print(f"{key}: {value}")
        
        print("\n=== 海量文档处理流水线模拟完成 ===")

if __name__ == "__main__":
    main()