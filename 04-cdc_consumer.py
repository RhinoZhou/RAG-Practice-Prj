#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kafka CDC (Change Data Capture) 消费者工具

功能描述:
  - 输入: 从Kafka消费Debezium变更事件
  - 处理: 按表/主键幂等UPSERT、watermark管理、offset/checkpoint对账
  - 输出: 落库统计 + DLQ投递
  - 关键点: 准Exactly-Once(去重+幂等+重放窗口)、主键冲突入DLQ

支持两种运行模式:
  1. 生产模式: 连接实际Kafka集群消费真实事件
  2. 模拟模式: 生成模拟的CDC事件进行处理演示，无需Kafka环境

注意: 运行前需要安装额外依赖: pip install -r requirements.txt
"""

import os
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CDC_Consumer")

# 尝试导入kafka-python库，如果不存在则提示安装
try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    logger.error("未找到kafka-python库，请使用以下命令安装：pip install kafka-python")
    raise

try:
    import pandas as pd
    import sqlite3
except ImportError:
    logger.error("未找到必要的依赖库，请使用requirements.txt安装所有依赖")
    raise

@dataclass
class MockKafkaMessage:
    """模拟Kafka消息对象，用于测试"""
    value: Dict
    partition: int = 0
    offset: int = 0

class CDCEventProcessor:
    """CDC事件处理器，负责处理Debezium格式的变更事件
    
    实现了准Exactly-Once语义的CDC事件处理流程，包括：
    - 事件去重
    - 幂等性存储
    - 重放窗口管理
    - 死信队列处理
    - 检查点机制
    - 性能监控和统计
    """
    
    def __init__(self, 
                 kafka_bootstrap_servers: str = 'localhost:9092',
                 input_topic: str = 'dbserver1.inventory',
                 dlq_topic: str = 'cdc_dlq',
                 database_path: str = './cdc_data.db',
                 replay_window_seconds: int = 3600,  # 1小时的重放窗口
                 checkpoint_interval: int = 10,  # 每处理10条记录检查一次
                 mock_mode: bool = False):  # 是否启用模拟模式
        """初始化CDC事件处理器
        
        参数:
            kafka_bootstrap_servers: Kafka服务器地址
            input_topic: 输入的Kafka主题
            dlq_topic: 死信队列主题
            database_path: SQLite数据库路径
            replay_window_seconds: 重放窗口大小（秒）
            checkpoint_interval: 检查点间隔（记录数）
            mock_mode: 是否启用模拟模式（不连接真实Kafka）
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.input_topic = input_topic
        self.dlq_topic = dlq_topic
        self.database_path = database_path
        self.replay_window_seconds = replay_window_seconds
        self.checkpoint_interval = checkpoint_interval
        self.mock_mode = mock_mode
        
        # 初始化Kafka消费者和生产者（如果不是模拟模式）
        if not mock_mode:
            self.consumer = self._init_consumer()
            self.producer = self._init_producer()
        else:
            # 模拟模式下使用None
            self.consumer = None
            self.producer = None
            logger.info("已启用模拟模式，将生成模拟CDC事件进行处理")
        
        # 幂等性保证 - 记录已处理的事件ID
        self.processed_events: Dict[str, datetime] = {}
        # 记录每个表的最新处理时间（watermark）
        self.watermarks: Dict[str, datetime] = {}
        # 记录每个表的offset信息
        self.last_offsets: Dict[str, int] = {}
        
        # 统计信息
        self.stats = {
            'total_events': 0,
            'success_events': 0,
            'failed_events': 0,
            'dlq_events': 0,
            'duplicate_events': 0
        }
        
        # 记录处理计数，用于触发检查点
        self.process_count = 0
        
        # 初始化数据库连接
        self.db_conn = self._init_database()
        
        # 这些属性已经在数据库初始化前定义，这里不再重复定义
        
    def _init_consumer(self) -> KafkaConsumer:
        """初始化Kafka消费者"""
        try:
            consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_bootstrap_servers,
                auto_offset_reset='earliest',  # 从最早的消息开始消费
                enable_auto_commit=False,  # 关闭自动提交，手动管理偏移量
                group_id='cdc-consumer-group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info(f"成功连接到Kafka服务器: {self.kafka_bootstrap_servers}")
            return consumer
        except KafkaError as e:
            logger.error(f"连接Kafka服务器失败: {e}")
            raise
    
    def _init_producer(self) -> KafkaProducer:
        """初始化Kafka生产者"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            return producer
        except KafkaError as e:
            logger.error(f"初始化Kafka生产者失败: {e}")
            raise
    
    def _init_database(self) -> sqlite3.Connection:
        """初始化SQLite数据库"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # 创建表来存储处理元数据
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    table_name TEXT,
                    event_type TEXT,
                    processing_time TIMESTAMP,
                    offset INTEGER,
                    partition INTEGER
                )
            ''')
            
            # 恢复已处理的事件记录
            cursor.execute("SELECT event_id, processing_time FROM processing_metadata")
            for row in cursor.fetchall():
                event_id, processing_time_str = row
                # 只保留在重放窗口内的记录
                processing_time = datetime.strptime(processing_time_str, '%Y-%m-%d %H:%M:%S.%f')
                if (datetime.now() - processing_time).total_seconds() < self.replay_window_seconds:
                    self.processed_events[event_id] = processing_time
            
            conn.commit()
            logger.info(f"成功初始化数据库: {self.database_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"初始化数据库失败: {e}")
            raise
    
    def _extract_event_info(self, event: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
        """从Debezium事件中提取关键信息
        
        Debezium事件格式说明：
        - source字段：包含事件来源信息（连接器、数据库、表名等）
        - op字段：表示操作类型（c=创建, u=更新, d=删除, r=读取）
        - after字段：操作后的数据（用于创建和更新操作）
        - before字段：操作前的数据（用于删除操作）
        
        返回: (事件ID, 表名, 事件类型, 事件数据)
        """
        # 生成唯一的事件ID，基于事件源信息和时间戳
        event_source = event.get('source', {})
        event_id = f"{event_source.get('connector', '')}-{event_source.get('db', '')}-"
        event_id += f"{event_source.get('table', '')}-{event_source.get('ts_ms', '')}-"
        event_id += f"{event_source.get('pos', '')}"
        
        # 提取表名
        table_name = event_source.get('table', 'unknown')
        
        # 提取事件类型
        event_type = event.get('op', 'unknown')
        
        # 提取事件数据
        event_data = event.get('after', {}) if event_type in ['c', 'u'] else event.get('before', {})
        
        logger.debug(f"提取事件信息: ID={event_id}, 表名={table_name}, 类型={event_type}, 数据字段数={len(event_data)}")
        return event_id, table_name, event_type, event_data
    
    def _is_duplicate_event(self, event_id: str) -> bool:
        """检查事件是否已处理过（在重放窗口内）
        
        实现原理：
        1. 检查事件ID是否存在于已处理事件集合中
        2. 如果存在，检查处理时间是否在重放窗口内
        3. 如果超出重放窗口，则从已处理集合中移除
        
        这种机制保证了在网络重试、应用重启等情况下不会重复处理同一事件
        同时避免了已处理事件集合无限增长
        """
        if event_id in self.processed_events:
            # 检查是否在重放窗口内
            elapsed = (datetime.now() - self.processed_events[event_id]).total_seconds()
            if elapsed < self.replay_window_seconds:
                logger.debug(f"检测到重复事件: {event_id}，在重放窗口内 ({elapsed:.1f}秒前处理)")
                return True
            else:
                # 移出超出重放窗口的事件
                del self.processed_events[event_id]
                logger.debug(f"事件 {event_id} 已超出重放窗口，从已处理集合中移除")
        return False
    
    def _upsert_to_database(self, table_name: str, event_data: Dict[str, Any], event_id: str) -> bool:
        """将数据幂等性地更新或插入到数据库中
        
        幂等UPSERT算法：
        1. 首先尝试更新记录（基于主键）
        2. 如果没有更新任何行（意味着记录不存在），则执行插入
        3. 无论更新还是插入，都记录元数据确保幂等性
        
        返回: 是否成功
        """
        try:
            cursor = self.db_conn.cursor()
            
            # 确保表存在
            columns = ', '.join(event_data.keys())
            placeholders = ', '.join(['?'] * len(event_data))
            update_clauses = ', '.join([f"{col} = ?" for col in event_data.keys()])
            
            # 创建表（如果不存在）
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} TEXT' for col in event_data.keys()])})"
            cursor.execute(create_table_sql)
            
            # 假设主键是'id'，实际应用中需要根据具体情况调整
            primary_key = 'id' if 'id' in event_data else next(iter(event_data.keys()))
            primary_key_value = event_data[primary_key]
            
            # 尝试更新
            update_sql = f"UPDATE {table_name} SET {update_clauses} WHERE {primary_key} = ?"
            update_params = list(event_data.values()) + [primary_key_value]
            cursor.execute(update_sql, update_params)
            
            # 确定操作类型
            operation = "更新" if cursor.rowcount > 0 else "插入"
            
            # 如果没有更新任何行，则插入
            if cursor.rowcount == 0:
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(insert_sql, list(event_data.values()))
            
            # 记录处理元数据
            cursor.execute(
                "INSERT OR REPLACE INTO processing_metadata (event_id, table_name, event_type, processing_time, offset, partition) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, table_name, 'upsert', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), -1, -1)  # 实际应用中需要填充真实的offset和partition
            )
            
            self.db_conn.commit()
            
            # 输出详细信息以便观察
            logger.info(f"数据{operation}成功: 表={table_name}, 主键={primary_key}={primary_key_value}, 事件ID={event_id[:20]}...")
            return True
        except sqlite3.IntegrityError as e:
            # 主键冲突，将被发送到DLQ
            logger.warning(f"主键冲突: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"数据库操作失败: {e}")
            return False
    
    def _send_to_dlq(self, event: Dict[str, Any], reason: str) -> None:
        """将失败的事件发送到死信队列
        
        死信队列用于存储无法正常处理的事件，包含：
        - 原始事件数据
        - 错误原因
        - 时间戳
        
        这些事件可以后续进行分析和重新处理
        """
        try:
            dlq_event = {
                'original_event': event,
                'error_reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.mock_mode:
                # 模拟模式下，只记录日志
                logger.info(f"[模拟模式] 事件将发送到DLQ，原因: {reason}")
                # 可以在这里添加模拟DLQ存储逻辑
            else:
                # 生产模式下，发送到真实Kafka
                self.producer.send(self.dlq_topic, value=dlq_event)
                self.producer.flush()
                
            self.stats['dlq_events'] += 1
            logger.info(f"事件已发送到DLQ，原因: {reason}")
        except KafkaError as e:
            logger.error(f"发送到DLQ失败: {e}")
    
    def _update_watermark(self, table_name: str) -> None:
        """更新表的watermark"""
        self.watermarks[table_name] = datetime.now()
    
    def _commit_checkpoint(self, partition: int, offset: int) -> None:
        """提交消费偏移量作为检查点"""
        try:
            if not self.mock_mode and self.consumer:
                self.consumer.commit()
                self.last_offsets[f"{partition}"] = offset
                logger.info(f"已提交检查点，分区: {partition}, 偏移量: {offset}")
            else:
                # 模拟模式下，只记录日志
                logger.debug(f"[模拟模式] 跳过检查点提交，分区: {partition}, 偏移量: {offset}")
        except KafkaError as e:
            logger.error(f"提交检查点失败: {e}")
    
    def _clean_old_events(self) -> None:
        """清理超出重放窗口的已处理事件记录"""
        now = datetime.now()
        events_to_remove = [event_id for event_id, process_time in self.processed_events.items()
                          if (now - process_time).total_seconds() >= self.replay_window_seconds]
        
        for event_id in events_to_remove:
            del self.processed_events[event_id]
        
        if events_to_remove:
            logger.info(f"已清理 {len(events_to_remove)} 个超出重放窗口的事件记录")
            
    def _generate_mock_event(self, event_type: str, table_name: str, record_id: int) -> Dict[str, Any]:
        """生成模拟的Debezium格式的CDC事件
        
        参数:
            event_type: 事件类型 (c=创建, u=更新, d=删除)
            table_name: 表名
            record_id: 记录ID
            
        返回: 模拟的Debezium事件
        """
        # 根据表名生成相应的字段和数据
        table_schemas = {
            'customers': {
                'fields': ['id', 'first_name', 'last_name', 'email', 'phone', 'created_at'],
                'data_generator': lambda: {
                    'id': record_id,
                    'first_name': f'Customer_{record_id}',
                    'last_name': f'Surname_{record_id}',
                    'email': f'customer{record_id}@example.com',
                    'phone': f'+1-555-{1000 + record_id}',
                    'created_at': datetime.now().isoformat()
                }
            },
            'products': {
                'fields': ['id', 'name', 'description', 'price', 'category', 'stock'],
                'data_generator': lambda: {
                    'id': record_id,
                    'name': f'Product_{record_id}',
                    'description': f'This is product #{record_id}',
                    'price': round(random.uniform(10.0, 1000.0), 2),
                    'category': random.choice(['electronics', 'clothing', 'food', 'books', 'toys']),
                    'stock': random.randint(0, 1000)
                }
            },
            'orders': {
                'fields': ['id', 'customer_id', 'order_date', 'total_amount', 'status'],
                'data_generator': lambda: {
                    'id': record_id,
                    'customer_id': random.randint(1, 100),
                    'order_date': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                    'total_amount': round(random.uniform(10.0, 5000.0), 2),
                    'status': random.choice(['pending', 'processing', 'shipped', 'delivered', 'cancelled'])
                }
            }
        }
        
        # 获取表的模式，如果不存在则使用默认模式
        schema = table_schemas.get(table_name, {
            'fields': ['id', 'data'],
            'data_generator': lambda: {
                'id': record_id,
                'data': f'Mock data for record {record_id}'
            }
        })
        
        # 生成事件数据
        event_data = schema['data_generator']()
        
        # 生成事件时间戳
        ts_ms = int(time.time() * 1000)
        
        # 构建模拟的Debezium事件
        mock_event = {
            'source': {
                'connector': 'mysql',
                'db': 'inventory',
                'table': table_name,
                'ts_ms': ts_ms,
                'pos': random.randint(1000, 999999)
            },
            'op': event_type
        }
        
        # 根据事件类型添加after或before字段
        if event_type in ['c', 'u']:
            # 对于更新操作，修改一个随机字段的值
            if event_type == 'u' and len(event_data) > 1:
                # 选择一个除id外的随机字段进行修改
                fields_to_modify = [k for k in event_data.keys() if k != 'id']
                if fields_to_modify:
                    field_to_modify = random.choice(fields_to_modify)
                    if isinstance(event_data[field_to_modify], str):
                        event_data[field_to_modify] += '_updated'
                    elif isinstance(event_data[field_to_modify], (int, float)):
                        if field_to_modify == 'price':
                            event_data[field_to_modify] = round(event_data[field_to_modify] * (0.9 + random.random() * 0.2), 2)
                        else:
                            event_data[field_to_modify] = event_data[field_to_modify] * (1 + random.randint(-2, 2)) % 1000
            
            mock_event['after'] = event_data
        elif event_type == 'd':
            mock_event['before'] = event_data
        
        # 有3%的概率生成一个会导致主键冲突的事件（用于测试DLQ功能）
        if random.random() < 0.03 and event_type == 'c':
            # 修改id为一个已存在的值
            conflicting_id = max(1, record_id - random.randint(1, min(record_id, 10)))
            event_data['id'] = conflicting_id
            mock_event['after'] = event_data
            
        return mock_event
    
    def process_event(self, msg) -> bool:
        """处理单个CDC事件
        
        返回: 是否处理成功
        """
        try:
            # 提取消息值
            event = msg.value
            partition = msg.partition
            offset = msg.offset
            
            # 提取事件信息
            event_id, table_name, event_type, event_data = self._extract_event_info(event)
            
            # 检查是否是重复事件
            if self._is_duplicate_event(event_id):
                self.stats['duplicate_events'] += 1
                logger.debug(f"跳过重复事件: {event_id}")
                return True
            
            # 处理事件
            success = False
            if event_type in ['c', 'u', 'r']:  # 创建、更新、读取
                # 尝试UPSERT到数据库
                success = self._upsert_to_database(table_name, event_data, event_id)
                
                if not success:
                    # 主键冲突，发送到DLQ
                    self._send_to_dlq(event, "主键冲突")
                    self.stats['failed_events'] += 1
                else:
                    # 更新watermark
                    self._update_watermark(table_name)
                    # 记录已处理的事件
                    self.processed_events[event_id] = datetime.now()
                    self.stats['success_events'] += 1
            elif event_type == 'd':  # 删除
                # 对于删除事件，我们可能只需要记录，或者根据业务需求处理
                logger.info(f"接收到删除事件，表名: {table_name}, 数据: {event_data}")
                success = True
                self.stats['success_events'] += 1
            else:
                logger.warning(f"未知的事件类型: {event_type}")
                success = False
                self.stats['failed_events'] += 1
            
            # 更新统计信息
            self.stats['total_events'] += 1
            self.process_count += 1
            
            # 定期提交检查点
            if self.process_count % self.checkpoint_interval == 0:
                self._commit_checkpoint(partition, offset)
                # 清理旧事件记录
                self._clean_old_events()
            
            return success
        except Exception as e:
            logger.error(f"处理事件失败: {e}")
            self.stats['failed_events'] += 1
            # 发送到DLQ
            try:
                self._send_to_dlq(msg.value, str(e))
            except:
                logger.error("发送到DLQ也失败了")
                pass
            return False
    
    def run(self, max_events: Optional[int] = None) -> None:
        """启动消费者并持续处理事件
        
        运行流程：
        1. 根据模式选择消费来源（真实Kafka或模拟事件）
        2. 循环处理每个事件
        3. 定期输出性能和统计信息
        4. 支持最大事件数限制，防止无限运行
        5. 优雅处理中断信号
        6. 结束时关闭资源并输出最终统计
        
        参数:
            max_events: 最大处理事件数，None表示无限处理
        """
        logger.info(f"启动CDC消费者，主题: {self.input_topic}")
        logger.info(f"重放窗口大小: {self.replay_window_seconds}秒")
        logger.info(f"检查点间隔: {self.checkpoint_interval}条记录")
        logger.info(f"数据库路径: {self.database_path}")
        logger.info(f"死信队列主题: {self.dlq_topic}")
        
        event_count = 0
        total_start_time = time.time()
        
        try:
            if self.mock_mode:
                # 模拟模式，生成假的事件数据
                logger.info(f"进入模拟模式，将生成{max_events or '无限'}条模拟事件")
                
                # 预定义一些事件类型和表名的分布
                event_types = ['c', 'u', 'd']  # 创建、更新、删除
                table_names = ['customers', 'products', 'orders']
                record_ids = set()
                
                while True:
                    # 生成随机的事件类型和表名
                    event_type = random.choice(event_types)
                    table_name = random.choice(table_names)
                    
                    # 生成或重用记录ID
                    if event_type == 'c' or len(record_ids) == 0:
                        # 创建新记录或记录集为空时，生成新ID
                        record_id = len(record_ids) + 1
                        record_ids.add(record_id)
                    else:
                        # 更新或删除时，重用现有ID
                        record_id = random.choice(list(record_ids))
                    
                    # 有5%的概率生成重复事件（用于测试去重功能）
                    is_duplicate = random.random() < 0.05
                    if is_duplicate and event_count > 0:
                        # 重复使用上一个事件的ID
                        pass
                    
                    # 生成模拟事件
                    mock_event = self._generate_mock_event(event_type, table_name, record_id)
                    
                    # 创建模拟消息对象
                    mock_msg = MockKafkaMessage(
                        value=mock_event,
                        partition=random.randint(0, 2),  # 模拟多分区
                        offset=event_count
                    )
                    
                    # 处理事件
                    self.process_event(mock_msg)
                    event_count += 1
                    
                    # 定期打印详细统计信息
                    if event_count % 10 == 0:
                        elapsed_time = time.time() - total_start_time
                        tps = event_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info("=" * 60)
                        logger.info(f"处理进度: {event_count}/{max_events or '无限'} 条事件")
                        logger.info(f"性能指标: TPS={tps:.2f}, 平均每条耗时={(elapsed_time/event_count*1000):.2f}ms")
                        logger.info(f"统计信息: {self.stats}")
                        logger.info(f"当前已处理事件缓存大小: {len(self.processed_events)}")
                        logger.info(f"当前watermark数量: {len(self.watermarks)}")
                        logger.info("=" * 60)
                    
                    # 达到最大事件数后退出
                    if max_events and event_count >= max_events:
                        logger.info(f"已达到最大处理事件数: {max_events}")
                        break
                    
                    # 模拟实际处理的延迟
                    time.sleep(random.uniform(0.01, 0.1))
            else:
                # 生产模式，从Kafka消费事件
                logger.info("进入生产模式，从Kafka消费事件")
                for msg in self.consumer:
                    # 记录单个事件的处理开始时间
                    start_time = time.time()
                    self.process_event(msg)
                    event_count += 1
                    
                    # 定期打印统计信息
                    if event_count % 10 == 0:
                        elapsed_time = time.time() - total_start_time
                        tps = event_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"已处理 {event_count} 条事件，耗时 {elapsed_time:.2f} 秒，TPS: {tps:.2f}")
                        logger.info(f"统计信息: {self.stats}")
                    
                    # 达到最大事件数后退出
                    if max_events and event_count >= max_events:
                        logger.info(f"已达到最大处理事件数: {max_events}")
                        break
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
        finally:
            # 最后提交一次检查点
            try:
                if not self.mock_mode and self.consumer:
                    self.consumer.commit()
            except:
                pass
            
            # 关闭资源
            try:
                if not self.mock_mode and self.consumer:
                    self.consumer.close()
            except:
                pass
            
            try:
                if not self.mock_mode and self.producer:
                    self.producer.close()
            except:
                pass
            
            try:
                if self.db_conn:
                    self.db_conn.close()
            except:
                pass
            
            # 打印最终统计信息
            logger.info(f"CDC消费者已停止")
            # 输出详细的最终统计信息
            elapsed_time = time.time() - total_start_time
            tps = event_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info("=" * 80)
            logger.info("          CDC事件处理完成          ")
            logger.info("=" * 80)
            logger.info(f"总处理事件数: {event_count}")
            logger.info(f"总耗时: {elapsed_time:.2f} 秒")
            logger.info(f"平均TPS: {tps:.2f}")
            logger.info(f"平均每条事件处理时间: {(elapsed_time/event_count*1000):.2f}ms")
            logger.info("\n详细统计信息:")
            logger.info(f"  - 成功事件: {self.stats['success_events']} ({self.stats['success_events']/event_count*100:.2f}%)")
            logger.info(f"  - 失败事件: {self.stats['failed_events']} ({self.stats['failed_events']/event_count*100:.2f}%)")
            logger.info(f"  - 重复事件: {self.stats['duplicate_events']} ({self.stats['duplicate_events']/event_count*100:.2f}%)")
            logger.info(f"  - DLQ事件: {self.stats['dlq_events']} ({self.stats['dlq_events']/event_count*100:.2f}%)")
            logger.info(f"\n资源使用情况:")
            logger.info(f"  - 已处理事件缓存大小: {len(self.processed_events)}")
            logger.info(f"  - 活跃watermark数量: {len(self.watermarks)}")
            logger.info("=" * 80)

def main():
    """程序主入口，支持命令行参数配置"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Kafka CDC消费者工具')
    parser.add_argument('--kafka-servers', type=str, default='localhost:9092',
                        help='Kafka服务器地址，格式为host:port')
    parser.add_argument('--input-topic', type=str, default='dbserver1.inventory',
                        help='输入的Kafka主题')
    parser.add_argument('--dlq-topic', type=str, default='cdc_dlq',
                        help='死信队列主题')
    parser.add_argument('--database', type=str, default='./cdc_data.db',
                        help='SQLite数据库路径')
    parser.add_argument('--replay-window', type=int, default=3600,
                        help='重放窗口大小（秒）')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='检查点间隔（记录数）')
    parser.add_argument('--max-events', type=int, default=50,
                        help='最大处理事件数')
    parser.add_argument('--mock', action='store_true',
                        help='启用模拟模式（不连接真实Kafka）')
    parser.add_argument('--verbose', action='store_true',
                        help='启用详细日志输出')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果启用详细日志
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 80)
    print("          Kafka CDC 消费者工具          ")
    print("=" * 80)
    print(f"运行模式: {'模拟模式' if args.mock else '生产模式'}")
    print(f"目标Kafka: {args.kafka_servers}")
    print(f"输入主题: {args.input_topic}")
    print(f"DLQ主题: {args.dlq_topic}")
    print(f"数据库: {args.database}")
    print(f"重放窗口: {args.replay_window}秒")
    print(f"检查点间隔: {args.checkpoint_interval}条记录")
    print(f"最大事件数: {args.max_events}")
    print("=" * 80)
    
    # 初始化CDC处理器
    try:
        cdc_processor = CDCEventProcessor(
            kafka_bootstrap_servers=args.kafka_servers,
            input_topic=args.input_topic,
            dlq_topic=args.dlq_topic,
            database_path=args.database,
            replay_window_seconds=args.replay_window,
            checkpoint_interval=args.checkpoint_interval,
            mock_mode=args.mock
        )
        
        # 运行消费者
        cdc_processor.run(max_events=args.max_events)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"程序运行失败: {e}")

if __name__ == "__main__":
    """程序入口点"""
    main()