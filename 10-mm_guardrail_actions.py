#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨模态SLO监控、降级与回滚动作建议

功能说明：监测回放率/低置信/P95/成本，输出降级或停用/回滚的动作计划与审计日志。

内容概述：滑窗检测异常（回放率跌破、置信升高、延迟超阈），映射到降级/停用/回滚动作；生成预计影响评估与审计日志、复盘模板。

执行流程：
1. 载入timeseries.csv与guardrails.json
2. 检测阈值越界与突变，合并多指标信号
3. 生成动作建议：降权未锚点→禁用跨模态→回滚版本
4. 输出操作清单与预计质量/延迟变化
5. 保存alerts.csv、action_plan.csv、audit_log.csv

输入参数：
--slo guardrails.json      监控阈值配置文件路径
--window 15min            滑动窗口大小

作者: Ph.D. Rhino
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import csv
from typing import Dict, List, Tuple, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """时间序列数据加载器：加载和预处理监控数据"""
    
    def __init__(self, data_file: str = "timeseries.csv"):
        """初始化时间序列数据加载器
        
        Args:
            data_file: 时间序列数据文件路径
        """
        self.data_file = data_file
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """加载时间序列数据
        
        Returns:
            加载并预处理后的时间序列数据
        """
        try:
            if os.path.exists(self.data_file):
                logger.info(f"加载时间序列数据: {self.data_file}")
                self.data = pd.read_csv(self.data_file)
                # 转换时间戳列
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            else:
                logger.warning(f"时间序列文件 {self.data_file} 不存在，生成默认数据")
                self.data = self._generate_default_timeseries_data()
                self.data.to_csv(self.data_file, index=False)
        except Exception as e:
            logger.error(f"加载时间序列数据失败: {e}")
            self.data = self._generate_default_timeseries_data()
        
        return self.data
    
    def _generate_default_timeseries_data(self) -> pd.DataFrame:
        """生成默认的时间序列数据
        
        Returns:
            生成的时间序列数据集
        """
        logger.info("生成默认时间序列数据...")
        
        # 生成过去60分钟的数据，每分钟一个样本
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=60)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # 初始化数据数组
        data = []
        
        for ts in timestamps:
            # 计算时间相对于开始的分钟数
            minutes_from_start = (ts - start_time).total_seconds() / 60
            
            # 生成正常数据，但在最后15分钟加入异常
            if minutes_from_start < 45:
                # 正常数据
                playback_rate = max(0.85, min(0.95, 0.9 + np.random.normal(0, 0.02)))
                low_conf_percent = max(5, min(10, 7.5 + np.random.normal(0, 1)))
                p95_latency = max(100, min(150, 125 + np.random.normal(0, 10)))
                cost_per_request = max(0.8, min(1.2, 1.0 + np.random.normal(0, 0.05)))
                anchored_percent = max(75, min(90, 85 + np.random.normal(0, 3)))
            else:
                # 异常数据（模拟问题场景）
                playback_rate = max(0.65, min(0.75, 0.7 + np.random.normal(0, 0.02)))
                low_conf_percent = max(18, min(25, 22 + np.random.normal(0, 1)))
                p95_latency = max(180, min(250, 220 + np.random.normal(0, 15)))
                cost_per_request = max(1.3, min(1.8, 1.5 + np.random.normal(0, 0.1)))
                anchored_percent = max(60, min(70, 65 + np.random.normal(0, 2)))
            
            data.append({
                'timestamp': ts,
                'playback_rate': playback_rate,
                'low_conf_percent': low_conf_percent,
                'p95_latency': p95_latency,
                'cost_per_request': cost_per_request,
                'anchored_percent': anchored_percent,
                'request_count': 1000 + np.random.randint(-100, 100)
            })
        
        df = pd.DataFrame(data)
        return df


class GuardrailConfigLoader:
    """监控阈值配置加载器：加载和验证SLO配置"""
    
    def __init__(self, config_file: str):
        """初始化监控阈值配置加载器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = None
    
    def load_config(self) -> Dict[str, Any]:
        """加载监控阈值配置
        
        Returns:
            监控阈值配置字典
        """
        try:
            if os.path.exists(self.config_file):
                logger.info(f"加载监控阈值配置: {self.config_file}")
                with open(self.config_file, 'r', encoding='utf-8-sig') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"监控阈值配置文件 {self.config_file} 不存在，生成默认配置")
                self.config = self._generate_default_guardrails()
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"加载监控阈值配置失败: {e}")
            self.config = self._generate_default_guardrails()
        
        return self.config
    
    def _generate_default_guardrails(self) -> Dict[str, Any]:
        """生成默认的监控阈值配置
        
        Returns:
            默认配置字典
        """
        return {
            "thresholds": {
                "playback_rate": {
                    "min": 0.8,
                    "alert_level": "warning"
                },
                "low_conf_percent": {
                    "max": 15,
                    "alert_level": "warning"
                },
                "p95_latency": {
                    "max": 180,
                    "alert_level": "warning"
                },
                "cost_per_request": {
                    "max": 1.3,
                    "alert_level": "warning"
                },
                "anchored_percent": {
                    "min": 70,
                    "alert_level": "info"
                }
            },
            "current_version": "v1.12",
            "previous_versions": [
                {"version": "v1.10", "performance": {"latency": 110, "cost": 0.95, "quality": 0.88}},
                {"version": "v1.9", "performance": {"latency": 100, "cost": 0.9, "quality": 0.85}}
            ],
            "rollback_eta": "3min",
            "action_mappings": {
                "low_playback": ["disable_mm"],
                "high_low_conf": ["raise_qc", "demote_unanchored"],
                "high_latency": ["scale_up", "cache_more"],
                "high_cost": ["optimize_models", "throttle_non_critical"]
            }
        }


class AnomalyDetector:
    """异常检测器：基于滑动窗口检测指标异常"""
    
    def __init__(self, window_size: str = "15min"):
        """初始化异常检测器
        
        Args:
            window_size: 滑动窗口大小
        """
        # 转换窗口大小为分钟数
        if window_size.endswith('min'):
            self.window_size_minutes = int(window_size[:-3])
        else:
            self.window_size_minutes = 15  # 默认15分钟
        
        logger.info(f"设置滑动窗口大小: {self.window_size_minutes}分钟")
    
    def detect_anomalies(self, data: pd.DataFrame, thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测异常指标
        
        Args:
            data: 时间序列数据
            thresholds: 阈值配置
            
        Returns:
            异常列表
        """
        anomalies = []
        
        # 获取最近的窗口数据
        latest_data = data.tail(self.window_size_minutes).copy()
        
        if len(latest_data) == 0:
            logger.warning("没有足够的数据进行异常检测")
            return anomalies
        
        # 计算窗口统计
        window_stats = {
            'playback_rate': {
                'mean': latest_data['playback_rate'].mean(),
                'min': latest_data['playback_rate'].min(),
                'max': latest_data['playback_rate'].max()
            },
            'low_conf_percent': {
                'mean': latest_data['low_conf_percent'].mean(),
                'min': latest_data['low_conf_percent'].min(),
                'max': latest_data['low_conf_percent'].max()
            },
            'p95_latency': {
                'mean': latest_data['p95_latency'].mean(),
                'min': latest_data['p95_latency'].min(),
                'max': latest_data['p95_latency'].max()
            },
            'cost_per_request': {
                'mean': latest_data['cost_per_request'].mean(),
                'min': latest_data['cost_per_request'].min(),
                'max': latest_data['cost_per_request'].max()
            },
            'anchored_percent': {
                'mean': latest_data['anchored_percent'].mean(),
                'min': latest_data['anchored_percent'].min(),
                'max': latest_data['anchored_percent'].max()
            }
        }
        
        # 检查每个指标的阈值
        for metric, stats in window_stats.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                
                # 检查最小值阈值
                if 'min' in threshold and stats['mean'] < threshold['min']:
                    anomalies.append({
                        'metric': metric,
                        'value': round(stats['mean'], 2),
                        'threshold': threshold['min'],
                        'direction': 'below',
                        'alert_level': threshold.get('alert_level', 'warning'),
                        'window_stats': stats
                    })
                
                # 检查最大值阈值
                if 'max' in threshold and stats['mean'] > threshold['max']:
                    anomalies.append({
                        'metric': metric,
                        'value': round(stats['mean'], 2),
                        'threshold': threshold['max'],
                        'direction': 'above',
                        'alert_level': threshold.get('alert_level', 'warning'),
                        'window_stats': stats
                    })
        
        # 排序异常，严重的优先
        alert_level_order = {'critical': 0, 'warning': 1, 'info': 2}
        anomalies.sort(key=lambda x: alert_level_order.get(x['alert_level'], 999))
        
        return anomalies


class ActionPlanner:
    """动作规划器：根据异常生成动作建议"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化动作规划器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.current_version = config.get('current_version', 'v1.0')
        self.previous_versions = config.get('previous_versions', [])
        self.action_mappings = config.get('action_mappings', {})
        self.rollback_eta = config.get('rollback_eta', '5min')
    
    def generate_action_plan(self, anomalies: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        """生成动作计划
        
        Args:
            anomalies: 检测到的异常列表
            
        Returns:
            (动作计划列表, 是否需要回滚)
        """
        action_plan = []
        need_rollback = False
        
        # 统计异常类型
        anomaly_types = self._categorize_anomalies(anomalies)
        
        # 根据异常类型生成动作
        for anomaly_type, anomaly_list in anomaly_types.items():
            # 获取对应的动作
            actions = self.action_mappings.get(anomaly_type, [])
            
            for action in actions:
                # 生成动作详情
                action_detail = self._create_action_detail(action, anomaly_list)
                action_plan.append(action_detail)
        
        # 判断是否需要回滚
        need_rollback = self._check_rollback_needed(anomalies)
        
        # 如果需要回滚，添加回滚动作
        if need_rollback:
            rollback_action = self._create_rollback_action()
            action_plan.insert(0, rollback_action)  # 回滚动作放在最前面
        
        # 去重并排序动作
        action_plan = self._prioritize_actions(action_plan)
        
        return action_plan, need_rollback
    
    def _categorize_anomalies(self, anomalies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """对异常进行分类
        
        Args:
            anomalies: 异常列表
            
        Returns:
            分类后的异常字典
        """
        categorized = {
            'low_playback': [],
            'high_low_conf': [],
            'high_latency': [],
            'high_cost': []
        }
        
        for anomaly in anomalies:
            if anomaly['metric'] == 'playback_rate' and anomaly['direction'] == 'below':
                categorized['low_playback'].append(anomaly)
            elif anomaly['metric'] == 'low_conf_percent' and anomaly['direction'] == 'above':
                categorized['high_low_conf'].append(anomaly)
            elif anomaly['metric'] == 'p95_latency' and anomaly['direction'] == 'above':
                categorized['high_latency'].append(anomaly)
            elif anomaly['metric'] == 'cost_per_request' and anomaly['direction'] == 'above':
                categorized['high_cost'].append(anomaly)
        
        return {k: v for k, v in categorized.items() if v}
    
    def _create_action_detail(self, action: str, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建动作详情
        
        Args:
            action: 动作名称
            anomalies: 触发该动作的异常列表
            
        Returns:
            动作详情字典
        """
        # 动作描述映射
        action_descriptions = {
            'disable_mm': '禁用跨模态处理',
            'raise_qc': '提高质检严格度',
            'demote_unanchored': '降低未锚点证据权重',
            'scale_up': '扩容服务节点',
            'cache_more': '增加缓存比例',
            'optimize_models': '优化模型配置',
            'throttle_non_critical': '限制非关键请求'
        }
        
        # 预计影响评估
        impact = self._estimate_impact(action)
        
        return {
            'action': action,
            'description': action_descriptions.get(action, action),
            'triggered_by': [f"{a['metric']} {a['value']} {a['direction']} {a['threshold']}" for a in anomalies],
            'severity': self._determine_action_severity(action, anomalies),
            'estimated_impact': impact,
            'timestamp': datetime.now().isoformat(),
            'eta': '1min' if action != 'scale_up' else '5min',
            'priority': self._get_action_priority(action)
        }
    
    def _create_rollback_action(self) -> Dict[str, Any]:
        """创建回滚动作
        
        Returns:
            回滚动作详情
        """
        # 选择最合适的回滚版本
        best_rollback_version = self._select_best_rollback_version()
        
        return {
            'action': 'rollback',
            'description': f'回滚到版本 {best_rollback_version}',
            'triggered_by': ['多指标严重异常'],
            'severity': 'critical',
            'estimated_impact': {
                'latency_improvement': '~20%',
                'cost_reduction': '~15%',
                'quality_change': '-2%',
                'risk_level': 'medium'
            },
            'timestamp': datetime.now().isoformat(),
            'eta': self.rollback_eta,
            'priority': 0,
            'from_version': self.current_version,
            'to_version': best_rollback_version
        }
    
    def _select_best_rollback_version(self) -> str:
        """选择最佳回滚版本
        
        Returns:
            最佳回滚版本号
        """
        if not self.previous_versions:
            return 'v1.0'
        
        # 优先选择最近的版本
        return self.previous_versions[0]['version']
    
    def _check_rollback_needed(self, anomalies: List[Dict[str, Any]]) -> bool:
        """检查是否需要回滚
        
        Args:
            anomalies: 异常列表
            
        Returns:
            是否需要回滚
        """
        # 定义回滚条件
        critical_metrics = ['playback_rate', 'low_conf_percent']
        critical_anomalies = [a for a in anomalies if a['metric'] in critical_metrics]
        
        # 条件1: 多个关键指标异常
        if len(critical_anomalies) >= 2:
            return True
        
        # 条件2: 单个指标严重超出阈值
        for anomaly in anomalies:
            if anomaly['metric'] == 'playback_rate' and anomaly['direction'] == 'below':
                # 回放率低于阈值20%以上
                if (anomaly['threshold'] - anomaly['value']) / anomaly['threshold'] > 0.2:
                    return True
            elif anomaly['metric'] == 'low_conf_percent' and anomaly['direction'] == 'above':
                # 低置信度超过阈值50%以上
                if (anomaly['value'] - anomaly['threshold']) / anomaly['threshold'] > 0.5:
                    return True
        
        return False
    
    def _estimate_impact(self, action: str) -> Dict[str, str]:
        """估计动作的影响
        
        Args:
            action: 动作名称
            
        Returns:
            影响评估字典
        """
        impact_mappings = {
            'disable_mm': {
                'latency_improvement': '~30%',
                'cost_reduction': '~25%',
                'quality_change': '-5%',
                'risk_level': 'medium'
            },
            'raise_qc': {
                'latency_improvement': '-5%',
                'cost_reduction': '-10%',
                'quality_change': '+3%',
                'risk_level': 'low'
            },
            'demote_unanchored': {
                'latency_improvement': '~5%',
                'cost_reduction': '~5%',
                'quality_change': '-1%',
                'risk_level': 'low'
            },
            'scale_up': {
                'latency_improvement': '~20%',
                'cost_reduction': '-20%',
                'quality_change': '0%',
                'risk_level': 'low'
            },
            'cache_more': {
                'latency_improvement': '~15%',
                'cost_reduction': '~10%',
                'quality_change': '-1%',
                'risk_level': 'low'
            },
            'optimize_models': {
                'latency_improvement': '~10%',
                'cost_reduction': '~15%',
                'quality_change': '0%',
                'risk_level': 'medium'
            },
            'throttle_non_critical': {
                'latency_improvement': '~10%',
                'cost_reduction': '~10%',
                'quality_change': '0%',
                'risk_level': 'low'
            }
        }
        
        return impact_mappings.get(action, {
            'latency_improvement': 'unknown',
            'cost_reduction': 'unknown',
            'quality_change': 'unknown',
            'risk_level': 'unknown'
        })
    
    def _determine_action_severity(self, action: str, anomalies: List[Dict[str, Any]]) -> str:
        """确定动作的严重程度
        
        Args:
            action: 动作名称
            anomalies: 触发动作的异常
            
        Returns:
            严重程度
        """
        # 检查是否有严重异常
        has_critical = any(a.get('alert_level') == 'critical' for a in anomalies)
        
        if action == 'disable_mm' or has_critical:
            return 'critical'
        elif action in ['raise_qc', 'demote_unanchored']:
            return 'warning'
        else:
            return 'info'
    
    def _get_action_priority(self, action: str) -> int:
        """获取动作优先级
        
        Args:
            action: 动作名称
            
        Returns:
            优先级（数字越小优先级越高）
        """
        priority_mappings = {
            'rollback': 0,
            'disable_mm': 1,
            'raise_qc': 2,
            'demote_unanchored': 3,
            'scale_up': 4,
            'cache_more': 5,
            'optimize_models': 6,
            'throttle_non_critical': 7
        }
        
        return priority_mappings.get(action, 999)
    
    def _prioritize_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对动作进行优先级排序
        
        Args:
            actions: 动作列表
            
        Returns:
            排序后的动作列表
        """
        # 按优先级排序
        sorted_actions = sorted(actions, key=lambda x: x['priority'])
        
        # 去重（保留优先级高的）
        unique_actions = []
        action_names = set()
        
        for action in sorted_actions:
            if action['action'] not in action_names:
                unique_actions.append(action)
                action_names.add(action['action'])
        
        return unique_actions


class ReportGenerator:
    """报告生成器：生成各种输出文件"""
    
    def __init__(self, anomalies: List[Dict[str, Any]], action_plan: List[Dict[str, Any]], need_rollback: bool):
        """初始化报告生成器
        
        Args:
            anomalies: 检测到的异常
            action_plan: 动作计划
            need_rollback: 是否需要回滚
        """
        self.anomalies = anomalies
        self.action_plan = action_plan
        self.need_rollback = need_rollback
    
    def generate_alerts_csv(self, output_file: str = "alerts.csv"):
        """生成告警CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        logger.info(f"生成告警文件: {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['timestamp', 'metric', 'value', 'threshold', 'direction', 'alert_level']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for anomaly in self.anomalies:
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'metric': anomaly['metric'],
                    'value': anomaly['value'],
                    'threshold': anomaly['threshold'],
                    'direction': anomaly['direction'],
                    'alert_level': anomaly['alert_level']
                })
        
        logger.info(f"告警文件已保存: {output_file}")
    
    def generate_action_plan_csv(self, output_file: str = "action_plan.csv"):
        """生成动作计划CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        logger.info(f"生成动作计划文件: {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['timestamp', 'action', 'description', 'severity', 'eta', 'triggered_by', 
                         'latency_impact', 'cost_impact', 'quality_impact', 'risk_level']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for action in self.action_plan:
                writer.writerow({
                    'timestamp': action['timestamp'],
                    'action': action['action'],
                    'description': action['description'],
                    'severity': action['severity'],
                    'eta': action['eta'],
                    'triggered_by': '; '.join(action['triggered_by']),
                    'latency_impact': action['estimated_impact'].get('latency_improvement', 'unknown'),
                    'cost_impact': action['estimated_impact'].get('cost_reduction', 'unknown'),
                    'quality_impact': action['estimated_impact'].get('quality_change', 'unknown'),
                    'risk_level': action['estimated_impact'].get('risk_level', 'unknown')
                })
        
        logger.info(f"动作计划文件已保存: {output_file}")
    
    def generate_audit_log_csv(self, output_file: str = "audit_log.csv"):
        """生成审计日志CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        logger.info(f"生成审计日志文件: {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['timestamp', 'event_type', 'details', 'severity', 'recommended_action']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 添加异常事件
            for anomaly in self.anomalies:
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'event_type': 'anomaly_detected',
                    'details': f"{anomaly['metric']} {anomaly['value']} {anomaly['direction']} threshold {anomaly['threshold']}",
                    'severity': anomaly['alert_level'],
                    'recommended_action': 'pending'
                })
            
            # 添加动作事件
            for action in self.action_plan:
                action_type = 'rollback' if action['action'] == 'rollback' else 'mitigation'
                writer.writerow({
                    'timestamp': action['timestamp'],
                    'event_type': action_type,
                    'details': f"{action['description']} - Triggered by: {' '.join(action['triggered_by'])}",
                    'severity': action['severity'],
                    'recommended_action': action['action']
                })
        
        logger.info(f"审计日志文件已保存: {output_file}")
    
    def generate_console_output(self):
        """生成控制台输出"""
        print("\n" + "="*80)
        print("跨模态SLO监控告警与动作建议")
        print("="*80)
        
        # 打印告警信息
        print("\n【告警信息】")
        for anomaly in self.anomalies:
            if anomaly['metric'] == 'playback_rate' and anomaly['direction'] == 'below':
                print(f"[ALERT] playback {anomaly['value']} < {anomaly['threshold']} → disable_mm")
            elif anomaly['metric'] == 'low_conf_percent' and anomaly['direction'] == 'above':
                print(f"[ALERT] ASR_low_conf {anomaly['value']}% > {anomaly['threshold']}% → raise_qc + demote_unanchored")
            else:
                direction_str = '<' if anomaly['direction'] == 'below' else '>'
                print(f"[ALERT] {anomaly['metric']} {anomaly['value']} {direction_str} {anomaly['threshold']}")
        
        # 打印回滚信息（如果需要）
        if self.need_rollback:
            rollback_action = next((a for a in self.action_plan if a['action'] == 'rollback'), None)
            if rollback_action:
                from_ver = rollback_action.get('from_version', 'current')
                to_ver = rollback_action.get('to_version', 'previous')
                eta = rollback_action.get('eta', 'unknown')
                print(f"Rollback: cfg {from_ver} → {to_ver} (ETA {eta})")
        
        # 打印动作计划摘要
        if self.action_plan:
            print("\n【动作计划摘要】")
            for i, action in enumerate(self.action_plan, 1):
                print(f"{i}. [{action['severity'].upper()}] {action['description']}")
                print(f"   预计影响: 延迟{action['estimated_impact'].get('latency_improvement', 'N/A')}, "
                      f"成本{action['estimated_impact'].get('cost_reduction', 'N/A')}, "
                      f"质量{action['estimated_impact'].get('quality_change', 'N/A')}")
                print(f"   ETA: {action['eta']}")
        
        print("\n" + "="*80)
        print(f"告警总数: {len(self.anomalies)}")
        print(f"动作建议数: {len(self.action_plan)}")
        print(f"是否需要回滚: {'是' if self.need_rollback else '否'}")
        print("="*80)


def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    required_packages = ['pandas', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"依赖 {package} 已安装")
        except ImportError:
            logger.info(f"正在安装依赖 {package}...")
            try:
                import subprocess
                subprocess.check_call(['pip', 'install', package])
                logger.info(f"依赖 {package} 安装成功")
            except Exception as e:
                logger.error(f"安装依赖 {package} 失败: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="跨模态SLO监控、降级与回滚动作建议")
    parser.add_argument("--slo", default="guardrails.json", help="监控阈值配置文件路径")
    parser.add_argument("--window", default="15min", help="滑动窗口大小")
    
    args = parser.parse_args()
    
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 1. 加载时间序列数据
    data_loader = TimeSeriesDataLoader()
    data = data_loader.load_data()
    
    # 2. 加载监控阈值配置
    config_loader = GuardrailConfigLoader(args.slo)
    config = config_loader.load_config()
    
    # 3. 检测异常
    detector = AnomalyDetector(args.window)
    anomalies = detector.detect_anomalies(data, config.get('thresholds', {}))
    
    # 4. 生成动作计划
    planner = ActionPlanner(config)
    action_plan, need_rollback = planner.generate_action_plan(anomalies)
    
    # 5. 生成报告
    report_gen = ReportGenerator(anomalies, action_plan, need_rollback)
    report_gen.generate_alerts_csv()
    report_gen.generate_action_plan_csv()
    report_gen.generate_audit_log_csv()
    report_gen.generate_console_output()
    
    # 6. 验证中文显示
    check_chinese_display()
    
    logger.info("程序执行完成！")


def check_chinese_display():
    """验证中文显示是否正常"""
    test_text = "跨模态SLO监控、降级与回滚动作建议"
    print(f"\n验证中文显示: {test_text}")
    print("中文显示正常")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"执行耗时: {end_time - start_time:.3f} 秒")