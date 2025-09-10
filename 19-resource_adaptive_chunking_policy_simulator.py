#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源自适应分块策略模拟器

功能：
- 基于模拟资源状态（QPS、内存使用、延迟），动态调整分块策略参数
- 支持调整chunk_size、并发数、top-k等参数
- 实现简单的规则引擎和状态机机制
- 提供可视化的策略调整建议

使用示例：
```python
from resource_adaptive_chunking_policy_simulator import ResourceAdaptiveSimulator

# 创建模拟器实例
simulator = ResourceAdaptiveSimulator()

# 模拟当前资源状态
resource_status = {
    "qps": 120,  # 查询每秒
    "mem_usage": 85,  # 内存使用率(%)
    "latency": 1500  # 延迟(ms)
}

# 获取新的参数配置
new_config = simulator.get_adaptive_config(resource_status)
print(f"新配置: {new_config}")
```

依赖：
- numpy
- json
"""

import numpy as np
import json
from typing import Dict, List, Any, Tuple
import time


class ResourceAdaptiveSimulator:
    """资源自适应分块策略模拟器
    
    该类用于模拟基于资源状态的动态分块策略调整，根据当前系统资源状况
    （QPS、内存使用率、延迟）来动态调整分块大小、并发数和top-k值等参数。
    通过状态机机制实现不同资源压力下的参数优化，确保系统在资源受限情况下
    仍能保持稳定运行，同时在资源充足时提供更好的处理质量。
    """
    
    def __init__(self):
        """初始化资源自适应策略模拟器
        
        初始化基础配置参数、资源阈值、权重配置、调整步长和参数约束等。
        """
        # 基础配置参数 - 系统正常状态下的默认参数值
        self.base_config = {
            "chunk_size": 200,  # 分块大小（字符数）
            "overlap": 20,      # 分块重叠大小（字符数）
            "concurrency": 8,   # 并发处理数
            "top_k": 5,         # 检索返回的top-k数量
            "rerank_top_k": 3   # 重排序后的top-k数量
        }
        
        # 资源阈值配置 - 用于触发不同级别的策略调整
        # 分为低(low)、中(medium)、高(high)、临界(critical)四个等级
        self.resource_thresholds = {
            "qps": {
                "low": 30,        # 低QPS阈值
                "medium": 80,     # 中等QPS阈值
                "high": 120,      # 高QPS阈值
                "critical": 150   # 临界QPS阈值
            },
            "mem_usage": {
                "low": 40,        # 低内存使用率阈值(%)
                "medium": 65,     # 中等内存使用率阈值(%)
                "high": 80,       # 高内存使用率阈值(%)
                "critical": 90    # 临界内存使用率阈值(%)
            },
            "latency": {
                "low": 300,       # 低延迟阈值(ms)
                "medium": 800,    # 中等延迟阈值(ms)
                "high": 1200,     # 高延迟阈值(ms)
                "critical": 1800  # 临界延迟阈值(ms)
            }
        }
        
        # 资源权重配置 - 影响策略调整的优先级
        # 权重总和为1.0，内存使用率权重最高(0.4)，QPS和延迟权重次之(0.3)
        self.resource_weights = {
            "qps": 0.3,        # QPS权重
            "mem_usage": 0.4,  # 内存使用率权重
            "latency": 0.3     # 延迟权重
        }
        
        # 参数调整步长配置 - 定义每次调整时参数变化的幅度
        # 减少步长通常大于增加步长，实现快速降载和渐进式升载
        self.adjustment_steps = {
            "chunk_size": {
                "decrease": 20,  # 每次减少的chunk_size（快速减少以降低处理压力）
                "increase": 10   # 每次增加的chunk_size（缓慢增加以保持系统稳定）
            },
            "overlap": {
                "decrease": 5,   # 每次减少的overlap
                "increase": 5    # 每次增加的overlap
            },
            "concurrency": {
                "decrease": 2,   # 每次减少的并发数（快速降低并发以减轻系统压力）
                "increase": 1    # 每次增加的并发数（缓慢增加以避免突发负载）
            },
            "top_k": {
                "decrease": 1,   # 每次减少的top_k
                "increase": 1    # 每次增加的top_k
            },
            "rerank_top_k": {
                "decrease": 1,   # 每次减少的rerank_top_k
                "increase": 1    # 每次增加的rerank_top_k
            }
        }
        
        # 参数约束配置 - 确保参数在合理范围内，防止极端值
        self.parameter_constraints = {
            "chunk_size": {
                "min": 50,   # 最小分块大小（保证基本语义完整性）
                "max": 500   # 最大分块大小（避免处理过大的文本块）
            },
            "overlap": {
                "min": 0,    # 最小重叠大小
                "max": 100   # 最大重叠大小（避免过多冗余）
            },
            "concurrency": {
                "min": 1,    # 最小并发数
                "max": 32    # 最大并发数（避免系统资源耗尽）
            },
            "top_k": {
                "min": 1,    # 最小top_k值
                "max": 20    # 最大top_k值
            },
            "rerank_top_k": {
                "min": 1,    # 最小rerank_top_k值
                "max": 10    # 最大rerank_top_k值
            }
        }
        
        # 当前状态机状态 - 初始状态为正常
        self.current_state = "normal"  # 可选值: idle, normal, pressure, critical
        
        # 历史状态记录，用于平滑策略调整
        self.history: List[Dict[str, Any]] = []  # 存储历史配置和资源状态
        self.history_window = 5  # 历史记录窗口大小（最近5次调整）
        
    def _classify_resource_level(self, resource_name: str, value: float) -> str:
        """
        根据资源值判断资源等级
        
        该方法将具体的资源数值映射到四个等级之一：低(low)、中(medium)、高(high)、临界(critical)
        对于QPS，数值越高表示资源压力越大；对于内存使用率和延迟，数值越高同样表示资源压力越大
        
        参数:
            resource_name: 资源名称 (qps, mem_usage, latency)
            value: 资源值
        
        返回:
            资源等级 (low, medium, high, critical)
        """
        thresholds = self.resource_thresholds[resource_name]
        
        # 对于所有资源类型，使用相同的判断逻辑（值越高，等级越高）
        if value < thresholds["low"]:
            return "low"
        elif value < thresholds["medium"]:
            return "medium"
        elif value < thresholds["high"]:
            return "high"
        else:
            return "critical"
        
    def _calculate_resource_score(self, resource_status: Dict[str, float]) -> float:
        """
        计算资源状态综合评分
        
        该方法将多个资源指标（QPS、内存使用率、延迟）综合为一个0-100的评分。
        评分越高表示资源越充足，系统负载越低；评分越低表示资源越紧张，系统负载越高。
        
        参数:
            resource_status: 资源状态字典 {"qps": 值, "mem_usage": 值, "latency": 值}
        
        返回:
            综合评分 (0-100)
        """
        # 对每个资源进行标准化评分 (0-100)
        scores = {}
        for resource_name, value in resource_status.items():
            thresholds = self.resource_thresholds[resource_name]
            
            # 所有资源使用相同的评分逻辑：
            # 值低于low阈值：资源非常充足，得100分
            # 值高于等于critical阈值：资源极度紧张，得0分
            # 值在low和critical之间：线性映射到0-100分
            if value < thresholds["low"]:
                scores[resource_name] = 100  # 资源充足
            elif value >= thresholds["critical"]:
                scores[resource_name] = 0    # 资源严重不足
            else:
                # 线性映射到0-100分
                # 公式：100 - ((当前值 - 低值)/(临界值 - 低值)) * 100
                # 确保值越高（资源越紧张），分数越低
                scores[resource_name] = 100 - ((value - thresholds["low"]) / 
                                             (thresholds["critical"] - thresholds["low"])) * 100
        
        # 加权计算综合评分 - 根据预设的资源权重进行加权平均
        total_score = 0
        for resource_name, score in scores.items():
            total_score += score * self.resource_weights[resource_name]
        
        return total_score
        
    def _update_state(self, resource_score: float) -> str:
        """
        更新状态机状态
        
        该方法根据资源综合评分更新系统状态，状态机包含四个状态：
        - idle: 资源充足（评分>=70）
        - normal: 资源正常（评分>=40且<70）
        - pressure: 资源压力（评分>=20且<40）
        - critical: 资源危急（评分<20）
        
        参数:
            resource_score: 资源综合评分 (0-100)
        
        返回:
            新的状态 (idle, normal, pressure, critical)
        """
        # 根据资源综合评分确定系统状态
        if resource_score >= 70:
            new_state = "idle"       # 资源充足状态
        elif resource_score >= 40:
            new_state = "normal"     # 资源正常状态
        elif resource_score >= 20:
            new_state = "pressure"   # 资源压力状态
        else:
            new_state = "critical"   # 资源危急状态
        
        # 保存旧状态并更新当前状态
        old_state = self.current_state
        self.current_state = new_state
        
        # 如果状态发生变化，打印状态转换日志
        if old_state != new_state:
            print(f"状态变化: {old_state} -> {new_state} (资源评分: {resource_score:.2f})")
            
        return new_state
        
    def _smooth_with_history(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用历史记录平滑策略调整
        
        该方法通过历史配置记录对新配置进行平滑处理，避免参数的剧烈波动。
        采用加权平均策略：70%新配置值 + 30%历史平均值，确保系统参数变化更加平稳。
        
        参数:
            new_config: 新计算出的配置参数
        
        返回:
            平滑处理后的配置参数
        """
        # 如果历史记录不足2条，无法进行有效平滑，直接返回新配置
        if len(self.history) < 2:
            return new_config
            
        smoothed_config = {}
        # 对每个参数进行平滑处理
        for param_name, param_value in new_config.items():
            # 收集最近3条历史记录中的对应参数值
            history_values = []
            for history_entry in self.history[-3:]:  # 取最近3条记录
                if param_name in history_entry['config']:
                    history_values.append(history_entry['config'][param_name])
                    
            # 如果有历史值，进行平滑处理
            if history_values:
                # 计算历史平均值
                avg_history_value = sum(history_values) / len(history_values)
                # 加权平均：70%新配置 + 30%历史平均
                # 这种加权策略既保证了对当前资源状态的响应，又避免了参数的剧烈波动
                smoothed_value = 0.7 * param_value + 0.3 * avg_history_value
                # 取整，确保参数为整数
                smoothed_config[param_name] = int(smoothed_value)
            else:
                # 没有历史记录，直接使用新配置值
                smoothed_config[param_name] = param_value
                
        return smoothed_config
        
    def _apply_constraints(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用参数约束
        
        该方法确保所有参数都在预设的合理范围内，防止极端值导致系统不稳定。
        对于每个参数，将其限制在对应的最小值和最大值之间。
        
        参数:
            config: 需要应用约束的配置参数
        
        返回:
            应用约束后确保在合理范围内的配置参数
        """
        constrained_config = {}
        # 对每个参数应用约束
        for param_name, param_value in config.items():
            if param_name in self.parameter_constraints:
                # 获取该参数的约束范围（最小值和最大值）
                constraints = self.parameter_constraints[param_name]
                # 使用max和min函数确保参数值在约束范围内
                constrained_value = max(constraints["min"], min(constraints["max"], param_value))
                constrained_config[param_name] = constrained_value
            else:
                # 没有约束的参数，保持原值
                constrained_config[param_name] = param_value
                
        return constrained_config
        
    def _generate_adjustment_reasons(self, resource_status: Dict[str, float], 
                                    current_config: Dict[str, Any], 
                                    new_config: Dict[str, Any]) -> List[str]:
        """
        生成配置调整的原因说明
        
        该方法分析当前资源状态和配置变化，生成人类可读的调整原因说明。
        包括两部分：资源状态分析和配置参数变化分析。
        
        参数:
            resource_status: 当前资源状态字典
            current_config: 调整前的配置参数
            new_config: 调整后的配置参数
        
        返回:
            原因说明文本列表，每条说明包含一个调整原因
        """
        reasons = []
        
        # 第一部分：分析资源状态，生成资源相关的原因说明
        resource_levels = {}
        for resource_name, value in resource_status.items():
            resource_levels[resource_name] = self._classify_resource_level(resource_name, value)
            
        # 根据资源等级生成不同级别的说明
        for resource_name, level in resource_levels.items():
            value = resource_status[resource_name]
            if level == "critical":  # 临界状态 - 需要立即采取措施
                if resource_name == "qps":
                    reasons.append(f"QPS过高 ({value})，系统负载严重，需要降低处理复杂度")
                elif resource_name == "mem_usage":
                    reasons.append(f"内存使用率过高 ({value}%)，系统内存紧张，需要减少内存占用")
                else:  # latency
                    reasons.append(f"延迟过高 ({value}ms)，用户体验差，需要优化响应速度")
            elif level == "high":  # 高负载状态 - 需要注意并适当调整
                if resource_name == "qps":
                    reasons.append(f"QPS较高 ({value})，系统负载较重")
                elif resource_name == "mem_usage":
                    reasons.append(f"内存使用率较高 ({value}%)，系统内存压力较大")
                else:  # latency
                    reasons.append(f"延迟较高 ({value}ms)，需要注意用户体验")
            elif level == "low" and resource_name != "qps":  # 低负载状态 - 可以考虑提升性能
                # QPS低通常是好事，不需要特别说明
                if resource_name == "mem_usage":
                    reasons.append(f"内存使用率较低 ({value}%)，有足够空间提升处理能力")
                else:  # latency
                    reasons.append(f"延迟较低 ({value}ms)，系统响应良好")
        
        # 第二部分：分析配置变化，生成参数调整相关的原因说明
        for param_name, new_value in new_config.items():
            if param_name in current_config:
                old_value = current_config[param_name]
                if new_value != old_value:  # 只记录发生变化的参数
                    change = new_value - old_value
                    if change > 0:
                        reasons.append(f"{param_name}从 {old_value} 增加到 {new_value}")
                    else:
                        reasons.append(f"{param_name}从 {old_value} 减少到 {new_value}")
                        
        return reasons
        
    def get_adaptive_config(self, resource_status: Dict[str, float], 
                           current_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        获取基于资源状态的自适应配置
        
        该方法是模拟器的核心方法，根据当前资源状态（QPS、内存使用率、延迟）
        动态调整分块策略参数。实现了从资源状态到配置参数的完整映射过程。
        
        参数:
            resource_status: 资源状态字典 {"qps": 值, "mem_usage": 值, "latency": 值}
            current_config: 当前配置（可选，默认为基础配置）
        
        返回:
            包含新配置、资源评分、系统状态和调整原因的结果字典
        """
        # 步骤1: 确定当前配置 - 使用提供的配置或默认的基础配置
        if current_config is None:
            current_config = self.base_config.copy()
        else:
            # 合并当前配置和基础配置，确保所有必要参数都存在
            # 使用字典解包操作符(**)合并配置，优先使用current_config中的值
            current_config = {**self.base_config, **current_config}
        
        # 步骤2: 计算资源综合评分 - 将多个资源指标综合为一个0-100的评分
        resource_score = self._calculate_resource_score(resource_status)
        
        # 步骤3: 更新状态机状态 - 根据资源评分确定当前系统状态
        state = self._update_state(resource_score)
        
        # 步骤4: 根据状态生成新配置 - 实现不同状态下的策略调整逻辑
        new_config = current_config.copy()
        
        if state == "critical":
            # 临界状态: 资源严重不足，需要激进调整以确保系统稳定
            # 采用双倍步长进行快速降载
            new_config["chunk_size"] -= self.adjustment_steps["chunk_size"]["decrease"] * 2  # 减小分块大小，降低处理复杂度
            new_config["overlap"] -= self.adjustment_steps["overlap"]["decrease"] * 2      # 减小重叠，减少重复处理
            new_config["concurrency"] -= self.adjustment_steps["concurrency"]["decrease"] * 2  # 降低并发，减轻系统压力
            new_config["top_k"] -= self.adjustment_steps["top_k"]["decrease"] * 2          # 减少返回结果，降低处理和传输成本
            new_config["rerank_top_k"] -= self.adjustment_steps["rerank_top_k"]["decrease"] * 2  # 减少重排序数量，降低计算成本
        elif state == "pressure":
            # 压力状态: 资源压力较大，需要适度调整以平衡性能和稳定性
            # 采用标准步长进行降载
            new_config["chunk_size"] -= self.adjustment_steps["chunk_size"]["decrease"]  # 适度减小分块大小
            new_config["overlap"] -= self.adjustment_steps["overlap"]["decrease"]      # 适度减小重叠
            new_config["concurrency"] -= self.adjustment_steps["concurrency"]["decrease"]  # 适度降低并发
            new_config["top_k"] -= self.adjustment_steps["top_k"]["decrease"]          # 适度减少返回结果
        elif state == "idle":
            # 空闲状态: 资源充足，可以适当提升处理质量和能力
            # 采用增加步长进行升载
            new_config["chunk_size"] += self.adjustment_steps["chunk_size"]["increase"]  # 增大分块，保留更多上下文
            new_config["overlap"] += self.adjustment_steps["overlap"]["increase"]      # 增加重叠，减少信息丢失
            new_config["concurrency"] += self.adjustment_steps["concurrency"]["increase"]  # 提高并发，加速处理
            new_config["top_k"] += self.adjustment_steps["top_k"]["increase"]          # 增加返回结果，提高召回率
            new_config["rerank_top_k"] += self.adjustment_steps["rerank_top_k"]["increase"]  # 增加重排序数量，提高排序质量
        # 正常状态(state == "normal"): 不做调整，保持当前配置
        
        # 步骤5: 应用参数约束 - 确保参数在合理范围内
        new_config = self._apply_constraints(new_config)
        
        # 步骤6: 使用历史记录平滑策略调整 - 避免参数的剧烈波动
        new_config = self._smooth_with_history(new_config)
        
        # 步骤7: 生成调整原因 - 提供人类可读的调整解释
        reasons = self._generate_adjustment_reasons(resource_status, current_config, new_config)
        
        # 步骤8: 记录本次调整到历史 - 用于后续的平滑处理和趋势分析
        self.history.append({
            "timestamp": time.time(),     # 调整时间戳
            "resource_status": resource_status,  # 当前资源状态
            "config": new_config,        # 调整后的配置
            "state": state,              # 当前系统状态
            "score": resource_score      # 资源综合评分
        })
        
        # 保持历史记录窗口大小 - 只保留最近的历史记录
        if len(self.history) > self.history_window:
            self.history.pop(0)  # 移除最旧的记录
            
        # 步骤9: 返回结果 - 包含完整的调整信息
        return {
            "config": new_config,          # 调整后的新配置
            "resource_score": round(resource_score, 2),  # 四舍五入的资源综合评分
            "state": state,                # 系统状态
            "reasons": reasons             # 调整原因列表
        }
        
    def simulate_load_changes(self, load_profile: List[Dict[str, float]], 
                             initial_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        模拟负载变化情况下的策略调整
        
        该方法用于模拟一系列资源状态变化下的策略调整过程，帮助评估
        自适应策略在不同负载场景下的表现。支持从低负载到高负载再到恢复正常
        的完整负载变化周期模拟。
        
        参数:
            load_profile: 负载变化配置文件，包含不同时间点的资源状态列表
            initial_config: 初始配置（可选，默认为基础配置）
        
        返回:
            模拟结果列表，包含每个时间点的完整调整结果
        """
        results = []  # 存储所有模拟步骤的结果
        current_config = initial_config or self.base_config.copy()  # 使用提供的初始配置或基础配置
        
        # 打印模拟开始信息
        print("开始负载变化模拟...")
        print(f"初始配置: {current_config}")
        
        # 逐个模拟负载变化场景
        for i, resource_status in enumerate(load_profile):
            # 打印当前模拟步骤信息
            print(f"\n模拟步骤 {i+1}/{len(load_profile)} - 资源状态: {resource_status}")
            
            # 获取当前资源状态下的自适应配置
            result = self.get_adaptive_config(resource_status, current_config)
            results.append(result)  # 保存当前步骤的结果
            
            # 更新当前配置，用于下一步模拟
            current_config = result["config"]
            
            # 打印当前步骤的详细结果
            print(f"  资源评分: {result['resource_score']}")
            print(f"  系统状态: {result['state']}")
            print(f"  新配置: {result['config']}")
            print("  调整原因:")
            for reason in result['reasons']:
                print(f"    - {reason}")
                
            # 模拟时间间隔 - 增加延迟以更好地观察模拟过程
            # 注释：这行可以根据需要取消注释，实际生产环境中通常不需要延迟
            # time.sleep(0.5)  # 可选：添加延迟以观察模拟过程
            
        return results  # 返回完整的模拟结果列表
        
    def save_simulation_results(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """
        保存模拟结果到JSON文件
        
        该方法将模拟过程中生成的所有策略调整结果保存到指定的JSON文件中，
        方便后续分析和可视化。使用UTF-8编码确保中文字符正常显示。
        
        参数:
            results: 模拟结果列表，包含每个时间点的完整调整结果
            file_path: 保存文件路径，通常使用.json扩展名
        """
        try:
            # 转换不可JSON序列化的对象
            serializable_results = []
            for result in results:
                serializable_result = {
                    "config": result["config"],
                    "resource_score": result["resource_score"],
                    "state": result["state"],
                    "reasons": result["reasons"]
                }
                serializable_results.append(serializable_result)
            
            # 以写入模式打开文件，使用UTF-8编码确保中文正常显示
            with open(file_path, 'w', encoding='utf-8') as f:
                # 保存结果为格式化的JSON，ensure_ascii=False确保中文不转义
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            # 打印保存成功信息
            print(f"\n模拟结果已保存到 {file_path}")
        except Exception as e:
            # 捕获并打印可能的异常
            print(f"保存结果时出错: {e}")



# 示例用法
if __name__ == "__main__":
    # 创建模拟器实例 - 使用默认配置初始化
    simulator = ResourceAdaptiveSimulator()
    
    # 定义负载变化配置文件 - 模拟从低负载到高负载再恢复的完整过程
    # 该配置文件设计了6个负载变化场景，全面测试自适应策略的响应能力
    load_profile = [
        # 场景1: 低负载 - 系统资源充足，可提升处理质量
        {"qps": 20, "mem_usage": 30, "latency": 200},
        # 场景2: 正常负载 - 系统运行稳定，无需调整策略
        {"qps": 70, "mem_usage": 55, "latency": 700},
        # 场景3: 高负载 - 系统压力明显，需要适度降载
        {"qps": 130, "mem_usage": 85, "latency": 1300},
        # 场景4: 临界负载 - 系统资源严重不足，需要激进调整
        {"qps": 160, "mem_usage": 95, "latency": 2000},
        # 场景5: 恢复到正常负载 - 系统压力减轻，逐步恢复正常配置
        {"qps": 80, "mem_usage": 60, "latency": 800},
        # 场景6: 恢复到低负载 - 系统资源充足，可再次提升处理能力
        {"qps": 30, "mem_usage": 40, "latency": 300}
    ]
    
    # 打印模拟器基本信息 - 展示初始配置参数
    print("===== 资源自适应分块策略模拟器 =====")
    print(f"基础配置: {simulator.base_config}")
    print(f"资源阈值: {simulator.resource_thresholds}")
    print(f"参数调整步长: {simulator.adjustment_steps}")
    print("\n开始模拟不同负载场景下的策略调整...\n")
    
    # 运行模拟 - 执行完整的负载变化模拟过程
    # 该过程会依次处理load_profile中的每个资源状态场景
    simulation_results = simulator.simulate_load_changes(load_profile)
    
    # 保存结果 - 将模拟过程的详细结果保存到JSON文件
    # 便于后续分析策略调整的效果和系统状态变化趋势
    simulator.save_simulation_results(simulation_results, "simulation_results.json")
    
    # 单独演示单个资源状态的调整 - 展示特定资源状态下的策略调整过程
    # 这里选择了一个高负载场景进行详细展示
    print("\n===== 演示单个资源状态调整 =====")
    test_resource_status = {"qps": 120, "mem_usage": 85, "latency": 1500}  # 测试高负载状态
    print(f"测试资源状态: {test_resource_status}")
    
    # 获取自适应配置 - 根据当前资源状态计算最优参数配置
    adaptive_result = simulator.get_adaptive_config(test_resource_status)
    
    # 打印结果 - 展示资源评分、系统状态、推荐配置和调整原因
    print(f"资源综合评分: {adaptive_result['resource_score']}")
    print(f"系统状态: {adaptive_result['state']}")
    print(f"推荐配置: {adaptive_result['config']}")
    print("调整原因:")
    for reason in adaptive_result['reasons']:
        print(f"  - {reason}")
        
    print("\n===== 模拟结束 =====")