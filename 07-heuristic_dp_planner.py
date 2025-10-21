#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应检索路径启发式规划

功能说明：
    以启发式代价函数选择"下一步动作"，输出最优可行路径。

作者：Ph.D. Rhino

内容概述：
    定义动作集合与代价/收益估计函数：(C = w_q · Δquality - w_l · Δlatency - w_c · Δcost)；
    在状态空间中逐步选取收益最大的动作，直到终止；记录每步边际收益与全局结果。

执行流程：
    1. 定义状态（覆盖、矛盾、预算）与动作（向量/图谱/跨模态/自检/回退）。
    2. 计算各动作的边际收益与代价分数。
    3. 选择最大(C)动作并转移状态，检查终止条件。
    4. 输出路径、累计收益、SLO合规性。
    5. 保存代价曲线与收益递减点。

输入说明：
    --weights 0.6 0.3 0.1 --budget 1500 --init 0.55

输出展示：
    Chosen path: [vec_expand, graph, self_check, targeted_ocr]
    Marginal gains: [ +0.06, +0.11, +0.03, contradiction -2.2% ]
    Stop: diminishing_returns
"""

import os
import json
import time
import random
import argparse
import numpy as np
from collections import defaultdict, deque


def install_dependencies():
    """
    检查并安装必要的依赖包
    """
    try:
        import numpy as np
        print("所有依赖包已安装完成。")
    except ImportError:
        print("正在安装依赖包...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "numpy"])
            print("依赖包安装成功。")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            exit(1)


def generate_action_space():
    """
    生成动作空间配置
    
    Returns:
        dict: 动作空间配置
    """
    action_space = {
        "vec_expand": {
            "name": "向量扩展检索",
            "effects": {
                "quality_improvement": 0.06,
                "latency_increase": 50,
                "cost_increase": 0.05,
                "contradiction_change": 0.5
            },
            "description": "扩展向量检索范围，提高召回率"
        },
        "graph": {
            "name": "图谱推理",
            "effects": {
                "quality_improvement": 0.11,
                "latency_increase": 80,
                "cost_increase": 0.12,
                "contradiction_change": -1.0
            },
            "description": "利用知识图谱进行关联推理，提升准确性"
        },
        "cross_modal": {
            "name": "跨模态验证",
            "effects": {
                "quality_improvement": 0.08,
                "latency_increase": 120,
                "cost_increase": 0.20,
                "contradiction_change": -3.5
            },
            "description": "跨模态信息交叉验证，解决复杂问题"
        },
        "self_check": {
            "name": "自检优化",
            "effects": {
                "quality_improvement": 0.03,
                "latency_increase": 30,
                "cost_increase": 0.02,
                "contradiction_change": -0.8
            },
            "description": "基于现有结果进行自检和局部优化"
        },
        "targeted_ocr": {
            "name": "定向OCR验证",
            "effects": {
                "quality_improvement": 0.04,
                "latency_increase": 70,
                "cost_increase": 0.08,
                "contradiction_change": -2.2
            },
            "description": "对矛盾点进行定向OCR内容验证"
        },
        "fallback": {
            "name": "回退策略",
            "effects": {
                "quality_improvement": 0.0,
                "latency_increase": 20,
                "cost_increase": 0.01,
                "contradiction_change": 0.0
            },
            "description": "使用预设的回退方案"
        }
    }
    return action_space


class State:
    """
    状态类，表示当前系统的状态
    """
    def __init__(self, initial_coverage=0.55, initial_contradiction=3.5, 
                 initial_budget=1500, initial_cost=0.0):
        self.coverage = initial_coverage  # 覆盖度
        self.contradiction = initial_contradiction  # 矛盾率(%)
        self.budget = initial_budget  # 预算(毫秒)
        self.used_budget = 0  # 已使用预算
        self.cost = initial_cost  # 成本因子
        self.history = []  # 历史动作记录
        self.quality_history = [initial_coverage]  # 质量历史
        self.contradiction_history = [initial_contradiction]  # 矛盾率历史
        self.budget_history = [initial_budget]  # 预算历史
        self.marginal_gains = []  # 边际收益记录
        self.steps = 0  # 执行步数
    
    def update(self, action, action_effects):
        """
        根据执行的动作更新状态
        
        Args:
            action: 执行的动作名称
            action_effects: 动作效果字典
        """
        # 计算实际效果（添加一些随机性模拟真实场景）
        quality_improvement = action_effects["quality_improvement"] * (0.9 + 0.2 * random.random())
        latency_increase = int(action_effects["latency_increase"] * (0.95 + 0.1 * random.random()))
        cost_increase = action_effects["cost_increase"] * (0.9 + 0.2 * random.random())
        contradiction_change = action_effects["contradiction_change"] * (0.9 + 0.2 * random.random())
        
        # 更新状态
        self.coverage = min(1.0, self.coverage + quality_improvement)
        self.contradiction = max(0.0, self.contradiction + contradiction_change)
        self.used_budget += latency_increase
        self.budget = max(0, self.budget - latency_increase)
        self.cost += cost_increase
        self.steps += 1
        
        # 记录边际收益
        if action == "targeted_ocr" and abs(contradiction_change) > 1.0:
            # 对于定向OCR，重点记录矛盾率改善
            self.marginal_gains.append(f"contradiction {contradiction_change:+.1f}%")
        else:
            self.marginal_gains.append(f"{quality_improvement:+.2f}")
        
        # 记录历史
        self.history.append({
            "step": self.steps,
            "action": action,
            "coverage": self.coverage,
            "contradiction": self.contradiction,
            "used_budget": self.used_budget,
            "remaining_budget": self.budget,
            "quality_improvement": quality_improvement,
            "contradiction_change": contradiction_change,
            "latency_increase": latency_increase
        })
        
        self.quality_history.append(self.coverage)
        self.contradiction_history.append(self.contradiction)
        self.budget_history.append(self.budget)
    
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            dict: 状态字典
        """
        return {
            "coverage": self.coverage,
            "contradiction": self.contradiction,
            "budget": self.budget,
            "used_budget": self.used_budget,
            "cost": self.cost,
            "steps": self.steps
        }
    
    def should_terminate(self, max_steps=10, min_improvement=0.015):
        """
        判断是否应该终止执行
        
        Args:
            max_steps: 最大步数
            min_improvement: 最小改进阈值
        
        Returns:
            tuple: (是否终止, 终止原因)
        """
        # 步数达到上限
        if self.steps >= max_steps:
            return True, "max_steps_reached"
        
        # 预算耗尽
        if self.budget <= 0:
            return True, "budget_exhausted"
        
        # 覆盖度已很高
        if self.coverage >= 0.95:
            return True, "high_coverage"
        
        # 矛盾率已很低
        if self.contradiction <= 0.5:
            return True, "low_contradiction"
        
        # 收益递减
        if len(self.quality_history) >= 3:
            recent_improvements = [
                self.quality_history[-1] - self.quality_history[-2],
                self.quality_history[-2] - self.quality_history[-3]
            ]
            if all(imp < min_improvement for imp in recent_improvements):
                return True, "diminishing_returns"
        
        return False, "continue"


class HeuristicPlanner:
    """
    启发式规划器，使用代价函数选择最优动作
    """
    def __init__(self, action_space, weights=(0.6, 0.3, 0.1)):
        self.action_space = action_space
        self.weights = {
            "quality": weights[0],  # 质量权重
            "latency": weights[1],  # 延迟权重
            "cost": weights[2]      # 成本权重
        }
    
    def calculate_action_score(self, state, action_name, action_effects):
        """
        计算动作的启发式代价分数
        
        代价函数: C = w_q · Δquality - w_l · Δlatency - w_c · Δcost
        
        Args:
            state: 当前状态
            action_name: 动作名称
            action_effects: 动作效果
        
        Returns:
            float: 动作分数
        """
        # 检查预算是否足够
        if state.budget < action_effects["latency_increase"]:
            return -float('inf')  # 预算不足，不考虑此动作
        
        # 计算分数
        quality_score = self.weights["quality"] * action_effects["quality_improvement"]
        latency_score = -self.weights["latency"] * (action_effects["latency_increase"] / 1000)  # 转换为秒
        cost_score = -self.weights["cost"] * action_effects["cost_increase"]
        
        # 综合分数
        total_score = quality_score + latency_score + cost_score
        
        # 对矛盾率高的情况，优先考虑能降低矛盾率的动作
        if state.contradiction > 2.0 and action_effects["contradiction_change"] < -1.0:
            total_score += 0.05  # 矛盾率高时，给能降低矛盾率的动作额外加分
        
        # 对高质量状态，优先考虑低成本动作
        if state.coverage > 0.8:
            cost_score *= 1.5  # 高质量时，增加成本权重
            total_score = quality_score + latency_score + cost_score
        
        return total_score
    
    def choose_next_action(self, state):
        """
        选择下一个最优动作
        
        Args:
            state: 当前状态
        
        Returns:
            tuple: (最佳动作名称, 最佳动作效果, 最佳分数)
        """
        best_action = None
        best_effects = None
        best_score = -float('inf')
        
        # 计算每个动作的分数
        for action_name, action_info in self.action_space.items():
            score = self.calculate_action_score(state, action_name, action_info["effects"])
            
            if score > best_score:
                best_score = score
                best_action = action_name
                best_effects = action_info["effects"]
        
        # 如果所有动作分数都很低，考虑回退
        if best_score < 0.01 and "fallback" in self.action_space:
            return "fallback", self.action_space["fallback"]["effects"], best_score
        
        return best_action, best_effects, best_score


def format_output_path(path):
    """
    格式化输出路径
    
    Args:
        path: 动作路径列表
    
    Returns:
        str: 格式化的路径字符串
    """
    return f"Chosen path: [{', '.join(path)}]"


def format_marginal_gains(gains):
    """
    格式化边际收益
    
    Args:
        gains: 边际收益列表
    
    Returns:
        str: 格式化的边际收益字符串
    """
    return f"Marginal gains: [ {', '.join(gains)} ]"


def format_stop_reason(reason):
    """
    格式化终止原因
    
    Args:
        reason: 终止原因
    
    Returns:
        str: 格式化的终止原因字符串
    """
    reason_map = {
        "max_steps_reached": "max_steps",
        "budget_exhausted": "budget_exhausted",
        "high_coverage": "high_coverage",
        "low_contradiction": "low_contradiction",
        "diminishing_returns": "diminishing_returns"
    }
    return f"Stop: {reason_map.get(reason, reason)}"


def save_planning_results(state, stop_reason, action_space, output_file="planning_results.json"):
    """
    保存规划结果到JSON文件
    
    Args:
        state: 最终状态
        stop_reason: 终止原因
        action_space: 动作空间
        output_file: 输出文件名
    """
    results = {
        "initial_state": {
            "coverage": state.quality_history[0],
            "contradiction": state.contradiction_history[0],
            "budget": state.budget_history[0]
        },
        "final_state": {
            "coverage": state.coverage,
            "contradiction": state.contradiction,
            "remaining_budget": state.budget,
            "used_budget": state.used_budget,
            "steps": state.steps
        },
        "planning_path": [h["action"] for h in state.history],
        "marginal_gains": state.marginal_gains,
        "stop_reason": stop_reason,
        "action_details": [{
            "step": h["step"],
            "action": h["action"],
            "action_name": action_space[h["action"]]["name"],
            "coverage_improvement": h["quality_improvement"],
            "contradiction_change": h["contradiction_change"],
            "latency_increase": h["latency_increase"]
        } for h in state.history],
        "metrics": {
            "total_coverage_improvement": state.coverage - state.quality_history[0],
            "total_contradiction_reduction": state.contradiction_history[0] - state.contradiction,
            "efficiency": (state.coverage - state.quality_history[0]) / state.used_budget if state.used_budget > 0 else 0,
            "quality_cost_ratio": (state.coverage - state.quality_history[0]) / state.cost if state.cost > 0 else 0
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"规划结果已保存到: {output_file}")
    return results


def generate_cost_curve(state, output_file="cost_curve.csv"):
    """
    生成代价曲线数据
    
    Args:
        state: 状态对象
        output_file: 输出文件名
    """
    with open(output_file, "w", encoding="utf-8-sig") as f:
        # 写入标题
        f.write("step,coverage,contradiction,used_budget,remaining_budget\n")
        
        # 写入初始状态
        f.write(f"0,{state.quality_history[0]:.4f},{state.contradiction_history[0]:.4f},0,{state.budget_history[0]}\n")
        
        # 写入每一步的状态
        for i, history in enumerate(state.history):
            f.write(f"{i+1},{history['coverage']:.4f},{history['contradiction']:.4f},{history['used_budget']},{history['remaining_budget']}\n")
    
    print(f"代价曲线已保存到: {output_file}")


def analyze_diminishing_returns(state, output_file="diminishing_returns_analysis.txt"):
    """
    分析收益递减点
    
    Args:
        state: 状态对象
        output_file: 输出文件名
    """
    # 计算每一步的边际改进
    marginal_improvements = []
    for i in range(1, len(state.quality_history)):
        improvement = state.quality_history[i] - state.quality_history[i-1]
        marginal_improvements.append(improvement)
    
    # 找到收益递减点
    diminishing_points = []
    if len(marginal_improvements) >= 2:
        for i in range(1, len(marginal_improvements)):
            # 如果当前改进小于前一次的80%，认为出现递减
            if marginal_improvements[i] < marginal_improvements[i-1] * 0.8:
                diminishing_points.append({
                    "step": i+1,
                    "current_improvement": marginal_improvements[i],
                    "previous_improvement": marginal_improvements[i-1],
                    "ratio": marginal_improvements[i] / marginal_improvements[i-1]
                })
    
    # 保存分析结果
    with open(output_file, "w", encoding="utf-8-sig") as f:
        f.write("===== 收益递减点分析 =====\n\n")
        
        # 边际改进列表
        f.write("边际改进序列:\n")
        for i, imp in enumerate(marginal_improvements):
            f.write(f"  第{i+1}步: +{imp:.4f}\n")
        f.write("\n")
        
        # 收益递减点
        f.write("收益递减点:\n")
        if diminishing_points:
            for point in diminishing_points:
                f.write(f"  第{point['step']}步: 比例={point['ratio']:.2f} (当前={point['current_improvement']:.4f}, 前一步={point['previous_improvement']:.4f})\n")
        else:
            f.write("  未检测到明显的收益递减点\n")
        f.write("\n")
        
        # 累计改进曲线分析
        f.write("累计改进分析:\n")
        total_improvement = state.quality_history[-1] - state.quality_history[0]
        f.write(f"  总改进: +{total_improvement:.4f}\n")
        
        if state.steps > 0:
            avg_improvement = total_improvement / state.steps
            f.write(f"  平均每步改进: +{avg_improvement:.4f}\n")
            
            # 计算前半段和后半段的平均改进
            half_steps = state.steps // 2
            if half_steps > 0:
                first_half_avg = (state.quality_history[half_steps] - state.quality_history[0]) / half_steps
                second_half_avg = (state.quality_history[-1] - state.quality_history[half_steps]) / (state.steps - half_steps)
                f.write(f"  前半段平均改进: +{first_half_avg:.4f}\n")
                f.write(f"  后半段平均改进: +{second_half_avg:.4f}\n")
                
                if second_half_avg < first_half_avg * 0.7:
                    f.write("  结论: 存在明显的收益递减现象\n")
                elif second_half_avg < first_half_avg:
                    f.write("  结论: 存在轻微的收益递减现象\n")
                else:
                    f.write("  结论: 未发现收益递减现象\n")
    
    print(f"收益递减分析已保存到: {output_file}")


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="自适应检索路径启发式规划")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.6, 0.3, 0.1], 
                        help="质量、延迟、成本的权重，三个浮点数，和为1")
    parser.add_argument("--budget", type=int, default=1500, help="预算限制(毫秒)")
    parser.add_argument("--init", type=float, default=0.55, help="初始覆盖度")
    parser.add_argument("--max-steps", type=int, default=10, help="最大执行步数")
    args = parser.parse_args()
    
    # 检查依赖
    install_dependencies()
    
    # 验证权重和
    weights_sum = sum(args.weights)
    if abs(weights_sum - 1.0) > 1e-6:
        print(f"警告: 权重和 {weights_sum:.3f} 不等于 1.0，将自动归一化")
        # 归一化权重
        args.weights = [w / weights_sum for w in args.weights]
        print(f"归一化后的权重: {args.weights}")
    
    # 生成动作空间
    action_space = generate_action_space()
    
    # 初始化状态
    state = State(
        initial_coverage=args.init,
        initial_contradiction=3.5,
        initial_budget=args.budget
    )
    
    # 初始化启发式规划器
    planner = HeuristicPlanner(action_space, weights=args.weights)
    
    print(f"\n===== 开始自适应检索路径启发式规划 =====")
    print(f"初始状态: 覆盖度={state.coverage:.2f}, 矛盾率={state.contradiction:.1f}%, 预算={state.budget}ms")
    print(f"权重设置: 质量={args.weights[0]:.1f}, 延迟={args.weights[1]:.1f}, 成本={args.weights[2]:.1f}")
    print(f"最大步数: {args.max_steps}")
    print()
    
    # 主循环
    start_time = time.time()
    
    while True:
        # 检查终止条件
        should_terminate, reason = state.should_terminate(args.max_steps)
        if should_terminate:
            print(f"终止执行: {reason}")
            break
        
        # 选择下一个最优动作
        action, effects, score = planner.choose_next_action(state)
        
        # 执行动作并更新状态
        state.update(action, effects)
        
        # 输出当前步骤信息
        action_name = action_space[action]["name"]
        last_history = state.history[-1]
        print(f"步骤 {state.steps}: 选择 {action} ({action_name}) - 分数: {score:.4f}")
        print(f"  覆盖度: {last_history['coverage']:.4f} (+{last_history['quality_improvement']:.4f})")
        print(f"  矛盾率: {last_history['contradiction']:.2f}% ({last_history['contradiction_change']:+.2f}%)")
        print(f"  延迟增加: {last_history['latency_increase']}ms, 剩余预算: {last_history['remaining_budget']}ms")
        print()
    
    execution_time = time.time() - start_time
    
    # 格式化输出最终结果
    path_str = format_output_path([h["action"] for h in state.history])
    gains_str = format_marginal_gains(state.marginal_gains)
    stop_str = format_stop_reason(reason)
    
    print("\n===== 规划结果摘要 =====")
    print(path_str)
    print(gains_str)
    print(stop_str)
    print(f"最终覆盖度: {state.coverage:.4f} (+{state.coverage - state.quality_history[0]:.4f})")
    print(f"最终矛盾率: {state.contradiction:.2f}% (-{state.contradiction_history[0] - state.contradiction:.2f}%)")
    print(f"总延迟: {state.used_budget}ms, 剩余预算: {state.budget}ms")
    print(f"执行步数: {state.steps}")
    
    # 保存结果
    results = save_planning_results(state, reason, action_space)
    generate_cost_curve(state)
    analyze_diminishing_returns(state)
    
    print(f"\n===== 执行完成 =====")
    print(f"程序执行时间: {execution_time:.3f}秒")
    
    # 验证中文显示
    print("\n验证中文显示：")
    sample_cn_text = "启发式规划 边际收益 代价函数 收益递减 最优路径"
    print(f"样本中文文本: {sample_cn_text}")


if __name__ == "__main__":
    main()