#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent策略调度（信号-动作-预算-回退）

功能说明：
    基于信号（覆盖、不确定度、矛盾）与预算约束调度动作序列。

作者：Ph.D. Rhino

内容概述：
    定义状态字典（覆盖度、矛盾率、预算余量），设策略映射（增检、图谱、跨模态、自检、回退）；
    迭代执行动作并更新状态；步数上限与收益递减终止；导出策略轨迹与指标变化。

执行流程：
    1. 初始化状态（覆盖=0.x、矛盾= y%、budget=ms/￥）
    2. 基于阈值选择动作；执行后更新状态与预算
    3. 触发必要节点才启跨模态/图谱；否则轻增检
    4. 终止条件：步数上限/收益递减/预算耗尽
    5. 导出轨迹日志与最终决策摘要

输入说明：
    --thresholds conf.json --budget 1500ms --max-steps 5

输出展示：
    Step1: action=light_expand_k → coverage 0.62 (+0.05), P95 +40ms
    Step2: action=graph_lookup → coverage 0.74 (+0.12), +70ms
    Step3: action=ocr_verify(targeted) → contradiction 2.6% (-2.1%), +80ms
    Final: coverage=0.83, contradiction=2.6%, P95_delta=+190ms (within SLO)
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


def generate_config_file():
    """
    生成配置文件conf.json
    """
    config = {
        "thresholds": {
            "coverage": {
                "low": 0.6,      # 低覆盖阈值
                "medium": 0.75,  # 中等覆盖阈值
                "high": 0.85     # 高覆盖阈值
            },
            "contradiction": {
                "warning": 3.0,   # 矛盾率警告阈值
                "critical": 5.0   # 矛盾率严重阈值
            },
            "uncertainty": {
                "high": 0.3,      # 高不确定性阈值
                "medium": 0.2     # 中等不确定性阈值
            }
        },
        "action_costs": {
            "light_expand_k": {"time_ms": 40, "description": "轻度扩展候选数量"},
            "heavy_expand_k": {"time_ms": 90, "description": "重度扩展候选数量"},
            "graph_lookup": {"time_ms": 70, "description": "图谱检索"},
            "ocr_verify": {"time_ms": 80, "description": "OCR验证"},
            "asr_verify": {"time_ms": 100, "description": "ASR验证"},
            "cross_verify": {"time_ms": 150, "description": "跨模态验证"},
            "self_check": {"time_ms": 30, "description": "自检"},
            "fallback": {"time_ms": 20, "description": "回退策略"}
        },
        "action_effects": {
            "light_expand_k": {"coverage_improvement": 0.05, "contradiction_change": 0.0},
            "heavy_expand_k": {"coverage_improvement": 0.1, "contradiction_change": 1.0},
            "graph_lookup": {"coverage_improvement": 0.12, "contradiction_change": -0.5},
            "ocr_verify": {"coverage_improvement": 0.03, "contradiction_change": -2.1},
            "asr_verify": {"coverage_improvement": 0.04, "contradiction_change": -1.8},
            "cross_verify": {"coverage_improvement": 0.08, "contradiction_change": -3.5},
            "self_check": {"coverage_improvement": 0.01, "contradiction_change": -0.3},
            "fallback": {"coverage_improvement": 0.0, "contradiction_change": 0.0}
        }
    }
    
    with open("conf.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("配置文件已生成: conf.json")
    return config


class AgentState:
    """
    Agent状态类，用于跟踪和管理系统状态
    """
    def __init__(self, initial_coverage=0.57, initial_contradiction=4.7, 
                 initial_uncertainty=0.25, budget_ms=1500):
        self.coverage = initial_coverage  # 覆盖度
        self.contradiction = initial_contradiction  # 矛盾率(%)
        self.uncertainty = initial_uncertainty  # 不确定度
        self.budget_ms = budget_ms  # 预算(毫秒)
        self.used_budget = 0  # 已使用预算
        self.history = []  # 历史动作记录
        self.coverage_history = [initial_coverage]  # 覆盖度历史
        self.contradiction_history = [initial_contradiction]  # 矛盾率历史
        self.budget_history = [budget_ms]  # 预算历史
        self.steps = 0  # 执行步数
    
    def update(self, action, coverage_change, contradiction_change, cost_ms):
        """
        更新状态
        
        Args:
            action: 执行的动作
            coverage_change: 覆盖度变化
            contradiction_change: 矛盾率变化
            cost_ms: 消耗的时间
        """
        self.coverage = min(1.0, max(0.0, self.coverage + coverage_change))
        self.contradiction = max(0.0, self.contradiction + contradiction_change)
        self.used_budget += cost_ms
        self.budget_ms = max(0, self.budget_ms - cost_ms)
        self.steps += 1
        
        # 记录历史
        self.history.append({
            "step": self.steps,
            "action": action,
            "coverage": self.coverage,
            "coverage_change": coverage_change,
            "contradiction": self.contradiction,
            "contradiction_change": contradiction_change,
            "cost_ms": cost_ms,
            "remaining_budget": self.budget_ms
        })
        
        self.coverage_history.append(self.coverage)
        self.contradiction_history.append(self.contradiction)
        self.budget_history.append(self.budget_ms)
    
    def get_state(self):
        """
        获取当前状态
        
        Returns:
            dict: 状态字典
        """
        return {
            "coverage": self.coverage,
            "contradiction": self.contradiction,
            "uncertainty": self.uncertainty,
            "remaining_budget": self.budget_ms,
            "used_budget": self.used_budget,
            "steps": self.steps
        }
    
    def should_terminate(self, max_steps, min_improvement=0.01):
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
            return True, "steps_exceeded"
        
        # 预算耗尽
        if self.budget_ms <= 0:
            return True, "budget_exhausted"
        
        # 覆盖度已很高
        if self.coverage >= 0.95:
            return True, "high_coverage"
        
        # 收益递减
        if len(self.coverage_history) >= 3:
            recent_improvements = [
                self.coverage_history[-1] - self.coverage_history[-2],
                self.coverage_history[-2] - self.coverage_history[-3]
            ]
            if all(imp < min_improvement for imp in recent_improvements):
                return True, "diminishing_returns"
        
        return False, "continue"


class ActionSelector:
    """
    动作选择器，基于状态和阈值选择合适的动作
    """
    def __init__(self, thresholds, action_costs, action_effects):
        self.thresholds = thresholds
        self.action_costs = action_costs
        self.action_effects = action_effects
    
    def select_action(self, state):
        """
        根据当前状态选择最合适的动作
        
        Args:
            state: AgentState对象
        
        Returns:
            str: 选择的动作名称
        """
        coverage = state.coverage
        contradiction = state.contradiction
        remaining_budget = state.budget_ms
        steps = state.steps
        
        # 首先检查必要的紧急动作
        if contradiction >= self.thresholds["contradiction"]["critical"]:
            # 矛盾率严重，需要跨模态验证
            if remaining_budget >= self.action_costs["cross_verify"]["time_ms"]:
                return "cross_verify"
            else:
                # 预算不足，选择次优动作
                return "ocr_verify"
        
        if contradiction >= self.thresholds["contradiction"]["warning"]:
            # 矛盾率警告，需要针对性验证
            if remaining_budget >= self.action_costs["ocr_verify"]["time_ms"]:
                return "ocr_verify"
            else:
                return "self_check"
        
        # 根据覆盖度选择动作
        if coverage < self.thresholds["coverage"]["low"]:
            # 低覆盖度
            if remaining_budget >= self.action_costs["graph_lookup"]["time_ms"]:
                # 优先使用图谱检索
                return "graph_lookup"
            elif remaining_budget >= self.action_costs["heavy_expand_k"]["time_ms"]:
                return "heavy_expand_k"
            else:
                return "light_expand_k"
        elif coverage < self.thresholds["coverage"]["medium"]:
            # 中等覆盖度
            if steps % 2 == 0 and remaining_budget >= self.action_costs["graph_lookup"]["time_ms"]:
                # 每隔一步使用图谱检索
                return "graph_lookup"
            else:
                return "light_expand_k"
        else:
            # 较高覆盖度
            if remaining_budget >= self.action_costs["self_check"]["time_ms"]:
                return "self_check"
            else:
                return "fallback"
    
    def get_action_info(self, action):
        """
        获取动作的详细信息
        
        Args:
            action: 动作名称
        
        Returns:
            dict: 动作信息
        """
        return {
            "cost_ms": self.action_costs[action]["time_ms"],
            "coverage_improvement": self.action_effects[action]["coverage_improvement"],
            "contradiction_change": self.action_effects[action]["contradiction_change"],
            "description": self.action_costs[action]["description"]
        }


def format_step_output(step, action, state_before, state_after, cost_ms):
    """
    格式化步骤输出
    
    Args:
        step: 步骤号
        action: 执行的动作
        state_before: 执行前的状态
        state_after: 执行后的状态
        cost_ms: 消耗的时间
    
    Returns:
        str: 格式化的输出字符串
    """
    coverage_change = state_after.coverage - state_before.coverage
    contradiction_change = state_after.contradiction - state_before.contradiction
    
    output = f"Step{step}: action={action} → "
    
    # 主要显示覆盖度变化
    output += f"coverage {state_after.coverage:.2f} ({'+' if coverage_change >= 0 else ''}{coverage_change:.2f}), "
    
    # 如果矛盾率有显著变化，也显示
    if abs(contradiction_change) > 0.1:
        output += f"contradiction {state_after.contradiction:.1f}% ({'+' if contradiction_change >= 0 else ''}{contradiction_change:.1f}%), "
    
    # 显示时间消耗
    output += f"P95 +{cost_ms}ms"
    
    return output


def format_final_output(state, total_cost, slo_ms=200):
    """
    格式化最终输出
    
    Args:
        state: 最终状态
        total_cost: 总消耗时间
        slo_ms: SLO阈值
    
    Returns:
        str: 格式化的输出字符串
    """
    within_slo = total_cost <= slo_ms
    slo_status = "within SLO" if within_slo else "exceeds SLO"
    
    return f"Final: coverage={state.coverage:.2f}, contradiction={state.contradiction:.1f}%, " \
           f"P95_delta=+{total_cost}ms ({slo_status})"


def export_trajectory(state, output_file="policy_trajectory.json"):
    """
    导出策略轨迹到JSON文件
    
    Args:
        state: AgentState对象
        output_file: 输出文件名
    """
    trajectory = {
        "initial_state": {
            "coverage": state.coverage_history[0],
            "contradiction": state.contradiction_history[0],
            "budget": state.budget_history[0]
        },
        "final_state": {
            "coverage": state.coverage,
            "contradiction": state.contradiction,
            "remaining_budget": state.budget_ms,
            "used_budget": state.used_budget,
            "steps": state.steps
        },
        "history": state.history,
        "metrics": {
            "coverage_improvement": state.coverage - state.coverage_history[0],
            "contradiction_reduction": state.contradiction_history[0] - state.contradiction,
            "efficiency": (state.coverage - state.coverage_history[0]) / state.used_budget if state.used_budget > 0 else 0
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
    
    print(f"策略轨迹已导出到: {output_file}")
    return trajectory


def generate_summary_report(trajectory, output_file="agent_policy_summary.txt"):
    """
    生成策略执行摘要报告
    
    Args:
        trajectory: 轨迹数据
        output_file: 输出文件名
    """
    with open(output_file, "w", encoding="utf-8-sig") as f:
        f.write("===== Agent策略调度执行摘要 =====\n\n")
        
        # 初始状态
        f.write("初始状态:\n")
        f.write(f"  覆盖度: {trajectory['initial_state']['coverage']:.2f}\n")
        f.write(f"  矛盾率: {trajectory['initial_state']['contradiction']:.1f}%\n")
        f.write(f"  预算: {trajectory['initial_state']['budget']}ms\n\n")
        
        # 最终状态
        f.write("最终状态:\n")
        f.write(f"  覆盖度: {trajectory['final_state']['coverage']:.2f} (+{trajectory['metrics']['coverage_improvement']:.2f})\n")
        f.write(f"  矛盾率: {trajectory['final_state']['contradiction']:.1f}% (-{trajectory['metrics']['contradiction_reduction']:.1f}%)\n")
        f.write(f"  剩余预算: {trajectory['final_state']['remaining_budget']}ms\n")
        f.write(f"  总消耗: {trajectory['final_state']['used_budget']}ms\n")
        f.write(f"  执行步数: {trajectory['final_state']['steps']}\n\n")
        
        # 动作统计
        f.write("动作统计:\n")
        action_counts = defaultdict(int)
        for step in trajectory['history']:
            action_counts[step['action']] += 1
        
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {action}: {count}次\n")
        f.write("\n")
        
        # 执行效率
        f.write("执行效率分析:\n")
        efficiency = trajectory['metrics']['efficiency'] * 1000  # 转换为每1000ms的改进
        f.write(f"  覆盖度提升效率: {efficiency:.4f} per 1000ms\n")
        
        # 策略建议
        f.write("\n策略建议:\n")
        if trajectory['final_state']['coverage'] < 0.8:
            f.write("  - 建议增加预算或调整动作选择策略以提高覆盖度\n")
        if trajectory['final_state']['contradiction'] > 3.0:
            f.write("  - 矛盾率仍然较高，建议增强验证动作\n")
        if trajectory['final_state']['used_budget'] > trajectory['initial_state']['budget'] * 0.9:
            f.write("  - 预算几乎耗尽，建议优化动作选择\n")
        else:
            f.write("  - 当前策略执行良好，覆盖度和矛盾率控制在合理范围内\n")
    
    print(f"策略执行摘要已导出到: {output_file}")


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Agent策略调度（信号-动作-预算-回退）")
    parser.add_argument("--thresholds", default="conf.json", help="阈值配置文件路径")
    parser.add_argument("--budget", default="1500ms", help="预算限制")
    parser.add_argument("--max-steps", type=int, default=5, help="最大执行步数")
    args = parser.parse_args()
    
    # 检查依赖
    install_dependencies()
    
    # 提取预算数值
    budget_str = args.budget.replace("ms", "").strip()
    try:
        budget_ms = int(budget_str)
    except ValueError:
        print("预算格式错误，使用默认值1500ms")
        budget_ms = 1500
    
    # 检查配置文件是否存在，不存在则生成
    if not os.path.exists(args.thresholds):
        print("配置文件不存在，正在生成默认配置...")
        config = generate_config_file()
    else:
        # 加载配置文件
        try:
            with open(args.thresholds, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"成功加载配置文件: {args.thresholds}")
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            print("使用默认配置...")
            config = generate_config_file()
    
    # 初始化状态
    state = AgentState(
        initial_coverage=0.57,
        initial_contradiction=4.7,
        initial_uncertainty=0.25,
        budget_ms=budget_ms
    )
    
    # 初始化动作选择器
    selector = ActionSelector(
        thresholds=config["thresholds"],
        action_costs=config["action_costs"],
        action_effects=config["action_effects"]
    )
    
    print(f"\n===== 开始Agent策略调度 =====")
    print(f"初始状态: 覆盖度={state.coverage:.2f}, 矛盾率={state.contradiction:.1f}%, 预算={state.budget_ms}ms")
    print(f"最大步数: {args.max_steps}")
    print()
    
    # 主循环
    start_time = time.time()
    step_outputs = []
    
    while True:
        # 检查终止条件
        should_terminate, reason = state.should_terminate(args.max_steps)
        if should_terminate:
            print(f"\n终止执行: {reason}")
            break
        
        # 选择动作
        action = selector.select_action(state)
        
        # 获取动作信息
        action_info = selector.get_action_info(action)
        
        # 保存执行前状态
        state_before = AgentState(
            initial_coverage=state.coverage,
            initial_contradiction=state.contradiction,
            initial_uncertainty=state.uncertainty,
            budget_ms=state.budget_ms
        )
        
        # 执行动作并更新状态
        # 添加一些随机性以模拟真实情况
        coverage_change = action_info["coverage_improvement"] * (0.9 + 0.2 * random.random())
        contradiction_change = action_info["contradiction_change"] * (0.9 + 0.2 * random.random())
        actual_cost = int(action_info["cost_ms"] * (0.95 + 0.1 * random.random()))
        
        state.update(action, coverage_change, contradiction_change, actual_cost)
        
        # 格式化输出
        step_output = format_step_output(state.steps, action, state_before, state, actual_cost)
        print(step_output)
        step_outputs.append(step_output)
    
    execution_time = time.time() - start_time
    
    # 输出最终结果
    final_output = format_final_output(state, state.used_budget)
    print("\n" + final_output)
    
    # 导出轨迹
    trajectory = export_trajectory(state)
    
    # 生成摘要报告
    generate_summary_report(trajectory)
    
    print(f"\n===== 执行完成 =====")
    print(f"程序执行时间: {execution_time:.3f}秒")
    
    # 验证中文显示
    print("\n验证中文显示：")
    sample_cn_text = "策略调度 覆盖度 矛盾率 预算控制 回退机制"
    print(f"样本中文文本: {sample_cn_text}")


if __name__ == "__main__":
    main()