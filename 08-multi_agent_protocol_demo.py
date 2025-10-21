#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三Agent协作时序与去重控制

功能说明：实现检索/校验/整合三Agent接口契约、步数上限、候选去重与时序日志。
内容概述：定义统一证据对象与锚点四元；检索Agent合并去重；校验Agent检测缺口与矛盾触发补证；整合Agent绑定引用生成回答；输出时序轨迹与成本摘要。
作者：Ph.D. Rhino
"""

import os
import sys
import json
import csv
import time
import random
import argparse
from typing import List, Dict, Any, Set, Tuple
import numpy as np

# 依赖检查与自动安装
def check_dependencies():
    """检查必要的依赖项并自动安装"""
    required_packages = ['numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("依赖包安装完成")

# 证据对象与锚点定义
class EvidenceObject:
    """统一证据对象定义"""
    def __init__(self, evidence_id: str, content: str, source: str, confidence: float, timestamp: float):
        self.evidence_id = evidence_id  # 证据唯一标识
        self.content = content          # 证据内容
        self.source = source            # 证据来源
        self.confidence = confidence    # 置信度
        self.timestamp = timestamp      # 时间戳
        self.references = set()         # 引用集合
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'evidence_id': self.evidence_id,
            'content': self.content,
            'source': self.source,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'references': list(self.references)
        }

class AnchorQuaternion:
    """锚点四元组定义"""
    def __init__(self, start_pos: int, end_pos: int, evidence_id: str, relation_type: str):
        self.start_pos = start_pos      # 起始位置
        self.end_pos = end_pos          # 结束位置
        self.evidence_id = evidence_id  # 关联证据ID
        self.relation_type = relation_type  # 关系类型
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'evidence_id': self.evidence_id,
            'relation_type': self.relation_type
        }

# Agent基类与接口契约
class BaseAgent:
    """Agent基类，定义统一接口契约"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cost = 0.0  # 成本累计
        self.request_count = 0  # 请求计数
        self.tokens_processed = 0  # 处理的token数
    
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """处理输入并返回输出和上下文更新"""
        raise NotImplementedError("子类必须实现process方法")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取Agent性能指标"""
        return {
            'agent_id': self.agent_id,
            'cost': self.cost,
            'request_count': self.request_count,
            'tokens_processed': self.tokens_processed
        }

class RetrieveAgent(BaseAgent):
    """检索Agent，负责召回与去重"""
    def __init__(self):
        super().__init__("retrieve")
        self.dedup_enabled = True  # 默认启用去重
        # 显式初始化属性
        self.cost = 0.0
        self.request_count = 0
        self.tokens_processed = 0
    
    def set_dedup_enabled(self, enabled: bool):
        """设置是否启用去重"""
        self.dedup_enabled = enabled
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（模拟实现）"""
        # 实际项目中可使用更复杂的相似度算法
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def _deduplicate_evidence(self, evidences: List[EvidenceObject], threshold: float = 0.7) -> Tuple[List[EvidenceObject], int]:
        """对证据列表进行去重"""
        if not self.dedup_enabled:
            return evidences, 0
        
        unique_evidences = []
        removed_count = 0
        
        for new_evidence in evidences:
            is_duplicate = False
            for existing_evidence in unique_evidences:
                similarity = self._calculate_similarity(new_evidence.content, existing_evidence.content)
                if similarity > threshold:
                    is_duplicate = True
                    removed_count += 1
                    break
            if not is_duplicate:
                unique_evidences.append(new_evidence)
        
        return unique_evidences, removed_count
    
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """处理检索请求"""
        query = input_data.get('query', '')
        
        # 模拟检索过程
        self.request_count += 1
        tokens = len(query) * 1.5  # 估算token数
        self.tokens_processed += tokens
        cost = random.uniform(0.5, 1.2)  # 模拟成本
        self.cost += cost
        
        print(f"  [检索Agent] 查询: {query}, 请求: {self.request_count}, Token: {tokens:.1f}, 成本: {cost:.2f}")
        
        # 生成模拟证据
        evidences = []
        for i in range(25):  # 生成25条证据，包含一些重复的
            evidence_id = f"evidence_{i}"
            # 模拟一些重复内容
            if i % 5 == 0:
                content = f"关于{query}的重要信息，这是一条重复内容示例"
            else:
                content = f"{query}相关信息 #{i}: 包含{query}的详细说明和背景知识"
            
            evidence = EvidenceObject(
                evidence_id=evidence_id,
                content=content,
                source=f"source_{i % 10}",
                confidence=random.uniform(0.6, 0.95),
                timestamp=time.time() - i
            )
            evidences.append(evidence)
        
        # 去重处理
        unique_evidences, removed_count = self._deduplicate_evidence(evidences)
        
        # 更新上下文
        context['evidences'] = [e.to_dict() for e in unique_evidences]
        context['retrieve_metrics'] = {
            'total_evidences': len(evidences),
            'unique_evidences': len(unique_evidences),
            'removed_duplicates': removed_count,
            'prune_rate': (removed_count / len(evidences)) * 100 if evidences else 0
        }
        
        # 返回结果
        output = {
            'status': 'success',
            'unique_count': len(unique_evidences),
            'prune_rate': context['retrieve_metrics']['prune_rate']
        }
        
        return output, context

class ValidateAgent(BaseAgent):
    """校验Agent，负责检测缺口与矛盾并触发补证"""
    def __init__(self):
        super().__init__("validate")
        # 显式初始化属性
        self.cost = 0.0
        self.request_count = 0
        self.tokens_processed = 0
    
    def _detect_gaps_and_contradictions(self, evidences: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """检测证据中的缺口和矛盾"""
        gaps = []
        contradictions = []
        
        # 模拟缺口检测
        if len(evidences) < 10:
            gaps.append("证据数量不足，需要补充专业领域知识")
        
        # 模拟矛盾检测（通过置信度差异）
        high_confidence = [e for e in evidences if e['confidence'] > 0.85]
        low_confidence = [e for e in evidences if e['confidence'] < 0.65]
        
        # 如果存在高置信度和低置信度的证据，模拟存在矛盾
        if high_confidence and low_confidence:
            contradictions.append((high_confidence[0]['evidence_id'], low_confidence[0]['evidence_id']))
        
        return gaps, contradictions
    
    def _trigger_supplement_verification(self, gaps: List[str], contradictions: List[Tuple[str, str]], context: Dict[str, Any]) -> str:
        """触发补证策略"""
        # 根据缺口和矛盾情况决定补证策略
        if contradictions:
            # 存在矛盾时优先使用图谱补证
            return "graph"
        elif gaps:
            # 存在缺口时使用跨模态补证
            return "targeted_ocr"
        return None
    
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """处理校验请求"""
        evidences = context.get('evidences', [])
        
        # 模拟校验过程
        self.request_count += 1
        tokens = sum(len(e['content']) for e in evidences) * 0.8
        self.tokens_processed += tokens
        cost = random.uniform(0.8, 1.5)
        self.cost += cost
        
        print(f"  [校验Agent] 证据数: {len(evidences)}, Token: {tokens:.1f}, 成本: {cost:.2f}")
        
        # 检测缺口和矛盾
        gaps, contradictions = self._detect_gaps_and_contradictions(evidences)
        
        # 触发补证策略
        supplement_action = self._trigger_supplement_verification(gaps, contradictions, context)
        
        # 更新上下文
        context['validation_result'] = {
            'gaps': gaps,
            'contradictions': contradictions,
            'supplement_action': supplement_action
        }
        
        # 返回结果
        output = {
            'status': 'success',
            'gaps_detected': len(gaps),
            'contradictions_detected': len(contradictions),
            'supplement_action': supplement_action
        }
        
        return output, context

class SupplementAgent(BaseAgent):
    """补证Agent，负责执行具体的补证操作"""
    def __init__(self):
        super().__init__("supplement")
        # 显式初始化属性
        self.cost = 0.0
        self.request_count = 0
        self.tokens_processed = 0
    
    def _perform_graph_verification(self, contradictions: List[Tuple[str, str]], context: Dict[str, Any]) -> List[EvidenceObject]:
        """执行图谱补证"""
        new_evidences = []
        
        # 模拟图谱补证过程
        for idx, (high_id, low_id) in enumerate(contradictions):
            evidence = EvidenceObject(
                evidence_id=f"graph_evidence_{idx}",
                content=f"图谱验证结果：解决{high_id}与{low_id}之间的矛盾，提供可靠关联",
                source="graph_database",
                confidence=0.92,
                timestamp=time.time()
            )
            new_evidences.append(evidence)
        
        return new_evidences
    
    def _perform_ocr_verification(self, gaps: List[str], context: Dict[str, Any]) -> List[EvidenceObject]:
        """执行OCR补证"""
        new_evidences = []
        
        # 模拟OCR补证过程
        for idx, gap in enumerate(gaps):
            evidence = EvidenceObject(
                evidence_id=f"ocr_evidence_{idx}",
                content=f"OCR识别结果：补充{gap}缺失的信息，提供视觉佐证",
                source="document_scanner",
                confidence=0.87,
                timestamp=time.time()
            )
            new_evidences.append(evidence)
        
        return new_evidences
    
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """处理补证请求"""
        action_type = input_data.get('action_type')
        validation_result = context.get('validation_result', {})
        
        # 模拟补证过程
        self.request_count += 1
        cost = random.uniform(1.0, 2.0)
        self.cost += cost
        
        new_evidences = []
        tokens = 0
        if action_type == "graph":
            new_evidences = self._perform_graph_verification(validation_result.get('contradictions', []), context)
            tokens = 500  # 估算图谱处理token数
            self.tokens_processed += tokens
            print(f"  [补证Agent] 动作: 图谱推理, 新增证据: {len(new_evidences)}, Token: {tokens}, 成本: {cost:.2f}")
        elif action_type == "targeted_ocr":
            new_evidences = self._perform_ocr_verification(validation_result.get('gaps', []), context)
            tokens = 800  # 估算OCR处理token数
            self.tokens_processed += tokens
            print(f"  [补证Agent] 动作: 定向OCR验证, 新增证据: {len(new_evidences)}, Token: {tokens}, 成本: {cost:.2f}")
        
        # 更新证据列表
        existing_evidences = []
        for e_dict in context.get('evidences', []):
            # 创建一个新的字典，排除references字段
            e_data = e_dict.copy()
            if 'references' in e_data:
                del e_data['references']
            # 创建EvidenceObject实例
            evidence = EvidenceObject(**e_data)
            # 如果有references字段，手动设置
            if 'references' in e_dict:
                evidence.references = set(e_dict['references'])
            existing_evidences.append(evidence)
        
        existing_evidences.extend(new_evidences)
        
        # 转换回字典格式
        context['evidences'] = [e.to_dict() for e in existing_evidences]
        context['supplement_result'] = {
            'action_type': action_type,
            'new_evidence_count': len(new_evidences)
        }
        
        # 返回结果
        output = {
            'status': 'success',
            'action_type': action_type,
            'new_evidence_count': len(new_evidences)
        }
        
        return output, context

class IntegrateAgent(BaseAgent):
    """整合Agent，负责绑定引用锚点并生成答案"""
    def __init__(self):
        super().__init__("integrate")
        # 显式初始化属性
        self.cost = 0.0
        self.request_count = 0
        self.tokens_processed = 0
    
    def _generate_answer(self, query: str, evidences: List[Dict[str, Any]]) -> Tuple[str, List[AnchorQuaternion]]:
        """生成答案并创建锚点引用"""
        # 筛选高置信度证据
        high_conf_evidences = sorted(
            [e for e in evidences if e['confidence'] > 0.8],
            key=lambda x: x['confidence'],
            reverse=True
        )[:5]  # 选取前5个高置信度证据
        
        # 生成模拟答案
        answer_parts = [f"根据检索和验证结果，关于'{query}'的回答如下：\n\n"]
        anchor_quaternions = []
        current_pos = len(answer_parts[0])
        
        # 整合证据内容
        for idx, evidence in enumerate(high_conf_evidences):
            # 模拟整合证据内容
            part = f"{idx + 1}. {evidence['content'][:100]}... [引用: {evidence['evidence_id']}]\n"
            answer_parts.append(part)
            
            # 创建锚点
            start_pos = current_pos
            end_pos = current_pos + len(part)
            anchor = AnchorQuaternion(
                start_pos=start_pos,
                end_pos=end_pos,
                evidence_id=evidence['evidence_id'],
                relation_type="citation"
            )
            anchor_quaternions.append(anchor)
            
            current_pos = end_pos
        
        # 添加结论
        conclusion = "\n综上所述，我们通过多源证据验证，提供了关于该问题的全面解答。"
        answer_parts.append(conclusion)
        
        return "".join(answer_parts), anchor_quaternions
    
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """处理整合请求"""
        query = input_data.get('query', '')
        evidences = context.get('evidences', [])
        
        # 模拟整合过程
        self.request_count += 1
        tokens = 1000  # 估算token数
        self.tokens_processed += tokens
        cost = random.uniform(1.2, 2.5)
        self.cost += cost
        
        print(f"  [整合Agent] 证据数: {len(evidences)}, Token: {tokens}, 成本: {cost:.2f}")
        
        # 生成答案和锚点
        answer, anchors = self._generate_answer(query, evidences)
        
        # 更新上下文
        context['answer'] = answer
        context['anchors'] = [a.to_dict() for a in anchors]
        context['integration_result'] = {
            'answer_length': len(answer),
            'anchor_count': len(anchors)
        }
        
        # 返回结果
        output = {
            'status': 'success',
            'answer_length': len(answer),
            'anchor_count': len(anchors)
        }
        
        return output, context

# 多Agent协议管理器
class MultiAgentProtocol:
    """多Agent协作协议管理器"""
    def __init__(self, max_steps: int, dedup_enabled: bool = True):
        self.max_steps = max_steps
        self.dedup_enabled = dedup_enabled
        self.sequence_log = []
        self.step_count = 0
        
        # 初始化各Agent
        self.retrieve_agent = RetrieveAgent()
        self.retrieve_agent.set_dedup_enabled(dedup_enabled)
        self.validate_agent = ValidateAgent()
        self.supplement_agent = SupplementAgent()
        self.integrate_agent = IntegrateAgent()
        
        self.agents = [
            self.retrieve_agent,
            self.validate_agent,
            self.supplement_agent,
            self.integrate_agent
        ]
    
    def _log_step(self, step_name: str, output: Dict[str, Any], cost: float):
        """记录步骤日志"""
        log_entry = {
            'step': self.step_count,
            'step_name': step_name,
            'timestamp': time.time(),
            'output': output,
            'cost': cost
        }
        self.sequence_log.append(log_entry)
        self.step_count += 1
    
    def _check_step_limit(self) -> bool:
        """检查是否超过最大步数限制"""
        return self.step_count >= self.max_steps
    
    def run_protocol(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行多Agent协议流程"""
        context = {}
        final_result = {
            'status': 'success',
            'sequence': [],
            'metrics': {
                'total_steps': 0,
                'total_cost': 0,
                'total_requests': 0,
                'total_tokens': 0,
                'prune_rate': 0,
                'agents': {}
            }
        }
        
        try:
            # 1. 执行检索Agent
            if not self._check_step_limit():
                retrieve_output, context = self.retrieve_agent.process(input_data, context)
                self._log_step('retrieve', retrieve_output, self.retrieve_agent.cost)
                final_result['sequence'].append('retrieve')
            
            # 2. 执行校验Agent
            validate_output = None
            if not self._check_step_limit():
                validate_output, context = self.validate_agent.process({}, context)
                self._log_step('validate', validate_output, self.validate_agent.cost)
                final_result['sequence'].append('validate')
            
            # 3. 检查是否需要补证（注意这里不放在校验Agent的if块中，确保即使校验失败也能正确处理）
            supplement_action = validate_output.get('supplement_action') if validate_output else None
            if supplement_action and not self._check_step_limit():
                supplement_input = {'action_type': supplement_action}
                supplement_output, context = self.supplement_agent.process(supplement_input, context)
                self._log_step(supplement_action, supplement_output, self.supplement_agent.cost)
                final_result['sequence'].append(supplement_action)
            
            # 4. 执行整合Agent
            if not self._check_step_limit():
                integrate_output, context = self.integrate_agent.process(input_data, context)
                self._log_step('integrate', integrate_output, self.integrate_agent.cost)
                final_result['sequence'].append('integrate')
            
            # 收集所有指标
            total_cost = sum(agent.cost for agent in self.agents)
            total_requests = sum(agent.request_count for agent in self.agents)
            total_tokens = sum(agent.tokens_processed for agent in self.agents)
            
            # 获取去重率
            prune_rate = context.get('retrieve_metrics', {}).get('prune_rate', 0)
            
            # 确保即使在没有证据的情况下也能得到正确的结果
            answer = context.get('answer', '')
            evidence_count = len(context.get('evidences', []))
            
            # 更新指标
            final_result['metrics']['total_steps'] = self.step_count
            final_result['metrics']['total_cost'] = total_cost
            final_result['metrics']['total_requests'] = total_requests
            final_result['metrics']['total_tokens'] = total_tokens
            final_result['metrics']['prune_rate'] = prune_rate
            final_result['metrics']['agents'] = {agent.agent_id: agent.get_metrics() for agent in self.agents}
            
            # 添加最终答案和证据
            final_result['answer'] = answer
            final_result['evidence_count'] = evidence_count
            
        except Exception as e:
            final_result['status'] = 'error'
            final_result['error'] = str(e)
            print(f"错误: {str(e)}")
        
        return final_result
    
    def export_sequence_log(self, filename: str = 'sequence_log.csv'):
        """导出序列日志到CSV文件"""
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['step', 'step_name', 'timestamp', 'status', 'cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for log_entry in self.sequence_log:
                row = {
                    'step': log_entry['step'],
                    'step_name': log_entry['step_name'],
                    'timestamp': log_entry['timestamp'],
                    'status': log_entry['output'].get('status', 'unknown'),
                    'cost': log_entry['cost']
                }
                writer.writerow(row)
    
    def export_metrics_summary(self, filename: str = 'protocol_metrics.json'):
        """导出指标摘要到JSON文件"""
        metrics_summary = {
            'sequence_log': self.sequence_log,
            'agents': {agent.agent_id: agent.get_metrics() for agent in self.agents}
        }
        
        with open(filename, 'w', encoding='utf-8-sig') as jsonfile:
            json.dump(metrics_summary, jsonfile, ensure_ascii=False, indent=2)

# 工具函数
class Toolkit:
    """工具函数集合"""
    
    @staticmethod
    def generate_demo_case(filename: str = 'case.json') -> Dict[str, Any]:
        """生成演示用例数据"""
        case_data = {
            "query": "药物相互作用对心脏病患者的影响",
            "context": {
                "patient_info": {
                    "age": 65,
                    "condition": "冠心病",
                    "medications": ["阿司匹林", "他汀类药物", "β受体阻滞剂"]
                },
                "question_type": "药物安全性",
                "priority": "high"
            }
        }
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8-sig') as jsonfile:
            json.dump(case_data, jsonfile, ensure_ascii=False, indent=2)
        
        return case_data
    
    @staticmethod
    def load_case(filename: str) -> Dict[str, Any]:
        """加载用例数据"""
        try:
            # 使用utf-8-sig编码来处理可能包含BOM的文件
            with open(filename, 'r', encoding='utf-8-sig') as jsonfile:
                return json.load(jsonfile)
        except FileNotFoundError:
            print(f"警告：未找到用例文件 {filename}，生成默认用例")
            return Toolkit.generate_demo_case(filename)
        except json.JSONDecodeError:
            print(f"警告：用例文件 {filename} 格式错误，生成默认用例")
            return Toolkit.generate_demo_case(filename)

# 主函数
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='三Agent协作时序与去重控制')
    parser.add_argument('--flow', type=str, default='case.json', help='流程配置文件路径')
    parser.add_argument('--max-steps', type=int, default=5, help='最大执行步数')
    parser.add_argument('--dedup', type=str, default='on', choices=['on', 'off'], help='是否启用去重')
    args = parser.parse_args()
    
    # 检查依赖
    check_dependencies()
    
    # 加载或生成用例
    case_data = Toolkit.load_case(args.flow)
    
    # 初始化多Agent协议
    dedup_enabled = args.dedup == 'on'
    protocol = MultiAgentProtocol(max_steps=args.max_steps, dedup_enabled=dedup_enabled)
    
    # 运行协议
    print(f"开始执行多Agent协作流程，最大步数: {args.max_steps}, 去重: {'启用' if dedup_enabled else '禁用'}")
    start_time = time.time()
    result = protocol.run_protocol(case_data)
    end_time = time.time()
    
    # 导出日志和指标
    protocol.export_sequence_log()
    protocol.export_metrics_summary()
    
    # 输出执行结果摘要
    print(f"\n执行完成！耗时: {end_time - start_time:.3f} 秒")
    print(f"Sequence: {' -> '.join(result['sequence'])}")
    
    metrics = result['metrics']
    prune_rate = metrics.get('prune_rate', 0)
    total_tokens = metrics.get('total_tokens', 0)
    total_requests = metrics.get('total_requests', 0)
    print(f"Duplicates pruned={prune_rate:.1f}%, tokens={total_tokens/1000:.1f}k, req={total_requests}")
    
    # 检查输出文件
    print(f"\n输出文件:")
    print(f"- sequence_log.csv: 时序日志")
    print(f"- protocol_metrics.json: 详细指标")
    
    # 验证中文显示
    print(f"\n验证中文显示:")
    print(f"答案长度: {len(result.get('answer', ''))} 字符")
    print(f"证据数量: {result.get('evidence_count', 0)}")

if __name__ == "__main__":
    main()