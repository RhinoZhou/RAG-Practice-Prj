#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多场景最小验证集生成器

功能：生成用于RAG系统评估的多场景最小验证集，包含法律、科研和客服对话三个典型场景

特点：
- 每个场景包含3-5段精心设计的典型文本
- 提供详细的标注信息，包括关键短语和答案范围
- 支持JSON和CSV两种输出格式
- 内置高质量的样本文本，可直接用于演示和评估

使用方法：
1. 直接运行脚本：python 03-min_evalset_builder_multiscenario.py
2. 结果将保存到results目录下

输出文件：
- multiscenario_evalset.json: JSON格式的验证集数据
- multiscenario_evalset.csv: CSV格式的验证集数据

依赖说明：
- 仅使用Python标准库，无需安装额外依赖

作者：自动生成
日期：2023-06
"""
import json
import csv
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ScenarioType(Enum):
    LEGAL = "legal"
    SCIENTIFIC = "scientific"
    CUSTOMER_SERVICE = "customer_service"

@dataclass
class SampleText:
    """表示一个样本文本的数据类"""
    text: str
    label: str
    questions: List[str]
    key_phrases: Optional[List[str]] = None
    answer_ranges: Optional[List[Tuple[int, int]]] = None

@dataclass
class ScenarioSample:
    """表示一个场景样本包的数据类"""
    scenario_type: str
    name: str
    description: str
    samples: List[SampleText]

class EvalSetBuilder:
    """用于构建多场景验证集的工具类
    
    主要功能：
    - 管理多个场景的样本文本库
    - 提供灵活的验证集构建方法
    - 支持多种格式的结果输出
    
    数据结构：
    - self.scenarios: 存储不同场景的样本集合
    
    典型用法：
    ```python
    builder = EvalSetBuilder()
    # 构建所有场景的验证集
    evalset_data = builder.build_evalset()
    # 或构建特定场景的验证集
    # evalset_data = builder.build_evalset(['legal', 'scientific'])
    ```
    """
    
    def __init__(self):
        """初始化验证集构建器，加载内置的示例素材库"""
        self.scenarios = {}
        self._load_example_library()
    
    def _load_example_library(self):
        """加载内置的示例素材库"""
        # 法律场景样本
        legal_samples = [
            SampleText(
                text="根据《中华人民共和国民法典》第一千零四十二条规定，禁止包办、买卖婚姻和其他干涉婚姻自由的行为。禁止借婚姻索取财物。禁止重婚。禁止有配偶者与他人同居。禁止家庭暴力。禁止家庭成员间的虐待和遗弃。",
                label="婚姻家庭法律条款",
                questions=[
                    "民法典中关于婚姻自由有哪些禁止性规定？",
                    "民法典如何规定家庭暴力问题？"
                ],
                key_phrases=["民法典", "婚姻自由", "家庭暴力", "禁止性规定"],
                answer_ranges=[(10, 32), (74, 82)]
            ),
            SampleText(
                text="合同纠纷案件中，当事人应当按照约定全面履行自己的义务。当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。当事人在履行合同过程中，应当避免浪费资源、污染环境和破坏生态。",
                label="合同履行原则",
                questions=[
                    "合同履行的基本原则是什么？",
                    "当事人在合同履行过程中应承担哪些附随义务？"
                ],
                key_phrases=["合同履行", "诚信原则", "附随义务", "全面履行"],
                answer_ranges=[(0, 22), (23, 61)]
            ),
            SampleText(
                text="知识产权包括著作权、专利权、商标权、商业秘密、集成电路布图设计等。侵犯知识产权的，应当依法承担停止侵害、消除影响、赔礼道歉、赔偿损失等民事责任。情节严重的，可能构成犯罪。",
                label="知识产权保护",
                questions=[
                    "知识产权包含哪些类型？",
                    "侵犯知识产权需要承担哪些法律责任？"
                ],
                key_phrases=["知识产权", "著作权", "专利权", "侵权责任"],
                answer_ranges=[(0, 28), (29, 67)]
            )
        ]
        
        # 科研场景样本
        scientific_samples = [
            SampleText(
                text="量子计算是一种遵循量子力学规律调控量子信息单元进行计算的新型计算模式。与传统计算机使用二进制位（比特）不同，量子计算机使用量子位（ qubits），可以处于叠加态，从而实现并行计算。量子计算在密码破解、材料设计、药物研发等领域有潜在的革命性应用。",
                label="量子计算基础",
                questions=[
                    "量子计算机与传统计算机的主要区别是什么？",
                    "量子计算有哪些潜在的应用领域？"
                ],
                key_phrases=["量子计算", "量子位", "叠加态", "并行计算"],
                answer_ranges=[(24, 62), (63, 94)]
            ),
            SampleText(
                text="人工智能大语言模型（LLM）通过预训练海量文本数据，学习语言的统计规律和语义表示。Transformer架构是现代LLM的核心，其自注意力机制允许模型捕捉长距离依赖关系。尽管LLM取得了巨大成功，但仍面临可解释性差、事实幻觉、偏见等挑战。",
                label="大语言模型特性",
                questions=[
                    "现代大语言模型的核心架构是什么？",
                    "大语言模型面临哪些主要挑战？"
                ],
                key_phrases=["大语言模型", "Transformer", "自注意力机制", "事实幻觉"],
                answer_ranges=[(35, 61), (76, 106)]
            ),
            SampleText(
                text="碳中和是指企业、团体或个人测算在一定时间内直接或间接产生的温室气体排放总量，通过植树造林、节能减排等形式，以抵消自身产生的二氧化碳排放量，实现二氧化碳\"零排放\"。碳中和目标的实现需要能源结构转型、技术创新和社会各界的共同努力。",
                label="碳中和概念",
                questions=[
                    "什么是碳中和？",
                    "实现碳中和目标需要哪些措施？"
                ],
                key_phrases=["碳中和", "温室气体排放", "能源结构转型", "零排放"],
                answer_ranges=[(0, 82), (83, 119)]
            )
        ]
        
        # 客服对话场景样本
        cs_samples = [
            SampleText(
                text="用户：你好，我想咨询一下我的订单什么时候能发货？\n客服：您好！很高兴为您服务。请提供一下您的订单号，我帮您查询一下。\n用户：订单号是OD20230516001。\n客服：好的，我正在为您查询。根据系统显示，您的订单已经安排发货，预计明天下午可以送达。\n用户：好的，谢谢！\n客服：不客气，祝您生活愉快！",
                label="订单查询对话",
                questions=[
                    "用户的订单号是什么？",
                    "用户的订单预计什么时候送达？"
                ],
                key_phrases=["订单号", "发货", "送达时间", "OD20230516001"],
                answer_ranges=[(62, 76), (110, 125)]
            ),
            SampleText(
                text="用户：我的手机充电时发热很严重，这正常吗？\n客服：您好！手机在充电过程中会有一定程度的发热，但如果感觉异常烫，可能是以下原因：充电环境温度过高、使用非原装充电器、后台运行程序过多等。建议您在通风良好的环境下充电，使用原装充电器，并关闭不必要的后台程序。\n用户：好的，我试试，谢谢！\n客服：不客气，如果问题仍然存在，请随时联系我们。",
                label="产品使用问题咨询",
                questions=[
                    "用户遇到了什么问题？",
                    "客服给出了哪些解决建议？"
                ],
                key_phrases=["手机充电", "发热严重", "原装充电器", "后台程序"],
                answer_ranges=[(0, 30), (58, 134)]
            ),
            SampleText(
                text="用户：我想申请退货，商品和描述不符。\n客服：您好！请您详细说明一下商品与描述不符的情况，以便我们为您处理退货申请。\n用户：我购买的是蓝色的耳机，但收到的是黑色的。\n客服：非常抱歉给您带来不便。根据我们的退货政策，您需要在收到商品7天内申请退货，并且商品需要保持完好。请您提供一下订单号，我将为您办理退货手续。\n用户：订单号是OD20230520005。\n客服：好的，已为您提交退货申请，请您按照系统提示完成后续操作。",
                label="退货申请对话",
                questions=[
                    "用户为什么要申请退货？",
                    "退货需要满足什么条件？"
                ],
                key_phrases=["退货申请", "商品与描述不符", "7天内", "保持完好"],
                answer_ranges=[(0, 20), (86, 126)]
            )
        ]
        
        # 将样本添加到场景字典中
        self.scenarios[ScenarioType.LEGAL.value] = ScenarioSample(
            scenario_type=ScenarioType.LEGAL.value,
            name="法律场景",
            description="包含婚姻家庭、合同纠纷、知识产权等法律领域的典型文本",
            samples=legal_samples
        )
        
        self.scenarios[ScenarioType.SCIENTIFIC.value] = ScenarioSample(
            scenario_type=ScenarioType.SCIENTIFIC.value,
            name="科研场景",
            description="包含量子计算、人工智能、环境保护等科研领域的典型文本",
            samples=scientific_samples
        )
        
        self.scenarios[ScenarioType.CUSTOMER_SERVICE.value] = ScenarioSample(
            scenario_type=ScenarioType.CUSTOMER_SERVICE.value,
            name="客服对话场景",
            description="包含订单查询、产品咨询、退换货等客服领域的典型对话",
            samples=cs_samples
        )
    
    def build_evalset(self, scenarios: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """构建验证集数据
        
        功能：根据指定的场景列表，构建并返回验证集数据
        
        实现细节：
        1. 如果未指定场景列表，默认包含所有可用场景
        2. 遍历指定的场景，检查场景是否存在
        3. 对每个存在的场景，提取其所有样本
        4. 将样本转换为统一的字典格式
        5. 合并所有场景的样本数据并返回
        
        Args:
            scenarios: 要包含的场景列表，可选值为['legal', 'scientific', 'customer_service']，
                      如果为None则包含所有场景
        
        Returns:
            验证集数据列表，每个元素是包含完整样本信息的字典
            
        示例返回值：
        [
            {
                "scenario_type": "legal",
                "scenario_name": "法律场景",
                "label": "婚姻家庭法律条款",
                "text": "根据《中华人民共和国民法典》...",
                "questions": [...],
                "key_phrases": [...],
                "answer_ranges": [...]
            },
            ...
        ]
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        result = []
        for scenario_type in scenarios:
            if scenario_type not in self.scenarios:
                print(f"警告：场景类型 {scenario_type} 不存在")
                continue
            
            scenario = self.scenarios[scenario_type]
            for sample in scenario.samples:
                # 转换为字典格式
                sample_dict = {
                    "scenario_type": scenario_type,
                    "scenario_name": scenario.name,
                    "label": sample.label,
                    "text": sample.text,
                    "questions": sample.questions,
                    "key_phrases": sample.key_phrases if sample.key_phrases else [],
                    "answer_ranges": sample.answer_ranges if sample.answer_ranges else []
                }
                result.append(sample_dict)
        
        return result
    
    def save_to_json(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """保存验证集数据到JSON文件
        
        功能：将验证集数据以格式化的JSON格式保存到指定路径
        
        实现细节：
        1. 确保输出目录存在，如果不存在则创建
        2. 以UTF-8编码打开输出文件
        3. 使用json.dump方法将数据写入文件，保持中文显示正常
        4. 设置缩进为2，提高可读性
        5. 打印保存成功的信息
        
        Args:
            data: 验证集数据，通常是build_evalset方法的返回值
            output_path: 输出文件路径，包括文件名
        
        注意事项：
        - 目录路径可以是相对路径或绝对路径
        - 程序会自动创建不存在的中间目录
        """
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"验证集数据已保存到 {output_path}")
    
    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """保存验证集数据到CSV文件
        
        功能：将验证集数据以CSV格式保存到指定路径，便于在电子表格软件中查看和分析
        
        实现细节：
        1. 检查数据是否为空，如果为空则打印警告并返回
        2. 确保输出目录存在，如果不存在则创建
        3. 定义CSV文件的所有字段名称
        4. 以UTF-8编码创建CSV写入器
        5. 写入表头行
        6. 遍历数据，处理列表类型的字段，将其转换为字符串
        7. 写入每一行数据
        8. 打印保存成功的信息
        
        Args:
            data: 验证集数据，通常是build_evalset方法的返回值
            output_path: 输出文件路径，包括文件名
        
        注意事项：
        - 列表类型的字段（如questions、key_phrases等）会被转换为字符串格式
        - CSV文件使用逗号作为分隔符
        """
        if not data:
            print("警告：没有数据可保存到CSV文件")
            return
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 提取所有可能的字段
        fields = ["scenario_type", "scenario_name", "label", "text", "questions", "key_phrases", "answer_ranges"]
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for row in data:
                # 处理列表类型的数据
                formatted_row = {}
                for field in fields:
                    if field in row:
                        if isinstance(row[field], list):
                            formatted_row[field] = str(row[field])
                        else:
                            formatted_row[field] = row[field]
                    else:
                        formatted_row[field] = ""
                
                writer.writerow(formatted_row)
        
        print(f"验证集数据已保存到 {output_path}")

def main():
    """主函数
    
    功能：程序的主要入口，演示如何使用EvalSetBuilder类生成多场景验证集
    
    执行流程：
    1. 创建EvalSetBuilder实例
    2. 调用build_evalset方法构建验证集数据
    3. 创建输出目录（如果不存在）
    4. 以JSON格式保存验证集数据
    5. 以CSV格式保存验证集数据
    6. 打印验证集的统计信息
    
    使用示例：
    - 运行完整程序：python 03-min_evalset_builder_multiscenario.py
    - 生成特定场景验证集：修改build_evalset参数为['legal', 'scientific']
    
    输出：
    - JSON格式的验证集文件：results/multiscenario_evalset.json
    - CSV格式的验证集文件：results/multiscenario_evalset.csv
    - 控制台输出的验证集统计信息
    """
    # 创建验证集构建器
    builder = EvalSetBuilder()
    
    # 构建所有场景的验证集
    evalset_data = builder.build_evalset()
    
    # 创建results目录（如果不存在）
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存为JSON和CSV格式
    json_path = os.path.join(results_dir, "multiscenario_evalset.json")
    csv_path = os.path.join(results_dir, "multiscenario_evalset.csv")
    
    builder.save_to_json(evalset_data, json_path)
    builder.save_to_csv(evalset_data, csv_path)
    
    # 打印统计信息
    print(f"\n验证集生成完成！")
    print(f"- 总样本数：{len(evalset_data)}")
    
    # 按场景统计
    scenario_counts = {}
    for item in evalset_data:
        scenario_type = item["scenario_type"]
        scenario_counts[scenario_type] = scenario_counts.get(scenario_type, 0) + 1
    
    for scenario, count in scenario_counts.items():
        print(f"- {scenario}场景：{count}个样本")

if __name__ == "__main__":
    main()