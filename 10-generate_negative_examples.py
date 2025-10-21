#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负例生成函数及调用示例

作者: Ph.D. Rhino

功能说明:
    演示FAQ数据集构建的关键代码，通过随机错配问答对来创建负面样本，
    使模型能够学习区分相关和不相关的回答，减少幻觉现象。

内容概述:
    1. 创建企业FAQ数据集
    2. 实现多种负例生成方法
    3. 演示不同负例比例对幻觉率的影响
    4. 分析负例质量与模型性能关系

执行流程:
    1. 创建数据目录和基础FAQ数据集
    2. 生成不同比例的负例样本
    3. 模拟模型训练和评估
    4. 分析负例比例与幻觉率的关系
    5. 可视化实验结果

输入说明:
    无需外部输入，程序自动生成测试数据

输出展示:
    - 生成的正例和负例数据集
    - 不同负例比例下的幻觉率统计
    - 实验结果可视化图表
"""

import os
import json
import random
import time
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 依赖包列表
required_packages = ['matplotlib', 'numpy']


def check_dependencies():
    """
    检查并安装必要的依赖包
    """
    print("正在检查依赖...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"发现缺失的依赖包: {', '.join(missing_packages)}")
        print("正在安装...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {package} 安装失败，请手动安装")
                sys.exit(1)
    else:
        print("所有必要的依赖包已安装完成。")


# 确保依赖已安装
check_dependencies()


class FAQDatasetGenerator:
    """
    FAQ数据集生成器，用于创建企业FAQ问答对数据集
    """
    
    def __init__(self, data_dir="data"):
        """
        初始化数据集生成器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        self.faq_categories = [
            "产品咨询", "技术支持", "售后服务", "账户管理", 
            "订单处理", "支付问题", "配送信息", "退款政策"
        ]
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_enterprise_faq(self, num_questions=50):
        """
        生成企业FAQ数据集
        
        Args:
            num_questions: 生成的问题数量
            
        Returns:
            list: FAQ问答对列表
        """
        print(f"正在生成企业FAQ数据集，共{num_questions}个问答对...")
        
        # 基础问题模板
        question_templates = {
            "产品咨询": [
                "{product}的主要功能是什么？",
                "{product}适合哪些用户使用？",
                "{product}和{competitor}相比有什么优势？",
                "如何选择适合我的{product}版本？",
                "{product}的价格是多少？"
            ],
            "技术支持": [
                "如何解决{product}的{issue}问题？",
                "{product}无法启动怎么办？",
                "如何更新{product}到最新版本？",
                "{product}出现错误代码{code}是什么意思？",
                "如何优化{product}的性能？"
            ],
            "售后服务": [
                "{product}的保修期是多久？",
                "如何申请{product}的售后维修？",
                "售后服务的工作时间是什么时候？",
                "售后问题一般多久能解决？",
                "如何联系在线客服？"
            ],
            "账户管理": [
                "如何注册新账户？",
                "忘记密码怎么办？",
                "如何修改账户信息？",
                "如何绑定手机/邮箱？",
                "如何注销账户？"
            ],
            "订单处理": [
                "如何查询订单状态？",
                "订单提交后可以修改吗？",
                "如何取消订单？",
                "订单一般多久能确认？",
                "如何合并多个订单？"
            ]
        }
        
        # 产品和问题类型
        products = ["智能助手软件", "数据分析平台", "客户关系管理系统", "云存储服务", "安全防护软件"]
        competitors = ["市场同类产品A", "市场同类产品B", "行业领先解决方案"]
        issues = ["登录失败", "系统崩溃", "数据丢失", "功能异常", "响应缓慢"]
        codes = ["E001", "E002", "E104", "F201", "P302"]
        
        faq_data = []
        
        for i in range(num_questions):
            # 随机选择分类
            category = random.choice(list(question_templates.keys()))
            
            # 随机选择问题模板
            template = random.choice(question_templates[category])
            
            # 填充模板变量
            question = template.format(
                product=random.choice(products),
                competitor=random.choice(competitors),
                issue=random.choice(issues),
                code=random.choice(codes)
            )
            
            # 生成对应的答案
            answer = self._generate_answer(category, question)
            
            # 添加到数据集
            faq_data.append({
                "id": f"faq_{i+1}",
                "category": category,
                "question": question,
                "answer": answer,
                "is_relevant": True  # 标记为正例
            })
        
        # 保存数据到文件
        output_file = os.path.join(self.data_dir, "enterprise_faq.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        
        print(f"FAQ数据集已保存至: {output_file}")
        print(f"共生成 {len(faq_data)} 个问答对")
        
        return faq_data
    
    def _generate_answer(self, category, question):
        """
        根据分类和问题生成答案
        
        Args:
            category: 问题分类
            question: 问题文本
            
        Returns:
            str: 生成的答案
        """
        # 基于分类的答案模板
        answer_templates = {
            "产品咨询": [
                "该产品主要提供{feature1}和{feature2}功能，非常适合{user_type}使用。",
                "我们的产品采用了先进的{tech}技术，能够有效解决{problem}问题。",
                "价格方面，我们提供多种套餐选择，从{min_price}到{max_price}不等，可以满足不同需求。",
                "与竞品相比，我们的产品在{advantage1}和{advantage2}方面具有明显优势。"
            ],
            "技术支持": [
                "针对您提到的问题，建议您先尝试{solution1}，如果问题仍然存在，可以{solution2}。",
                "这个错误通常是由于{reason}导致的，您可以通过{steps}来解决。",
                "请确保您的系统满足最低要求：{requirements}。如仍有问题，请联系技术支持。",
                "最新版本已经修复了这个问题，建议您立即更新到版本{version}。"
            ],
            "售后服务": [
                "我们提供{warranty}的保修期，涵盖{coverage}范围。",
                "您可以通过{channel1}或{channel2}申请售后服务，我们会在{timeframe}内响应。",
                "售后服务热线工作时间为{hours}，也可以通过官网在线客服获得支持。",
                "维修完成后，我们会提供{additional}服务，确保您的使用体验。"
            ],
            "账户管理": [
                "要完成此操作，请登录您的账户，然后在{location}找到相关设置选项。",
                "出于安全考虑，您需要验证您的身份，然后才能进行此操作。",
                "此更改将在{time}内生效，请刷新页面查看更新后的信息。",
                "如果您遇到任何困难，可以通过{contact}联系我们的客户支持团队。"
            ],
            "订单处理": [
                "您可以在'我的账户'-'订单管理'页面查看订单的实时状态。",
                "订单提交后30分钟内可以修改，超过时间需要联系客服协助处理。",
                "取消订单的申请将在{process_time}内处理，退款将在{refund_time}内原路返回。",
                "如需合并订单，请在提交前选择'合并配送'选项，或联系客服协助。"
            ]
        }
        
        # 答案变量池
        features = ["数据分析", "自动报告生成", "智能预测", "用户行为分析", "多平台同步"]
        user_types = ["企业用户", "数据分析人员", "市场决策者", "普通个人用户", "专业技术人员"]
        techs = ["机器学习", "人工智能", "云计算", "大数据分析", "区块链"]
        problems = ["数据管理", "效率低下", "成本控制", "决策支持", "用户体验"]
        prices = ["999元/年", "1999元/年", "2999元/年", "4999元/年", "9999元/年"]
        advantages = ["性能稳定性", "用户友好度", "功能丰富度", "价格优势", "技术创新"]
        solutions = ["重启系统", "清除缓存", "更新驱动", "检查网络连接", "重新安装软件"]
        reasons = ["软件冲突", "系统资源不足", "网络问题", "配置错误", "权限设置不当"]
        steps = ["按照向导指引操作", "参考用户手册第X章", "使用诊断工具扫描", "重置相关设置"]
        requirements = ["Windows 10以上系统", "8GB以上内存", "500MB可用磁盘空间", "稳定的网络连接"]
        warranties = ["一年", "两年", "三年", "终身维护"]
        coverages = ["硬件故障", "软件更新", "技术支持", "配件更换"]
        channels = ["官方网站", "客服热线", "微信公众号", "线下服务中心"]
        timeframes = ["24小时", "48小时", "3个工作日", "5个工作日"]
        
        # 选择模板并填充变量
        template = random.choice(answer_templates[category])
        
        # 随机选择变量值
        answer = template.format(
            feature1=random.choice(features),
            feature2=random.choice(features),
            user_type=random.choice(user_types),
            tech=random.choice(techs),
            problem=random.choice(problems),
            min_price=random.choice(prices),
            max_price=random.choice(prices),
            advantage1=random.choice(advantages),
            advantage2=random.choice(advantages),
            solution1=random.choice(solutions),
            solution2=random.choice(solutions),
            reason=random.choice(reasons),
            steps=random.choice(steps),
            requirements=random.choice(requirements),
            version=f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
            warranty=random.choice(warranties),
            coverage=random.choice(coverages),
            channel1=random.choice(channels),
            channel2=random.choice(channels),
            timeframe=random.choice(timeframes),
            hours="周一至周五 9:00-18:00",
            additional="7天无理由退换",
            location="账户设置",
            time="24小时",
            contact="客服热线400-123-4567",
            process_time="1个工作日",
            refund_time="3-7个工作日"
        )
        
        return answer


class NegativeExampleGenerator:
    """
    负例生成器，用于创建不相关的问答对样本
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
    
    def generate_negative_examples(self, faq_data, negative_ratio=1.0, method="random"):
        """
        生成负例样本
        
        Args:
            faq_data: 原始FAQ数据（正例）
            negative_ratio: 负例比例（相对于正例数量）
            method: 负例生成方法 (random, cross_category, similar_question)
            
        Returns:
            list: 包含正例和负例的混合数据集
        """
        num_positive = len(faq_data)
        num_negative = int(num_positive * negative_ratio)
        
        print(f"\n使用 {method} 方法生成负例...")
        print(f"正例数量: {num_positive}")
        print(f"目标负例数量: {num_negative} (比例: {negative_ratio:.2f})")
        
        # 初始化结果列表（包含所有正例）
        mixed_dataset = faq_data.copy()
        
        # 生成负例
        negative_examples = []
        questions = [item["question"] for item in faq_data]
        answers = [item["answer"] for item in faq_data]
        categories = [item["category"] for item in faq_data]
        
        # 确保有足够的问答对可以错配
        if num_negative > 0:
            # 根据选择的方法生成负例
            if method == "random":
                # 随机错配方法
                negative_examples = self._generate_random_negatives(
                    faq_data, questions, answers, num_negative
                )
            elif method == "cross_category":
                # 跨类别错配方法
                negative_examples = self._generate_cross_category_negatives(
                    faq_data, categories, num_negative
                )
            elif method == "similar_question":
                # 相似问题错配方法（这里简化实现为随机错配的一种变体）
                negative_examples = self._generate_similar_question_negatives(
                    faq_data, questions, num_negative
                )
            else:
                raise ValueError(f"不支持的负例生成方法: {method}")
        
        # 添加负例到混合数据集
        mixed_dataset.extend(negative_examples)
        
        # 打乱数据集顺序
        random.shuffle(mixed_dataset)
        
        # 保存混合数据集
        output_file = os.path.join(self.data_dir, f"mixed_dataset_{method}_{negative_ratio:.2f}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mixed_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n混合数据集已保存至: {output_file}")
        print(f"数据集总大小: {len(mixed_dataset)} (正例: {num_positive}, 负例: {len(negative_examples)})")
        
        return mixed_dataset
    
    def _generate_random_negatives(self, faq_data, questions, answers, num_negative):
        """
        随机错配生成负例
        
        Args:
            faq_data: 原始FAQ数据
            questions: 问题列表
            answers: 答案列表
            num_negative: 需要生成的负例数量
            
        Returns:
            list: 负例样本列表
        """
        negative_examples = []
        negative_count = 0
        
        # 跟踪已使用的（问题索引，答案索引）对，避免重复
        used_pairs = set()
        
        while negative_count < num_negative:
            # 随机选择一个问题
            q_idx = random.randint(0, len(faq_data) - 1)
            question_item = faq_data[q_idx]
            
            # 随机选择一个答案，但不能是问题对应的正确答案
            a_idx = random.randint(0, len(faq_data) - 1)
            
            # 确保不是同一个问答对，并且没有重复使用
            if q_idx != a_idx and (q_idx, a_idx) not in used_pairs:
                used_pairs.add((q_idx, a_idx))
                
                # 创建负例
                negative_example = {
                    "id": f"neg_{negative_count+1}",
                    "category": question_item["category"],
                    "question": question_item["question"],
                    "answer": answers[a_idx],
                    "is_relevant": False,  # 标记为负例
                    "original_answer": question_item["answer"]  # 记录原始正确答案
                }
                
                negative_examples.append(negative_example)
                negative_count += 1
        
        return negative_examples
    
    def _generate_cross_category_negatives(self, faq_data, categories, num_negative):
        """
        跨类别错配生成负例
        
        Args:
            faq_data: 原始FAQ数据
            categories: 类别列表
            num_negative: 需要生成的负例数量
            
        Returns:
            list: 负例样本列表
        """
        negative_examples = []
        negative_count = 0
        
        # 按类别分组问题和答案
        category_items = {}
        for item in faq_data:
            cat = item["category"]
            if cat not in category_items:
                category_items[cat] = []
            category_items[cat].append(item)
        
        # 所有可用类别
        all_categories = list(category_items.keys())
        
        while negative_count < num_negative:
            # 随机选择一个问题
            q_idx = random.randint(0, len(faq_data) - 1)
            question_item = faq_data[q_idx]
            q_category = question_item["category"]
            
            # 随机选择一个不同类别的答案
            other_categories = [cat for cat in all_categories if cat != q_category]
            if not other_categories:
                continue  # 如果只有一个类别，则回退到随机错配
            
            a_category = random.choice(other_categories)
            a_item = random.choice(category_items[a_category])
            
            # 创建负例
            negative_example = {
                "id": f"neg_{negative_count+1}",
                "category": q_category,
                "question": question_item["question"],
                "answer": a_item["answer"],
                "is_relevant": False,
                "original_answer": question_item["answer"],
                "mismatch_category": a_category  # 记录不匹配的类别
            }
            
            negative_examples.append(negative_example)
            negative_count += 1
        
        return negative_examples
    
    def _generate_similar_question_negatives(self, faq_data, questions, num_negative):
        """
        相似问题错配生成负例（简化版）
        
        Args:
            faq_data: 原始FAQ数据
            questions: 问题列表
            num_negative: 需要生成的负例数量
            
        Returns:
            list: 负例样本列表
        """
        # 这里使用简化实现，选择长度相近的问题进行错配
        # 在实际应用中，可以使用更复杂的文本相似度算法
        negative_examples = []
        negative_count = 0
        
        # 按问题长度分组
        length_groups = {}
        for i, item in enumerate(faq_data):
            length = len(item["question"])
            length_range = length // 5 * 5  # 每5个字符为一个范围
            if length_range not in length_groups:
                length_groups[length_range] = []
            length_groups[length_range].append(i)
        
        # 可用的长度范围
        available_ranges = list(length_groups.keys())
        
        while negative_count < num_negative:
            # 随机选择一个问题
            q_idx = random.randint(0, len(faq_data) - 1)
            question_item = faq_data[q_idx]
            q_length = len(question_item["question"])
            q_length_range = q_length // 5 * 5
            
            # 在相似长度范围内选择不同的问题对应的答案
            if q_length_range in length_groups and len(length_groups[q_length_range]) > 1:
                # 从相似长度范围中选择一个不同的问题索引
                candidates = [idx for idx in length_groups[q_length_range] if idx != q_idx]
                if candidates:
                    a_idx = random.choice(candidates)
                    a_item = faq_data[a_idx]
                    
                    # 创建负例
                    negative_example = {
                        "id": f"neg_{negative_count+1}",
                        "category": question_item["category"],
                        "question": question_item["question"],
                        "answer": a_item["answer"],
                        "is_relevant": False,
                        "original_answer": question_item["answer"],
                        "similar_length": True  # 标记为相似长度错配
                    }
                    
                    negative_examples.append(negative_example)
                    negative_count += 1
            else:
                # 如果相似长度范围内没有足够的问题，则回退到随机错配
                a_idx = random.randint(0, len(faq_data) - 1)
                if a_idx != q_idx:
                    a_item = faq_data[a_idx]
                    
                    negative_example = {
                        "id": f"neg_{negative_count+1}",
                        "category": question_item["category"],
                        "question": question_item["question"],
                        "answer": a_item["answer"],
                        "is_relevant": False,
                        "original_answer": question_item["answer"]
                    }
                    
                    negative_examples.append(negative_example)
                    negative_count += 1
        
        return negative_examples


class HallucinationAnalyzer:
    """
    幻觉率分析器，用于模拟不同负例比例下的模型性能
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
    
    def simulate_model_performance(self, faq_data, negative_ratios=[0.5, 1.0, 2.0, 3.0, 5.0]):
        """
        模拟不同负例比例下的模型性能
        
        Args:
            faq_data: 原始FAQ数据
            negative_ratios: 要测试的负例比例列表
            
        Returns:
            dict: 不同比例下的性能统计
        """
        print("\n===== 开始模拟不同负例比例下的模型性能 =====")
        
        # 结果存储
        performance_results = {}
        visualization_data = []
        
        # 基础幻觉率（无负例时）
        base_hallucination_rate = 0.35  # 假设无负例时模型幻觉率为35%
        base_accuracy = 0.75  # 假设基础准确率为75%
        
        # 模拟不同负例比例
        for ratio in negative_ratios:
            print(f"\n测试负例比例: {ratio:.2f}")
            
            # 生成该比例下的负例
            neg_generator = NegativeExampleGenerator(self.data_dir)
            mixed_dataset = neg_generator.generate_negative_examples(
                faq_data, negative_ratio=ratio, method="random"
            )
            
            # 模拟模型训练和评估（这里使用数学模型模拟效果）
            # 随着负例比例增加，幻觉率降低，但准确率提升会逐渐饱和
            hallucination_rate = base_hallucination_rate * (1 - min(ratio / 10, 0.8))
            
            # 准确率提升，但也有上限
            accuracy_boost = min(ratio / 8, 0.15)  # 最大提升15%
            accuracy = base_accuracy + accuracy_boost
            
            # 计算其他指标
            precision = accuracy * (1 - hallucination_rate)
            recall = accuracy * (1 - 0.1 * ratio / 5)  # 过度负例可能降低召回率
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 记录结果
            performance_results[ratio] = {
                "hallucination_rate": hallucination_rate,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "dataset_size": len(mixed_dataset),
                "positive_count": sum(1 for item in mixed_dataset if item["is_relevant"]),
                "negative_count": sum(1 for item in mixed_dataset if not item["is_relevant"])
            }
            
            visualization_data.append({
                "ratio": ratio,
                "hallucination_rate": hallucination_rate,
                "accuracy": accuracy,
                "f1_score": f1_score
            })
            
            # 输出当前比例的结果
            print(f"  幻觉率: {hallucination_rate:.2%}")
            print(f"  准确率: {accuracy:.2%}")
            print(f"  精确率: {precision:.2%}")
            print(f"  召回率: {recall:.2%}")
            print(f"  F1分数: {f1_score:.4f}")
        
        # 可视化结果
        self._visualize_results(visualization_data)
        
        # 保存性能结果
        output_file = os.path.join(self.data_dir, "performance_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(performance_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n性能结果已保存至: {output_file}")
        
        return performance_results
    
    def _visualize_results(self, data):
        """
        可视化不同负例比例下的幻觉率变化
        
        Args:
            data: 性能数据列表
        """
        ratios = [item["ratio"] for item in data]
        hallucination_rates = [item["hallucination_rate"] * 100 for item in data]  # 转换为百分比
        accuracies = [item["accuracy"] * 100 for item in data]  # 转换为百分比
        f1_scores = [item["f1_score"] * 100 for item in data]  # 转换为百分比
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 创建两个子图
        ax1 = plt.subplot(111)
        
        # 绘制幻觉率曲线
        color1 = 'tab:red'
        ax1.set_xlabel('负例比例')
        ax1.set_ylabel('幻觉率 (%)', color=color1)
        ax1.plot(ratios, hallucination_rates, 'o-', color=color1, label='幻觉率')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 40)  # 设置y轴范围
        
        # 创建第二个y轴
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('准确率/ F1分数 (%)', color=color2)
        ax2.plot(ratios, accuracies, 's-', color='tab:blue', label='准确率')
        ax2.plot(ratios, f1_scores, '^-', color='tab:green', label='F1分数')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(70, 100)  # 设置y轴范围
        
        # 添加标题和网格
        plt.title('不同负例比例下的幻觉率与模型性能变化')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
        
        # 添加标注
        for i, ratio in enumerate(ratios):
            ax1.annotate(f'{hallucination_rates[i]:.1f}%', 
                        xy=(ratio, hallucination_rates[i]),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        color=color1)
            
            ax2.annotate(f'{accuracies[i]:.1f}%', 
                        xy=(ratio, accuracies[i]),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        color='tab:blue')
            
            ax2.annotate(f'{f1_scores[i]:.1f}%', 
                        xy=(ratio, f1_scores[i]),
                        xytext=(0, -15),
                        textcoords='offset points',
                        ha='center',
                        color='tab:green')
        
        # 保存图表
        output_file = os.path.join(self.data_dir, "hallucination_analysis.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存至: {output_file}")
    
    def analyze_negative_impact(self, performance_results):
        """
        分析负例对幻觉率的影响规律
        
        Args:
            performance_results: 不同比例下的性能统计
        """
        print("\n===== 负例比例对幻觉率的影响分析 =====")
        
        # 提取数据
        ratios = list(performance_results.keys())
        hallucination_rates = [results["hallucination_rate"] for results in performance_results.values()]
        
        # 计算幻觉率下降幅度
        base_rate = hallucination_rates[0]
        reduction_rates = [(base_rate - rate) / base_rate for rate in hallucination_rates]
        
        print(f"\n幻觉率变化趋势:")
        for i, ratio in enumerate(ratios):
            print(f"  负例比例 {ratio:.2f}: 幻觉率 {hallucination_rates[i]:.2%}, "
                  f"相对基准下降 {reduction_rates[i]:.2%}")
        
        # 分析最佳负例比例
        # 找到幻觉率下降最显著的点
        best_ratio_index = 0
        max_reduction_rate = reduction_rates[0]
        
        for i, reduction in enumerate(reduction_rates):
            if reduction > max_reduction_rate:
                max_reduction_rate = reduction
                best_ratio_index = i
        
        best_ratio = ratios[best_ratio_index]
        best_hallucination = hallucination_rates[best_ratio_index]
        
        print(f"\n关键发现:")
        print(f"1. 基础幻觉率（负例比例 {ratios[0]:.2f}）: {hallucination_rates[0]:.2%}")
        print(f"2. 最低幻觉率（负例比例 {best_ratio:.2f}）: {best_hallucination:.2%}")
        print(f"3. 最大幻觉率降低幅度: {max_reduction_rate:.2%}")
        
        # 分析边际效益递减点
        if len(ratios) >= 3:
            # 计算相邻点之间的变化率
            marginal_changes = []
            for i in range(1, len(ratios)):
                change = (hallucination_rates[i-1] - hallucination_rates[i]) / hallucination_rates[i-1]
                marginal_changes.append(change)
            
            # 寻找边际效益开始明显下降的点
            diminishing_point = -1
            for i in range(1, len(marginal_changes)):
                if marginal_changes[i] < marginal_changes[i-1] * 0.5:  # 变化率下降超过50%
                    diminishing_point = i
                    break
            
            if diminishing_point >= 0:
                print(f"4. 边际效益递减点: 负例比例 {ratios[diminishing_point+1]:.2f}")
                print(f"   此比例后，每增加相同比例的负例，幻觉率降低的效果显著减弱")
            else:
                print("4. 在测试范围内未观察到明显的边际效益递减现象")
        
        print(f"\n建议:")
        if best_ratio <= 2.0:
            print(f"- 建议负例比例: {best_ratio:.2f}")
            print(f"- 理由: 在该比例下可以获得最佳的幻觉率降低效果，同时不会显著增加数据规模")
        else:
            print(f"- 建议负例比例: 2.0")
            print(f"- 理由: 虽然负例比例 {best_ratio:.2f} 可以获得最低幻觉率，但考虑到数据规模和训练成本，")
            print(f"  使用2.0的比例可以在幻觉率和计算效率之间取得较好平衡")


def demonstrate_negative_example_quality(faq_data, data_dir="data"):
    """
    演示不同质量负例的效果
    
    Args:
        faq_data: 原始FAQ数据
        data_dir: 数据保存目录
    """
    print("\n===== 不同负例生成方法质量对比 =====")
    
    # 定义要测试的方法
    methods = ["random", "cross_category", "similar_question"]
    negative_ratio = 1.0  # 固定负例比例
    
    # 生成不同方法的负例
    results = {}
    
    for method in methods:
        neg_generator = NegativeExampleGenerator(data_dir)
        mixed_dataset = neg_generator.generate_negative_examples(
            faq_data, negative_ratio=negative_ratio, method=method
        )
        
        # 提取负例
        negatives = [item for item in mixed_dataset if not item["is_relevant"]]
        
        # 模拟不同质量负例的效果（这里简化为基于方法类型的假设）
        if method == "random":
            quality_score = 65  # 基础质量分数
            hallucination_reduction = 20  # 幻觉率降低百分比
        elif method == "cross_category":
            quality_score = 75  # 中等质量
            hallucination_reduction = 25
        elif method == "similar_question":
            quality_score = 85  # 高质量
            hallucination_reduction = 30
        
        # 保存结果
        results[method] = {
            "negative_count": len(negatives),
            "quality_score": quality_score,
            "hallucination_reduction": hallucination_reduction
        }
        
        print(f"\n方法: {method}")
        print(f"  生成负例数: {len(negatives)}")
        print(f"  估计负例质量分数: {quality_score}/100")
        print(f"  估计幻觉率降低: {hallucination_reduction}%")
    
    # 创建简单的质量对比图
    plt.figure(figsize=(10, 5))
    
    # 提取数据
    method_names = ["随机错配", "跨类别错配", "相似问题错配"]
    quality_scores = [results[m]["quality_score"] for m in methods]
    hallucination_reductions = [results[m]["hallucination_reduction"] for m in methods]
    
    # 创建子图
    ax1 = plt.subplot(111)
    
    # 绘制质量分数
    color1 = 'tab:blue'
    ax1.set_xlabel('负例生成方法')
    ax1.set_ylabel('负例质量分数', color=color1)
    bars1 = ax1.bar(method_names, quality_scores, color=color1, alpha=0.6, label='负例质量分数')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}', ha='center', va='bottom', color=color1)
    
    # 创建第二个y轴
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('幻觉率降低百分比 (%)', color=color2)
    bars2 = ax2.bar([x + 0.2 for x in range(len(method_names))], 
                   hallucination_reductions, color=color2, alpha=0.6, label='幻觉率降低')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 40)
    
    # 在柱状图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', color=color2)
    
    # 添加标题和图例
    plt.title('不同负例生成方法的质量与幻觉率降低效果对比')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图表
    output_file = os.path.join(data_dir, "negative_example_quality.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n质量对比图已保存至: {output_file}")


def main():
    """
    主函数
    """
    print("===== 负例生成函数及FAQ数据集构建演示 =====")
    
    # 创建数据目录
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 生成企业FAQ数据集
    print("\n1. 生成企业FAQ数据集")
    faq_generator = FAQDatasetGenerator(data_dir)
    faq_data = faq_generator.generate_enterprise_faq(num_questions=30)
    
    # 2. 演示不同负例生成方法
    print("\n2. 演示不同负例生成方法")
    demonstrate_negative_example_quality(faq_data, data_dir)
    
    # 3. 测试不同负例比例对幻觉率的影响
    print("\n3. 测试不同负例比例对幻觉率的影响")
    analyzer = HallucinationAnalyzer(data_dir)
    negative_ratios = [0.5, 1.0, 2.0, 3.0, 5.0]
    performance_results = analyzer.simulate_model_performance(faq_data, negative_ratios)
    
    # 4. 分析负例对幻觉率的影响规律
    analyzer.analyze_negative_impact(performance_results)
    
    # 5. 生成示例负例样本用于展示
    print("\n4. 生成示例负例用于展示")
    neg_generator = NegativeExampleGenerator(data_dir)
    mixed_dataset = neg_generator.generate_negative_examples(
        faq_data[:5], negative_ratio=1.0, method="random"
    )
    
    # 显示正例和负例样本
    print("\n===== 样本展示 =====")
    
    # 显示正例
    print("\n正例样本:")
    positive_examples = [item for item in mixed_dataset if item["is_relevant"]][:2]
    for i, example in enumerate(positive_examples):
        print(f"\n正例 {i+1}:")
        print(f"  问题: {example['question']}")
        print(f"  答案: {example['answer']}")
    
    # 显示负例
    print("\n负例样本:")
    negative_examples = [item for item in mixed_dataset if not item["is_relevant"]][:2]
    for i, example in enumerate(negative_examples):
        print(f"\n负例 {i+1}:")
        print(f"  问题: {example['question']}")
        print(f"  错误答案: {example['answer']}")
        print(f"  正确答案: {example['original_answer']}")
    
    print("\n===== 演示完成 =====")
    print("\n关键结论:")
    print("1. 适当比例的负例可以有效降低模型幻觉率")
    print("2. 高质量的负例（如跨类别错配、相似问题错配）具有更好的训练效果")
    print("3. 负例比例通常在1.0-2.0之间可以取得较好的幻觉率降低效果和计算效率平衡")
    print("4. 建议在实际应用中根据具体任务和数据特点调整负例比例和生成策略")


if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    
    try:
        main()
        
        # 计算执行时间
        execution_time = time.time() - start_time
        print(f"\n程序执行时间: {execution_time:.2f} 秒")
        
        # 检查执行效率
        if execution_time > 30:
            print("警告: 程序执行时间超过30秒，可能需要优化代码")
        else:
            print("✓ 程序执行效率良好")
    
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")