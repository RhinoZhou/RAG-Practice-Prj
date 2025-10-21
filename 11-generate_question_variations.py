#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题多样化生成功能演示

作者: Ph.D. Rhino

功能说明:
    展示了问题多样化生成代码，通过模板变换创建同一问题的不同表达方式，
    增强模型对多种问法的理解能力。支持多种变体生成策略，包括模板替换、
    同义词替换、句式转换等方法，适用于扩展训练数据集和提高模型鲁棒性。

内容概述:
    1. 实现多种问题变体生成方法
    2. 创建示例数据集并应用变体生成
    3. 评估变体质量和多样性
    4. 可视化变体生成效果
    5. 提供统计分析和应用建议

执行流程:
    1. 检查并安装必要依赖
    2. 创建示例问题数据集
    3. 应用不同的变体生成策略
    4. 评估和比较变体质量
    5. 生成可视化结果和统计报告

输入说明:
    无需外部输入，程序自动生成测试数据

输出展示:
    - 原始问题和生成的变体示例
    - 变体质量和多样性统计
    - 可视化图表展示不同生成策略效果
"""

import os
import json
import random
import time
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Set

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


class QuestionVariationGenerator:
    """
    问题变体生成器，支持多种策略生成问题的不同表达方式
    """
    
    def __init__(self):
        """
        初始化问题变体生成器，加载必要的模板和词表
        """
        # 基础问题模板
        self.base_templates = [
            "请问{q}？",
            "我想知道{q}？",
            "{q}是什么？",
            "能告诉我{q}吗？",
            "关于{q}的问题？",
            "如何{q}？",
            "有没有关于{q}的信息？",
            "我想了解{q}。",
            "{q}的具体情况是怎样的？",
            "可以详细说明一下{q}吗？"
        ]
        
        # 疑问词替换映射
        self.question_word_mapping = {
            "什么": ["哪些", "啥", "什么东西", "什么内容", "什么情况"],
            "怎么": ["如何", "怎样", "咋样", "怎么弄", "怎么操作"],
            "为什么": ["为啥", "为何", "原因是什么", "为什么会这样"],
            "哪里": ["哪儿", "什么地方", "在何处", "在什么位置"],
            "什么时候": ["何时", "什么时间", "什么时候开始", "什么时候结束"],
            "谁": ["哪个", "哪位", "哪个人", "什么人"]
        }
        
        # 句式转换模板
        self.sentence_transform_templates = {
            # 主动转被动
            "如何{action}{target}": ["{target}如何被{action}"],
            # 直接问法转间接问法
            "{q}需要多久": ["我想知道{q}需要花费多长时间", "{q}大概需要多少时间"],
            # 简单问法转复杂问法
            "{q}多少钱": ["{q}的价格是多少", "{q}的费用如何", "{q}大概需要花费多少"],
            # 口语化转书面化
            "{q}咋弄": ["{q}应该如何操作", "{q}的正确操作方法是什么"]
        }
        
        # 常用前缀和后缀
        self.prefixes = [
            "打扰一下，", "您好，", "请教一下，", "请问，",
            "想咨询一个问题，", "我有个疑问，", "不好意思，", "麻烦问一下，"
        ]
        
        self.suffixes = [
            "，谢谢！", "，非常感谢。", "，期待您的回复。", 
            "，请告知。", "，麻烦详细说明一下。"
        ]
    
    def generate_question_variations(self, question: str, n: int = 3) -> List[str]:
        """
        生成原始问题的多种变体
        
        Args:
            question: 原始问题文本
            n: 要生成的变体数量
            
        Returns:
            List[str]: 问题变体列表
        """
        variations = []
        generated_variations = set()  # 用于去重
        
        # 清理原始问题
        q_clean = self._clean_question(question)
        
        # 1. 基于模板的变体生成
        template_variations = self._generate_template_variations(q_clean, n=max(1, n//3))
        for var in template_variations:
            if var not in generated_variations and var != q_clean:
                generated_variations.add(var)
                variations.append(var)
            if len(variations) >= n:
                return variations
        
        # 2. 疑问词替换变体
        question_word_variations = self._generate_question_word_variations(q_clean, n=max(1, n//3))
        for var in question_word_variations:
            if var not in generated_variations and var != q_clean:
                generated_variations.add(var)
                variations.append(var)
            if len(variations) >= n:
                return variations
        
        # 3. 句式转换变体
        transform_variations = self._generate_sentence_transform_variations(q_clean, n=max(1, n//3))
        for var in transform_variations:
            if var not in generated_variations and var != q_clean:
                generated_variations.add(var)
                variations.append(var)
            if len(variations) >= n:
                return variations
        
        # 4. 添加前缀后缀变体
        if len(variations) < n:
            prefix_suffix_variations = self._generate_prefix_suffix_variations(
                q_clean, n=n-len(variations)
            )
            for var in prefix_suffix_variations:
                if var not in generated_variations and var != q_clean:
                    generated_variations.add(var)
                    variations.append(var)
                if len(variations) >= n:
                    break
        
        # 5. 如果还不够，复制已有变体并稍作修改
        if len(variations) < n:
            additional_variations = self._generate_additional_variations(
                q_clean, variations, n=n-len(variations)
            )
            for var in additional_variations:
                if var not in generated_variations and var != q_clean:
                    generated_variations.add(var)
                    variations.append(var)
                if len(variations) >= n:
                    break
        
        return variations
    
    def _clean_question(self, question: str) -> str:
        """
        清理问题文本，去除多余的标点符号等
        
        Args:
            question: 原始问题
            
        Returns:
            str: 清理后的问题
        """
        # 去除首尾空白字符
        q = question.strip()
        
        # 去除末尾的问号
        q = q.rstrip('？').rstrip('?')
        
        return q
    
    def _generate_template_variations(self, question: str, n: int = 3) -> List[str]:
        """
        使用模板生成问题变体
        
        Args:
            question: 清理后的问题
            n: 生成数量
            
        Returns:
            List[str]: 基于模板的变体列表
        """
        variations = []
        
        # 随机选择模板
        selected_templates = random.sample(self.base_templates, min(n, len(self.base_templates)))
        
        for template in selected_templates:
            try:
                variation = template.format(q=question)
                variations.append(variation)
            except (KeyError, IndexError):
                # 如果模板格式化失败，则跳过
                continue
        
        return variations
    
    def _generate_question_word_variations(self, question: str, n: int = 3) -> List[str]:
        """
        通过替换疑问词生成问题变体
        
        Args:
            question: 清理后的问题
            n: 生成数量
            
        Returns:
            List[str]: 替换疑问词后的变体列表
        """
        variations = []
        
        # 检查问题中包含的疑问词
        for question_word, synonyms in self.question_word_mapping.items():
            if question_word in question:
                # 为每个找到的疑问词生成替换变体
                for synonym in synonyms:
                    variation = question.replace(question_word, synonym)
                    variations.append(variation)
                    if len(variations) >= n:
                        return variations
        
        return variations
    
    def _generate_sentence_transform_variations(self, question: str, n: int = 3) -> List[str]:
        """
        通过句式转换生成问题变体
        
        Args:
            question: 清理后的问题
            n: 生成数量
            
        Returns:
            List[str]: 句式转换后的变体列表
        """
        variations = []
        
        # 简单的句式转换规则
        # 这里使用一些启发式规则，实际应用中可以使用更复杂的NLP技术
        
        # 规则1: 主动转被动 (简单实现)
        if question.startswith("如何"):
            # 尝试拆分动宾结构
            parts = question[2:].split("的")
            if len(parts) > 1:
                variation = f"{parts[1]}如何被{parts[0]}"
                variations.append(variation)
        
        # 规则2: 调整句子结构
        if "需要多久" in question:
            variation = question.replace("需要多久", "大概需要多长时间")
            variations.append(variation)
        
        if "多少钱" in question:
            variation = question.replace("多少钱", "的价格是多少")
            variations.append(variation)
        
        # 规则3: 添加修饰词
        if not question.startswith("我想知道"):
            variations.append(f"我想知道{question}")
        
        # 规则4: 改变句子结尾
        if not question.endswith("吗"):
            variations.append(f"{question}吗")
        
        # 过滤并限制数量
        variations = list(set(variations))[:n]
        return variations
    
    def _generate_prefix_suffix_variations(self, question: str, n: int = 3) -> List[str]:
        """
        通过添加前缀和后缀生成问题变体
        
        Args:
            question: 清理后的问题
            n: 生成数量
            
        Returns:
            List[str]: 添加前缀后缀后的变体列表
        """
        variations = []
        
        # 添加前缀
        selected_prefixes = random.sample(self.prefixes, min(n, len(self.prefixes)))
        for prefix in selected_prefixes:
            variation = f"{prefix}{question}"
            variations.append(variation)
            if len(variations) >= n:
                return variations
        
        # 添加后缀
        selected_suffixes = random.sample(self.suffixes, min(n, len(self.suffixes)))
        for suffix in selected_suffixes:
            variation = f"{question}{suffix}"
            variations.append(variation)
            if len(variations) >= n:
                return variations
        
        # 添加前缀和后缀
        if len(variations) < n:
            prefix = random.choice(self.prefixes)
            suffix = random.choice(self.suffixes)
            variation = f"{prefix}{question}{suffix}"
            variations.append(variation)
        
        return variations
    
    def _generate_additional_variations(self, question: str, existing_variations: List[str], n: int = 3) -> List[str]:
        """
        生成额外的变体，用于填充不足的数量
        
        Args:
            question: 清理后的问题
            existing_variations: 已有的变体列表
            n: 需要额外生成的数量
            
        Returns:
            List[str]: 额外生成的变体列表
        """
        variations = []
        
        # 简单的额外变体生成策略
        strategies = [
            lambda q: f"关于{q}，您能详细说明一下吗？",
            lambda q: f"请问{q}具体是怎么回事？",
            lambda q: f"能不能解释一下{q}？",
            lambda q: f"我想请教一下{q}的问题。",
            lambda q: f"对于{q}，有什么建议吗？",
        ]
        
        # 随机选择策略生成变体
        random.shuffle(strategies)
        for strategy in strategies:
            try:
                variation = strategy(question)
                variations.append(variation)
                if len(variations) >= n:
                    break
            except Exception:
                continue
        
        # 如果还不够，复制已有变体并调整标点符号
        if len(variations) < n and existing_variations:
            existing_copy = existing_variations.copy()
            random.shuffle(existing_copy)
            
            for var in existing_copy:
                # 简单调整标点或添加修饰
                if var.endswith('。'):
                    new_var = var[:-1] + '？'
                elif var.endswith('？'):
                    new_var = var[:-1] + '。'
                else:
                    new_var = var + '？'
                
                variations.append(new_var)
                if len(variations) >= n:
                    break
        
        return variations


class DatasetGenerator:
    """
    数据集生成器，用于创建测试问题集
    """
    
    def __init__(self, data_dir="data"):
        """
        初始化数据集生成器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def generate_sample_questions(self, num_questions: int = 50) -> List[Dict[str, str]]:
        """
        生成样本问题数据集
        
        Args:
            num_questions: 生成的问题数量
            
        Returns:
            List[Dict]: 问题数据集列表
        """
        print(f"正在生成样本问题数据集，共{num_questions}个问题...")
        
        # 问题类别和模板
        question_categories = {
            "产品咨询": [
                "{product}的主要功能是什么？",
                "{product}适合哪些用户使用？",
                "如何选择适合我的{product}？",
                "{product}的价格是多少？",
                "{product}和{competitor}相比有什么优势？"
            ],
            "技术支持": [
                "如何解决{product}的{issue}问题？",
                "{product}无法启动怎么办？",
                "如何更新{product}到最新版本？",
                "{product}出现错误代码{code}是什么意思？",
                "使用{product}需要什么系统要求？"
            ],
            "服务咨询": [
                "售后服务的工作时间是什么时候？",
                "如何申请退款？",
                "配送需要多长时间？",
                "如何联系客服？",
                "保修政策是怎样的？"
            ],
            "使用方法": [
                "如何使用{product}的{feature}功能？",
                "{action}需要哪些步骤？",
                "使用{product}有什么技巧？",
                "如何优化{product}的性能？",
                "{product}的快捷键有哪些？"
            ],
            "常见问题": [
                "为什么{product}会出现{problem}？",
                "如何避免{issue}问题？",
                "{error}是怎么回事？",
                "遇到{problem}应该怎么办？",
                "{warning}提示是什么意思？"
            ]
        }
        
        # 填充变量
        products = ["智能手机", "笔记本电脑", "智能手表", "平板电脑", "智能家居系统"]
        competitors = ["品牌A", "品牌B", "同类产品", "其他品牌"]
        features = ["拍照", "数据备份", "语音助手", "多任务处理", "电池优化"]
        issues = ["系统崩溃", "无法连接网络", "应用闪退", "充电缓慢", "屏幕黑屏"]
        codes = ["E001", "F102", "P303", "S404", "U505"]
        actions = ["重置设备", "更新系统", "备份数据", "恢复出厂设置", "连接Wi-Fi"]
        problems = ["闪退", "卡顿", "发热", "耗电快", "无法开机"]
        errors = ["登录失败", "网络错误", "存储空间不足", "权限受限", "同步失败"]
        warnings = ["低电量", "系统更新", "存储空间不足", "高温警告", "连接不稳定"]
        
        dataset = []
        
        for i in range(num_questions):
            # 随机选择类别
            category = random.choice(list(question_categories.keys()))
            
            # 随机选择模板
            template = random.choice(question_categories[category])
            
            # 填充模板变量
            question = template.format(
                product=random.choice(products),
                competitor=random.choice(competitors),
                feature=random.choice(features),
                issue=random.choice(issues),
                code=random.choice(codes),
                action=random.choice(actions),
                problem=random.choice(problems),
                error=random.choice(errors),
                warning=random.choice(warnings)
            )
            
            # 生成简单的答案
            answer = self._generate_simple_answer(category, question)
            
            # 添加到数据集
            dataset.append({
                "id": f"q_{i+1}",
                "category": category,
                "question": question,
                "answer": answer
            })
        
        # 保存数据集
        output_file = os.path.join(self.data_dir, "sample_questions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"样本问题数据集已保存至: {output_file}")
        print(f"共生成 {len(dataset)} 个问题")
        
        return dataset
    
    def _generate_simple_answer(self, category: str, question: str) -> str:
        """
        生成简单的答案
        
        Args:
            category: 问题类别
            question: 问题文本
            
        Returns:
            str: 生成的答案
        """
        # 基于类别的简单答案模板
        answer_templates = {
            "产品咨询": [
                "该产品具有{feature}功能，非常适合{user_type}使用。",
                "价格方面，我们提供多种选择，从{min_price}到{max_price}不等。",
                "与竞品相比，我们的产品在{advantage}方面具有明显优势。"
            ],
            "技术支持": [
                "针对这个问题，建议您先尝试{solution}，如果问题仍然存在，请联系客服。",
                "这个错误通常是由于{reason}导致的，可以通过{steps}来解决。",
                "请确保您的系统满足最低要求：{requirements}。"
            ],
            "服务咨询": [
                "我们的服务时间是{hours}，您可以通过{channel}联系我们。",
                "退款申请将在{process_time}内处理完成。",
                "配送时间通常为{days}个工作日，具体视地区而定。"
            ],
            "使用方法": [
                "要使用该功能，请按照以下步骤操作：{steps}。",
                "您可以通过{method}来实现这个操作。",
                "为了获得最佳体验，建议您{tip}。"
            ],
            "常见问题": [
                "这个问题通常是由于{cause}造成的，解决方案是{solution}。",
                "为了避免这个问题，建议您{prevention}。",
                "这是一个常见现象，您可以通过{method}来解决。"
            ]
        }
        
        # 答案变量
        features = ["强大的性能", "优秀的用户体验", "丰富的功能", "高性价比", "稳定可靠"]
        user_types = ["普通用户", "专业人士", "商务人士", "学生群体", "游戏玩家"]
        prices = ["999元", "1999元", "2999元", "3999元", "4999元"]
        advantages = ["性能", "价格", "功能丰富度", "用户体验", "售后服务"]
        solutions = ["重启设备", "更新软件", "检查网络连接", "清理缓存", "恢复出厂设置"]
        reasons = ["软件冲突", "网络不稳定", "系统版本过低", "硬件故障", "设置错误"]
        steps = ["打开设置", "选择相应选项", "按照提示操作", "重启设备"]
        requirements = ["Windows 10或更高版本", "8GB以上内存", "500MB可用磁盘空间", "稳定网络连接"]
        hours = ["周一至周五 9:00-18:00", "7x24小时服务", "工作日 8:00-20:00", "每天 10:00-22:00"]
        channels = ["客服热线400-123-4567", "官方网站在线客服", "微信公众号", "线下服务中心"]
        process_times = ["1-3个工作日", "3-5个工作日", "5-7个工作日", "7-10个工作日"]
        days = ["1-2", "2-3", "3-5", "5-7"]
        method = ["点击菜单按钮", "使用快捷键Ctrl+S", "通过设置面板", "使用语音命令"]
        tips = ["定期更新软件", "清理不必要的文件", "优化系统设置", "使用官方推荐的配件"]
        causes = ["操作不当", "软件缺陷", "系统冲突", "硬件老化", "环境因素"]
        preventions = ["定期备份数据", "避免安装不明软件", "保持系统更新", "使用安全软件"]
        
        # 选择模板并填充变量
        template = random.choice(answer_templates[category])
        
        try:
            answer = template.format(
                feature=random.choice(features),
                user_type=random.choice(user_types),
                min_price=random.choice(prices),
                max_price=random.choice([p for p in prices if p > random.choice(prices)] or prices),
                advantage=random.choice(advantages),
                solution=random.choice(solutions),
                reason=random.choice(reasons),
                steps=" → ".join(random.sample(steps, k=random.randint(2, 4))),
                requirements=random.choice(requirements),
                hours=random.choice(hours),
                channel=random.choice(channels),
                process_time=random.choice(process_times),
                days=random.choice(days),
                method=random.choice(method),
                tip=random.choice(tips),
                cause=random.choice(causes),
                prevention=random.choice(preventions)
            )
        except Exception:
            # 如果格式化失败，返回通用答案
            answer = "感谢您的提问。针对您的问题，我们建议您参考官方使用手册或联系客服获取详细解答。"
        
        return answer


class VariationEvaluator:
    """
    变体评估器，用于评估生成的问题变体质量和多样性
    """
    
    def __init__(self):
        """
        初始化变体评估器
        """
        pass
    
    def evaluate_variations(self, original_question: str, variations: List[str]) -> Dict:
        """
        评估问题变体的质量和多样性
        
        Args:
            original_question: 原始问题
            variations: 生成的变体列表
            
        Returns:
            Dict: 评估结果
        """
        # 基本统计
        num_variations = len(variations)
        
        # 计算多样性指标
        # 1. 字符长度多样性
        lengths = [len(v) for v in variations]
        length_mean = np.mean(lengths) if lengths else 0
        length_std = np.std(lengths) if lengths else 0
        
        # 2. 词级别多样性（简单实现）
        all_words = ' '.join(variations).split()
        word_counter = Counter(all_words)
        vocabulary_size = len(word_counter)
        
        # 3. 与原始问题的相似度（简单实现：共同字符比例）
        similarity_scores = []
        original_chars = set(original_question)
        for variation in variations:
            variation_chars = set(variation)
            common_chars = original_chars.intersection(variation_chars)
            similarity = len(common_chars) / len(original_chars) if original_chars else 0
            similarity_scores.append(similarity)
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # 4. 语法正确性（简化评估，实际应用中可以使用NLP工具）
        # 这里使用简单的启发式规则
        valid_variations = 0
        for variation in variations:
            # 基本的语法检查：是否以问号结尾或包含疑问词
            if variation.endswith(('？', '?')) or any(qw in variation for qw in ['什么', '怎么', '如何', '为什么', '哪里', '谁']):
                valid_variations += 1
        
        validity_rate = valid_variations / num_variations if num_variations > 0 else 0
        
        # 5. 去重率
        unique_variations = set(variations)
        uniqueness_rate = len(unique_variations) / num_variations if num_variations > 0 else 0
        
        # 综合质量评分（0-100）
        quality_score = (
            0.3 * validity_rate +  # 语法正确性权重
            0.3 * avg_similarity +  # 语义相关性权重
            0.2 * uniqueness_rate +  # 唯一性权重
            0.2 * min(1.0, vocabulary_size / len(original_question.split()))  # 多样性权重
        ) * 100
        
        return {
            "num_variations": num_variations,
            "length_mean": length_mean,
            "length_std": length_std,
            "vocabulary_size": vocabulary_size,
            "avg_similarity": avg_similarity,
            "validity_rate": validity_rate,
            "uniqueness_rate": uniqueness_rate,
            "quality_score": quality_score,
            "similarity_scores": similarity_scores
        }
    
    def evaluate_dataset_variations(self, dataset: List[Dict], variations_data: List[Dict]) -> Dict:
        """
        评估整个数据集的变体生成效果
        
        Args:
            dataset: 原始数据集
            variations_data: 变体数据
            
        Returns:
            Dict: 数据集级别的评估结果
        """
        # 统计数据
        total_questions = len(dataset)
        total_variations = len(variations_data)
        avg_variations_per_question = total_variations / total_questions if total_questions > 0 else 0
        
        # 按类别统计
        category_stats = {}
        for item in variations_data:
            category = item.get('category', 'Unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'quality_scores': []
                }
            category_stats[category]['count'] += 1
            if 'quality_score' in item:
                category_stats[category]['quality_scores'].append(item['quality_score'])
        
        # 计算类别级别的统计
        for category, stats in category_stats.items():
            if stats['quality_scores']:
                stats['avg_quality'] = np.mean(stats['quality_scores'])
                stats['quality_std'] = np.std(stats['quality_scores'])
            else:
                stats['avg_quality'] = 0
                stats['quality_std'] = 0
        
        # 整体质量统计
        all_quality_scores = [item.get('quality_score', 0) for item in variations_data]
        overall_quality = np.mean(all_quality_scores) if all_quality_scores else 0
        quality_std = np.std(all_quality_scores) if all_quality_scores else 0
        
        # 变体覆盖率（有多少原始问题生成了变体）
        original_ids = set(item['id'] for item in dataset)
        covered_ids = set(item.get('original_id', '') for item in variations_data)
        coverage_rate = len(covered_ids.intersection(original_ids)) / len(original_ids) if original_ids else 0
        
        return {
            "total_questions": total_questions,
            "total_variations": total_variations,
            "avg_variations_per_question": avg_variations_per_question,
            "overall_quality": overall_quality,
            "quality_std": quality_std,
            "coverage_rate": coverage_rate,
            "category_stats": category_stats
        }


class VariationVisualizer:
    """
    变体可视化工具，用于生成变体效果的可视化图表
    """
    
    def __init__(self, data_dir="data"):
        """
        初始化可视化工具
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
    
    def visualize_variation_quality(self, dataset_eval_results: Dict):
        """
        可视化不同类别的变体质量
        
        Args:
            dataset_eval_results: 数据集评估结果
        """
        plt.figure(figsize=(12, 6))
        
        # 提取类别和质量数据
        categories = []
        avg_qualities = []
        counts = []
        
        for category, stats in dataset_eval_results['category_stats'].items():
            categories.append(category)
            avg_qualities.append(stats['avg_quality'])
            counts.append(stats['count'])
        
        # 创建双轴图
        ax1 = plt.subplot(111)
        
        # 绘制质量柱状图
        x = np.arange(len(categories))
        bars = ax1.bar(x, avg_qualities, width=0.6, color='skyblue', alpha=0.7, label='平均质量分数')
        ax1.set_ylabel('平均质量分数', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='y')
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 创建第二个y轴显示变体数量
        ax2 = ax1.twinx()
        ax2.plot(x, counts, 'o-', color='red', label='变体数量')
        ax2.set_ylabel('变体数量', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 设置x轴标签
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        
        # 添加标题和图例
        plt.title('不同类别问题的变体质量分析', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 保存图表
        output_file = os.path.join(self.data_dir, "variation_quality_by_category.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n变体质量可视化图已保存至: {output_file}")
    
    def visualize_variation_examples(self, variations_data: List[Dict], num_examples: int = 3):
        """
        可视化变体生成示例
        
        Args:
            variations_data: 变体数据
            num_examples: 展示的例子数量
        """
        # 选择几个有代表性的例子
        selected_examples = random.sample(variations_data, min(num_examples, len(variations_data)))
        
        # 创建可视化
        plt.figure(figsize=(12, 4 * num_examples))
        
        for i, example in enumerate(selected_examples):
            # 准备数据
            original = example.get('original_question', 'Unknown')
            variations = example.get('variations', [])
            
            # 限制变体数量，避免图表过大
            variations = variations[:5]
            
            # 创建子图
            ax = plt.subplot(num_examples, 1, i+1)
            
            # 计算文本长度作为条形长度
            original_length = len(original)
            variation_lengths = [len(v) for v in variations]
            
            # 绘制原始问题
            ax.barh(0, original_length, color='blue', alpha=0.7, label='原始问题')
            ax.text(original_length + 5, 0, original[:30] + '...' if len(original) > 30 else original,
                   va='center', fontsize=10)
            
            # 绘制变体
            for j, (length, variation) in enumerate(zip(variation_lengths, variations)):
                ax.barh(j+1, length, color='green', alpha=0.5, label='变体问题' if j == 0 else "")
                ax.text(length + 5, j+1, variation[:30] + '...' if len(variation) > 30 else variation,
                       va='center', fontsize=9)
            
            # 设置图表属性
            ax.set_yticks(range(len(variations) + 1))
            ax.set_yticklabels(['原始问题'] + [f'变体 {j+1}' for j in range(len(variations))])
            ax.set_xlabel('文本长度')
            ax.set_title(f'变体生成示例 {i+1}')
            ax.grid(True, linestyle='--', alpha=0.3, axis='x')
            
            if i == 0:  # 只在第一个图添加图例
                ax.legend(loc='upper right')
        
        # 保存图表
        output_file = os.path.join(self.data_dir, "variation_examples.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"变体示例可视化图已保存至: {output_file}")
    
    def visualize_variation_statistics(self, dataset_eval_results: Dict):
        """
        可视化变体生成的整体统计信息
        
        Args:
            dataset_eval_results: 数据集评估结果
        """
        # 提取关键统计数据
        total_questions = dataset_eval_results['total_questions']
        total_variations = dataset_eval_results['total_variations']
        avg_variations = dataset_eval_results['avg_variations_per_question']
        overall_quality = dataset_eval_results['overall_quality']
        coverage_rate = dataset_eval_results['coverage_rate'] * 100
        
        # 创建统计图表
        plt.figure(figsize=(12, 8))
        
        # 1. 总体统计信息
        stats = [
            ("原始问题数", total_questions),
            ("生成变体数", total_variations),
            ("平均变体数/问题", round(avg_variations, 2)),
            ("平均变体质量", round(overall_quality, 1)),
            ("覆盖率", round(coverage_rate, 1))
        ]
        
        # 绘制统计信息
        plt.subplot(2, 1, 1)
        bars = plt.bar([s[0] for s in stats], [s[1] for s in stats], color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.ylabel('数值')
        plt.title('问题变体生成统计信息', fontsize=14)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(stats, key=lambda x: x[1])[1] * 0.02,
                    f'{height}', ha='center', va='bottom')
        
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # 2. 类别分布饼图
        plt.subplot(2, 1, 2)
        categories = []
        counts = []
        for category, stats in dataset_eval_results['category_stats'].items():
            categories.append(category)
            counts.append(stats['count'])
        
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')  # 确保饼图是圆的
        plt.title('变体类别分布', fontsize=14)
        
        # 保存图表
        output_file = os.path.join(self.data_dir, "variation_statistics.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"变体统计图表已保存至: {output_file}")


def generate_diverse_dataset(original_dataset: List[Dict], variations_per_question: int = 5) -> List[Dict]:
    """
    为原始数据集生成多样化问法并保存
    
    Args:
        original_dataset: 原始数据集
        variations_per_question: 每个问题生成的变体数量
        
    Returns:
        List[Dict]: 包含变体的数据集
    """
    print(f"\n正在为数据集生成多样化问法，每个问题生成 {variations_per_question} 个变体...")
    
    # 创建问题变体生成器
    generator = QuestionVariationGenerator()
    evaluator = VariationEvaluator()
    
    # 生成变体数据
    variations_data = []
    dataset_variations = []
    
    for i, item in enumerate(original_dataset):
        if i > 0 and i % 10 == 0:
            print(f"  已处理 {i}/{len(original_dataset)} 个问题")
        
        original_question = item['question']
        
        # 生成变体
        variations = generator.generate_question_variations(original_question, n=variations_per_question)
        
        # 评估变体质量
        eval_results = evaluator.evaluate_variations(original_question, variations)
        
        # 保存变体数据
        variations_data.append({
            "original_id": item['id'],
            "category": item['category'],
            "original_question": original_question,
            "variations": variations,
            "quality_score": eval_results['quality_score'],
            "evaluation": eval_results
        })
        
        # 创建扩充后的数据集条目
        for variation in variations:
            dataset_variations.append({
                "id": f"{item['id']}_var_{variations.index(variation)+1}",
                "original_id": item['id'],
                "category": item['category'],
                "question": variation,
                "answer": item['answer'],
                "original_question": original_question
            })
    
    print(f"\n变体生成完成！共处理 {len(original_dataset)} 个问题，生成 {len(dataset_variations)} 个变体")
    
    # 保存变体数据
    os.makedirs("data", exist_ok=True)
    
    # 保存变体评估数据
    with open("data/variations_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(variations_data, f, ensure_ascii=False, indent=2)
    
    # 保存扩充后的数据集
    with open("data/diverse_question_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_variations, f, ensure_ascii=False, indent=2)
    
    print("\n已保存文件：")
    print("  - data/variations_evaluation.json: 变体评估数据")
    print("  - data/diverse_question_dataset.json: 扩充后的多样化问题数据集")
    
    return variations_data


def analyze_variation_results(variations_data: List[Dict], original_dataset: List[Dict]):
    """
    分析变体生成结果
    
    Args:
        variations_data: 变体数据
        original_dataset: 原始数据集
    """
    print("\n===== 变体生成结果分析 =====")
    
    # 评估数据集级别的变体效果
    evaluator = VariationEvaluator()
    dataset_eval = evaluator.evaluate_dataset_variations(original_dataset, variations_data)
    
    # 输出总体统计
    print(f"\n总体统计：")
    print(f"  原始问题数量: {dataset_eval['total_questions']}")
    print(f"  生成变体数量: {dataset_eval['total_variations']}")
    print(f"  平均每个问题生成变体数: {dataset_eval['avg_variations_per_question']:.2f}")
    print(f"  平均变体质量分数: {dataset_eval['overall_quality']:.1f}/100")
    print(f"  问题覆盖率: {dataset_eval['coverage_rate']:.1%}")
    
    # 输出各类别的统计
    print(f"\n各类别变体质量统计：")
    for category, stats in dataset_eval['category_stats'].items():
        print(f"  {category}:")
        print(f"    变体数量: {stats['count']}")
        print(f"    平均质量: {stats['avg_quality']:.1f}/100")
        print(f"    质量标准差: {stats['quality_std']:.1f}")
    
    # 找出质量最高和最低的变体
    quality_scores = [(item['quality_score'], item['original_question']) for item in variations_data]
    if quality_scores:
        best_quality = max(quality_scores, key=lambda x: x[0])
        worst_quality = min(quality_scores, key=lambda x: x[0])
        
        print(f"\n质量最高的变体（{best_quality[0]:.1f}/100）:")
        print(f"  原始问题: {best_quality[1]}")
        
        # 找到对应的变体
        for item in variations_data:
            if item['original_question'] == best_quality[1]:
                print(f"  生成的变体:")
                for i, var in enumerate(item['variations'], 1):
                    print(f"    {i}. {var}")
                break
        
        print(f"\n质量最低的变体（{worst_quality[0]:.1f}/100）:")
        print(f"  原始问题: {worst_quality[1]}")
    
    # 生成可视化
    visualizer = VariationVisualizer("data")
    visualizer.visualize_variation_quality(dataset_eval)
    visualizer.visualize_variation_examples(variations_data, num_examples=3)
    visualizer.visualize_variation_statistics(dataset_eval)
    
    # 提供改进建议
    print(f"\n===== 改进建议 =====")
    
    if dataset_eval['overall_quality'] < 70:
        print("1. 变体质量有待提高，建议：")
        print("   - 增加更多样化的模板")
        print("   - 改进疑问词替换策略")
        print("   - 引入更复杂的句式转换规则")
    
    if dataset_eval['avg_variations_per_question'] < 3:
        print("2. 建议增加每个问题的变体数量，以提高数据集多样性")
    
    # 分析各类别的表现
    for category, stats in dataset_eval['category_stats'].items():
        if stats['avg_quality'] < 65:
            print(f"3. {category}类别的变体质量较低，建议针对该类别优化生成策略")
    
    print("\n总体结论：")
    if dataset_eval['overall_quality'] >= 80:
        print("✓ 变体生成效果优秀，可以有效提高模型对多样化问法的理解能力")
    elif dataset_eval['overall_quality'] >= 60:
        print("✓ 变体生成效果良好，但仍有优化空间")
    else:
        print("! 变体生成效果需要改进，建议调整生成策略")


def main():
    """
    主函数
    """
    print("===== 问题多样化生成功能演示 =====")
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 生成样本问题数据集
    print("\n1. 生成样本问题数据集")
    dataset_generator = DatasetGenerator("data")
    original_dataset = dataset_generator.generate_sample_questions(num_questions=30)
    
    # 2. 演示单个问题的变体生成
    print("\n2. 演示单个问题的变体生成")
    generator = QuestionVariationGenerator()
    
    # 选择一个示例问题
    sample_question = original_dataset[0]['question']
    print(f"\n原始问题: {sample_question}")
    
    # 生成变体
    variations = generator.generate_question_variations(sample_question, n=5)
    print(f"\n生成的5个变体:")
    for i, variation in enumerate(variations, 1):
        print(f"  {i}. {variation}")
    
    # 评估变体质量
    evaluator = VariationEvaluator()
    eval_results = evaluator.evaluate_variations(sample_question, variations)
    print(f"\n变体质量评估:")
    print(f"  质量分数: {eval_results['quality_score']:.1f}/100")
    print(f"  语法有效率: {eval_results['validity_rate']:.1%}")
    print(f"  唯一性: {eval_results['uniqueness_rate']:.1%}")
    print(f"  与原始问题平均相似度: {eval_results['avg_similarity']:.1%}")
    
    # 3. 为整个数据集生成变体
    print("\n3. 为整个数据集生成变体")
    variations_data = generate_diverse_dataset(original_dataset, variations_per_question=3)
    
    # 4. 分析变体生成结果
    analyze_variation_results(variations_data, original_dataset)
    
    # 计算执行时间
    execution_time = time.time() - start_time
    print(f"\n程序执行时间: {execution_time:.2f} 秒")
    
    # 检查执行效率
    if execution_time > 30:
        print("警告: 程序执行时间超过30秒，可能需要优化代码")
    else:
        print("✓ 程序执行效率良好")
    
    print("\n===== 演示完成 =====")
    print("\n关键结论:")
    print("1. 通过多样化问法生成，可以有效扩充训练数据集")
    print("2. 模板替换、疑问词替换和句式转换是有效的变体生成策略")
    print("3. 生成的变体保留了原始问题的语义，同时增加了表达多样性")
    print("4. 建议在实际应用中根据具体任务需求调整变体生成参数")
    print("5. 生成的多样化数据集可以用于训练更鲁棒的问答模型")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()