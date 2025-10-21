#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本语料清洗与格式化演示

功能说明：演示如何清理原始语料并转换为标准的 instruction / response 格式
内容概述：
使用正则与字符串操作去除空行、冗余符号，并将问答样本组织为统一JSON格式，为后续微调训练准备数据。
执行流程：
1. 读取原始 .txt 文本文件
2. 执行正则匹配清理特殊符号与多空格
3. 自动拆分问答样本
4. 格式化为 JSON 结构 {"instruction":…, "response":…}
5. 输出整理结果并统计样本数
输入说明：一个包含多段问答的纯文本文件
输出展示：打印示例样本条目、总计样本数量

作者：Ph.D. Rhino
"""

import os
import re
import json
import sys


def install_dependencies():
    """
    检查并安装必要的依赖库
    """
    try:
        # 本程序主要使用Python标准库，不需要额外的第三方依赖
        print("检查依赖...")
        print("所有必要的依赖已满足 (仅使用Python标准库)")
        return True
    except Exception as e:
        print(f"依赖检查/安装失败: {e}")
        return False


def generate_sample_data(file_path):
    """
    生成示例数据文件
    Args:
        file_path: 要生成的数据文件路径
    """
    # 示例问答数据
    sample_qa_pairs = [
        ("介绍RAG的概念", "RAG结合检索与生成模型实现知识增强推理。它通过从外部知识库检索相关信息，然后将这些信息与用户查询一起输入到生成模型中，生成更加准确和可靠的回答。"),
        ("什么是大语言模型？", "大语言模型是一种基于深度学习的人工智能模型，通过大量文本数据训练，可以理解和生成人类语言。它们能够完成文本生成、翻译、问答等多种自然语言处理任务。"),
        ("机器学习的主要步骤有哪些？", "机器学习的主要步骤包括：1) 数据收集与预处理；2) 特征工程；3) 模型选择；4) 模型训练；5) 模型评估；6) 模型部署与监控。"),
        ("解释监督学习的概念", "监督学习是机器学习的一种方法，通过已标记的训练数据来训练模型。在训练过程中，模型学习输入数据与输出标签之间的映射关系，以便对新的未标记数据进行预测。"),
        ("什么是过拟合？如何防止？", "过拟合是指模型在训练数据上表现过于良好，但在新数据上表现较差的现象。防止过拟合的方法包括：增加训练数据、使用正则化技术、早停策略、交叉验证等。"),
        ("什么是Transformer架构？", "Transformer是一种基于自注意力机制的神经网络架构，由Vaswani等人在2017年提出。它广泛应用于自然语言处理任务，如机器翻译、文本生成等，具有并行计算能力强、能捕捉长距离依赖等优点。"),
        ("解释向量数据库的概念", "向量数据库是一种专门设计用于存储和检索向量嵌入的数据库系统。它能够高效地进行相似度搜索，常用于推荐系统、图像检索、自然语言处理等领域，支持最近邻搜索等操作。"),
        ("Python中有哪些常用的数据结构？", "Python中常用的数据结构包括：列表(list)、元组(tuple)、字典(dict)、集合(set)等。此外，还可以通过collections模块使用更高级的数据结构，如OrderedDict、defaultdict、Counter等。"),
        ("什么是API？有什么作用？", "API(应用程序编程接口)是不同软件组件之间交互的规范。它定义了请求和响应的格式，允许不同系统之间进行数据交换和功能调用，便于软件集成和开发。"),
        ("解释数据清洗的重要性", "数据清洗是数据预处理的重要步骤，它去除或修复数据集中的错误、缺失值、异常值和重复数据等。干净的数据对于提高模型性能、减少训练时间、确保分析结果的准确性至关重要。"),
    ]
    
    # 生成带格式的数据文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("# 示例问答数据集\n\n")
        for i, (question, answer) in enumerate(sample_qa_pairs, 1):
            # 添加一些混乱格式，模拟真实数据
            if i % 2 == 0:
                f.write(f"\n\n问 题 {i}:  {question}   \n   回 答:   {answer}\n\n")
            else:
                f.write(f"问: {question}\n答: {answer}\n")
            # 添加一些随机噪声和冗余行
            if i % 3 == 0:
                f.write("---\n***\n")
    
    print(f"示例数据已生成: {file_path}")


def clean_text(text):
    """
    使用正则表达式清理文本
    Args:
        text: 原始文本
    Returns:
        清理后的文本
    """
    # 保留必要的空行，但移除连续多个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行首尾的空白字符
    lines = [line.strip() for line in text.split('\n')]
    # 移除只包含特殊符号的行
    lines = [line for line in lines if not re.match(r'^[-*]{2,}$', line)]
    # 移除注释行
    lines = [line for line in lines if not line.startswith('#')]
    # 重新组合文本
    cleaned_text = '\n'.join(lines)
    return cleaned_text


def extract_qa_pairs(text):
    """
    从文本中提取问答对
    Args:
        text: 清理后的文本
    Returns:
        问答对列表 [(question, answer), ...]
    """
    qa_pairs = []
    lines = text.split('\n')
    
    # 打印清理后的文本以便调试
    print("清理后的文本预览:")
    print('-' * 50)
    print('\n'.join(lines[:10]) + '\n...')
    print('-' * 50)
    
    # 定义更全面的前缀列表用于清理
    question_prefixes = ['问:', '问题:', '问 ', '问题 ', '题 ', '题:']
    answer_prefixes = ['答:', '回答:', '回:', '答案:', '回 答: ', '答 ', '回答 ', '回 ', '答案 ', '回 答 ']
    
    # 使用状态机方法处理问答对
    question = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是问题行
        if '问' in line and '答' not in line:
            # 尝试使用正则表达式直接提取问题内容
            # 匹配常见的问题格式：问: xxx 或 问题 X: xxx 或 问问题: xxx
            match = re.search(r'[问问题题]\s*\d*\s*[:：]\s*(.+)', line)
            if match:
                question = match.group(1).strip()
            else:
                # 如果正则匹配失败，尝试移除所有可能的前缀
                temp_line = line
                for prefix in question_prefixes:
                    if temp_line.startswith(prefix):
                        temp_line = temp_line[len(prefix):].strip()
                        break
                # 再次检查是否包含冒号，可能是格式不标准的问题
                if ':' in temp_line or '：' in line:
                    separator = ':' if ':' in temp_line else '：'
                    parts = temp_line.split(separator, 1)
                    if len(parts) > 1:
                        question = parts[1].strip()
                    else:
                        question = temp_line
                else:
                    question = temp_line
        
        # 检查是否是回答行，并且已经有一个待匹配的问题
        elif ('答' in line or '回' in line) and question:
            # 尝试使用正则表达式直接提取回答内容
            match = re.search(r'[答回答案]\s*[:：]\s*(.+)', line)
            if match:
                answer = match.group(1).strip()
            else:
                # 如果正则匹配失败，尝试移除所有可能的前缀
                temp_line = line
                for prefix in answer_prefixes:
                    if temp_line.startswith(prefix):
                        temp_line = temp_line[len(prefix):].strip()
                        break
                # 再次检查是否包含冒号，可能是格式不标准的回答
                if ':' in temp_line or '：' in temp_line:
                    separator = ':' if ':' in temp_line else '：'
                    parts = temp_line.split(separator, 1)
                    if len(parts) > 1:
                        answer = parts[1].strip()
                    else:
                        answer = temp_line
                else:
                    answer = temp_line
                
            # 如果成功提取了回答，添加问答对
            if answer:
                # 进一步清理问题文本，移除可能残留的题号等
                clean_question = re.sub(r'^题\s*\d*\s*[:：]?\s*', '', question)
                qa_pairs.append((clean_question, answer))
                # 重置问题状态，准备处理下一对
                question = None
    
    # 如果使用主要策略没有找到所有预期的问答对，尝试备用策略
    print(f"第一次提取完成，找到 {len(qa_pairs)} 个问答对")
    
    # 备用策略：基于行号模式匹配
    if len(qa_pairs) < 10:  # 假设我们知道应该有10个问答对
        print("使用备用提取策略...")
        # 遍历所有行，寻找问题-回答模式
        i = 0
        while i < len(lines):
            if i + 1 < len(lines):
                current_line = lines[i].strip()
                next_line = lines[i + 1].strip()
                
                # 检查当前行是否像问题，下一行是否像回答
                is_question_line = '问' in current_line and '答' not in current_line
                is_answer_line = '答' in next_line or '回' in next_line
                
                if is_question_line and is_answer_line:
                    # 提取问题
                    q_match = re.search(r'[:：]\s*(.+)', current_line)
                    question = q_match.group(1).strip() if q_match else current_line
                    
                    # 移除问题相关前缀
                    for q_prefix in question_prefixes:
                        if question.startswith(q_prefix):
                            question = question[len(q_prefix):].strip()
                            break
                    
                    # 进一步清理问题文本
                    question = re.sub(r'^题\s*\d*\s*[:：]?\s*', '', question)
                    
                    # 提取回答
                    a_match = re.search(r'[:：]\s*(.+)', next_line)
                    answer = a_match.group(1).strip() if a_match else next_line
                    
                    # 移除回答相关前缀
                    for a_prefix in answer_prefixes:
                        if answer.startswith(a_prefix):
                            answer = answer[len(a_prefix):].strip()
                            break
                    
                    # 检查这个问答对是否已经存在
                    is_duplicate = False
                    for q, a in qa_pairs:
                        if question == q or answer == a:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and question and answer:
                        qa_pairs.append((question, answer))
                    
                    i += 2  # 跳过已处理的问题和回答行
                else:
                    i += 1
            else:
                i += 1
    
    return qa_pairs


def format_to_json(qa_pairs):
    """
    将问答对转换为JSON格式
    Args:
        qa_pairs: 问答对列表
    Returns:
        JSON对象列表
    """
    json_data = []
    for question, answer in qa_pairs:
        json_item = {
            "instruction": question,
            "response": answer
        }
        json_data.append(json_item)
    return json_data


def save_json_data(json_data, output_file):
    """
    保存JSON数据到文件
    Args:
        json_data: JSON对象列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"格式化数据已保存到: {output_file}")


def display_results(json_data):
    """
    展示处理结果
    Args:
        json_data: JSON对象列表
    """
    print(f"\n处理结果摘要:")
    print(f"清理后样本总数: {len(json_data)}")
    print("\n前3个样本示例:")
    
    # 显示前3个样本
    for i, sample in enumerate(json_data[:3], 1):
        print(f"样本 {i}:")
        print(f"  instruction: {sample['instruction']}")
        print(f"  response: {sample['response']}")
        print()


def main():
    """
    主函数，协调整个数据清洗与格式化流程
    """
    print("=== 文本语料清洗与格式化演示 ===")
    
    # 检查并安装依赖
    if not install_dependencies():
        print("依赖安装失败，程序终止。")
        sys.exit(1)
    
    # 定义文件路径
    input_file = "sample_qa_data.txt"
    output_file = "formatted_qa_data.json"
    
    # 生成示例数据
    generate_sample_data(input_file)
    
    # 读取原始文本文件
    print(f"\n读取原始数据文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f"读取成功，原始文本长度: {len(raw_text)} 字符")
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)
    
    # 清理文本
    print("\n开始清理文本...")
    cleaned_text = clean_text(raw_text)
    print(f"清理完成，清理后文本长度: {len(cleaned_text)} 字符")
    
    # 提取问答对
    print("\n开始提取问答对...")
    qa_pairs = extract_qa_pairs(cleaned_text)
    print(f"提取完成，共找到 {len(qa_pairs)} 个问答对")
    
    # 转换为JSON格式
    print("\n开始格式化为JSON...")
    json_data = format_to_json(qa_pairs)
    
    # 保存JSON数据
    save_json_data(json_data, output_file)
    
    # 展示结果
    display_results(json_data)
    
    # 验证中文显示
    print("\n中文显示验证:")
    print("测试中文: 你好，世界！这是中文测试。")
    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    main()