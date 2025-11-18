#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行版智能医疗助手演示程序

功能：展示医疗信息检索的基本功能，包括知识库管理和简单的问答功能

"""

import os
import json
import time

# 配置设置
KB_BASE_DIR = "knowledge_bases"
DEFAULT_KB = "default"
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)

# 确保知识库目录存在
os.makedirs(KB_BASE_DIR, exist_ok=True)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 简单的知识库管理函数
def get_knowledge_bases() -> list:
    """获取所有知识库名称"""
    try:
        return [d for d in os.listdir(KB_BASE_DIR) if os.path.isdir(os.path.join(KB_BASE_DIR, d))]
    except Exception as e:
        print(f"获取知识库列表失败: {e}")
        return [DEFAULT_KB]

def create_knowledge_base(kb_name: str) -> str:
    """创建新的知识库"""
    try:
        if not kb_name:
            return "知识库名称不能为空"
        
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 已存在"
        
        os.makedirs(kb_path)
        return f"成功创建知识库: {kb_name}"
    except Exception as e:
        return f"创建知识库失败: {str(e)}"

def delete_knowledge_base(kb_name: str) -> str:
    """删除知识库"""
    try:
        if kb_name == DEFAULT_KB:
            return "默认知识库不能删除"
        
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 不存在"
        
        # 删除所有文件
        for file in os.listdir(kb_path):
            os.remove(os.path.join(kb_path, file))
        # 删除目录
        os.rmdir(kb_path)
        return f"成功删除知识库: {kb_name}"
    except Exception as e:
        return f"删除知识库失败: {str(e)}"

def get_kb_files(kb_name: str) -> list:
    """获取知识库中的所有文件"""
    try:
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return []
        return [f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))]
    except Exception as e:
        print(f"获取知识库文件列表失败: {e}")
        return []

def read_file_content(kb_name: str, filename: str) -> str:
    """读取文件内容"""
    try:
        file_path = os.path.join(KB_BASE_DIR, kb_name, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def save_to_kb(kb_name: str, content: str, filename: str = "new_file.txt") -> str:
    """保存内容到知识库文件"""
    try:
        if not content.strip():
            return "内容不能为空"
        
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        file_path = os.path.join(kb_path, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"成功保存到文件: {filename}"
    except Exception as e:
        return f"保存文件失败: {str(e)}"

# 简单的问答功能
def simple_qa(question: str, kb_name: str = DEFAULT_KB) -> str:
    """基于知识库内容的简单问答"""
    if not question.strip():
        return "请输入您的问题"
    
    try:
        # 获取知识库中的所有文件
        files = get_kb_files(kb_name)
        if not files:
            return "当前知识库为空，请先添加内容"
        
        # 简单的关键词匹配
        question_lower = question.lower()
        relevant_content = []
        
        for file in files:
            content = read_file_content(kb_name, file)
            content_lower = content.lower()
            
            # 检查是否包含相关关键词
            if any(keyword in content_lower for keyword in ["糖尿病", "高血压", "饮食", "治疗", "症状", "血糖", "血压"]):
                relevant_content.append(f"[{file}]: {content}")
        
        if relevant_content:
            return "\n\n".join(relevant_content)
        else:
            return "在知识库中未找到相关信息"
    except Exception as e:
        return f"回答问题时出错: {str(e)}"

# 主函数
def main():
    print("=" * 50)
    print("智能医疗助手演示程序")
    print("=" * 50)
    print()
    
    # 创建示例知识库内容
    print("正在创建示例知识库内容...")
    
    # 创建糖尿病信息
    diabetes_content = "糖尿病是一种慢性疾病，患者需要控制饮食和血糖水平。\n\n" \
                     "饮食建议：\n" \
                     "1. 减少糖分摄入\n" \
                     "2. 增加膳食纤维\n" \
                     "3. 控制碳水化合物摄入量\n" \
                     "4. 定期监测血糖"
    
    # 创建高血压信息
    hypertension_content = "高血压患者应该减少盐的摄入，定期测量血压。\n\n" \
                          "饮食建议：\n" \
                          "1. 每日盐摄入量不超过5克\n" \
                          "2. 多吃蔬菜水果\n" \
                          "3. 减少高脂肪食物摄入\n" \
                          "4. 适量运动"
    
    # 保存示例内容
    save_to_kb(DEFAULT_KB, diabetes_content, "diabetes.txt")
    save_to_kb(DEFAULT_KB, hypertension_content, "hypertension.txt")
    
    print("示例知识库内容创建完成！")
    print()
    
    # 显示当前知识库信息
    print("当前知识库:")
    kbs = get_knowledge_bases()
    for kb in kbs:
        print(f"- {kb}")
        files = get_kb_files(kb)
        for file in files:
            print(f"  └─ {file}")
    
    print()
    print("=" * 50)
    print("演示问答功能")
    print("=" * 50)
    print()
    
    # 示例问答
    example_questions = [
        "糖尿病患者应该如何控制饮食？",
        "高血压患者需要注意什么？"
    ]
    
    for i, question in enumerate(example_questions, 1):
        print(f"问题 {i}: {question}")
        print("正在回答...")
        time.sleep(1)  # 模拟处理时间
        answer = simple_qa(question, DEFAULT_KB)
        print(f"回答: {answer}")
        print()
    
    print("=" * 50)
    print("演示完成！")
    print("=" * 50)
    print()
    print("您可以在knowledge_bases目录下查看和编辑知识库内容。")
    print("使用 'python demo_cli.py' 命令可以再次运行此演示。")

if __name__ == "__main__":
    main()