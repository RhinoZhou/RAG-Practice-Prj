#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版智能医疗助手演示程序

功能：展示医疗信息检索的基本功能，包括知识库管理和简单的问答功能

"""

import os
import gradio as gr
import json

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

# 创建Gradio界面
with gr.Blocks(title="智能医疗助手", theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    # 智能医疗助手演示
    
    **支持功能：**
    - 多知识库管理（创建、删除、查看）
    - 知识库内容管理（添加、查看文件）
    - 基于知识库的简单问答
    """)
    
    with gr.Tab("知识库管理"):
        with gr.Row():
            with gr.Column(scale=1):
                kb_name_input = gr.Textbox(label="新知识库名称")
                create_kb_btn = gr.Button("创建知识库")
                create_result = gr.Textbox(label="创建结果", interactive=False)
                
                delete_kb_dropdown = gr.Dropdown(label="选择要删除的知识库", choices=get_knowledge_bases())
                delete_kb_btn = gr.Button("删除知识库")
                delete_result = gr.Textbox(label="删除结果", interactive=False)
            
            with gr.Column(scale=2):
                kb_dropdown = gr.Dropdown(label="选择知识库", choices=get_knowledge_bases(), value=DEFAULT_KB)
                
                with gr.Row():
                    filename_input = gr.Textbox(label="文件名", value="medical_info.txt")
                    content_input = gr.Textbox(label="文件内容", lines=10, placeholder="输入医疗信息内容...")
                    save_btn = gr.Button("保存到知识库")
                
                save_result = gr.Textbox(label="保存结果", interactive=False)
                
                files_list = gr.Dropdown(label="知识库文件", choices=get_kb_files(DEFAULT_KB))
                view_content_btn = gr.Button("查看文件内容")
                file_content_output = gr.Textbox(label="文件内容", interactive=False, lines=10)
    
    with gr.Tab("医疗问答"):
        with gr.Row():
            with gr.Column(scale=1):
                qa_kb_dropdown = gr.Dropdown(label="选择知识库", choices=get_knowledge_bases(), value=DEFAULT_KB)
            
            with gr.Column(scale=2):
                question_input = gr.Textbox(label="您的问题", placeholder="例如：糖尿病患者应该如何控制饮食？")
                ask_btn = gr.Button("提问")
                answer_output = gr.Textbox(label="回答", interactive=False, lines=10)
    
    # 事件处理
    def update_kb_dropdowns():
        """更新所有知识库下拉菜单"""
        kbs = get_knowledge_bases()
        return [
            gr.Dropdown.update(choices=kbs),
            gr.Dropdown.update(choices=kbs, value=DEFAULT_KB),
            gr.Dropdown.update(choices=kbs, value=DEFAULT_KB)
        ]
    
    def update_files_list(kb_name):
        """更新文件列表"""
        files = get_kb_files(kb_name)
        return gr.Dropdown.update(choices=files)
    
    # 知识库管理事件
    create_kb_btn.click(
        fn=lambda name: [create_knowledge_base(name), *update_kb_dropdowns()],
        inputs=[kb_name_input],
        outputs=[create_result, kb_dropdown, delete_kb_dropdown, qa_kb_dropdown]
    )
    
    delete_kb_btn.click(
        fn=lambda name: [delete_knowledge_base(name), *update_kb_dropdowns(), update_files_list(DEFAULT_KB)],
        inputs=[delete_kb_dropdown],
        outputs=[delete_result, kb_dropdown, delete_kb_dropdown, qa_kb_dropdown, files_list]
    )
    
    kb_dropdown.change(
        fn=update_files_list,
        inputs=[kb_dropdown],
        outputs=[files_list]
    )
    
    save_btn.click(
        fn=save_to_kb,
        inputs=[kb_dropdown, content_input, filename_input],
        outputs=[save_result]
    ).then(
        fn=update_files_list,
        inputs=[kb_dropdown],
        outputs=[files_list]
    )
    
    view_content_btn.click(
        fn=lambda kb, file: read_file_content(kb, file) if file else "请选择一个文件",
        inputs=[kb_dropdown, files_list],
        outputs=[file_content_output]
    )
    
    # 问答事件
    ask_btn.click(
        fn=simple_qa,
        inputs=[question_input, qa_kb_dropdown],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)