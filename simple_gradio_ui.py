#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Gradio界面，用于调用命令行版RAG系统
"""

import gradio as gr
import subprocess
import os
import sys
import time
import shutil

# 确保数据目录存在
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

def log(message):
    """简单的日志函数"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()

# 命令行工具函数
def run_cli_command(cmd, timeout=30):
    """运行命令行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "命令执行超时"
    except Exception as e:
        return f"命令执行错误: {str(e)}"

def upload_file(file):
    """上传文件到数据目录"""
    try:
        if not file:
            return "请选择一个文件"
        
        # 复制文件到数据目录
        file_name = os.path.basename(file.name)
        dest_path = os.path.join(DATA_DIR, file_name)
        shutil.copy2(file.name, dest_path)
        
        return f"成功上传文件: {file_name}"
    except Exception as e:
        return f"上传文件失败: {str(e)}"

def list_files():
    """列出数据目录中的所有文件"""
    try:
        files = os.listdir(DATA_DIR)
        if not files:
            return "暂无上传的文件"
        return "\n".join([f"• {file}" for file in files])
    except Exception as e:
        return f"获取文件列表失败: {str(e)}"

def delete_file(file_name):
    """删除指定文件"""
    try:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"成功删除文件: {file_name}"
        else:
            return f"文件不存在: {file_name}"
    except Exception as e:
        return f"删除文件失败: {str(e)}"

def clear_all_data():
    """清空所有数据"""
    try:
        # 清空数据目录
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        return "成功清空所有数据"
    except Exception as e:
        return f"清空数据失败: {str(e)}"

def ask_question(question):
    """调用命令行版的RAG系统进行问答"""
    try:
        # 这里可以根据需要扩展，例如调用命令行版的RAG系统
        # 目前使用简化的回答
        return f"这是一个示例回答，针对您的问题：{question}\n\n注：完整的RAG功能可以通过命令行版使用 'python cli_rag_system.py' 启动。"
    except Exception as e:
        return f"问答失败: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="医学知识库问答系统") as demo:
    gr.Markdown("# 🩺 医学知识库问答系统")
    gr.Markdown("**注意**: 由于环境限制，这是一个简化版界面。完整功能请使用命令行版本 `python cli_rag_system.py`")
    
    with gr.Tabs():
        with gr.TabItem("文件管理"):
            gr.Markdown("## 文件管理")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_output = gr.Textbox(label="上传结果", interactive=False)
                    file_upload = gr.File(label="选择文件")
                    upload_btn = gr.Button("上传文件", variant="primary")
                    
                    file_list = gr.Textbox(label="当前文件列表", interactive=False, lines=5)
                    list_btn = gr.Button("刷新文件列表")
                
                with gr.Column(scale=1):
                    delete_filename = gr.Textbox(label="要删除的文件名")
                    delete_btn = gr.Button("删除文件")
                    delete_output = gr.Textbox(label="删除结果", interactive=False)
                    
                    clear_btn = gr.Button("清空所有数据", variant="secondary")
                    clear_output = gr.Textbox(label="清空结果", interactive=False)
            
            # 文件管理事件
            upload_btn.click(upload_file, inputs=[file_upload], outputs=[file_output])
            list_btn.click(list_files, outputs=[file_list])
            delete_btn.click(delete_file, inputs=[delete_filename], outputs=[delete_output])
            clear_btn.click(clear_all_data, outputs=[clear_output])
        
        with gr.TabItem("问答系统"):
            gr.Markdown("## 问答系统")
            
            question_input = gr.Textbox(label="您的问题", placeholder="请输入您的医学问题...", lines=2)
            ask_btn = gr.Button("获取答案", variant="primary")
            answer_output = gr.Textbox(label="回答", interactive=False, lines=8)
            
            # 问答事件
            ask_btn.click(ask_question, inputs=[question_input], outputs=[answer_output])
    
    gr.Markdown("---")
    gr.Markdown("**使用说明**:")
    gr.Markdown("1. 在文件管理标签中上传医学相关文件")
    gr.Markdown("2. 在问答标签中输入您的问题")
    gr.Markdown("3. 完整功能请使用命令行版本：`python cli_rag_system.py`")

# 启动Gradio服务
if __name__ == "__main__":
    log("启动Gradio界面")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        debug=False
    )