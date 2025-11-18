#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版医学RAG系统 - 稳定版本
"""

import time
import sys
import os
import shutil
from datetime import datetime

def log(message):
    """带时间戳的日志函数"""
    timestamp = time.strftime("[%H:%M:%S]", time.localtime())
    print(f"{timestamp} {message}")
    sys.stdout.flush()

# 程序启动
log("程序启动")

# 基础库导入
try:
    import torch
    log("✓ 导入 torch 成功")
except Exception as e:
    log(f"✗ 导入 torch 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import faiss
    log("✓ 导入 faiss 成功")
except Exception as e:
    log(f"✗ 导入 faiss 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import numpy as np
    log("✓ 导入 numpy 成功")
except Exception as e:
    log(f"✗ 导入 numpy 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import pandas as pd
    log("✓ 导入 pandas 成功")
except Exception as e:
    log(f"✗ 导入 pandas 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import nltk
    log("✓ 导入 nltk 成功")
except Exception as e:
    log(f"✗ 导入 nltk 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    log("✓ 导入 tqdm 成功")
except Exception as e:
    log(f"✗ 导入 tqdm 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import PyPDF2
    log("✓ 导入 PyPDF2 成功")
except Exception as e:
    log(f"✗ 导入 PyPDF2 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import docx
    log("✓ 导入 docx 成功")
except Exception as e:
    log(f"✗ 导入 docx 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# llama_index 相关库导入
try:
    from llama_index.core.node_parser import SentenceSplitter
    log("✓ 导入 SentenceSplitter 成功")
except Exception as e:
    log(f"✗ 导入 SentenceSplitter 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    log("✓ 导入 VectorStoreIndex 和 StorageContext 成功")
except Exception as e:
    log(f"✗ 导入 VectorStoreIndex 和 StorageContext 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import load_index_from_storage
    log("✓ 导入 load_index_from_storage 成功")
except Exception as e:
    log(f"✗ 导入 load_index_from_storage 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.vector_stores.faiss import FaissVectorStore
    log("✓ 导入 FaissVectorStore 成功")
except Exception as e:
    log(f"✗ 导入 FaissVectorStore 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# gradio 导入
log("开始导入 gradio...")
try:
    import gradio as gr
    log(f"✓ 导入 gradio 成功！版本: {gr.__version__}")
except Exception as e:
    log(f"✗ gradio 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log("所有必要库导入完成！")

# 全局变量
data_path = "./data"
index_path = "./index"
current_index = None
file_indexes = {}  # 存储每个文件的索引

# 创建必要的目录
os.makedirs(data_path, exist_ok=True)
os.makedirs(index_path, exist_ok=True)

# 简单的文本处理函数
def read_pdf(file_path):
    """读取PDF文件内容"""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        log(f"读取PDF文件失败: {e}")
        return ""

def read_docx(file_path):
    """读取Word文件内容"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        log(f"读取Word文件失败: {e}")
        return ""

def read_txt(file_path):
    """读取TXT文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()
    except Exception as e:
        log(f"读取TXT文件失败: {e}")
        return ""

def process_file(file_obj):
    """处理上传的文件"""
    try:
        # 保存文件
        file_path = os.path.join(data_path, file_obj.name)
        with open(file_path, 'wb') as f:
            f.write(file_obj.read())
        
        # 读取文件内容
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            content = read_pdf(file_path)
        elif file_ext == '.docx':
            content = read_docx(file_path)
        elif file_ext == '.txt':
            content = read_txt(file_path)
        else:
            return f"不支持的文件格式: {file_ext}", False
        
        if not content:
            return f"文件内容为空或无法读取: {file_obj.name}", False
        
        return f"成功上传并处理文件: {file_obj.name}", True
    except Exception as e:
        return f"处理文件失败: {str(e)}", False

def delete_file(file_name):
    """删除指定文件"""
    try:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            # 同时删除对应的索引
            if file_name in file_indexes:
                del file_indexes[file_name]
            return f"成功删除文件: {file_name}", True
        else:
            return f"文件不存在: {file_name}", False
    except Exception as e:
        return f"删除文件失败: {str(e)}", False

def list_files():
    """列出所有上传的文件"""
    try:
        files = os.listdir(data_path)
        if not files:
            return "暂无上传的文件"
        else:
            return "\n".join([f"- {file}" for file in files])
    except Exception as e:
        return f"获取文件列表失败: {str(e)}"

def clear_all_data():
    """清空所有数据"""
    try:
        # 清空数据目录
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)
        
        # 清空索引目录
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
        
        # 清空全局变量
        global current_index, file_indexes
        current_index = None
        file_indexes = {}
        
        return "成功清空所有数据", True
    except Exception as e:
        return f"清空数据失败: {str(e)}", False

def main():
    """主函数，创建Gradio界面"""
    log("开始创建Gradio界面")
    
    with gr.Blocks(title="医学知识库问答系统") as demo:
        # 页面标题
        gr.Markdown("# 🩺 医学知识库问答系统")
        gr.Markdown("基于RAG技术的智能医学问答平台")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 文件管理")
                
                # 文件上传组件
                file_input = gr.File(
                    label="上传医学文献（支持PDF、DOCX、TXT）",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="binary"
                )
                
                upload_btn = gr.Button("📤 上传并处理")
                upload_output = gr.Textbox(
                    label="上传结果",
                    interactive=False,
                    placeholder="上传结果将显示在这里..."
                )
                
                # 文件列表和删除功能
                list_btn = gr.Button("📋 列出所有文件")
                file_list_output = gr.Textbox(
                    label="文件列表",
                    interactive=False,
                    placeholder="文件列表将显示在这里..."
                )
                
                delete_file_input = gr.Textbox(
                    label="输入要删除的文件名",
                    placeholder="例如：document.pdf"
                )
                delete_btn = gr.Button("🗑️ 删除文件")
                delete_output = gr.Textbox(
                    label="删除结果",
                    interactive=False,
                    placeholder="删除结果将显示在这里..."
                )
                
                # 清空所有数据
                clear_btn = gr.Button("🧹 清空所有数据", variant="danger")
                clear_output = gr.Textbox(
                    label="清空结果",
                    interactive=False,
                    placeholder="清空结果将显示在这里..."
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 💬 问答功能")
                
                # 对话历史记录
                chat_history = gr.Chatbot(
                    label="对话历史",
                    height=400,
                    placeholder="开始与医学知识库对话..."
                )
                
                # 问题输入框
                question_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="例如：什么是高血压？",
                    lines=2
                )
                
                # 提问按钮
                ask_btn = gr.Button("❓ 提问", variant="primary")
                
                # 回答输出框
                answer_output = gr.Textbox(
                    label="回答",
                    interactive=False,
                    placeholder="回答将显示在这里..."
                )
        
        # 绑定事件处理函数
        upload_btn.click(
            fn=process_file,
            inputs=[file_input],
            outputs=[upload_output]
        )
        
        list_btn.click(
            fn=list_files,
            inputs=[],
            outputs=[file_list_output]
        )
        
        delete_btn.click(
            fn=delete_file,
            inputs=[delete_file_input],
            outputs=[delete_output]
        )
        
        clear_btn.click(
            fn=clear_all_data,
            inputs=[],
            outputs=[clear_output]
        )
        
        # 简单的问答函数
        def simple_qa(question, history):
            """简单的问答函数，仅用于演示"""
            answer = f"这是一个示例回答，针对您的问题：{question}\n\n注：当前使用的是简化版本，完整功能需要transformers和sentence_transformers库支持。"
            return answer, history + [[question, answer]]
        
        ask_btn.click(
            fn=simple_qa,
            inputs=[question_input, chat_history],
            outputs=[answer_output, chat_history]
        )
    
    # 启动Gradio界面
    log("准备启动Gradio界面")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            debug=True
        )
    except Exception as e:
        log(f"启动Gradio界面失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    log("程序开始执行")
    main()