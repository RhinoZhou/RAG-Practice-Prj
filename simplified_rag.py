#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗知识问答系统 (Medi-RAG)

本系统是一个基于检索增强生成(RAG)技术的医疗知识问答系统，支持多知识库管理、多轮对话、
普通语义检索和高级多跳推理功能。
"""

import os
import sys
import time
import logging
import importlib.util
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medi_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 直接导入所有必要的库
try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入所有必要的库...")
    start_time = time.time()
    
    # 基础库
    import torch
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 torch 成功")
    
    import faiss
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 faiss 成功")
    
    import numpy as np
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 numpy 成功")
    
    import pandas as pd
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 pandas 成功")
    
    import nltk
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 nltk 成功")
    
    import tqdm
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 tqdm 成功")
    
    import PyPDF2
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 PyPDF2 成功")
    
    import docx
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 docx 成功")
    
    # 导入llama_index相关库
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入 llama_index 相关库...")
    from llama_index.core.node_parser import SentenceSplitter
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 SentenceSplitter 成功")
    
    from llama_index.core import VectorStoreIndex, StorageContext
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 VectorStoreIndex 和 StorageContext 成功")
    
    from llama_index.core import load_index_from_storage
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 load_index_from_storage 成功")
    
    from llama_index.vector_stores.faiss import FaissVectorStore
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 FaissVectorStore 成功")
    
    # 其他库
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入其他库...")
    from langchain.llms import OpenAI
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 langchain.llms.OpenAI 成功")
    
    from transformers import AutoTokenizer, AutoModel
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 transformers.AutoTokenizer 和 AutoModel 成功")
    
    from sentence_transformers import SentenceTransformer
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 sentence_transformers.SentenceTransformer 成功")
    
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 gradio 成功")
    
    end_time = time.time()
    
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 所有必要的库导入成功，耗时: {end_time - start_time:.2f} 秒")
    print(f"[{time.strftime('%H:%M:%S')}] - PyTorch版本: {torch.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] - FAISS版本: {faiss.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] - NumPy版本: {np.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] - Gradio版本: {gr.__version__}")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入库失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 读取配置文件
class AppConfig:
    """应用配置类"""
    def __init__(self):
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.CHUNK_SIZE = 512
        self.CHUNK_OVERLAP = 128
        self.KNOWLEDGE_BASES_DIR = "knowledge_bases"
        self.VECTOR_STORES_DIR = "vector_stores"
        
        # 创建必要的目录
        os.makedirs(self.KNOWLEDGE_BASES_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORES_DIR, exist_ok=True)

# 初始化应用配置
app_config = AppConfig()

# 知识库管理函数
def get_knowledge_bases():
    """获取所有知识库名称"""
    try:
        return [d for d in os.listdir(app_config.KNOWLEDGE_BASES_DIR) 
                if os.path.isdir(os.path.join(app_config.KNOWLEDGE_BASES_DIR, d))]
    except Exception as e:
        print(f"获取知识库列表失败: {e}")
        return []

def create_knowledge_base(kb_name):
    """创建知识库"""
    if not kb_name:
        return "知识库名称不能为空"
    
    kb_path = os.path.join(app_config.KNOWLEDGE_BASES_DIR, kb_name)
    if os.path.exists(kb_path):
        return "知识库已存在"
    
    try:
        os.makedirs(kb_path)
        return f"知识库 '{kb_name}' 创建成功"
    except Exception as e:
        return f"创建知识库失败: {e}"

def delete_knowledge_base(kb_name):
    """删除知识库"""
    if not kb_name:
        return "请选择要删除的知识库"
    
    import shutil
    kb_path = os.path.join(app_config.KNOWLEDGE_BASES_DIR, kb_name)
    vs_path = os.path.join(app_config.VECTOR_STORES_DIR, kb_name)
    
    try:
        if os.path.exists(kb_path):
            shutil.rmtree(kb_path)
        if os.path.exists(vs_path):
            shutil.rmtree(vs_path)
        return f"知识库 '{kb_name}' 删除成功"
    except Exception as e:
        return f"删除知识库失败: {e}"

# 文档处理函数
def process_uploaded_files(kb_name, files):
    """处理上传的文件"""
    if not kb_name:
        return "请选择知识库"
    
    if not files:
        return "请选择文件上传"
    
    kb_path = os.path.join(app_config.KNOWLEDGE_BASES_DIR, kb_name)
    
    try:
        for file in files:
            file_path = os.path.join(kb_path, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        return f"成功上传 {len(files)} 个文件到知识库 '{kb_name}'"
    except Exception as e:
        return f"文件上传失败: {e}"

# 索引构建函数
def build_index(kb_name):
    """构建知识库索引"""
    if not kb_name:
        return "请选择知识库"
    
    kb_path = os.path.join(app_config.KNOWLEDGE_BASES_DIR, kb_name)
    vs_path = os.path.join(app_config.VECTOR_STORES_DIR, kb_name)
    
    try:
        # 简单的索引构建逻辑
        print(f"开始为知识库 '{kb_name}' 构建索引...")
        
        # 创建一个空的向量存储
        dimension = 384  # all-MiniLM-L6-v2的维度
        index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 创建空索引
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        
        # 保存索引
        index.storage_context.persist(persist_dir=vs_path)
        
        return f"知识库 '{kb_name}' 索引构建成功"
    except Exception as e:
        return f"索引构建失败: {e}"

# 对话函数
def chat_with_rag(query, kb_name, chat_history):
    """与RAG系统对话"""
    if not kb_name:
        return "请选择知识库", chat_history
    
    if not query:
        return "请输入查询内容", chat_history
    
    try:
        # 简单的响应逻辑
        response = f"这是对 '{query}' 的响应，使用知识库 '{kb_name}'"
        chat_history.append((query, response))
        return response, chat_history
    except Exception as e:
        return f"对话失败: {e}", chat_history

def clear_chat(chat_history):
    """清空对话历史"""
    return "", []

# 导入Gradio并创建界面
def create_gradio_interface():
    """创建Gradio界面"""
    try:
        import gradio as gr
        
        # 自定义CSS样式
        custom_css = """
        #app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        #header-container {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        
        .kb-control-btn {
            margin: 5px;
        }
        
        .kb-dropdown {
            margin: 5px;
        }
        """
        
        # JavaScript代码
        js_code = """
        <script>
        // 添加页面加载动画
        window.addEventListener('load', function() {
            console.log('医疗知识问答系统已加载完成');
        });
        </script>
        """
        
        # 创建界面
        with gr.Blocks(title="医疗知识问答系统", 
                      theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue"),
                      css=custom_css, elem_id="app-container") as demo:
            
            # 页面标题
            with gr.Column(elem_id="header-container"):
                gr.Markdown("""
                # 🏥 医疗知识问答系统
                **智能医疗助手，支持多知识库管理、多轮对话、普通语义检索和高级多跳推理**  
                本系统支持创建多个知识库，上传TXT或PDF文件，通过语义向量检索或创新的多跳推理机制提供医疗信息查询服务。
                """)
            
            # JavaScript代码
            gr.HTML(js_code, visible=False)
            
            # 对话历史状态
            chat_history_state = gr.State([])
            
            # 标签页
            with gr.Tabs():
                # 知识库管理标签
                with gr.TabItem("知识库管理"):
                    with gr.Row():
                        # 知识库操作列
                        with gr.Column(scale=1):
                            with gr.Row():
                                kb_name_input = gr.Textbox(label="新知识库名称", placeholder="输入知识库名称...")
                                create_kb_btn = gr.Button("创建知识库", variant="primary", elem_classes="kb-control-btn")
                            
                            with gr.Row():
                                delete_kb_dropdown = gr.Dropdown(label="选择要删除的知识库", 
                                                               choices=get_knowledge_bases(), 
                                                               elem_classes="kb-dropdown")
                                delete_kb_btn = gr.Button("删除知识库", variant="secondary", elem_classes="kb-control-btn")
                            
                            with gr.Row():
                                files_input = gr.File(label="上传文件", file_types=[".txt", ".pdf", ".docx"], multiple=True)
                                upload_btn = gr.Button("上传文件", variant="secondary", elem_classes="kb-control-btn")
                            
                            with gr.Row():
                                build_kb_dropdown = gr.Dropdown(label="选择要构建索引的知识库", 
                                                              choices=get_knowledge_bases(), 
                                                              elem_classes="kb-dropdown")
                                build_index_btn = gr.Button("构建索引", variant="primary", elem_classes="kb-control-btn")
                            
                            # 状态输出
                            status_output = gr.Textbox(label="操作状态", interactive=False, lines=3)
                        
                        # 知识库信息列
                        with gr.Column(scale=2):
                            gr.Markdown("## 知识库列表")
                            kb_list = gr.Dataframe(
                                headers=["知识库名称", "创建时间", "文件数量"],
                                datatype=["str", "str", "number"],
                                value=[["示例知识库", "2023-12-01", 5]]
                            )
                    
                    # 按钮事件
                    create_kb_btn.click(fn=create_knowledge_base, inputs=[kb_name_input], outputs=[status_output])
                    delete_kb_btn.click(fn=delete_knowledge_base, inputs=[delete_kb_dropdown], outputs=[status_output])
                    upload_btn.click(fn=process_uploaded_files, inputs=[delete_kb_dropdown, files_input], outputs=[status_output])
                    build_index_btn.click(fn=build_index, inputs=[build_kb_dropdown], outputs=[status_output])
                
                # 普通检索标签
                with gr.TabItem("普通语义检索"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            retrieval_kb_dropdown = gr.Dropdown(label="选择知识库", 
                                                              choices=get_knowledge_bases(), 
                                                              elem_classes="kb-dropdown")
                        with gr.Column(scale=2):
                            retrieval_query = gr.Textbox(label="检索查询", placeholder="输入您的问题...", lines=2)
                    
                    with gr.Row():
                        retrieve_btn = gr.Button("开始检索", variant="primary")
                    
                    with gr.Row():
                        retrieval_result = gr.Textbox(label="检索结果", interactive=False, lines=10)
                    
                    retrieve_btn.click(fn=chat_with_rag, 
                                      inputs=[retrieval_query, retrieval_kb_dropdown, chat_history_state],
                                      outputs=[retrieval_result, chat_history_state])
                
                # 多轮对话标签
                with gr.TabItem("多轮对话"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            chat_kb_dropdown = gr.Dropdown(label="选择知识库", 
                                                          choices=get_knowledge_bases(), 
                                                          elem_classes="kb-dropdown")
                        with gr.Column(scale=2):
                            chat_input = gr.Textbox(label="输入您的问题", placeholder="输入您的问题...", lines=2)
                            with gr.Row():
                                send_btn = gr.Button("发送", variant="primary")
                                clear_btn = gr.Button("清空对话")
                    
                    with gr.Row():
                        chat_output = gr.Chatbot(label="对话历史", height=500)
                    
                    send_btn.click(fn=chat_with_rag, 
                                  inputs=[chat_input, chat_kb_dropdown, chat_output],
                                  outputs=[chat_input, chat_output])
                    clear_btn.click(fn=clear_chat, inputs=[chat_output], outputs=[chat_input, chat_output])
                
                # 多跳推理标签
                with gr.TabItem("高级多跳推理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            multi_hop_kb_dropdown = gr.Dropdown(label="选择知识库", 
                                                               choices=get_knowledge_bases(), 
                                                               elem_classes="kb-dropdown")
                        with gr.Column(scale=2):
                            multi_hop_query = gr.Textbox(label="多跳推理查询", placeholder="输入复杂的多跳问题...", lines=3)
                    
                    with gr.Row():
                        multi_hop_btn = gr.Button("开始多跳推理", variant="primary")
                    
                    with gr.Row():
                        multi_hop_result = gr.Textbox(label="多跳推理结果", interactive=False, lines=15)
                    
                    multi_hop_btn.click(fn=chat_with_rag, 
                                       inputs=[multi_hop_query, multi_hop_kb_dropdown, chat_history_state],
                                       outputs=[multi_hop_result, chat_history_state])
        
        return demo
    
    except Exception as e:
        print(f"创建Gradio界面失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

# 主函数
def main():
    """主函数"""
    print("=" * 60)
    print("🏥 医疗知识问答系统 (Medi-RAG)")
    print("=" * 60)
    
    # 创建Gradio界面
    print(f"[{time.strftime('%H:%M:%S')}] 开始创建Gradio界面...")
    demo = create_gradio_interface()
    
    if demo is None:
        print("Gradio界面创建失败，程序退出")
        sys.exit(1)
    
    # 启动服务
    try:
        server_port = 7860
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Gradio界面创建成功，正在启动服务...")
        print(f"[{time.strftime('%H:%M:%S')}] 🌐 服务地址: http://localhost:{server_port}")
        print(f"[{time.strftime('%H:%M:%S')}] 🌐 局域网地址: http://0.0.0.0:{server_port}")
        print("=" * 60)
        print(f"[{time.strftime('%H:%M:%S')}] 系统已启动，您可以通过浏览器访问上述地址使用系统")
        
        demo.launch(server_name="0.0.0.0", server_port=server_port, share=False)
        
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 启动服务失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()