#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medi-RAG 应用程序入口
分离Gradio界面到单独文件，使核心功能与界面解耦
"""

# 先导入Gradio
import sys
import time
import subprocess

print(f"[{time.strftime('%H:%M:%S')}] 开始导入Gradio...")
try:
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] [OK] 成功导入Gradio版本: {gr.__version__}")
    
    # 导入rag模块
    print(f"[{time.strftime('%H:%M:%S')}] 导入rag核心模块...")
    import rag
    print(f"[{time.strftime('%H:%M:%S')}] [OK] 成功导入rag模块")
    
    # 创建主题和界面
    print(f"[{time.strftime('%H:%M:%S')}] 创建Gradio主题...")
    try:
        custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue")
        print(f"[{time.strftime('%H:%M:%S')}] [OK] 主题创建成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [WARNING] 主题创建失败，使用默认主题: {e}")
        custom_theme = None
    
    print(f"[{time.strftime('%H:%M:%S')}] 创建Gradio界面...")
    with gr.Blocks(title="企业知识库问答系统", theme=custom_theme, css="", elem_id="app-container") as demo:
        # 使用rag模块中的自定义CSS
        gr.HTML(f"<style>{rag.custom_css}</style>")
        
        # 创建状态存储
        chat_history = gr.State([])
        
        # 创建标签页
        with gr.Tabs(elem_id="main-tabs") as tabs:
            # 问答标签页
            with gr.TabItem("问答系统", elem_id="qa-tab"):
                gr.Markdown("# 企业知识库问答系统")
                gr.Markdown("**智能助手，支持多知识库管理、多轮对话、普通语义检索和高级多跳推理**")
                gr.Markdown("本系统支持创建多个知识库，上传TXT或PDF文件，通过语义向量检索或创新的多跳推理机制提供信息查询服务。")
                
                # 知识库选择器
                with gr.Row():
                    kb_name_input = gr.Dropdown(
                        choices=rag.get_knowledge_bases(),
                        label="选择知识库",
                        value=rag.DEFAULT_KB,
                        interactive=True
                    )
                
                # 问题输入和回答输出
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="请输入您的问题",
                            placeholder="例如：请解释什么是RAG技术？",
                            lines=3,
                            interactive=True
                        )
                        
                        with gr.Row():
                            # 提交按钮
                            submit_btn = gr.Button("提交问题", variant="primary", elem_classes=["submit-btn"])
                            # 清空按钮
                            clear_btn = gr.Button("清空对话")
                            # 表格格式输出开关
                            use_table_output = gr.Checkbox(
                                label="使用表格格式输出",
                                value=False
                            )
                            # 多跳推理开关
                            use_multi_hop = gr.Checkbox(
                                label="使用多跳推理",
                                value=False
                            )
                    
                    with gr.Column(scale=2):
                        answer_output = gr.Markdown(
                            label="回答",
                            interactive=False
                        )
                        
                        # 推理步骤和检索内容显示区域
                        with gr.Row():
                            with gr.Column(scale=1):
                                reasoning_steps_output = gr.Markdown(
                                    label="推理步骤",
                                    interactive=False,
                                    elem_classes=["reasoning-steps"]
                                )
                            with gr.Column(scale=1):
                                retrieved_content_output = gr.Markdown(
                                    label="检索内容",
                                    interactive=False,
                                    elem_classes=["reasoning-steps"]
                                )
                
                # 对话历史
                chat_history_output = gr.Chatbot(
                    label="对话历史",
                    interactive=False,
                    height=300
                )
                
                # 响应状态显示
                status_output = gr.Markdown(
                    label="处理状态",
                    interactive=False
                )
            
            # 知识库管理标签页
            with gr.TabItem("知识库管理", elem_id="kb-tab"):
                gr.Markdown("# 知识库管理")
                gr.Markdown("**管理您的知识库，上传文件并进行索引**")
                
                # 当前知识库列表
                current_kbs = gr.Markdown(
                    label="当前知识库",
                    value="\n".join([f"- {kb}" for kb in rag.get_knowledge_bases()])
                )
                
                # 知识库操作
                with gr.Row():
                    # 创建知识库
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["kb-management"]):
                            gr.Markdown("## 创建新知识库")
                            new_kb_name = gr.Textbox(
                                label="知识库名称",
                                placeholder="输入新知识库名称",
                                interactive=True
                            )
                            create_kb_btn = gr.Button("创建知识库")
                            create_kb_status = gr.Markdown(
                                label="创建状态",
                                interactive=False
                            )
                    
                    # 删除知识库
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["kb-management"]):
                            gr.Markdown("## 删除知识库")
                            delete_kb_name = gr.Dropdown(
                                choices=rag.get_knowledge_bases(),
                                label="选择要删除的知识库",
                                value=rag.DEFAULT_KB if rag.DEFAULT_KB in rag.get_knowledge_bases() else None,
                                interactive=True
                            )
                            delete_kb_btn = gr.Button("删除知识库", variant="stop")
                            delete_kb_status = gr.Markdown(
                                label="删除状态",
                                interactive=False
                            )
                
                # 文件上传
                with gr.Group(elem_classes=["kb-management"]):
                    gr.Markdown("## 上传文件到知识库")
                    
                    with gr.Row():
                        upload_kb_name = gr.Dropdown(
                            choices=rag.get_knowledge_bases(),
                            label="选择知识库",
                            value=rag.DEFAULT_KB,
                            interactive=True
                        )
                    
                    file_upload = gr.File(
                        label="上传文件 (支持PDF/TXT)",
                        type="file",
                        file_types=[".pdf", ".txt"],
                        multiple=True,
                        interactive=True,
                        elem_classes=["compact-upload"]
                    )
                    
                    upload_btn = gr.Button("上传并索引文件", variant="primary")
                    upload_status = gr.Markdown(
                        label="上传状态",
                        interactive=False
                    )
                
                # 查看文件列表
                with gr.Group(elem_classes=["kb-management"]):
                    gr.Markdown("## 查看知识库文件")
                    
                    with gr.Row():
                        view_kb_name = gr.Dropdown(
                            choices=rag.get_knowledge_bases(),
                            label="选择知识库",
                            value=rag.DEFAULT_KB,
                            interactive=True
                        )
                    
                    view_files_btn = gr.Button("查看文件列表")
                    files_list_output = gr.Markdown(
                        label="文件列表",
                        interactive=False
                    )
                
            # 检索内容标签页
            with gr.TabItem("检索内容", elem_id="retrieval-tab"):
                gr.Markdown("# 检索内容")
                gr.Markdown("**查看当前检索到的知识库内容**")
                
                retrieved_chunks_output = gr.Markdown(
                    label="检索到的内容",
                    interactive=False,
                    height=400
                )
        
        # 刷新知识库列表的函数
        def refresh_kb_lists():
            kbs = rag.get_knowledge_bases()
            return (
                gr.Dropdown(choices=kbs, value=rag.DEFAULT_KB if rag.DEFAULT_KB in kbs else kbs[0] if kbs else None),
                gr.Dropdown(choices=kbs, value=rag.DEFAULT_KB if rag.DEFAULT_KB in kbs else kbs[0] if kbs else None),
                gr.Dropdown(choices=kbs, value=rag.DEFAULT_KB if rag.DEFAULT_KB in kbs else kbs[0] if kbs else None),
                gr.Dropdown(choices=kbs, value=rag.DEFAULT_KB if rag.DEFAULT_KB in kbs else kbs[0] if kbs else None),
                gr.Markdown(value="\n".join([f"- {kb}" for kb in kbs]))
            )
        
        # 绑定刷新按钮
        refresh_btn = gr.Button("刷新知识库列表")
        refresh_btn.click(
            refresh_kb_lists,
            [],
            [kb_name_input, delete_kb_name, upload_kb_name, view_kb_name, current_kbs]
        )
        
        # 创建知识库的函数
        def create_kb(kb_name):
            if not kb_name:
                return "请输入知识库名称"
            
            try:
                result = rag.create_knowledge_base(kb_name)
                return result
            except Exception as e:
                return f"创建失败: {str(e)}"
        
        # 删除知识库的函数
        def delete_kb(kb_name):
            if not kb_name:
                return "请选择要删除的知识库"
            
            try:
                result = rag.delete_knowledge_base(kb_name)
                return result
            except Exception as e:
                return f"删除失败: {str(e)}"
        
        # 上传文件的函数
        def upload_files(kb_name, files):
            if not kb_name:
                return "请选择知识库"
            
            if not files:
                return "请选择要上传的文件"
            
            try:
                # 使用正确的函数名
                result = rag.process_and_index_files(files, kb_name)
                return result
            except Exception as e:
                return f"上传失败: {str(e)}"
        
        # 查看文件列表的函数
        def view_files(kb_name):
            if not kb_name:
                return "请选择知识库"
            
            try:
                # 使用正确的函数名
                files = rag.get_kb_files(kb_name)
                if not files:
                    return "该知识库中没有文件"
                return "\n".join([f"- {file}" for file in files])
            except Exception as e:
                return f"查看失败: {str(e)}"
        
        # 对话函数
        def chat_with_rag(question, chat_history, kb_name, use_table_output, use_multi_hop):
            try:
                # 使用rag模块的对话函数
                answer = rag.answer_question(question, kb_name, use_table_format=use_table_output, multi_hop=use_multi_hop)
                
                # 格式化回答
                chat_history.append((question, answer))
                
                return (
                    chat_history,
                    answer,
                    "使用简单检索模式，无推理步骤" if not use_multi_hop else "多跳推理已启用",
                    "未实现检索内容展示",
                    "完成"
                )
            except Exception as e:
                return (
                    chat_history,
                    f"处理失败: {str(e)}",
                    f"错误: {str(e)}",
                    "",
                    "处理失败"
                )
        
        # 清空对话函数
        def clear_chat():
            return [], [], "", "", ""
        
        # 绑定事件
        submit_btn.click(
            chat_with_rag,
            [question_input, chat_history, kb_name_input, use_table_output, use_multi_hop],
            [chat_history_output, answer_output, reasoning_steps_output, retrieved_content_output, status_output]
        )
        
        clear_btn.click(
            clear_chat,
            [],
            [chat_history_output, answer_output, reasoning_steps_output, retrieved_content_output, status_output]
        )
        
        create_kb_btn.click(
            create_kb,
            [new_kb_name],
            [create_kb_status]
        )
        
        delete_kb_btn.click(
            delete_kb,
            [delete_kb_name],
            [delete_kb_status]
        )
        
        upload_btn.click(
            upload_files,
            [upload_kb_name, file_upload],
            [upload_status]
        )
        
        view_files_btn.click(
            view_files,
            [view_kb_name],
            [files_list_output]
        )
    
    # 启动界面
    print(f"[{time.strftime('%H:%M:%S')}] 启动Gradio界面...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 启动失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)