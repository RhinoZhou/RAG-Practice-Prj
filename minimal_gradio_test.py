#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最基本的Gradio测试，只导入Gradio并创建一个简单界面
"""

import gradio as gr
import time

def log(message):
    """简单的日志函数"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def greet(name):
    """简单的问候函数"""
    return f"Hello, {name}!"

# 程序启动
log("程序启动 - 最基本的Gradio测试")
log(f"Gradio版本: {gr.__version__}")

# 创建简单的Gradio界面
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="您的名字"),
    outputs=gr.Textbox(label="问候"),
    title="最简单的Gradio测试"
)

# 启动服务
log("启动Gradio服务...")
demo.launch(
    server_name="127.0.0.1",
    server_port=7861,
    share=False,
    debug=True
)