#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版RAG应用 - 最小功能集
仅包含基础功能，用于测试程序启动和运行
"""

import time
import sys

def log(message):
    """带时间戳的日志函数"""
    timestamp = time.strftime("[%H:%M:%S]", time.localtime())
    print(f"{timestamp} {message}")
    sys.stdout.flush()

# 程序启动
log("程序启动")

# 测试Gradio
log("开始测试Gradio")
try:
    import gradio as gr
    log(f"✓ 导入 gradio 成功！版本: {gr.__version__}")
    
    # 创建一个简单的界面
    def greet(name):
        return f"Hello, {name}!"
    
    # 启动Gradio服务
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    log("正在启动Gradio服务...")
    iface.launch(server_port=7861, share=False)
    log("Gradio服务启动成功！")
except Exception as e:
    log(f"✗ gradio 导入或启动失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()