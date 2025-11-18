#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的Gradio测试脚本
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始测试Gradio")

try:
    print(f"[{time.strftime('%H:%M:%S')}] 正在导入gradio...")
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入gradio成功")
    print(f"[{time.strftime('%H:%M:%S')}] gradio版本: {gr.__version__}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入gradio失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建一个简单的界面
def greet(name):
    return f"Hello, {name}!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output)

print(f"[{time.strftime('%H:%M:%S')}] 正在启动Gradio服务...")
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,  # 使用不同的端口
    share=False,
    debug=False
)