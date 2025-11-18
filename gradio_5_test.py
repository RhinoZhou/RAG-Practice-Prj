#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 5.x API测试脚本
使用Gradio 5.x的新API创建界面
"""

import sys
import time

print(f"Python版本: {sys.version.split()[0]}")
print("\n尝试使用Gradio 5.x API创建界面...")

try:
    # Gradio 5.x的导入方式
    import gradio as gr
    print(f"已导入Gradio 5.x，版本: {gr.__version__}")
    
    # 创建一个简单的界面
    def greet(name):
        return f"你好, {name}!"
    
    # 使用Gradio 5.x的Interface API
    demo = gr.Interface(
        fn=greet,
        inputs=gr.Textbox(label="你的名字"),
        outputs=gr.Textbox(label="问候语"),
        title="Gradio 5.x 测试界面",
        description="这是一个使用Gradio 5.x API创建的简单界面"
    )
    
    print("\n✅ 成功创建Gradio 5.x界面！")
    print("\nGradio 5.x API测试通过！")
    
    # 启动界面（可选）
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    
except Exception as e:
    import traceback
    print(f"\n❌ 错误: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\nGradio 5.x API测试失败！")