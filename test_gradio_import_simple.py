#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简Gradio导入测试脚本
"""

import time
import sys
import traceback

print(f"[{time.strftime('%H:%M:%S')}] 开始测试Gradio导入...")
print(f"Python版本: {sys.version}")

# 测试1: 直接导入Gradio
try:
    start_time = time.time()
    import gradio as gr
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 成功导入 gradio (版本: {gr.__version__})，耗时: {end_time - start_time:.2f} 秒")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 gradio 失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试2: 尝试创建简单组件
try:
    print(f"[{time.strftime('%H:%M:%S')}] 尝试创建Gradio组件...")
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="gray",
        text_size="lg",
        spacing_size="md",
        radius_size="md"
    )
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 成功创建Gradio主题")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 创建Gradio主题失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] Gradio导入测试全部通过！")