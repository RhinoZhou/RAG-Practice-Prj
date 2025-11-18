#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简Gradio测试脚本
"""

print("测试Gradio导入...")
try:
    import gradio as gr
    print(f"Gradio导入成功！版本: {gr.__version__}")
    print("Gradio测试通过！")
except Exception as e:
    print(f"Gradio导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()