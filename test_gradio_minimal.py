#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简Gradio测试脚本
"""

print("测试Gradio导入...")

# 只测试基本导入，不做其他操作
try:
    import gradio
    print("Gradio核心模块导入成功")
except Exception as e:
    print(f"Gradio核心模块导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("测试完成！")