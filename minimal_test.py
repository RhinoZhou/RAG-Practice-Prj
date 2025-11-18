#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简测试：只测试Gradio导入
"""

import sys
import time
import os

print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
print(f"[{time.strftime('%H:%M:%S')}] 当前目录: {os.getcwd()}")
print(f"[{time.strftime('%H:%M:%S')}] sys.path: {sys.path[:3]}")

# 强制刷新pip并重新安装Gradio
print(f"\n[{time.strftime('%H:%M:%S')}] 尝试重新安装Gradio...")
import subprocess
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'gradio', '--force-reinstall', '--no-cache-dir'],
    capture_output=True,
    text=True
)
print(f"[{time.strftime('%H:%M:%S')}] 安装输出: {result.stdout[:500]}...")
if result.stderr:
    print(f"[{time.strftime('%H:%M:%S')}] 安装错误: {result.stderr[:500]}...")

# 再次测试导入
print(f"\n[{time.strftime('%H:%M:%S')}] 测试Gradio导入...")
try:
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Gradio导入成功，版本: {gr.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Gradio路径: {gr.__file__}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ Gradio导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[{time.strftime('%H:%M:%S')}] 测试完成！")