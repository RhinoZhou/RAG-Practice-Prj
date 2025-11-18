#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简测试：不包含安装步骤，只检查和导入
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")

# 检查已安装的Gradio版本
print(f"\n[{time.strftime('%H:%M:%S')}] 检查已安装的Gradio版本...")
import pkg_resources
try:
    gradio_version = pkg_resources.get_distribution("gradio").version
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 已安装Gradio版本: {gradio_version}")
except pkg_resources.DistributionNotFound:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 未找到Gradio安装")
    sys.exit(1)

# 测试导入
print(f"\n[{time.strftime('%H:%M:%S')}] 测试Gradio导入...")
try:
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Gradio导入成功，版本: {gr.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Gradio路径: {gr.__file__}")
    
    # 测试基本功能
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 测试Gradio版本: {gr.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 测试Gradio模块内容: {dir(gr)[:20]}...")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ Gradio导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[{time.strftime('%H:%M:%S')}] 测试完成！Gradio可以正常使用。")