#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细Gradio测试脚本 - 逐步测试各个功能
"""

import time
import sys

print("=== 详细Gradio测试 ===")
print(f"Python版本: {sys.version}")
print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 40)

# 步骤1: 尝试导入gradio核心模块
print("\n步骤1: 导入gradio核心模块...")
try:
    import gradio
    print(f"✓ gradio模块导入成功！")
except Exception as e:
    print(f"✗ gradio模块导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤2: 尝试导入gradio as gr
print("\n步骤2: 导入gradio as gr...")
try:
    import gradio as gr
    print(f"✓ gradio as gr导入成功！")
    print(f"  Gradio版本: {gr.__version__}")
except Exception as e:
    print(f"✗ gradio as gr导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤3: 尝试访问gradio的基本属性
print("\n步骤3: 访问Gradio基本属性...")
try:
    print(f"✓ gr.Interface存在: {hasattr(gr, 'Interface')}")
    print(f"✓ gr.Blocks存在: {hasattr(gr, 'Blocks')}")
    print(f"✓ gr.Textbox存在: {hasattr(gr, 'Textbox')}")
    print(f"✓ gr.Button存在: {hasattr(gr, 'Button')}")
except Exception as e:
    print(f"✗ 访问Gradio属性失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤4: 尝试创建简单的函数
print("\n步骤4: 创建简单函数...")
try:
    def greet(name):
        return f"Hello, {name}!"
    print(f"✓ 函数创建成功")
except Exception as e:
    print(f"✗ 函数创建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤5: 尝试创建Interface实例
print("\n步骤5: 创建Interface实例...")
try:
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    print(f"✓ Interface实例创建成功")
except Exception as e:
    print(f"✗ Interface实例创建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤6: 尝试启动服务（不阻塞）
print("\n步骤6: 启动Gradio服务...")
try:
    # 这里我们不实际启动服务，只是测试launch方法是否存在
    print(f"✓ iface.launch方法存在: {hasattr(iface, 'launch')}")
    print("注意: 我们没有实际启动服务，只是测试了launch方法的存在性")
except Exception as e:
    print(f"✗ 测试launch方法失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 40)
print("🎉 所有Gradio测试通过！")
print(f"测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")