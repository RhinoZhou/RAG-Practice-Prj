#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：逐步测试导入过程
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始诊断...")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")

# 1. 测试基本导入
print(f"\n[{time.strftime('%H:%M:%S')}] 1. 测试基本导入...")
try:
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Gradio导入成功，版本: {gr.__version__}")
    print(f"[{time.strftime('%H:%M:%S')}]    Gradio路径: {gr.__file__}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ Gradio导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试rag模块导入
print(f"\n[{time.strftime('%H:%M:%S')}] 2. 测试rag模块导入...")
try:
    import rag
    print(f"[{time.strftime('%H:%M:%S')}] ✅ rag模块导入成功")
    print(f"[{time.strftime('%H:%M:%S')}]    rag路径: {rag.__file__}")
    
    # 测试rag模块的基本功能
    print(f"[{time.strftime('%H:%M:%S')}]    测试rag.get_knowledge_bases(): {rag.get_knowledge_bases()}")
    print(f"[{time.strftime('%H:%M:%S')}]    测试rag.DEFAULT_KB: {rag.DEFAULT_KB}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ rag模块导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试Gradio界面元素创建
print(f"\n[{time.strftime('%H:%M:%S')}] 3. 测试Gradio界面元素创建...")
try:
    # 测试主题创建
    try:
        custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue")
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 主题创建成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️  主题创建失败，使用默认主题: {e}")
        custom_theme = None
    
    # 测试简单Blocks创建
    with gr.Blocks(title="测试界面", theme=custom_theme) as demo:
        gr.Markdown("# 测试界面")
        gr.Textbox(label="测试输入")
        gr.Button("测试按钮")
    
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Blocks创建成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ Gradio界面创建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[{time.strftime('%H:%M:%S')}] 所有测试通过！导入过程正常。")
print(f"[{time.strftime('%H:%M:%S')}] 开始启动完整界面...")

# 4. 测试完整界面启动（非阻塞模式）
try:
    # 导入app.py中的界面
    from app import demo
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 从app.py导入demo成功")
    
    # 启动界面（测试10秒后自动关闭）
    import threading
    import time
    
    def launch_and_stop():
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860, inbrowser=False)
    
    thread = threading.Thread(target=launch_and_stop)
    thread.daemon = True
    thread.start()
    
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 界面已启动（线程模式）")
    print(f"[{time.strftime('%H:%M:%S')}] 测试运行10秒后自动结束...")
    
    # 运行10秒后退出
    time.sleep(10)
    print(f"[{time.strftime('%H:%M:%S')}] 诊断完成！")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 完整界面测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)