#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Gradio 4.36.1导入
"""

import sys
import time

def test_gradio_import():
    print(f"[{time.strftime('%H:%M:%S')}] 开始测试Gradio导入...")
    print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
    
    try:
        # 导入gradio
        import gradio as gr
        print(f"[{time.strftime('%H:%M:%S')}] [成功] 导入gradio版本: {gr.__version__}")
        print(f"[{time.strftime('%H:%M:%S')}] Gradio模块路径: {gr.__file__}")
        
        # 测试简单的Gradio组件创建
        print(f"[{time.strftime('%H:%M:%S')}] 测试创建简单Gradio组件...")
        demo = gr.Interface(fn=lambda x: x, inputs="text", outputs="text")
        print(f"[{time.strftime('%H:%M:%S')}] [成功] 创建Gradio界面组件")
        
        print(f"[{time.strftime('%H:%M:%S')}] Gradio测试完成！")
        return True
        
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [错误] Gradio导入失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_gradio_import()