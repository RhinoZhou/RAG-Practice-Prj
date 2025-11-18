import time
import sys
import os

print(f"[{time.strftime('%H:%M:%S')}] 程序开始执行")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
print(f"[{time.strftime('%H:%M:%S')}] 当前工作目录: {os.getcwd()}")

# 安装依赖的简化版本
def install_dependencies():
    try:
        print(f"[{time.strftime('%H:%M:%S')}] 开始安装依赖...")
        
        # 安装核心依赖
        dependencies = [
            "torch",
            "faiss-cpu",
            "numpy",
            "pymupdf",
            "chardet",
            "concurrent.futures",
            "gradio==5.49.1"
        ]
        
        # 只检查gradio是否已安装
        try:
            import gradio as gr
            print(f"[{time.strftime('%H:%M:%S')}] ✓ gradio已安装 (版本: {gr.__version__})")
        except ImportError:
            print(f"[{time.strftime('%H:%M:%S')}] ✗ gradio未安装，开始安装...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", "gradio==5.49.1"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[{time.strftime('%H:%M:%S')}] ✓ gradio安装成功")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ✗ gradio安装失败: {result.stderr}")
                sys.exit(1)
                
        print(f"[{time.strftime('%H:%M:%S')}] 依赖安装完成")
        
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 安装依赖时出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 执行依赖安装
try:
    install_dependencies()
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 执行依赖安装时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入核心库
try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入核心库...")
    
    import torch
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入torch成功 (版本: {torch.__version__})")
    
    import faiss
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入faiss成功")
    
    import numpy as np
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入numpy成功 (版本: {np.__version__})")
    
    import fitz  # PyMuPDF
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入fitz成功 (版本: {fitz.__version__})")
    
    import chardet
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入chardet成功 (版本: {chardet.__version__})")
    
    from typing import List, Dict, Any, Optional, Tuple
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入typing模块成功")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入concurrent.futures成功")
    
    print(f"[{time.strftime('%H:%M:%S')}] 核心库导入完成")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入核心库时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入配置文件
try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入配置文件...")
    from config import AppConfig
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入AppConfig成功")
    print(f"[{time.strftime('%H:%M:%S')}]   KB_BASE_DIR: {AppConfig.knowledge_base_root}")
    print(f"[{time.strftime('%H:%M:%S')}]   DEFAULT_KB: {AppConfig.default_knowledge_base}")
    
    # 创建必要的目录
    KB_BASE_DIR = AppConfig.knowledge_base_root
    os.makedirs(KB_BASE_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 知识库目录创建成功: {KB_BASE_DIR}")
    
    DEFAULT_KB = AppConfig.default_knowledge_base
    DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
    os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 默认知识库目录创建成功: {DEFAULT_KB_DIR}")
    
    OUTPUT_DIR = AppConfig.temp_output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 临时输出目录创建成功: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入配置文件时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入gradio
try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始导入gradio...")
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✓ gradio导入成功 (版本: {gr.__version__})")
    print(f"[{time.strftime('%H:%M:%S')}]   Gradio模块路径: {gr.__file__}")
    
    # 检查gradio的关键组件
    print(f"[{time.strftime('%H:%M:%S')}] 检查gradio组件...")
    components_to_check = ['Blocks', 'Textbox', 'Button', 'Markdown', 'File', 'Tab', 'Tabs', 'State']
    for component in components_to_check:
        if hasattr(gr, component):
            print(f"[{time.strftime('%H:%M:%S')}]   ✓ {component}组件存在")
        else:
            print(f"[{time.strftime('%H:%M:%S')}]   ✗ {component}组件不存在")
    
    # 检查themes
    if hasattr(gr, 'themes'):
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ themes属性存在")
        if hasattr(gr.themes, 'Soft'):
            print(f"[{time.strftime('%H:%M:%S')}]   ✓ themes.Soft存在")
    
    print(f"[{time.strftime('%H:%M:%S')}] gradio组件检查完成")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入gradio时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建一个简单的Gradio界面
try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始创建Gradio界面...")
    
    # 简单的CSS
    custom_css = """
    .container { max-width: 800px !important; margin: 0 auto !important; }
    .submit-btn { background-color: #2196F3 !important; border: none !important; }
    """
    
    # 创建主题
    try:
        print(f"[{time.strftime('%H:%M:%S')}]   创建主题...")
        custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="blue")
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ 主题创建成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   ✗ 主题创建失败，使用默认主题: {e}")
        custom_theme = None
    
    # 创建界面
    print(f"[{time.strftime('%H:%M:%S')}]   创建界面...")
    with gr.Blocks(title="测试界面", theme=custom_theme, css=custom_css) as demo:
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ Blocks创建成功")
        
        gr.Markdown("# 测试界面")
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ Markdown添加成功")
        
        # 简单的文本输入和按钮
        with gr.Row():
            input_text = gr.Textbox(label="输入")
            output_text = gr.Textbox(label="输出")
            submit_btn = gr.Button("提交")
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ 输入输出组件添加成功")
        
        # 简单的点击事件
        def process_input(input_val):
            return f"你输入了: {input_val}"
        
        submit_btn.click(fn=process_input, inputs=input_text, outputs=output_text)
        print(f"[{time.strftime('%H:%M:%S')}]   ✓ 事件绑定成功")
    
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 界面创建完成")
    
    # 启动界面
    print(f"[{time.strftime('%H:%M:%S')}] 开始启动界面...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 界面启动成功")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 创建或启动界面时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 程序正常结束")