import time
import sys

try:
    print(f"[{time.strftime('%H:%M:%S')}] 开始测试Gradio导入...")
    
    # 尝试导入gradio
    import gradio as gr
    
    print(f"[{time.strftime('%H:%M:%S')}] Gradio导入成功！")
    
    # 检查Gradio属性
    print(f"[{time.strftime('%H:%M:%S')}] 检查Gradio属性...")
    
    # 检查__version__属性
    if hasattr(gr, '__version__'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 版本: {gr.__version__}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 版本属性不存在")
    
    # 检查__file__属性
    if hasattr(gr, '__file__'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 模块路径: {gr.__file__}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 模块路径属性不存在")
    
    # 检查themes属性
    if hasattr(gr, 'themes'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ themes属性存在")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ themes属性不存在")
    
    # 检查Blocks类
    if hasattr(gr, 'Blocks'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Blocks类存在")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ Blocks类不存在")
    
    # 检查其他常用组件
    components = ['Textbox', 'Button', 'Markdown', 'FileUpload', 'Tab', 'Tabs', 'State']
    for component in components:
        if hasattr(gr, component):
            print(f"[{time.strftime('%H:%M:%S')}] ✓ {component}组件存在")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ✗ {component}组件不存在")
    
    print(f"[{time.strftime('%H:%M:%S')}] 测试完成！")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)