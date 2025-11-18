import sys
import time
import traceback

# 非常简单的Gradio测试脚本
print(f"[{time.strftime('%H:%M:%S')}] 开始测试Gradio导入...")

try:
    # 添加Python路径，确保能找到安装的包
    import os
    sys.path.append(os.path.abspath('.'))
    
    # 先导入一些基础库
    print(f"[{time.strftime('%H:%M:%S')}] 导入基础库...")
    import threading
    import multiprocessing
    
    print(f"[{time.strftime('%H:%M:%S')}] 尝试导入gradio...")
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 成功导入gradio (版本: {gr.__version__})")
    
    print(f"[{time.strftime('%H:%M:%S')}] Gradio导入测试成功！")
    sys.exit(0)
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ Gradio导入失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
except KeyboardInterrupt:
    print(f"[{time.strftime('%H:%M:%S')}] 测试被中断")
    sys.exit(0)