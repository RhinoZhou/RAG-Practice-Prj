import time
import sys

print(f"[{time.strftime('%H:%M:%S')}] 开始测试")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
print(f"[{time.strftime('%H:%M:%S')}] Python可执行文件路径: {sys.executable}")

# 尝试导入gradio
print(f"[{time.strftime('%H:%M:%S')}] 准备导入gradio...")

# 使用importlib动态导入，以便更好地控制和捕获错误
import importlib

try:
    # 记录开始时间
    start_time = time.time()
    
    # 动态导入gradio
    gr_module = importlib.import_module("gradio")
    
    # 记录导入完成时间
    import_time = time.time() - start_time
    
    print(f"[{time.strftime('%H:%M:%S')}] ✓ gradio导入成功！")
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入耗时: {import_time:.2f}秒")
    
    # 检查模块属性
    if hasattr(gr_module, '__version__'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 版本: {gr_module.__version__}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 版本属性不存在")
    
    if hasattr(gr_module, '__file__'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 文件路径: {gr_module.__file__}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 文件路径属性不存在")
    
    # 尝试访问一个简单的属性
    if hasattr(gr_module, 'Blocks'):
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Blocks类存在")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ Blocks类不存在")
    
    print(f"[{time.strftime('%H:%M:%S')}] 测试完成！所有检查通过。")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 如果程序执行到这里，说明导入成功
print(f"[{time.strftime('%H:%M:%S')}] 程序执行完成，没有退出！")

# 等待几秒钟再退出，以便观察
import time
print(f"[{time.strftime('%H:%M:%S')}] 等待3秒后退出...")
time.sleep(3)
print(f"[{time.strftime('%H:%M:%S')}] 程序正常退出！")