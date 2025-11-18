import sys
import os
print(f"Python版本: {sys.version}")
print(f"当前目录: {os.getcwd()}")
print(f"环境变量: {os.environ.get('OPENAI_API_KEY') is not None}")

print("\n尝试导入openai库...")
try:
    import openai
    print(f"成功导入openai库，版本: {openai.__version__}")
    print(f"OpenAI模块路径: {openai.__file__}")
except Exception as e:
    print(f"导入openai库失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n尝试从openai导入OpenAI类...")
try:
    from openai import OpenAI
    print("成功导入OpenAI类")
    # 尝试实例化OpenAI对象
    try:
        client = OpenAI()
        print("成功实例化OpenAI对象")
    except Exception as e:
        print(f"实例化OpenAI对象失败: {type(e).__name__}: {e}")
except Exception as e:
    print(f"导入OpenAI类失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")