import sys
print(f"Python版本: {sys.version}")

# 测试完整的导入流程，从numpy到配置文件
print("\n测试完整导入流程:")

# 1. 测试numpy
print("\n1. 测试numpy导入:")
try:
    import numpy as np
    print("✓ numpy导入成功")
except Exception as e:
    print(f"✗ numpy导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试llama_index组件
print("\n2. 测试llama_index组件导入:")
try:
    from llama_index.core.node_parser import SentenceSplitter
    print("✓ SentenceSplitter导入成功")
except Exception as e:
    print(f"✗ SentenceSplitter导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试其他标准库
print("\n3. 测试其他标准库导入:")
try:
    import re
    print("✓ re导入成功")
    from typing import List, Dict, Any, Optional, Tuple
    print("✓ typing模块导入成功")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("✓ concurrent.futures导入成功")
    import json
    print("✓ json导入成功")
    import shutil
    print("✓ shutil导入成功")
except Exception as e:
    print(f"✗ 标准库导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试OpenAI局部导入
print("\n4. 测试OpenAI局部导入:")
try:
    from openai import OpenAI
    print("✓ OpenAI局部导入成功")
except Exception as e:
    print(f"✗ OpenAI局部导入失败: {e}")
    import traceback
    traceback.print_exc()
    # 不退出，因为我们已经改为延迟导入

# 5. 测试gradio
print("\n5. 测试gradio导入:")
try:
    import gradio as gr
    print(f"✓ gradio导入成功，版本: {gr.__version__}")
except Exception as e:
    print(f"✗ gradio导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试fitz
print("\n6. 测试fitz导入:")
try:
    import fitz  # PyMuPDF
    print(f"✓ fitz (PyMuPDF)导入成功，版本: {fitz.__version__}")
except Exception as e:
    print(f"✗ fitz (PyMuPDF)导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 测试chardet
print("\n7. 测试chardet导入:")
try:
    import chardet
    print(f"✓ chardet导入成功，版本: {chardet.__version__}")
except Exception as e:
    print(f"✗ chardet导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. 测试traceback
print("\n8. 测试traceback导入:")
try:
    import traceback
    print("✓ traceback导入成功")
except Exception as e:
    print(f"✗ traceback导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 9. 测试配置文件
print("\n9. 测试配置文件导入:")
try:
    from config import AppConfig  # 导入配置文件
    print("✓ AppConfig导入成功")
    # 测试访问配置项
    print(f"  KB_BASE_DIR: {AppConfig.knowledge_base_root}")
    print(f"  DEFAULT_KB: {AppConfig.default_knowledge_base}")
    print(f"  LLM_BASE_URL: {AppConfig.llm_base_url}")
except Exception as e:
    print(f"✗ 配置文件导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 所有导入测试都通过了！")

# 10. 测试延迟初始化的OpenAI客户端
print("\n10. 测试延迟初始化的OpenAI客户端:")
try:
    class TestConfig:
        llm_api_key = "test_key"
        llm_base_url = "http://example.com"
    
    class TestAppConfig:
        llm_api_key = "test_key"
        llm_base_url = "http://example.com"
    
    # 测试get_client函数的逻辑
    test_client = None
    def get_test_client():
        global test_client
        if test_client is None:
            print("  初始化测试OpenAI客户端...")
            from openai import OpenAI
            test_client = OpenAI(
                api_key=TestAppConfig.llm_api_key,
                base_url=TestAppConfig.llm_base_url
            )
            print("  测试OpenAI客户端初始化完成")
        return test_client
    
    # 模拟导入后立即调用get_client
    # 这会失败，因为我们使用的是测试密钥
    print("  测试get_client函数调用...")
    try:
        client = get_test_client()
        print("  ✓ get_client函数调用成功")
    except Exception as e:
        print(f"  ⚠️ get_client调用抛出异常（预期行为，因为使用了测试密钥）: {e}")
        print("  这表明OpenAI客户端在需要时才会初始化")
        
except Exception as e:
    print(f"✗ 延迟初始化测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")