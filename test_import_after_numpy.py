import sys
print(f"Python版本: {sys.version}")

# 测试numpy之后的导入
print("\n测试numpy导入:")
try:
    import numpy as np
    print("✓ numpy导入成功")
except Exception as e:
    print(f"✗ numpy导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试llama_index.core导入:")
try:
    import llama_index.core
    print("✓ llama_index.core导入成功")
    print(f"  版本: {llama_index.core.__version__}")
except Exception as e:
    print(f"✗ llama_index.core导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试SentenceSplitter导入:")
try:
    from llama_index.core.node_parser import SentenceSplitter
    print("✓ SentenceSplitter导入成功")
except Exception as e:
    print(f"✗ SentenceSplitter导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试其他库导入:")
try:
    import re
    print("✓ re导入成功")
except Exception as e:
    print(f"✗ re导入失败: {e}")

print("\n所有导入测试完成！")