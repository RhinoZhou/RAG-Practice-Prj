import sys
print(f"Python版本: {sys.version}")

try:
    print("尝试导入 llama_index.core...")
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    print("成功导入 VectorStoreIndex 和 SimpleDirectoryReader")
except Exception as e:
    print(f"导入 llama_index.core 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n尝试导入 SentenceSplitter...")
    from llama_index.core.node_parser import SentenceSplitter
    print("成功导入 SentenceSplitter")
except Exception as e:
    print(f"导入 SentenceSplitter 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")