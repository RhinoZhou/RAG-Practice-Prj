import sys
import time

def log(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

log("开始测试SentenceSplitter导入...")
log(f"Python版本: {sys.version}")

# 首先测试基本的llama_index导入
try:
    log("尝试导入llama_index...")
    import llama_index
    log("成功导入llama_index")
except Exception as e:
    log(f"导入llama_index失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 然后测试核心模块导入
try:
    log("尝试导入llama_index.core...")
    from llama_index import core
    log("成功导入llama_index.core")
except Exception as e:
    log(f"导入llama_index.core失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 然后测试node_parser导入
try:
    log("尝试导入llama_index.core.node_parser...")
    from llama_index.core import node_parser
    log("成功导入llama_index.core.node_parser")
except Exception as e:
    log(f"导入llama_index.core.node_parser失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 最后测试SentenceSplitter导入
try:
    log("尝试导入SentenceSplitter from llama_index.core.node_parser...")
    from llama_index.core.node_parser import SentenceSplitter
    log("成功导入SentenceSplitter")
    # 测试创建实例
    log("尝试创建SentenceSplitter实例...")
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=20)
    log("成功创建SentenceSplitter实例")
    # 测试基本功能
    test_text = "这是一个测试文本，用于测试SentenceSplitter的功能。我们希望它能够正确地将文本分割成多个块。"
    chunks = splitter.split_text(test_text)
    log(f"成功分割文本，得到 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        log(f"块 {i+1}: {chunk[:50]}...")
except Exception as e:
    log(f"导入或使用SentenceSplitter失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log("SentenceSplitter测试完成，所有功能正常！")