#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试llama_index导入的简单脚本
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始测试llama_index导入")

# 测试基本导入
try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 llama_index.core...")
    from llama_index.core.node_parser import SentenceSplitter
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 SentenceSplitter 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 SentenceSplitter 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 VectorStoreIndex 和 StorageContext...")
    from llama_index.core import VectorStoreIndex, StorageContext
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 VectorStoreIndex 和 StorageContext 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 VectorStoreIndex 和 StorageContext 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 load_index_from_storage...")
    from llama_index.core import load_index_from_storage
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 load_index_from_storage 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 load_index_from_storage 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 FaissVectorStore...")
    from llama_index.vector_stores.faiss import FaissVectorStore
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 FaissVectorStore 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 FaissVectorStore 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试其他库
try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 langchain.llms...")
    from langchain.llms import OpenAI
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 OpenAI 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 OpenAI 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 transformers...")
    from transformers import AutoTokenizer, AutoModel
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 AutoTokenizer 和 AutoModel 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 AutoTokenizer 和 AutoModel 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 SentenceTransformer 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 SentenceTransformer 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入 gradio...")
    import gradio as gr
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入 gradio 成功")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入 gradio 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 所有库导入测试完成！")