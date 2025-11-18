#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐步导入测试脚本 - 找出导入顺序问题
"""

import time
import sys

def log(message):
    """带时间戳的日志函数"""
    timestamp = time.strftime("[%H:%M:%S]", time.localtime())
    print(f"{timestamp} {message}")
    sys.stdout.flush()

# 程序启动
log("程序启动")

# 步骤1: 导入基础库
log("\n步骤1: 导入基础库")
try:
    import os
    import sys
    import time
    import shutil
    import datetime
    log("✓ 导入基础库成功")
except Exception as e:
    log(f"✗ 导入基础库失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤2: 导入gradio（先单独导入，确认它能正常工作）
log("\n步骤2: 单独导入gradio")
try:
    import gradio as gr
    log(f"✓ 导入 gradio 成功！版本: {gr.__version__}")
except Exception as e:
    log(f"✗ gradio 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤3: 导入科学计算库
log("\n步骤3: 导入科学计算库")
try:
    import numpy as np
    log("✓ 导入 numpy 成功")
except Exception as e:
    log(f"✗ 导入 numpy 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import pandas as pd
    log("✓ 导入 pandas 成功")
except Exception as e:
    log(f"✗ 导入 pandas 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤4: 导入torch
log("\n步骤4: 导入torch")
try:
    import torch
    log("✓ 导入 torch 成功")
except Exception as e:
    log(f"✗ 导入 torch 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤5: 导入faiss
log("\n步骤5: 导入faiss")
try:
    import faiss
    log("✓ 导入 faiss 成功")
except Exception as e:
    log(f"✗ 导入 faiss 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤6: 导入nltk
log("\n步骤6: 导入nltk")
try:
    import nltk
    log("✓ 导入 nltk 成功")
except Exception as e:
    log(f"✗ 导入 nltk 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤7: 导入tqdm
log("\n步骤7: 导入tqdm")
try:
    from tqdm import tqdm
    log("✓ 导入 tqdm 成功")
except Exception as e:
    log(f"✗ 导入 tqdm 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤8: 导入文件处理库
log("\n步骤8: 导入文件处理库")
try:
    import PyPDF2
    log("✓ 导入 PyPDF2 成功")
except Exception as e:
    log(f"✗ 导入 PyPDF2 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import docx
    log("✓ 导入 docx 成功")
except Exception as e:
    log(f"✗ 导入 docx 失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 步骤9: 导入llama_index库
log("\n步骤9: 导入llama_index库")
try:
    from llama_index.core.node_parser import SentenceSplitter
    log("✓ 导入 SentenceSplitter 成功")
except Exception as e:
    log(f"✗ 导入 SentenceSplitter 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    log("✓ 导入 VectorStoreIndex 和 StorageContext 成功")
except Exception as e:
    log(f"✗ 导入 VectorStoreIndex 和 StorageContext 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import load_index_from_storage
    log("✓ 导入 load_index_from_storage 成功")
except Exception as e:
    log(f"✗ 导入 load_index_from_storage 失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.vector_stores.faiss import FaissVectorStore
    log("✓ 导入 FaissVectorStore 成功")
except Exception as e:
    log(f"✗ 导入 FaissVectorStore 失败: {type(e).__name__}: {e}")
    sys.exit(1)

log("\n🎉 所有库导入成功！")
log("测试完成")