import time
import sys
import os

print(f"[{time.strftime('%H:%M:%S')}] 开始测试核心RAG功能")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")

# 导入核心库
try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入核心库...")
    
    import torch
    print(f"[{time.strftime('%H:%M:%S')}] ✓ torch (版本: {torch.__version__})")
    
    import faiss
    print(f"[{time.strftime('%H:%M:%S')}] ✓ faiss")
    
    import numpy as np
    print(f"[{time.strftime('%H:%M:%S')}] ✓ numpy (版本: {np.__version__})")
    
    import fitz  # PyMuPDF
    print(f"[{time.strftime('%H:%M:%S')}] ✓ fitz (PyMuPDF, 版本: {fitz.__version__})")
    
    import chardet
    print(f"[{time.strftime('%H:%M:%S')}] ✓ chardet (版本: {chardet.__version__})")
    
    from typing import List, Dict, Any, Optional, Tuple
    print(f"[{time.strftime('%H:%M:%S')}] ✓ typing模块")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"[{time.strftime('%H:%M:%S')}] ✓ concurrent.futures")
    
    print(f"[{time.strftime('%H:%M:%S')}] 核心库导入完成")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入核心库时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入配置文件
try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入配置文件...")
    from config import AppConfig
    print(f"[{time.strftime('%H:%M:%S')}] ✓ AppConfig (版本: {getattr(AppConfig, 'version', '未知')})")
    print(f"[{time.strftime('%H:%M:%S')}]   KB_BASE_DIR: {AppConfig.knowledge_base_root}")
    print(f"[{time.strftime('%H:%M:%S')}]   DEFAULT_KB: {AppConfig.default_knowledge_base}")
    
    # 创建必要的目录
    KB_BASE_DIR = AppConfig.knowledge_base_root
    os.makedirs(KB_BASE_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 知识库目录创建成功: {KB_BASE_DIR}")
    
    DEFAULT_KB = AppConfig.default_knowledge_base
    DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
    os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 默认知识库目录创建成功: {DEFAULT_KB_DIR}")
    
    OUTPUT_DIR = AppConfig.temp_output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 临时输出目录创建成功: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入配置文件时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入RAG核心功能
try:
    print(f"[{time.strftime('%H:%M:%S')}] 导入RAG核心功能...")
    
    # 导入部分核心函数（不导入Gradio相关的函数）
    import rag
    
    # 测试知识库管理功能
    print(f"[{time.strftime('%H:%M:%S')}] 测试知识库管理功能...")
    
    # 获取知识库列表
    kbs = rag.get_knowledge_bases()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 当前知识库列表: {kbs}")
    
    # 测试文本分割功能
    test_text = "这是一个测试文本，用于测试文本分割功能。文本分割是RAG系统中的重要组成部分，它将长文本分割成适合处理的小块。"
    chunks = rag.split_text_semantically(test_text, chunk_size=100, chunk_overlap=20)
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 文本分割成功，得到 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"[{time.strftime('%H:%M:%S')}]   块 {i+1}: {chunk['chunk'][:50]}...")
    
    print(f"[{time.strftime('%H:%M:%S')}] 核心RAG功能测试完成！")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 导入或测试RAG核心功能时出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 所有测试通过！核心RAG功能正常工作。")
print(f"[{time.strftime('%H:%M:%S')}] 程序结束")