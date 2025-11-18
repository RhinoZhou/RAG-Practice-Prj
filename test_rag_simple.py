#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import time

def run_test():
    """运行简单的rag.py测试"""
    print(f"[{time.strftime('%H:%M:%S')}] 开始测试rag.py...")
    
    # 先测试基本导入
    print(f"[{time.strftime('%H:%M:%S')}] 测试1: 基本库导入")
    try:
        import torch
        import faiss
        import numpy as np
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 基本库导入成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 基本库导入失败: {e}")
        return False
    
    # 测试llama_index导入
    print(f"[{time.strftime('%H:%M:%S')}] 测试2: llama_index导入")
    try:
        from llama_index.core.node_parser import SentenceSplitter
        print(f"[{time.strftime('%H:%M:%S')}] ✓ SentenceSplitter导入成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ SentenceSplitter导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试配置文件导入
    print(f"[{time.strftime('%H:%M:%S')}] 测试3: 配置文件导入")
    try:
        from config import AppConfig
        print(f"[{time.strftime('%H:%M:%S')}] ✓ AppConfig导入成功，KB_BASE_DIR={AppConfig.knowledge_base_root}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 配置文件导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试目录创建
    print(f"[{time.strftime('%H:%M:%S')}] 测试4: 目录创建")
    try:
        KB_BASE_DIR = AppConfig.knowledge_base_root
        os.makedirs(KB_BASE_DIR, exist_ok=True)
        DEFAULT_KB = AppConfig.default_knowledge_base
        DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
        os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
        OUTPUT_DIR = AppConfig.temp_output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 目录创建成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ 目录创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"[{time.strftime('%H:%M:%S')}] 所有测试通过！")
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)