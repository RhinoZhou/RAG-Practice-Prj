print("开始测试导入流程...")

try:
    import torch
    print("成功导入 torch")
except Exception as e:
    print(f"导入 torch 失败: {e}")

try:
    import faiss
    print("成功导入 faiss")
except Exception as e:
    print(f"导入 faiss 失败: {e}")

try:
    import numpy as np
    print("成功导入 numpy")
except Exception as e:
    print(f"导入 numpy 失败: {e}")

try:
    from llama_index.core.node_parser import SentenceSplitter
    print("成功导入 SentenceSplitter")
except Exception as e:
    print(f"导入 SentenceSplitter 失败: {e}")

try:
    import re
    print("成功导入 re")
except Exception as e:
    print(f"导入 re 失败: {e}")

try:
    from typing import List, Dict, Any, Optional, Tuple
    print("成功导入 typing 模块")
except Exception as e:
    print(f"导入 typing 模块失败: {e}")

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("成功导入 concurrent.futures")
except Exception as e:
    print(f"导入 concurrent.futures 失败: {e}")

try:
    import json
    print("成功导入 json")
except Exception as e:
    print(f"导入 json 失败: {e}")

try:
    import shutil
    print("成功导入 shutil")
except Exception as e:
    print(f"导入 shutil 失败: {e}")

try:
    from openai import OpenAI
    print("成功导入 OpenAI")
except Exception as e:
    print(f"导入 OpenAI 失败: {e}")

try:
    import gradio as gr
    print("成功导入 gradio")
except Exception as e:
    print(f"导入 gradio 失败: {e}")

try:
    import fitz  # PyMuPDF
    print("成功导入 fitz (PyMuPDF)")
except Exception as e:
    print(f"导入 fitz (PyMuPDF) 失败: {e}")

try:
    import chardet  # 用于自动检测编码
    print("成功导入 chardet")
except Exception as e:
    print(f"导入 chardet 失败: {e}")

try:
    import traceback
    print("成功导入 traceback")
except Exception as e:
    print(f"导入 traceback 失败: {e}")

print("\n所有基础库导入完成")

# 尝试导入配置文件
try:
    from config import AppConfig  # 导入配置文件
    print("成功导入 AppConfig")
    # 测试配置文件是否可用
    config = AppConfig()
    print(f"配置文件可用，API_KEY: {config.API_KEY[:5]}...")
except Exception as e:
    print(f"导入配置文件失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n测试流程完成")