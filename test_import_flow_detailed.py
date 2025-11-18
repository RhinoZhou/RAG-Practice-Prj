import sys
import time
import os

def log(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

log("开始模拟rag.py的完整导入流程...")
log(f"Python版本: {sys.version}")

# 记录每个步骤的时间
timings = {}

def time_step(step_name):
    timings[step_name] = time.time()
    log(f"开始步骤: {step_name}")

# 测试1: 基本导入
time_step("基本导入")
try:
    import subprocess
    import sys
    import os
    log("成功导入基本模块")
except Exception as e:
    log(f"基本导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: install_dependencies函数
time_step("定义install_dependencies函数")
try:
    def install_dependencies():
        required_packages = [
            "torch",
            "faiss-cpu",
            "numpy",
            "llama-index-core",
            "PyMuPDF",
            "chardet",
            "gradio"
        ]
        log(f"依赖包列表: {required_packages}")
        log("注意: 实际安装已跳过，假设依赖已安装")
    
    install_dependencies()
    log("成功定义并调用install_dependencies函数")
except Exception as e:
    log(f"install_dependencies函数测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: time和log_import函数
time_step("导入time和定义log_import函数")
try:
    import time
    
    def log_import(module_name):
        """记录导入操作的时间戳"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] 正在导入 {module_name}...")
    
    log("成功导入time和定义log_import函数")
except Exception as e:
    log(f"time和log_import函数测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 逐个导入核心模块
core_modules = [
    ("torch", "import torch"),
    ("faiss", "import faiss"),
    ("numpy", "import numpy as np"),
    ("SentenceSplitter from llama_index.core.node_parser", "from llama_index.core.node_parser import SentenceSplitter"),
    ("re", "import re"),
    ("typing modules", "from typing import List, Dict, Any, Optional, Tuple"),
    ("concurrent.futures", "from concurrent.futures import ThreadPoolExecutor, as_completed"),
    ("json", "import json"),
    ("shutil", "import shutil"),
    ("fitz (PyMuPDF)", "import fitz"),
    ("chardet", "import chardet"),
    ("traceback", "import traceback"),
]

time_step("导入核心模块")
for module_name, import_code in core_modules:
    log_import(module_name)
    try:
        exec(import_code)
        log(f"✓ 成功导入 {module_name}")
    except Exception as e:
        log(f"✗ 导入 {module_name} 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 测试5: 导入配置文件
time_step("导入配置文件")
try:
    from config import AppConfig
    log("✓ 成功导入 AppConfig")
    log(f"  配置项测试: KB_BASE_DIR={AppConfig.knowledge_base_root}")
except Exception as e:
    log(f"✗ 导入配置文件失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 创建目录
time_step("创建目录")
try:
    KB_BASE_DIR = AppConfig.knowledge_base_root
    os.makedirs(KB_BASE_DIR, exist_ok=True)
    log(f"✓ 成功创建知识库根目录: {KB_BASE_DIR}")
    
    DEFAULT_KB = AppConfig.default_knowledge_base
    DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
    os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
    log(f"✓ 成功创建默认知识库目录: {DEFAULT_KB_DIR}")
    
    OUTPUT_DIR = AppConfig.temp_output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"✓ 成功创建临时输出目录: {OUTPUT_DIR}")
except Exception as e:
    log(f"✗ 创建目录失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试7: 检查是否能继续执行后续代码
time_step("测试后续代码执行")
try:
    # 定义一些后续代码中会用到的变量
    client = None
    
    # 定义一个简单的函数
    def test_function():
        return "测试函数执行成功"
    
    result = test_function()
    log(f"✓ 后续代码执行成功: {result}")
    
    # 测试是否能访问导入的模块
    log(f"✓ 能访问numpy: {np.array([1, 2, 3])}")
    log(f"✓ 能访问SentenceSplitter: {SentenceSplitter.__name__}")
except Exception as e:
    log(f"✗ 后续代码执行失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 打印时间统计
log("\n=== 导入流程时间统计 ===")
total_time = 0
for step, start_time in timings.items():
    end_time = time.time()
    step_time = end_time - start_time
    total_time += step_time
    log(f"{step}: {step_time:.2f}秒")
log(f"总时间: {total_time:.2f}秒")

log("\n🎉 所有测试步骤都成功完成！rag.py的导入流程没有问题。")
log("如果程序仍然提前退出，问题可能在:")
log("1. 实际运行环境与测试环境的差异")
log("2. 某些模块的导入有副作用")
log("3. 程序在导入完成后有其他逻辑导致退出")