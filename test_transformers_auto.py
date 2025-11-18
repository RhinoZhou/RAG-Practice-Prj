#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试transformers AutoTokenizer和AutoModel导入的脚本
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始测试transformers AutoTokenizer和AutoModel导入")

try:
    print(f"[{time.strftime('%H:%M:%S')}] 正在导入transformers...")
    start_time = time.time()
    import transformers
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ transformers基础导入成功！耗时: {end_time - start_time:.2f} 秒")
    print(f"[{time.strftime('%H:%M:%S')}] transformers版本: {transformers.__version__}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ transformers基础导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"[{time.strftime('%H:%M:%S')}] 正在导入AutoTokenizer和AutoModel...")
    start_time = time.time()
    from transformers import AutoTokenizer, AutoModel
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ AutoTokenizer和AutoModel导入成功！耗时: {end_time - start_time:.2f} 秒")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ AutoTokenizer和AutoModel导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 所有测试完成！")