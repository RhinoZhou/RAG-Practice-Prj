#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试sentence_transformers导入的简单脚本
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始测试sentence_transformers导入")

try:
    print(f"[{time.strftime('%H:%M:%S')}] 正在导入sentence_transformers...")
    start_time = time.time()
    from sentence_transformers import SentenceTransformer
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入成功！耗时: {end_time - start_time:.2f} 秒")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 测试完成！")