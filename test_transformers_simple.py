#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试transformers基础导入的简单脚本
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] 开始测试transformers基础导入")

try:
    print(f"[{time.strftime('%H:%M:%S')}] 正在导入transformers基础模块...")
    start_time = time.time()
    import transformers
    end_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 导入成功！耗时: {end_time - start_time:.2f} 秒")
    print(f"[{time.strftime('%H:%M:%S')}] transformers版本: {transformers.__version__}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ✗ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] 测试完成！")