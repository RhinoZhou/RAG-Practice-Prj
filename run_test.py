#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
"""
import os
import sys

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("开始运行程序...")

try:
    # 导入并运行主程序
    exec(open('03-hybrid_retrieval_rerank.py', encoding='utf-8').read())
    print("程序执行完成")
except Exception as e:
    print(f"程序执行出错: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
