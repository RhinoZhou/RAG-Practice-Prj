#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查知识库文件的内容和编码
"""

import os
import sys

KB_DIR = "knowledge_bases/default"

print(f"Python解释器默认编码: {sys.getdefaultencoding()}")
print()

# 读取并显示文件内容
for filename in ["diabetes.txt", "hypertension.txt"]:
    file_path = os.path.join(KB_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"=== 文件: {filename} ===")
        print(f"内容:\n{content}")
        print(f"编码: UTF-8")
        print("-" * 50)
    except Exception as e:
        print(f"读取 {filename} 失败: {e}")
        print("-" * 50)