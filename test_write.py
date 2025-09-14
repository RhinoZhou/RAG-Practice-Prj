#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件写入
"""
import os

print("开始测试文件写入...")

# 创建目录
os.makedirs("outputs", exist_ok=True)
print("✓ 创建目录")

# 测试写入文件
try:
    with open("outputs/test.txt", "w", encoding="utf-8") as f:
        f.write("测试文件写入\n")
        f.write("这是中文测试\n")
    print("✓ 成功写入文件")
except Exception as e:
    print(f"✗ 写入文件失败: {e}")

# 检查文件是否存在
if os.path.exists("outputs/test.txt"):
    print("✓ 文件存在")
    with open("outputs/test.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(f"文件内容: {content}")
else:
    print("✗ 文件不存在")

print("测试完成")

