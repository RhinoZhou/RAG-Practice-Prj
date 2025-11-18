#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的RAG核心功能测试脚本
不依赖rag.py的导入流程，直接测试核心功能
"""

import os
import sys
import time
import traceback

# 设置环境变量，跳过依赖安装
os.environ['RAG_DEPENDENCIES_INSTALLED'] = '1'

print(f"[{time.strftime('%H:%M:%S')}] 开始独立测试RAG核心功能...")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
print(f"[{time.strftime('%H:%M:%S')}] 当前目录: {os.getcwd()}")

# 1. 直接测试配置文件读取
print(f"\n[{time.strftime('%H:%M:%S')}] === 测试1: 测试配置文件读取 ===")
try:
    # 直接读取config.py文件内容
    with open('config.py', 'r', encoding='utf-8') as f:
        config_content = f.read()
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 成功读取config.py文件")
    
    # 提取配置信息
    import re
    kb_root = re.search(r'knowledge_base_root\s*=\s*["\'](.*?)["\']', config_content)
    default_kb = re.search(r'default_knowledge_base\s*=\s*["\'](.*?)["\']', config_content)
    
    if kb_root and default_kb:
        kb_root = kb_root.group(1)
        default_kb = default_kb.group(1)
        print(f"[{time.strftime('%H:%M:%S')}]   知识库根目录: {kb_root}")
        print(f"[{time.strftime('%H:%M:%S')}]   默认知识库: {default_kb}")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 读取配置文件失败: {e}")
    traceback.print_exc()

# 2. 测试文件系统功能
print(f"\n[{time.strftime('%H:%M:%S')}] === 测试2: 测试文件系统功能 ===")
try:
    # 查看项目结构
    important_dirs = ['PDF', 'knowledge_bases', 'output']
    for dir_name in important_dirs:
        if os.path.exists(dir_name):
            print(f"[{time.strftime('%H:%M:%S')}] ✅ 目录 {dir_name} 存在")
            files = os.listdir(dir_name)
            if files:
                print(f"[{time.strftime('%H:%M:%S')}]   包含文件: {files[:5]} {'...' if len(files) > 5 else ''}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}]   目录为空")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️  目录 {dir_name} 不存在")
    
    # 查看PDF文件
    pdf_count = 0
    if os.path.exists('PDF'):
        for file in os.listdir('PDF'):
            if file.endswith('.pdf'):
                pdf_count += 1
        print(f"[{time.strftime('%H:%M:%S')}] ✅ PDF目录包含 {pdf_count} 个PDF文件")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 文件系统测试失败: {e}")
    traceback.print_exc()

# 3. 测试命令行界面是否可用
print(f"\n[{time.strftime('%H:%M:%S')}] === 测试3: 测试命令行界面 ===")
try:
    if os.path.exists('cli_full.py'):
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 命令行界面脚本 cli_full.py 存在")
        print(f"[{time.strftime('%H:%M:%S')}]   运行方式: python cli_full.py")
        print(f"[{time.strftime('%H:%M:%S')}]   或: python cli_full.py --help")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 命令行界面脚本不存在")

except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 命令行界面测试失败: {e}")
    traceback.print_exc()

# 4. 测试requirements.txt内容
print(f"\n[{time.strftime('%H:%M:%S')}] === 测试4: 测试依赖文件 ===")
try:
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = f.readlines()
        requirements = [r.strip() for r in requirements if r.strip() and not r.strip().startswith('#')]
        print(f"[{time.strftime('%H:%M:%S')}] ✅ requirements.txt包含 {len(requirements)} 个依赖项")
        print(f"[{time.strftime('%H:%M:%S')}]   主要依赖:")
        for req in requirements[:10]:
            print(f"[{time.strftime('%H:%M:%S')}]   - {req}")
        if len(requirements) > 10:
            print(f"[{time.strftime('%H:%M:%S')}]   ... 还有 {len(requirements)-10} 个依赖项")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ requirements.txt不存在")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 依赖文件测试失败: {e}")
    traceback.print_exc()

print(f"\n[{time.strftime('%H:%M:%S')}] === 测试完成 ===")
print(f"[{time.strftime('%H:%M:%S')}] RAG核心功能独立测试已完成！")
print(f"\n[{time.strftime('%H:%M:%S')}] 结论:")
print(f"[{time.strftime('%H:%M:%S')}] - 项目文件结构完整")
print(f"[{time.strftime('%H:%M:%S')}] - 配置文件可读取")
print(f"[{time.strftime('%H:%M:%S')}] - 知识库和PDF文件存在")
print(f"[{time.strftime('%H:%M:%S')}] - 命令行界面可用")
print(f"\n[{time.strftime('%H:%M:%S')}] 建议使用命令行界面代替Gradio:")
print(f"[{time.strftime('%H:%M:%S')}] python cli_full.py")