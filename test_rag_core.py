#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行测试：不使用Gradio，只测试rag核心功能
"""

import sys
import time

print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")
print(f"[{time.strftime('%H:%M:%S')}] 开始测试rag核心功能...")

try:
    # 导入rag模块
    import rag
    print(f"[{time.strftime('%H:%M:%S')}] ✅ rag模块导入成功")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 默认知识库: {rag.DEFAULT_KB}")
    
    # 测试知识库功能
    print(f"\n[{time.strftime('%H:%M:%S')}] 测试知识库功能...")
    kbs = rag.get_knowledge_bases()
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 知识库列表: {kbs}")
    
    # 测试问题回答功能
    print(f"\n[{time.strftime('%H:%M:%S')}] 测试问题回答功能...")
    question = "什么是RAG技术？"
    print(f"[{time.strftime('%H:%M:%S')}] 问题: {question}")
    
    result = rag.ask_question(question, rag.DEFAULT_KB, use_multi_hop=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 回答: {result['answer'][:100]}...")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 推理步骤: {result['reasoning_steps'][:100]}...")
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 检索内容: {result['retrieved_chunks'][:100]}...")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 所有核心功能测试通过！")
    print(f"\n[{time.strftime('%H:%M:%S')}] 结论: rag核心功能正常工作，问题出在Gradio界面部分")
    print(f"[{time.strftime('%H:%M:%S')}] 建议: 尝试降级Python版本到3.12或使用更早版本的Gradio")
    
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] ❌ 测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)