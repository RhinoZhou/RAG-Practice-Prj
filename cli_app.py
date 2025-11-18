#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medi-RAG 命令行界面
不依赖Gradio，提供基本的知识库问答功能
"""

import sys
import time
import rag

def main():
    """命令行界面主函数"""
    print("=" * 60)
    print("🎯 Medi-RAG 命令行界面")
    print("=" * 60)
    print(f"当前Python版本: {sys.version.split()[0]}")
    print(f"可用知识库: {rag.get_knowledge_bases()}")
    print("=" * 60)
    
    # 选择知识库
    kb_name = input("请选择知识库 (默认: default): ").strip()
    if not kb_name:
        kb_name = rag.DEFAULT_KB
    
    print(f"\n✅ 已选择知识库: {kb_name}")
    
    while True:
        print("\n" + "-" * 40)
        question = input("请输入您的问题 (输入 'exit' 退出): ").strip()
        
        if question.lower() == 'exit':
            print("\n👋 感谢使用Medi-RAG！")
            break
        
        if not question:
            print("⚠️  问题不能为空，请重新输入")
            continue
        
        print(f"\n🔍 正在处理问题: {question}")
        
        try:
            # 使用简单模式回答
            start_time = time.time()
            answer = rag.answer_question(question, kb_name, multi_hop=False)
            end_time = time.time()
            
            print(f"\n💡 回答: ")
            print(answer)
            print(f"\n⏱️  处理时间: {end_time - start_time:.2f}秒")
            
            # 询问是否需要多跳推理
            multi_hop_choice = input("\n是否需要使用多跳推理重新回答？(y/n): ").strip().lower()
            if multi_hop_choice == 'y':
                print(f"\n🔍 正在使用多跳推理处理问题...")
                start_time = time.time()
                answer, debug_info = rag.generate_answer_with_multi_hop(question, kb_name)
                end_time = time.time()
                
                print(f"\n💡 多跳推理回答: ")
                print(answer)
                print(f"\n🔧 推理步骤: ")
                for i, step in enumerate(debug_info["reasoning_steps"]):
                    print(f"  步骤 {i+1}: {step['thought'][:100]}...")
                print(f"\n⏱️  处理时间: {end_time - start_time:.2f}秒")
                
        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()