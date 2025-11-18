#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medi-RAG 命令行界面 (CLI)
无需Gradio，可直接使用所有RAG功能
"""

import os
import sys
import time
import shutil
import argparse
import importlib.metadata

# 设置环境变量以跳过重复的依赖安装
os.environ['RAG_DEPENDENCIES_INSTALLED'] = '1'

class MediRAGCLI:
    """Medi-RAG 命令行界面类"""
    
    def __init__(self):
        self.rag = None
        self.initialize()
    
    def initialize(self):
        """初始化RAG系统"""
        print("=" * 60)
        print("🎯 Medi-RAG 命令行界面")
        print("=" * 60)
        print(f"Python版本: {sys.version.split()[0]}")
        
        try:
            # 导入rag模块
            import rag
            self.rag = rag
            
            # 获取版本信息
            try:
                version = importlib.metadata.version('medi-rag')
                print(f"Medi-RAG版本: {version}")
            except:
                print("Medi-RAG版本: 开发版")
            
            print("=" * 60)
            print("系统初始化成功！")
            
        except Exception as e:
            print(f"\n❌ 初始化失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def list_knowledge_bases(self):
        """列出所有知识库"""
        try:
            kbs = self.rag.get_knowledge_bases()
            print(f"\n📚 知识库列表 ({len(kbs)}个):")
            for i, kb in enumerate(kbs, 1):
                print(f"   {i}. {kb}")
            return kbs
        except Exception as e:
            print(f"\n❌ 获取知识库列表失败: {type(e).__name__}: {e}")
            return []
    
    def create_knowledge_base(self, kb_name):
        """创建知识库"""
        try:
            result = self.rag.create_knowledge_base(kb_name)
            print(f"\n✅ 成功创建知识库: {kb_name}")
            return True
        except Exception as e:
            print(f"\n❌ 创建知识库失败: {type(e).__name__}: {e}")
            return False
    
    def delete_knowledge_base(self, kb_name):
        """删除知识库"""
        try:
            # 确认操作
            confirm = input(f"\n⚠️  确定要删除知识库 '{kb_name}' 吗？(y/n): ").strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return False
            
            result = self.rag.delete_knowledge_base(kb_name)
            print(f"\n✅ 成功删除知识库: {kb_name}")
            return True
        except Exception as e:
            print(f"\n❌ 删除知识库失败: {type(e).__name__}: {e}")
            return False
    
    def list_files_in_kb(self, kb_name):
        """列出知识库中的文件"""
        try:
            files = self.rag.get_kb_files(kb_name)
            if files:
                print(f"\n📄 知识库 '{kb_name}' 中的文件 ({len(files)}个):")
                for i, file in enumerate(files, 1):
                    print(f"   {i}. {file}")
            else:
                print(f"\n📄 知识库 '{kb_name}' 中没有文件")
            return files
        except Exception as e:
            print(f"\n❌ 获取文件列表失败: {type(e).__name__}: {e}")
            return []
    
    def add_files_to_kb(self, kb_name, file_paths):
        """添加文件到知识库"""
        try:
            # 检查文件是否存在
            existing_files = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                else:
                    print(f"\n⚠️ 文件不存在: {file_path}")
            
            if not existing_files:
                print("没有有效文件可添加")
                return False
            
            print(f"\n🔄 正在处理 {len(existing_files)} 个文件...")
            result = self.rag.process_and_index_files(existing_files, kb_name)
            print(f"\n✅ 文件处理完成: {result}")
            return True
        except Exception as e:
            print(f"\n❌ 添加文件失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def answer_question(self, question, kb_name, use_multi_hop=False, use_table_format=False):
        """回答问题"""
        try:
            print(f"\n🔍 正在处理问题: {question}")
            print(f"   知识库: {kb_name}")
            print(f"   多跳推理: {'开启' if use_multi_hop else '关闭'}")
            print(f"   表格格式: {'开启' if use_table_format else '关闭'}")
            
            start_time = time.time()
            
            if use_multi_hop:
                # 使用多跳推理
                answer, debug_info = self.rag.generate_answer_with_multi_hop(question, kb_name)
                
                print(f"\n💡 多跳推理回答:")
                print(f"\n{answer}")
                
                if debug_info and 'reasoning_steps' in debug_info:
                    print(f"\n🔧 推理步骤:")
                    for i, step in enumerate(debug_info['reasoning_steps'], 1):
                        print(f"   步骤 {i}: {step['thought']}")
                        if 'query' in step:
                            print(f"      查询: {step['query']}")
                        if 'results' in step:
                            print(f"      结果: {step['results'][:100]}...")
            else:
                # 使用简单检索
                answer = self.rag.answer_question(question, kb_name, use_table_format=use_table_format)
                print(f"\n💡 回答:")
                print(f"\n{answer}")
            
            end_time = time.time()
            print(f"\n⏱️  处理时间: {end_time - start_time:.2f}秒")
            
            return True
        except Exception as e:
            print(f"\n❌ 回答问题失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("🎉 进入交互式模式")
        print("=" * 60)
        print("输入 'help' 查看可用命令")
        print("输入 'exit' 退出程序")
        print("=" * 60)
        
        # 默认知识库
        current_kb = self.rag.DEFAULT_KB if hasattr(self.rag, 'DEFAULT_KB') else 'default'
        
        while True:
            try:
                command = input(f"\n{current_kb}> ").strip().lower()
                
                if not command:
                    continue
                
                if command == 'exit' or command == 'quit':
                    print("\n👋 感谢使用Medi-RAG！")
                    break
                
                elif command == 'help':
                    self.show_help()
                
                elif command == 'kb list':
                    self.list_knowledge_bases()
                
                elif command.startswith('kb create '):
                    kb_name = command[10:].strip()
                    if kb_name:
                        self.create_knowledge_base(kb_name)
                    else:
                        print("请指定知识库名称")
                
                elif command.startswith('kb delete '):
                    kb_name = command[10:].strip()
                    if kb_name:
                        self.delete_knowledge_base(kb_name)
                    else:
                        print("请指定知识库名称")
                
                elif command.startswith('kb use '):
                    kb_name = command[7:].strip()
                    if kb_name:
                        # 检查知识库是否存在
                        kbs = self.rag.get_knowledge_bases()
                        if kb_name in kbs:
                            current_kb = kb_name
                            print(f"\n✅ 已切换到知识库: {current_kb}")
                        else:
                            print(f"\n❌ 知识库不存在: {kb_name}")
                    else:
                        print("请指定知识库名称")
                
                elif command == 'kb files':
                    self.list_files_in_kb(current_kb)
                
                elif command.startswith('kb add '):
                    file_paths = command[7:].strip().split()
                    if file_paths:
                        self.add_files_to_kb(current_kb, file_paths)
                    else:
                        print("请指定文件路径")
                
                elif command.startswith('ask '):
                    question = command[4:].strip()
                    if question:
                        self.answer_question(question, current_kb)
                    else:
                        print("请输入问题")
                
                elif command.startswith('ask --multi-hop '):
                    question = command[15:].strip()
                    if question:
                        self.answer_question(question, current_kb, use_multi_hop=True)
                    else:
                        print("请输入问题")
                
                elif command.startswith('ask --table '):
                    question = command[12:].strip()
                    if question:
                        self.answer_question(question, current_kb, use_table_format=True)
                    else:
                        print("请输入问题")
                
                else:
                    print(f"\n❌ 未知命令: {command}")
                    print("输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用Medi-RAG！")
                break
            except Exception as e:
                print(f"\n❌ 命令执行失败: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📋 可用命令:")
        print("=" * 40)
        print("基础命令:")
        print("  help              - 显示此帮助信息")
        print("  exit/quit         - 退出程序")
        print("\n知识库管理:")
        print("  kb list           - 列出所有知识库")
        print("  kb create <name>  - 创建新知识库")
        print("  kb delete <name>  - 删除知识库")
        print("  kb use <name>     - 切换当前知识库")
        print("  kb files          - 查看当前知识库中的文件")
        print("  kb add <files>    - 添加文件到当前知识库")
        print("\n问答功能:")
        print("  ask <question>                - 提问（简单检索）")
        print("  ask --multi-hop <question>    - 提问（多跳推理）")
        print("  ask --table <question>        - 提问（表格格式）")
        print("=" * 40)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Medi-RAG 命令行界面')
    parser.add_argument('--mode', choices=['interactive', 'single'], default='interactive',
                        help='运行模式: interactive（交互式）或 single（单次查询）')
    parser.add_argument('--kb', help='指定知识库名称')
    parser.add_argument('--question', help='问题内容（仅在single模式下使用）')
    parser.add_argument('--multi-hop', action='store_true', help='使用多跳推理')
    parser.add_argument('--table', action='store_true', help='使用表格格式')
    
    args = parser.parse_args()
    
    # 创建CLI实例
    cli = MediRAGCLI()
    
    if args.mode == 'single':
        # 单次查询模式
        if not args.question:
            parser.error('--question 参数是必需的（在single模式下）')
        
        kb_name = args.kb or (cli.rag.DEFAULT_KB if hasattr(cli.rag, 'DEFAULT_KB') else 'default')
        cli.answer_question(args.question, kb_name, use_multi_hop=args.multi_hop, use_table_format=args.table)
    else:
        # 交互式模式
        cli.interactive_mode()

if __name__ == "__main__":
    main()