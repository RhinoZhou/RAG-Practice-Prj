#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业智能知识库管理系统
基于向量检索和生成式AI的企业知识问答平台
"""

import time
import sys
import os
import shutil

def console_log(message):
    """控制台日志记录函数，带时间戳"""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    print(f"{timestamp} {message}")
    sys.stdout.flush()

# 程序初始化
console_log("企业智能知识库管理系统启动中...")

# 系统依赖加载
console_log("正在加载系统依赖...")

try:
    import torch
    console_log("✅ 机器学习框架加载成功")
except Exception as e:
    console_log(f"❌ 机器学习框架加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import faiss
    console_log("✅ 向量检索引擎加载成功")
except Exception as e:
    console_log(f"❌ 向量检索引擎加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import numpy as np
    console_log("✅ 数值计算库加载成功")
except Exception as e:
    console_log(f"❌ 数值计算库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import pandas as pd
    console_log("✅ 数据处理库加载成功")
except Exception as e:
    console_log(f"❌ 数据处理库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import nltk
    console_log("✅ 自然语言处理库加载成功")
except Exception as e:
    console_log(f"❌ 自然语言处理库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    console_log("✅ 进度显示库加载成功")
except Exception as e:
    console_log(f"❌ 进度显示库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import PyPDF2
    console_log("✅ PDF处理库加载成功")
except Exception as e:
    console_log(f"❌ PDF处理库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    import docx
    console_log("✅ Word处理库加载成功")
except Exception as e:
    console_log(f"❌ Word处理库加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

# 文档处理框架加载
try:
    from llama_index.core.node_parser import SentenceSplitter
    console_log("✅ 文档分块器加载成功")
except Exception as e:
    console_log(f"❌ 文档分块器加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex, StorageContext
    console_log("✅ 向量存储索引加载成功")
except Exception as e:
    console_log(f"❌ 向量存储索引加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.core import load_index_from_storage
    console_log("✅ 索引加载工具加载成功")
except Exception as e:
    console_log(f"❌ 索引加载工具加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    from llama_index.vector_stores.faiss import FaissVectorStore
    console_log("✅ FAISS向量存储加载成功")
except Exception as e:
    console_log(f"❌ FAISS向量存储加载失败: {type(e).__name__}: {e}")
    sys.exit(1)

console_log("✅ 所有系统依赖加载完成！")

# 系统配置
KNOWLEDGE_STORE = "./knowledge_repo"
VECTOR_INDEX_STORE = "./vector_indexes"

# 系统状态
active_index = None
file_registry = {}  # 文件索引注册表

# 创建系统目录结构
console_log(f"正在初始化系统目录...")
os.makedirs(KNOWLEDGE_STORE, exist_ok=True)
os.makedirs(VECTOR_INDEX_STORE, exist_ok=True)
console_log(f"✅ 系统目录初始化完成")

# 文档内容提取模块
def extract_pdf_content(file_path):
    """从PDF文件中提取文本内容"""
    try:
        with open(file_path, 'rb') as doc_file:
            pdf_reader = PyPDF2.PdfReader(doc_file)
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text() or ""
        return extracted_text
    except Exception as e:
        console_log(f"PDF文件读取错误: {e}")
        return ""

def extract_docx_content(file_path):
    """从Word文档中提取文本内容"""
    try:
        word_doc = docx.Document(file_path)
        extracted_text = ""
        for paragraph in word_doc.paragraphs:
            extracted_text += paragraph.text + "\n"
        return extracted_text
    except Exception as e:
        console_log(f"Word文档读取错误: {e}")
        return ""

def extract_txt_content(file_path):
    """从文本文件中提取内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as txt_file:
            return txt_file.read()
    except Exception as e:
        console_log(f"文本文件读取错误: {e}")
        return ""

def import_document(file_path):
    """导入并处理企业文档"""
    try:
        # 验证文件存在性
        if not os.path.exists(file_path):
            return f"❌ 指定的文件不存在: {file_path}", False
        
        # 获取文件名并复制到知识库
        doc_name = os.path.basename(file_path)
        target_path = os.path.join(KNOWLEDGE_STORE, doc_name)
        shutil.copy2(file_path, target_path)
        
        # 提取文件内容
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            doc_content = extract_pdf_content(target_path)
        elif file_ext == '.docx':
            doc_content = extract_docx_content(target_path)
        elif file_ext == '.txt':
            doc_content = extract_txt_content(target_path)
        else:
            # 清理已复制的文件
            os.remove(target_path)
            return f"❌ 不支持的文件格式: {file_ext}，仅支持PDF、Word和TXT格式", False
        
        if not doc_content or len(doc_content.strip()) < 10:
            # 清理已复制的文件
            os.remove(target_path)
            return f"❌ 文件内容为空或无法有效读取: {doc_name}", False
        
        # 注册文件
        file_registry[doc_name] = {
            'path': target_path,
            'size': os.path.getsize(target_path),
            'import_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'content_length': len(doc_content)
        }
        
        return f"✅ 文档导入成功: {doc_name}", True
    except Exception as e:
        return f"❌ 文档导入失败: {str(e)}", False

def remove_document(doc_name):
    """从知识库中移除指定文档"""
    try:
        doc_path = os.path.join(KNOWLEDGE_STORE, doc_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)
            # 清理文件注册表
            if doc_name in file_registry:
                del file_registry[doc_name]
            return f"✅ 文档已成功删除: {doc_name}", True
        else:
            return f"❌ 未找到指定文档: {doc_name}", False
    except Exception as e:
        return f"❌ 文档删除失败: {str(e)}", False

def list_documents():
    """列出知识库中的所有文档"""
    try:
        docs = os.listdir(KNOWLEDGE_STORE)
        if not docs:
            return "📁 知识库中暂无文档"
        else:
            result = "📋 知识库文档列表:\n"
            for doc in sorted(docs):
                if doc in file_registry:
                    result += f"• {doc} (导入时间: {file_registry[doc]['import_time']})\n"
                else:
                    result += f"• {doc}\n"
            return result
    except Exception as e:
        return f"❌ 获取文档列表失败: {str(e)}"

def clear_all_documents():
    """清空知识库中的所有文档和索引"""
    try:
        # 清空知识库目录
        if os.path.exists(KNOWLEDGE_STORE):
            shutil.rmtree(KNOWLEDGE_STORE)
        os.makedirs(KNOWLEDGE_STORE, exist_ok=True)
        
        # 清空向量索引目录
        if os.path.exists(VECTOR_INDEX_STORE):
            shutil.rmtree(VECTOR_INDEX_STORE)
        os.makedirs(VECTOR_INDEX_STORE, exist_ok=True)
        
        # 重置系统状态
        global active_index, file_registry
        active_index = None
        file_registry = {}
        
        return "✅ 知识库已完全清空", True
    except Exception as e:
        return f"❌ 清空知识库失败: {str(e)}", False

def enterprise_qa(query):
    """企业知识问答功能"""
    response = f"💡 关于'{query}'的智能回答：\n\n这是基于企业知识库的智能回复示例。\n\n📌 系统说明：\n- 目前使用演示模式，完整功能需要配置AI模型\n- 支持transformers和sentence_transformers等高级模型集成\n- 可根据企业需求定制问答逻辑和响应格式"
    return response

def display_main_menu():
    """显示主菜单界面"""
    print("\n" + "=" * 60)
    print("🎯 企业智能知识库管理系统 v1.0")
    print("=" * 60)
    print("1. 导入企业文档（支持PDF、Word、TXT格式）")
    print("2. 查看知识库")
    print("3. 删除指定文档")
    print("4. 清空知识库")
    print("5. 智能问答")
    print("0. 退出系统")
    print("=" * 60)

def main():
    """系统主入口函数"""
    console_log("企业智能知识库管理系统已就绪")
    
    while True:
        display_main_menu()
        user_choice = input("请选择操作 (0-5): ")
        
        if user_choice == "0":
            console_log("系统正在关闭...")
            print("\n👋 感谢使用企业智能知识库管理系统！")
            break
        elif user_choice == "1":
            doc_path = input("请输入文档路径: ")
            print("\n📥 正在导入文档...")
            result, success = import_document(doc_path)
            print(result)
        elif user_choice == "2":
            print("\n" + list_documents())
        elif user_choice == "3":
            doc_name = input("请输入要删除的文档名: ")
            result, success = remove_document(doc_name)
            print(result)
        elif user_choice == "4":
            confirm = input("⚠️  确定要清空所有文档和索引吗？此操作不可恢复！(yes/no): ")
            if confirm.lower() == "yes":
                result, success = clear_all_documents()
                print(result)
            else:
                print("✅ 操作已取消")
        elif user_choice == "5":
            user_query = input("请输入您的问题: ")
            print("\n🤖 正在生成智能回答...")
            response = enterprise_qa(user_query)
            print(f"\n{response}")
        else:
            print("❌ 无效的选择，请重新输入")
        
        # 添加操作完成提示
        if user_choice in ["1", "2", "3", "4", "5"]:
            input("\n📌 按回车键继续...")

if __name__ == "__main__":
    try:
        console_log("企业智能知识库管理系统启动成功！")
        main()
    except KeyboardInterrupt:
        console_log("系统已被用户中断")
        print("\n👋 感谢使用企业智能知识库管理系统！")
    except Exception as e:
        console_log(f"系统发生错误: {type(e).__name__}: {e}")
        print("\n❌ 系统异常退出，请检查日志信息")