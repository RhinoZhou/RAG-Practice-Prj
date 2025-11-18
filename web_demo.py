# -*- coding: utf-8 -*-
'''
法律智能问答系统Web演示界面，基于Streamlit构建。
主要功能包括：
1. 用户友好的对话界面，支持多轮对话交互
2. 集成法律问题分类器，自动识别问题类型
3. 基于RAG技术，从法律知识库中检索相关内容
4. 调用大语言模型生成准确的法律回答
5. 展示回答的参考资料来源
该演示界面展示了完整的法律RAG系统工作流程。
'''

import json
import torch
import streamlit as st
import os
import sys
from typing import Tuple, List, Dict, Any

# 导入自定义模块
from RAG import *
from Questionary import QuestionClassifier


def setup_dependencies():
    """
    检查并自动安装必要的依赖包
    """
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'sentence-transformers',
        'pymilvus'
    ]
    
    print("正在检查并安装依赖包...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装，正在安装...")
            try:
                os.system(f"pip install {package}")
                print(f"✓ {package} 安装成功")
            except Exception as e:
                print(f"✗ {package} 安装失败: {str(e)}")


# 自动安装依赖
setup_dependencies()


# 配置页面
st.set_page_config(
    page_title="法律智能助手 - LawBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# 设置页面标题和介绍
st.title("法律智能助手 🤖")
st.markdown("专业的法律问答系统，基于RAG技术提供准确的法律信息和建议")


@st.cache_resource
def init_model() -> Tuple[Any, Any, Any, Any, QuestionClassifier]:
    """
    初始化模型组件，使用缓存避免重复加载
    
    Returns:
        Tuple: 包含LLM、嵌入模型、向量数据库、重排序模型和分类器的元组
    """
    try:
        # 使用相对路径，提高代码可移植性
        base_model_dir = './model_hub' if os.path.exists('./model_hub') else './models'
        
        print("正在加载模型组件...")
        
        # 加载大语言模型
        print("加载大语言模型...")
        try:
            # 尝试使用原始路径，同时提供相对路径选项
            llm_paths = [
                '/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct',
                os.path.join(base_model_dir, 'qwen/Qwen2-7B-Instruct'),
                os.path.join(base_model_dir, 'qwen')
            ]
            
            for path in llm_paths:
                try:
                    if os.path.exists(path):
                        llm = QwenModelChat(path)
                        print(f"成功加载LLM模型: {path}")
                        break
                except Exception:
                    continue
            else:
                # 如果所有路径都失败，尝试使用简单实现
                print("无法找到LLM模型，使用默认配置")
                llm = QwenModelChat()  # 假设构造函数有默认参数
                
        except Exception as e:
            print(f"LLM模型加载警告: {str(e)}")
            # 继续执行，希望能够使用默认构造函数
            llm = QwenModelChat()
        
        # 加载嵌入模型
        print("加载嵌入模型...")
        try:
            embedding_paths = [
                '/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/',
                os.path.join(base_model_dir, 'ZhipuAI/bge-large-zh-v1.5'),
                os.path.join(base_model_dir, 'bge-large-zh')
            ]
            
            for path in embedding_paths:
                try:
                    if os.path.exists(path):
                        embedding = BGEVectorizer(path)
                        print(f"成功加载嵌入模型: {path}")
                        break
                except Exception:
                    continue
            else:
                embedding = BGEVectorizer()
                
        except Exception as e:
            print(f"嵌入模型加载警告: {str(e)}")
            embedding = BGEVectorizer()
        
        # 加载向量数据库
        print("加载向量数据库...")
        try:
            vector_db_paths = [
                '/root/sunyd/llms/TinyRAG-master/storage/milvus_law.db',
                './storage/milvus_law.db',
                './data/milvus_law.db'
            ]
            
            for path in vector_db_paths:
                try:
                    if os.path.exists(os.path.dirname(path)) or os.path.exists(path):
                        vector = VectorStore(uri=path)
                        print(f"成功加载向量数据库: {path}")
                        break
                except Exception:
                    continue
            else:
                vector = VectorStore(uri='./milvus_law.db')
                
        except Exception as e:
            print(f"向量数据库加载警告: {str(e)}")
            vector = VectorStore(uri='./milvus_law.db')
        
        # 加载重排序模型
        print("加载重排序模型...")
        try:
            reranker_paths = [
                '/root/sunyd/model_hub/Xorbits/bge-reranker-base',
                os.path.join(base_model_dir, 'Xorbits/bge-reranker-base'),
                os.path.join(base_model_dir, 'bge-reranker-base')
            ]
            
            for path in reranker_paths:
                try:
                    if os.path.exists(path):
                        relevance_reranker = BgeReranker(path=path)
                        print(f"成功加载重排序模型: {path}")
                        break
                except Exception:
                    continue
            else:
                relevance_relevance_reranker = BgeReranker()
                
        except Exception as e:
            print(f"重排序模型加载警告: {str(e)}")
            reranker = BgeReranker()
        
        # 初始化问题分类器
        print("初始化问题分类器...")
        classifier = QuestionClassifier()
        
        print("所有模型组件加载完成！")
        return llm, embedding, vector, relevance_reranker, classifier
    
    except Exception as e:
        print(f"模型初始化错误: {str(e)}")
        st.error(f"系统初始化错误: {str(e)}")
        # 尝试返回可用的组件
        try:
            return llm, embedding, vector, relevance_reranker, QuestionClassifier()
        except:
            return None, None, None, None, None


def clear_chat_history():
    """
    清空对话历史记录
    """
    if "messages" in st.session_state:
        del st.session_state.messages
        st.success("对话历史已清空")


def init_chat_history() -> List[Dict[str, str]]:
    """
    初始化对话历史，显示欢迎消息和之前的对话内容
    
    Returns:
        List[Dict]: 对话历史消息列表
    """
    # 显示欢迎消息
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好！我是法律智能助手，很高兴为您提供法律咨询服务。请输入您的法律问题，我将尽力为您解答。")

    # 显示之前的对话消息
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        # 初始化空的对话历史
        st.session_state.messages = []

    return st.session_state.messages


def process_query(question: str, llm: Any, embedding: Any, vector: Any, 
                 relevance_reranker: Any, classifier: QuestionClassifier) -> str:
    """
    处理用户查询，包括问题分类、知识检索和回答生成
    
    Args:
        question (str): 用户问题
        llm: 大语言模型实例
        embedding: 嵌入模型实例
        vector: 向量数据库实例
        reranker: 重排序模型实例
        classifier: 问题分类器实例
        
    Returns:
        str: 生成的回答
    """
    try:
        # 对问题进行分类
        res_classify = classifier.classify(question)
        
        # 检查分类结果
        if len(res_classify['kg_names']) == 0:
            # 未识别到特定类别，直接使用LLM回答
            print(f"未识别到特定类别，直接使用LLM回答问题: {question}")
            prompt = llm.generate_prompt(question, "")
            answer = llm.chat(prompt)
            return answer
        else:
            print(f"识别到问题类别: {res_classify['kg_names']}")
            
            # 从向量数据库检索相关内容
            contents = []
            sim_query = []
            
            for collection_name in res_classify['kg_names']:
                print(f"从集合 {collection_name} 中检索相关内容...")
                try:
                    # 检索前k个最相关的内容
                    for content in vector.query(question, collection_name=collection_name, 
                                             vectorizer=embedding, k=3):
                        sim_query.append(content.key)
                        contents.append(content.value)
                except Exception as e:
                    print(f"检索集合 {collection_name} 时出错: {str(e)}")
            
            # 检查是否检索到内容
            if len(contents) == 0:
                # 未检索到相关内容，直接使用LLM回答
                print("未检索到相关内容，直接使用LLM回答")
                prompt = llm.generate_prompt(question, "")
                answer = llm.chat(prompt)
                return answer
            else:
                # 构建参考资料文本
                best_content = "参考资料："
                for i, sq in enumerate(contents, 1):
                    best_content += f'\n\n{i}. {sq}'
                
                # 显示参考资料
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("**参考资料：**")
                    for i, sq in enumerate(contents, 1):
                        st.markdown(f"{i}. {sq}")
                
                # 使用检索到的内容构建提示并生成回答
                print(f"使用检索到的{len(contents)}条内容生成回答")
                prompt = llm.generate_prompt(question, best_content)
                answer = llm.chat(prompt)
                return answer
    
    except Exception as e:
        print(f"处理查询时出错: {str(e)}")
        return f"很抱歉，处理您的问题时发生错误: {str(e)}"


def main():
    """
    主函数，运行Streamlit应用
    """
    # 初始回答文本
    default_answer = '您好！我是法律智能助手，请问有什么需要咨询的法律问题？'
    
    # 加载模型组件
    llm, embedding, vector, relevance_reranker, classifier = init_model()
    
    # 如果模型加载失败，显示错误信息并退出
    if llm is None:
        st.error("系统初始化失败，请检查模型文件路径和依赖安装情况。")
        return
    
    # 初始化对话历史
    messages = init_chat_history()
    
    # 显示示例问题提示
    with st.expander("💡 示例问题", expanded=False):
        st.markdown("""
        - 《民法典》中关于合同违约的规定有哪些？
        - 如何撰写一份有效的遗嘱？
        - 张三因盗窃罪被起诉，可能会面临什么处罚？
        - 推荐几本关于商法的经典著作
        - 法考题目：以下关于正当防卫的说法正确的是？
        """)
    
    # 处理用户输入
    if question := st.chat_input("请输入您的法律问题，按Enter发送"):
        # 显示用户消息
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(question)
        
        # 处理查询并生成回答
        answer = process_query(question, llm, embedding, vector, relevance_reranker, classifier)
        
        # 添加到对话历史
        messages.append({"role": "user", "content": question})
        
        # 显示助手回答
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown(answer)
            
            # 清理缓存（如果可用）
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # 添加回答到对话历史
        messages.append({"role": "assistant", "content": answer})
        
        # 打印对话历史到控制台（用于调试）
        print(json.dumps(messages, ensure_ascii=False), flush=True)
    
    # 清空对话按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("🔄 清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    # 确保中文显示正常
    st.markdown("<style>body { font-family: 'SimHei', 'WenQuanYi Micro Hei', sans-serif; }</style>", 
                unsafe_allow_html=True)
    
    # 运行主函数
    main()