# -*- coding: utf-8 -*-
'''
RAG（检索增强生成）系统示例程序。
本程序展示了如何使用向量数据库存储文档向量，并基于用户查询检索相关内容，
然后使用大语言模型生成回答。主要功能包括：
1. 文档读取与分割
2. 文本向量化与存储
3. 向量数据库的持久化与加载
4. 基于相似度的文档检索
5. 结合检索内容与LLM生成回答
该示例展示了RAG系统的完整工作流程，适用于各类知识库问答场景。
'''

import os
import sys
import time
from typing import Any, List, Dict

# 导入RAG系统组件
from RAG.VecDB import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.TxtEmbedding import JinaVectorizer, ZhipuVectorizer


def setup_dependencies():
    """
    检查并自动安装必要的依赖包
    """
    required_packages = [
        'numpy',
        'pandas',
        'transformers',
        'sentence-transformers',
        'openai',
        'pymilvus',
        'tiktoken'
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


def setup_vector_db() -> VectorStore:
    """
    设置向量数据库，包括文档读取、向量化和持久化
    
    Returns:
        VectorStore: 初始化并存储了向量的数据库实例
    """
    try:
        # 检查storage目录是否存在，如果存在则跳过重建
        if os.path.exists('./storage') and os.listdir('./storage'):
            print("检测到已存在的向量数据库，将直接加载...")
            vector_db = VectorStore()
            vector_db.load_vector('./storage')
            print("向量数据库加载完成")
            return vector_db
        
        # 读取文档并分割
        print("正在读取文档...")
        # 检查data目录是否存在
        if not os.path.exists('./data'):
            print("警告: ./data目录不存在，创建示例数据...")
            os.makedirs('./data', exist_ok=True)
            # 创建示例文档
            with open('./data/example_legal_docs.txt', 'w', encoding='utf-8') as f:
                f.write("""《民法典》第一千零一十二条 自然人享有姓名权，有权依法决定、使用、变更或者许可他人使用自己的姓名，但是不得违背公序良俗。

《刑法》第二百六十四条 盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。

正当防卫，指对正在进行不法侵害行为的人，而采取的制止不法侵害的行为，对不法侵害人造成损害的，属于正当防卫，不负刑事责任。

合同违约是指合同当事人完全没有履行合同或者履行合同义务不符合约定的行为。一般说来，违约行为从属于违法行为。民事违法行为就包括民事违约和民事侵权两类。

遗嘱是指遗嘱人生前在法律允许的范围内，按照法律规定的方式对其遗产或其他事务所作的个人处分，并于遗嘱人死亡时发生效力的法律行为。
                """)
            print("示例数据创建完成")
        
        # 读取并分割文档
        docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
        print(f"成功读取并分割 {len(docs)} 个文档片段")
        
        # 创建向量数据库并添加文档
        vector_db = VectorStore(docs)
        
        # 创建嵌入模型
        embedding = ZhipuVectorizer()
        print("正在生成文档向量...")
        
        # 生成向量
        start_time = time.time()
        vector_db.get_vector(vectorizer=embedding)
        end_time = time.time()
        print(f"向量生成完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 持久化向量数据库
        print("正在保存向量数据库...")
        vector_db.persist(path='storage')
        print("向量数据库保存完成")
        
        return vector_db
        
    except Exception as e:
        print(f"设置向量数据库时出错: {str(e)}")
        # 返回空的向量数据库实例
        return VectorStore()


def perform_rag_query(question: str, vector_db: VectorStore) -> str:
    """
    执行RAG查询流程：检索相关文档并生成回答
    
    Args:
        question (str): 用户问题
        vector_db: 向量数据库实例
        
    Returns:
        str: 生成的回答
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 创建嵌入模型
        embedding = ZhipuVectorizer()
        
        # 从向量数据库检索相关内容
        print(f"\n正在检索与问题相关的文档: {question}")
        search_start = time.time()
        
        # 尝试检索，如果失败则使用备用方法
        try:
            # 检索最相关的3个文档
            relevant_contents = vector_db.query(question, vectorizer=embedding, k=3)
            print(f"检索完成，耗时: {time.time() - search_start:.2f} 秒")
            print(f"找到 {len(relevant_contents)} 个相关文档")
            
            # 打印检索到的内容摘要
            for i, content in enumerate(relevant_contents, 1):
                summary = str(content)[:100] + '...' if len(str(content)) > 100 else str(content)
                print(f"相关文档 {i}: {summary}")
                
        except Exception as e:
            print(f"检索出错: {str(e)}")
            # 使用备用回答
            return f"抱歉，无法从知识库检索相关信息: {str(e)}"
        
        # 如果没有检索到内容，返回相应提示
        if not relevant_contents:
            return "未找到与问题相关的信息"
        
        # 选择最相关的内容
        best_content = relevant_contents[0]
        
        # 创建LLM实例
        print("正在初始化语言模型...")
        try:
            # 尝试使用OpenAI模型
            chat = OpenAIChat(model='gpt-3.5-turbo-1106')
        except Exception as e:
            print(f"OpenAI模型初始化失败，尝试使用InternLM模型: {str(e)}")
            try:
                # 尝试使用InternLM模型
                chat = InternLMChat()
            except Exception as e2:
                print(f"InternLM模型初始化失败: {str(e2)}")
                # 直接返回检索到的内容
                return f"无法初始化语言模型，以下是检索到的相关信息：\n{best_content}"
        
        # 生成回答
        print("正在生成回答...")
        gen_start = time.time()
        
        # 构建对话历史（空）
        history = []
        
        # 生成回答
        answer = chat.chat(question, history, best_content)
        
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        print(f"回答生成完成，耗时: {gen_time:.2f} 秒")
        print(f"总耗时: {total_time:.2f} 秒")
        
        return answer
        
    except Exception as e:
        print(f"执行RAG查询时出错: {str(e)}")
        return f"处理查询时发生错误: {str(e)}"


def run_example_queries():
    """
    运行示例查询，展示RAG系统的功能
    """
    print("\n===== RAG系统示例程序 =====")
    
    # 设置向量数据库
    vector_db = setup_vector_db()
    
    # 示例问题列表
    example_questions = [
        "《民法典》中关于姓名权的规定是什么？",
        "盗窃罪的量刑标准是什么？",
        "正当防卫的定义是什么？",
        "合同违约指的是什么？",
        "什么是遗嘱？"
    ]
    
    print("\n准备执行示例查询...")
    print(f"共有 {len(example_questions)} 个示例问题")
    
    # 执行每个示例问题
    for i, question in enumerate(example_questions, 1):
        print(f"\n\n=== 问题 {i}/{len(example_questions)} ===")
        print(f"问题: {question}")
        
        # 执行RAG查询
        answer = perform_rag_query(question, vector_db)
        
        # 打印回答
        print(f"\n回答: {answer}")
        print("=" * 50)
    
    print("\n示例查询执行完成！")


if __name__ == "__main__":
    # 运行示例查询
    run_example_queries()

