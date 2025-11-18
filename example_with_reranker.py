# -*- coding: utf-8 -*-
'''
带重排序器(Reranker)的RAG（检索增强生成）系统示例程序。
本程序展示了如何结合向量数据库和重排序器进行更精确的文档检索，
然后使用大语言模型生成高质量回答。主要功能包括：
1. 文档读取与分割
2. 文本向量化与向量数据库存储
3. 两阶段检索：向量相似度搜索 + 重排序
4. 结合检索内容与LLM生成回答
5. 性能优化和效率提升
该示例特别强调了重排序在提升检索精度中的重要作用，适用于对检索准确性要求较高的场景。
'''

import os
import sys
import time
import re
from typing import Any, List, Dict, Optional

# 导入RAG系统组件
from RAG.VecDB import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import ZhipuChat, OpenAIChat, InternLMChat
from RAG.TxtEmbedding import BGEVectorizer, ZhipuVectorizer, JinaVectorizer
from RAG.Reranker import BgeReranker


def setup_dependencies():
    """
    检查并自动安装必要的依赖包，包括重排序器所需的额外依赖
    """
    # 基础依赖包
    required_packages = [
        'numpy',
        'pandas',
        'transformers',
        'sentence-transformers',
        'openai',
        'pymilvus',
        'tiktoken'
    ]
    
    # 重排序器额外依赖
    reranker_packages = [
        'torch',
        'scikit-learn',
        'bge-reranker'
    ]
    
    all_packages = required_packages + reranker_packages
    
    print("正在检查并安装依赖包...")
    for package in all_packages:
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


def setup_api_keys():
    """
    设置必要的API密钥
    """
    print("正在设置API密钥...")
    
    # 从环境变量获取API密钥或使用默认值
    if 'ZHIPUAI_API_KEY' not in os.environ or os.environ['ZHIPUAI_API_KEY'] == '*':
        # 尝试从配置文件读取
        if os.path.exists('config.py'):
            try:
                import config
                if hasattr(config, 'ZHIPUAI_API_KEY'):
                    os.environ['ZHIPUAI_API_KEY'] = config.ZHIPUAI_API_KEY
                    print("✓ 成功从config.py加载智谱AI API密钥")
            except Exception as e:
                print(f"✗ 无法从config.py加载API密钥: {str(e)}")
        
        # 如果还是没有API密钥，提示用户
        if 'ZHIPUAI_API_KEY' not in os.environ or os.environ['ZHIPUAI_API_KEY'] == '*':
            print("⚠ 警告: 智谱AI API密钥未设置，可能影响功能使用")
            print("  请在代码中或环境变量中设置ZHIPUAI_API_KEY")


def setup_vector_db_with_reranker(recreate_db: bool = False) -> VectorStore:
    """
    设置向量数据库，包括文档读取、向量化和持久化
    
    Args:
        recreate_db (bool): 是否重新创建向量数据库
        
    Returns:
        VectorStore: 初始化并存储了向量的数据库实例
    """
    try:
        # 检查storage目录是否存在，如果存在且不重建则跳过重建
        if not recreate_db and os.path.exists('./storage') and os.listdir('./storage'):
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
        embedding = BGEVectorizer()
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


def create_embedding_and_reranker() -> tuple[Any, Any]:
    """
    创建嵌入模型和重排序器
    
    Returns:
        tuple: (嵌入模型实例, 重排序器实例)
    """
    try:
        # 创建嵌入模型
        print("正在初始化嵌入模型...")
        embedding = BGEVectorizer()
        print("✓ 嵌入模型初始化完成")
        
        # 创建重排序器
        print("正在初始化重排序器...")
        relevance_reranker = BgeReranker()
        print("✓ 重排序器初始化完成")
        
        return embedding, relevance_reranker
    except Exception as e:
        print(f"初始化嵌入模型或重排序器时出错: {str(e)}")
        # 返回None值，后续会进行处理
        return None, None


def create_llm_instance() -> Optional[Any]:
    """
    创建语言模型实例，尝试多个模型作为备选
    
    Returns:
        语言模型实例或None
    """
    try:
        # 首先尝试ZhipuChat
        print("正在初始化语言模型 (ZhipuChat)...")
        llm = ZhipuChat()
        print("✓ ZhipuChat初始化成功")
        return llm
    except Exception as e1:
        print(f"ZhipuChat初始化失败，尝试OpenAIChat: {str(e1)}")
        try:
            # 尝试OpenAIChat
            llm = OpenAIChat(model='gpt-3.5-turbo-1106')
            print("✓ OpenAIChat初始化成功")
            return llm
        except Exception as e2:
            print(f"OpenAIChat初始化失败，尝试InternLMChat: {str(e2)}")
            try:
                # 尝试InternLMChat
                llm = InternLMChat()
                print("✓ InternLMChat初始化成功")
                return llm
            except Exception as e3:
                print(f"所有语言模型初始化失败: {str(e3)}")
                return None


def perform_rag_with_reranker(question: str, vector_db: VectorStore, embedding: Any, relevance_reranker: Any, llm: Any) -> str:
    """
    执行带重排序的RAG查询流程：向量检索 + 重排序 + 生成回答
    
    Args:
        question (str): 用户问题
        vector_db: 向量数据库实例
        embedding: 嵌入模型实例
        reranker: 重排序器实例
        llm: 语言模型实例
        
    Returns:
        str: 生成的回答
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 第一阶段：向量相似度搜索
        print(f"\n正在执行第一阶段检索（向量相似度搜索）: {question}")
        vector_search_start = time.time()
        
        # 从向量数据库检索相关内容
        try:
            # 检索最相似的3个文档
            initial_contents = vector_db.query(question, vectorizer=embedding, k=3)
            vector_search_time = time.time() - vector_search_start
            print(f"向量搜索完成，耗时: {vector_search_time:.2f} 秒")
            print(f"找到 {len(initial_contents)} 个候选文档")
            
            # 打印检索到的内容摘要
            for i, content in enumerate(initial_contents, 1):
                summary = str(content)[:80] + '...' if len(str(content)) > 80 else str(content)
                print(f"候选文档 {i}: {summary}")
                
        except Exception as e:
            print(f"向量搜索出错: {str(e)}")
            # 使用备用回答
            return f"抱歉，无法从知识库检索相关信息: {str(e)}"
        
        # 如果没有检索到内容，返回相应提示
        if not initial_contents:
            return "未找到与问题相关的信息"
        
        # 第二阶段：重排序
        print("\n正在执行第二阶段检索（重排序）...")
        rerank_start = time.time()
        
        try:
            # 从初始检索结果中用重排序器再次筛选出最相似的2个文档
            reranked_contents = relevance_reranker.rerank(question, initial_contents, k=2)
            rerank_time = time.time() - rerank_start
            print(f"重排序完成，耗时: {rerank_time:.2f} 秒")
            print(f"重排序后保留 {len(reranked_contents)} 个最相关文档")
            
            # 打印重排序后的内容摘要
            for i, content in enumerate(reranked_contents, 1):
                summary = str(content)[:80] + '...' if len(str(content)) > 80 else str(content)
                print(f"重排序后文档 {i}: {summary}")
                
        except Exception as e:
            print(f"重排序出错: {str(e)}")
            print("将使用初始检索结果继续处理")
            reranked_contents = initial_contents[:1]  # 使用第一个作为备选
        
        # 选择最相关的内容
        best_content = reranked_contents[0]
        
        # 如果没有LLM实例，直接返回检索到的内容
        if llm is None:
            return f"无法初始化语言模型，以下是检索到的最相关信息：\n{best_content}"
        
        # 生成回答
        print("\n正在生成回答...")
        gen_start = time.time()
        
        # 构建对话历史（空）
        history = []
        
        # 生成回答
        answer = llm.chat(question, history, best_content)
        
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        print(f"回答生成完成，耗时: {gen_time:.2f} 秒")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"  - 向量搜索: {vector_search_time:.2f} 秒")
        if 'rerank_time' in locals():
            print(f"  - 重排序: {rerank_time:.2f} 秒")
        print(f"  - 回答生成: {gen_time:.2f} 秒")
        
        return answer
        
    except Exception as e:
        print(f"执行RAG查询时出错: {str(e)}")
        return f"处理查询时发生错误: {str(e)}"


def run_example_queries_with_reranker():
    """
    运行带重排序器的示例查询，展示增强型RAG系统的功能
    """
    print("\n===== 带重排序器的RAG系统示例程序 =====")
    print("本示例展示了如何通过向量检索和重排序的两阶段检索策略提高RAG系统的准确性")
    
    # 设置API密钥
    setup_api_keys()
    
    # 设置向量数据库
    vector_db = setup_vector_db_with_reranker(recreate_db=False)
    
    # 创建嵌入模型和重排序器
    embedding, relevance_reranker = create_embedding_and_reranker()
    
    # 如果嵌入模型或重排序器创建失败，退出
    if embedding is None or relevance_reranker is None:
        print("\n错误: 无法初始化嵌入模型或重排序器，程序终止")
        return
    
    # 创建语言模型实例
    llm = create_llm_instance()
    
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
    print("\n执行流程：向量检索 → 重排序 → 回答生成")
    
    # 执行每个示例问题
    for i, question in enumerate(example_questions, 1):
        print(f"\n\n=== 问题 {i}/{len(example_questions)} ===")
        print(f"问题: {question}")
        
        # 执行带重排序的RAG查询
        answer = perform_rag_with_reranker(question, vector_db, embedding, relevance_reranker, llm)
        
        # 打印回答
        print(f"\n回答: {answer}")
        print("=" * 60)
    
    print("\n示例查询执行完成！")
    print("\n性能分析:")
    print("  带重排序器的RAG系统通过两阶段检索显著提高了检索精度")
    print("  1. 第一阶段向量检索快速缩小候选范围")
    print("  2. 第二阶段重排序对候选文档进行深度语义匹配")
    print("  这种方法在保持检索效率的同时，大幅提升了回答质量")


if __name__ == "__main__":
    # 运行带重排序器的示例查询
    run_example_queries_with_reranker()
