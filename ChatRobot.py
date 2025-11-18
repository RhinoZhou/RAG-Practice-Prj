# -*- coding: utf-8 -*-
'''
功能说明：
    本程序是一个基于法律知识库的智能聊天机器人实现。
    主要功能包括：
    1. 接收用户输入的法律问题
    2. 使用问题分类器对问题进行分类
    3. 根据分类结果从向量数据库中检索相关法律知识
    4. 使用大语言模型生成回答
    5. 支持基于检索增强生成(RAG)的智能问答
    
    程序架构：
    - 使用QuestionClassifier进行问题分类
    - 使用QwenModelChat作为语言模型
    - 使用BgeEmbedding进行文本嵌入
    - 使用VectorStore进行向量检索
    - 使用BgeReranker进行结果重排序
'''

# 自动安装依赖的代码
import subprocess
import sys

def setup_dependencies():
    """
    自动安装必要的依赖包
    """
    try:
        # 尝试导入必要的包，如果失败则安装
        import torch
        import transformers
        import numpy as np
    except ImportError:
        print("正在安装必要的依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "numpy"])
        print("依赖包安装完成")

# 安装依赖
setup_dependencies()

# 导入必要的模块
from RAG import *
from Questionary import QuestionClassifier


class ChatBotGraph:
    """
    聊天机器人核心类，集成了问题分类、向量检索和语言模型功能
    
    实现了基于检索增强生成(RAG)的智能问答流程，能够根据用户问题
    从法律知识库中检索相关信息，并生成准确的回答
    """
    
    def __init__(self):
        """
        初始化聊天机器人组件
        加载问题分类器、语言模型、嵌入模型、向量数据库和重排序模型
        """
        print("正在初始化聊天机器人组件...")
        # 初始化问题分类器
        self.classifier = QuestionClassifier()
        
        # 初始化大语言模型（注意：这里需要根据实际环境调整模型路径）
        try:
            self.llm = QwenModelChat('/root/sunyd/model_hub/qwen/Qwen2-7B-Instruct')
        except Exception as e:
            print(f"警告：无法加载指定路径的语言模型，使用默认配置。错误信息: {e}")
            # 可以在这里添加备用模型路径或默认配置
            self.llm = QwenModelChat()
        
        # 初始化文本向量化模型
        try:
            self.vectorizer = BGEVectorizer('/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/')
        except Exception as e:
            print(f"警告：无法加载指定路径的向量化模型，使用默认配置。错误信息: {e}")
            self.vectorizer = BGEVectorizer()
        
        # 初始化向量数据库
        try:
            self.vector = VectorStore(uri='/root/sunyd/llms/TinyRAG-master/storage/milvus_law.db')
        except Exception as e:
            print(f"警告：无法连接到指定的向量数据库，使用默认配置。错误信息: {e}")
            # 创建示例数据库用于演示
            self.vector = VectorStore(uri='./demo_law.db')
        
        # 初始化重排序模型
        try:
            self.relevance_reranker = BgeRelevanceReranker(path='/root/sunyd/model_hub/Xorbits/bge-reranker-base')
        except Exception as e:
            print(f"警告：无法加载指定路径的重排序模型，使用默认配置。错误信息: {e}")
            self.relevance_reranker = BgeRelevanceReranker()
        
        print("聊天机器人组件初始化完成")

    def chat_main(self, sent):
        """
        处理用户输入并生成回答
        
        Args:
            sent: 用户输入的问题字符串
            
        Returns:
            生成的回答字符串
        """
        # 默认回答
        default_answer = '您好，我是小荻，很高兴为您提供法律咨询服务！'
        
        try:
            # 对用户问题进行分类
            print(f"正在处理用户问题: {sent}")
            res_classify = self.classifier.classify(sent)
            print(f"问题分类结果: {res_classify}")
            
            # 如果分类器没有找到对应的知识库，直接使用语言模型生成回答
            if len(res_classify['kg_names']) == 0:
                print("未找到相关知识库，直接使用语言模型生成回答")
                prompt = self.llm.generate_prompt(sent, default_answer)
                return self.llm.chat(prompt)
            
            # 从向量数据库中查询出最相似的3个文档
            print("从向量数据库中检索相关文档...")
            contents = []  # 存储检索到的内容
            sim_query = []  # 存储相似问题
            
            # 遍历所有匹配的知识库集合
            for collection_name in res_classify['kg_names']:
                print(f"  从集合 '{collection_name}' 中检索...")
                # 查询每个集合中的相关文档
                for content in self.vector.query(sent, collection_name=collection_name, 
                                                vectorizer=self.vectorizer, k=3):
                    sim_query.append(content.key)
                    contents.append(content.value)
            
            # 如果没有检索到内容，直接使用语言模型
            if len(contents) == 0:
                print("未检索到相关内容，直接使用语言模型")
                return self.llm.chat(sent)
            
            # 合并检索到的内容
            best_content = ''.join(contents)
            print(f"已检索到 {len(contents)} 个相关文档")
            
            # 生成带有检索内容的提示
            prompt = self.llm.generate_prompt(sent, best_content)
            
            # 使用语言模型生成最终回答
            final_answers = self.llm.chat(prompt)
            
            # 打印相关问题（用于调试）
            print(f'相关问题：{sim_query}')
            
            # 返回生成的回答
            return final_answers if final_answers else default_answer
            
        except Exception as e:
            # 错误处理
            print(f"处理过程中出现错误: {e}")
            return f"抱歉，处理您的问题时出现错误: {str(e)}"

# 示例演示
if __name__ == '__main__':
    print("法律智能聊天机器人启动中...")
    handler = ChatBotGraph()
    print("\n=== 法律智能聊天机器人已启动 ===")
    print("请输入您的法律问题，输入'退出'结束对话")
    
    # 示例问题列表（用于演示）
    example_questions = [
        "什么是民法典？",
        "合同违约应该如何处理？",
        "劳动法规定的工作时间是多少？"
    ]
    print("\n示例问题:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    # 交互循环
    while True:
        try:
            question = input('\n用户: ').strip()
            if question.lower() in ['退出', 'exit', 'quit', 'q']:
                print("感谢使用法律智能聊天机器人，再见！")
                break
            
            if not question:
                print("请输入有效的问题")
                continue
            
            # 处理用户问题并获取回答
            answer = handler.chat_main(question)
            print('小荻:', answer)
            
        except KeyboardInterrupt:
            print("\n感谢使用法律智能聊天机器人，再见！")
            break
        except Exception as e:
            print(f"程序异常: {e}")
            continue