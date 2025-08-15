# 1. 加载环境变量
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
from langchain_community.document_loaders import TextLoader  # 用于加载本地文本文件
# 从本地加载法律文档
loader = TextLoader(
    file_path="../20-Data/法律文档/中华人民共和国民营经济促进法.txt",
    encoding='utf-8'  # 指定编码，避免中文乱码问题
)
docs = loader.load()
# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
# 5. 构建用户查询
question = "哪些部门负责促进民营经济发展的工作？"
# 6. 搜索相关文档
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )
# 8. 使用大语言模型生成答案
from langchain_deepseek import ChatDeepSeek  # pip install langchain-deepseek

# 显式获取并验证API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    # 直接设置API密钥（仅用于测试）
    api_key = "sk-XXXXXXXX"  # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
    print("警告：从环境变量加载API密钥失败，使用备用密钥")
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=api_key  # 这里使用前面验证过的api_key变量
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
