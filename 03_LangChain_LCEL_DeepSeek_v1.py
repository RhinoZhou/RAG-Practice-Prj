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

# 2. 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型（保持使用本地模型，无需API）
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_splits)

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 创建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:""")

# 7. 设置语言模型（和输出解析器
from langchain_deepseek import ChatDeepSeek  # 导入DeepSeek适配器
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 获取并验证API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    # 备用API密钥设置
    api_key = "sk-XXXXXXXX"  # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
    print("警告：从环境变量加载API密钥失败，使用备用密钥")

# 初始化DeepSeek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=api_key  # 使用验证过的API密钥
)

# 8. 构建 LCEL 链（保持链结构不变，仅替换了LLM）
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 执行查询（问题改为与法律文档相关）
question = "哪些部门负责促进民营经济发展的工作？"
response = chain.invoke(question)
print(response)
