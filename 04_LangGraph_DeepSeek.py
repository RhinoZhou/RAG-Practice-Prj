# 1. 加载文档（改为本地文件）
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# 加载环境变量
load_dotenv()

# 从本地加载法律文档
loader = TextLoader(
    file_path="../20-Data/法律文档/中华人民共和国民营经济促进法.txt",
    encoding='utf-8'  # 指定编码，避免中文乱码
)
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 定义RAG提示词
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# 6. 定义应用状态
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# 7. 定义检索步骤
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# 8. 定义生成步骤
def generate(state: State):
    from langchain_deepseek import ChatDeepSeek

    # 获取并验证API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # 使用备用API密钥
        api_key = "sk-XXXXXXXX"  # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
        print("警告：从环境变量加载API密钥失败，使用备用密钥")

    # 初始化DeepSeek模型
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2048,
        api_key=api_key
    )

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# 9. 构建和编译应用
from langgraph.graph import START, StateGraph  # pip install langgraph

graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)

# 10. 运行查询（改为与法律文档相关的问题）
question = "哪些部门负责促进民营经济发展的工作？"
response = graph.invoke({"question": question})
print(f"\n问题: {question}")
print(f"答案: {response['answer']}")
