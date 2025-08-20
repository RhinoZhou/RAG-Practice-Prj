# 1. 加载基础库与初始化计时
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_deepseek import ChatDeepSeek

# 记录总耗时
total_start = time.time()

# 加载环境变量
load_dotenv()

# 配置路径与初始化计时
current_dir = Path(__file__).resolve().parent
LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"
DOC_PATH = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"

# 2. 验证本地资源（带计时）
start_time = time.time()
if not LOCAL_EMBEDDING_PATH.exists():
    raise FileNotFoundError(
        f"本地嵌入模型不存在：{LOCAL_EMBEDDING_PATH}\n"
        f"下载命令：modelscope download --model AI-ModelScope/bge-small-zh-v1.5 --local_dir {LOCAL_EMBEDDING_PATH}"
    )
if not DOC_PATH.exists():
    raise FileNotFoundError(f"文档不存在：{DOC_PATH}")
validate_time = time.time() - start_time
print(f"【1/8】资源验证耗时：{validate_time:.4f}秒")

# 3. 加载本地文档（带计时）
start_time = time.time()
loader = TextLoader(
    file_path=str(DOC_PATH),
    encoding='utf-8'
)
docs = loader.load()
load_time = time.time() - start_time
print(f"【2/8】文档加载耗时：{load_time:.4f}秒（文档数量：{len(docs)}）")

# 4. 文档分块（带计时）
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(docs)
split_time = time.time() - start_time
print(f"【3/8】文档分块耗时：{split_time:.4f}秒（分块数量：{len(all_splits)}）")

# 5. 初始化本地嵌入模型（带计时）
start_time = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=str(LOCAL_EMBEDDING_PATH),
    model_kwargs={
        'device': 'cpu',
        'local_files_only': True
    },
    encode_kwargs={'normalize_embeddings': True}
)
embed_init_time = time.time() - start_time
print(f"【4/8】嵌入模型初始化耗时：{embed_init_time:.4f}秒（模型路径：{LOCAL_EMBEDDING_PATH.name}）")

# 6. 创建向量存储（带计时）
start_time = time.time()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
vector_time = time.time() - start_time
print(f"【5/8】向量存储创建耗时：{vector_time:.4f}秒")

# 7. 加载提示词模板（带计时）
start_time = time.time()
prompt = hub.pull("rlm/rag-prompt")
prompt_time = time.time() - start_time
print(f"【6/8】提示词模板加载耗时：{prompt_time:.4f}秒")

# 8. 定义应用状态与工作流
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 检索步骤（带内部计时）
def retrieve(state: State):
    start = time.time()
    retrieved_docs = vector_store.similarity_search(state["question"])
    end = time.time()
    print(f"【7/8】文档检索耗时：{end - start:.4f}秒（返回文档数：{len(retrieved_docs)}）")
    return {"context": retrieved_docs}

# 生成步骤（带内部计时）
def generate(state: State):
    start = time.time()
    # 初始化DeepSeek
    api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXX"
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2048,
        api_key=api_key
    )
    # 生成回答
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    end = time.time()
    print(f"【8/8】LLM生成耗时：{end - start:.4f}秒")
    return {"answer": response.content}

# 9. 构建图工作流（带计时）
start_time = time.time()
graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)
graph_time = time.time() - start_time
print(f"【9/9】工作流构建耗时：{graph_time:.4f}秒")

# 10. 运行查询并输出结果
question = "哪些部门负责促进民营经济发展的工作？"
print(f"\n===== 开始查询：{question} =====")
response = graph.invoke({"question": question})

# 输出总耗时
total_time = time.time() - total_start
print(f"\n===== 总流程耗时：{total_time:.4f}秒 =====")
print(f"问题：{question}")
print(f"答案：{response['answer']}")