import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 记录总耗时
start_total = time.time()

# 加载环境变量
load_dotenv()

# 获取当前脚本路径，统一处理绝对路径
current_dir = Path(__file__).resolve().parent

# 配置本地模型路径（请确保已下载到这些路径）
LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"
# 文档路径
DOCUMENT_PATH = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"

# 验证本地模型是否存在
if not LOCAL_EMBEDDING_PATH.exists():
    raise FileNotFoundError(
        f"本地嵌入模型不存在于路径: {LOCAL_EMBEDDING_PATH}\n"
        f"请先下载模型到该路径，可使用命令:\n"
        f"modelscope download --model AI-ModelScope/bge-small-zh-v1.5 --local_dir {LOCAL_EMBEDDING_PATH}"
    )

# 验证文档是否存在
if not DOCUMENT_PATH.exists():
    raise FileNotFoundError(f"文档文件不存在: {DOCUMENT_PATH}")

# 1. 加载文档
start_time = time.time()
loader = TextLoader(
    file_path=str(DOCUMENT_PATH),
    encoding='utf-8'  # 确保中文正常加载
)
docs = loader.load()
load_time = time.time() - start_time
print(f"1、文档加载耗时：{load_time:.4f}秒")

# 2. 文档分块（优化中文分割）
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]  # 中文优先分割符
)
all_splits = text_splitter.split_documents(docs)
split_time = time.time() - start_time
print(f"2、文档分块耗时：{split_time:.4f}秒，共分{len(all_splits)}块")

# 3. 初始化本地嵌入模型（完全不联网）
start_time = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=str(LOCAL_EMBEDDING_PATH),  # 使用本地模型路径
    model_kwargs={
        'device': 'cpu',  # 强制使用CPU，无需GPU
        'local_files_only': True  # 仅使用本地文件，不检查远程
    },
    encode_kwargs={'normalize_embeddings': True}
)
embed_time = time.time() - start_time
print(f"3、本地嵌入模型初始化耗时：{embed_time:.4f}秒")

# 4. 创建向量存储
start_time = time.time()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
vector_time = time.time() - start_time
print(f"4、向量存储创建耗时：{vector_time:.4f}秒")

# 5. 检索相关文档
question = "哪些部门负责促进民营经济发展的工作？"
start_time = time.time()
retrieved_docs = vector_store.similarity_search(question, k=3)
retrieve_time = time.time() - start_time
print(f"5、文档检索耗时：{retrieve_time:.4f}秒，返回{len(retrieved_docs)}条结果")

# 准备上下文内容
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. 构建提示模板
prompt = ChatPromptTemplate.from_template("""
基于以下上下文，用中文简洁准确地回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:"""
                                          )

# 7. 调用DeepSeek生成答案
start_time = time.time()
# 获取API密钥（DeepSeek API仍需联网，如要完全离线需替换为本地部署的LLM）
api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXX"

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.0,  # 事实性问答适合低随机性
    max_tokens=2048,
    api_key=api_key
)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
llm_time = time.time() - start_time
print(f"6、LLM生成耗时：{llm_time:.4f}秒")

# 输出结果
print("\n===== 问答结果 =====")
print(f"问题：{question}")
print(f"回答：{answer.content}")

# 总耗时统计
total_time = time.time() - start_total
print(f"\n总流程耗时：{total_time:.4f}秒")
