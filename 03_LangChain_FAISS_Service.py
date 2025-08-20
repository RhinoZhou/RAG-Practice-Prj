import os
import time
import torch
from pathlib import Path  # 用于处理文件路径
from dotenv import load_dotenv  # 用于加载环境变量
from fastapi import FastAPI, HTTPException, Request, Response  # FastAPI核心组件
from fastapi.middleware.cors import CORSMiddleware  # 处理跨域请求
from pydantic import BaseModel  # 用于数据验证和模型定义
from typing import List, Dict, Any, Optional  # 类型注解
import uvicorn  # ASGI服务器，用于运行FastAPI应用
from contextlib import asynccontextmanager  # 用于定义异步上下文管理器

# LangChain相关导入 - 用于构建RAG(检索增强生成)系统
from langchain_community.document_loaders import TextLoader  # 用于加载文本文件
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 用于文档分块
from langchain_huggingface import HuggingFaceEmbeddings  # 用于加载HuggingFace嵌入模型
from langchain_community.vectorstores import FAISS  # 用于向量存储和检索
from langchain_core.prompts import ChatPromptTemplate  # 用于构建提示模板
from langchain_deepseek import ChatDeepSeek  # DeepSeek大语言模型集成
from sentence_transformers import SentenceTransformer as ST  # 用于获取嵌入模型设备信息

# 全局变量，用于存储初始化好的组件
# 这些组件在应用启动时初始化，整个应用生命周期内复用
vector_store = None  # FAISS向量存储实例
llm = None  # 大语言模型实例
prompt = None  # 提示模板实例
init_info = {}  # 存储初始化信息的字典


# 使用lifespan替代传统的on_event("startup")和on_event("shutdown")
# 这是FastAPI推荐的新方式，用于管理应用的生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 声明使用全局变量
    global vector_store, llm, prompt, init_info

    # 记录总初始化开始时间
    start_total = time.time()

    # 加载环境变量（从.env文件）
    # 通常用于存储敏感信息如API密钥，避免硬编码
    load_dotenv()

    # 获取当前脚本所在目录的绝对路径
    # 这样可以确保无论从哪个目录运行脚本，路径都能正确解析
    current_dir = Path(__file__).resolve().parent

    # 配置各种路径
    # 本地嵌入模型路径 - BGE-small-zh-v1.5是一个中文嵌入模型
    LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"
    # 法律文档路径 - 这里使用的是《中华人民共和国民营经济促进法》
    DOCUMENT_PATH = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"
    # FAISS向量库保存路径 - 用于存储文档的向量表示
    FAISS_DB_PATH = current_dir / "../23-faiss_db"

    # 验证本地嵌入模型是否存在
    if not LOCAL_EMBEDDING_PATH.exists():
        # 如果模型不存在，抛出异常并给出下载提示
        raise FileNotFoundError(
            f"本地嵌入模型不存在于路径: {LOCAL_EMBEDDING_PATH}\n"
            f"请先下载模型到该路径，可使用命令:\n"
            f"modelscope download --model AI-ModelScope/bge-small-zh-v1.5 --local_dir {LOCAL_EMBEDDING_PATH}"
        )

    # 验证文档是否存在
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"文档文件不存在: {DOCUMENT_PATH}")

    # 1. 加载文档
    start_time = time.time()  # 记录开始时间
    loader = TextLoader(
        file_path=str(DOCUMENT_PATH),  # 文档路径转换为字符串
        encoding='utf-8'  # 指定编码为utf-8，确保中文正常加载
    )
    docs = loader.load()  # 加载文档内容
    load_time = time.time() - start_time  # 计算耗时
    print(f"1、文档加载耗时：{load_time:.4f}秒")
    init_info["load_time"] = load_time  # 保存到初始化信息

    # 2. 文档分块
    # 长文档需要分割成小块，以便更好地进行检索和处理
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的大小（字符数）
        chunk_overlap=200,  # 块之间的重叠部分（字符数），有助于保持上下文连续性
        # 分割符优先级列表，按照中文文本特点设置
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    all_splits = text_splitter.split_documents(docs)  # 执行分块
    split_time = time.time() - start_time
    print(f"2、文档分块耗时：{split_time:.4f}秒，共分{len(all_splits)}块")
    init_info["split_time"] = split_time
    init_info["split_count"] = len(all_splits)  # 记录分块数量

    # 3. 初始化本地嵌入模型
    # 嵌入模型用于将文本转换为向量表示
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model=str(LOCAL_EMBEDDING_PATH),  # 本地模型路径
        model_kwargs={
            # 根据是否有GPU自动选择设备
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'local_files_only': True  # 只使用本地文件，不尝试下载
        },
        encode_kwargs={
            'batch_size': 32,  # 批处理大小，影响编码速度和内存使用
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'normalize_embeddings': True  # 归一化嵌入向量，便于计算相似度
        }
    )

    # 获取设备信息（用于显示）
    # 临时加载模型以获取设备信息
    temp_model = ST(str(LOCAL_EMBEDDING_PATH))
    device = temp_model.device
    del temp_model  # 及时删除临时模型，释放资源

    embed_time = time.time() - start_time
    print(f"3、本地嵌入模型初始化耗时：{embed_time:.4f}秒，使用设备：{device}")
    init_info["embed_time"] = embed_time
    init_info["device"] = str(device)  # 记录使用的设备（CPU/GPU）

    # 4. 创建或加载FAISS向量存储
    # FAISS是用于高效相似性搜索的库，这里用于存储文档块的向量
    start_time = time.time()
    # 检查是否已存在向量库
    if FAISS_DB_PATH.exists() and len(os.listdir(FAISS_DB_PATH)) > 0:
        # 加载已存在的向量库
        vector_store = FAISS.load_local(
            str(FAISS_DB_PATH),
            embeddings,
            # 允许危险的反序列化（因为向量库是本地生成的，相对安全）
            allow_dangerous_deserialization=True
        )
        print(f"加载已存在的FAISS向量库，路径：{FAISS_DB_PATH}")
    else:
        # 不存在则创建新的向量库
        vector_store = FAISS.from_documents(all_splits, embeddings)
        # 确保保存目录存在
        FAISS_DB_PATH.mkdir(parents=True, exist_ok=True)
        # 保存向量库到本地
        vector_store.save_local(str(FAISS_DB_PATH))
        print(f"新建并保存FAISS向量库，路径：{FAISS_DB_PATH}")

    vector_time = time.time() - start_time
    print(f"4、FAISS向量存储处理耗时：{vector_time:.4f}秒")
    init_info["vector_time"] = vector_time

    # 5. 创建LLM（大语言模型）实例
    start_time = time.time()
    # 从环境变量获取API密钥，如果没有则使用默认值（仅作示例，实际应用中不应硬编码）
    api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXX"

    # 初始化DeepSeek大语言模型
    llm = ChatDeepSeek(
        model="deepseek-chat",  # 指定模型名称
        temperature=0.0,  # 温度参数，0表示结果更确定，1表示更多样化
        max_tokens=2048,  # 最大生成token数
        api_key=api_key  # API密钥
    )
    llm_time = time.time() - start_time
    print(f"5、创建LLM耗时：{llm_time:.4f}秒")
    init_info["llm_time"] = llm_time

    # 6. 构建提示模板
    # 提示模板定义了给大语言模型的指令格式
    prompt = ChatPromptTemplate.from_template("""
    基于以下上下文，用中文简洁准确地回答问题。如果上下文中没有相关信息，
    请说"我无法从提供的上下文中找到相关信息"。
    上下文: {context}
    问题: {question}
    回答:"""
                                              )

    # 计算总初始化耗时
    total_init_time = time.time() - start_total
    print(f"\n===== 系统初始化完成（总耗时：{total_init_time:.4f}秒） =====")
    init_info["total_init_time"] = total_init_time

    yield  # 程序运行期间保持，控制权交还给FastAPI
    # 如果需要在应用关闭时执行清理操作，可以放在yield之后


# 初始化FastAPI应用，指定生命周期管理器
app = FastAPI(
    title="法律文档问答服务",  # API标题
    description="基于本地文档的法律问答API",  # API描述
    lifespan=lifespan  # 指定生命周期管理器
)

# 配置CORS（跨域资源共享）
# 允许前端应用从不同的域名访问后端API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境应指定具体域名
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)


# 定义请求和响应模型
# 使用Pydantic模型可以自动进行数据验证和文档生成

class QueryRequest(BaseModel):
    """查询请求模型，包含用户的问题"""
    question: str  # 用户的问题文本


class QueryResponse(BaseModel):
    """查询响应模型，包含回答及相关信息"""
    question: str  # 回应用户的问题
    answer: str  # 生成的回答
    retrieval_time: float  # 文档检索耗时（秒）
    llm_time: float  # LLM生成耗时（秒）
    total_time: float  # 总耗时（秒）
    retrieved_docs_count: int  # 检索到的相关文档数量


# 健康检查接口
# 用于监控服务是否正常运行
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """健康检查接口，返回服务状态"""
    return {"status": "healthy", "message": "法律文档问答服务运行正常"}


# 获取初始化信息接口
# 用于获取系统初始化过程中的各项指标
@app.get("/init-info", response_model=Dict[str, Any])
async def get_init_info():
    """获取系统初始化信息，包括各步骤耗时等"""
    return init_info


# 问答接口（核心功能）
# 接收用户问题，返回基于文档的回答
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """处理用户查询，返回基于法律文档的回答"""
    # 去除问题前后的空白字符
    question = request.question.strip()

    # 打印从客户端传递来的问题文本
    print(f"\n收到客户端查询: {question}")

    # 检查问题是否为空
    if not question:
        # 如果问题为空，返回400错误
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 记录单次查询开始时间
    query_start = time.time()

    try:
        # 1. 检索相关文档
        # 从向量库中查找与问题最相似的文档块
        start_time = time.time()
        # k=3表示返回最相关的3个文档块
        retrieved_docs = vector_store.similarity_search(question, k=3)
        retrieve_time = time.time() - start_time
        print(f"文档检索耗时：{retrieve_time:.4f}秒，返回{len(retrieved_docs)}条相关结果")

        # 2. 准备上下文内容
        # 将检索到的文档内容合并为一个字符串，作为LLM的上下文
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 3. 调用LLM生成答案
        # 将问题和上下文传入提示模板，调用大语言模型生成回答
        start_time = time.time()
        answer = llm.invoke(prompt.format(question=question, context=docs_content))
        llm_time = time.time() - start_time
        print(f"答案生成耗时：{llm_time:.4f}秒")

        # 计算总耗时
        query_total_time = time.time() - query_start

        # 返回结果
        return QueryResponse(
            question=question,
            answer=answer.content,  # 提取LLM生成的回答内容
            retrieval_time=retrieve_time,
            llm_time=llm_time,
            total_time=query_total_time,
            retrieved_docs_count=len(retrieved_docs)
        )
    except Exception as e:
        # 捕获并记录所有异常
        print(f"处理查询时发生错误：{str(e)}")
        # 向客户端返回500错误
        raise HTTPException(status_code=500, detail=f"处理查询时发生错误：{str(e)}")


# 主函数，当脚本直接运行时启动服务
if __name__ == "__main__":
    # 启动uvicorn服务器
    # host="0.0.0.0"表示允许所有网络接口访问
    # port=8848使用一个不常用的端口
    # log_level="info"设置日志级别
    uvicorn.run(app, host="0.0.0.0", port=8848, log_level="info")
