import os                                                                                          # 操作系统接口模块，用于文件和目录操作
import time                                                                                       # 时间相关函数，用于性能计时
import torch                                                                                      # PyTorch深度学习框架，用于GPU检测
import socket                                                                                     # 网络通信模块，用于端口检查
from pathlib import Path                                                                          # 面向对象的路径操作模块
from dotenv import load_dotenv                                                                    # 环境变量加载模块，用于读取.env文件
from fastapi import FastAPI, HTTPException                                                        # FastAPI框架和HTTP异常处理
from fastapi.middleware.cors import CORSMiddleware                                                # CORS中间件，处理跨域请求
from pydantic import BaseModel                                                                    # Pydantic模型基类，用于数据验证
from typing import Dict, Any                                                                      # 类型提示，用于函数签名
import uvicorn                                                                                    # ASGI服务器，用于运行FastAPI应用
from contextlib import asynccontextmanager                                                        # 异步上下文管理器，用于应用生命周期管理

# LangChain相关导入
from langchain_huggingface import HuggingFaceEmbeddings                                           # HuggingFace嵌入模型封装
from langchain_community.vectorstores import FAISS                                                # FAISS向量数据库实现
from langchain_core.prompts import ChatPromptTemplate                                               # 聊天提示模板
from langchain_deepseek import ChatDeepSeek                                                       # DeepSeek聊天模型
from sentence_transformers import SentenceTransformer as ST                                       # 句子变换器，用于获取设备信息

# 全局变量
vector_store = None                                                                              # 全局向量存储实例
llm = None                                                                                       # 全局语言模型实例
prompt = None                                                                                    # 全局提示模板实例
init_info = {}                                                                                   # 全局初始化信息字典


def is_port_in_use(port: int) -> bool:                                                             # 检查指定端口是否被占用的函数
    """检查端口是否被占用"""                                                                       # 函数说明：检查TCP端口是否已被其他进程占用
    try:                                                                                            # 异常处理开始
        # 创建socket对象
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:                               # 创建IPv4 TCP socket对象
            # 尝试绑定端口，若成功则端口未被占用
            s.bind(("0.0.0.0", port))                                                             # 尝试绑定到所有网络接口的指定端口
        return False                                                                                # 绑定成功，端口未被占用
    except OSError:                                                                                 # 捕获操作系统错误（端口被占用）
        return True                                                                                 # 端口已被占用


@asynccontextmanager                                                                               # 异步上下文管理器装饰器
async def lifespan(app: FastAPI):                                                                   # FastAPI应用生命周期管理函数
    global vector_store, llm, prompt, init_info                                                     # 声明使用全局变量

    start_total = time.time()                                                                       # 记录初始化总开始时间

    load_dotenv()                                                                                   # 加载环境变量配置文件
    current_dir = Path(__file__).resolve().parent                                                   # 获取当前脚本所在目录的绝对路径

    # 配置路径
    LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"                    # 本地嵌入模型路径配置
    FAISS_DB_PATH = current_dir / "../23-faiss_db"                                                  # FAISS向量数据库路径配置

    # 验证本地嵌入模型是否存在
    start_time = time.time()                                                                        # 记录模型验证开始时间
    if not LOCAL_EMBEDDING_PATH.exists() or len(os.listdir(LOCAL_EMBEDDING_PATH)) == 0:            # 检查模型目录是否存在且非空
        raise FileNotFoundError(f"本地嵌入模型不存在于路径: {LOCAL_EMBEDDING_PATH}")                # 模型不存在时抛出异常

    print("本地嵌入模型已存在，无需下载")                                                             # 提示模型已存在
    model_prep_time = time.time() - start_time                                                      # 计算模型验证耗时
    init_info["model_prep_time"] = model_prep_time                                                 # 记录模型验证时间到初始化信息

    # 初始化本地嵌入模型
    start_time = time.time()                                                                        # 记录嵌入模型初始化开始时间
    embeddings = HuggingFaceEmbeddings(                                                             # 创建HuggingFace嵌入模型实例
        model=str(LOCAL_EMBEDDING_PATH),                                                             # 指定本地模型路径
        model_kwargs={
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',                                # 自动选择GPU或CPU设备
            'local_files_only': True                                                                  # 强制使用本地文件，不联网下载
        },
        encode_kwargs={
            'batch_size': 32,                                                                         # 批处理大小为32
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',                                # 再次指定设备确保一致性
            'normalize_embeddings': True                                                              # 启用嵌入向量归一化
        }
    )

    # 获取设备信息
    temp_model = ST(str(LOCAL_EMBEDDING_PATH))                                                       # 临时加载模型获取设备信息
    device = temp_model.device                                                                       # 获取模型所在设备（GPU/CPU）
    del temp_model  # 释放资源                                                                       # 删除临时模型释放内存

    embed_time = time.time() - start_time                                                           # 计算嵌入模型初始化耗时
    print(f"嵌入模型初始化耗时：{embed_time:.4f}秒，使用设备：{device}")                              # 打印初始化耗时和设备信息
    init_info["embed_time"] = embed_time                                                            # 记录初始化时间到全局信息
    init_info["device"] = str(device)                                                               # 记录设备信息到全局信息

    # 加载向量库
    start_time = time.time()                                                                        # 记录向量库加载开始时间
    if not FAISS_DB_PATH.exists() or len(os.listdir(FAISS_DB_PATH)) == 0:                           # 检查向量库目录是否存在且非空
        raise FileNotFoundError(f"未找到FAISS向量库，路径：{FAISS_DB_PATH}")                        # 向量库不存在时抛出异常

    vector_store = FAISS.load_local(                                                                # 从本地加载FAISS向量库
        str(FAISS_DB_PATH),                                                                         # 向量库目录路径
        embeddings,                                                                                 # 使用的嵌入模型
        allow_dangerous_deserialization=True                                                        # 允许反序列化（注意安全风险）
    )
    print(f"加载已存在的FAISS向量库，路径：{FAISS_DB_PATH}")                                         # 提示向量库加载成功
    vector_time = time.time() - start_time                                                          # 计算向量库加载耗时
    init_info["vector_time"] = vector_time                                                          # 记录加载时间到全局信息

    # 初始化LLM
    start_time = time.time()                                                                        # 记录LLM初始化开始时间
    api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXXX"                                            # 获取API密钥（环境变量或默认值）

    llm = ChatDeepSeek(                                                                             # 创建DeepSeek聊天模型实例
        model="deepseek-chat",                                                                      # 指定使用deepseek-chat模型
        temperature=0.0,                                                                            # 设置温度为0，确保确定性输出
        max_tokens=2048,                                                                            # 最大输出令牌数限制为2048
        api_key=api_key                                                                             # 传入API密钥
    )
    llm_time = time.time() - start_time                                                             # 计算LLM初始化耗时
    print(f"创建LLM耗时：{llm_time:.4f}秒")                                                          # 打印LLM初始化耗时
    init_info["llm_time"] = llm_time                                                               # 记录初始化时间到全局信息

    # 构建提示模板
    prompt = ChatPromptTemplate.from_template("""                                                    
    基于以下上下文，用中文简洁准确地回答问题。如果上下文中没有相关信息，                     
    请说"我无法从提供的上下文中找到相关信息"。                                                
    上下文: {context}                                                                                 
    问题: {question}                                                                                
    回答:"""                                                                                         # 回答开始标记
                                              )

    # 计算总初始化耗时
    total_init_time = time.time() - start_total                                                     # 计算整个初始化流程总耗时
    print(f"\n===== 服务初始化完成（总耗时：{total_init_time:.4f}秒） =====")                         # 打印初始化完成信息
    init_info["total_init_time"] = total_init_time                                                  # 记录总初始化时间到全局信息

    yield  # 应用运行期间保持资源                                                                    # 应用启动完成，开始处理请求


# 初始化FastAPI应用
app = FastAPI(                                                                                      # 创建FastAPI应用实例
    title="法律文档问答服务",                                                                        # 设置API文档标题
    description="基于本地文档的法律问答API",                                                           # 设置API文档描述
    lifespan=lifespan                                                                              # 指定生命周期管理函数
)

# 配置CORS
app.add_middleware(                                                                               # 添加CORS中间件
    CORSMiddleware,                                                                                # CORS中间件类
    allow_origins=["*"],                                                                           # 允许所有来源的跨域请求
    allow_credentials=True,                                                                        # 允许携带凭证
    allow_methods=["*"],                                                                           # 允许所有HTTP方法
    allow_headers=["*"],                                                                           # 允许所有请求头
)


# 定义请求和响应模型
class QueryRequest(BaseModel):                                                                      # 查询请求数据模型
    question: str                                                                                  # 用户问题字段


class QueryResponse(BaseModel):                                                                     # 查询响应数据模型
    question: str                                                                                  # 原始问题
    answer: str                                                                                    # 生成的答案
    retrieval_time: float                                                                          # 文档检索耗时（秒）
    llm_time: float                                                                                # LLM生成耗时（秒）
    total_time: float                                                                              # 总处理耗时（秒）
    retrieved_docs_count: int                                                                      # 检索到的相关文档数量


# 健康检查接口
@app.get("/health", response_model=Dict[str, str])                                                # 健康检查GET接口
async def health_check():                                                                           # 健康检查处理函数
    return {"status": "healthy", "message": "法律文档问答服务运行正常"}                            # 返回健康状态和服务信息


# 获取初始化信息接口
@app.get("/init-info", response_model=Dict[str, Any])                                              # 初始化信息GET接口
async def get_init_info():                                                                          # 初始化信息处理函数
    return init_info                                                                                # 返回全局初始化信息


# 问答接口
@app.post("/query", response_model=QueryResponse)                                                 # 问答POST接口
async def query(request: QueryRequest):                                                             # 问答处理函数
    question = request.question.strip()                                                             # 获取并清理用户问题
    print(f"\n收到客户端查询: {question}")                                                          # 打印收到的查询

    if not question:                                                                                # 检查问题是否为空
        raise HTTPException(status_code=400, detail="问题不能为空")                               # 空问题返回400错误

    query_start = time.time()                                                                       # 记录查询处理开始时间

    try:                                                                                            # 异常处理开始
        # 检索相关文档
        start_time = time.time()                                                                    # 记录检索开始时间
        retrieved_docs = vector_store.similarity_search(question, k=3)                             # 执行相似度搜索，返回前3个相关文档
        retrieve_time = time.time() - start_time                                                    # 计算检索耗时
        print(f"文档检索耗时：{retrieve_time:.4f}秒，返回{len(retrieved_docs)}条相关结果")          # 打印检索结果

        # 准备上下文内容
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)                     # 将检索到的文档内容合并为上下文

        # 调用LLM生成答案
        start_time = time.time()                                                                    # 记录LLM调用开始时间
        answer = llm.invoke(prompt.format(question=question, context=docs_content))                 # 格式化提示并调用LLM
        llm_time = time.time() - start_time                                                         # 计算LLM调用耗时
        print(f"答案生成耗时：{llm_time:.4f}秒")                                                     # 打印LLM调用耗时

        # 计算总耗时
        query_total_time = time.time() - query_start                                                # 计算整个查询处理耗时

        return QueryResponse(                                                                        # 返回结构化的响应数据
            question=question,                                                                       # 原始问题
            answer=answer.content,                                                                   # 生成的答案内容
            retrieval_time=retrieve_time,                                                          # 检索耗时
            llm_time=llm_time,                                                                       # LLM调用耗时
            total_time=query_total_time,                                                             # 总处理耗时
            retrieved_docs_count=len(retrieved_docs)                                                 # 检索到的文档数量
        )
    except Exception as e:                                                                          # 捕获所有异常
        print(f"处理查询时发生错误：{str(e)}")                                                       # 打印错误信息
        raise HTTPException(status_code=500, detail=f"处理查询时发生错误：{str(e)}")                  # 返回500服务器错误


if __name__ == "__main__":                                                                          # 主程序入口
    # 配置服务端口
    PORT = 8848                                                                                     # 设置服务监听端口为8848

    # 检查端口是否被占用
    if is_port_in_use(PORT):                                                                        # 检查8848端口是否已被占用
        # 端口被占用时抛出明确错误
        raise OSError(f"端口 {PORT} 已被占用，请释放该端口或更换其他端口后重试。")                      # 端口占用异常提示

    # 端口可用时启动服务
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")                                  # 启动Uvicorn服务器，监听所有网络接口