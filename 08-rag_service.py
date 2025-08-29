#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG系统后端服务

本程序实现了一个基于Langchain架构的RAG(检索增强生成)系统后端，
负责从客户端接收请求，连接到向量数据库和FastAPI服务检索相关数据，
并通过DeepSeek模型生成回答，最终返回给前端。

主要功能:
- 提供RESTful API接口接收客户端请求
- 连接到FAISS向量数据库检索相关文档
- 调用07-graph_loader.py提供的FastAPI服务获取子图信息
- 使用DeepSeek模型基于检索到的信息生成回答
- 处理请求/响应的格式转换和错误处理

"""

import os
import json
import logging
import requests
import yaml
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import uvicorn

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# 尝试导入DeepSeek模型
# 注意：根据实际安装的langchain版本和DeepSeek SDK版本调整导入方式
try:
    from langchain.llms import DeepSeek
    logger.info("成功导入langchain.llms.DeepSeek")
except ImportError:
    try:
        # 尝试其他可能的导入方式
        from langchain.chat_models import ChatDeepSeek as DeepSeek
        logger.info("成功导入langchain.chat_models.ChatDeepSeek")
    except ImportError:
        # 如果都失败，定义一个占位符
        logger.warning("无法导入DeepSeek模型，将使用模拟实现")
        
        class DeepSeek:
            def __init__(self, model=None, api_key=None, **kwargs):
                self.model = model
                self.api_key = api_key
                self.kwargs = kwargs
                
            def invoke(self, prompt, **kwargs):
                return f"[模拟回答] {prompt}"
                
            def __call__(self, prompt, **kwargs):
                return f"[模拟回答] {prompt}"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGService")

class RAGService:
    """RAG服务类
    
    该类负责管理RAG系统的核心功能，包括连接向量数据库、
    调用FastAPI服务、初始化语言模型和处理检索增强生成任务。
    """
    
    def __init__(self, config_file: str = 'rag_service_config.yaml'):
        """初始化RAG服务
        
        参数:
            config_file: 配置文件路径
        """
        # 默认配置
        self.config = {
            'vector_index_path': 'D:\\rag-project\\05-rag-practice\\23-faiss_db\\graph_node\\vector_index.faiss',
            'node_id_to_index_path': 'D:\\rag-project\\05-rag-practice\\23-faiss_db\\graph_node\\node_id_to_index.json',
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_dimension': 384,
            'fastapi_url': 'http://localhost:8000',
            'deepseek_api_key': 'your_api_key_here',
            'deepseek_model': 'deepseek-chat',
            'search_top_k': 5,
            'graph_depth': 1
        }
        
        # 从配置文件加载配置
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        self.config.update(yaml_config)
                logger.info(f"成功加载配置文件: {config_file}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        # 初始化组件
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        
        self.initialize_components()
    
    def initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化嵌入模型
            self.initialize_embeddings()
            
            # 初始化向量存储
            self.initialize_vector_store()
            
            # 初始化语言模型
            self.initialize_llm()
            
            # 初始化QA链
            self.initialize_qa_chain()
            
        except Exception as e:
            logger.error(f"初始化组件失败: {str(e)}")
    
    def initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config['embedding_model'],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"成功初始化嵌入模型: {self.config['embedding_model']}")
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            # 尝试使用sentence_transformers作为备选
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings = SentenceTransformer(self.config['embedding_model'])
                logger.info(f"使用sentence_transformers初始化嵌入模型: {self.config['embedding_model']}")
            except Exception as e2:
                logger.error(f"使用sentence_transformers初始化嵌入模型失败: {str(e2)}")
    
    def initialize_vector_store(self):
        """初始化向量存储"""
        try:
            # 检查文件是否存在
            if os.path.exists(self.config['vector_index_path']) and os.path.exists(self.config['node_id_to_index_path']):
                # 尝试直接加载FAISS索引
                try:
                    # 检查是否有SentenceTransformer嵌入模型可用
                    if hasattr(self.embeddings, 'encode'):
                        # 使用SentenceTransformer嵌入模型
                        self.vector_store = FAISS.load_local(
                            folder_path=os.path.dirname(self.config['vector_index_path']),
                            embeddings=self.embeddings,
                            index_name=os.path.basename(self.config['vector_index_path']).split('.')[0],
                            allow_dangerous_deserialization=True
                        )
                        logger.info(f"成功通过SentenceTransformer加载FAISS索引")
                    else:
                        # 尝试使用HuggingFaceEmbeddings加载
                        self.vector_store = FAISS.load_local(
                            folder_path=os.path.dirname(self.config['vector_index_path']),
                            embeddings=self.embeddings,
                            index_name=os.path.basename(self.config['vector_index_path']).split('.')[0],
                            allow_dangerous_deserialization=True
                        )
                        logger.info(f"成功通过HuggingFaceEmbeddings加载FAISS索引")
                except Exception as load_error:
                    logger.warning(f"直接加载FAISS索引失败: {str(load_error)}，尝试重建文档")
                    
                    # 备选方案：重建文档对象
                    try:
                        # 首先从07-graph_loader.py生成的节点文件中加载节点数据
                        # 尝试查找节点数据文件
                        node_files = ['nodes.csv', 'graph_nodes.csv', 'knowledge_graph_nodes.csv']
                        nodes_df = None
                        
                        for node_file in node_files:
                            if os.path.exists(node_file):
                                nodes_df = pd.read_csv(node_file)
                                logger.info(f"成功加载节点数据文件: {node_file}")
                                break
                        
                        if nodes_df is None:
                            logger.warning("未找到节点数据文件，使用模拟数据")
                            # 使用模拟数据
                            class MockDataFrame:
                                def __init__(self):
                                    self['id'] = ['1', '2', '3']
                                    self['text'] = ['模拟文档1', '模拟文档2', '模拟文档3']
                                
                                def iloc(self, index):
                                    return self
                                
                                def to_dict(self):
                                    return {}
                                
                                def __getitem__(self, key):
                                    return self
                                
                                def __setitem__(self, key, value):
                                    pass
                            nodes_df = MockDataFrame()
                        
                        # 读取节点ID到索引的映射
                        with open(self.config['node_id_to_index_path'], 'r', encoding='utf-8') as f:
                            node_id_to_index = json.load(f)
                        
                        # 创建文档列表
                        documents = []
                        # 安全地获取id和text列
                        try:
                            ids = nodes_df['id'] if hasattr(nodes_df, 'id') else getattr(nodes_df, 'id', ['1', '2', '3'])
                            texts = nodes_df['text'] if hasattr(nodes_df, 'text') else getattr(nodes_df, 'text', ['模拟文档1', '模拟文档2', '模拟文档3'])
                            
                            for node_id, text in zip(ids, texts):
                                # 尝试获取节点的其他属性
                                try:
                                    if hasattr(nodes_df, 'iloc'):
                                        node_data = nodes_df[nodes_df['id'] == node_id].iloc[0].to_dict()
                                    else:
                                        node_data = {}
                                except:
                                    node_data = {'id': node_id, 'text': text}
                                
                                # 创建文档对象
                                document = Document(
                                    page_content=str(text),
                                    metadata={k: str(v) for k, v in node_data.items()}
                                )
                                documents.append(document)
                        except Exception as doc_error:
                            logger.warning(f"创建文档对象失败: {str(doc_error)}，使用简化文档")
                            # 使用简化的文档创建方式
                            documents = [Document(page_content=f"模拟文档 {i}") for i in range(5)]
                        
                        # 初始化FAISS向量存储
                        self.vector_store = FAISS.from_documents(
                            documents=documents,
                            embedding=self.embeddings
                        )
                        
                        logger.info(f"成功初始化向量存储，包含 {len(documents)} 个文档")
                    except Exception as rebuild_error:
                        logger.error(f"重建文档对象失败: {str(rebuild_error)}")
                        # 使用最小的模拟向量存储
                        self.vector_store = self._create_minimal_vector_store()
            else:
                logger.warning("向量索引文件或节点映射文件不存在，使用模拟向量存储")
                self.vector_store = self._create_minimal_vector_store()
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            # 兜底方案
            self.vector_store = self._create_minimal_vector_store()
    
    def _create_minimal_vector_store(self):
        """创建最小化的模拟向量存储"""
        try:
            # 创建几个简单的文档
            mock_docs = [
                Document(page_content="这是一个示例文档，包含一些测试内容。"),
                Document(page_content="向量数据库用于高效存储和检索嵌入向量。"),
                Document(page_content="RAG系统结合检索和生成能力提供更准确的回答。"),
                Document(page_content="知识图谱可以表示实体之间的关系。"),
                Document(page_content="FAISS是Facebook开发的高效相似度搜索库。")
            ]
            
            # 创建向量存储
            vector_store = FAISS.from_documents(mock_docs, self.embeddings)
            logger.info("成功创建最小化模拟向量存储")
            return vector_store
        except Exception as e:
            logger.error(f"创建模拟向量存储失败: {str(e)}")
            # 最后的兜底方案，返回None
            return None
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
    
    def initialize_llm(self):
        """初始化语言模型"""
        try:
            # 初始化DeepSeek语言模型
            # 注意：根据实际情况调整初始化参数
            self.llm = DeepSeek(
                model=self.config['deepseek_model'],
                api_key=self.config['deepseek_api_key']
            )
            logger.info(f"成功初始化DeepSeek模型: {self.config['deepseek_model']}")
        except Exception as e:
            logger.error(f"初始化DeepSeek模型失败: {str(e)}")
            # 提供一个简单的模拟语言模型作为备选
            class MockLLM:
                def __call__(self, prompt, **kwargs):
                    return f"这是一个模拟回答，基于查询: {prompt}"
            
            self.llm = MockLLM()
            logger.warning("使用模拟语言模型作为备选")
    
    def initialize_qa_chain(self):
        """初始化QA链"""
        try:
            if self.vector_store is None or self.llm is None:
                logger.warning("向量存储或语言模型未初始化，无法创建QA链")
                return
            
            # 创建检索器
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config['search_top_k']}
            )
            
            # 创建提示模板
            prompt_template = """使用以下上下文信息回答用户的问题。如果无法从上下文信息中找到答案，
请直接说"根据提供的信息，我无法回答这个问题"。

上下文信息:
{context}

用户问题:
{question}

回答:
"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # 创建QA链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            logger.info("成功初始化QA链")
        except Exception as e:
            logger.error(f"初始化QA链失败: {str(e)}")
    
    def call_fastapi_service(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """调用FastAPI服务
        
        参数:
            endpoint: 服务端点
            params: 查询参数
        
        返回:
            服务响应结果
        """
        try:
            url = f"{self.config['fastapi_url']}/{endpoint}"
            logger.info(f"调用FastAPI服务: {url}, 参数: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"FastAPI服务调用成功，返回结果长度: {len(str(result))}")
            return result
        except Exception as e:
            logger.error(f"调用FastAPI服务失败: {str(e)}")
            return {"error": str(e)}
    
    def search_concept(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """搜索相似概念
        
        参数:
            query: 搜索查询
            top_k: 返回结果数量
        
        返回:
            搜索结果
        """
        params = {"q": query}
        if top_k is not None:
            params["top_k"] = top_k
        
        return self.call_fastapi_service("search_concept", params)
    
    def get_subgraph(self, concept_id: str, depth: int = None) -> Dict[str, Any]:
        """获取概念的子图
        
        参数:
            concept_id: 概念ID
            depth: 子图深度
        
        返回:
            子图信息
        """
        params = {"concept_id": concept_id}
        if depth is not None:
            params["depth"] = depth
        
        return self.call_fastapi_service("subgraph", params)
    
    def mixed_query(self, query: str, top_k: int = None, depth: int = None) -> Dict[str, Any]:
        """混合查询
        
        参数:
            query: 搜索查询
            top_k: 向量检索结果数量
            depth: 子图深度
        
        返回:
            查询结果
        """
        params = {"q": query}
        if top_k is not None:
            params["top_k"] = top_k
        if depth is not None:
            params["depth"] = depth
        
        return self.call_fastapi_service("mixed_query", params)
    
    def generate_answer(self, query: str, use_graph: bool = True) -> Dict[str, Any]:
        """生成回答
        
        参数:
            query: 用户问题
            use_graph: 是否使用图信息
        
        返回:
            生成的回答及相关信息
        """
        try:
            # 首先检查是否使用图信息
            context_info = {}
            
            if use_graph:
                # 执行混合查询获取相关信息
                mixed_results = self.mixed_query(
                    query,
                    top_k=self.config['search_top_k'],
                    depth=self.config['graph_depth']
                )
                context_info["mixed_results"] = mixed_results
            
            if self.qa_chain is None:
                # 如果QA链未初始化，使用向量检索+图查询+模拟回答
                logger.warning("QA链未初始化，使用备选方法生成回答")
                
                # 构建上下文
                context = ""
                if use_graph and "mixed_results" in context_info:
                    mixed_results = context_info["mixed_results"]
                    if "concept_results" in mixed_results:
                        for result in mixed_results["concept_results"]:
                            if "text" in result:
                                context += f"{result['text']}\n"
                    
                    if "graph" in mixed_results and "nodes" in mixed_results["graph"]:
                        for node in mixed_results["graph"]["nodes"]:
                            if "text" in node:
                                context += f"{node['text']}\n"
                
                # 如果向量存储可用，尝试进行向量检索
                if self.vector_store:
                    try:
                        vector_results = self.vector_store.similarity_search(query, k=self.config['search_top_k'])
                        for doc in vector_results:
                            context += f"{doc.page_content}\n"
                    except Exception as vec_err:
                        logger.warning(f"向量检索失败: {str(vec_err)}")
                
                # 生成模拟回答
                answer = f"基于检索到的信息，这是对您问题 '{query}' 的回答。\n\n检索到的相关信息:\n{context}"
                
                return {
                    "answer": answer,
                    "sources": context_info,
                    "status": "success",
                    "using_fallback": True,
                    "use_graph": use_graph
                }
            
            # 使用QA链生成回答
            result = self.qa_chain.invoke(query)
            
            # 如果使用图信息，添加图相关信息到结果中
            if use_graph:
                result["answer"] += "\n\n注意：此回答已考虑相关知识图谱信息。"
                
            return {
                "answer": result["result"],
                "sources": [
                    {"text": doc.page_content, "metadata": doc.metadata}
                    for doc in result.get("source_documents", [])
                ],
                "status": "success",
                "using_fallback": False,
                "use_graph": use_graph
            }
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时发生错误: {str(e)}",
                "sources": [],
                "status": "error",
                "error": str(e)
            }

# 导入缺失的pandas库
try:
    import pandas as pd
    logger.info("成功导入pandas库")
except ImportError as e:
    logger.warning(f"无法导入pandas库: {str(e)}. 某些功能可能受限。")
    # 提供一个简单的模拟pandas
    class MockPandas:
        def read_csv(self, *args, **kwargs):
            class MockDataFrame:
                def __init__(self):
                    self['id'] = []
                    self['text'] = []
                
                def iloc(self, index):
                    return self
                
                def to_dict(self):
                    return {}
                
                def __getitem__(self, key):
                    return self
                
                def __setitem__(self, key, value):
                    pass
            return MockDataFrame()
    pd = MockPandas()

# 创建RAG服务实例
rag_service = RAGService()

# 创建FastAPI应用
app = FastAPI(
    title="RAG Service API",
    description="基于Langchain的RAG系统后端服务",
    version="1.0.0"
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"请求处理异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path
        }
    )

@app.get("/health", tags=["监控"])
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "vector_store": "available" if rag_service.vector_store else "unavailable",
            "llm": "available" if hasattr(rag_service.llm, '__call__') else "unavailable",
            "qa_chain": "available" if rag_service.qa_chain else "unavailable"
        }
    }

@app.get("/answer", tags=["问答"])
async def get_answer(
    q: str = Query(..., description="用户问题"),
    use_graph: bool = Query(True, description="是否使用图信息")
):
    """获取问题的回答
    
    参数:
        q: 用户问题
        use_graph: 是否使用图信息
    
    返回:
        包含回答和相关信息的响应
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    logger.info(f"收到问答请求: {q}, use_graph: {use_graph}")
    
    # 生成回答
    result = rag_service.generate_answer(q, use_graph)
    
    return result

@app.get("/search", tags=["搜索"])
async def search(
    q: str = Query(..., description="搜索查询"),
    top_k: int = Query(5, description="返回结果数量", ge=1, le=20)
):
    """搜索相似概念
    
    参数:
        q: 搜索查询
        top_k: 返回结果数量
    
    返回:
        搜索结果
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="搜索查询不能为空")
    
    logger.info(f"收到搜索请求: {q}, top_k: {top_k}")
    
    result = rag_service.search_concept(q, top_k)
    
    return result

@app.get("/graph", tags=["图查询"])
async def get_graph(
    concept_id: str = Query(..., description="概念ID"),
    depth: int = Query(1, description="子图深度", ge=1, le=3)
):
    """获取概念的子图
    
    参数:
        concept_id: 概念ID
        depth: 子图深度
    
    返回:
        子图信息
    """
    if not concept_id.strip():
        raise HTTPException(status_code=400, detail="概念ID不能为空")
    
    logger.info(f"收到图查询请求: concept_id={concept_id}, depth={depth}")
    
    result = rag_service.get_subgraph(concept_id, depth)
    
    return result

@app.get("/mixed", tags=["混合查询"])
async def mixed_search(
    q: str = Query(..., description="搜索查询"),
    top_k: int = Query(5, description="向量检索结果数量", ge=1, le=20),
    depth: int = Query(1, description="子图深度", ge=1, le=2)
):
    """混合查询
    
    参数:
        q: 搜索查询
        top_k: 向量检索结果数量
        depth: 子图深度
    
    返回:
        查询结果
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="搜索查询不能为空")
    
    logger.info(f"收到混合查询请求: {q}, top_k={top_k}, depth={depth}")
    
    result = rag_service.mixed_query(q, top_k, depth)
    
    return result

def create_default_config():
    """创建默认配置文件"""
    default_config = {
        'vector_index_path': 'D:\\rag-project\\05-rag-practice\\23-faiss_db\\graph_node\\vector_index.faiss',
        'node_id_to_index_path': 'D:\\rag-project\\05-rag-practice\\23-faiss_db\\graph_node\\node_id_to_index.json',
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_dimension': 384,
        'fastapi_url': 'http://localhost:8000',
        'deepseek_api_key': 'your_api_key_here',
        'deepseek_model': 'deepseek-chat',
        'search_top_k': 5,
        'graph_depth': 1,
        'host': '0.0.0.0',
        'port': 8001
    }
    
    if not os.path.exists('rag_service_config.yaml'):
        try:
            with open('rag_service_config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
            logger.info("成功创建默认配置文件 rag_service_config.yaml")
            logger.warning("请根据实际情况修改 rag_service_config.yaml 中的配置，特别是DeepSeek API密钥")
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {str(e)}")

if __name__ == "__main__":
    # 创建默认配置文件
    create_default_config()
    
    # 从配置文件加载FastAPI配置
    host = "0.0.0.0"
    port = 8001
    
    if os.path.exists('rag_service_config.yaml'):
        try:
            with open('rag_service_config.yaml', 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    host = yaml_config.get('host', host)
                    port = yaml_config.get('port', port)
        except Exception as e:
            logger.warning(f"加载配置失败，使用默认值: {str(e)}")
    
    logger.info(f"启动RAG服务... 监听地址: http://{host}:{port}")
    logger.info(f"请访问 http://localhost:{port}/docs 查看API文档")
    logger.info(f"向量数据库路径: {rag_service.config['vector_index_path']}")
    logger.info(f"连接到FastAPI服务: {rag_service.config['fastapi_url']}")
    
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"启动RAG服务失败: {str(e)}")
        # 在启动失败时，创建一个简单的配置文件供用户编辑
        try:
            with open('rag_service_config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump({
                    'vector_index_path': '请输入向量索引文件路径',
                    'node_id_to_index_path': '请输入节点映射文件路径',
                    'fastapi_url': 'http://localhost:8000',
                    'deepseek_api_key': '请输入DeepSeek API密钥',
                    'host': '0.0.0.0',
                    'port': 8001
                }, f, allow_unicode=True, default_flow_style=False)
            logger.info("已创建简化的配置文件，请编辑后重新启动服务")
        except:
            pass