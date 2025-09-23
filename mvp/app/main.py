#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI入口文件 - RAG演示系统
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import time
import uuid
import logging
import asyncio

# 导入项目组件
from app.config import config
from app.observability import setup_logging, RequestContext, log_request, Obs
from app.utils import generate_request_id, truncate_text
from app.models import Query, Evidence, PipelineResult
from app.preprocess import normalize
from app.rewrite import rewrite
from app.self_query import SelfQueryParser
from app.router import decide
from app.search_bm25 import bm25_searcher
from app.search_vector import search_vector
from app.fuse import rrf_and_aggregate
from app.rerank import rerank
from app.answer import compose

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG演示系统",
    description="检索增强生成演示系统API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询文本")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    use_cache: bool = Field(True, description="是否使用缓存")

class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""
    queries: List[Dict[str, Any]] = Field(
        ..., 
        description="查询列表，每个查询包含'query'和可选的'context'",
        example=[{"query": "什么是RAG", "context": {"domain": "AI"}}]
    )
    parallel: bool = Field(False, description="是否并行处理")

class LogLevelRequest(BaseModel):
    """日志级别请求模型"""
    level: str = Field(..., description="日志级别", example="INFO")

# 完整RAGPipeline类实现
class RAGPipeline:
    """完整的RAG流水线类，实现端到端的查询处理流程"""
    
    def __init__(self):
        """初始化RAG流水线"""
        logger.info("RAGPipeline已初始化")
        # 初始化各组件
        self.initialize_components()
    
    def initialize_components(self):
        """初始化流水线组件"""
        # 初始化SelfQueryParser
        self.self_query_parser = SelfQueryParser()
        # 其他组件已经通过全局实例初始化
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        """处理单个查询请求，实现完整的端到端流程"""
        # 创建观测对象
        obs = Obs(request_id=generate_request_id())
        obs.start_step("total")
        
        try:
            # 1. 预处理查询
            obs.start_step("preprocess")
            preprocessed_query = normalize(query, context)
            obs.end_step()
            
            # 2. 创建Query对象
            q = Query(
                text=preprocessed_query,
                filters={},
                rewrite_candidates=[],
                top_k=config.TOP_K,
                user_id=context.get("user_id") if context else None,
                session_id=context.get("session_id") if context else None,
                metadata=context or {}
            )
            
            # 3. 查询重写
            obs.start_step("rewrite")
            rewritten_query = rewrite(q)
            q.rewritten_query = rewritten_query.text
            obs.add_event("query_rewritten", {"original": q.text, "rewritten": rewritten_query.text})
            obs.end_step()
            
            # 4. Self-Query扩展
            obs.start_step("self_query")
            parsed_result = self.self_query_parser.parse_query(rewritten_query.text)
            q.filters = parsed_result.get("constraints", {})
            q.metadata.update(parsed_result.get("metadata", {}))
            obs.add_event("query_enriched", {"filters": q.filters})
            obs.end_step()
            
            # 5. 路由决策
            obs.start_step("route")
            route_decision = decide(q, SearchParams(query=q.text, top_k=q.top_k), {})
            # 使用target_tools作为策略信息
            obs.add_event("route_decision", {"strategies": route_decision.target_tools})
            obs.end_step()
            
            # 6. 执行检索
            obs.start_step("retrieval")
            all_evidences = []
            
            # 执行BM25检索
            bm25_evidences = bm25_searcher.search(q.text, {"top_k": q.top_k}, q.filters)
            all_evidences.extend(bm25_evidences)
            obs.add_event("bm25_retrieval", {"count": len(bm25_evidences)})
            
            # 执行向量检索
            vector_evidences = search_vector(q)
            all_evidences.extend(vector_evidences)
            obs.add_event("vector_retrieval", {"count": len(vector_evidences)})
            obs.end_step()
            
            # 7. 结果融合
            obs.start_step("fuse")
            fused_evidences = rrf_and_aggregate(bm25_evidences, vector_evidences, top_k=config.TOP_K * 2)
            obs.add_event("results_fused", {"count": len(fused_evidences)})
            obs.end_step()
            
            # 8. 重排序
            obs.start_step("rerank")
            reranked_evidences = rerank(q, fused_evidences, top_k=config.TOP_K)
            obs.add_event("results_reranked", {"count": len(reranked_evidences)})
            obs.end_step()
            
            # 9. 生成回答
            obs.start_step("generate")
            answer, citations = compose(q, reranked_evidences)
            obs.add_event("answer_generated", {"citation_count": len(citations)})
            obs.end_step()
            
            # 10. 创建流水线结果
            pipeline_result = PipelineResult(
                query=q.text,
                answer=answer,
                evidences=reranked_evidences,
                logs=[event for event in obs.events],
                timings_ms=obs.timings(),
                status="success",
                error=None,
                route_decision=route_decision,
                model_usage=None
            )
            
            # 添加额外的元数据到结果字典中
            result_dict = pipeline_result.dict()
            result_dict["rewritten_query"] = q.rewritten_query
            result_dict["citations"] = citations
            result_dict["strategy"] = route_decision.target_tools[0] if route_decision.target_tools else "default"
            result_dict["metadata"] = {
                "processing_time": obs.timings().get("total", 0),
                "evidences_count": len(reranked_evidences),
                "request_id": obs.request_id,
                "citation_count": len(citations)
            }
            
            # 记录观测数据
            obs.end_step()  # 结束total步骤
            obs.write_to_log()
            
            # 返回结果字典
            return result_dict
            
        except Exception as e:
            # 记录错误
            logger.error(f"Query processing failed: {str(e)}")
            obs.add_event("error", {"type": type(e).__name__, "message": str(e)})
            obs.end_step()  # 结束total步骤
            obs.write_to_log()
            
            # 返回错误结果
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "metadata": {
                    "request_id": obs.request_id,
                    "processing_time": obs.timings().get("total", 0)
                }
            }
    
    async def batch_process(self, queries: List[Dict[str, Any]], parallel: bool = False) -> List[Dict[str, Any]]:
        """批量处理查询"""
        results = []
        
        if parallel:
            # 并行处理
            tasks = []
            for i, q in enumerate(queries):
                query = q.get("query", "")
                context = q.get("context", None)
                use_cache = q.get("use_cache", True)
                # 使用线程池执行同步函数
                tasks.append(
                    asyncio.to_thread(
                        self._process_query_with_index, 
                        query, context, use_cache, i
                    )
                )
            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
        else:
            # 串行处理
            for i, q in enumerate(queries):
                result = self._process_query_with_index(
                    q.get("query", ""), 
                    q.get("context", None), 
                    q.get("use_cache", True),
                    i
                )
                results.append(result)
        
        return results
        
    def _process_query_with_index(self, query: str, context: Optional[Dict[str, Any]], 
                                use_cache: bool, index: int) -> Dict[str, Any]:
        """带索引的查询处理，用于批量处理"""
        try:
            result = self.process_query(query, context, use_cache)
            result["batch_index"] = index
            result["status"] = "success"
        except Exception as e:
            result = {
                "query": query,
                "status": "error",
                "error": str(e),
                "batch_index": index
            }
        return result

# 创建全局流水线实例
pipeline = RAGPipeline()

# HTTP请求日志中间件
@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    """HTTP请求日志中间件"""
    # 生成请求ID
    request_id = generate_request_id()
    
    # 设置请求上下文
    with RequestContext(request_id):
        # 记录请求信息
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": str(request.client)
            }
        )
        
        # 处理请求
        try:
            response = await call_next(request)
        except Exception as e:
            # 记录异常
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e)
                }
            )
            raise
        
        # 记录响应信息
        processing_time = time.time() - start_time
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "processing_time": processing_time
            }
        )
        
        # 在响应头中添加请求ID
        response.headers["X-Request-ID"] = request_id
        
        return response

# 健康检查接口
@app.get("/health")
def health_check():
    """
    健康检查接口
    返回系统状态信息
    """
    return {
        "ok": True,
        "version": "0.1.0",
        "services": {
            "api": "running",
            "pipeline": "initialized"
        }
    }

# 单轮查询接口
@app.post("/query")
def query_handler(request: QueryRequest):
    """
    单轮查询处理接口
    处理用户的查询请求并返回回答
    """
    try:
        result = pipeline.process_query(
            query=request.query,
            context=request.context,
            use_cache=request.use_cache
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 批量查询接口
@app.post("/batch_query")
async def batch_query_handler(request: BatchQueryRequest):
    """
    批量查询处理接口
    批量处理多个查询请求
    """
    try:
        results = await pipeline.batch_process(
            queries=request.queries,
            parallel=request.parallel
        )
        return JSONResponse(content={
            "total_queries": len(request.queries),
            "results": results
        })
    except Exception as e:
        logger.error(f"Batch query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取系统配置接口
@app.get("/config")
def get_config():
    """
    获取系统配置信息
    返回当前系统的配置参数
    """
    return {
        "api": {
            "host": config.API_HOST,
            "port": config.API_PORT
        },
        "retrieval": {
            "top_k": config.TOP_K
        },
        "paths": {
            "data_dir": config.DATA_DIR,
            "index_dir": config.INDEX_DIR
        },
        "logging": {
            "level": config.LOG_LEVEL,
            "log_dir": config.LOG_DIR
        }
    }

# 设置日志级别接口
@app.post("/set_log_level")
def set_log_level(request: LogLevelRequest):
    """
    设置日志级别接口
    动态调整系统日志级别
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if request.level.upper() not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
        )
    
    try:
        config.LOG_LEVEL = request.level.upper()
        root_logger = logging.getLogger()
        root_logger.setLevel(config.LOG_LEVEL)
        logger.info(f"Log level changed to {config.LOG_LEVEL}")
        return {"success": True, "message": f"Log level set to {config.LOG_LEVEL}"}
    except Exception as e:
        logger.error(f"Failed to change log level: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 应用启动事件
@app.on_event("startup")
def startup_event():
    """应用启动事件处理"""
    logger.info("RAG Demo System starting up")
    # 确保必要的目录存在
    config.ensure_directories()
    
    # 初始化各个组件
    # TODO: 实现组件的初始化逻辑
    logger.info("All components initialized")

# 应用关闭事件
@app.on_event("shutdown")
def shutdown_event():
    """应用关闭事件处理"""
    logger.info("RAG Demo System shutting down")
    # 清理资源
    # TODO: 实现资源清理逻辑

# 根路径
@app.get("/")
def root():
    """API根路径"""
    return {
        "name": "RAG Demo System",
        "version": "0.1.0",
        "description": "检索增强生成演示系统",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)