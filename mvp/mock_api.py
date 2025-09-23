#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Mock API服务，用于演示curl命令调用
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

# 创建FastAPI应用实例
app = FastAPI(
    title="Mock RAG API",
    description="用于演示curl命令的简化版API服务",
    version="0.1.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 单轮查询接口
@app.post("/query")
async def query_handler(request: Request):
    """
    模拟处理查询请求的接口
    """
    try:
        # 获取请求体
        body = await request.json()
        print(f"收到查询请求: {body}")
        
        # 返回模拟响应
        response = {
            "query": body.get("query", ""),
            "answer": "根据最新退款政策，用户可在购买后7天内申请全额退款。退款流程为：1. 登录账户；2. 找到订单；3. 点击申请退款；4. 等待审核，审核通过后1-3个工作日内到账。",
            "evidences": [
                {
                    "id": "doc1",
                    "content": "退款政策说明：用户可在购买后7天内申请全额退款。",
                    "score": 0.95,
                    "source": "policy_doc_2025"
                },
                {
                    "id": "doc2",
                    "content": "退款流程：登录账户->找到订单->点击申请退款->等待审核。",
                    "score": 0.92,
                    "source": "process_guide"
                }
            ],
            "status": "success",
            "metadata": {
                "request_id": "req-mock-001",
                "processing_time": 123
            },
            "route_decision": {
                "target_tools": ["bm25", "vector"],
                "top_k_destinations": [
                    {"name": "bm25", "weight": 0.6, "rank": 1},
                    {"name": "vector", "weight": 0.4, "rank": 2}
                ]
            }
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        print(f"处理请求失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# 健康检查接口
@app.get("/health")
def health_check():
    """
    健康检查接口
    """
    return {"ok": True, "version": "0.1.0"}

# 根路径
@app.get("/")
def root():
    """
    API根路径
    """
    return {
        "name": "Mock RAG API",
        "version": "0.1.0",
        "description": "用于演示curl命令的简化版API服务"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)