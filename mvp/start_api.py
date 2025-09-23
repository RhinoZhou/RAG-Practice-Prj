#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API服务启动脚本
"""

import os
import sys
import uvicorn
from app.config import config

def main():
    """启动API服务"""
    # 确保Python路径正确
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 打印启动信息
    print(f"\n=== RAG Demo API 服务启动 ===")
    print(f"API版本: 0.1.0")
    print(f"监听地址: {config.API_HOST}:{config.API_PORT}")
    print(f"日志级别: {config.LOG_LEVEL}")
    print(f"文档地址: http://{config.API_HOST}:{config.API_PORT}/docs")
    print(f"==========================\n")
    
    # 启动服务
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,  # 生产环境关闭热重载
        workers=1,     # 可以根据需要调整工作进程数
        log_level=config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()