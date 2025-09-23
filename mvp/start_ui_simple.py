#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版UI启动脚本，直接启动Streamlit服务
"""

import os
import sys
import subprocess
import time
from app.config import config

# 设置环境变量
os.environ["RAG_API_URL"] = f"http://{config.API_HOST}:{config.API_PORT}"
# 跳过Streamlit的各种提示和统计
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# 打印启动信息
print("\n=== RAG Demo UI 界面启动 ===")
print(f"UI版本: 0.1.0")
print(f"API服务地址: {os.environ['RAG_API_URL']}")
print(f"启动命令: streamlit run ui/app.py --server.port 8501 --server.headless true")
print(f"==========================\n")

# 直接启动Streamlit，使用--server.headless true选项避免邮箱提示
process = subprocess.Popen(
    [
        sys.executable,  # 使用当前Python解释器
        "-m",
        "streamlit", 
        "run", 
        "ui/app.py", 
        "--server.port", 
        "8501",
        "--server.headless",
        "true"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 实时打印输出
while True:
    # 读取输出行
    stdout_line = process.stdout.readline()
    stderr_line = process.stderr.readline()
    
    # 打印输出
    if stdout_line:
        print(stdout_line.strip())
        # 检查是否包含URL信息
        if "Local URL" in stdout_line or "Network URL" in stdout_line:
            print("\nStreamlit UI已成功启动！")
            print("请在浏览器中访问上述URL查看界面。")
    if stderr_line:
        print(f"错误: {stderr_line.strip()}")
    
    # 检查进程是否结束
    if process.poll() is not None:
        # 读取剩余输出
        for line in process.stdout.readlines():
            print(line.strip())
        for line in process.stderr.readlines():
            print(f"错误: {line.strip()}")
        print(f"\nStreamlit服务已停止，退出代码: {process.returncode}")
        sys.exit(process.returncode)
    
    # 短暂休眠，避免CPU占用过高
    time.sleep(0.1)