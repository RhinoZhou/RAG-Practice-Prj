#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI界面启动脚本
"""

import os
import sys
import subprocess
from app.config import config

def main():
    """启动Streamlit UI界面"""
    # 确保Python路径正确
    project_root = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(project_root, "ui")
    
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 检查ui目录是否存在，如果不存在则创建
    if not os.path.exists(ui_dir):
        print(f"警告: UI目录不存在，创建目录 - {ui_dir}")
        os.makedirs(ui_dir)
    
    # 检查app.py文件是否存在，如果不存在则创建一个简单的演示应用
    app_py_path = os.path.join(ui_dir, "app.py")
    if not os.path.exists(app_py_path):
        print(f"警告: UI应用文件不存在，创建演示应用 - {app_py_path}")
        with open(app_py_path, "w", encoding="utf-8") as f:
            f.write("""
import streamlit as st
import requests
import os

# 设置页面配置
st.set_page_config(
    page_title="RAG Demo",
    page_icon="🤖",
    layout="wide"
)

# 设置API服务地址
def get_api_url():
    return os.environ.get("RAG_API_URL", "http://localhost:8000")

# 页面标题
st.title("RAG 问答系统演示")

# 侧边栏
with st.sidebar:
    st.header("系统配置")
    st.write(f"当前连接API: {get_api_url()}")
    st.divider()
    st.info("这是一个RAG系统的演示界面，提供问答功能。")

# 主界面
query = st.text_input("请输入您的问题:", "什么是RAG技术?")

if st.button("获取回答"):
    if not query.strip():
        st.error("请输入问题后再查询")
    else:
        try:
            # 显示加载状态
            with st.spinner("正在处理您的问题..."):
                # 调用API
                api_url = get_api_url()
                response = requests.post(
                    f"{api_url}/query",
                    json={"query": query},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # 显示结果
                st.subheader("回答")
                st.write(result.get("answer", "未获取到回答"))
                
                # 显示查询信息
                with st.expander("查询详情"):
                    st.write(f"原始查询: {result.get('query', '')}")
                    st.write(f"重写查询: {result.get('rewritten_query', '')}")
                    st.write(f"处理时间: {result.get('metadata', {}).get('processing_time', 0):.3f}秒")
                    
        except Exception as e:
            st.error(f"查询过程中出错: {str(e)}")

# 底部信息
st.divider()
st.caption("RAG Demo System v0.1.0 | 简化版演示")
""")
    
    # 打印启动信息
    print(f"\n=== RAG Demo UI 界面启动 ===")
    print(f"UI版本: 0.1.0")
    print(f"API服务地址: http://{config.API_HOST}:{config.API_PORT}")
    print(f"正在启动Streamlit界面...")
    print(f"Streamlit应用: {app_py_path}")
    print(f"==========================\n")
    
    # 启动Streamlit服务
    try:
        # 传递API地址作为环境变量
        env = os.environ.copy()
        env["RAG_API_URL"] = f"http://{config.API_HOST}:{config.API_PORT}"
        # 跳过Streamlit邮箱提示
        env["STREAMLIT_THEME_BASE"] = "light"
        env["STREAMLIT_SAVE_WARNINGS_TO_FILE"] = "false"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
        
        # 启动Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_py_path, "--server.port", "8501"],
            env=env,
            cwd=project_root
        )
    except KeyboardInterrupt:
        print("\nUI服务已停止")
    except Exception as e:
        print(f"启动UI服务失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()