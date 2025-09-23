#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI面板
用于与RAG系统交互的用户界面
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional, List

# 设置页面配置
st.set_page_config(
    page_title="RAG系统演示",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局设置
API_URL = "http://127.0.0.1:8000/query"
MAX_HISTORY_SIZE = 10

class RAGUI:
    """RAG系统用户界面"""
    
    def __init__(self):
        """初始化UI"""
        # 初始化会话状态
        if "history" not in st.session_state:
            st.session_state.history = []
        
        if "pipeline_results" not in st.session_state:
            st.session_state.pipeline_results = None
        
        if "api_url" not in st.session_state:
            st.session_state.api_url = API_URL
        
    def render_header(self):
        """渲染页面头部"""
        st.title("RAG系统演示")
        st.subheader("基于检索增强生成的智能问答系统")
        st.markdown("---")
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.header("功能开关")
            
            # 功能开关
            options = {
                "rewrite": st.checkbox("启用查询重写 (Rewrite)", value=True),
                "self_query": st.checkbox("启用Self-Query", value=True),
                "moe": st.checkbox("启用MoE路由", value=True),
                "rerank": st.checkbox("启用重排序 (Rerank)", value=True),
                "hyde": st.checkbox("启用HyDE", value=False),
                "degrade": st.checkbox("启用降级模式", value=False)
            }
            
            st.markdown("---")
            
            # 检索参数
            st.header("检索参数")
            params = {
                "k": st.slider("返回结果数量 (k)", min_value=1, max_value=20, value=5),
                "similarity_threshold": st.slider("相似度阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05),
                "retrievers": st.multiselect(
                    "检索器选择",
                    options=["BM25", "向量检索", "混合检索"],
                    default=["BM25", "向量检索"]
                )
            }
            
            return options, params
    
    def render_main_content(self, options: Dict[str, bool], params: Dict[str, Any]):
        """渲染主内容区"""
        # 输入区域
        self._render_input_area(options, params)
        
        # 结果展示区域
        if st.session_state.pipeline_results:
            self._render_pipeline_results(st.session_state.pipeline_results)
    
    def _render_input_area(self, options: Dict[str, bool], params: Dict[str, Any]):
        """渲染输入区域"""
        with st.form("query_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_query = st.text_input("问题", placeholder="请输入您的问题...")
            
            with col2:
                request_id = st.text_input("Request ID", placeholder="可选")
            
            user_profile = st.text_area(
                "用户画像 JSON", 
                placeholder='例如: {"user_id": "test_user", "domain": "tech"}',
                height=100
            )
            
            # 提交按钮
            submitted = st.form_submit_button("运行端到端管道", type="primary")
            
            if submitted and user_query:
                # 构建请求参数
                try:
                    context = json.loads(user_profile) if user_profile else {}
                    if request_id:
                        context["request_id"] = request_id
                except json.JSONDecodeError:
                    st.error("用户画像 JSON 格式错误")
                    return
                
                # 构建完整的查询参数
                query_params = {
                    "query": user_query,
                    "top_k": params["k"],
                    "options": options,
                    "params": params,
                    "context": context
                }
                
                # 调用API获取管道结果
                with st.spinner("正在运行端到端管道..."):
                    try:
                        pipeline_results = self._call_api(query_params)
                        st.session_state.pipeline_results = pipeline_results
                        
                        # 添加到历史记录
                        st.session_state.history.append({
                            "role": "user",
                            "content": user_query,
                            "params": query_params,
                            "timestamp": time.time()
                        })
                        
                    except Exception as e:
                        st.error(f"API调用失败: {str(e)}")
                        
    def _render_pipeline_results(self, results: Dict[str, Any]):
        """渲染管道结果"""
        st.markdown("## 端到端管道结果")
        
        # 1. 预处理与重写候选
        with st.expander("1. 预处理与重写候选", expanded=True):
            if results.get("rewritten_query"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始查询")
                    st.write(results.get("query", ""))
                with col2:
                    st.subheader("重写查询")
                    st.write(results.get("rewritten_query", ""))
            else:
                st.info("查询重写功能未启用或无重写候选")
        
        # 2. Self-Query 过滤
        with st.expander("2. Self-Query 过滤", expanded=False):
            if results.get("filters"):
                st.json(results.get("filters", {}))
            else:
                st.info("Self-Query 功能未启用或无过滤条件")
        
        # 3. 路由决策
        with st.expander("3. 路由决策", expanded=False):
            route_decision = results.get("route_decision", {})
            if route_decision:
                st.subheader("选择的策略")
                st.write(results.get("strategy", ""))
                
                if route_decision.get("target_tools"):
                    st.subheader("目标工具")
                    st.write(route_decision["target_tools"])
                
                if route_decision.get("rules_applied"):
                    st.subheader("应用的规则")
                    st.write(route_decision["rules_applied"])
            else:
                st.info("路由决策信息不可用")
        
        # 4. 检索候选（原始）
        with st.expander("4. 检索候选（原始）", expanded=False):
            if "evidences" in results and len(results["evidences"]) > 0:
                for i, evidence in enumerate(results["evidences"]):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**[{i+1}] {evidence.get('title', '未命名文档')}**")
                            st.caption(evidence.get('content', '')[:300] + "...")
                        with col2:
                            st.write(f"分数: {evidence.get('score', 0):.4f}")
                            st.write(f"来源: {evidence.get('source', '')}")
            else:
                st.info("未检索到相关文档")
        
        # 5. 融合与重排 Top 列表
        with st.expander("5. 融合与重排 Top 列表", expanded=False):
            if "evidences" in results and len(results["evidences"]) > 0:
                # 模拟重排结果（实际项目中应该从API获取）
                sorted_evidences = sorted(results["evidences"], key=lambda x: x.get('score', 0), reverse=True)
                
                col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
                col1.write("**排名**")
                col2.write("**标题**")
                col3.write("**分数**")
                col4.write("**来源**")
                
                for i, evidence in enumerate(sorted_evidences[:params["k"]]):
                    col1.write(f"{i+1}")
                    col2.write(evidence.get('title', '未命名文档'))
                    col3.write(f"{evidence.get('score', 0):.4f}")
                    col4.write(evidence.get('source', ''))
            else:
                st.info("无融合与重排结果")
        
        # 6. 最终答案与引文
        with st.expander("6. 最终答案与引文", expanded=True):
            if results.get("answer"):
                st.subheader("答案")
                st.write(results["answer"])
                
                if results.get("citations"):
                    st.subheader("引文")
                    for citation in results["citations"]:
                        st.markdown(f"- [{citation.get('id', '')}] {citation.get('title', '')}")
            else:
                st.info("未生成答案")
        
        # 7. 链路日志与耗时（折叠）
        with st.expander("7. 链路日志与耗时", expanded=False):
            if results.get("logs"):
                st.subheader("处理日志")
                for log in results.get("logs", []):
                    st.json(log)
                
                if results.get("timings_ms"):
                    st.subheader("处理耗时 (ms)")
                    st.json(results["timings_ms"])
            else:
                st.info("链路日志不可用")
    
    def _call_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用RAG API
        
        Args:
            params: 查询参数
            
        Returns:
            API返回结果
        """
        try:
            url = st.session_state.api_url
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=params, headers=headers, timeout=120)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API请求失败: HTTP {response.status_code}, {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"网络错误: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("API返回的不是有效的JSON格式")
        except Exception as e:
            raise Exception(f"未知错误: {str(e)}")
    
    def run(self):
        """运行UI应用"""
        self.render_header()
        options, params = self.render_sidebar()
        self.render_main_content(options, params)

# 创建并运行UI
if __name__ == "__main__":
    rag_ui = RAGUI()
    rag_ui.run()