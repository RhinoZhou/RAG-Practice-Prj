#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地推理服务与调用示例

作者: Ph.D. Rhino

功能说明:
    展示如何封装模型预测函数为可访问API服务与简单HTTP调用

内容概述:
    用Flask轻量服务框架构建/predict POST接口，模拟返回模型预测结果
    包含服务端实现和客户端调用示例

执行流程:
    1. 定义基础推理函数（此处使用规则输出）
    2. 启动本地API服务监听端口
    3. 编写客户端脚本发送POST请求
    4. 服务端解析请求并返回JSON结果
    5. 打印响应耗时统计

输入说明:
    JSON请求格式: {"query": "请输入问题"}

输出格式:
    JSON响应格式: {"result": "预测结果", "processing_time": "处理时间"}
"""

import json
import time
import random
import threading
import requests
import sys
import subprocess

# 确保中文显示正常
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 依赖包列表
required_packages = ['flask']


def check_dependencies():
    """
    检查并安装必要的依赖包
    """
    print("正在检查依赖...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"发现缺失的依赖包: {', '.join(missing_packages)}")
        print("正在安装...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {package} 安装失败，请手动安装")
                sys.exit(1)
    else:
        print("所有必要的依赖包已安装完成。")


# 确保依赖已安装后再导入Flask
check_dependencies()
from flask import Flask, request, jsonify


class ModelSimulator:
    """
    模型预测模拟器，模拟真实模型的预测行为
    """
    
    def __init__(self):
        # 预定义的问题-答案映射，模拟知识库
        self.knowledge_base = {
            "什么是LoRA?": "LoRA（Low-Rank Adaptation）通过低秩分解实现高效微调，" \
                          "在保持模型性能的同时大幅减少可训练参数量。",
            "什么是RAG?": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的方法，" \
                        "通过从外部知识库检索相关信息来增强大型语言模型的回答质量。",
            "什么是注意力机制?": "注意力机制是一种让模型能够选择性地关注输入数据中重要部分的技术，" \
                              "在序列到序列任务中特别有效，使模型能够处理长距离依赖关系。",
            "什么是Transformer?": "Transformer是一种基于自注意力机制的神经网络架构，" \
                                "能够并行处理序列数据，已成为NLP领域的主流模型架构。",
            "什么是微调?": "微调是指在预训练模型的基础上，使用特定领域或任务的数据进行进一步训练，" \
                         "使模型更好地适应特定任务需求的过程。"
        }
        
        # 默认回复模板
        self.default_templates = [
            "根据我的理解，这个问题涉及到{topic}领域。",
            "关于这个问题，我需要更多的上下文信息来提供准确回答。",
            "这个问题比较复杂，通常来说，{topic}相关的解决方案包括多种方法。",
            "您的问题很有意思，{topic}是当前研究的热点之一。"
        ]
    
    def predict(self, query):
        """
        模拟模型预测过程
        
        Args:
            query: 输入的查询文本
            
        Returns:
            str: 模型预测结果
        """
        # 模拟处理时间，增加一点随机性
        processing_time = random.uniform(0.05, 0.2)
        time.sleep(processing_time)
        
        # 检查是否有预定义的回答
        for key, answer in self.knowledge_base.items():
            if key in query:
                return answer
        
        # 对于未知查询，根据关键词生成回复
        topics = ["机器学习", "深度学习", "自然语言处理", "模型优化"]
        topic = random.choice(topics)
        template = random.choice(self.default_templates)
        
        return template.format(topic=topic)


# 初始化Flask应用和模型模拟器
app = Flask(__name__)
model = ModelSimulator()


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    预测API端点，接收POST请求并返回预测结果
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 解析请求数据
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "请求格式错误，缺少'query'字段"}), 400
        
        # 获取查询文本
        query = data['query']
        
        # 调用模型进行预测
        result = model.predict(query)
        
        # 计算处理时间
        processing_time = round(time.time() - start_time, 4)
        
        # 返回JSON格式的响应
        return jsonify({
            "result": result,
            "processing_time": f"{processing_time}s"
        })
    
    except Exception as e:
        # 错误处理
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点，用于验证服务是否正常运行
    """
    return jsonify({"status": "ok", "service": "model-inference-api"})


class APIClient:
    """
    API客户端，用于向服务端发送请求
    """
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
        self.health_url = f"{base_url}/health"
    
    def check_health(self):
        """
        检查服务健康状态
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict(self, query):
        """
        发送预测请求
        
        Args:
            query: 查询文本
            
        Returns:
            dict: 包含预测结果的字典
        """
        payload = {"query": query}
        headers = {"Content-Type": "application/json"}
        
        # 记录请求时间
        start_time = time.time()
        
        try:
            response = requests.post(
                self.predict_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            # 计算响应时间
            response_time = round(time.time() - start_time, 4)
            
            if response.status_code == 200:
                result = response.json()
                result['response_time'] = f"{response_time}s"
                return result
            else:
                return {
                    "error": f"请求失败，状态码: {response.status_code}",
                    "response_time": f"{response_time}s"
                }
        except Exception as e:
            # 计算响应时间
            response_time = round(time.time() - start_time, 4)
            return {
                "error": str(e),
                "response_time": f"{response_time}s"
            }


def run_server():
    """
    运行Flask服务
    """
    # 关闭调试模式以避免重载问题
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_client_demo():
    """
    运行客户端演示
    """
    # 等待服务启动
    print("\n等待服务启动...")
    time.sleep(3)
    
    client = APIClient()
    
    # 检查服务健康状态
    print("检查服务健康状态...")
    if not client.check_health():
        print("错误: 无法连接到API服务，请检查服务是否正常启动")
        return
    print("✓ 服务健康状态正常")
    
    # 示例查询列表
    sample_queries = [
        "什么是LoRA?",
        "什么是RAG?",
        "解释一下注意力机制",
        "如何优化模型性能?",
        "Transformer架构的主要组件"
    ]
    
    print("\n===== 开始API调用演示 =====")
    
    # 测试查询
    total_response_time = 0
    successful_requests = 0
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n请求 {i}/{len(sample_queries)}:")
        print(f"Request: {{\"query\": \"{query}\"}}")
        
        # 发送请求
        result = client.predict(query)
        
        # 显示结果
        if "result" in result:
            print(f"Response: {{\"result\":\"{result['result']}\"}}")
            print(f"处理时间: {result['processing_time']}")
            print(f"响应时间: {result['response_time']}")
            
            # 解析时间字符串，计算总响应时间
            response_time = float(result['response_time'].replace('s', ''))
            total_response_time += response_time
            successful_requests += 1
        else:
            print(f"错误: {result.get('error', '未知错误')}")
    
    # 显示统计信息
    print("\n===== API调用统计 =====")
    if successful_requests > 0:
        avg_response_time = total_response_time / successful_requests
        print(f"成功请求数: {successful_requests}/{len(sample_queries)}")
        print(f"平均响应时间: {avg_response_time:.4f}秒")
    else:
        print("所有请求均失败")
    
    print("\n===== 演示完成 =====")


def main():
    """
    主函数，启动服务和客户端
    """
    print("===== 本地推理服务与调用示例 =====")
    
    # 在单独的线程中启动Flask服务
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
    server_thread.start()
    
    try:
        # 运行客户端演示
        run_client_demo()
        
        # 等待用户输入后退出
        print("\n按Enter键退出程序...")
        input()
    
    except KeyboardInterrupt:
        print("\n程序已中断")
    
    finally:
        print("程序已退出")


if __name__ == "__main__":
    main()