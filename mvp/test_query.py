#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试/query端点的脚本
"""

import requests
import json
import sys

# 设置请求URL
url = "http://127.0.0.1:8001/query"

# 读取请求体数据
try:
    with open("replay/demo_policy.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"读取请求数据成功: {data}")
except Exception as e:
    print(f"读取请求数据失败: {e}")
    sys.exit(1)

# 发送请求
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    
    # 打印响应
    print(f"\n响应状态码: {response.status_code}")
    print("响应内容:")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
except Exception as e:
    print(f"发送请求失败: {e}")
    sys.exit(1)