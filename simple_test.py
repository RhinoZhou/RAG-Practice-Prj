#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试程序
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

print("开始简化测试...")

# 创建目录
os.makedirs("outputs", exist_ok=True)

# 生成简单测试数据
corpus = [
    {"id": "doc1", "title": "人工智能", "content": "人工智能是计算机科学的一个分支"},
    {"id": "doc2", "title": "机器学习", "content": "机器学习是人工智能的核心技术"},
    {"id": "doc3", "title": "深度学习", "content": "深度学习是机器学习的一个子领域"}
]

queries = [
    {"id": "q1", "text": "什么是人工智能？"},
    {"id": "q2", "text": "机器学习技术"}
]

print("✓ 生成测试数据")

# 构建TF-IDF模型
texts = [doc["title"] + ": " + doc["content"] for doc in corpus]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

print("✓ 构建TF-IDF模型")

# 执行检索
query = "人工智能"
query_vector = vectorizer.transform([query])
similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
top_indices = similarities.argsort()[-3:][::-1]

results = []
for i, idx in enumerate(top_indices):
    results.append({
        "doc_id": corpus[idx]["id"],
        "score": similarities[idx],
        "rank": i + 1
    })

print("✓ 执行检索")

# 保存结果
with open("outputs/simple_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✓ 保存结果到 outputs/simple_test_results.json")

# 创建简单图表
plt.figure(figsize=(10, 6))
scores = [r["score"] for r in results]
doc_ids = [r["doc_id"] for r in results]

plt.bar(doc_ids, scores)
plt.title("检索结果分数", fontsize=16)
plt.xlabel("文档ID", fontsize=14)
plt.ylabel("相似度分数", fontsize=14)
plt.grid(True, alpha=0.3)

plt.savefig("outputs/simple_test_chart.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ 生成图表 outputs/simple_test_chart.png")
print("简化测试完成！")

