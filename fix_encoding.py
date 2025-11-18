#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复知识库文件的中文编码问题
"""

import os

KB_DIR = "knowledge_bases/default"

# 创建示例内容
example_content = {
    "diabetes.txt": "糖尿病是一种慢性疾病，患者需要控制饮食和血糖水平。\n\n" \
                   "饮食建议：\n" \
                   "1. 减少糖分摄入\n" \
                   "2. 增加膳食纤维\n" \
                   "3. 控制碳水化合物摄入量\n" \
                   "4. 定期监测血糖",
    
    "hypertension.txt": "高血压患者应该减少盐的摄入，定期测量血压。\n\n" \
                        "饮食建议：\n" \
                        "1. 每日盐摄入量不超过5克\n" \
                        "2. 多吃蔬菜水果\n" \
                        "3. 减少高脂肪食物摄入\n" \
                        "4. 适量运动"
}

# 重新创建文件
for filename, content in example_content.items():
    file_path = os.path.join(KB_DIR, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"成功创建 {filename}")
    except Exception as e:
        print(f"创建 {filename} 失败: {e}")

print("\n所有文件已重新创建完成！")