#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 读取并打印策略建议树
print("=== 策略建议树 ===")
with open("results/strategy_recommendation_tree.json", "r", encoding="utf-8") as f:
    tree = json.load(f)
    print(json.dumps(tree, ensure_ascii=False, indent=2))

# 读取并打印摘要报告
print("\n=== 摘要报告 ===")
with open("results/summary_report.json", "r", encoding="utf-8") as f:
    report = json.load(f)
    print(json.dumps(report, ensure_ascii=False, indent=2))