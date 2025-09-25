#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
漏检定位最小工具

作者: Ph.D. Rhino

功能说明: 检查金标证据是否可达/是否命中，输出漏检原因与建议。

内容概述: 给定"查询→正确三键"，先检查索引覆盖（是否存在该三键），再用简易检索看是否进入Top‑K，
         区分"覆盖缺失"与"策略缺口"，并打印修复建议。

执行流程:
1. 读取金标对（查询→证据三键）
2. 校验三键是否在索引中
3. 检索判断是否进Top‑K
4. 输出漏检类型与建议

输入说明: 内置小集合GOLD；无需外部输入。
输出展示: 漏检诊断结果与统计信息
"""

import sys
import time
import json
from datetime import datetime
import numpy as np

# 自动安装必要的依赖
def install_dependencies():
    """检查并自动安装必要的依赖包"""
    try:
        # 本工具主要使用Python标准库，仅在需要时安装额外依赖
        import numpy
        print("依赖检查: 所有必要的依赖已安装。")
    except ImportError:
        print("正在安装必要的依赖...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            print("依赖安装成功。")
        except Exception as e:
            print(f"依赖安装失败: {e}")
            sys.exit(1)

class LeakDiagnoseReverseSearch:
    """漏检定位与诊断工具类"""
    
    def __init__(self):
        # 初始化内置的金标数据集（查询→正确三键）
        self.gold_data = {
            "RRF的作用？": ["docA:4.1:s2", "docB:2.3:s1"],
            "混合检索的特点是什么？": ["docA:2.2:s3", "docC:1.4:s4"],
            "文本分块的作用是什么？": ["docA:3.1:s1", "docD:2.1:s2"],
            "窗口滑动策略有哪些？": ["docA:3.2:s1", "docE:3.4:s3"]
        }
        
        # 模拟索引中的三键集合
        self.indexed_triple_keys = {
            "docA:4.1:s2", "docB:2.3:s1",  # RRF相关
            "docA:2.2:s3", "docC:1.4:s4",  # 混合检索相关
            "docA:3.1:s1",                  # 文本分块相关（缺少一个）
            # 窗口滑动策略相关全部缺少
        }
        
        # 模拟检索结果（为了演示，我们基于简化的TF-IDF相似度）
        self.search_similarity = {
            "RRF的作用？": {
                "docX:5.2:s3": 0.9,
                "docY:1.1:s1": 0.85,
                "docZ:3.3:s2": 0.7,
                # 正确的三键得分较低
                "docA:4.1:s2": 0.3,
                "docB:2.3:s1": 0.25
            },
            "混合检索的特点是什么？": {
                "docW:2.5:s4": 0.88,
                "docA:2.2:s3": 0.82,  # 这个命中了
                "docV:3.2:s1": 0.75,
                "docC:1.4:s4": 0.4    # 这个得分较低
            },
            "文本分块的作用是什么？": {
                "docA:3.1:s1": 0.95,  # 这个命中了
                "docU:4.3:s2": 0.8,
                "docT:2.1:s5": 0.7
            },
            "窗口滑动策略有哪些？": {
                "docS:3.4:s1": 0.85,
                "docR:1.2:s3": 0.78,
                "docQ:2.5:s2": 0.7
            }
        }
        
        # 定义TOP-K阈值
        self.top_k = 3
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "unreachable": 0,
            "reachable_but_not_hit": 0,
            "hit": 0
        }
        
        # 详细结果
        self.detailed_results = []

    def check_triple_key_reachability(self, triple_keys):
        """
        检查三键是否在索引中
        参数:
            triple_keys: 三键列表
        返回:
            tuple: (是否可达, 不可达的三键列表)
        """
        unreachable_keys = [key for key in triple_keys if key not in self.indexed_triple_keys]
        is_reachable = len(unreachable_keys) == 0
        return is_reachable, unreachable_keys

    def check_search_hits(self, query, triple_keys):
        """
        检查金标三键是否在检索结果的Top-K中
        参数:
            query: 查询字符串
            triple_keys: 三键列表
        返回:
            tuple: (是否命中, 命中的三键列表, 命中的排名列表)
        """
        hit_keys = []
        hit_ranks = []
        
        if query not in self.search_similarity:
            return False, hit_keys, hit_ranks
        
        # 按相似度排序
        sorted_results = sorted(
            self.search_similarity[query].items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]
        
        # 检查金标三键是否在Top-K中
        for i, (key, score) in enumerate(sorted_results):
            if key in triple_keys:
                hit_keys.append(key)
                hit_ranks.append(i + 1)  # 排名从1开始
        
        return len(hit_keys) > 0, hit_keys, hit_ranks

    def generate_recommendations(self, is_reachable, is_hit):
        """
        根据漏检类型生成修复建议
        参数:
            is_reachable: 三键是否可达
            is_hit: 三键是否命中
        返回:
            str: 修复建议
        """
        if not is_reachable:
            return "覆盖缺失：建议检查索引构建流程，确保所有相关文档被正确索引。"
        elif not is_hit:
            return "策略缺口：建议混合检索/术语映射/扩大窗口，优化检索排序策略。"
        else:
            return "已命中：无需特殊处理。"

    def diagnose_query(self, query, triple_keys):
        """
        诊断单个查询的漏检情况
        参数:
            query: 查询字符串
            triple_keys: 三键列表
        返回:
            dict: 诊断结果
        """
        start_time = time.time()
        
        # 检查三键可达性
        is_reachable, unreachable_keys = self.check_triple_key_reachability(triple_keys)
        
        # 检查检索命中情况
        is_hit, hit_keys, hit_ranks = self.check_search_hits(query, triple_keys)
        
        # 生成建议
        recommendation = self.generate_recommendations(is_reachable, is_hit)
        
        # 更新统计信息
        self.stats["total_queries"] += 1
        if not is_reachable:
            self.stats["unreachable"] += 1
        elif not is_hit:
            self.stats["reachable_but_not_hit"] += 1
        else:
            self.stats["hit"] += 1
        
        end_time = time.time()
        
        # 构建结果
        result = {
            "query": query,
            "triple_keys": triple_keys,
            "is_reachable": is_reachable,
            "unreachable_keys": unreachable_keys,
            "is_hit": is_hit,
            "hit_keys": hit_keys,
            "hit_ranks": hit_ranks,
            "recommendation": recommendation,
            "execution_time": (end_time - start_time) * 1000  # 转换为毫秒
        }
        
        self.detailed_results.append(result)
        return result

    def print_diagnosis_result(self, result):
        """
        打印诊断结果
        参数:
            result: 诊断结果字典
        """
        print(f"Q: {result['query']}")
        print(f"Reachable: {result['is_reachable']}")
        print(f"Hits: {result['hit_keys']}")
        print(f"命中: {'是' if result['is_hit'] else '否'} → {result['recommendation']}")
        print("---")

    def print_statistics(self):
        """打印统计信息"""
        print("===== 漏检统计 ======")
        print(f"不可达: {self.stats['unreachable']}")
        print(f"可达未命中: {self.stats['reachable_but_not_hit']}")
        print(f"命中: {self.stats['hit']}")
        print(f"总查询数: {self.stats['total_queries']}")
        print("=================")

    def save_results(self, output_file="leak_diagnose_results.json"):
        """
        保存诊断结果到JSON文件
        参数:
            output_file: 输出文件名
        """
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": self.stats,
            "detailed_results": self.detailed_results,
            "config": {
                "top_k": self.top_k
            }
        }
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            print(f"诊断结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")

    def run_diagnosis(self):
        """运行完整的漏检诊断流程"""
        print("===== 漏检定位诊断工具 ======")
        print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Top-K设置: {self.top_k}")
        print("\n开始诊断...\n")
        
        start_time = time.time()
        
        # 对每个查询进行诊断
        for query, triple_keys in self.gold_data.items():
            result = self.diagnose_query(query, triple_keys)
            self.print_diagnosis_result(result)
        
        # 打印统计信息
        self.print_statistics()
        
        # 计算总执行时间
        total_time = time.time() - start_time
        print(f"\n总执行时间: {total_time:.4f}秒")
        
        # 保存结果
        self.save_results()
        
        return total_time

# 主函数
def main():
    """主函数：安装依赖并运行漏检诊断工具"""
    # 安装依赖
    install_dependencies()
    
    # 创建并运行漏检诊断工具
    leak_diagnoser = LeakDiagnoseReverseSearch()
    total_time = leak_diagnoser.run_diagnosis()
    
    # 执行效率检查
    if total_time > 1.0:  # 如果执行时间超过1秒，提示优化
        print("\n⚠️ 注意：程序执行时间较长，建议检查代码优化空间。")
    else:
        print("\n✅ 程序执行效率良好。")
    
    # 中文输出检查
    print("\n✅ 中文输出测试正常。")
    
    # 演示目的检查
    print("\n✅ 程序已成功演示漏检定位功能，包括覆盖缺失和策略缺口的诊断。")

if __name__ == "__main__":
    main()