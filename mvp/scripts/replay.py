#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询重放脚本
负责重放历史查询并评估系统性能
"""

import os
import json
import argparse
import time
import uuid
import requests
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from app.config import config

class QueryReplayer:
    """查询重放器"""
    
    def __init__(self, queries_file: str, api_url: str = "http://localhost:8000"):
        """
        初始化查询重放器
        
        Args:
            queries_file: 包含查询的文件路径
            api_url: API服务地址
        """
        self.queries_file = queries_file
        self.api_url = api_url
        self.queries = []
        self.results = []
    
    def load_queries(self) -> int:
        """
        加载查询列表
        
        Returns:
            加载的查询数量
        """
        # TODO: 实现查询加载功能
        try:
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                self.queries = json.load(f)
            
            # 确保查询是列表格式
            if not isinstance(self.queries, list):
                self.queries = [self.queries]
            
            print(f"成功加载 {len(self.queries)} 个查询")
            return len(self.queries)
        except Exception as e:
            print(f"加载查询文件失败: {e}")
            self.queries = []
            return 0
    
    def replay_queries(self, delay: float = 0.5, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        重放查询
        
        Args:
            delay: 查询之间的延迟（秒）
            timeout: 请求超时时间（秒）
        
        Returns:
            重放结果列表
        """
        # TODO: 实现查询重放功能
        if not self.queries:
            print("没有加载任何查询，无法重放")
            return []
        
        self.results = []
        total_time = 0.0
        
        print(f"开始重放查询，目标API: {self.api_url}")
        
        for i, query_data in enumerate(self.queries, 1):
            # 准备查询参数
            query_id = str(uuid.uuid4())
            query_text = query_data.get("query", "")
            
            if not query_text:
                print(f"跳过第 {i} 个查询（无查询文本）")
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 发送查询请求，传入原始查询数据以支持更多参数
                response = self._send_query(query_text, query_id, timeout, query_data)
                
                # 记录结束时间
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                
                # 处理响应
                if response.status_code == 200:
                    result = response.json()
                    result["query_id"] = query_id
                    result["duration"] = duration
                    result["timestamp"] = start_time
                    result["original_query"] = query_data
                    self.results.append(result)
                    
                    print(f"查询 {i}/{len(self.queries)}: 成功，耗时: {duration:.2f}秒")
                else:
                    print(f"查询 {i}/{len(self.queries)}: 失败，状态码: {response.status_code}")
                    
                    # 记录失败结果
                    error_result = {
                        "query_id": query_id,
                        "query": query_text,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}",
                        "duration": duration,
                        "timestamp": start_time,
                        "original_query": query_data
                    }
                    self.results.append(error_result)
                    
            except Exception as e:
                # 记录异常
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                
                print(f"查询 {i}/{len(self.queries)}: 异常，错误: {str(e)}")
                
                error_result = {
                    "query_id": query_id,
                    "query": query_text,
                    "error": str(e),
                    "duration": duration,
                    "timestamp": start_time,
                    "original_query": query_data
                }
                self.results.append(error_result)
            
            # 延迟下一个查询
            if i < len(self.queries) and delay > 0:
                time.sleep(delay)
        
        avg_time = total_time / len(self.queries) if self.queries else 0
        print(f"\n查询重放完成！")
        print(f"总查询数: {len(self.queries)}")
        print(f"成功数: {sum(1 for r in self.results if 'error' not in r)}")
        print(f"失败数: {sum(1 for r in self.results if 'error' in r)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {avg_time:.2f}秒/查询")
        
        return self.results
    
    def _send_query(self, query: str, query_id: str, timeout: float, query_data: Dict = None) -> requests.Response:
        """
        发送查询请求到API
        
        Args:
            query: 查询文本
            query_id: 查询ID
            timeout: 超时时间
            query_data: 原始查询数据，包含可能的额外参数
        
        Returns:
            HTTP响应对象
        """
        url = f"{self.api_url}/query"
        headers = {
            "Content-Type": "application/json"
        }
        
        # 构建请求数据，确保包含所有必要字段
        data = {
            "query": query,
            "query_id": query_id,
            "top_k": 5
        }
        
        # 如果有原始查询数据，合并额外参数
        if query_data:
            # 合并user_profile、filters等字段
            if "user_profile" in query_data:
                data["user_profile"] = query_data["user_profile"]
            if "filters" in query_data:
                data["filters"] = query_data["filters"]
            if "top_k" in query_data:
                data["top_k"] = query_data["top_k"]
            
            # 合并其他自定义字段
            for key, value in query_data.items():
                if key not in ["query", "query_id", "original_query"]:
                    data[key] = value
        
        return requests.post(url, json=data, headers=headers, timeout=timeout)
    
    def save_results(self, output_file: str = "replay_results.json") -> str:
        """
        保存重放结果
        
        Args:
            output_file: 输出文件路径
        
        Returns:
            输出文件路径
        """
        # TODO: 实现结果保存功能
        if not self.results:
            print("没有重放结果，无法保存")
            return ""
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"重放结果已保存到 {output_file}")
        return output_file
    
    def generate_report(self, results_file: str = "replay_report.json") -> Dict[str, Any]:
        """
        生成重放报告
        
        Args:
            results_file: 报告文件路径
        
        Returns:
            报告数据
        """
        # TODO: 实现报告生成功能
        if not self.results:
            print("没有重放结果，无法生成报告")
            return {}
        
        # 计算统计信息
        report = {
            "total_queries": len(self.queries),
            "successful_queries": sum(1 for r in self.results if 'error' not in r),
            "failed_queries": sum(1 for r in self.results if 'error' in r),
            "response_times": [r.get("duration", 0) for r in self.results],
            "error_types": {},
            "timestamp": self._get_current_timestamp()
        }
        
        # 计算平均响应时间
        if report["response_times"]:
            report["avg_response_time"] = sum(report["response_times"]) / len(report["response_times"])
            report["min_response_time"] = min(report["response_times"])
            report["max_response_time"] = max(report["response_times"])
        else:
            report["avg_response_time"] = 0
            report["min_response_time"] = 0
            report["max_response_time"] = 0
        
        # 统计错误类型
        for result in self.results:
            if "error" in result:
                error_type = str(result["error"])
                if error_type not in report["error_types"]:
                    report["error_types"][error_type] = 0
                report["error_types"][error_type] += 1
        
        # 保存报告
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"重放报告已保存到 {results_file}")
        return report
    
    def generate_charts(self, output_dir: str = "charts") -> List[str]:
        """
        生成可视化图表
        
        Args:
            output_dir: 图表输出目录
        
        Returns:
            图表文件路径列表
        """
        # TODO: 实现图表生成功能
        os.makedirs(output_dir, exist_ok=True)
        chart_files = []
        
        # 如果没有结果，跳过图表生成
        if not self.results:
            return chart_files
        
        # 响应时间分布图
        try:
            response_times = [r.get("duration", 0) for r in self.results]
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=20, alpha=0.7)
            plt.title('Response Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            rt_file = os.path.join(output_dir, 'response_time_distribution.png')
            plt.savefig(rt_file)
            chart_files.append(rt_file)
            plt.close()
            print(f"响应时间分布图已保存到 {rt_file}")
        except Exception as e:
            print(f"生成响应时间分布图失败: {e}")
        
        # 成功/失败饼图
        try:
            success_count = sum(1 for r in self.results if 'error' not in r)
            fail_count = sum(1 for r in self.results if 'error' in r)
            
            plt.figure(figsize=(8, 8))
            plt.pie([success_count, fail_count], labels=['Successful', 'Failed'], autopct='%1.1f%%',
                    startangle=90, colors=['#4CAF50', '#F44336'])
            plt.title('Query Success Rate')
            plt.axis('equal')  # 保证饼图是正圆形
            
            sr_file = os.path.join(output_dir, 'success_rate.png')
            plt.savefig(sr_file)
            chart_files.append(sr_file)
            plt.close()
            print(f"成功率饼图已保存到 {sr_file}")
        except Exception as e:
            print(f"生成成功率饼图失败: {e}")
        
        return chart_files
    
    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳字符串
        
        Returns:
            时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="重放历史查询并评估系统性能")
    parser.add_argument("--replay-dir", type=str, default="replay",
                        help="包含查询JSON文件的目录")
    parser.add_argument("--output-dir", type=str, default="replay_out",
                        help="结果输出目录")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000",
                        help="API服务地址")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="查询之间的延迟（秒）")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="请求超时时间（秒）")
    parser.add_argument("--generate-charts", action="store_true",
                        help="是否生成可视化图表")
    
    args = parser.parse_args()
    
    # 确保replay目录存在
    if not os.path.exists(args.replay_dir):
        print(f"重放目录 {args.replay_dir} 不存在，程序退出")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取replay目录下所有的JSON文件
    json_files = []
    for file in os.listdir(args.replay_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(args.replay_dir, file))
    
    if not json_files:
        print(f"重放目录 {args.replay_dir} 中没有找到JSON文件，程序退出")
        return
    
    print(f"找到 {len(json_files)} 个查询文件，开始批量重放...")
    
    # 处理每个JSON文件
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        print(f"\n处理文件: {file_name}")
        
        # 创建查询重放器
        replayer = QueryReplayer(json_file, args.api_url)
        
        # 加载查询
        query_count = replayer.load_queries()
        if query_count == 0:
            print(f"文件 {file_name} 中没有查询可重放，跳过")
            continue
        
        # 重放查询
        print(f"开始重放 {query_count} 个查询...")
        replayer.replay_queries(args.delay, args.timeout)
        
        # 保存结果
        output_file = os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}_results.json")
        replayer.save_results(output_file)
        
        # 生成报告
        report_file = os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}_report.json")
        replayer.generate_report(report_file)
        
        # 生成图表
        if args.generate_charts:
            charts_dir = os.path.join(args.output_dir, os.path.splitext(file_name)[0], "charts")
            replayer.generate_charts(charts_dir)
    
    print("\n批量查询重放任务已完成！所有结果已保存到 replay_out 目录")

if __name__ == "__main__":
    main()

# TODO: 实现更多重放功能
# 1. 支持批量查询重放
# 2. 支持查询过滤和排序
# 3. 更详细的性能指标计算
# 4. 自动生成PDF格式的报告