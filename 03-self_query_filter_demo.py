#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我查询过滤演示工具

功能说明：
此程序将自然语言查询（特别是与产品、年份、地区相关的查询）解析为结构化的filter JSON，
然后基于内存中的元数据执行过滤操作，并对符合条件的文档进行简易打分，最终返回Top-K结果。

执行流程：
1. 初始化依赖检查与安装
2. 定义内存中的示例元数据（产品、年份、地区等信息）
3. 将输入的自然语言查询解析为结构化的filter字典
4. 基于filter字典执行元数据过滤
5. 对过滤后的文档进行简易打分（基于关键词命中和元数据匹配）
6. 按照分数排序，返回Top-K结果

作者：Ph.D. Rhino
"""

import re
import sys
import json
import subprocess
import importlib.util
import time


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    本程序主要使用Python标准库，但我们仍保留此函数以演示依赖检查功能
    """
    required_packages = []  # 目前此程序仅使用标准库，无需外部依赖
    missing_packages = []
    
    # 检查所需的包是否已安装
    for package in required_packages:
        try:
            # 使用importlib代替废弃的pkg_resources
            importlib.util.find_spec(package)
        except ModuleNotFoundError:
            missing_packages.append(package)
    
    # 安装缺失的包
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            print("依赖包安装成功!")
        except subprocess.CalledProcessError as e:
            print(f"依赖包安装失败: {e}")
            sys.exit(1)
    else:
        print("所有依赖包已安装完成。")


class SelfQueryFilter:
    """自我查询过滤主类，负责解析自然语言并执行过滤与打分"""
    
    def __init__(self):
        # 初始化内存中的示例元数据
        self.documents = self._initialize_documents()
        
        # 定义产品、年份、地区的映射字典，用于自然语言解析
        self.product_mapping = {
            'iphone': ['iphone', '苹果手机', '苹果'],
            'samsung': ['samsung', '三星'],
            'xiaomi': ['xiaomi', '小米'],
            'huawei': ['huawei', '华为'],
            'oppo': ['oppo', '欧珀'],
            'vivo': ['vivo', '维沃']
        }
        
        self.year_pattern = r'(202[0-9]|201[0-9])'  # 匹配2010-2029年
        
        self.region_mapping = {
            'north_america': ['北美', '美国', '加拿大', '墨西哥'],
            'europe': ['欧洲', '欧盟', '英国', '德国', '法国', '意大利', '西班牙'],
            'asia': ['亚洲', '中国', '日本', '韩国', '印度', '新加坡'],
            'oceania': ['大洋洲', '澳大利亚', '新西兰'],
            'south_america': ['南美', '巴西', '阿根廷'],
            'africa': ['非洲', '南非', '埃及']
        }
    
    def _initialize_documents(self):
        """初始化内存中的示例文档元数据"""
        # 模拟一些产品文档及其元数据
        return [
            {"id": "doc_1", "title": "iPhone 15 详细评测", "product": "iphone", "version": "15", "year": 2023, "region": "north_america", "keywords": ["摄像头", "处理器", "电池"]},
            {"id": "doc_2", "title": "三星Galaxy S24 上手体验", "product": "samsung", "version": "S24", "year": 2024, "region": "europe", "keywords": ["屏幕", "性能", "AI功能"]},
            {"id": "doc_3", "title": "小米14 Pro 深度测评", "product": "xiaomi", "version": "14 Pro", "year": 2023, "region": "asia", "keywords": ["充电速度", "摄影系统", "屏幕素质"]},
            {"id": "doc_4", "title": "华为Mate 60 Pro 全面解析", "product": "huawei", "version": "Mate 60 Pro", "year": 2023, "region": "asia", "keywords": ["卫星通信", "鸿蒙系统", "影像" ]},
            {"id": "doc_5", "title": "OPPO Find X7 Ultra 相机评测", "product": "oppo", "version": "Find X7 Ultra", "year": 2024, "region": "asia", "keywords": ["徕卡", "长焦", "暗光拍摄"]},
            {"id": "doc_6", "title": "iPhone 15 Pro Max 专业用户体验", "product": "iphone", "version": "15 Pro Max", "year": 2023, "region": "north_america", "keywords": ["专业摄影", "视频制作", "A17芯片"]},
            {"id": "doc_7", "title": "三星Galaxy Z Fold 5 折叠屏体验", "product": "samsung", "version": "Z Fold 5", "year": 2023, "region": "europe", "keywords": ["折叠屏", "生产力", "多任务"]},
            {"id": "doc_8", "title": "小米Redmi K70 Pro 游戏性能测试", "product": "xiaomi", "version": "Redmi K70 Pro", "year": 2024, "region": "asia", "keywords": ["游戏", "性能", "散热"]},
            {"id": "doc_9", "title": "iPhone 16 曝光信息汇总", "product": "iphone", "version": "16", "year": 2024, "region": "north_america", "keywords": ["新功能", "设计变化", "规格"]},
            {"id": "doc_10", "title": "华为P70系列前瞻报告", "product": "huawei", "version": "P70", "year": 2024, "region": "asia", "keywords": ["相机升级", "设计", "性能"]},
            {"id": "doc_11", "title": "2024年北美手机市场分析", "product": "all", "version": "", "year": 2024, "region": "north_america", "keywords": ["市场份额", "趋势", "竞争"]},
            {"id": "doc_12", "title": "iPhone 15 在亚洲市场的表现", "product": "iphone", "version": "15", "year": 2023, "region": "asia", "keywords": ["销量", "用户反馈", "市场策略"]}
        ]
    
    def parse_natural_language(self, natural_language):
        """
        解析自然语言查询，提取产品、年份和地区信息
        参数: natural_language - 自然语言查询
        返回: dict - 结构化的filter字典
        """
        filter_dict = {}
        
        # 转换为小写以便匹配
        nl_lower = natural_language.lower()
        
        # 提取产品信息
        for product, keywords in self.product_mapping.items():
            if any(keyword in nl_lower for keyword in keywords):
                filter_dict['product'] = product
                break
        
        # 提取产品型号信息
        # 检查iPhone型号 - 改进正则表达式以更准确地提取型号
        iphone_model_match = re.search(r'iphone\s+(\d+(?:\s+\w+)?)', nl_lower)
        if iphone_model_match and filter_dict.get('product') == 'iphone':
            model = iphone_model_match.group(1).strip()
            filter_dict['model'] = f'iPhone {model}'
        
        # 提取年份信息
        year_match = re.search(self.year_pattern, nl_lower)
        if year_match:
            filter_dict['year'] = int(year_match.group(1))
        
        # 提取地区信息
        for region, keywords in self.region_mapping.items():
            if any(keyword in nl_lower for keyword in keywords):
                filter_dict['region'] = region
                break
        
        # 提取关键词
        query_keywords = self._extract_keywords(nl_lower)
        if query_keywords:
            filter_dict['keywords'] = query_keywords
        
        return filter_dict
    
    def _extract_keywords(self, text):
        """从文本中提取关键词"""
        # 移除停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 分割文本为单词 - 使用更精确的分词方式
        words = []
        # 首先尝试提取有意义的中文词语
        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text)
        if chinese_words:
            words.extend(chinese_words)
        
        # 再提取其他单词
        other_words = re.findall(r'[a-zA-Z0-9]+', text)
        words.extend(other_words)
        
        # 过滤停用词和太短的单词
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords[:5]  # 返回前5个关键词
    
    def filter_documents(self, filter_dict):
        """
        基于filter字典过滤文档
        参数: filter_dict - 结构化的过滤条件
        返回: list - 过滤后的文档列表
        """
        filtered_docs = []
        
        for doc in self.documents:
            # 检查产品过滤条件
            if 'product' in filter_dict and doc['product'] != 'all' and doc['product'] != filter_dict['product']:
                continue
            
            # 检查年份过滤条件
            # 如果找不到严格匹配的年份，考虑放宽条件以避免返回空结果
            if 'year' in filter_dict and doc['year'] != filter_dict['year']:
                # 允许年份相差±1年的文档
                if abs(doc['year'] - filter_dict['year']) > 1:
                    continue
            
            # 检查地区过滤条件
            if 'region' in filter_dict and doc['region'] != filter_dict['region']:
                continue
            
            # 检查产品型号匹配
            if 'model' in filter_dict:
                model_text = filter_dict['model'].lower()
                if model_text not in doc['title'].lower():
                    # 如果文档是通用分析报告，也应包含
                    if '市场分析' not in doc['title'] and '分析' not in doc['title']:
                        continue
            
            # 添加到过滤结果
            filtered_docs.append(doc)
        
        return filtered_docs
    
    def score_documents(self, docs, filter_dict):
        """
        对文档进行简易打分
        参数: docs - 文档列表
              filter_dict - 过滤条件字典
        返回: list - 包含(文档ID, 分数)的列表，按分数降序排列
        """
        scored_docs = []
        
        for doc in docs:
            score = 0.0
            
            # 基础分
            score += 0.5
            
            # 关键词匹配加分
            if 'keywords' in filter_dict:
                query_keywords = filter_dict['keywords']
                doc_keywords = doc.get('keywords', [])
                
                # 计算关键词匹配数量
                matched_keywords = set(query_keywords) & set(doc_keywords)
                score += len(matched_keywords) * 0.1
            
            # 产品型号匹配加分
            if 'model' in filter_dict:
                model_text = filter_dict['model'].lower()
                if model_text in doc['title'].lower():
                    score += 0.3
            
            # 确保分数在0-1之间
            score = min(1.0, score)
            
            scored_docs.append((doc['id'], round(score, 2)))
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def get_top_k(self, scored_docs, k=3):
        """
        获取Top-K结果
        参数: scored_docs - 已打分排序的文档列表
              k - 返回的结果数量
        返回: list - Top-K结果
        """
        return scored_docs[:k]
    
    def process_query(self, natural_language, k=3):
        """
        处理完整的查询流程
        参数: natural_language - 自然语言查询
              k - 返回的结果数量
        返回: (filter_dict, top_k_results)
        """
        # 解析自然语言为filter字典
        filter_dict = self.parse_natural_language(natural_language)
        
        # 过滤文档
        filtered_docs = self.filter_documents(filter_dict)
        
        # 打分并排序
        scored_docs = self.score_documents(filtered_docs, filter_dict)
        
        # 获取Top-K结果
        top_k_results = self.get_top_k(scored_docs, k)
        
        return filter_dict, top_k_results


def main():
    """主函数，处理命令行参数并执行查询流程"""
    # 记录开始时间，用于性能分析
    start_time = time.time()
    
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 创建SelfQueryFilter实例
    query_filter = SelfQueryFilter()
    
    # 从命令行参数获取自然语言查询，如果没有则使用示例
    if len(sys.argv) > 1:
        natural_language = ' '.join(sys.argv[1:])
    else:
        # 使用内置示例
        natural_language = "查找2024年关于iPhone 15的亚洲市场分析"
    
    print(f"NL: {natural_language}")
    
    # 处理查询
    filter_dict, top_k_results = query_filter.process_query(natural_language)
    
    # 格式化输出结果
    print(f"filter={json.dumps(filter_dict, ensure_ascii=False)}")
    print(f"results={top_k_results}")
    
    # 计算执行时间
    execution_time = time.time() - start_time
    print(f"执行时间: {execution_time:.4f}秒")
    
    # 检查执行效率
    if execution_time > 1.0:  # 如果执行时间超过1秒，提示优化
        print("警告: 程序执行时间较长，可考虑优化数据结构和算法。")


if __name__ == "__main__":
    main()