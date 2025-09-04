#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG数据处理最小闭环流水线示例

本脚本实现了RAG（检索增强生成）数据处理的完整最小闭环流水线，包括：
1. 数据采集：从模拟数据源获取原始文档
2. 数据清洗：处理和标准化采集到的文档
3. 数据索引：创建向量索引以支持高效检索
4. 数据检索：基于用户查询从索引中检索相关文档
5. 结果评测：评估检索结果的质量和相关性
6. 告警机制：当评测结果不满足预设阈值时触发告警

这个最小示例展示了RAG系统的核心数据处理流程，为实际应用提供了基础框架。
"""
import os
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MinimalRAGPipeline:
    """
    RAG数据处理最小闭环流水线类
    """
    
    def __init__(self):
        """初始化RAG流水线"""
        # 设置随机种子以确保结果可复现
        random.seed(42)
        np.random.seed(42)
        
        # 配置参数
        self.config = {
            'collection_size': 100,  # 采集的文档数量
            'embedding_dim': 768,    # 嵌入向量维度
            'top_k': 5,              # 检索时返回的文档数量
            'relevance_threshold': 0.7,  # 相关性阈值
            'alert_threshold': 0.5    # 告警阈值
        }
        
        # 初始化数据存储
        self.raw_documents = []
        self.clean_documents = []
        self.vector_index = []
        self.retrieved_results = []
        self.evaluation_results = []
        
        print("RAG最小闭环流水线初始化完成")
        print(f"配置参数: {json.dumps(self.config, ensure_ascii=False, indent=2)}")
    
    def collect_data(self):
        """
        数据采集阶段：从模拟数据源获取原始文档
        """
        print("\n=== 数据采集阶段开始 ===")
        
        # 模拟从不同来源采集医疗相关文档
        sources = ['medical_journal', 'clinical_trials', 'patient_records', 'research_papers']
        topics = ['心脏病', '糖尿病', '癌症', '高血压', '肺炎', '阿尔茨海默病', '帕金森病', '哮喘']
        
        for i in range(self.config['collection_size']):
            # 随机生成文档内容
            doc = {
                'doc_id': f'doc_{i:03d}',
                'source': random.choice(sources),
                'topic': random.choice(topics),
                'title': f'{random.choice(topics)}的最新研究进展_{i}',
                'content': self._generate_content(random.choice(topics)),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'source_type': random.choice(sources),
                    'collection_date': datetime.now().date().isoformat(),
                    'priority': random.randint(1, 5)
                }
            }
            
            self.raw_documents.append(doc)
            
            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"已采集 {i + 1}/{self.config['collection_size']} 个文档")
        
        print(f"数据采集完成，共采集 {len(self.raw_documents)} 个文档")
        
        # 保存原始数据
        with open(os.path.join(OUTPUT_DIR, 'raw_documents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.raw_documents, f, ensure_ascii=False, indent=2)
        
        return self.raw_documents
    
    def _generate_content(self, topic):
        """
        生成模拟文档内容
        
        Args:
            topic: 文档主题
        
        Returns:
            str: 生成的文档内容
        """
        templates = {
            '心脏病': "心脏病是一种常见的慢性疾病，近年来发病率呈上升趋势。最新研究表明，合理的饮食和规律的运动可以有效降低心脏病的发病风险。此外，早期筛查和干预对于提高患者生存率至关重要。",
            '糖尿病': "糖尿病是一种代谢性疾病，主要分为1型和2型。据统计，全球有超过4亿糖尿病患者。控制血糖水平、定期监测并发症是糖尿病管理的关键环节。",
            '癌症': "癌症是全球范围内的主要死亡原因之一。近年来，免疫治疗和靶向治疗取得了突破性进展，为癌症患者带来了新的希望。早期诊断和综合治疗是提高癌症治愈率的关键。",
            '高血压': "高血压是心脑血管疾病的重要危险因素。保持健康的生活方式，如低盐饮食、适量运动、戒烟限酒等，可以有效控制血压。对于需要药物治疗的患者，应严格遵医嘱服药。",
            '肺炎': "肺炎是一种常见的肺部感染性疾病，可由细菌、病毒或真菌引起。特别是在流感季节，肺炎的发病率明显增加。接种疫苗是预防肺炎的有效手段之一。",
            '阿尔茨海默病': "阿尔茨海默病是一种进行性神经退行性疾病，主要影响记忆力和认知功能。目前尚无治愈方法，但早期干预和支持治疗可以延缓病情进展，提高患者生活质量。",
            '帕金森病': "帕金森病是一种常见的神经系统变性疾病，主要表现为震颤、肌肉僵硬和运动迟缓。药物治疗和康复训练是控制症状的主要方法，近年来深部脑刺激等手术治疗也取得了一定进展。",
            '哮喘': "哮喘是一种慢性气道炎症性疾病，主要表现为反复发作的喘息、气急、胸闷或咳嗽等症状。避免接触过敏原、规范使用吸入性药物是哮喘管理的核心。"
        }
        
        # 基础内容
        base_content = templates.get(topic, "这是一篇关于{topic}的医学文档。")
        
        # 添加一些随机变化以增加内容多样性
        variations = [
            "近年来，随着医学研究的深入，我们对该疾病的认识有了显著提高。",
            "临床实践表明，综合治疗方案比单一治疗方法更有效。",
            "最新发表在《医学期刊》上的研究提供了新的治疗思路。",
            "然而，目前仍有许多未解之谜需要进一步研究。",
            "患者教育和自我管理在疾病控制中发挥着重要作用。"
        ]
        
        # 随机选择2-3个变化添加到基础内容
        num_variations = random.randint(2, 3)
        selected_variations = random.sample(variations, num_variations)
        
        return base_content + " " + " ".join(selected_variations)
    
    def clean_data(self):
        """
        数据清洗阶段：处理和标准化采集到的文档
        """
        print("\n=== 数据清洗阶段开始 ===")
        
        if not self.raw_documents:
            print("没有可清洗的数据，请先执行数据采集阶段")
            return []
        
        for i, doc in enumerate(self.raw_documents):
            # 创建清洗后的文档
            clean_doc = {
                'doc_id': doc['doc_id'],
                'source': doc['source'],
                'topic': doc['topic'],
                'title': doc['title'].strip(),  # 去除首尾空格
                'content': self._clean_text(doc['content']),  # 清洗文本内容
                'timestamp': doc['timestamp'],
                'metadata': doc['metadata'],
                'cleaned_at': datetime.now().isoformat()
            }
            
            self.clean_documents.append(clean_doc)
            
            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"已清洗 {i + 1}/{len(self.raw_documents)} 个文档")
        
        print(f"数据清洗完成，共清洗 {len(self.clean_documents)} 个文档")
        
        # 保存清洗后的数据
        with open(os.path.join(OUTPUT_DIR, 'clean_documents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.clean_documents, f, ensure_ascii=False, indent=2)
        
        return self.clean_documents
    
    def _clean_text(self, text):
        """
        文本清洗函数
        
        Args:
            text: 原始文本
        
        Returns:
            str: 清洗后的文本
        """
        # 去除多余空格
        text = ' '.join(text.split())
        # 转换为小写（在实际应用中可能需要保留大小写）
        # text = text.lower()
        # 去除特殊字符（保留中文、英文、数字和基本标点）
        # import re
        # text = re.sub(r'[^一-龥a-zA-Z0-9，。,.!?；;]', ' ', text)
        
        return text
    
    def create_index(self):
        """
        数据索引阶段：创建向量索引以支持高效检索
        """
        print("\n=== 数据索引阶段开始 ===")
        
        if not self.clean_documents:
            print("没有可索引的数据，请先执行数据清洗阶段")
            return []
        
        # 模拟向量嵌入（在实际应用中应使用真实的嵌入模型）
        for i, doc in enumerate(self.clean_documents):
            # 为每个文档生成随机向量（模拟嵌入结果）
            vector = self._generate_embedding(doc['content'])
            
            # 构建索引项
            index_item = {
                'doc_id': doc['doc_id'],
                'vector': vector.tolist(),
                'metadata': {
                    'topic': doc['topic'],
                    'source': doc['source']
                }
            }
            
            self.vector_index.append(index_item)
            
            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"已索引 {i + 1}/{len(self.clean_documents)} 个文档")
        
        print(f"数据索引完成，共创建 {len(self.vector_index)} 个索引项")
        
        # 保存向量索引
        with open(os.path.join(OUTPUT_DIR, 'vector_index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vector_index, f, ensure_ascii=False, indent=2)
        
        return self.vector_index
    
    def _generate_embedding(self, text):
        """
        生成文档的向量表示（模拟）
        
        Args:
            text: 文档内容
        
        Returns:
            np.ndarray: 文档的向量表示
        """
        # 模拟向量生成 - 在实际应用中应使用真实的嵌入模型
        # 这里我们基于文本长度和字符分布生成伪随机向量
        text_len = len(text)
        hash_val = hash(text) % (10 ** 9)
        
        # 设置随机种子以确保相同文本生成相同向量
        np.random.seed(hash_val)
        
        # 生成随机向量并归一化
        vector = np.random.rand(self.config['embedding_dim'])
        vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def retrieve_documents(self, queries=None):
        """
        数据检索阶段：基于用户查询从索引中检索相关文档
        
        Args:
            queries: 用户查询列表，如果为None则使用默认查询
        
        Returns:
            list: 检索结果
        """
        print("\n=== 数据检索阶段开始 ===")
        
        if not self.vector_index:
            print("没有可检索的索引，请先执行数据索引阶段")
            return []
        
        # 如果没有提供查询，使用默认查询
        if queries is None:
            queries = [
                "心脏病的最新治疗方法有哪些？",
                "如何有效预防糖尿病？",
                "癌症免疫治疗的进展如何？",
                "高血压患者的饮食注意事项",
                "肺炎疫苗的接种指南"
            ]
        
        for query in queries:
            print(f"\n查询: {query}")
            
            # 模拟查询向量生成
            query_vector = self._generate_embedding(query)
            
            # 计算相似度并排序
            similarities = []
            for item in self.vector_index:
                doc_vector = np.array(item['vector'])
                # 计算余弦相似度
                similarity = np.dot(query_vector, doc_vector)
                similarities.append((item['doc_id'], similarity))
            
            # 按相似度降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 取top-k个结果
            top_results = similarities[:self.config['top_k']]
            
            # 查找对应的文档内容
            retrieved_docs = []
            for doc_id, score in top_results:
                # 找到对应的清洗后文档
                doc = next((d for d in self.clean_documents if d['doc_id'] == doc_id), None)
                if doc:
                    retrieved_docs.append({
                        'doc_id': doc_id,
                        'score': float(score),
                        'topic': doc['topic'],
                        'title': doc['title'],
                        'content': doc['content'][:100] + "..."  # 只保留前100个字符
                    })
            
            # 存储检索结果
            self.retrieved_results.append({
                'query': query,
                'results': retrieved_docs,
                'timestamp': datetime.now().isoformat()
            })
            
            # 打印检索结果
            print(f"找到 {len(retrieved_docs)} 个相关文档:")
            for i, result in enumerate(retrieved_docs):
                print(f"  {i+1}. [{result['score']:.4f}] {result['title']}")
        
        print(f"\n数据检索完成，处理了 {len(queries)} 个查询")
        
        # 保存检索结果
        with open(os.path.join(OUTPUT_DIR, 'retrieval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self.retrieved_results, f, ensure_ascii=False, indent=2)
        
        return self.retrieved_results
    
    def evaluate_results(self):
        """
        结果评测阶段：评估检索结果的质量和相关性
        """
        print("\n=== 结果评测阶段开始 ===")
        
        if not self.retrieved_results:
            print("没有可评测的检索结果，请先执行数据检索阶段")
            return []
        
        for i, retrieval in enumerate(self.retrieved_results):
            query = retrieval['query']
            results = retrieval['results']
            
            # 计算平均相关性分数
            if results:
                avg_score = sum(r['score'] for r in results) / len(results)
            else:
                avg_score = 0.0
            
            # 计算命中率（满足相关性阈值的文档比例）
            relevant_docs = [r for r in results if r['score'] >= self.config['relevance_threshold']]
            hit_rate = len(relevant_docs) / len(results) if results else 0.0
            
            # 存储评测结果
            evaluation = {
                'query_id': f'query_{i:03d}',
                'query': query,
                'avg_relevance_score': float(avg_score),
                'hit_rate': float(hit_rate),
                'num_results': len(results),
                'num_relevant_results': len(relevant_docs),
                'relevance_threshold': self.config['relevance_threshold'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.evaluation_results.append(evaluation)
            
            # 打印评测结果
            print(f"\n查询 {i+1}: {query}")
            print(f"  平均相关性分数: {avg_score:.4f}")
            print(f"  命中率: {hit_rate:.4f} ({len(relevant_docs)}/{len(results)})")
        
        # 计算总体评测指标
        if self.evaluation_results:
            overall_avg_score = sum(e['avg_relevance_score'] for e in self.evaluation_results) / len(self.evaluation_results)
            overall_hit_rate = sum(e['hit_rate'] for e in self.evaluation_results) / len(self.evaluation_results)
        else:
            overall_avg_score = 0.0
            overall_hit_rate = 0.0
        
        print(f"\n总体评测结果:")
        print(f"  平均相关性分数: {overall_avg_score:.4f}")
        print(f"  平均命中率: {overall_hit_rate:.4f}")
        
        # 保存评测结果
        with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'individual_evaluations': self.evaluation_results,
                'overall_metrics': {
                    'avg_relevance_score': float(overall_avg_score),
                    'avg_hit_rate': float(overall_hit_rate),
                    'total_queries': len(self.evaluation_results),
                    'timestamp': datetime.now().isoformat()
                }
            }, f, ensure_ascii=False, indent=2)
        
        return self.evaluation_results
    
    def check_alerts(self):
        """
        告警机制：当评测结果不满足预设阈值时触发告警
        """
        print("\n=== 告警检查阶段开始 ===")
        
        if not self.evaluation_results:
            print("没有可检查的评测结果，请先执行结果评测阶段")
            return []
        
        alerts = []
        
        # 检查每个查询的评测结果
        for evaluation in self.evaluation_results:
            query = evaluation['query']
            avg_score = evaluation['avg_relevance_score']
            hit_rate = evaluation['hit_rate']
            
            # 检查是否触发告警
            alert_reasons = []
            if avg_score < self.config['alert_threshold']:
                alert_reasons.append(f"平均相关性分数低于阈值 ({avg_score:.4f} < {self.config['alert_threshold']})")
            
            if hit_rate < self.config['alert_threshold']:
                alert_reasons.append(f"命中率低于阈值 ({hit_rate:.4f} < {self.config['alert_threshold']})")
            
            # 如果有告警原因，创建告警
            if alert_reasons:
                alert = {
                    'alert_id': f'alert_{int(time.time())}_{random.randint(1000, 9999)}',
                    'query': query,
                    'reasons': alert_reasons,
                    'evaluation_metrics': {
                        'avg_relevance_score': avg_score,
                        'hit_rate': hit_rate
                    },
                    'timestamp': datetime.now().isoformat(),
                    'status': 'unresolved'
                }
                
                alerts.append(alert)
                print(f"\n触发告警！")
                print(f"查询: {query}")
                for reason in alert_reasons:
                    print(f"  - {reason}")
        
        # 如果没有告警，打印成功信息
        if not alerts:
            print("恭喜！所有评测指标均满足要求，未触发任何告警。")
        else:
            print(f"\n告警检查完成，共触发 {len(alerts)} 个告警。")
        
        # 保存告警信息
        with open(os.path.join(OUTPUT_DIR, 'alerts.json'), 'w', encoding='utf-8') as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
        
        return alerts
    
    def run_full_pipeline(self, queries=None):
        """
        运行完整的RAG流水线
        
        Args:
            queries: 用户查询列表，如果为None则使用默认查询
        
        Returns:
            dict: 包含所有阶段结果的字典
        """
        print("\n=== 开始运行完整RAG流水线 ===")
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 数据采集
        self.collect_data()
        
        # 2. 数据清洗
        self.clean_data()
        
        # 3. 数据索引
        self.create_index()
        
        # 4. 数据检索
        self.retrieve_documents(queries)
        
        # 5. 结果评测
        self.evaluate_results()
        
        # 6. 告警检查
        alerts = self.check_alerts()
        
        # 记录结束时间
        end_time = time.time()
        
        print(f"\n=== RAG流水线运行完成 ===")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"生成的所有输出文件已保存至: {OUTPUT_DIR}")
        
        # 返回所有阶段的结果
        return {
            'raw_documents': self.raw_documents,
            'clean_documents': self.clean_documents,
            'vector_index': self.vector_index,
            'retrieval_results': self.retrieved_results,
            'evaluation_results': self.evaluation_results,
            'alerts': alerts,
            'execution_time': end_time - start_time
        }

if __name__ == "__main__":
    """主程序入口，运行RAG最小闭环流水线"""
    
    # 创建RAG流水线实例
    rag_pipeline = MinimalRAGPipeline()
    
    # 运行完整流水线
    pipeline_results = rag_pipeline.run_full_pipeline()
    
    print("\nRAG最小闭环流水线示例演示完成！")