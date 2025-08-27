# -*- coding: utf-8 -*-

"""法律文档混合检索系统

功能概述：
该系统基于08-legal-vector-db.py，增加了混合检索方案，实现BM25和向量相似度的双通道召回，
归一化融合打分后去重合并，再用跨编码器重排，输出可引用的Top-N文段。

主要功能：
- 支持从PostgreSQL/pgvector数据库加载向量数据
- 实现BM25文本检索算法
- 支持向量相似度检索
- 双通道召回结果归一化融合
- 结果去重合并
- 跨编码器重排
- 输出可引用的Top-N文段

使用方法：
1. 确保已安装PostgreSQL和pgvector扩展
2. 配置数据库连接参数
3. 确保已有向量化数据存储在数据库中
4. 运行程序，执行混合检索查询

输入：查询文本
输出：Top-N相关法律条款文段
"""

import os
import re
import json
import time
import hashlib
import threading
import psycopg2
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi

# LangChain相关导入
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_community.vectorstores import PGVector
from langchain.schema import Document

# 文档处理相关导入
import docx

# 用于模型下载的导入
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

class LegalHybridRetriever:
    """法律文档混合检索类
    
    提供BM25和向量相似度的双通道召回、归一化融合打分、去重合并和跨编码器重排功能
    """
    
    def __init__(self, db_connection_string: str, collection_name: str = "legal_documents"):
        """初始化法律文档混合检索器
        
        Args:
            db_connection_string: PostgreSQL数据库连接字符串
            collection_name: 向量集合名称
        """
        # 数据库配置
        self.db_connection_string = db_connection_string
        self.collection_name = collection_name
        
        # 模型相关配置
        self.bge_model_name = "BAAI/bge-m3"
        self.cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # 修改本地模型存储路径
        self.local_model_dir = os.path.join(os.path.dirname(os.getcwd()), "11_local_models")
        self.local_bge_model_dir = os.path.join(self.local_model_dir, "bge-m3")
        self.local_cross_encoder_dir = os.path.join(self.local_model_dir, "cross-encoder-ms-marco")
        
        # 检查本地是否有模型文件，如果没有则下载
        self._check_and_download_models()
        
        # 初始化嵌入模型 - 使用本地bge-m3模型
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.local_bge_model_dir,  # 使用本地模型目录
            model_kwargs={"device": "cpu"},  # 可根据实际情况改为"cuda"
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 初始化跨编码器模型
        self.cross_encoder = self._load_cross_encoder()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["。", "；", "！", "？", "\n", "\r", " "]
        )
        
        # 向量数据库实例
        self.vector_store = None
        
        # BM25检索器实例
        self.bm25_retriever = None
        
        # 存储所有文档的文本和元数据，用于BM25检索
        self.all_documents = []
        self.all_document_texts = []
        self.all_document_metadata = []
        
        # 初始化数据库连接
        self._init_vector_store()
        
        # 加载所有文档数据用于BM25检索
        self._load_documents_for_bm25()
        
        # 初始化BM25检索器
        self._init_bm25_retriever()
    
    def _check_and_download_models(self):
        """检查本地是否有模型文件，如果没有则下载"""
        # 确保本地模型目录存在
        os.makedirs(self.local_model_dir, exist_ok=True)
        
        # 检查并下载bge-m3模型
        self._check_and_download_bge_model()
        
        # 检查并下载cross-encoder模型
        self._check_and_download_cross_encoder_model()
    
    def _check_and_download_bge_model(self):
        """检查并下载bge-m3模型"""
        # 检查本地模型目录是否存在且包含必要文件
        required_files = ["config.json", "tokenizer.json", "model.safetensors"]
        model_exists = True
        
        if not os.path.exists(self.local_bge_model_dir):
            model_exists = False
        else:
            # 检查是否包含必要的模型文件
            for file in required_files:
                if not os.path.exists(os.path.join(self.local_bge_model_dir, file)):
                    model_exists = False
                    break
        
        if not model_exists:
            print(f"本地未找到完整的bge-m3模型，正在从Hugging Face下载到 {self.local_bge_model_dir}...")
            try:
                # 下载模型和分词器
                tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
                model = AutoModel.from_pretrained(self.bge_model_name)
                
                # 保存到本地目录
                os.makedirs(self.local_bge_model_dir, exist_ok=True)
                tokenizer.save_pretrained(self.local_bge_model_dir)
                model.save_pretrained(self.local_bge_model_dir)
                
                print("bge-m3模型下载完成！")
            except Exception as e:
                print(f"bge-m3模型下载失败: {str(e)}")
                print("将尝试使用HuggingFaceBgeEmbeddings的默认加载方式")
        else:
            print(f"找到本地bge-m3模型，将使用本地模型进行向量化")
    
    def _check_and_download_cross_encoder_model(self):
        """检查并下载cross-encoder模型"""
        # 检查本地模型目录是否存在且包含必要文件
        required_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
        model_exists = True
        
        if not os.path.exists(self.local_cross_encoder_dir):
            model_exists = False
        else:
            # 检查是否包含必要的模型文件
            for file in required_files:
                if not os.path.exists(os.path.join(self.local_cross_encoder_dir, file)):
                    model_exists = False
                    break
        
        if not model_exists:
            print(f"本地未找到完整的cross-encoder模型，正在从Hugging Face下载到 {self.local_cross_encoder_dir}...")
            try:
                # 下载模型和分词器
                tokenizer = AutoTokenizer.from_pretrained(self.cross_encoder_model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.cross_encoder_model_name)
                
                # 保存到本地目录
                os.makedirs(self.local_cross_encoder_dir, exist_ok=True)
                tokenizer.save_pretrained(self.local_cross_encoder_dir)
                model.save_pretrained(self.local_cross_encoder_dir)
                
                print("cross-encoder模型下载完成！")
            except Exception as e:
                print(f"cross-encoder模型下载失败: {str(e)}")
                print("将尝试使用默认加载方式")
        else:
            print(f"找到本地cross-encoder模型，将使用本地模型进行重排")
    
    def _load_cross_encoder(self):
        """加载跨编码器模型"""
        try:
            # 尝试从本地加载模型
            tokenizer = AutoTokenizer.from_pretrained(self.local_cross_encoder_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.local_cross_encoder_dir)
            
            # 将模型移至适当的设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            return (tokenizer, model, device)
        except Exception as e:
            print(f"从本地加载cross-encoder模型失败: {str(e)}")
            try:
                # 尝试从Hugging Face直接加载
                print("尝试直接从Hugging Face加载cross-encoder模型...")
                tokenizer = AutoTokenizer.from_pretrained(self.cross_encoder_model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.cross_encoder_model_name)
                
                # 将模型移至适当的设备
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                
                return (tokenizer, model, device)
            except Exception as e2:
                print(f"加载cross-encoder模型失败: {str(e2)}")
                print("重排功能将不可用，将使用混合分数直接排序")
                return None
    
    def _init_vector_store(self):
        """初始化向量数据库连接"""
        try:
            print("正在连接PostgreSQL/pgvector数据库...")
            self.vector_store = PGVector(
                connection_string=self.db_connection_string,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            print("数据库连接成功！")
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            print("请确保已安装PostgreSQL和pgvector扩展，并正确配置连接字符串")
            # 即使连接失败也继续执行，后面会检查vector_store是否为None
            self.vector_store = None
    
    def _load_documents_for_bm25(self):
        """加载所有文档数据用于BM25检索"""
        if not self.vector_store:
            print("向量数据库未初始化，无法加载文档数据")
            return
        
        try:
            print("正在加载文档数据用于BM25检索...")
            
            # 使用psycopg2直接查询数据库获取所有文档
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            # 查询指定集合中的所有文档
            cursor.execute("""SELECT c.page_content, c.metadata FROM langchain_pg_embedding e
                            JOIN langchain_pg_collection c ON e.collection_id = c.id
                            WHERE c.name = %s""", (self.collection_name,))
            
            # 解析结果
            for row in cursor.fetchall():
                page_content = row[0]
                metadata = row[1]
                
                # 将元数据字符串转换为字典
                if isinstance(metadata, str):
                    try:
                        metadata_dict = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata_dict = {}
                else:
                    metadata_dict = metadata
                
                self.all_documents.append(Document(page_content=page_content, metadata=metadata_dict))
                self.all_document_texts.append(page_content)
                self.all_document_metadata.append(metadata_dict)
            
            cursor.close()
            conn.close()
            
            print(f"成功加载 {len(self.all_documents)} 个文档用于BM25检索")
        except Exception as e:
            print(f"加载文档数据时发生错误: {str(e)}")
    
    def _init_bm25_retriever(self):
        """初始化BM25检索器"""
        if not self.all_document_texts:
            print("没有文档数据，无法初始化BM25检索器")
            return
        
        try:
            print("正在初始化BM25检索器...")
            
            # 简单的中文分词函数（实际应用中可以使用更复杂的分词库如jieba）
            def tokenize_chinese(text):
                # 基本的中文分词，保留标点符号作为分隔符
                tokens = []
                for char in text:
                    # 使用正确的中文检测正则表达式
                    if re.match(r'[一-龥]', char):  # 中文字符
                        tokens.append(char)
                    elif re.match(r'[a-zA-Z0-9]', char):  # 英文或数字
                        # 尝试组合连续的英文或数字
                        if tokens and re.match(r'[a-zA-Z0-9]', tokens[-1]):
                            tokens[-1] += char
                        else:
                            tokens.append(char)
                    else:
                        # 其他字符作为单独的token
                        tokens.append(char)
                return tokens
            
            # 对所有文档进行分词
            tokenized_corpus = [tokenize_chinese(doc) for doc in self.all_document_texts]
            
            # 初始化BM25检索器
            self.bm25_retriever = BM25Okapi(tokenized_corpus)
            
            print("BM25检索器初始化完成！")
        except Exception as e:
            print(f"初始化BM25检索器时发生错误: {str(e)}")
            self.bm25_retriever = None
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """对分数进行归一化处理
        
        Args:
            scores: 原始分数列表
            
        Returns:
            List[float]: 归一化后的分数列表
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # 处理除以零的情况
        if max_score == min_score:
            return [0.5] * len(scores)
        
        # 归一化到0-1范围
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        return normalized_scores
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """对检索结果进行去重
        
        Args:
            results: 检索结果列表
            
        Returns:
            List[Dict]: 去重后的结果列表
        """
        if not results:
            return []
        
        # 使用集合记录已出现的文本哈希值
        seen_hashes = set()
        deduplicated_results = []
        
        for result in results:
            # 为文本生成哈希值用于去重
            text_hash = hashlib.sha256(result['text'].encode('utf-8')).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduplicated_results.append(result)
        
        return deduplicated_results
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """使用跨编码器对检索结果进行重排
        
        Args:
            query: 查询文本
            results: 原始检索结果列表
            top_k: 重排后返回的结果数量
            
        Returns:
            List[Dict]: 重排后的结果列表
        """
        if not results or len(results) <= 1:
            return results
        
        # 检查跨编码器是否可用
        if self.cross_encoder is None:
            print("跨编码器不可用，将使用原始混合分数排序")
            # 按混合分数排序
            return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:top_k]
        
        try:
            print("正在使用跨编码器进行结果重排...")
            
            tokenizer, model, device = self.cross_encoder
            
            # 准备用于重排的查询-文本对
            query_text_pairs = [(query, result['text']) for result in results]
            
            # 使用跨编码器计算相关性分数
            with torch.no_grad():
                # 对文本对进行编码
                inputs = tokenizer(
                    query_text_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(device)
                
                # 获取模型输出
                outputs = model(**inputs)
                
                # 提取分数（对于二分类模型，使用logits[0]）
                scores = outputs.logits.squeeze().tolist()
            
            # 更新结果中的重排分数
            for i, result in enumerate(results):
                result['rerank_score'] = scores[i]
            
            # 按重排分数排序并返回top_k结果
            reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:top_k]
            
            print("结果重排完成！")
            return reranked_results
        except Exception as e:
            print(f"使用跨编码器重排时发生错误: {str(e)}")
            print("将使用原始混合分数排序")
            # 按混合分数排序
            return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:top_k]
    
    def bm25_search(self, query: str, k: int = 20) -> List[Dict]:
        """执行BM25搜索
        
        Args:
            query: 搜索查询文本
            k: 返回的最相似结果数量
            
        Returns:
            List[Dict]: BM25搜索结果列表
        """
        if not self.bm25_retriever:
            print("BM25检索器未初始化，无法执行搜索")
            return []
        
        try:
            print(f"正在执行BM25搜索: {query}")
            
            # 对查询进行分词
            def tokenize_chinese(text):
                tokens = []
                for char in text:
                    if re.match(r'[一-龥]', char):  # 中文字符
                        tokens.append(char)
                    elif re.match(r'[a-zA-Z0-9]', char):  # 英文或数字
                        # 尝试组合连续的英文或数字
                        if tokens and re.match(r'[a-zA-Z0-9]', tokens[-1]):
                            tokens[-1] += char
                        else:
                            tokens.append(char)
                    else:
                        # 其他字符作为单独的token
                        tokens.append(char)
                return tokens
            
            tokenized_query = tokenize_chinese(query)
            
            # 执行BM25搜索
            bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
            
            # 获取top-k结果的索引
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            
            # 构建搜索结果
            results = []
            for idx in top_indices:
                doc = self.all_documents[idx]
                result = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'bm25_score': float(bm25_scores[idx])
                }
                results.append(result)
            
            return results
        except Exception as e:
            print(f"执行BM25搜索时发生错误: {str(e)}")
            return []
    
    def vector_search(self, query: str, k: int = 20) -> List[Dict]:
        """执行向量相似度搜索
        
        Args:
            query: 搜索查询文本
            k: 返回的最相似结果数量
            
        Returns:
            List[Dict]: 向量搜索结果列表
        """
        if not self.vector_store:
            print("向量数据库未初始化，无法执行搜索")
            return []
        
        try:
            print(f"正在执行向量相似度搜索: {query}")
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # 格式化搜索结果
            formatted_results = []
            for doc, score in results:
                result = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'vector_score': 1 - float(score)  # 转换为相似度得分（越高越相似）
                }
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"执行向量搜索时发生错误: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = 20, vector_weight: float = 0.5, rerank_k: int = 10) -> List[Dict]:
        """执行混合检索（BM25 + 向量相似度）
        
        Args:
            query: 搜索查询文本
            k: 每个检索通道返回的结果数量
            vector_weight: 向量相似度分数的权重（0-1之间）
            rerank_k: 重排后返回的最终结果数量
            
        Returns:
            List[Dict]: 混合检索结果列表
        """
        print(f"正在执行混合检索: {query}")
        
        # 并行执行BM25和向量搜索
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_search, query, k)
            vector_future = executor.submit(self.vector_search, query, k)
            
            # 获取搜索结果
            bm25_results = bm25_future.result()
            vector_results = vector_future.result()
        
        # 归一化分数
        if bm25_results:
            bm25_scores = [result['bm25_score'] for result in bm25_results]
            normalized_bm25_scores = self._normalize_scores(bm25_scores)
            for i, result in enumerate(bm25_results):
                result['normalized_bm25_score'] = normalized_bm25_scores[i]
        
        if vector_results:
            vector_scores = [result['vector_score'] for result in vector_results]
            normalized_vector_scores = self._normalize_scores(vector_scores)
            for i, result in enumerate(vector_results):
                result['normalized_vector_score'] = normalized_vector_scores[i]
        
        # 合并结果并去重
        all_results = bm25_results + vector_results
        deduplicated_results = self._deduplicate_results(all_results)
        
        # 计算混合分数
        for result in deduplicated_results:
            # 对于只在一个通道中出现的结果，使用该通道的归一化分数
            if 'normalized_bm25_score' in result and 'normalized_vector_score' in result:
                # 对于在两个通道中都出现的结果，计算加权平均分数
                result['hybrid_score'] = (result['normalized_bm25_score'] * (1 - vector_weight) + 
                                          result['normalized_vector_score'] * vector_weight)
            elif 'normalized_bm25_score' in result:
                result['hybrid_score'] = result['normalized_bm25_score']
            elif 'normalized_vector_score' in result:
                result['hybrid_score'] = result['normalized_vector_score']
            else:
                result['hybrid_score'] = 0
        
        # 使用跨编码器进行重排
        final_results = self._cross_encoder_rerank(query, deduplicated_results, top_k=rerank_k)
        
        # 格式化最终结果，添加引用信息
        for i, result in enumerate(final_results, 1):
            # 为结果添加排名
            result['rank'] = i
            
            # 提取引用信息
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知来源')
            article_id = metadata.get('article_id', '未知条款ID')
            
            # 构建引用字符串
            result['citation'] = f"来源: {os.path.basename(source)}, 条款ID: {article_id}"
            
            # 确保文本不超过一定长度
            if len(result['text']) > 300:
                result['text_preview'] = result['text'][:300] + "..."
            else:
                result['text_preview'] = result['text']
        
        print(f"混合检索完成，返回 {len(final_results)} 个结果")
        return final_results
    
    def count_documents(self) -> int:
        """获取数据库中的文档数量"""
        if not self.vector_store:
            print("向量数据库未初始化")
            return 0
        
        try:
            # 使用psycopg2直接查询数据库获取文档数量
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            # 查询指定集合中的文档数量
            cursor.execute("""SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = (
                SELECT id FROM langchain_pg_collection WHERE name = %s
            )""", (self.collection_name,))
            
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count
        except Exception as e:
            print(f"获取文档数量时发生错误: {str(e)}")
            return 0


def main():
    """主函数"""
    # 配置数据库连接参数
    DB_PARAMS = {
        "host": "localhost",
        "port": 5432,
        "dbname": "legal_database",
        "user": "postgres",
        "password": "your_password"  # 请替换为实际密码
    }
    
    # 构建数据库连接字符串
    db_connection_string = f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    
    # 创建法律文档混合检索器实例
    try:
        hybrid_retriever = LegalHybridRetriever(db_connection_string)
        
        # 统计数据库中的文档数量
        doc_count = hybrid_retriever.count_documents()
        print(f"向量数据库中共有 {doc_count} 个文档块")
        
        # 示例：执行混合检索
        while True:
            query = input("\n请输入搜索查询（输入'q'退出）: ")
            if query.lower() == 'q':
                break
            
            # 执行混合检索
            start_time = time.time()
            results = hybrid_retriever.hybrid_search(
                query=query, 
                k=20,  # 每个通道返回的结果数量
                vector_weight=0.6,  # 向量相似度的权重
                rerank_k=5  # 重排后返回的最终结果数量
            )
            end_time = time.time()
            
            # 显示检索结果
            print(f"\n混合检索结果 ({len(results)}):")
            print(f"检索耗时: {end_time - start_time:.2f} 秒")
            
            for result in results:
                print(f"\n排名 {result['rank']}:")
                print(f"混合分数: {result.get('hybrid_score', 0):.4f}")
                if 'rerank_score' in result:
                    print(f"重排分数: {result['rerank_score']:.4f}")
                print(f"文本预览: {result['text_preview']}")
                print(f"引用: {result['citation']}")
                print("=" * 50)
            
    except Exception as e:
        print(f"程序运行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()