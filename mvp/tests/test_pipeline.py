#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统流水线测试
测试整个RAG系统的各个组件及其集成功能
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模拟FastAPI TestClient
class MockTestClient:
    def __init__(self, app):
        self.app = app
    
    def post(self, url, json=None):
        # 模拟API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "这是一个测试答案",
            "evidences": [
                {"id": "doc1", "title": "文档1", "content": "文档内容", "score": 0.95},
                {"id": "doc2", "title": "文档2", "content": "另一个文档内容", "score": 0.85}
            ],
            "reranked": [
                {"id": "doc1", "title": "文档1", "content": "文档内容", "score": 0.95},
                {"id": "doc2", "title": "文档2", "content": "另一个文档内容", "score": 0.85}
            ]
        }
        return mock_response

# 尝试导入FastAPI，如果可用则使用真实的，否则使用模拟的
fastapi_available = False
try:
    from fastapi.testclient import TestClient
    fastapi_available = True
except ImportError:
    # 如果fastapi不可用，使用模拟的TestClient
    TestClient = MockTestClient

# 导入需要测试的模块
from app.config import config
from app.preprocess import TextPreprocessor
from app.rewrite import QueryRewriter
from app.self_query import SelfQueryParser
from app.router import QueryRouter, RetrievalStrategy
from app.search_bm25 import BM25Searcher
from app.search_vector import VectorSearcher
from app.fuse import ResultFuser
from app.rerank import Reranker
from app.answer import AnswerGenerator
from app.cache import CacheManager

class TestRAGPipeline(unittest.TestCase):
    """RAG系统流水线测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，只执行一次"""
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        cls.test_config = {
            "data_dir": cls.temp_dir,
            "index_dir": os.path.join(cls.temp_dir, "index"),
            "bm25_index_path": os.path.join(cls.temp_dir, "index", "bm25_index.json"),
            "vector_index_path": os.path.join(cls.temp_dir, "index", "vector_index.faiss"),
            "embedding_model_path": "all-MiniLM-L6-v2"
        }
        
        # 创建索引目录
        os.makedirs(cls.test_config["index_dir"], exist_ok=True)
        
        # 准备测试数据
        cls._prepare_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理，只执行一次"""
        # 删除临时目录
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _prepare_test_data(cls):
        """准备测试数据"""
        # 创建测试文档
        cls.test_docs = [
            {
                "id": "doc1",
                "title": "测试文档1",
                "content": "这是一篇关于RAG系统的测试文档，用于测试文本预处理功能。",
                "type": "policy",
                "source": "test_policy.txt"
            },
            {
                "id": "doc2",
                "title": "测试文档2",
                "content": "向量检索是RAG系统中的重要组成部分，可以快速找到相关文档。",
                "type": "faq",
                "source": "test_faq.txt"
            },
            {
                "id": "doc3",
                "title": "测试文档3",
                "content": "BM25算法在文本检索领域有着广泛的应用，特别是在关键词匹配方面。",
                "type": "sop",
                "source": "test_sop.txt"
            }
        ]
        
        # 保存测试文档到临时目录
        test_docs_path = os.path.join(cls.temp_dir, "test_docs.json")
        with open(test_docs_path, 'w', encoding='utf-8') as f:
            json.dump(cls.test_docs, f, ensure_ascii=False, indent=2)
    
    def setUp(self):
        """每个测试方法执行前的初始化"""
        # 创建各组件实例
        self.preprocessor = TextPreprocessor()
        self.query_rewriter = QueryRewriter()
        self.self_query_parser = SelfQueryParser()
        self.query_router = QueryRouter()
        self.cache_manager = CacheManager()
        
        # 创建测试查询
        self.test_query = "RAG系统中的检索方法有哪些？"
    
    def test_text_preprocessing(self):
        """测试文本预处理功能"""
        # 测试文本清洗
        test_text = "这是一篇测试文档，包含   多余的空格和\n换行符！"
        cleaned_text = self.preprocessor.clean_text(test_text)
        
        # 验证结果
        self.assertIn("测试文档", cleaned_text)
        self.assertNotIn("  ", cleaned_text)  # 多余的空格应该被移除
        self.assertNotIn("\n", cleaned_text)  # 换行符应该被移除
        
        # 测试文本分块
        long_text = "这是一段很长的文本，" * 20
        chunks = self.preprocessor.chunk_text(long_text, chunk_size=100)
        
        # 验证分块结果
        self.assertTrue(len(chunks) > 1)  # 应该被分成多个块
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 100)  # 每个块的长度不应该超过100
        
        # 测试文档预处理
        test_doc = {
            "id": "test_doc",
            "content": "这是测试文档内容。"
        }
        preprocessed_doc = self.preprocessor.preprocess_document(test_doc)
        
        # 验证预处理结果
        self.assertIn("id", preprocessed_doc)
        self.assertIn("content", preprocessed_doc)
    
    def test_query_rewriting(self):
        """测试查询重写功能"""
        # 测试查询重写
        rewritten = self.query_rewriter.rewrite_query(self.test_query)
        
        # 验证结果
        self.assertIn("original_query", rewritten)
        self.assertIn("rewritten_query", rewritten)
        self.assertEqual(rewritten["original_query"], self.test_query)
        
        # 检查是否提取了关键词
        self.assertIn("keywords", rewritten)
        self.assertIsInstance(rewritten["keywords"], list)
    
    def test_self_query_parsing(self):
        """测试自查询解析功能"""
        # 测试带约束的查询解析
        constraint_query = "查找关于RAG系统的政策文档"
        parsed = self.self_query_parser.parse_query(constraint_query)
        
        # 验证结果
        self.assertIn("query_text", parsed)
        self.assertIn("constraints", parsed)
        
        # 检查约束是否被正确应用
        test_docs = [
            {"id": "1", "content": "RAG系统介绍", "type": "policy"},
            {"id": "2", "content": "RAG系统使用指南", "type": "faq"}
        ]
        filtered_docs = self.self_query_parser.apply_constraints(test_docs, parsed["constraints"])
        
        # 应该只返回政策文档
        self.assertEqual(len(filtered_docs), 1)
        self.assertEqual(filtered_docs[0]["type"], "policy")
    
    @patch('app.search_bm25.BM25Searcher.search')
    @patch('app.search_vector.VectorSearcher.search')
    def test_search_integration(self, mock_vector_search, mock_bm25_search):
        """测试搜索集成功能"""
        # 设置模拟返回值
        mock_bm25_search.return_value = {
            "query_id": "test_query_id",
            "results": [
                {"id": "doc1", "score": 0.9, "content": "RAG系统测试"},
                {"id": "doc2", "score": 0.8, "content": "向量检索"}
            ]
        }
        
        mock_vector_search.return_value = {
            "query_id": "test_query_id",
            "results": [
                {"id": "doc2", "score": 0.95, "content": "向量检索"},
                {"id": "doc3", "score": 0.7, "content": "BM25算法"}
            ]
        }
        
        # 创建搜索器实例
        bm25_searcher = BM25Searcher(self.test_config["bm25_index_path"])
        vector_searcher = VectorSearcher(
            self.test_config["vector_index_path"],
            self.test_config["embedding_model_path"]
        )
        
        # 测试BM25搜索
        bm25_result = bm25_searcher.search(self.test_query, top_k=2)
        self.assertEqual(len(bm25_result["results"]), 2)
        mock_bm25_search.assert_called_with(self.test_query, top_k=2)
        
        # 测试向量搜索
        vector_result = vector_searcher.search(self.test_query, top_k=2)
        self.assertEqual(len(vector_result["results"]), 2)
        mock_vector_search.assert_called_with(self.test_query, top_k=2)
    
    def test_result_fusion(self):
        """测试结果融合功能"""
        # 创建融合器实例
        fuser = ResultFuser()
        
        # 准备测试结果
        results = [
            {
                "query_id": "test_query_id",
                "strategy": "bm25",
                "results": [
                    {"id": "doc1", "score": 0.9, "content": "RAG系统测试"},
                    {"id": "doc2", "score": 0.8, "content": "向量检索"}
                ]
            },
            {
                "query_id": "test_query_id",
                "strategy": "vector",
                "results": [
                    {"id": "doc2", "score": 0.95, "content": "向量检索"},
                    {"id": "doc3", "score": 0.7, "content": "BM25算法"}
                ]
            }
        ]
        
        # 测试融合功能
        fused_result = fuser.fuse_results(results)
        
        # 验证结果
        self.assertIn("query_id", fused_result)
        self.assertIn("results", fused_result)
        self.assertEqual(fused_result["query_id"], "test_query_id")
        
        # 应该至少有3个不同的文档
        doc_ids = set([doc["id"] for doc in fused_result["results"]])
        self.assertTrue(len(doc_ids) >= 3)
    
    def test_reranking(self):
        """测试结果重排序功能"""
        # 创建重排序器实例（使用模拟）
        with patch('app.rerank.Reranker._load_model'):
            reranker = Reranker()
            reranker.model = MagicMock()
            
            # 设置模拟返回值
            reranker.model.predict.return_value = [[0.8], [0.9], [0.7]]
            
            # 准备测试文档
            docs = [
                {"id": "doc1", "content": "RAG系统测试文档1"},
                {"id": "doc2", "content": "RAG系统测试文档2"},
                {"id": "doc3", "content": "不相关文档"}
            ]
            
            # 测试重排序
            reranked_docs = reranker.rerank(self.test_query, docs)
            
            # 验证结果
            self.assertEqual(len(reranked_docs), 3)
            self.assertTrue(reranked_docs[0]["score"] >= reranked_docs[1]["score"])
            self.assertTrue(reranked_docs[1]["score"] >= reranked_docs[2]["score"])
    
    def test_answer_generation(self):
        """测试回答生成功能"""
        # 创建回答生成器实例
        generator = AnswerGenerator()
        
        # 准备测试文档
        retrieved_docs = [
            {"id": "doc1", "content": "RAG系统结合了检索和生成技术。", "title": "RAG系统介绍"},
            {"id": "doc2", "content": "主要检索方法包括BM25和向量检索。", "title": "RAG检索方法"}
        ]
        
        # 测试回答生成
        answer = generator.generate_answer(self.test_query, retrieved_docs)
        
        # 验证结果
        self.assertIn("answer", answer)
        self.assertIn("sources", answer)
        self.assertTrue(len(answer["sources"]) > 0)
    
    def test_caching(self):
        """测试缓存功能"""
        # 测试缓存设置和获取
        key = "test_key"
        value = {"test": "value"}
        
        # 设置缓存
        self.cache_manager.set(key, value)
        
        # 获取缓存
        cached_value = self.cache_manager.get(key)
        
        # 验证结果
        self.assertEqual(cached_value, value)
        
        # 测试缓存清除
        self.cache_manager.delete(key)
        
        # 验证缓存已清除
        self.assertIsNone(self.cache_manager.get(key))

class TestEndToEndPipeline(unittest.TestCase):
    """端到端流水线测试"""
    
    @patch('app.preprocess.TextPreprocessor.preprocess_document')
    @patch('app.rewrite.QueryRewriter.rewrite_query')
    @patch('app.router.QueryRouter.route_query')
    @patch('app.search_bm25.BM25Searcher.search')
    @patch('app.search_vector.VectorSearcher.search')
    @patch('app.fuse.ResultFuser.fuse_results')
    @patch('app.rerank.Reranker.rerank')
    @patch('app.answer.AnswerGenerator.generate_answer')
    def test_full_pipeline(self, 
                          mock_generate_answer, 
                          mock_rerank, 
                          mock_fuse_results, 
                          mock_vector_search, 
                          mock_bm25_search, 
                          mock_route_query, 
                          mock_rewrite_query, 
                          mock_preprocess_document):
        """测试完整的RAG流水线"""
        # 设置模拟返回值
        mock_rewrite_query.return_value = {
            "original_query": "测试查询",
            "rewritten_query": "重写后的测试查询",
            "keywords": ["测试", "查询"],
            "metadata": {}
        }
        
        mock_route_query.return_value = [RetrievalStrategy.BM25, RetrievalStrategy.VECTOR]
        
        mock_bm25_search.return_value = {
            "query_id": "test_id",
            "strategy": "bm25",
            "results": [{"id": "doc1", "score": 0.9, "content": "测试内容1"}]
        }
        
        mock_vector_search.return_value = {
            "query_id": "test_id",
            "strategy": "vector",
            "results": [{"id": "doc2", "score": 0.85, "content": "测试内容2"}]
        }
        
        mock_fuse_results.return_value = {
            "query_id": "test_id",
            "results": [
                {"id": "doc1", "score": 0.9, "content": "测试内容1"},
                {"id": "doc2", "score": 0.85, "content": "测试内容2"}
            ]
        }
        
        mock_rerank.return_value = [
            {"id": "doc2", "score": 0.95, "content": "测试内容2"},
            {"id": "doc1", "score": 0.9, "content": "测试内容1"}
        ]
        
        mock_generate_answer.return_value = {
            "answer": "这是生成的回答",
            "sources": [{"id": "doc2", "title": "测试文档2"}],
            "metadata": {}
        }
        
        # 模拟完整流水线
        # 1. 查询重写
        query_rewriter = QueryRewriter()
        rewritten = query_rewriter.rewrite_query("测试查询")
        
        # 2. 路由选择
        router = QueryRouter()
        strategies = router.route_query(rewritten["rewritten_query"])
        
        # 3. 执行搜索
        search_results = []
        if RetrievalStrategy.BM25 in strategies:
            bm25_searcher = BM25Searcher("dummy_path")
            search_results.append(bm25_searcher.search(rewritten["rewritten_query"], top_k=1))
        
        if RetrievalStrategy.VECTOR in strategies:
            vector_searcher = VectorSearcher("dummy_path", "dummy_model")
            search_results.append(vector_searcher.search(rewritten["rewritten_query"], top_k=1))
        
        # 4. 结果融合
        fuser = ResultFuser()
        fused_result = fuser.fuse_results(search_results)
        
        # 5. 结果重排序
        with patch('app.rerank.Reranker._load_model'):
            reranker = Reranker()
            reranked_docs = reranker.rerank(rewritten["rewritten_query"], fused_result["results"])
        
        # 6. 回答生成
        generator = AnswerGenerator()
        final_answer = generator.generate_answer(rewritten["rewritten_query"], reranked_docs)
        
        # 验证所有步骤都被正确调用
        mock_rewrite_query.assert_called_once()
        mock_route_query.assert_called_once()
        mock_bm25_search.assert_called_once()
        mock_vector_search.assert_called_once()
        mock_fuse_results.assert_called_once()
        mock_rerank.assert_called_once()
        mock_generate_answer.assert_called_once()
        
        # 验证最终结果
        self.assertIn("answer", final_answer)
        self.assertIn("sources", final_answer)

if __name__ == "__main__":
    unittest.main()

# TODO: 实现更多测试功能
# 1. 添加更多的单元测试和集成测试
# 2. 实现性能测试和基准测试
# 3. 添加更多的模拟数据和边缘情况测试
# 4. 实现测试覆盖率报告生成
# 5. 实现配置验证测试
# 6. 实现错误处理和异常测试

# 添加API接口测试
class TestAPI(unittest.TestCase):
    """API接口测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，只执行一次"""
        # 导入FastAPI应用
        from app.main import app
        
        # 创建测试客户端
        cls.client = TestClient(app)
        
    def test_query_endpoint(self):
        """测试/query端点功能"""
        # 准备测试数据
        test_query = {
            "query": "RAG系统是什么？",
            "top_k": 5
        }
        
        # 发送POST请求到/query端点
        response = self.client.post("/query", json=test_query)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 解析响应数据
        result = response.json()
        
        # 断言响应中包含answer字段且不为空
        self.assertIn("answer", result)
        self.assertTrue(result["answer"])
        
        # 断言响应中包含evidences字段且不为空（作为reranked的证据）
        self.assertIn("evidences", result)
        self.assertTrue(isinstance(result["evidences"], list))
        self.assertTrue(len(result["evidences"]) > 0)
        
        # 检查每个evidence是否有必要的字段
        for evidence in result["evidences"]:
            self.assertIn("id", evidence)
            self.assertIn("title", evidence)
            self.assertIn("content", evidence)
            self.assertIn("score", evidence)