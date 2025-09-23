import unittest
from unittest.mock import patch, MagicMock, Mock
import json

class TestAPI(unittest.TestCase):
    """测试API接口功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的TestClient
        self.mock_client = Mock()
        
        # 设置模拟响应
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
        
        self.mock_client.post.return_value = mock_response
    
    def test_query_endpoint(self):
        """测试/query端点功能"""
        # 准备测试数据
        test_data = {
            "query": "什么是RAG技术？",
            "query_id": "test_001",
            "user_profile": "{\"domain\": \"人工智能\", \"expertise_level\": \"初级\"}"
        }
        
        # 发送请求
        response = self.mock_client.post("/query", json=test_data)
        
        # 验证请求是否正确发送
        self.mock_client.post.assert_called_once_with("/query", json=test_data)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        result = response.json()
        self.assertIn("answer", result)
        self.assertIsNotNone(result["answer"])
        self.assertNotEqual(result["answer"], "")
        
        # 验证evidences字段
        self.assertIn("evidences", result)
        self.assertIsInstance(result["evidences"], list)
        self.assertGreater(len(result["evidences"]), 0)
        
        # 验证每个evidence的字段
        for evidence in result["evidences"]:
            self.assertIn("id", evidence)
            self.assertIn("title", evidence)
            self.assertIn("content", evidence)
            self.assertIn("score", evidence)
        
        # 验证reranked字段非空
        self.assertIn("reranked", result)
        self.assertIsInstance(result["reranked"], list)
        self.assertGreater(len(result["reranked"]), 0)

if __name__ == "__main__":
    unittest.main()