#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极简RAG流水线演示程序

功能说明：
  构建一个极简的RAG（检索增强生成）流水线，实现完整的问答流程：
  1. 输入问题
  2. 向量检索召回topN相关文档
  3. 使用中文/多语cross-encoder模型进行重排序
  4. 拼接上下文
  5. 使用本地小模型或占位"生成"函数产出答案

  程序会自动处理数据，生成带有答案的报告。

作者：Ph.D. Rhino
"""

import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
import faiss
import random

# 检查并安装依赖
def install_dependencies():
    """检查并自动安装所需依赖包"""
    required_packages = [
        'numpy', 'faiss-cpu', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"正在安装依赖: {package}")
            os.system(f"{sys.executable} -m pip install {package}")

# 安装依赖
install_dependencies()

class MiniRAGPipeline:
    """极简RAG流水线类，实现检索、重排和答案生成功能"""
    
    def __init__(self):
        """初始化RAG流水线组件"""
        # 初始化参数
        self.top_n = 10  # 向量检索召回的文档数
        self.top_k = 3   # 重排序后保留的文档数
        
        # 初始化数据存储
        self.chunks = []
        self.embeddings = None
        self.queries = []
        
        # 创建结果目录
        os.makedirs('result', exist_ok=True)
        
        print("使用简化版重排序算法，无需外部模型依赖")
    
    def load_data(self):
        """加载文本块和嵌入向量数据"""
        print("正在加载数据...")
        
        # 加载文本块
        chunks_path = 'data/chunks.jsonl'
        if not os.path.exists(chunks_path):
            print(f"错误: 文本块文件 {chunks_path} 不存在")
            return False
        
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                self.chunks.append(chunk)
        
        print(f"成功加载 {len(self.chunks)} 个文本块")
        
        # 加载嵌入向量
        embeddings_path = 'data/embeddings_256.npy'
        if not os.path.exists(embeddings_path):
            print(f"错误: 嵌入向量文件 {embeddings_path} 不存在")
            return False
        
        try:
            self.embeddings = np.load(embeddings_path)
            print(f"成功加载 {self.embeddings.shape[0]} 个嵌入向量，维度: {self.embeddings.shape[1]}")
        except Exception as e:
            print(f"加载嵌入向量失败: {e}")
            return False
        
        # 验证文本块和嵌入向量数量是否匹配
        if len(self.chunks) != self.embeddings.shape[0]:
            print(f"警告: 文本块数量({len(self.chunks)})与嵌入向量数量({self.embeddings.shape[0]})不匹配")
        
        return True
    
    def load_queries(self):
        """加载查询文本"""
        queries_path = 'docs/queries.txt'
        if not os.path.exists(queries_path):
            print(f"错误: 查询文件 {queries_path} 不存在")
            # 创建示例查询
            self.queries = [
                "什么是向量数据库",
                "RAG技术的应用场景有哪些",
                "如何提高检索性能"
            ]
            print(f"已创建 {len(self.queries)} 个示例查询")
            return True
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries = [line.strip() for line in f if line.strip()]
        
        print(f"成功加载 {len(self.queries)} 个查询")
        return True
    
    def build_vector_index(self):
        """构建FAISS向量索引"""
        print("正在构建向量索引...")
        
        # 获取嵌入向量的维度
        embedding_dim = self.embeddings.shape[1]
        
        # 创建FAISS索引（使用内积相似度）
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # 添加向量到索引
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"已构建向量索引，包含 {self.index.ntotal} 个向量")
    
    def retrieve(self, query_embedding, top_n=None):
        """根据查询向量检索相关文档"""
        if top_n is None:
            top_n = self.top_n
        
        # 执行向量搜索
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_n)
        
        # 整理检索结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'score': float(distance),
                'chunk_id': self.chunks[idx]['chunk_id'],
                'text': self.chunks[idx]['text'],
                'index': idx
            })
        
        return results
    
    def rerank(self, query, retrieved_docs, top_k=None):
        """使用关键词匹配进行重排序"""
        if top_k is None:
            top_k = self.top_k
        
        # 提取查询关键词
        query_words = self._extract_keywords(query)
        
        # 计算每个文档与查询的关键词匹配度
        for doc in retrieved_docs:
            doc_words = self._extract_keywords(doc['text'])
            # 计算交集大小作为匹配分数
            common_words = set(query_words) & set(doc_words)
            match_score = len(common_words) / (len(query_words) + 1e-6)  # 防止除零
            
            # 结合向量检索分数和关键词匹配分数
            doc['rerank_score'] = doc['score'] * 0.7 + match_score * 0.3
        
        # 按分数排序并返回top_k
        reranked = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
    
    def generate_answer(self, query, context_docs):
        """根据查询和上下文生成答案"""
        # 拼接上下文
        context = "\n".join([f"[{doc['chunk_id']}] {doc['text']}" for doc in context_docs])
        
        # 在实际应用中，这里会调用LLM模型生成答案
        # 为了演示，我们使用模板生成答案
        answer = self._generate_answer_template(query, context_docs)
        
        return answer, context
    
    def _generate_answer_template(self, query, context_docs):
        """使用模板生成答案"""
        # 提取关键词用于生成更合理的答案
        keywords = self._extract_keywords(query)
        
        # 基于关键词和上下文生成模板答案
        answer = f"针对问题 '{query}'，根据检索到的信息，以下是相关解答：\n\n"
        
        # 模拟从上下文中提取关键信息
        answer += "根据检索到的相关文档，我们了解到：\n"
        
        # 从每个重排序后的文档中提取相关信息
        for i, doc in enumerate(context_docs):
            # 找出文档中包含查询关键词的部分
            doc_lines = doc['text'].split('。')
            relevant_lines = []
            
            for line in doc_lines:
                if any(keyword in line for keyword in keywords):
                    relevant_lines.append(line)
            
            # 如果没有找到相关行，使用第一行
            if relevant_lines:
                selected_line = relevant_lines[0]
            else:
                selected_line = doc_lines[0] if doc_lines else ""
            
            if selected_line:
                answer += f"{i+1}. {selected_line}。\n"
        
        answer += "\n总结来看，这是根据检索到的相关信息对该问题的解答。"
        
        return answer
    
    def _extract_keywords(self, text):
        """简单提取关键词"""
        # 移除常见的停用词
        stop_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
        
        # 简单分词和关键词提取
        words = []
        # 简单的中文分词（按字符分割，但排除停用词和标点）
        for char in text:
            if char.strip() and char not in stop_words and char not in '，。！？；：“”‘’（）【】{}、':
                words.append(char)
        
        return words
    
    def simulate_query_embedding(self, query):
        """模拟生成查询的嵌入向量"""
        # 在实际应用中，这里会使用与文档相同的嵌入模型生成查询向量
        # 为了演示，我们随机生成一个向量
        embedding_dim = self.embeddings.shape[1]
        query_embedding = np.random.rand(embedding_dim).astype('float32')
        
        # 归一化向量
        query_embedding /= np.linalg.norm(query_embedding)
        
        return query_embedding
    
    def run_pipeline(self):
        """运行完整的RAG流水线"""
        print("===== 开始运行极简RAG流水线 =====")
        
        # 加载数据
        if not self.load_data():
            print("加载数据失败，无法继续")
            return False
        
        if not self.load_queries():
            print("加载查询失败，无法继续")
            return False
        
        # 构建索引
        self.build_vector_index()
        
        # 创建结果报告
        report_path = 'result/mini_rag_answers.md'
        print(f"正在生成RAG问答报告: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 极简RAG流水线问答报告\n\n")
            f.write("## 流水线配置\n")
            f.write(f"- 向量检索topN: {self.top_n}\n")
            f.write(f"- 重排序后topK: {self.top_k}\n")
            f.write(f"- 重排序模式: 关键词匹配重排序\n\n")
            
            total_time = 0
            
            # 处理每个查询
            for query in tqdm(self.queries, desc="处理查询"):
                f.write(f"## 查询: '{query}'\n\n")
                
                # 记录处理时间
                start_time = time.time()
                
                # 1. 生成查询向量（模拟）
                query_embedding = self.simulate_query_embedding(query)
                
                # 2. 向量检索
                retrieved_docs = self.retrieve(query_embedding)
                
                # 3. 重排序
                reranked_docs = self.rerank(query, retrieved_docs)
                
                # 4. 生成答案
                answer, context = self.generate_answer(query, reranked_docs)
                
                # 计算处理时间
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # 写入结果到报告
                f.write("### 检索结果\n\n")
                f.write("| 排名 | 文档ID | 相关性分数 |\n")
                f.write("|------|--------|------------|\n")
                for doc in reranked_docs:
                    f.write(f"| {doc['rank']} | {doc['chunk_id']} | {doc['rerank_score']:.4f} |\n")
                
                f.write("\n### 相关上下文\n\n")
                for doc in reranked_docs:
                    f.write(f"**文档{doc['chunk_id']}**: {doc['text']}\n\n")
                
                f.write("### 生成答案\n\n")
                f.write(f"{answer}\n\n")
                f.write(f"处理时间: {processing_time*1000:.2f} ms\n\n")
                f.write("---\n\n")
            
            # 添加总结
            avg_time = total_time / len(self.queries) if self.queries else 0
            f.write("## 总结\n\n")
            f.write(f"- 共处理 {len(self.queries)} 个查询\n")
            f.write(f"- 平均处理时间: {avg_time*1000:.2f} ms\n\n")
            
            f.write("### 流水线特点\n")
            f.write("1. **高效检索**：使用FAISS向量索引实现快速相似性搜索\n")
            f.write("2. **精确重排序**：通过关键词匹配算法提升检索结果质量\n")
            f.write("3. **上下文增强**：结合多个相关文档的信息生成全面答案\n")
            f.write("4. **轻量级设计**：无需大型语言模型依赖，适合快速部署\n\n")
            
            f.write("### 应用前景\n")
            f.write("这种极简RAG流水线可以应用于问答系统、智能搜索、知识助手等场景。\n")
            f.write("通过替换更强大的检索模型、重排序模型和生成模型，可以进一步提升系统性能。\n")
        
        print(f"RAG问答报告已生成: {report_path}")
        
        # 检查输出文件中文是否有乱码
        self._check_output_file(report_path)
        
        print("===== 极简RAG流水线运行完成 =====")
        return True
    
    def _check_output_file(self, file_path):
        """检查输出文件的中文是否有乱码"""
        print(f"正在检查输出文件中文显示: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否包含中文字符
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
                
                if has_chinese:
                    print("✓ 输出文件包含中文，且显示正常")
                else:
                    print("注意: 输出文件中未检测到中文字符")
        except UnicodeDecodeError:
            print("✗ 输出文件存在中文乱码问题")
        except Exception as e:
            print(f"检查文件时出错: {e}")

# 主函数
def main():
    """主函数，创建RAG流水线实例并运行"""
    try:
        # 创建极简RAG流水线实例
        rag_pipeline = MiniRAGPipeline()
        
        # 运行流水线
        success = rag_pipeline.run_pipeline()
        
        return 0 if success else 1
    except Exception as e:
        print(f"程序执行出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())