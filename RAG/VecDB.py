# -*- coding: utf-8 -*-
'''
向量数据库模块，提供高效的向量存储、检索和持久化功能。
本模块实现了基于Milvus的向量存储系统，支持文档向量化、向量搜索、
结果重排序等核心功能，是RAG系统的基础组件。主要功能包括：
1. 文档向量的生成与存储
2. 基于向量相似度的高效检索
3. 向量数据库的持久化与加载
4. 支持多模态嵌入模型
该模块为RAG系统提供了可靠的数据存储和检索基础。
'''

import os
import json
import numpy as np
from typing import List, Optional, Any, Dict
from .TxtEmbedding import TextVectorizer

try:
    from pymilvus import MilvusClient
except ImportError:
    print("警告: pymilvus库未安装，将使用本地向量存储作为备选")
    

try:
    from pydantic import BaseModel
except ImportError:
    print("警告: pydantic库未安装，将使用字典替代数据模型")
    
    # 如果没有pydantic，创建一个简单的替代类
    class SimpleModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        
        def __repr__(self):
            return str(self.__dict__)
    
    BaseModel = SimpleModel


class ItemModel(BaseModel):
    """
    检索结果项的数据模型
    
    Attributes:
        key: 文档唯一标识符
        value: 文档内容
        score: 相似度得分
    """
    key: str
    value: str
    score: float


class ReturnModel(BaseModel):
    """
    检索结果返回模型
    
    Attributes:
        topK: 检索到的最相关文档列表
    """
    topK: List[ItemModel]


class LocalVectorStore:
    """
    本地向量存储实现，作为Milvus的备选方案
    当Milvus不可用时使用，存储向量和文档到本地文件系统
    """
    def __init__(self):
        self.vectors = []
        self.documents = []
        self.ids = []
        self.embedding_dim = None
    
    def add_vectors(self, vectors, ids, documents):
        """添加向量和文档"""
        self.vectors.extend(vectors)
        self.ids.extend(ids)
        self.documents.extend(documents)
        if vectors:
            self.embedding_dim = len(vectors[0])
    
    def search(self, data, limit, search_params, output_fields=None):
        """向量搜索实现"""
        query_vector = data[0]  # 假设只搜索一个向量
        
        # 计算余弦相似度
        results = []
        for i, vec in enumerate(self.vectors):
            # 归一化向量
            norm_query = np.linalg.norm(query_vector)
            norm_vec = np.linalg.norm(vec)
            
            if norm_query > 0 and norm_vec > 0:
                similarity = np.dot(query_vector, vec) / (norm_query * norm_vec)
            else:
                similarity = 0.0
            
            # 检查阈值
            if 'radius' in search_params.get('params', {}) and \
               similarity < search_params['params']['radius']:
                continue
            
            results.append({
                'entity': {
                    'key': str(self.ids[i]),
                    'value': self.documents[i]
                },
                'distance': similarity
            })
        
        # 按相似度排序并限制结果数量
        results.sort(key=lambda x: x['distance'], reverse=True)
        return [results[:limit]]
    
    def save(self, path):
        """保存到本地文件"""
        data = {
            'vectors': self.vectors,
            'documents': self.documents,
            'ids': self.ids,
            'embedding_dim': self.embedding_dim
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'vector_store.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    def load(self, path):
        """从本地文件加载"""
        try:
            with open(os.path.join(path, 'vector_store.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.vectors = data.get('vectors', [])
            self.documents = data.get('documents', [])
            self.ids = data.get('ids', [])
            self.embedding_dim = data.get('embedding_dim', None)
            return True
        except Exception:
            return False


class VectorStore:
    """
    向量存储管理类，支持向量数据库的创建、查询、持久化和加载
    
    该类是RAG系统的核心组件之一，负责文档向量的生成、存储、检索和管理。
    它支持使用Milvus向量数据库或本地向量存储作为后端。
    """
    
    def __init__(self, docs: Optional[List[str]] = None, uri: str = None, 
                 collection_name: str = "default_collection") -> None:
        """
        初始化向量存储
        
        Args:
            docs: 初始文档列表（可选）
            uri: Milvus服务器URI或本地存储路径
            collection_name: 集合名称
        """
        self.docs = docs if docs else []
        self.collection_name = collection_name
        self.use_local_store = False
        
        try:
            # 尝试使用Milvus
            if uri and (uri.startswith('http://') or uri.startswith('https://')):
                self.milvus_client = MilvusClient(uri=uri)
                self.use_local_store = False
                print(f"✓ 使用Milvus向量数据库: {uri}")
                
                # 如果集合不存在，创建集合
                try:
                    if not self.milvus_client.has_collection(collection_name=self.collection_name):
                        # 这里暂时不创建，等获取向量后再创建
                        pass
                except Exception as e:
                    print(f"警告: 检查Milvus集合时出错: {str(e)}")
                    # 切换到本地存储
                    self.milvus_client = None
                    self.use_local_store = True
                    self.local_store = LocalVectorStore()
                    print("⚠ 切换到本地向量存储")
            else:
                # 使用本地存储
                self.milvus_client = None
                self.use_local_store = True
                self.local_store = LocalVectorStore()
                print("✓ 使用本地向量存储")
                
        except Exception as e:
            # 出错时切换到本地存储
            print(f"错误: 初始化Milvus客户端失败: {str(e)}")
            self.milvus_client = None
            self.use_local_store = True
            self.local_store = LocalVectorStore()
            print("⚠ 切换到本地向量存储")
    
    def get_vector(self, vectorizer: TextVectorizer) -> None:
        """
        为文档生成向量并存储
        
        Args:
            vectorizer: 向量化模型实例，用于生成文本向量
        """
        try:
            print(f"开始为{len(self.docs)}个文档生成向量...")
            
            # 为每个文档生成向量
            vectors = []
            ids = []
            
            for i, doc in enumerate(self.docs):
                try:
                    # 生成向量
                    vec = vectorizer.get_embedding(doc)
                    vectors.append(vec)
                    ids.append(str(i))
                    
                    # 进度显示
                    if (i + 1) % 10 == 0 or i + 1 == len(self.docs):
                        print(f"  已处理 {i + 1}/{len(self.docs)} 个文档")
                except Exception as e:
                    print(f"警告: 为文档{i}生成向量失败: {str(e)}")
                    # 使用零向量作为备选
                    try:
                        # 尝试获取嵌入维度
                        dim = len(vectorizer.get_embedding("test"))
                        vectors.append([0.0] * dim)
                    except:
                        vectors.append([0.0] * 768)  # 默认使用768维
                    ids.append(str(i))
            
            # 存储向量和文档
            if self.use_local_store:
                # 使用本地存储
                self.local_store.add_vectors(vectors, ids, self.docs)
                print(f"✓ 成功将{len(vectors)}个向量存储到本地")
            else:
                # 使用Milvus
                try:
                    # 检查集合是否存在，如果不存在则创建
                    if not self.milvus_client.has_collection(collection_name=self.collection_name):
                        # 获取向量维度
                        dim = len(vectors[0]) if vectors else 768
                        
                        # 创建集合
                        self.milvus_client.create_collection(
                            collection_name=self.collection_name,
                            dimension=dim,
                            metric_type="IP"  # 内积距离
                        )
                        print(f"✓ 创建Milvus集合: {self.collection_name}")
                    
                    # 准备插入数据
                    data = []
                    for i, (vec, doc) in enumerate(zip(vectors, self.docs)):
                        data.append({
                            "id": i,
                            "vector": vec,
                            "key": str(i),
                            "value": doc
                        })
                    
                    # 插入数据
                    self.milvus_client.insert(
                        collection_name=self.collection_name,
                        data=data
                    )
                    print(f"✓ 成功将{len(data)}个向量插入到Milvus")
                except Exception as e:
                    print(f"错误: 存储向量到Milvus失败: {str(e)}")
                    # 尝试切换到本地存储
                    print("尝试切换到本地存储...")
                    self.use_local_store = True
                    self.local_store = LocalVectorStore()
                    self.local_store.add_vectors(vectors, ids, self.docs)
                    print(f"✓ 成功将{len(vectors)}个向量存储到本地")
        
        except Exception as e:
            print(f"错误: 生成向量时发生异常: {str(e)}")
    
    def persist(self, path: str = "storage") -> bool:
        """
        持久化向量数据库到本地文件系统
        
        Args:
            path: 存储路径
            
        Returns:
            bool: 是否成功持久化
        """
        try:
            print(f"开始持久化向量数据库到: {path}")
            
            # 创建存储目录
            os.makedirs(path, exist_ok=True)
            
            # 保存文档信息
            metadata = {
                "collection_name": self.collection_name,
                "doc_count": len(self.docs),
                "use_local_store": self.use_local_store
            }
            
            with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)
            
            # 保存文档内容
            with open(os.path.join(path, 'documents.json'), 'w', encoding='utf-8') as f:
                json.dump(self.docs, f, ensure_ascii=False)
            
            # 持久化向量
            if self.use_local_store:
                self.local_store.save(path)
            else:
                # 对于Milvus，我们只保存元数据和文档，向量仍然在Milvus中
                print("提示: Milvus向量数据仍然存储在Milvus服务器中")
            
            print(f"✓ 向量数据库持久化完成: {path}")
            return True
        except Exception as e:
            print(f"错误: 持久化向量数据库失败: {str(e)}")
            return False
    
    def load_vector(self, path: str = "storage") -> bool:
        """
        从本地文件系统加载向量数据库
        
        Args:
            path: 存储路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            print(f"开始加载向量数据库从: {path}")
            
            # 加载元数据
            with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.collection_name = metadata.get("collection_name", "default_collection")
            self.use_local_store = metadata.get("use_local_store", True)
            
            # 加载文档
            with open(os.path.join(path, 'documents.json'), 'r', encoding='utf-8') as f:
                self.docs = json.load(f)
            
            # 加载向量
            if self.use_local_store:
                if self.local_store.load(path):
                    print(f"✓ 成功加载本地向量存储，共{len(self.docs)}个文档")
                else:
                    print("错误: 加载本地向量存储失败")
                    return False
            else:
                # 对于Milvus，我们只加载元数据和文档，向量仍然在Milvus中
                print(f"✓ 成功加载Milvus元数据，共{len(self.docs)}个文档")
            
            print(f"✓ 向量数据库加载完成: {path}")
            return True
        except Exception as e:
            print(f"错误: 加载向量数据库失败: {str(e)}")
            return False
    
    def query(self, query: str, vectorizer: TextVectorizer, k: int = 1, threshold: float = 0.5) -> List[str]:
        """
        基于向量相似度查询相关文档
        
        Args:
            query: 查询文本
            vectorizer: 向量化模型实例
            k: 返回前k个最相关的文档
            threshold: 相似度阈值
            
        Returns:
            List[str]: 相关文档内容列表
        """
        try:
            # 生成查询向量
            query_vector = vectorizer.get_embedding(query)
            
            # 执行查询
            if self.use_local_store:
                # 使用本地存储查询
                search_results = self.local_store.search(
                    data=[query_vector],
                    limit=k,
                    search_params={"metric_type": "IP", "params": {'radius': threshold}}
                )
            else:
                # 使用Milvus查询
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=k,
                    search_params={"metric_type": "IP", "params": {'radius': threshold}},
                    output_fields=["key", "value"]
                )
            
            # 提取结果
            results = []
            for res in search_results[0]:
                try:
                    # 确保实体存在
                    if "entity" in res and "value" in res["entity"]:
                        results.append(res["entity"]["value"])
                    else:
                        # 兼容不同格式的返回结果
                        if hasattr(res, "value"):
                            results.append(res.value)
                        else:
                            print(f"警告: 结果格式不正确: {res}")
                except Exception as e:
                    print(f"警告: 处理查询结果时出错: {str(e)}")
            
            # 如果没有结果，返回空列表
            if not results and k > 0:
                print(f"警告: 未找到相似度大于{threshold}的文档")
            
            return results
            
        except Exception as e:
            print(f"错误: 查询向量数据库时出错: {str(e)}")
            # 返回空列表
            return []


if __name__ == '__main__':
    # 示例用法
    try:
        from .TxtEmbedding import BGEVectorizer
        
        # 创建嵌入模型（使用默认路径）
        vectorizer = BGEVectorizer()
        
        # 创建示例文档
        sample_docs = [
            "《民法典》是中国民法典的正式名称，自2021年1月1日起施行。",
            "盗窃罪是指以非法占有为目的，秘密窃取公私财物的行为。",
            "正当防卫是法律赋予公民的一项权利，用于保护自身合法权益。"
        ]
        
        # 创建向量存储
        vector_store = VectorStore(docs=sample_docs)
        
        # 生成向量
        vector_store.get_vector(vectorizer=vectorizer)
        
        # 持久化
        vector_store.persist(path='./test_storage')
        
        # 查询示例
        query = "民法典什么时候生效？"
        results = vector_store.query(query, vectorizer=vectorizer, k=2)
        
        print(f"\n查询结果:\n{results}")
        
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
    
    
