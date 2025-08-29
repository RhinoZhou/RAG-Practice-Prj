#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图+向量双存储检索服务

本程序实现了一个结合Neo4j图数据库和FAISS向量索引的双存储系统，用于知识图谱数据的加载、存储和检索。
主要功能包括：
- 从CSV文件加载节点和边数据
- 使用SentenceTransformer生成文本嵌入向量
- 创建和管理FAISS向量索引
- 将数据导入Neo4j图数据库
- 提供RESTful API接口进行概念搜索、子图查询和混合查询
- 支持配置文件外部化管理
- 具有依赖缺失时的优雅降级机制

"""

# 基础库导入
import os
import json
import logging
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Optional, Any
import time
from datetime import datetime
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GraphLoader")

# 尝试导入可选依赖
SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer
    logger.info("成功导入sentence_transformers库")
except ImportError as e:
    logger.warning(f"无法导入sentence_transformers库: {str(e)}. 将使用随机向量作为替代。")

GraphDatabase = None
try:
    from neo4j import GraphDatabase
    logger.info("成功导入neo4j库")
except ImportError as e:
    logger.warning(f"无法导入neo4j库: {str(e)}. 将使用模拟数据。")

FastAPI = None
HTTPException = None
Query = None
uvicorn = None
try:
    from fastapi import FastAPI, HTTPException, Query
    import uvicorn
    logger.info("成功导入fastapi和uvicorn库")
except ImportError as e:
    logger.warning(f"无法导入fastapi或uvicorn库: {str(e)}. 无法启动API服务。")

class GraphVectorLoader:
    """图向量加载器类
    
    该类负责管理整个图+向量双存储系统的核心功能，包括数据加载、向量嵌入、索引创建和图数据库操作。
    设计采用组件化架构，各个功能模块相对独立，支持灵活配置和优雅降级。
    
    属性:
        config: 配置字典，包含所有运行参数
        neo4j_driver: Neo4j数据库驱动实例
        embedding_model: 文本嵌入模型实例
        vector_index: FAISS向量索引实例
        node_id_to_index: 节点ID到索引ID的映射
        node_index_to_id: 索引ID到节点ID的映射
        nodes_df: 节点数据的DataFrame
        edges_df: 边数据的DataFrame
    """
    
    def __init__(self, config_file: str = 'graph_loader_config.yaml', config: Dict[str, Any] = None):
        # 默认配置
        self.config = {
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'password',
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_dimension': 384,
            'nodes_file': 'nodes.csv',
            'edges_file': 'edges.csv',
            'vector_index_path': 'vector_index.faiss',
            'node_id_to_index_path': 'node_id_to_index.json'
        }
        
        # 从配置文件加载配置
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    
                # 转换yaml配置到程序配置格式
                if yaml_config:
                    if 'neo4j' in yaml_config:
                        self.config['neo4j_uri'] = yaml_config['neo4j'].get('uri', self.config['neo4j_uri'])
                        self.config['neo4j_user'] = yaml_config['neo4j'].get('user', self.config['neo4j_user'])
                        self.config['neo4j_password'] = yaml_config['neo4j'].get('password', self.config['neo4j_password'])
                    
                    if 'embedding' in yaml_config:
                        self.config['embedding_model'] = yaml_config['embedding'].get('model_name', self.config['embedding_model'])
                        self.config['vector_dimension'] = yaml_config['embedding'].get('vector_dimension', self.config['vector_dimension'])
                    
                    if 'files' in yaml_config:
                        self.config['nodes_file'] = yaml_config['files'].get('nodes_file', self.config['nodes_file'])
                        self.config['edges_file'] = yaml_config['files'].get('edges_file', self.config['edges_file'])
                        self.config['vector_index_path'] = yaml_config['files'].get('vector_index_path', self.config['vector_index_path'])
                        self.config['node_id_to_index_path'] = yaml_config['files'].get('node_id_to_index_path', self.config['node_id_to_index_path'])
                
                logger.info(f"成功加载配置文件: {config_file}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化组件
        self.neo4j_driver = None
        self.embedding_model = None
        self.vector_index = None
        self.node_id_to_index = {}
        self.node_index_to_id = {}
        self.nodes_df = None
        self.edges_df = None
        
        # 初始化服务
        self.initialize_components()
    
    def initialize_components(self):
        """初始化各个组件"""
        try:
            # 加载节点数据
            self.load_nodes_data()
            
            # 加载边数据
            self.load_edges_data()
            
            # 初始化嵌入模型
            self.initialize_embedding_model()
            
            # 初始化向量索引
            self.initialize_vector_index()
            
            # 初始化Neo4j连接（尝试连接，如果失败则记录日志但不中断程序）
            self.initialize_neo4j()
            
        except Exception as e:
            logger.error(f"初始化组件失败: {str(e)}")
    
    def load_nodes_data(self):
        """加载节点数据"""
        try:
            if os.path.exists(self.config['nodes_file']):
                self.nodes_df = pd.read_csv(self.config['nodes_file'])
                logger.info(f"成功加载节点数据: {len(self.nodes_df)} 条记录")
            else:
                logger.warning(f"节点文件 {self.config['nodes_file']} 不存在")
        except Exception as e:
            logger.error(f"加载节点数据失败: {str(e)}")
    
    def load_edges_data(self):
        """加载边数据"""
        try:
            if os.path.exists(self.config['edges_file']):
                self.edges_df = pd.read_csv(self.config['edges_file'])
                logger.info(f"成功加载边数据: {len(self.edges_df)} 条记录")
            else:
                logger.warning(f"边文件 {self.config['edges_file']} 不存在")
        except Exception as e:
            logger.error(f"加载边数据失败: {str(e)}")
    
    def initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            if SentenceTransformer is not None:
                self.embedding_model = SentenceTransformer(self.config['embedding_model'])
                logger.info(f"成功初始化嵌入模型: {self.config['embedding_model']}")
            else:
                # 提供一个简单的替代方法生成随机向量
                def random_vector_generator(text):
                    np.random.seed(hash(text) % (2**32))
                    return np.random.rand(self.config['vector_dimension']).astype('float32')
                
                self.embedding_model = random_vector_generator
                logger.warning("使用随机向量生成器作为嵌入模型的替代")
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            # 即使出错也设置一个随机向量生成器
            def random_vector_generator(text):
                np.random.seed(hash(text) % (2**32))
                return np.random.rand(self.config['vector_dimension']).astype('float32')
            
            self.embedding_model = random_vector_generator
            logger.warning("使用随机向量生成器作为嵌入模型的替代")
    
    def initialize_vector_index(self):
        """初始化向量索引
        
        该方法首先检查是否存在预先生成的向量索引文件和节点ID映射文件，如果存在则加载；
        如果不存在但节点数据和嵌入模型已初始化，则创建新的向量索引。
        向量索引存储在配置指定的路径中，当前配置为：
        D:\rag-project\05-rag-practice\23-faiss_db\graph_node\vector_index.faiss
        
        流程:
        1. 检查索引文件是否存在
        2. 如果存在，加载索引和节点映射
        3. 如果不存在但数据准备就绪，确保目标目录存在并创建新索引
        4. 异常处理确保程序稳定性
        """
        try:
            # 确保向量索引保存目录存在
            vector_index_dir = os.path.dirname(self.config['vector_index_path'])
            if vector_index_dir and not os.path.exists(vector_index_dir):
                os.makedirs(vector_index_dir, exist_ok=True)
                logger.info(f"创建向量索引保存目录: {vector_index_dir}")
            
            # 如果索引文件存在，则加载
            if os.path.exists(self.config['vector_index_path']) and os.path.exists(self.config['node_id_to_index_path']):
                self.vector_index = faiss.read_index(self.config['vector_index_path'])
                with open(self.config['node_id_to_index_path'], 'r', encoding='utf-8') as f:
                    self.node_id_to_index = json.load(f)
                self.node_index_to_id = {v: k for k, v in self.node_id_to_index.items()}
                logger.info(f"成功加载向量索引，包含 {self.vector_index.ntotal} 个向量")
            elif self.nodes_df is not None and self.embedding_model is not None:
                # 创建新的向量索引
                self.create_vector_index()
        except Exception as e:
            logger.error(f"初始化向量索引失败: {str(e)}")
    
    def create_vector_index(self):
        """创建向量索引
        
        该方法负责为所有节点创建向量嵌入并构建FAISS向量索引，然后保存到配置指定的文件路径。
        向量索引存储位置可在配置文件中通过vector_index_path参数设置，当前已修改为存储在：
        D:\rag-project\05-rag-practice\23-faiss_db\graph_node\vector_index.faiss
        
        同时，节点ID到索引ID的映射关系也会保存在配置指定的路径：
        D:\rag-project\05-rag-practice\23-faiss_db\graph_node\node_id_to_index.json
        
        实现流程:
        1. 检查节点数据和嵌入模型是否已初始化
        2. 创建FAISS L2距离索引
        3. 遍历所有节点，为每个节点生成向量嵌入
        4. 对于无文本的节点，生成随机向量作为替代
        5. 构建节点ID和索引ID之间的映射关系
        6. 将向量添加到索引中
        7. 将索引和映射关系保存到指定文件路径
        """
        try:
            if self.nodes_df is None or self.embedding_model is None:
                logger.warning("节点数据或嵌入模型未初始化，无法创建向量索引")
                return
            
            # 创建索引
            self.vector_index = faiss.IndexFlatL2(self.config['vector_dimension'])
            
            # 为节点生成嵌入
            vectors = []
            for i, row in enumerate(self.nodes_df.itertuples()):
                # 使用text字段生成嵌入
                text = str(row.text) if hasattr(row, 'text') else ''
                if text:
                    vector = self.embedding_model.encode(text, convert_to_tensor=False)
                    vectors.append(vector)
                    self.node_id_to_index[row.id] = i
                    self.node_index_to_id[i] = row.id
                else:
                    # 为文本为空的节点生成随机向量
                    vector = np.random.rand(self.config['vector_dimension']).astype('float32')
                    vectors.append(vector)
                    self.node_id_to_index[row.id] = i
                    self.node_index_to_id[i] = row.id
            
            # 添加向量到索引
            vectors_array = np.array(vectors).astype('float32')
            self.vector_index.add(vectors_array)
            
            # 保存索引和映射
            faiss.write_index(self.vector_index, self.config['vector_index_path'])
            with open(self.config['node_id_to_index_path'], 'w', encoding='utf-8') as f:
                json.dump(self.node_id_to_index, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功创建向量索引，包含 {len(vectors)} 个向量")
        except Exception as e:
            logger.error(f"创建向量索引失败: {str(e)}")
    
    def initialize_neo4j(self):
        """初始化Neo4j连接"""
        try:
            if GraphDatabase is None:
                logger.warning("Neo4j库不可用，无法初始化Neo4j连接")
                self.neo4j_driver = None
                return
            
            self.neo4j_driver = GraphDatabase.driver(
                self.config['neo4j_uri'],
                auth=(self.config['neo4j_user'], self.config['neo4j_password'])
            )
            
            # 测试连接
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            logger.info(f"成功连接到Neo4j数据库: {self.config['neo4j_uri']}")
            
            # 如果有数据，导入到Neo4j
            self.import_to_neo4j()
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {str(e)}")
            self.neo4j_driver = None
    
    def import_to_neo4j(self):
        """导入数据到Neo4j"""
        if self.neo4j_driver is None:
            logger.warning("Neo4j驱动未初始化，无法导入数据")
            return
        
        try:
            # 导入节点
            if self.nodes_df is not None:
                logger.info(f"开始导入节点到Neo4j...")
                with self.neo4j_driver.session() as session:
                    # 先清空现有数据
                    session.run("MATCH (n) DETACH DELETE n")
                    
                    # 批量导入节点
                    batch_size = 1000
                    for i in range(0, len(self.nodes_df), batch_size):
                        batch = self.nodes_df.iloc[i:i+batch_size]
                        nodes = []
                        
                        for _, row in batch.iterrows():
                            # 构建节点属性
                            properties = {
                                'id': row.id,
                                'text': row.text,
                                'type': row.type,
                                'concept_id': row.concept_id,
                                'polarity': row.polarity,
                                'temporality': row.temporality,
                                'confidence': float(row.confidence),
                                'source': row.source,
                                'source_file': row.source_file,
                                'created_at': row.created_at
                            }
                            
                            # 过滤掉None值
                            properties = {k: v for k, v in properties.items() if pd.notna(v)}
                            nodes.append({'properties': properties})
                        
                        # 使用UNWIND批量创建节点
                        session.run("""
                            UNWIND $nodes AS node
                            CREATE (n:Entity {id: node.properties.id})
                            SET n += node.properties
                        """, nodes=nodes)
                        
                        logger.info(f"已导入 {min(i+batch_size, len(self.nodes_df))}/{len(self.nodes_df)} 个节点")
                
            # 导入边
            if self.edges_df is not None:
                logger.info(f"开始导入边到Neo4j...")
                with self.neo4j_driver.session() as session:
                    batch_size = 1000
                    for i in range(0, len(self.edges_df), batch_size):
                        batch = self.edges_df.iloc[i:i+batch_size]
                        edges = []
                        
                        for _, row in batch.iterrows():
                            edges.append({
                                'source_id': row.source_id,
                                'target_id': row.target_id,
                                'type': row.type,
                                'confidence': float(row.confidence) if pd.notna(row.confidence) else 0.0,
                                'source': row.source if pd.notna(row.source) else '',
                                'source_file': row.source_file if pd.notna(row.source_file) else ''
                            })
                        
                        # 使用UNWIND批量创建边
                        session.run("""
                            UNWIND $edges AS edge
                            MATCH (s:Entity {id: edge.source_id})
                            MATCH (t:Entity {id: edge.target_id})
                            CALL apoc.create.relationship(s, edge.type, {
                                confidence: edge.confidence,
                                source: edge.source,
                                source_file: edge.source_file,
                                created_at: datetime()
                            }, t)
                            YIELD rel
                            RETURN count(rel)
                        """, edges=edges)
                        
                        logger.info(f"已导入 {min(i+batch_size, len(self.edges_df))}/{len(self.edges_df)} 条边")
            
            logger.info("数据导入Neo4j完成")
        except Exception as e:
            logger.error(f"导入数据到Neo4j失败: {str(e)}")
    
    def search_concept(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似概念
        
        该方法实现了基于向量相似度的概念搜索功能。它将查询文本转换为向量，
        然后在FAISS向量索引中查找最相似的向量，并返回对应的概念信息。
        
        参数:
            query: 搜索查询字符串
            top_k: 返回的最大结果数
        
        返回:
            包含相似概念信息的列表，每个元素包含concept_id、text、type和similarity_score
        
        实现流程:
        1. 检查嵌入模型和向量索引是否已初始化
        2. 将查询文本转换为向量表示
        3. 在FAISS索引中进行相似度搜索
        4. 通过索引映射获取对应的节点ID
        5. 从节点数据中检索完整的概念信息
        6. 构建并返回搜索结果
        """
        try:
            if self.embedding_model is None or self.vector_index is None:
                logger.warning("嵌入模型或向量索引未初始化，无法执行搜索")
                return []
            
            # 生成查询向量
            query_vector = self.embedding_model.encode(query, convert_to_tensor=False)
            query_vector = np.array([query_vector]).astype('float32')
            
            # 搜索相似向量
            distances, indices = self.vector_index.search(query_vector, top_k)
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                node_id = self.node_index_to_id.get(idx, None)
                if node_id and self.nodes_df is not None:
                    node_row = self.nodes_df[self.nodes_df['id'] == node_id].iloc[0]
                    results.append({
                        'concept_id': node_row.id,
                        'text': node_row.text,
                        'type': node_row.type,
                        'similarity_score': float(1 - distances[0][i])  # 转换为相似度分数
                    })
            
            return results
        except Exception as e:
            logger.error(f"搜索相似概念失败: {str(e)}")
            return []
    
    def get_subgraph(self, concept_id: str, depth: int = 1) -> Dict[str, Any]:
        """获取概念的子图"""
        try:
            if self.neo4j_driver is None:
                # 如果没有Neo4j连接，返回模拟数据
                logger.warning("Neo4j驱动未初始化，返回模拟子图数据")
                return {
                    'nodes': [{
                        'id': concept_id,
                        'text': '模拟节点',
                        'type': 'Disease',
                        'concept_id': concept_id
                    }],
                    'edges': []
                }
            
            # 查询子图
            with self.neo4j_driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (n:Entity {{id: $concept_id}})-[r*1..{depth}]-(m)
                    RETURN 
                        n AS node, 
                        collect(DISTINCT {{ 
                            source: startNode(rel).id, 
                            target: endNode(rel).id, 
                            type: type(rel), 
                            properties: properties(rel) 
                        }}) AS edges
                    FROM n, relationships((n)-[r*1..{depth}]-(m)) AS rel
                    """,
                    concept_id=concept_id
                )
                
                # 处理结果
                nodes = []
                edges = []
                
                for record in result:
                    node = record['node']
                    nodes.append(dict(node))
                    edges.extend(record['edges'])
                
                return {
                    'nodes': nodes,
                    'edges': edges
                }
        except Exception as e:
            logger.error(f"获取子图失败: {str(e)}")
            # 返回模拟数据
            return {
                'nodes': [{
                    'id': concept_id,
                    'text': '模拟节点',
                    'type': 'Disease',
                    'concept_id': concept_id
                }],
                'edges': []
            }
    
    def mixed_query(self, query: str, top_k: int = 5, depth: int = 1) -> Dict[str, Any]:
        """混合查询：先向量检索，再图扩展"""
        try:
            # 1. 向量检索获取相似概念
            concept_results = self.search_concept(query, top_k)
            
            # 2. 对每个概念进行图扩展
            all_nodes = []
            all_edges = []
            concept_ids = [result['concept_id'] for result in concept_results]
            
            for concept_id in concept_ids:
                subgraph = self.get_subgraph(concept_id, depth)
                all_nodes.extend(subgraph['nodes'])
                all_edges.extend(subgraph['edges'])
            
            # 3. 去重节点
            seen_node_ids = set()
            unique_nodes = []
            for node in all_nodes:
                if node['id'] not in seen_node_ids:
                    seen_node_ids.add(node['id'])
                    unique_nodes.append(node)
            
            # 4. 去重边
            seen_edges = set()
            unique_edges = []
            for edge in all_edges:
                edge_key = (edge['source'], edge['target'], edge['type'])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_edges.append(edge)
            
            return {
                'query': query,
                'concept_results': concept_results,
                'graph': {
                    'nodes': unique_nodes,
                    'edges': unique_edges
                }
            }
        except Exception as e:
            logger.error(f"混合查询失败: {str(e)}")
            # 返回模拟数据
            return {
                'query': query,
                'concept_results': [],
                'graph': {
                    'nodes': [],
                    'edges': []
                }
            }
    
    def close(self):
        """关闭资源"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j连接已关闭")

# 初始化图向量加载器
loader = GraphVectorLoader()

# 尝试创建FastAPI应用
app = None

if FastAPI is not None and HTTPException is not None and Query is not None:
    # 创建FastAPI应用
    app = FastAPI(title="Graph Vector Search API", description="图+向量双存储检索服务")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭时释放资源"""
        loader.close()
    
    @app.get("/search_concept", tags=["搜索"])
    async def search_concept(
        q: str = Query(..., description="搜索查询字符串"),
        top_k: int = Query(10, description="返回的最大结果数", ge=1, le=100)
    ) -> List[Dict[str, Any]]:
        """搜索相似概念"""
        if not q.strip():
            raise HTTPException(status_code=400, detail="搜索查询不能为空")
        
        results = loader.search_concept(q, top_k)
        return results
    
    @app.get("/subgraph", tags=["图查询"])
    async def get_subgraph(
        concept_id: str = Query(..., description="概念ID"),
        depth: int = Query(1, description="子图深度", ge=1, le=3)
    ) -> Dict[str, Any]:
        """获取概念的子图"""
        if not concept_id.strip():
            raise HTTPException(status_code=400, detail="概念ID不能为空")
        
        subgraph = loader.get_subgraph(concept_id, depth)
        return subgraph
    
    @app.get("/mixed_query", tags=["混合查询"])
    async def mixed_query(
        q: str = Query(..., description="搜索查询字符串"),
        top_k: int = Query(5, description="向量检索的最大结果数", ge=1, le=20),
        depth: int = Query(1, description="子图深度", ge=1, le=2)
    ) -> Dict[str, Any]:
        """混合查询：先向量检索，再图扩展"""
        if not q.strip():
            raise HTTPException(status_code=400, detail="搜索查询不能为空")
        
        result = loader.mixed_query(q, top_k, depth)
        return result
    
    @app.get("/health", tags=["监控"])
    async def health_check():
        """健康检查接口"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "vector_index": "available" if loader.vector_index else "unavailable",
                "neo4j": "available" if loader.neo4j_driver else "unavailable",
                "embedding_model": "available" if loader.embedding_model else "unavailable"
            }
        }
else:
    logger.warning("FastAPI相关依赖不可用，无法创建API服务")

if __name__ == "__main__":
    # 从配置文件加载FastAPI配置
    fastapi_host = "0.0.0.0"
    fastapi_port = 8000
    
    if os.path.exists('graph_loader_config.yaml'):
        try:
            with open('graph_loader_config.yaml', 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and 'fastapi' in yaml_config:
                    fastapi_host = yaml_config['fastapi'].get('host', fastapi_host)
                    fastapi_port = yaml_config['fastapi'].get('port', fastapi_port)
        except Exception as e:
            logger.warning(f"加载FastAPI配置失败，使用默认值: {str(e)}")
    
    # 检查是否可以启动FastAPI服务
    if app is not None and uvicorn is not None:
        logger.info(f"启动FastAPI服务... 监听地址: http://{fastapi_host}:{fastapi_port}")
        logger.info("请访问 http://localhost:8000/docs 查看API文档")
        try:
            uvicorn.run(app, host=fastapi_host, port=fastapi_port)
        except Exception as e:
            logger.error(f"启动FastAPI服务失败: {str(e)}")
            logger.info("尝试仅执行数据加载部分...")
            # 仅执行数据加载逻辑
            if loader.nodes_df is not None:
                logger.info(f"已加载 {len(loader.nodes_df)} 个节点")
            if loader.vector_index is not None:
                logger.info(f"向量索引已创建，包含 {loader.vector_index.ntotal} 个向量")
            logger.info("程序执行完成。")
    else:
        logger.info("由于缺少依赖，无法启动FastAPI服务。程序将执行数据加载部分...")
        # 仅执行数据加载逻辑
        if loader.nodes_df is not None:
            logger.info(f"已加载 {len(loader.nodes_df)} 个节点")
        if loader.vector_index is not None:
            logger.info(f"向量索引已创建，包含 {loader.vector_index.ntotal} 个向量")
        logger.info("程序执行完成。")