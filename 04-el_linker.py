#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床文档实体关系挖掘程序

功能：
  - 从实体识别结果(mentions.jsonl)中读取医学实体
  - 实现实体链接功能，将识别出的实体映射到医学术语库概念
  - 计算实体与概念之间的相似度评分
  - 标记无法链接的实体(NIL标记)
  - 保存链接结果到linked_mentions.jsonl

输入：
  - mentions.jsonl：包含识别出的医学实体信息
  - 术语库索引(本程序使用模拟数据，可扩展为真实术语库连接)

输出：
  - linked_mentions.jsonl：包含实体链接结果，格式为{"doc_id": "", "offset": [], "text": "", "type": "", "polarity": "", "temporality": "", "concept_id": "", "score": 0.0, "is_NIL": false}

依赖：
  - transformers: 提供BERT模型用于生成向量表示
  - sentence-transformers: 用于计算文本相似度
  - faiss: 用于高效向量搜索
  - pymilvus: 可选，用于大规模向量存储和搜索
  - elasticsearch: 可选，用于术语库检索
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import faiss

# 设置日志配置
def setup_logging():
    """设置日志记录器，确保同时输出到文件和控制台
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 创建一个logger实例，命名为"EL_Linker"
    logger = logging.getLogger("EL_Linker")
    # 设置日志级别为INFO，捕获所有INFO及以上级别的日志
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器，将日志写入el_linker.log文件
        file_handler = logging.FileHandler("el_linker.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式，包含时间戳、logger名称、日志级别和消息内容
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将处理器添加到logger实例
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 先初始化logger
def setup_global_logger():
    """初始化全局logger，确保在其他模块导入前可用"""
    temp_logger = logging.getLogger("temp")
    temp_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    temp_logger.addHandler(handler)
    return temp_logger

temp_logger = setup_global_logger()

# 检查并导入必要的依赖库
# 添加依赖库检查，确保即使某些库不可用，程序也能以简化模式运行
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    temp_logger.warning("未找到faiss库，将使用简化的相似度计算方法")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    temp_logger.warning("未找到sentence-transformers库，将使用transformers进行向量生成")

try:
    import elasticsearch
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False
    temp_logger.warning("未找到elasticsearch库，无法连接elasticsearch服务")

try:
    import pymilvus
    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False
    temp_logger.warning("未找到pymilvus库，无法使用Milvus向量数据库")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    temp_logger.warning("未找到torch库，将使用简化模式")

# 现在初始化正式的logger
logger = setup_logging()
# 替换临时logger的引用
temp_logger.info("日志系统初始化完成")

class EntityLinker:
    """实体链接器类
    
    提供医学实体链接功能，将识别出的实体映射到医学术语库概念
    使用向量相似度计算和近似最近邻搜索实现高效的实体链接
    """
    def __init__(self, model_name: str = "bert-base-chinese", embedding_dim: int = 768):
        """
        初始化实体链接器
        
        Args:
            model_name: 用于生成向量表示的预训练模型名称
            embedding_dim: 向量嵌入维度
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # 初始化模型：根据可用的库选择合适的模型
        logger.info(f"初始化嵌入模型: {model_name}")
        if HAS_SENTENCE_TRANSFORMERS:
            # 如果有sentence-transformers，优先使用
            self.embedding_model = SentenceTransformer(model_name)
        else:
            # 否则使用基本的transformers模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        # 初始化术语库和向量索引
        self.terminology_db = []  # 术语库存储
        self.concept_embeddings = None  # 概念向量
        self.index = None  # 向量索引
        
        # 加载或创建术语库
        self._load_terminology_db()
        
        # 创建向量索引
        self._create_index()
    
    def _load_terminology_db(self):
        """加载或创建医学术语库
        
        实际应用中，这里可以连接到真实的医学术语库，如UMLS、SNOMED CT等
        当前实现使用模拟数据进行演示
        """
        logger.info("加载医学术语库")
        
        # 模拟医学术语库数据
        # 在实际应用中，这里应该从数据库或文件加载真实的术语库
        self.terminology_db = [
            {"concept_id": "C0001", "name": "高血压", "type": "Disease"},
            {"concept_id": "C0002", "name": "糖尿病", "type": "Disease"},
            {"concept_id": "C0003", "name": "冠心病", "type": "Disease"},
            {"concept_id": "C0004", "name": "肺炎", "type": "Disease"},
            {"concept_id": "C0005", "name": "胃炎", "type": "Disease"},
            {"concept_id": "C0006", "name": "头痛", "type": "Symptom"},
            {"concept_id": "C0007", "name": "发热", "type": "Symptom"},
            {"concept_id": "C0008", "name": "咳嗽", "type": "Symptom"},
            {"concept_id": "C0009", "name": "恶心", "type": "Symptom"},
            {"concept_id": "C0010", "name": "呕吐", "type": "Symptom"}
        ]
        
        logger.info(f"术语库加载完成，共 {len(self.terminology_db)} 个概念")
    
    def _create_index(self):
        """创建向量索引
        
        为术语库中的每个概念生成向量表示，并创建索引用于相似度搜索
        根据可用的库选择合适的索引方法
        """
        logger.info("创建术语库向量索引")
        
        # 为每个概念生成向量表示
        concept_names = [item["name"] for item in self.terminology_db]
        self.concept_embeddings = self._generate_embeddings(concept_names)
        
        # 根据可用的库选择合适的索引方法
        if HAS_FAISS:
            # 使用FAISS索引
            self.index = faiss.IndexFlatL2(self.concept_embeddings.shape[1])  # 使用L2距离
            self.index.add(self.concept_embeddings.astype('float32'))  # 添加向量到索引
            logger.info(f"FAISS向量索引创建完成，索引维度: {self.concept_embeddings.shape[1]}")
        else:
            # 不使用索引，直接进行线性搜索
            logger.info("使用线性搜索方法，未创建FAISS索引")
            self.index = None
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成文本的向量表示
        
        Args:
            texts: 文本列表
        
        Returns:
            np.ndarray: 文本的向量表示矩阵
        """
        # 根据可用的库选择合适的向量生成方法
        try:
            if HAS_SENTENCE_TRANSFORMERS:
                # 使用sentence-transformers生成向量
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                return np.array(embeddings)
            elif HAS_TORCH:
                # 使用基本的transformers模型生成向量
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    # 使用[CLS]标记的输出作为文本的向量表示
                    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
                    embeddings.append(cls_embedding[0])
                
                return np.array(embeddings)
            else:
                # 降级到简单的字符嵌入模式
                logger.warning("未找到足够的嵌入生成库，使用简化字符嵌入")
                embeddings = []
                for text in texts:
                    # 简单的字符频率嵌入
                    char_count = {}
                    for char in text:
                        char_count[char] = char_count.get(char, 0) + 1
                    # 构建固定长度的向量
                    embedding = [char_count.get(chr(i), 0) for i in range(ord('\u4e00'), ord('\u4e30'))][:30]  # 只取部分常用字符
                    if not embedding:
                        embedding = [0] * 30
                    # 归一化
                    norm = np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else 1
                    embeddings.append(np.array(embedding) / norm)
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"生成嵌入时出错: {str(e)}")
            # 降级到简单字符嵌入
            embeddings = []
            for text in texts:
                embedding = [0] * 30  # 固定30维零向量
                embeddings.append(np.array(embedding))
            return np.array(embeddings)
    
    def load_mentions(self, file_path: str) -> List[Dict]:
        """从文件加载实体识别结果
        
        Args:
            file_path: mentions.jsonl文件路径
        
        Returns:
            List[Dict]: 实体识别结果列表
        """
        logger.info(f"开始加载实体识别结果: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"实体识别结果文件不存在: {file_path}")
            raise FileNotFoundError(f"实体识别结果文件不存在: {file_path}")
        
        # 读取JSONL文件
        mentions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    mention = json.loads(line)
                    mentions.append(mention)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON行时出错: {e}")
                    continue
        
        logger.info(f"实体识别结果加载完成，共 {len(mentions)} 个实体")
        return mentions
    
    def link_entities(self, mentions: List[Dict], top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """将实体链接到术语库概念
        
        Args:
            mentions: 实体识别结果列表
            top_k: 每个实体返回的候选概念数量
            threshold: 相似度阈值，低于此值的链接被标记为NIL
        
        Returns:
            List[Dict]: 包含链接结果的实体列表
        """
        logger.info(f"开始实体链接，共 {len(mentions)} 个实体")
        
        linked_mentions = []
        
        for mention in mentions:
            try:
                # 为实体文本生成向量表示
                entity_text = mention["text"]
                entity_embedding = self._generate_embeddings([entity_text])
                
                # 搜索最相似的概念
                if HAS_FAISS and self.index is not None:
                    # 使用FAISS进行向量搜索
                    distances, indices = self.index.search(entity_embedding.astype('float32'), top_k)
                    
                    # 选择最相似的概念
                    if distances[0][0] < (2 - 2 * threshold) and indices[0][0] >= 0:  # 调整阈值以适应L2距离
                        # 找到合适的链接
                        best_match = self.terminology_db[indices[0][0]]
                        # 计算余弦相似度
                        cosine_sim = 1 - (distances[0][0] ** 2) / (2 * np.linalg.norm(entity_embedding) * np.linalg.norm(self.concept_embeddings[indices[0][0]]))
                        linked_mention = {
                            **mention,
                            "concept_id": best_match["concept_id"],
                            "score": float(max(0, cosine_sim)),  # 确保相似度为非负数
                            "is_NIL": False
                        }
                    else:
                        # 未找到合适的链接，标记为NIL
                        linked_mention = {
                            **mention,
                            "concept_id": "NIL",
                            "score": 0.0,
                            "is_NIL": True
                        }
                else:
                    # 不使用FAISS，进行线性搜索计算余弦相似度
                    similarities = []
                    for i, concept_emb in enumerate(self.concept_embeddings):
                        # 计算余弦相似度
                        if np.linalg.norm(entity_embedding) > 0 and np.linalg.norm(concept_emb) > 0:
                            sim = np.dot(entity_embedding[0], concept_emb) / (np.linalg.norm(entity_embedding[0]) * np.linalg.norm(concept_emb))
                        else:
                            sim = 0.0
                        similarities.append((i, sim))
                    
                    # 按相似度排序
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # 选择最相似的概念
                    if similarities and similarities[0][1] >= threshold:
                        # 找到合适的链接
                        best_idx = similarities[0][0]
                        best_match = self.terminology_db[best_idx]
                        linked_mention = {
                            **mention,
                            "concept_id": best_match["concept_id"],
                            "score": float(similarities[0][1]),
                            "is_NIL": False
                        }
                    else:
                        # 未找到合适的链接，标记为NIL
                        linked_mention = {
                            **mention,
                            "concept_id": "NIL",
                            "score": 0.0,
                            "is_NIL": True
                        }
                
                linked_mentions.append(linked_mention)
            except Exception as e:
                logger.error(f"处理实体 '{mention.get('text', 'N/A')}' 时出错: {str(e)}")
                # 出错时标记为NIL
                error_mention = {
                    **mention,
                    "concept_id": "NIL",
                    "score": 0.0,
                    "is_NIL": True
                }
                linked_mentions.append(error_mention)
        
        logger.info("实体链接完成")
        return linked_mentions
    
    def save_linked_mentions(self, linked_mentions: List[Dict], output_file: str = "linked_mentions.jsonl"):
        """保存实体链接结果到文件
        
        Args:
            linked_mentions: 包含链接结果的实体列表
            output_file: 输出文件路径
        """
        logger.info(f"保存实体链接结果到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for mention in linked_mentions:
                f.write(json.dumps(mention, ensure_ascii=False) + '\n')
        
        # 统计链接结果
        total = len(linked_mentions)
        linked = sum(1 for m in linked_mentions if not m["is_NIL"])
        nil_count = total - linked
        
        logger.info(f"链接结果保存完成，共 {total} 个实体，成功链接: {linked} 个，NIL: {nil_count} 个")

def main():
    """主函数，协调整个实体链接流程
    
    负责程序的整体执行流程，包括参数设置、数据加载、实体链接和结果保存
    包含完整的错误处理机制，确保程序稳定运行
    """
    logger.info("实体关系挖掘程序开始执行")
    
    try:
        # 配置参数
        input_file = "mentions.jsonl"
        output_file = "linked_mentions.jsonl"
        model_name = "bert-base-chinese"
        threshold = 0.5  # 相似度阈值
        
        # 初始化实体链接器
        logger.info(f"初始化实体链接器，使用模型: {model_name}")
        logger.info(f"可用功能: FAISS={HAS_FAISS}, Sentence-Transformers={HAS_SENTENCE_TRANSFORMERS}")
        linker = EntityLinker(model_name=model_name)
        
        # 加载实体识别结果
        mentions = linker.load_mentions(input_file)
        
        # 限制处理的数据量，避免处理过多数据
        max_mentions = 100  # 只处理前100个实体用于演示
        if len(mentions) > max_mentions:
            mentions = mentions[:max_mentions]
            logger.info(f"为了演示目的，限制处理 {max_mentions} 个实体")
        
        # 执行实体链接
        linked_mentions = linker.link_entities(mentions, threshold=threshold)
        
        # 保存链接结果
        linker.save_linked_mentions(linked_mentions, output_file)
        
        logger.info("实体关系挖掘程序执行完成")
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        print(f"错误: 文件未找到 - {str(e)}")
        print("请确保mentions.jsonl文件存在于当前目录中")
    except ImportError as e:
        logger.error(f"导入依赖库失败: {str(e)}")
        print(f"错误: 导入依赖库失败 - {str(e)}")
        print("请使用pip install -r requirements.txt安装所需的依赖库")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """程序入口点
    
    当脚本被直接执行时，调用main函数开始执行程序
    """
    main()