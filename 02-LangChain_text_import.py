#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统文本批量导入工具

使用典型的Loader → Cleaner → Splitter → Embedder → VectorStore架构，
实现文本批量写入、幂等处理、失败重试和去重功能。
"""

import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
import ftfy

# LangChain相关导入
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangChainTextImport")

class TextImportPipeline:
    """LangChain文本导入流水线，实现完整的文本处理和向量存储流程"""
    
    def __init__(self,
                 data_dir: str = "data",
                 vector_store_path: str = "vector_store",
                 cache_dir: str = "embedding_cache",
                 batch_size: int = 10,
                 max_retries: int = 3,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100):
        """初始化文本导入流水线
        
        参数:
            data_dir: 存放文本文件的目录
            vector_store_path: 向量存储的保存路径
            cache_dir: 嵌入缓存的保存路径
            batch_size: 批量处理的文件数量
            max_retries: 失败重试的最大次数
            chunk_size: 文本分块的大小
            chunk_overlap: 文本分块的重叠部分大小
        """
        self.data_dir = data_dir
        self.vector_store_path = vector_store_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 记录已处理的文件指纹，用于幂等和去重
        self.processed_file_hashes_path = os.path.join(self.vector_store_path, "processed_files.json")
        self.processed_hashes: Set[str] = self._load_processed_hashes()
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化流水线组件"""
        # ============ 流水线组件初始化：Loader → Splitter → Embedder → VectorStore ============
        
        # 1. Loader - 文本加载器
        # 负责从文件系统加载文本文件内容
        self.loader = DirectoryLoader(
            self.data_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            silent_errors=True
        )
        
        # 2. Splitter - 文本分割器
        # 负责将长文本分割成适合向量化的小块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 3. Embedder - 嵌入模型 (带缓存，使用HuggingFace本地模型)
        # 负责将文本转换为向量表示
        cache = LocalFileStore(self.cache_dir)
        try:
            underlying_embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # 一个轻量级的预训练模型
                model_kwargs={'device': 'cpu'},  # 确保在CPU上运行
                encode_kwargs={'normalize_embeddings': True}  # 对嵌入向量进行归一化
            )
            self.embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings, cache
            )
        except Exception as e:
            # 如果HuggingFace模型不可用，使用一个简单的占位符嵌入
            logger.warning(f"无法加载HuggingFace嵌入模型: {e}")
            from langchain.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)  # 创建一个假的嵌入模型，向量维度为384
        
        # 4. VectorStore - 向量存储
        # 负责存储和检索文本向量
        try:
            # 尝试加载已存在的向量存储
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"加载已存在的向量存储，包含 {self.vector_store.index.ntotal} 个文档")
        except Exception as e:
            # 创建新的向量存储
            logger.info(f"创建新的向量存储: {e}")
            self.vector_store = None
    
    def _clean_text(self, text: str) -> str:
        """Cleaner - 文本清洗器
        负责清洗和标准化文本内容（流水线的第二个环节）
        
        参数:
            text: 输入文本
        返回:
            清洗后的文本
        """
        # 使用ftfy修复编码问题
        cleaned_text = ftfy.fix_text(text)
        
        # 移除多余的空白字符
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 移除BOM头
        if cleaned_text.startswith('\ufeff'):
            cleaned_text = cleaned_text[1:]
        
        return cleaned_text
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的哈希值，用于幂等和去重
        
        参数:
            file_path: 文件路径
        返回:
            文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def _load_processed_hashes(self) -> Set[str]:
        """加载已处理文件的哈希值集合"""
        try:
            if os.path.exists(self.processed_file_hashes_path):
                with open(self.processed_file_hashes_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
        except Exception as e:
            logger.error(f"加载已处理文件哈希失败: {e}")
        return set()
    
    def _save_processed_hashes(self):
        """保存已处理文件的哈希值集合"""
        try:
            with open(self.processed_file_hashes_path, "w", encoding="utf-8") as f:
                json.dump(list(self.processed_hashes), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存已处理文件哈希失败: {e}")
    
    def _process_with_retry(self, func, *args, **kwargs):
        """带重试机制的函数执行
        
        参数:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        返回:
            函数执行结果
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"函数执行失败，已达到最大重试次数 {self.max_retries}: {e}")
                    raise
                logger.warning(f"函数执行失败，正在重试 ({retries}/{self.max_retries}): {e}")
                time.sleep(2 * retries)  # 指数退避
    
    def _get_all_text_files(self) -> List[str]:
        """获取所有待处理的文本文件
        
        返回:
            文本文件路径列表
        """
        text_files = []
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.data_dir, filename)
                    # 计算文件哈希
                    file_hash = self._calculate_file_hash(file_path)
                    if file_hash and file_hash not in self.processed_hashes:
                        text_files.append((file_path, file_hash))
            logger.info(f"找到 {len(text_files)} 个未处理的文本文件")
            return text_files
        except Exception as e:
            logger.error(f"获取文本文件列表失败: {e}")
            return []
    
    def process_single_file(self, file_path: str, file_hash: str) -> bool:
        """处理单个文本文件
        实现完整的流水线处理流程：Loader → Cleaner → Splitter → Embedder → VectorStore
        
        参数:
            file_path: 文件路径
            file_hash: 文件哈希值
        返回:
            处理是否成功
        """
        try:
            # ============ 完整流水线处理流程 ============
            # 1. Loader - 加载文本
            # 从文件系统读取原始文本内容
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            
            if not documents:
                logger.warning(f"文件为空: {file_path}")
                return False
            
            # 2. Cleaner - 清洗文本
            # 对加载的文本进行清洗和标准化处理
            cleaned_docs = []
            for doc in documents:
                cleaned_content = self._clean_text(doc.page_content)
                if cleaned_content:
                    # 保留元数据并添加文件路径
                    metadata = doc.metadata.copy()
                    metadata["source"] = file_path
                    metadata["file_hash"] = file_hash
                    cleaned_docs.append(Document(page_content=cleaned_content, metadata=metadata))
            
            if not cleaned_docs:
                logger.warning(f"文件清洗后为空: {file_path}")
                return False
            
            # 3. Splitter - 分割文本
            # 将清洗后的文本分割成适合向量化的小块
            split_docs = self.text_splitter.split_documents(cleaned_docs)
            
            if not split_docs:
                logger.warning(f"文件分割后为空: {file_path}")
                return False
            
            logger.info(f"文件 {os.path.basename(file_path)} 分割为 {len(split_docs)} 个文档块")
            
            # 4. Embedder + VectorStore - 向量化并存储
            # 使用嵌入模型将文本转换为向量，并存储到向量数据库中
            if self.vector_store is None:
                # 首次创建向量存储
                self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            else:
                # 添加到已有的向量存储
                self.vector_store.add_documents(split_docs)
            
            # 5. 标记为已处理
            self.processed_hashes.add(file_hash)
            self._save_processed_hashes()
            
            return True
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False
    
    def batch_process_files(self):
        """批量处理文本文件"""
        # 获取所有未处理的文本文件
        text_files = self._get_all_text_files()
        
        if not text_files:
            logger.info("没有发现未处理的文本文件")
            return
        
        # 分批处理文件
        total_batches = (len(text_files) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(text_files))
            batch_files = text_files[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}, 文件数量: {len(batch_files)}")
            
            # 使用tqdm显示进度
            for file_path, file_hash in tqdm(batch_files, desc=f"批次 {batch_idx + 1}"):
                # 使用带重试机制的处理函数
                try:
                    success = self._process_with_retry(
                        self.process_single_file, 
                        file_path=file_path,
                        file_hash=file_hash
                    )
                    if success:
                        logger.info(f"成功处理文件: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"文件处理失败，已达到最大重试次数: {os.path.basename(file_path)}")
            
            # 每批次保存一次向量存储
            try:
                self.vector_store.save_local(self.vector_store_path)
                logger.info(f"向量存储已保存到 {self.vector_store_path}")
            except Exception as e:
                logger.error(f"保存向量存储失败: {e}")
    
    def create_sample_text_files(self, num_files: int = 5):
        """创建示例文本文件用于测试
        
        参数:
            num_files: 要创建的示例文件数量
        """
        sample_contents = [
            "这是一个示例文本文件，包含一些测试内容。\nRAG系统可以有效地检索和生成相关信息。\n文本清洗和标准化是RAG系统中的重要步骤。",
            "LangChain是一个强大的框架，可以简化LLM应用的开发。\n它提供了丰富的组件和工具，支持各种文本处理任务。\n向量存储是RAG系统的核心组件之一。",
            "嵌入模型可以将文本转换为高维向量表示。\n相似的文本在向量空间中距离更近。\nFAISS是一个高效的向量相似度搜索库。",
            "幂等性是系统设计中的重要概念。\n它确保重复执行相同的操作不会产生副作用。\n在数据处理中，幂等性可以防止数据重复。",
            "失败重试机制可以提高系统的稳定性和可靠性。\n指数退避是一种常用的重试策略。\n它可以减少重试对系统的压力。"
        ]
        
        # 如果示例内容不足，重复使用
        while len(sample_contents) < num_files:
            sample_contents.extend(sample_contents)
        
        for i in range(num_files):
            file_name = f"sample_text_{i+1}.txt"
            file_path = os.path.join(self.data_dir, file_name)
            
            # 避免覆盖已有的文件
            if os.path.exists(file_path):
                logger.warning(f"文件已存在，跳过创建: {file_path}")
                continue
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(sample_contents[i])
                logger.info(f"创建示例文件: {file_path}")
            except Exception as e:
                logger.error(f"创建示例文件失败 {file_path}: {e}")

if __name__ == "__main__":
    # 创建文本导入流水线实例
    pipeline = TextImportPipeline()
    
    # 创建示例文本文件
    print("正在创建示例文本文件...")
    pipeline.create_sample_text_files(num_files=5)
    
    # 批量处理文本文件
    print("\n开始批量处理文本文件...")
    pipeline.batch_process_files()
    
    print("\n文本导入流程完成！")