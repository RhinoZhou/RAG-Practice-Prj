# -*- coding: utf-8 -*-
'''
文本向量化模块，提供文本到向量的转换功能。
本模块实现了多种嵌入模型的接口，用于将文本转换为向量表示，
支持语义相似度计算和批量处理，是RAG系统的核心组件之一。主要特性包括：
1. 统一的嵌入模型接口抽象
2. 支持多种预训练嵌入模型
3. 批量嵌入生成
4. 语义相似度计算
5. 自动依赖检查和安装
6. 性能优化和错误处理
'''

import os
import sys
import time
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod


# 自动安装依赖
def setup_dependencies():
    """
    自动安装所需依赖库
    """
    try:
        import pip
        
        required_packages = [
            'sentence-transformers>=2.2.0',
            'numpy>=1.24.0',
            'torch>=2.0.0',
            'transformers>=4.36.0',
            'tqdm>=4.65.0'
        ]
        
        print("正在检查并安装必要的依赖...")
        for package in required_packages:
            try:
                # 尝试导入包
                if package.split('>=')[0].replace('-', '_') == 'sentence_transformers':
                    import sentence_transformers
                elif package.split('>=')[0].replace('-', '_') == 'numpy':
                    import numpy
                elif package.split('>=')[0].replace('-', '_') == 'torch':
                    import torch
                elif package.split('>=')[0].replace('-', '_') == 'transformers':
                    import transformers
                elif package.split('>=')[0].replace('-', '_') == 'tqdm':
                    import tqdm
            except ImportError:
                print(f"安装 {package}...")
                pip.main(['install', package])
        
        print("✓ 依赖安装完成")
    except Exception as e:
        print(f"警告: 自动安装依赖时出错: {str(e)}")
        print("请手动安装所需依赖")


# 自动安装依赖
setup_dependencies()


# 导入必要的库
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import numpy as np


def determine_device():
    """
    获取可用的设备名称
    
    Returns:
        设备名称 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon GPU
    else:
        return 'cpu'


class TextVectorizer(ABC):
    """
    嵌入模型抽象基类，定义了所有嵌入模型必须实现的接口
    """
    
    def __init__(self):
        """初始化嵌入模型"""
        self.model_name = "Base Embedding"
        self.initialized = False
        self._model = None
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量（浮点数列表）
        """
        pass
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
        """
        results = []
        
        # 计时开始
        start_time = time.time()
        
        # 分批次处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                try:
                    embedding = self.get_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"警告: 获取文本嵌入时出错: {str(e)}")
                    # 如果出错，生成零向量作为替代
                    if self._model:
                        embedding_dim = self._model.get_sentence_embedding_dimension()
                    else:
                        embedding_dim = 768  # 默认维度
                    batch_embeddings.append([0.0] * embedding_dim)
            
            results.extend(batch_embeddings)
        
        # 计算耗时
        end_time = time.time()
        print(f"  批量嵌入生成耗时: {end_time - start_time:.4f}秒 (处理 {len(texts)} 个文本)")
        
        return results
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值（范围在[-1, 1]之间）
        """
        try:
            # 转换为numpy数组
            vec1 = np.array(vector1).reshape(1, -1)
            vec2 = np.array(vector2).reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cos_sim(vec1, vec2).numpy()[0][0]
            return float(similarity)  # 转换为Python原生浮点数
        except Exception as e:
            print(f"警告: 计算余弦相似度时出错: {str(e)}")
            return 0.0
    
    def batch_cosine_similarity(self, query_vector: List[float], vectors: List[List[float]]) -> List[float]:
        """
        批量计算查询向量与多个向量的余弦相似度
        
        Args:
            query_vector: 查询向量
            vectors: 向量列表
            
        Returns:
            相似度值列表
        """
        similarities = []
        for vector in vectors:
            try:
                similarity = self.cosine_similarity(query_vector, vector)
                similarities.append(similarity)
            except Exception as e:
                print(f"警告: 批量计算相似度时出错: {str(e)}")
                similarities.append(0.0)
        return similarities
    
    def health_check(self) -> bool:
        """
        检查嵌入模型健康状态
        
        Returns:
            嵌入模型是否正常
        """
        try:
            test_text = "测试文本"
            embedding = self.get_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            print(f"错误: 嵌入模型健康检查失败: {str(e)}")
            return False
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        Returns:
            嵌入维度
        """
        try:
            if self._model and hasattr(self._model, 'get_sentence_embedding_dimension'):
                return self._model.get_sentence_embedding_dimension()
            else:
                # 通过测试获取维度
                test_text = "测试"
                test_embedding = self.get_embedding(test_text)
                return len(test_embedding)
        except Exception as e:
            print(f"警告: 获取嵌入维度时出错: {str(e)}")
            return 768  # 默认维度


class BGEVectorizer(TextVectorizer):
    """
    BGE嵌入模型实现
    基于BAAI的BGE预训练模型，用于获取高质量的文本嵌入
    """
    
    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5') -> None:
        """
        初始化BGE嵌入模型
        
        Args:
            path: 模型路径或Hugging Face模型名称
        """
        super().__init__()
        self.model_name = "BGE Embedding"
        self.path = path
        
        # 默认模型路径列表（按优先级排序）
        self.default_model_paths = [
            path,
            "./models/bge-base-zh-v1.5",
            "./model_hub/ZhipuAI/bge-large-zh-v1.5",
            "/root/sunyd/model_hub/ZhipuAI/bge-large-zh-v1___5/",  # 处理路径中的下划线问题
            "BAAI/bge-base-zh-v1.5",
            "BAAI/bge-large-zh-v1.5"
        ]
        
        # 尝试加载模型
        self._model = None
        self.device = None
        self.load_model()
    
    def load_model(self) -> bool:
        """
        加载嵌入模型，尝试多个路径
        
        Returns:
            是否成功加载
        """
        try:
            print(f"正在加载{BGEVectorizer.__name__}...")
            
            # 确定设备
            self.device = determine_device()
            print(f"  使用设备: {self.device}")
            
            # 尝试加载模型
            model_loaded = False
            for model_path in self.default_model_paths:
                if not model_path:  # 跳过空路径
                    continue
                    
                try:
                    print(f"  尝试加载模型: {model_path}")
                    
                    # 特殊处理路径中的下划线问题
                    if "bge-large-zh-v1___5" in model_path:
                        model_path = model_path.replace("v1___5", "v1.5")
                    
                    # 加载模型
                    self._model = SentenceTransformer(
                        model_path,
                        device=self.device,
                        trust_remote_code=True
                    )
                    
                    # 进行一个小测试确保模型正常
                    test_embedding = self._model.encode("测试")
                    print(f"  模型加载成功，嵌入维度: {len(test_embedding)}")
                    
                    model_loaded = True
                    self.initialized = True
                    print(f"✓ 成功加载模型: {model_path}")
                    break
                    
                except Exception as e:
                    print(f"  加载失败: {str(e)}")
                    continue
            
            if not model_loaded:
                raise Exception("所有模型路径都加载失败")
                
            return True
            
        except Exception as e:
            print(f"错误: 加载{BGEVectorizer.__name__}失败: {str(e)}")
            print("请确保模型路径正确且已下载模型文件")
            self.initialized = False
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量（浮点数列表）
        """
        if not self.initialized:
            print("警告: 嵌入模型未初始化，尝试重新加载")
            if not self.load_model():
                # 如果加载失败，返回默认向量
                return [0.0] * 768
        
        try:
            # 验证输入
            if not text or not isinstance(text, str):
                text = ""  # 空文本处理
            
            # 生成嵌入
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,  # 转换为numpy数组
                normalize_embeddings=True  # 归一化嵌入向量
            )
            
            # 转换为Python原生列表
            return embedding.tolist()
            
        except Exception as e:
            print(f"错误: 获取文本嵌入时出错: {str(e)}")
            # 出错时返回零向量
            try:
                dim = self.get_embedding_dimension()
                return [0.0] * dim
            except:
                return [0.0] * 768  # 默认维度
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量获取文本嵌入向量，利用SentenceTransformer的批量处理优化
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
        """
        if not self.initialized:
            print("警告: 嵌入模型未初始化，尝试重新加载")
            if not self.load_model():
                # 如果加载失败，返回默认向量列表
                return [[0.0] * 768 for _ in texts]
        
        try:
            # 验证输入
            if not texts:
                return []
                
            # 计时开始
            start_time = time.time()
            
            # 清理和验证文本列表
            valid_texts = []
            for text in texts:
                if text and isinstance(text, str):
                    valid_texts.append(text)
                else:
                    valid_texts.append("")  # 空文本处理
            
            # 使用模型的批量encode功能
            embeddings = self._model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # 计算耗时
            end_time = time.time()
            print(f"  BGE批量向量化生成耗时: {end_time - start_time:.4f}秒 (处理 {len(texts)} 个文本)")
            
            # 转换为Python原生列表
            return embeddings.tolist()
            
        except Exception as e:
            print(f"错误: 批量获取文本嵌入时出错: {str(e)}")
            # 出错时返回零向量列表
            try:
                dim = self.get_embedding_dimension()
                return [[0.0] * dim for _ in texts]
            except:
                return [[0.0] * 768 for _ in texts]  # 默认维度


class UniversalVectorizer(TextVectorizer):
    """
    通用SentenceTransformer嵌入模型实现
    可以加载任意SentenceTransformer兼容的模型
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> None:
        """
        初始化SentenceTransformer嵌入模型
        
        Args:
            model_name: 模型名称或路径
        """
        super().__init__()
        self.model_name = "SentenceTransformer Embedding"
        self.model_path = model_name
        
        # 默认模型路径列表
        self.default_model_paths = [
            model_name,
            f"./models/{model_name}",
            model_name
        ]
        
        # 尝试加载模型
        self._model = None
        self.device = None
        self.load_model()
    
    def load_model(self) -> bool:
        """
        加载SentenceTransformer模型
        
        Returns:
            是否成功加载
        """
        try:
            print(f"正在加载{UniversalVectorizer.__name__}...")
            
            # 确定设备
            self.device = get_device_name()
            print(f"  使用设备: {self.device}")
            
            # 尝试加载模型
            model_loaded = False
            for model_path in self.default_model_paths:
                if not model_path:
                    continue
                    
                try:
                    print(f"  尝试加载模型: {model_path}")
                    
                    # 加载模型
                    self._model = SentenceTransformer(
                        model_path,
                        device=self.device,
                        trust_remote_code=True
                    )
                    
                    # 测试
                    test_embedding = self._model.encode("测试")
                    print(f"  模型加载成功，嵌入维度: {len(test_embedding)}")
                    
                    model_loaded = True
                    self.initialized = True
                    print(f"✓ 成功加载模型: {model_path}")
                    break
                    
                except Exception as e:
                    print(f"  加载失败: {str(e)}")
                    continue
            
            if not model_loaded:
                raise Exception("所有模型路径都加载失败")
                
            return True
            
        except Exception as e:
            print(f"错误: 加载{UniversalVectorizer.__name__}失败: {str(e)}")
            self.initialized = False
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量（浮点数列表）
        """
        if not self.initialized:
            print("警告: 嵌入模型未初始化，尝试重新加载")
            if not self.load_model():
                # 如果加载失败，返回默认向量
                return [0.0] * 384  # MiniLM默认维度
        
        try:
            # 验证输入
            if not text or not isinstance(text, str):
                text = ""  # 空文本处理
            
            # 生成嵌入
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # 转换为Python原生列表
            return embedding.tolist()
            
        except Exception as e:
            print(f"错误: 获取文本嵌入时出错: {str(e)}")
            # 出错时返回零向量
            try:
                dim = self.get_embedding_dimension()
                return [0.0] * dim
            except:
                return [0.0] * 384  # MiniLM默认维度


# 创建嵌入模型工厂函数
def create_vectorizer_instance(embedding_type: str = "bge", **kwargs) -> TextVectorizer:
    """
    创建嵌入模型实例的工厂函数
    
    Args:
        embedding_type: 嵌入模型类型，支持 "bge", "sentence-transformer"
        **kwargs: 嵌入模型初始化参数
        
    Returns:
        嵌入模型实例
    """
    try:
        if embedding_type.lower() == "bge":
            return BGEVectorizer(**kwargs)
        elif embedding_type.lower() == "sentence-transformer" or embedding_type.lower() == "st":
            return UniversalVectorizer(**kwargs)
        else:
            print(f"警告: 不支持的嵌入模型类型: {embedding_type}，使用默认BGE模型")
            return BgeEmbedding(**kwargs)
    except Exception as e:
        print(f"错误: 创建嵌入模型实例时出错: {str(e)}")
        # 尝试返回BGE模型作为备用
        try:
            return BgeEmbedding()
        except:
            print("严重错误: 无法创建任何嵌入模型实例")
            # 创建一个最小化的虚拟嵌入模型
            class DummyVectorizer(TextVectorizer):
                def get_embedding(self, text: str) -> List[float]:
                    return [0.0] * 768
            return DummyEmbedding()


if __name__ == "__main__":
    # 示例用法
    try:
        print("=== 测试文本向量化模块 ===")
        
        # 测试BGE嵌入模型
        print("\n1. 测试BGE向量化模型:")
        vectorizer_model = create_vectorizer_instance("bge")
        
        # 健康检查
        if vectorizer_model.health_check():
            print("向量化模型健康检查通过")
            print(f"向量维度: {vectorizer_model.get_embedding_dimension()}")
            
            # 测试单个文本嵌入
            text1 = "孙永荻"
            text2 = "吴娜"
            
            print(f"\n获取文本向量: '{text1}'")
            emb1 = vectorizer_model.get_embedding(text1)
            print(f"嵌入向量长度: {len(emb1)}")
            print(f"前5个值: {emb1[:5]}")
            
            print(f"\n获取文本向量: '{text2}'")
            emb2 = vectorizer_model.get_embedding(text2)
            print(f"嵌入向量长度: {len(emb2)}")
            
            # 计算余弦相似度
            similarity = embedding_model.cosine_similarity(emb1, emb2)
            print(f"\n'{text1}' 和 '{text2}' 的余弦相似度: {similarity:.6f}")
            
            # 测试批量嵌入
            print("\n测试批量向量化:")
            batch_texts = ["你好", "世界", "这是一个测试", "批量处理"]
            batch_embeddings = vectorizer_model.get_embeddings(batch_texts)
            print(f"批量嵌入完成，共 {len(batch_embeddings)} 个嵌入向量")
            print(f"每个向量长度: {len(batch_embeddings[0])}")
            
            # 批量相似度计算
            if batch_embeddings:
                query_vector = batch_embeddings[0]  # 使用第一个向量作为查询
                similarities = embedding_model.batch_cosine_similarity(query_vector, batch_embeddings[1:])
                print("\n与第一个文本的批量相似度:")
                for i, sim in enumerate(similarities):
                    print(f"  与 '{batch_texts[i+1]}' 的相似度: {sim:.6f}")
        else:
            print("向量化模型健康检查失败")
            
        # 测试SentenceTransformer通用模型
        print("\n2. 测试通用向量化模型:")
        st_model = create_vectorizer_instance("sentence-transformer")
        
        if st_model.health_check():
            print("通用向量化模型健康检查通过")
            print(f"嵌入维度: {st_model.get_embedding_dimension()}")
            
            # 简单测试
            st_embedding = st_model.get_embedding("这是一个测试文本")
            print(f"通用模型嵌入向量长度: {len(st_embedding)}")
        else:
            print("通用向量化模型健康检查失败")
            
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
