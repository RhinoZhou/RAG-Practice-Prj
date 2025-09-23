#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建向量索引脚本
负责创建和保存FAISS向量索引
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from app.config import config

class VectorIndexBuilder:
    """向量索引构建器"""
    
    def __init__(self, input_file: str, output_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量索引构建器
        
        Args:
            input_file: 输入的meta JSONL文件路径
            output_dir: 索引输出目录
            model_name: 使用的SentenceTransformer模型名称
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型
        print(f"加载SentenceTransformer模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 索引相关属性
        self.index = None
        self.vectors = []
        self.docs = []
    
    def load_corpus(self) -> Dict[str, Any]:
        """
        加载语料库
        
        Returns:
            加载统计信息
        """
        stats = {
            "total_docs": 0,
            "loaded_docs": 0,
            "failed_docs": 0
        }
        
        # 检查输入文件是否存在
        if not os.path.exists(self.input_file):
            print(f"错误: 输入文件 {self.input_file} 不存在")
            return stats
        
        print(f"开始读取语料库文件: {self.input_file}")
        
        # 读取meta JSONL文件
        self.docs = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    self.docs.append(doc)
                    stats["total_docs"] += 1
                    stats["loaded_docs"] += 1
                    
                    # 打印进度
                    if stats["total_docs"] % 100 == 0:
                        print(f"已处理 {stats['total_docs']} 个文档")
                        
                except Exception as e:
                    print(f"解析文档第{line_num}行失败: {e}")
                    stats["failed_docs"] += 1
        
        print(f"语料库加载完成！统计信息：")
        print(f"- 总文档数: {stats['total_docs']}")
        print(f"- 成功加载: {stats['loaded_docs']}")
        print(f"- 加载失败: {stats['failed_docs']}")
        
        return stats
    
    def encode_vectors(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        对文档进行向量编码
        
        Args:
            batch_size: 批量处理大小
        
        Returns:
            编码统计信息
        """
        stats = {
            "total_docs": len(self.docs),
            "encoded_docs": 0,
            "failed_docs": 0,
            "embedding_dim": 0
        }
        
        if not self.docs:
            print("错误: 没有加载任何文档，无法进行向量编码")
            return stats
        
        print(f"开始对 {len(self.docs)} 个文档进行向量编码")
        
        # 提取文本内容
        texts = []
        valid_indices = []
        
        for i, doc in enumerate(self.docs):
            content = doc.get("content", "")
            if content:
                texts.append(content)
                valid_indices.append(i)
            else:
                stats["failed_docs"] += 1
        
        # 批量编码
        self.vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                # 使用SentenceTransformer进行编码
                batch_vectors = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                self.vectors.extend(batch_vectors)
                stats["encoded_docs"] += len(batch_texts)
                
                # 更新嵌入维度
                if stats["embedding_dim"] == 0 and len(batch_vectors) > 0:
                    stats["embedding_dim"] = len(batch_vectors[0])
                    
                # 打印进度
                if stats["encoded_docs"] % 100 == 0:
                    progress = (stats["encoded_docs"] / len(texts)) * 100
                    print(f"向量编码进度: {progress:.1f}% ({stats['encoded_docs']}/{len(texts)})")
                    
            except Exception as e:
                print(f"编码批次失败 ({i//batch_size+1}): {e}")
                stats["failed_docs"] += len(batch_texts)
        
        # 将向量转换为numpy数组
        if self.vectors:
            self.vectors = np.array(self.vectors, dtype=np.float32)
        
        # 过滤出有效的文档（只保留成功编码的文档）
        valid_docs = [self.docs[i] for i in valid_indices if len(valid_indices) > i]
        self.docs = valid_docs[:len(self.vectors)]  # 确保文档和向量数量一致
        
        print(f"向量编码完成！统计信息：")
        print(f"- 总文档数: {stats['total_docs']}")
        print(f"- 成功编码: {stats['encoded_docs']}")
        print(f"- 编码失败: {stats['failed_docs']}")
        print(f"- 向量维度: {stats['embedding_dim']}")
        print(f"- 最终向量数量: {len(self.vectors)}")
        
        return stats
    
    def build_index(self) -> Dict[str, Any]:
        """
        构建FAISS向量索引
        
        Returns:
            索引构建统计信息
        """
        stats = {
            "vector_count": len(self.vectors),
            "index_type": "IndexFlatIP",
            "embedding_dim": 0,
            "status": "failed"
        }
        
        if not self.vectors.any():
            print("错误: 没有向量数据，无法构建索引")
            return stats
        
        # 获取向量维度
        embedding_dim = len(self.vectors[0])
        stats["embedding_dim"] = embedding_dim
        
        print(f"开始构建FAISS索引 (IndexFlatIP)，向量数: {len(self.vectors)}, 维度: {embedding_dim}")
        
        # 创建FAISS索引
        # 使用IndexFlatIP进行内积搜索（适合归一化向量）
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # 归一化向量以适应内积搜索
        faiss.normalize_L2(self.vectors)
        
        # 添加向量到索引
        self.index.add(self.vectors)
        
        stats["status"] = "success"
        print(f"FAISS索引构建完成！索引包含 {self.index.ntotal} 个向量")
        
        return stats
    
    def save_index(self) -> Dict[str, Any]:
        """
        保存索引和元数据
        
        Returns:
            保存统计信息
        """
        stats = {
            "index_file": "",
            "meta_file": "",
            "vector_count": len(self.vectors),
            "doc_count": len(self.docs),
            "status": "failed"
        }
        
        if self.index is None or not self.docs:
            print("错误: 索引或文档数据不存在，无法保存")
            return stats
        
        # 保存FAISS索引
        index_file = os.path.join(self.output_dir, "vector.faiss")
        try:
            faiss.write_index(self.index, index_file)
            stats["index_file"] = index_file
            print(f"FAISS索引已保存到: {index_file}")
        except Exception as e:
            print(f"保存FAISS索引失败: {e}")
            return stats
        
        # 保存元数据JSONL
        meta_file = os.path.join(self.output_dir, "meta.jsonl")
        try:
            with open(meta_file, 'w', encoding='utf-8') as f:
                for doc in self.docs:
                    # 确保doc是可序列化的
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            stats["meta_file"] = meta_file
            print(f"元数据已保存到: {meta_file}")
        except Exception as e:
            print(f"保存元数据失败: {e}")
            return stats
        
        # 保存索引配置信息
        config_file = os.path.join(self.output_dir, "index_config.json")
        config_data = {
            "model_name": self.model_name,
            "index_type": "IndexFlatIP",
            "vector_count": len(self.vectors),
            "embedding_dim": len(self.vectors[0]) if self.vectors.any() else 0,
            "created_at": str(os.path.getmtime(__file__))
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            print(f"索引配置已保存到: {config_file}")
        except Exception as e:
            print(f"保存索引配置失败: {e}")
        
        stats["status"] = "success"
        print(f"索引保存完成！")
        
        return stats
    
    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳字符串
        
        Returns:
            时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--corpus-dir", type=str, default=os.path.join(config.DATA_DIR, "processed"),
                        help="处理后的语料库目录")
    parser.add_argument("--index-dir", type=str, default=config.INDEX_DIR,
                        help="索引输出目录")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2",
                        help="用于生成嵌入的模型名称")
    parser.add_argument("--index-type", type=str, default="ivf",
                        help="索引类型 (flat, ivf, hnsw)")
    parser.add_argument("--n-list", type=int, default=100,
                        help="聚类数量（仅IVF索引使用）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批处理大小")
    
    args = parser.parse_args()
    
    # 创建索引构建器
    builder = VectorIndexBuilder(args.corpus_dir, args.index_dir, args.model_name)
    
    # 加载语料库
    print(f"开始加载语料库，目录: {args.corpus_dir}")
    start_time = time.time()
    
    stats = builder.load_corpus()
    load_time = time.time() - start_time
    
    print(f"语料库加载完成，耗时: {load_time:.2f}秒")
    print(f"总文件数: {stats['total_files']}")
    print(f"加载文档数: {stats['loaded_docs']}")
    print(f"跳过文件数: {stats['skipped_files']}")
    
    # 生成向量嵌入
    print(f"\n开始生成向量嵌入，使用模型: {args.model_name}")
    start_time = time.time()
    
    builder.generate_embeddings(args.batch_size)
    embedding_time = time.time() - start_time
    
    print(f"向量嵌入生成完成，耗时: {embedding_time:.2f}秒")
    
    # 构建索引
    print(f"\n开始构建{args.index_type.upper()}索引")
    start_time = time.time()
    
    builder.build_index(args.index_type, args.n_list)
    build_time = time.time() - start_time
    
    print(f"索引构建完成，耗时: {build_time:.2f}秒")
    
    # 保存索引
    print(f"\n开始保存索引，目录: {args.index_dir}")
    
    index_file, metadata_file = builder.save_index()
    
    print("\n向量索引构建流程已完成！")
    print(f"索引文件: {index_file}")
    print(f"元数据文件: {metadata_file}")
    print(f"总文档数: {len(builder.documents)}")
    print(f"总耗时: {load_time + embedding_time + build_time:.2f}秒")

if __name__ == "__main__":
    main()

# TODO: 实现更多索引构建功能
# 1. 支持向量压缩（PQ/SQ）
# 2. 多模态向量索引
# 3. 索引优化和参数调优
# 4. 分布式索引构建