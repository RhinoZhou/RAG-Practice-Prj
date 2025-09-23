#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语料准备脚本
负责处理原始文档，生成标准格式的语料库
"""

import os
import json
import argparse
import uuid
from typing import List, Dict, Any
from app.utils import clean_text, split_text
from app.config import config

class CorpusPreparer:
    """语料准备器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化语料准备器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """\读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                print(f"读取文件失败 {file_path}: {e}")
                return ""
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return ""
    
    def write_file(self, file_path: str, content: str):
        """写入文件内容"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def list_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        """列出目录下所有指定扩展名的文件"""
        if extensions is None:
            extensions = ['.md', '.txt']
        
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_paths.append(os.path.join(root, file))
        return file_paths
    
    def extract_metadata(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """从文件路径和内容中提取元数据"""
        # 提取文件名作为标题
        file_name = os.path.basename(file_path)
        title = os.path.splitext(file_name)[0]
        
        # 简单的年份提取（从文件名或路径中）
        year = ""
        year_match = re.search(r'(20\d{2})', file_path)
        if year_match:
            year = year_match.group(1)
        
        # 设置平台和区域
        platform = doc_type
        region = "cn"  # 默认中国
        
        return {
            "title": title,
            "source": file_path,
            "year": year,
            "platform": platform,
            "region": region
        }
    
    def prepare_corpus(self, chunk_size: int = 300, chunk_min: int = 200, chunk_max: int = 400) -> Dict[str, Any]:
        """
        准备语料库
        
        Args:
            chunk_size: 目标文本分块大小
            chunk_min: 最小块大小
            chunk_max: 最大块大小
        
        Returns:
            处理统计信息
        """
        stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "processed_docs": 0,
            "failed_docs": 0,
            "doc_types": {}
        }
        
        output_file = os.path.join(self.output_dir, "corpus_meta.jsonl")
        
        # 清理输出文件
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # 遍历数据目录下的不同文档类型
        doc_types = ["policy", "faq", "sop"]
        
        for doc_type in doc_types:
            doc_dir = os.path.join(self.data_dir, doc_type)
            if not os.path.exists(doc_dir):
                print(f"警告: 目录 {doc_dir} 不存在")
                continue
            
            # 初始化文档类型统计
            stats["doc_types"][doc_type] = {
                "docs": 0,
                "chunks": 0
            }
            
            # 获取文件列表
            files = self.list_files(doc_dir)
            stats["total_docs"] += len(files)
            stats["doc_types"][doc_type]["docs"] = len(files)
            
            # 处理每个文件
            for file_path in files:
                try:
                    # 读取文件内容
                    content = self.read_file(file_path)
                    if not content:
                        stats["failed_docs"] += 1
                        continue
                    
                    # 清理文本
                    content = clean_text(content)
                    
                    # 提取元数据
                    metadata = self.extract_metadata(file_path, doc_type)
                    
                    # 分块处理 (使用自定义的分块逻辑，确保块大小在200-400字之间)
                    chunks = []
                    start = 0
                    while start < len(content):
                        # 尝试在目标大小附近找一个合适的分割点
                        end = min(start + chunk_size, len(content))
                        
                        # 如果块太小，尝试扩展到最小大小
                        if end - start < chunk_min:
                            end = min(start + chunk_max, len(content))
                        
                        # 尽量在句子结束处分割（简化版）
                        if end < len(content):
                            # 查找最近的句号、问号、感叹号
                            punctuation_positions = [content.rfind(p, start, end) for p in ["。", "?", "!", "."]]
                            punctuation_positions = [pos for pos in punctuation_positions if pos > start + chunk_min]
                            
                            if punctuation_positions:
                                end = max(punctuation_positions) + 1
                        
                        chunks.append(content[start:end])
                        start = end
                    
                    # 写入JSONL文件
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for i, chunk in enumerate(chunks):
                            # 创建文档ID
                            doc_id = f"{doc_type}_{uuid.uuid4().hex[:8]}_{i}"
                            
                            # 构建文档对象
                            doc = {
                                "doc_id": doc_id,
                                "title": metadata["title"],
                                "source": metadata["source"],
                                "year": metadata["year"],
                                "platform": metadata["platform"],
                                "region": metadata["region"],
                                "content": chunk
                            }
                            
                            # 写入JSONL格式
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    
                    # 更新统计信息
                    stats["total_chunks"] += len(chunks)
                    stats["doc_types"][doc_type]["chunks"] += len(chunks)
                    stats["processed_docs"] += 1
                    
                    if stats["processed_docs"] % 10 == 0:
                        print(f"已处理 {stats['processed_docs']} 个文档，生成 {stats['total_chunks']} 个段落")
                    
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {e}")
                    stats["failed_docs"] += 1
        
        print(f"语料准备完成！统计信息：")
        print(f"- 总文档数: {stats['total_docs']}")
        print(f"- 成功处理: {stats['processed_docs']}")
        print(f"- 处理失败: {stats['failed_docs']}")
        print(f"- 生成段落数: {stats['total_chunks']}")
        for doc_type, type_stats in stats["doc_types"].items():
            print(f"  - {doc_type}: {type_stats['docs']} 文档, {type_stats['chunks']} 段落")
        
        return stats

# 简单的分词函数（用于BM25索引构建）
def simple_tokenize(text: str) -> List[str]:
    """
    简单的文本分词函数
    中文按字符分割，英文按空格分割
    
    Args:
        text: 输入文本
    
    Returns:
        分词后的词汇列表
    """
    # TODO: 这里可以替换为更复杂的分词库如jieba
    tokens = []
    temp_english = ""
    
    for char in text:
        # 检查是否为英文字符或数字
        if re.match(r'[a-zA-Z0-9]', char):
            temp_english += char
        else:
            if temp_english:
                tokens.append(temp_english.lower())
                temp_english = ""
            if char.strip():
                tokens.append(char)
    
    # 处理最后一个英文单词
    if temp_english:
        tokens.append(temp_english.lower())
    
    return tokens

import re

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语料准备脚本')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default=config.DATA_DIR,
                        help='输出目录')
    parser.add_argument('--chunk_size', type=int, default=300,
                        help='文本分块大小')
    
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建语料准备器
    preparer = CorpusPreparer(args.data_dir, args.output_dir)
    
    # 准备语料库
    stats = preparer.prepare_corpus(chunk_size=args.chunk_size)
    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, "corpus_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计信息已保存到 {stats_file}")

if __name__ == "__main__":
    main()
        for doc_type in ["policy", "faq", "sop"]:
            doc_dir = os.path.join(self.data_dir, doc_type)
            if not os.path.exists(doc_dir):
                print(f"警告: 目录 {doc_dir} 不存在，跳过...")
                continue
            
            # 列出所有文本文件
            text_files = list_files(doc_dir, [".txt", ".md", ".json"])
            stats["doc_types"][doc_type] = {
                "total": len(text_files),
                "processed": 0
            }
            
            # 处理每个文件
            for file_path in text_files:
                try:
                    # 读取文件内容
                    content = read_file(file_path)
                    if not content:
                        continue
                    
                    # 清理文本
                    cleaned_text = clean_text(content)
                    
                    # 分块处理
                    chunks = split_text(cleaned_text, chunk_size, overlap)
                    
                    # 生成文档ID
                    file_name = os.path.basename(file_path)
                    doc_id = file_name.split('.')[0]
                    
                    # 保存处理后的文档
                    processed_docs = []
                    for i, chunk in enumerate(chunks):
                        doc = {
                            "id": f"{doc_id}_{i}",
                            "text": chunk,
                            "metadata": {
                                "source": file_path,
                                "type": doc_type,
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        }
                        processed_docs.append(doc)
                    
                    # 保存到输出目录
                    output_file = os.path.join(self.output_dir, f"{doc_id}.json")
                    write_file(output_file, json.dumps(processed_docs, ensure_ascii=False, indent=2))
                    
                    # 更新统计信息
                    stats["total_docs"] += 1
                    stats["total_chunks"] += len(chunks)
                    stats["processed_docs"] += 1
                    stats["doc_types"][doc_type]["processed"] += 1
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")
                    stats["failed_docs"] += 1
        
        return stats
    
    def generate_manifest(self, stats: Dict[str, Any]) -> None:
        """
        生成语料库清单
        
        Args:
            stats: 统计信息
        """
        manifest = {
            "version": "1.0",
            "created_at": self._get_current_timestamp(),
            "corpus_dir": self.output_dir,
            "stats": stats
        }
        
        manifest_file = os.path.join(self.output_dir, "manifest.json")
        write_file(manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"语料库清单已保存到 {manifest_file}")
    
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
    parser = argparse.ArgumentParser(description="准备RAG系统语料库")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default=os.path.join(config.DATA_DIR, "processed"),
                        help="处理后的数据输出目录")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="文本分块大小")
    parser.add_argument("--overlap", type=int, default=50,
                        help="文本块重叠大小")
    
    args = parser.parse_args()
    
    # 创建语料准备器
    preparer = CorpusPreparer(args.data_dir, args.output_dir)
    
    # 准备语料库
    print(f"开始准备语料库，数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分块大小: {args.chunk_size}，重叠大小: {args.overlap}")
    
    stats = preparer.prepare_corpus(args.chunk_size, args.overlap)
    
    # 生成清单
    preparer.generate_manifest(stats)
    
    # 打印统计信息
    print("\n语料准备完成！")
    print(f"总文档数: {stats['total_docs']}")
    print(f"总块数: {stats['total_chunks']}")
    print(f"成功处理: {stats['processed_docs']}")
    print(f"处理失败: {stats['failed_docs']}")
    print("文档类型统计:")
    for doc_type, type_stats in stats['doc_types'].items():
        print(f"  {doc_type}: {type_stats['processed']}/{type_stats['total']}")

if __name__ == "__main__":
    main()

# TODO: 实现更多语料处理功能
# 1. 支持更多文件格式（PDF, DOCX等）
# 2. 自动识别文档类型
# 3. 文档元数据自动提取
# 4. 多线程/多进程处理