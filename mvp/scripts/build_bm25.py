#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建BM25索引脚本
负责从预处理的语料库构建BM25索引
"""

import os
import json
import pickle
import argparse
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.config import config

class BM25IndexBuilder:
    """BM25索引构建器"""
    
    def __init__(self, input_file: str, output_dir: str):
        """
        初始化BM25索引构建器
        
        Args:
            input_file: 输入的meta JSONL文件路径
            output_dir: 索引输出目录
        """
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def simple_tokenize(self, text: str) -> List[str]:
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
    
    def build_index(self) -> Dict[str, Any]:
        """
        构建BM25索引
        
        Returns:
            索引构建统计信息
        """
        stats = {
            "total_docs": 0,
            "indexed_docs": 0,
            "failed_docs": 0
        }
        
        # 读取meta JSONL文件
        docs = []
        tokenized_corpus = []
        
        print(f"开始读取语料库文件: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            print(f"错误: 输入文件 {self.input_file} 不存在")
            return stats
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    docs.append(doc)
                    stats["total_docs"] += 1
                    
                    # 提取文本内容并进行分词
                    content = doc.get("content", "")
                    if content:
                        # 分词处理
                        tokens = self.simple_tokenize(content)
                        tokenized_corpus.append(tokens)
                        stats["indexed_docs"] += 1
                    else:
                        # 空文档，添加空的token列表以保持索引一致性
                        tokenized_corpus.append([])
                        stats["failed_docs"] += 1
                    
                    # 打印进度
                    if stats["total_docs"] % 100 == 0:
                        print(f"已处理 {stats['total_docs']} 个文档")
                        
                except Exception as e:
                    print(f"解析文档第{line_num}行失败: {e}")
                    stats["failed_docs"] += 1
                    # 添加空的token列表以保持索引一致性
                    tokenized_corpus.append([])
        
        # 构建BM25索引
        if tokenized_corpus:
            print(f"开始构建BM25索引，文档数: {len(tokenized_corpus)}")
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 准备保存的数据
            index_data = {
                "bm25": bm25,
                "docs": docs,
                "stats": stats,
                "version": "1.0",
                "build_time": str(os.path.getmtime(__file__))
            }
            
            # 保存索引
            output_file = os.path.join(self.output_dir, "bm25.pkl")
            print(f"保存BM25索引到: {output_file}")
            
            try:
                with open(output_file, 'wb') as f:
                    pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"BM25索引保存成功")
            except Exception as e:
                print(f"保存BM25索引失败: {e}")
        else:
            print("错误: 没有有效的文档用于构建索引")
        
        print(f"BM25索引构建完成！统计信息：")
        print(f"- 总文档数: {stats['total_docs']}")
        print(f"- 成功索引: {stats['indexed_docs']}")
        print(f"- 索引失败: {stats['failed_docs']}")
        
        return stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建BM25索引脚本')
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join(config.DATA_DIR, "corpus_meta.jsonl"),
                        help='输入的meta JSONL文件路径')
    parser.add_argument('--output_dir', type=str, default=config.INDEX_DIR,
                        help='索引输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建BM25索引构建器
    builder = BM25IndexBuilder(args.input_file, args.output_dir)
    
    # 构建索引
    stats = builder.build_index()
    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, "bm25_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计信息已保存到 {stats_file}")

if __name__ == "__main__":
    main()