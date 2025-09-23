#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩充数据集工具

功能说明：
  1. 从corpus.txt提取内容扩充chunks.jsonl文件
  2. 生成相应的嵌入向量扩充embeddings_256.npy文件
  3. 保持数据格式一致性

作者：Ph.D. Rhino
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

# 检查并安装依赖
def install_dependencies():
    """检查并自动安装所需依赖包"""
    required_packages = [
        'numpy', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装依赖: {package}")
            os.system(f"{sys.executable} -m pip install {package}")

# 安装依赖
install_dependencies()

class DatasetExpander:
    """数据集扩充工具类"""
    
    def __init__(self):
        """初始化数据集扩充工具"""
        # 文件路径
        self.corpus_path = 'docs/corpus.txt'
        self.chunks_path = 'data/chunks.jsonl'
        self.embeddings_path = 'data/embeddings_256.npy'
        
        # 配置参数
        self.target_size = 200  # 目标数据量
        self.embedding_dim = 49  # 嵌入维度
        self.overlap_range = (0.1, 0.3)  # overlap范围
        
        # 确保目录存在
        os.makedirs('data', exist_ok=True)
        
        print("数据集扩充工具初始化完成")
        print(f"目标数据量: {self.target_size}")
        print(f"嵌入维度: {self.embedding_dim}")
    
    def load_existing_data(self):
        """加载现有数据"""
        print("正在加载现有数据...")
        
        # 加载chunks.jsonl
        existing_chunks = []
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line.strip())
                        existing_chunks.append(chunk)
                    except json.JSONDecodeError:
                        continue
        
        print(f"已加载 {len(existing_chunks)} 个现有文本块")
        
        # 加载embeddings_256.npy
        existing_embeddings = None
        if os.path.exists(self.embeddings_path):
            try:
                existing_embeddings = np.load(self.embeddings_path)
                print(f"已加载 {len(existing_embeddings)} 个现有嵌入向量")
            except Exception as e:
                print(f"加载嵌入向量失败: {e}")
                # 重新生成
                existing_embeddings = None
        
        # 确保chunks和embeddings数量一致
        if existing_embeddings is not None and len(existing_embeddings) != len(existing_chunks):
            print(f"警告: chunks数量({len(existing_chunks)})与embeddings数量({len(existing_embeddings)})不一致")
            # 以chunks数量为准
            existing_embeddings = existing_embeddings[:len(existing_chunks)]
        
        return existing_chunks, existing_embeddings
    
    def extract_new_content(self, existing_chunks):
        """从corpus.txt提取新内容"""
        print("正在从corpus.txt提取新内容...")
        
        # 加载corpus.txt
        new_content = []
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 提取问答对
            for i in range(0, len(lines), 2):
                if i+1 < len(lines):
                    question = lines[i].strip()
                    answer = lines[i+1].strip()
                    if question and answer:
                        new_content.append(f"{question}\n{answer}")
        
        # 如果corpus.txt内容不够，就基于现有内容生成更多
        existing_texts = [chunk['text'] for chunk in existing_chunks]
        additional_content = []
        
        # 简单的文本增强技术：句子重排和组合
        while len(new_content) + len(additional_content) < (self.target_size - len(existing_chunks)):
            # 随机选择几个现有文本块
            sample_size = min(3, len(existing_texts))
            selected_texts = np.random.choice(existing_texts, sample_size, replace=False)
            # 组合这些文本块
            combined_text = '\n'.join(selected_texts)
            # 限制长度
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000] + '...'
            additional_content.append(combined_text)
        
        print(f"已提取 {len(new_content)} 条新内容，生成 {len(additional_content)} 条增强内容")
        
        return new_content + additional_content
    
    def generate_new_chunks(self, existing_chunks, new_content):
        """生成新的文本块"""
        print("正在生成新的文本块...")
        
        # 获取现有最大chunk_id
        existing_ids = []
        for chunk in existing_chunks:
            if 'chunk_id' in chunk and chunk['chunk_id']:
                try:
                    existing_ids.append(int(chunk['chunk_id']))
                except ValueError:
                    continue
        
        next_id = max(existing_ids) + 1 if existing_ids else 1
        
        # 生成新的chunks
        new_chunks = []
        for i, text in enumerate(tqdm(new_content, desc="生成文本块")):
            chunk_id = str(next_id + i).zfill(3)
            overlap = np.random.uniform(self.overlap_range[0], self.overlap_range[1])
            
            new_chunk = {
                'text': text,
                'chunk_id': chunk_id,
                'overlap': round(overlap, 2)
            }
            new_chunks.append(new_chunk)
        
        # 合并现有和新的chunks
        all_chunks = existing_chunks + new_chunks
        print(f"总共生成 {len(all_chunks)} 个文本块")
        
        return all_chunks
    
    def generate_new_embeddings(self, existing_embeddings, num_new, total_size):
        """生成新的嵌入向量"""
        print("正在生成新的嵌入向量...")
        
        # 生成随机嵌入向量（实际应用中应使用真实的嵌入模型）
        new_embeddings = np.random.rand(num_new, self.embedding_dim).astype('float32')
        
        # 归一化向量，与现有向量保持一致
        for i in range(num_new):
            norm = np.linalg.norm(new_embeddings[i])
            if norm > 0:
                new_embeddings[i] /= norm
        
        # 合并现有和新的嵌入向量
        if existing_embeddings is not None:
            all_embeddings = np.concatenate([existing_embeddings, new_embeddings])
        else:
            # 如果没有现有嵌入向量，就生成全部
            all_embeddings = np.random.rand(total_size, self.embedding_dim).astype('float32')
            # 归一化
            for i in range(total_size):
                norm = np.linalg.norm(all_embeddings[i])
                if norm > 0:
                    all_embeddings[i] /= norm
        
        # 确保数量与chunks一致
        all_embeddings = all_embeddings[:total_size]
        
        print(f"总共生成 {len(all_embeddings)} 个嵌入向量，维度: {self.embedding_dim}")
        
        return all_embeddings
    
    def save_data(self, chunks, embeddings):
        """保存数据"""
        print("正在保存数据...")
        
        # 保存chunks.jsonl
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"已保存 {len(chunks)} 个文本块到 {self.chunks_path}")
        
        # 保存embeddings_256.npy
        np.save(self.embeddings_path, embeddings)
        
        print(f"已保存 {len(embeddings)} 个嵌入向量到 {self.embeddings_path}")
        
        # 验证保存是否成功
        try:
            # 验证chunks
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                saved_chunks = [json.loads(line.strip()) for line in f]
            assert len(saved_chunks) == len(chunks), f"保存的chunks数量不匹配: {len(saved_chunks)} vs {len(chunks)}"
            
            # 验证embeddings
            saved_embeddings = np.load(self.embeddings_path)
            assert len(saved_embeddings) == len(embeddings), f"保存的embeddings数量不匹配: {len(saved_embeddings)} vs {len(embeddings)}"
            assert saved_embeddings.shape[1] == self.embedding_dim, f"保存的embeddings维度不匹配: {saved_embeddings.shape[1]} vs {self.embedding_dim}"
            
            print("✓ 数据保存验证成功")
        except Exception as e:
            print(f"✗ 数据保存验证失败: {e}")
    
    def expand(self):
        """执行数据集扩充"""
        print("===== 开始执行数据集扩充 =====")
        
        # 加载现有数据
        existing_chunks, existing_embeddings = self.load_existing_data()
        
        # 计算需要新增的数量
        num_existing = len(existing_chunks)
        if num_existing >= self.target_size:
            print(f"现有数据量({num_existing})已达到或超过目标数据量({self.target_size})，无需扩充")
            return False
        
        num_new = self.target_size - num_existing
        print(f"需要新增 {num_new} 条数据")
        
        # 提取新内容
        new_content = self.extract_new_content(existing_chunks)
        
        # 生成新的文本块
        all_chunks = self.generate_new_chunks(existing_chunks, new_content[:num_new])
        
        # 生成新的嵌入向量
        all_embeddings = self.generate_new_embeddings(existing_embeddings, num_new, len(all_chunks))
        
        # 保存数据
        self.save_data(all_chunks, all_embeddings)
        
        print("===== 数据集扩充完成 =====")
        return True

# 主函数
if __name__ == "__main__":
    try:
        # 创建数据集扩充工具实例
        expander = DatasetExpander()
        
        # 执行扩充
        success = expander.expand()
        
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)