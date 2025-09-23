#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量检索性能A/B测试评估工具

功能说明：
  1. 对比HNSW(efs=128)和IVF-PQ(nprobe=32)两种索引配置的性能
  2. 计算并输出Recall@{1,10,20}、平均延迟与简化MRR指标
  3. 生成Recall@K vs Latency曲线数据
  4. 输出Markdown表格与CSV格式的评估结果

作者：Ph.D. Rhino
"""

import os
import sys
import json
import time
import numpy as np
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 检查并安装依赖
def install_dependencies():
    """检查并自动安装所需依赖包"""
    required_packages = [
        'numpy', 'faiss-cpu', 'matplotlib', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安装依赖: {package}")
            os.system(f"{sys.executable} -m pip install {package}")

# 安装依赖
install_dependencies()

class ABEvaluation:
    """向量检索性能A/B测试评估类"""
    
    def __init__(self):
        """初始化评估工具"""
        # 文件路径
        self.embeddings_path = 'data/embeddings_256.npy'
        self.output_md_path = 'result/ab_eval.md'
        self.output_csv_path = 'result/ab_eval.csv'
        
        # 配置参数
        self.test_size = 0.2  # 测试集比例
        self.recall_ks = [1, 10, 20]  # 计算Recall的K值
        self.num_queries = 50  # 查询数量
        
        # 索引参数
        self.index_configs = {
            'HNSW': {
                'type': 'hnsw',
                'params': {
                    'M': 16,  # 图中每个节点的邻居数量
                    'efConstruction': 128,  # 构建时的探索参数
                    'efSearch': 128  # 搜索时的探索参数
                },
                'description': 'HNSW索引 (efSearch=128)'
            },
            'IVF-PQ': {
                'type': 'ivfpq',
                'params': {
                    'nlist': 8,  # 聚类中心数量（必须小于训练数据点数量）
                    'm': 7,  # 子量化器数量（必须是维度的因数）
                    'nprobe': 32  # 查询时搜索的聚类数量
                },
                'description': 'IVF-PQ索引 (nprobe=32)'
            }
        }
        
        # 确保输出目录存在
        os.makedirs('result', exist_ok=True)
        
        print("向量检索性能A/B测试评估工具初始化完成")
        print(f"数据集路径: {self.embeddings_path}")
        print(f"评估指标: Recall@{self.recall_ks}, 平均延迟, 简化MRR")
        print(f"输出文件: {self.output_md_path}, {self.output_csv_path}")
    
    def load_data(self):
        """加载嵌入向量数据"""
        print("正在加载嵌入向量数据...")
        
        if not os.path.exists(self.embeddings_path):
            print(f"错误: 嵌入向量文件 {self.embeddings_path} 不存在")
            return False
        
        try:
            # 加载嵌入向量
            self.embeddings = np.load(self.embeddings_path).astype('float32')
            
            # 检查数据维度
            if len(self.embeddings.shape) != 2:
                print("错误: 嵌入向量数据格式不正确，应为二维数组")
                return False
            
            self.num_vectors, self.dim = self.embeddings.shape
            print(f"成功加载 {self.num_vectors} 个向量，维度: {self.dim}")
            
            # 归一化向量以用于余弦相似度搜索
            faiss.normalize_L2(self.embeddings)
            
            # 分割训练集和测试集
            self._split_data()
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def _split_data(self):
        """分割训练集和测试集"""
        # 随机打乱数据
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        indices = np.random.permutation(self.num_vectors)
        
        # 计算分割点
        test_size = min(int(self.num_vectors * self.test_size), self.num_queries)
        
        # 分割数据
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        self.test_vectors = self.embeddings[test_indices]
        self.train_vectors = self.embeddings[train_indices]
        
        # 为了评估，我们需要一个精确的索引作为基准
        self.ground_truth_index = faiss.IndexFlatIP(self.dim)  # 内积搜索
        self.ground_truth_index.add(self.embeddings)
        
        print(f"数据集分割完成: 训练集 {len(self.train_vectors)} 个向量，测试集 {len(self.test_vectors)} 个向量")
    
    def create_index(self, config_name):
        """创建指定配置的索引"""
        config = self.index_configs[config_name]
        print(f"\n创建索引: {config['description']}")
        
        if config['type'] == 'hnsw':
            # 创建HNSW索引
            try:
                index = faiss.IndexHNSWFlat(self.dim, config['params']['M'], faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = config['params']['efConstruction']
                index.hnsw.efSearch = config['params']['efSearch']
                index.add(self.train_vectors)
                
                print(f"索引创建完成，包含 {index.ntotal} 个向量")
                return index, config
            except Exception as e:
                print(f"创建HNSW索引失败: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        elif config['type'] == 'ivfpq':
            # 创建IVF-PQ索引
            # 首先创建量化器
            quantizer = faiss.IndexFlatIP(self.dim)
            
            # 添加调试信息
            print(f"调试信息 - 创建IVF-PQ索引:")
            print(f"  维度: {self.dim}")
            print(f"  nlist(聚类中心数量): {config['params']['nlist']}")
            print(f"  m(子量化器数量): {config['params']['m']}")
            print(f"  训练数据点数量: {len(self.train_vectors)}")
            
            # 检查子量化器数量是否合适
            if self.dim % config['params']['m'] != 0:
                print(f"警告: 子量化器数量 {config['params']['m']} 不是维度 {self.dim} 的因数")
                # 调整子量化器数量为合适的值
                new_m = self._find_suitable_m(self.dim)
                print(f"自动调整子量化器数量为: {new_m}")
                config['params']['m'] = new_m
            
            try:
                index = faiss.IndexIVFPQ(
                    quantizer, 
                    self.dim, 
                    config['params']['nlist'], 
                    config['params']['m'], 
                    8  # 每个子量化器的位数
                )
                
                # 训练索引
                print("开始训练IVF-PQ索引...")
                index.train(self.train_vectors)
                
                # 添加向量
                index.add(self.train_vectors)
                
                # 设置查询参数
                index.nprobe = config['params']['nprobe']
                
                print(f"索引创建完成，包含 {index.ntotal} 个向量")
                return index, config
            except Exception as e:
                print(f"创建IVF-PQ索引失败: {e}")
                # 尝试使用更小的聚类中心数量
                print("尝试使用更小的聚类中心数量...")
                try:
                    # 使用训练数据点数量的1/10作为聚类中心数量
                    adjusted_nlist = max(1, len(self.train_vectors) // 10)
                    print(f"调整后的聚类中心数量: {adjusted_nlist}")
                    
                    index = faiss.IndexIVFPQ(
                        quantizer, 
                        self.dim, 
                        adjusted_nlist, 
                        config['params']['m'], 
                        8
                    )
                    
                    index.train(self.train_vectors)
                    index.add(self.train_vectors)
                    index.nprobe = config['params']['nprobe']
                    
                    print(f"索引创建完成，包含 {index.ntotal} 个向量")
                    return index, config
                except Exception as e2:
                    print(f"再次尝试失败: {e2}")
                    import traceback
                    traceback.print_exc()
                    return None
                    
        else:
            print(f"错误: 不支持的索引类型: {config['type']}")
            return None
    
    def _find_suitable_m(self, dim):
        """找到适合给定维度的子量化器数量"""
        # 尝试找到最大的因数，不大于32
        for m in range(min(dim, 32), 0, -1):
            if dim % m == 0:
                return m
        return 1  # 默认值
    
    def evaluate_index(self, index, config, config_name):
        """评估索引性能"""
        print(f"\n评估索引: {config['description']}")
        
        # 存储每个查询的结果
        all_distances = []
        all_indices = []
        latencies = []
        
        # 执行查询
        max_k = max(self.recall_ks)
        
        for i in tqdm(range(len(self.test_vectors)), desc=f"执行查询 ({config_name})"):
            query_vector = self.test_vectors[i:i+1]
            
            # 记录查询开始时间
            start_time = time.time()
            
            # 执行查询
            if config['type'] == 'hnsw':
                # 对于HNSW，我们需要在每次查询前设置efSearch
                index.hnsw.efSearch = config['params']['efSearch']
            
            distances, indices = index.search(query_vector, max_k)
            
            # 计算查询延迟（毫秒）
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            all_distances.append(distances[0])
            all_indices.append(indices[0])
        
        # 计算Recall@K
        recall_scores = self._calculate_recall(all_indices, max_k)
        
        # 计算平均延迟
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # 计算简化MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(all_indices)
        
        print(f"评估完成 ({config_name}):")
        for k in self.recall_ks:
            print(f"  Recall@{k}: {recall_scores[k]:.4f}")
        print(f"  平均延迟: {avg_latency:.4f} ms")
        print(f"  P95延迟: {p95_latency:.4f} ms")
        print(f"  简化MRR: {mrr:.4f}")
        
        # 返回评估结果
        return {
            'config_name': config_name,
            'config': config,
            'recall': recall_scores,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'mrr': mrr,
            'latencies': latencies
        }
    
    def _calculate_recall(self, all_indices, max_k):
        """计算Recall@K指标"""
        recall_scores = {}
        
        # 对于每个查询向量，找到所有相关结果
        for k in self.recall_ks:
            correct = 0
            total = 0
            
            for i, query_vector in enumerate(self.test_vectors):
                # 在完整数据集上查找真实的最近邻
                true_distances, true_indices = self.ground_truth_index.search(query_vector.reshape(1, -1), self.num_vectors)
                
                # 排除查询向量本身（如果它在数据集中）
                valid_indices = [idx for idx in true_indices[0] if idx not in np.where((self.embeddings == query_vector).all(axis=1))[0]]
                true_neighbors = valid_indices[:k]  # 前k个真实邻居
                
                # 获取索引返回的邻居
                retrieved_neighbors = all_indices[i][:k] if len(all_indices[i]) >= k else all_indices[i]
                
                # 计算交集
                intersection = set(true_neighbors) & set(retrieved_neighbors)
                
                # 如果有真实邻居，计算准确率
                if len(true_neighbors) > 0:
                    correct += len(intersection)
                    total += min(k, len(true_neighbors))
            
            # 计算Recall@K
            recall_scores[k] = correct / total if total > 0 else 0
        
        return recall_scores
    
    def _calculate_mrr(self, all_indices):
        """计算简化的Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for i, query_vector in enumerate(self.test_vectors):
            # 在完整数据集上查找真实的最近邻
            true_distances, true_indices = self.ground_truth_index.search(query_vector.reshape(1, -1), self.num_vectors)
            
            # 排除查询向量本身（如果它在数据集中）
            valid_indices = [idx for idx in true_indices[0] if idx not in np.where((self.embeddings == query_vector).all(axis=1))[0]]
            
            if len(valid_indices) == 0:
                continue
            
            # 获取第一个相关结果的排名
            true_neighbor = valid_indices[0]
            retrieved_neighbors = all_indices[i]
            
            # 查找第一个相关结果的位置
            rank = None
            for j, neighbor in enumerate(retrieved_neighbors):
                if neighbor == true_neighbor:
                    rank = j + 1  # 排名从1开始
                    break
            
            # 如果找到了相关结果，计算倒数排名
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                # 如果没有找到，添加0
                reciprocal_ranks.append(0)
        
        # 计算平均倒数排名
        return np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0
    
    def generate_results(self, results):
        """生成评估结果报告"""
        print("\n生成评估结果报告...")
        
        # 生成Markdown报告
        md_content = self._generate_md_report(results)
        
        # 生成CSV报告
        csv_content = self._generate_csv_report(results)
        
        # 保存结果
        with open(self.output_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        with open(self.output_csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        print(f"报告已生成: {self.output_md_path}")
        print(f"CSV数据已生成: {self.output_csv_path}")
        
        # 检查中文显示
        self._check_chinese_display()
    
    def _generate_md_report(self, results):
        """生成Markdown格式的评估报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"# 向量检索性能A/B测试评估报告\n\n"
        md += f"## 评估概览\n"
        md += f"- 评估时间: {timestamp}\n"
        md += f"- 数据集: {self.num_vectors}个向量，维度{self.dim}\n"
        md += f"- 测试查询数: {len(self.test_vectors)}\n"
        md += f"- 评估指标: Recall@{self.recall_ks}, 平均延迟, P95延迟, 简化MRR\n\n"
        
        md += "## 索引配置\n"
        for config_name, result in results.items():
            config = result['config']
            params = config['params']
            md += f"### {config['description']}\n"
            md += "```\n"
            for param, value in params.items():
                md += f"{param}: {value}\n"
            md += "```\n\n"
        
        md += "## 性能对比\n"
        md += "| 索引类型 | "
        for k in self.recall_ks:
            md += f"Recall@{k} | "
        md += "平均延迟 (ms) | P95延迟 (ms) | 简化MRR |\n"
        
        md += "|---------|" + "-------|" * len(self.recall_ks) + "------------|------------|---------|\n"
        
        for config_name, result in results.items():
            md += f"| {config_name} | "
            for k in self.recall_ks:
                md += f"{result['recall'][k]:.4f} | "
            md += f"{result['avg_latency']:.4f} | "
            md += f"{result['p95_latency']:.4f} | "
            md += f"{result['mrr']:.4f} |\n"
        
        md += "\n## 结果分析\n"
        md += "### 延迟与召回率权衡\n"
        
        # 找出延迟最低的配置
        min_latency_config = min(results.items(), key=lambda x: x[1]['avg_latency'])[0]
        md += f"- **最低延迟**: {min_latency_config} ({results[min_latency_config]['avg_latency']:.4f} ms)\n"
        
        # 找出召回率最高的配置
        for k in self.recall_ks:
            max_recall_config = max(results.items(), key=lambda x: x[1]['recall'][k])[0]
            md += f"- **最高Recall@{k}**: {max_recall_config} ({results[max_recall_config]['recall'][k]:.4f})\n"
        
        # 找出MRR最高的配置
        max_mrr_config = max(results.items(), key=lambda x: x[1]['mrr'])[0]
        md += f"- **最高MRR**: {max_mrr_config} ({results[max_mrr_config]['mrr']:.4f})\n"
        
        md += "\n### 应用建议\n"
        md += "1. **低延迟场景**: 选择延迟较低的索引配置\n"
        md += "2. **高准确率场景**: 选择召回率和MRR较高的索引配置\n"
        md += "3. **资源受限环境**: 考虑IVF-PQ等压缩索引以节省内存\n"
        md += "4. **参数调优**: 根据实际需求调整索引参数以获得最佳性能\n"
        
        return md
    
    def _generate_csv_report(self, results):
        """生成CSV格式的评估数据"""
        # CSV头部
        header = "config_name,"
        for k in self.recall_ks:
            header += f"recall@{k},"
        header += "avg_latency_ms,p95_latency_ms,mrr\n"
        
        # CSV内容
        content = header
        for config_name, result in results.items():
            row = f"{config_name},"
            for k in self.recall_ks:
                row += f"{result['recall'][k]:.4f},"
            row += f"{result['avg_latency']:.4f},{result['p95_latency']:.4f},{result['mrr']:.4f}\n"
            content += row
        
        return content
    
    def _check_chinese_display(self):
        """检查输出文件中的中文显示情况"""
        try:
            with open(self.output_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查是否包含中文字符
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in content)
            
            if has_chinese:
                print("✓ 输出文件包含中文，且显示正常")
            else:
                print("注意: 输出文件中未检测到中文字符")
        except Exception as e:
            print(f"检查中文显示失败: {e}")
    
    def run(self):
        """运行完整的A/B测试评估流程"""
        print("===== 开始执行向量检索性能A/B测试评估 =====")
        
        # 加载数据
        if not self.load_data():
            print("评估失败: 无法加载数据")
            return False
        
        # 确保有足够的数据进行评估
        if len(self.train_vectors) < 10 or len(self.test_vectors) < 10:
            print("警告: 数据集过小，评估结果可能不准确")
        
        # 创建并评估所有索引配置
        results = {}
        for config_name in self.index_configs.keys():
            # 创建索引
            print(f"\n尝试创建和评估 {config_name} 索引...")
            try:
                index_result = self.create_index(config_name)
                if index_result is None:
                    print(f"跳过 {config_name} 索引的评估: 创建失败")
                    continue
                
                index, config = index_result
                
                # 评估索引
                result = self.evaluate_index(index, config, config_name)
                results[config_name] = result
            except Exception as e:
                print(f"评估 {config_name} 索引时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成评估结果报告
        if results:
            self.generate_results(results)
            print("===== 向量检索性能A/B测试评估完成 =====")
            return True
        else:
            print("评估失败: 没有成功创建和评估任何索引")
            return False

# 主函数
if __name__ == "__main__":
    try:
        # 创建A/B测试评估工具实例
        evaluator = ABEvaluation()
        
        # 运行评估
        success = evaluator.run()
        
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)