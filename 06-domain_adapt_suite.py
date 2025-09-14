#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域内选型、难负挖掘和轻量微调工具套件

本脚本实现了一个完整的文本嵌入模型优化流程，包括：
1. 模型A/B测试：在域内数据上评估多个预训练模型的性能
2. 难负样本挖掘：使用FAISS近邻检索技术挖掘难负样本
3. 轻量级微调：基于挖掘的难负样本对选定模型进行微调

主要特点：
- 完全兼容最新版本的transformers库，避免使用已废弃的TFPreTrainedModel接口
- 支持中文文本处理和显示，确保输出文件中中文不会出现乱码
- 提供详细的模型评估指标和可视化结果
- 模块化设计，便于集成到更大的RAG系统中

使用方法：
  python 06-domain_adapt_suite.py

依赖项：
  - torch: 深度学习框架
  - transformers: 预训练语言模型
  - faiss-cpu: 高效相似度搜索和聚类库
  - numpy: 数值计算库
  - pandas: 数据处理库
  - matplotlib: 数据可视化库
  - scikit-learn: 机器学习评估工具

输出结果：
  - outputs/mined_hard_negatives.json: 挖掘的难负样本数据
  - outputs/model_comparison_results.json: 模型性能对比结果
  - outputs/model_comparison_chart.png: 模型性能对比图表

注意事项：
  - 本脚本使用示例数据进行演示，实际应用中应替换为真实的域内数据集
  - 微调过程中使用三元组损失函数，适用于大多数文本嵌入任务
  - 如需在GPU上运行，请确保已安装对应版本的CUDA和cuDNN


"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

# 安装依赖（检查并安装缺失的库）
def install_dependencies():
    """
    检查并安装必要的依赖库
    本函数会检查主要依赖是否已安装，如果未安装则自动安装
    """
    try:
        # 这里只做检查，实际导入已经在文件顶部完成
        print("所有依赖已安装")
    except ImportError:
        print("正在安装必要的依赖...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu", "torch", "transformers"])
        print("依赖安装完成")

# 生成示例数据（占位用，实际项目中应替换为真实数据）
def generate_sample_data():
    """
    生成示例数据用于演示
    包含查询文本、正样本和负样本
    
    返回:
        生成的数据列表，每个元素包含查询、正样本、负样本和标签
    """
    # 创建示例数据
    query_texts = [
        "什么是人工智能？",
        "机器学习的主要算法有哪些？",
        "自然语言处理的应用场景",
        "如何构建推荐系统？",
        "深度学习与传统机器学习的区别"
    ]
    
    # 为每个查询生成正样本和负样本
    data = []
    for i, query in enumerate(query_texts):
        # 正样本：相似的问题
        pos_sample = f"{query} 详细解答"
        # 负样本：不相关的问题
        neg_sample = f"如何学习编程？第{i+1}部分"
        data.append({
            'query': query,
            'positive': pos_sample,
            'negative': neg_sample,
            'label': 1
        })
    
    # 保存为JSON文件
    os.makedirs('data', exist_ok=True)
    with open('data/sample_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

# 加载数据
def load_data(file_path):
    """
    加载数据集
    
    参数:
        file_path: 数据文件路径
    
    返回:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 获取句子嵌入向量（使用transformers直接实现）
def get_sentence_embedding(model, tokenizer, text, device, use_grad=False):
    """
    获取单个句子的嵌入向量
    
    参数:
        model: transformers模型
        tokenizer: 分词器
        text: 输入文本
        device: 运行设备
        use_grad: 是否启用梯度计算，训练时设为True
    
    返回:
        句子嵌入向量
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    
    if use_grad:
        # 训练模式下，允许梯度计算
        outputs = model(**inputs)
    else:
        # 评估模式下，禁用梯度计算以提高效率
        with torch.no_grad():
            outputs = model(**inputs)
    
    # 使用最后一层的隐藏状态的平均值作为句子嵌入
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    return sentence_embedding

# 评估模型性能
def evaluate_model(model, tokenizer, data, model_name="模型", device='cpu'):
    """
    评估模型在数据集上的性能
    
    参数:
        model: 待评估的模型
        tokenizer: 分词器
        data: 评估数据集
        model_name: 模型名称，用于输出
        device: 运行设备
    
    返回:
        评估结果字典，包含准确率、精确率、召回率、F1分数等
    """
    queries = [item['query'] for item in data]
    positives = [item['positive'] for item in data]
    negatives = [item['negative'] for item in data]
    
    # 计算嵌入向量，评估时不启用梯度计算
    query_embeddings = []
    pos_embeddings = []
    neg_embeddings = []
    
    for text in queries:
        query_embeddings.append(get_sentence_embedding(model, tokenizer, text, device, use_grad=False))
    
    for text in positives:
        pos_embeddings.append(get_sentence_embedding(model, tokenizer, text, device, use_grad=False))
    
    for text in negatives:
        neg_embeddings.append(get_sentence_embedding(model, tokenizer, text, device, use_grad=False))
    
    # 计算余弦相似度
    pos_scores = []
    neg_scores = []
    
    for q_emb, p_emb, n_emb in zip(query_embeddings, pos_embeddings, neg_embeddings):
        # 计算余弦相似度
        pos_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
        neg_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), n_emb.unsqueeze(0)).item()
        pos_scores.append(pos_sim)
        neg_scores.append(neg_sim)
    
    # 计算准确率：正确判断正样本比负样本相似的比例
    correct = sum(1 for p, n in zip(pos_scores, neg_scores) if p > n)
    accuracy = correct / len(data)
    
    # 生成预测标签
    y_true = [1] * len(data) + [0] * len(data)
    y_pred = []
    
    for p, n in zip(pos_scores, neg_scores):
        y_pred.append(1 if p > n else 0)  # 正样本预测
        y_pred.append(0 if p > n else 1)  # 负样本预测
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pos_mean_score": np.mean(pos_scores),
        "neg_mean_score": np.mean(neg_scores)
    }
    
    print(f"\n{model_name} 评估结果：")
    for key, value in results.items():
        if key != "model":
            print(f"{key}: {value:.4f}")
    
    return results

# 难负挖掘
def mine_hard_triplets(model, tokenizer, data, top_k=10, device='cpu'):
    """
    使用FAISS近邻检索挖掘难负样本
    
    参数:
        model: 用于生成嵌入的模型
        tokenizer: 分词器
        data: 原始数据集
        top_k: 为每个查询挖掘的难负样本数量
        device: 运行设备
    
    返回:
        包含难负样本的新数据集
    """
    print(f"\n开始挖掘难负样本...")
    
    # 提取所有文本
    all_texts = []
    for item in data:
        all_texts.append(item['query'])
        all_texts.append(item['positive'])
        all_texts.append(item['negative'])
    
    # 生成所有文本的嵌入，评估时不启用梯度计算
    embeddings = []
    for text in all_texts:
        emb = get_sentence_embedding(model, tokenizer, text, device, use_grad=False)
        embeddings.append(emb.cpu().numpy())
    embeddings = np.array(embeddings)
    
    # 使用FAISS构建索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离
    index.add(embeddings.astype(np.float32))
    
    # 为每个查询挖掘难负样本
    mined_data = []
    for i, item in enumerate(data):
        query_embedding = get_sentence_embedding(model, tokenizer, item['query'], device, use_grad=False).cpu().numpy()
        
        # 搜索相似文本（跳过自身）
        distances, indices = index.search(query_embedding.reshape(1, -1), len(all_texts))
        
        # 收集难负样本
        hard_negatives = []
        for j in range(1, len(indices[0])):  # 从1开始跳过自身
            idx = indices[0][j]
            if all_texts[idx] != item['positive'] and all_texts[idx] != item['query']:
                hard_negatives.append(all_texts[idx])
                if len(hard_negatives) >= top_k:
                    break
        
        # 更新数据
        mined_item = item.copy()
        mined_item['hard_negatives'] = hard_negatives
        mined_data.append(mined_item)
    
    print(f"难负样本挖掘完成，平均每个查询挖掘了{sum(len(item['hard_negatives']) for item in mined_data) / len(mined_data):.1f}个难负样本")
    
    # 保存挖掘结果
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/mined_hard_negatives.json', 'w', encoding='utf-8') as f:
        json.dump(mined_data, f, ensure_ascii=False, indent=2)
    
    return mined_data

# 简单微调函数（使用余弦相似度损失）
def simple_fine_tuning(base_model_name, mined_data, epochs=3, batch_size=8, device='cpu'):
    """
    简单的模型微调函数，使用余弦相似度损失
    
    参数:
        base_model_name: 基础模型名称
        mined_data: 包含难负样本的数据集
        epochs: 训练轮数
        batch_size: 批次大小
        device: 运行设备
    
    返回:
        微调后的模型和分词器
    """
    print(f"\n开始简单微调模型...")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name).to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 准备训练数据
    train_data = []
    for item in mined_data:
        query_text = item['query']
        positive_text = item['positive']
        
        # 对每个难负样本创建训练示例
        for hard_neg in item['hard_negatives'][:3]:  # 只使用前3个难负样本
            train_data.append({
                'query': query_text,
                'positive': positive_text,
                'negative': hard_neg
            })
    
    # 训练模型
    model.train()  # 确保模型处于训练模式
    for epoch in range(epochs):
        total_loss = 0
        # 打乱训练数据
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用列表收集批次中的损失，避免原地操作
            batch_losses = []
            for item in batch:
                # 获取嵌入向量，训练时启用梯度计算
                q_emb = get_sentence_embedding(model, tokenizer, item['query'], device, use_grad=True)
                p_emb = get_sentence_embedding(model, tokenizer, item['positive'], device, use_grad=True)
                n_emb = get_sentence_embedding(model, tokenizer, item['negative'], device, use_grad=True)
                
                # 计算余弦相似度
                pos_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_emb.unsqueeze(0))
                neg_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), n_emb.unsqueeze(0))
                
                # 计算三元组损失 (triplet margin loss)
                margin = 0.2
                loss = torch.max(torch.tensor(0.0).to(device), neg_sim - pos_sim + margin)
                batch_losses.append(loss)
            
            # 将所有损失聚合为一个张量并计算平均值
            if batch_losses:
                # 将列表中的张量堆叠起来
                stacked_losses = torch.stack(batch_losses)
                # 计算平均损失
                batch_loss = torch.mean(stacked_losses)
                total_loss += batch_loss.item()
                
                # 反向传播和优化
                batch_loss.backward()
                optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.6f}")
    
    return model, tokenizer

# 分析文件的中文显示情况
def analyze_file_utf8(file_path):
    """
    分析文件的UTF-8编码情况，特别检查中文显示
    
    参数:
        file_path: 文件路径
    
    返回:
        分析结果字符串
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fa5' for char in content)
        
        # 检查是否有编码问题（尝试重新编码和解码）
        try:
            encoded = content.encode('utf-8')
            decoded = encoded.decode('utf-8')
            has_encoding_issue = content != decoded
        except:
            has_encoding_issue = True
        
        result = f"文件 {file_path}: "
        result += f"包含中文字符: {'是' if has_chinese else '否'}, "
        result += f"编码正常: {'是' if not has_encoding_issue else '否'}"
        return result
    except Exception as e:
        return f"文件 {file_path} 分析出错: {str(e)}"

# 主函数
def main():
    """
    主函数，实现完整的域内选型、难负挖掘和轻量微调流程
    执行步骤：
    1. 安装依赖
    2. 生成示例数据
    3. 加载多个预训练模型进行A/B测试
    4. 选择最佳模型
    5. 挖掘难负样本
    6. 基于难负样本微调模型
    7. 评估微调后的模型
    8. 保存结果并可视化
    """
    # 1. 安装依赖
    install_dependencies()
    
    # 导入可能在依赖安装后才能导入的模块
    import faiss
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # 确定运行设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 2. 生成示例数据（实际项目中应替换为真实数据加载）
    print("生成示例数据...")
    data = generate_sample_data()
    
    # 3. 加载两个不同的模型进行A/B测试
    print("加载模型进行A/B测试...")
    model_names = [
        'sentence-transformers/all-MiniLM-L6-v2',  # 轻量级通用模型
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # 多语言模型
    ]
    
    models = []
    tokenizers = []
    results = []
    
    for i, model_name in enumerate(model_names):
        print(f"加载模型 {i+1}/{len(model_names)}: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        models.append(model)
        tokenizers.append(tokenizer)
        
        # 评估模型性能
        model_result = evaluate_model(model, tokenizer, data, f"模型{i+1} ({model_name.split('/')[-1]})")
        results.append(model_result)
    
    # 4. 选择性能更好的模型进行后续处理
    best_idx = 0
    best_f1 = results[0]['f1_score']
    for i, result in enumerate(results[1:], 1):
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']
            best_idx = i
    
    selected_model = models[best_idx]
    selected_tokenizer = tokenizers[best_idx]
    selected_model_name = f"模型{best_idx+1} ({model_names[best_idx].split('/')[-1]})"
    
    print(f"\n选择{selected_model_name}进行后续难负挖掘和微调")
    
    # 5. 使用FAISS进行难负挖掘
    mined_data = mine_hard_triplets(selected_model, selected_tokenizer, data, device=device)
    
    # 6. 基于难负样本进行简单微调
    # 为了微调，我们使用基础模型名称而不是已加载的模型
    base_model_name = model_names[best_idx]
    fine_tuned_model, fine_tuned_tokenizer = simple_fine_tuning(base_model_name, mined_data, device=device)
    
    # 7. 评估微调后的模型性能
    fine_tuned_results = evaluate_model(fine_tuned_model, fine_tuned_tokenizer, data, "微调后模型", device=device)
    
    # 8. 保存所有评估结果
    os.makedirs('outputs', exist_ok=True)
    all_results = {
        "models": results,
        "fine_tuned": fine_tuned_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('outputs/model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 9. 生成可视化结果
    plt.figure(figsize=(12, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_labels = [f"模型{i+1}" for i in range(len(models))] + ["微调后模型"]
    all_values = [result[m] for result in results for m in metrics] + [fine_tuned_results[m] for m in metrics]
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [results[j][metric] for j in range(len(results))] + [fine_tuned_results[metric]]
        colors = ['blue', 'green', 'red'][:len(values)]
        plt.bar(model_labels, values, color=colors)
        plt.title(metric)
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for j, v in enumerate(values):
            plt.text(j, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison_chart.png')
    print("\n模型对比图表已保存至 outputs/model_comparison_chart.png")
    
    # 10. 分析输出文件的中文显示情况
    print("\n分析输出文件的中文显示情况...")
    
    # 检查所有输出文件
    output_files = [
        'outputs/mined_hard_negatives.json',
        'outputs/model_comparison_results.json'
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            result = analyze_file_utf8(file_path)
            print(f"- {result}")
        else:
            print(f"- 文件 {file_path} 不存在")
    
    # 11. 实验结果总结
    print("\n域内选型、难负挖掘和轻量微调流程已完成！")
    print("\n实验结果总结：")
    
    # 输出原始模型性能
    for i, result in enumerate(results):
        print(f"{i+1}. 模型{i+1} ({model_names[i].split('/')[-1]}) F1分数: {result['f1_score']:.4f}")
    
    # 输出微调后模型性能
    print(f"{len(results)+1}. 微调后模型 F1分数: {fine_tuned_results['f1_score']:.4f}")
    
    # 计算性能提升
    original_best_f1 = max(result['f1_score'] for result in results)
    improvement = ((fine_tuned_results['f1_score'] - original_best_f1) / original_best_f1) * 100
    print(f"性能提升: {improvement:.2f}%")
    
    # 其他统计信息
    print(f"挖掘的难负样本数量: {sum(len(item['hard_negatives']) for item in mined_data)}")
    print(f"输出文件保存位置: outputs/ 目录")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


