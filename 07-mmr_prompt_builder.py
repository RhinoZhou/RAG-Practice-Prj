#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MMR选择与RAG端到端链路构建器

该脚本实现了检索增强生成(RAG)系统中的关键组件：
1. 对检索到的候选文档进行最大边际相关性(MMR)选择，平衡相关性与多样性
2. 实现完整的RAG端到端链路，包括文本去重、内容拼接和Token窗口优化
3. 基于配置的提示模板生成最终的提示文本

作者：Ph.D. Rhino

使用方法：
    python 07-mmr_prompt_builder.py

输入文件：
    data/query.json - 包含查询文本和向量
    data/candidates.jsonl - 包含候选文档ID、向量和文本
    configs/prompt.json - 包含提示模板和配置参数

输出文件：
    outputs/prompt.txt - 生成的完整提示文本
    outputs/selected_ids.json - 选择的文档ID列表

依赖项：
    numpy - 用于向量计算和余弦相似度计算
    tqdm - 用于显示进度条

功能特点：
    - 自动检查和安装依赖
    - 实现MMR算法选择文档
    - 文本去重和Token估算优化
    - 详细的日志输出和执行时间统计
"""

import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm


def install_dependencies():
    """检查并安装必要的依赖包

    自动检测numpy和tqdm是否已安装，如未安装则通过pip安装
    """
    try:
        import numpy
        import tqdm
        print("所有必要的依赖已安装。")
    except ImportError:
        print("正在安装必要的依赖...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "tqdm"])
            print("依赖安装成功。")
        except Exception as e:
            print(f"依赖安装失败: {e}")
            sys.exit(1)


def load_json_file(file_path):
    """加载JSON文件内容

    Args:
        file_path: JSON文件路径

    Returns:
        文件内容的Python对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        sys.exit(1)


def load_jsonl_file(file_path):
    """加载JSONL文件内容

    Args:
        file_path: JSONL文件路径

    Returns:
        包含所有行内容的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"加载 {os.path.basename(file_path)}"):
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        sys.exit(1)


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度

    Args:
        vec1: 第一个向量
        vec2: 第二个向量

    Returns:
        余弦相似度值
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算向量范数
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 避免除以零的情况
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    # 计算点积并除以范数乘积
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)


def mmr_selection(query_vector, candidates, lambda_param=0.5, top_k=5):
    """使用最大边际相关性(MMR)算法选择候选文档

    Args:
        query_vector: 查询向量
        candidates: 候选文档列表，每个文档包含id、vector和text
        lambda_param: 相关性和多样性的权衡参数，范围0-1
        top_k: 选择的文档数量

    Returns:
        选择的文档列表和选择的文档ID列表
    """
    # 存储已选择的文档
    selected = []
    selected_ids = []
    # 存储未选择的文档
    remaining = candidates.copy()
    
    # 计算每个候选文档与查询的相关性分数
    for doc in tqdm(candidates, desc="计算文档-查询相关性"):
        doc['relevance_score'] = cosine_similarity(query_vector, doc['vector'])
    
    # MMR选择过程
    for i in range(min(top_k, len(candidates))):
        if not remaining:
            break
            
        # 第一个文档选择与查询最相关的
        if i == 0:
            best_doc = max(remaining, key=lambda x: x['relevance_score'])
            selected.append(best_doc)
            selected_ids.append(best_doc['id'])
            remaining.remove(best_doc)
        else:
            # 对剩余文档计算MMR分数
            best_score = -1
            best_doc = None
            
            for doc in tqdm(remaining, desc=f"MMR选择第{i+1}个文档"):
                # 计算文档与查询的相关性
                relevance = doc['relevance_score']
                
                # 计算文档与已选文档的最大相似度
                max_similarity = 0
                for s_doc in selected:
                    sim = cosine_similarity(doc['vector'], s_doc['vector'])
                    if sim > max_similarity:
                        max_similarity = sim
                
                # 计算MMR分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            
            # 选择MMR分数最高的文档
            if best_doc:
                selected.append(best_doc)
                selected_ids.append(best_doc['id'])
                remaining.remove(best_doc)
    
    return selected, selected_ids


def deduplicate_texts(texts):
    """对文本列表进行去重

    Args:
        texts: 文本列表

    Returns:
        去重后的文本列表
    """
    seen = set()
    unique_texts = []
    
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    return unique_texts


def estimate_tokens(text, chars_per_token=2.5):
    """估算文本的Token数量

    Args:
        text: 输入文本
        chars_per_token: 每个Token平均包含的字符数

    Returns:
        估算的Token数量
    """
    return len(text) / chars_per_token


def build_prompt(query, selected_docs, prompt_config):
    """根据查询和选择的文档构建最终提示

    Args:
        query: 查询文本
        selected_docs: 选择的文档列表
        prompt_config: 提示配置

    Returns:
        构建的提示文本
    """
    # 提取文档文本
    doc_texts = [doc['text'] for doc in selected_docs]
    
    # 文本去重
    unique_texts = deduplicate_texts(doc_texts)
    
    # 估算Token数量并确保不超过预算
    token_budget = prompt_config['token_budget']
    avg_chars_per_token = prompt_config['token_estimation']['avg_chars_per_token']
    
    # 计算查询和模板的Token数
    query_tokens = estimate_tokens(query, avg_chars_per_token)
    template_without_context = prompt_config['template'].replace('{context}', '').replace('{query}', '')
    template_tokens = estimate_tokens(template_without_context, avg_chars_per_token)
    
    remaining_tokens = token_budget - query_tokens - template_tokens
    
    # 选择能够容纳的文本
    selected_texts = []
    current_tokens = 0
    
    for text in unique_texts:
        text_tokens = estimate_tokens(text, avg_chars_per_token)
        if current_tokens + text_tokens <= remaining_tokens:
            selected_texts.append(text)
            current_tokens += text_tokens
        else:
            # 如果无法完全容纳当前文本，但剩余空间足够容纳部分文本
            if remaining_tokens - current_tokens > 0:
                # 估算可以容纳的字符数
                remaining_chars = (remaining_tokens - current_tokens) * avg_chars_per_token
                # 截取文本并添加省略号
                truncated_text = text[:int(remaining_chars)] + "..."
                selected_texts.append(truncated_text)
                current_tokens += estimate_tokens(truncated_text, avg_chars_per_token)
            break
    
    # 拼接上下文文本
    context = prompt_config['context_separator'].join(selected_texts)
    
    # 构建最终提示
    prompt = prompt_config['template'].format(query=query, context=context)
    
    return prompt


def save_outputs(prompt, selected_ids, output_dir):
    """保存输出文件

    Args:
        prompt: 构建的提示文本
        selected_ids: 选择的文档ID列表
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存提示文本
    prompt_path = os.path.join(output_dir, 'prompt.txt')
    try:
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"提示文本已保存至 {prompt_path}")
    except Exception as e:
        print(f"保存提示文本失败: {e}")
    
    # 保存选择的文档ID
    selected_ids_path = os.path.join(output_dir, 'selected_ids.json')
    try:
        with open(selected_ids_path, 'w', encoding='utf-8') as f:
            json.dump(selected_ids, f, ensure_ascii=False, indent=2)
        print(f"选择的文档ID已保存至 {selected_ids_path}")
    except Exception as e:
        print(f"保存选择的文档ID失败: {e}")
    
    return prompt_path, selected_ids_path


def main():
    """主函数，协调整个MMR选择和RAG提示构建流程
    """
    start_time = time.time()
    
    # 检查并安装依赖
    install_dependencies()
    
    # 定义文件路径
    query_path = os.path.join('data', 'query.json')
    candidates_path = os.path.join('data', 'candidates.jsonl')
    config_path = os.path.join('configs', 'prompt.json')
    output_dir = os.path.join('outputs')
    
    # 加载输入数据
    print("开始加载输入数据...")
    query_data = load_json_file(query_path)
    candidates = load_jsonl_file(candidates_path)
    config = load_json_file(config_path)
    
    print(f"加载完成: 查询1条, 候选文档{len(candidates)}条")
    
    # 执行MMR选择
    print("开始MMR文档选择...")
    lambda_param = config['mmr_params']['lambda']
    top_k = config['mmr_params']['top_k']
    
    selected_docs, selected_ids = mmr_selection(
        query_data['vector'],
        candidates,
        lambda_param=lambda_param,
        top_k=top_k
    )
    
    print(f"MMR选择完成: 选择了{len(selected_docs)}条文档")
    print(f"选择的文档ID: {selected_ids}")
    
    # 构建提示
    print("开始构建RAG提示...")
    prompt = build_prompt(query_data['text'], selected_docs, config)
    
    # 保存输出
    prompt_path, selected_ids_path = save_outputs(prompt, selected_ids, output_dir)
    
    # 检查输出文件的中文是否有乱码
    print("检查输出文件的中文编码...")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        print(f"提示文件编码检查通过，文件大小: {len(prompt_content)} 字符")
    except UnicodeDecodeError:
        print("警告: 提示文件可能存在中文乱码")
    
    # 打印执行统计
    end_time = time.time()
    print(f"\n执行统计:")
    print(f"- 总耗时: {end_time - start_time:.2f} 秒")
    print(f"- 选择的文档数: {len(selected_docs)}")
    print(f"- 生成的提示Token估算: {estimate_tokens(prompt, config['token_estimation']['avg_chars_per_token']):.2f} tokens")
    print(f"- 输出文件: {prompt_path}, {selected_ids_path}")
    
    print("\nMMR提示构建流程已完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)