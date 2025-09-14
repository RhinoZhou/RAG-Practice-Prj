#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 和相似度计算快速入门示例

功能说明:
- 生成 RAG 流程图 SVG: "查询→嵌入→检索→重排→拼接→生成"的完整流程可视化
- 相似度计算与比较: 对多组句对计算余弦相似度、内积(IP)和欧氏距离(L2)
- 排序一致性评估: 比较不同相似度度量的排序结果，计算Kendall和Spearman相关系数
- 结果可视化: 打印Top-N相似对与一致性统计信息

输入:
- 内置示例句子列表(可通过 data/examples.json 文件覆盖)

输出:
- outputs/rag_pipeline.svg: RAG流程图
- 控制台输出: 相似度对比表格和排序一致性报告

依赖包:
- numpy: 用于数值计算
- scipy: 用于统计计算和相似度度量
- scikit-learn: 用于文本向量化


"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Tuple

# 检查并安装依赖包
def check_and_install_dependencies():
    """检查并自动安装必要的依赖包"""
    required_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0"
    ]
    
    try:
        # 检查是否已安装所有必要的包
        import numpy as np
        import scipy
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ 所有依赖包已安装完成")
    except ImportError:
        print("正在安装必要的依赖包...")
        # 使用pip安装缺失的包
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ 成功安装 {package}")
            except Exception as e:
                print(f"✗ 安装 {package} 失败: {str(e)}")
                raise RuntimeError(f"无法安装必要的依赖包，请手动安装: {package}") from e

# 创建输出目录
def create_output_directory(directory="outputs"):
    """创建输出目录，用于保存SVG文件等"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ 创建输出目录: {directory}")
    return directory

# 生成RAG流程图SVG
def generate_rag_flow_svg(output_path="outputs/rag_pipeline.svg"):
    """
    生成RAG流程的SVG示意图
    流程: 查询 → 嵌入 → 检索 → 重排 → 拼接 → 生成
    """
    svg_content = """
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200" viewBox="0 0 800 200">
    <!-- 背景 -->
    <rect width="800" height="200" fill="#f9f9f9" rx="10" ry="10"/>
    
    <!-- 标题 -->
    <text x="400" y="30" font-family="Arial" font-size="20" text-anchor="middle" font-weight="bold">RAG流程示意图</text>
    
    <!-- 节点定义 -->
    <g>
        <!-- 查询节点 -->
        <circle cx="100" cy="100" r="40" fill="#ff9f9f" stroke="#333" stroke-width="2"/>
        <text x="100" y="105" font-family="Arial" font-size="16" text-anchor="middle">查询</text>
        
        <!-- 嵌入节点 -->
        <circle cx="230" cy="100" r="40" fill="#9fff9f" stroke="#333" stroke-width="2"/>
        <text x="230" y="105" font-family="Arial" font-size="16" text-anchor="middle">嵌入</text>
        
        <!-- 检索节点 -->
        <circle cx="360" cy="100" r="40" fill="#9f9fff" stroke="#333" stroke-width="2"/>
        <text x="360" y="105" font-family="Arial" font-size="16" text-anchor="middle">检索</text>
        
        <!-- 重排节点 -->
        <circle cx="490" cy="100" r="40" fill="#ffff9f" stroke="#333" stroke-width="2"/>
        <text x="490" y="105" font-family="Arial" font-size="16" text-anchor="middle">重排</text>
        
        <!-- 拼接节点 -->
        <circle cx="620" cy="100" r="40" fill="#9fffff" stroke="#333" stroke-width="2"/>
        <text x="620" y="105" font-family="Arial" font-size="16" text-anchor="middle">拼接</text>
        
        <!-- 生成节点 -->
        <circle cx="750" cy="100" r="40" fill="#ff9fff" stroke="#333" stroke-width="2"/>
        <text x="750" y="105" font-family="Arial" font-size="16" text-anchor="middle">生成</text>
    </g>
    
    <!-- 箭头连接 -->
    <g>
        <line x1="140" y1="100" x2="190" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
        <line x1="270" y1="100" x2="320" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
        <line x1="400" y1="100" x2="450" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
        <line x1="530" y1="100" x2="580" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
        <line x1="660" y1="100" x2="710" y2="100" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
        
        <!-- 箭头标记定义 -->
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
            </marker>
        </defs>
    </g>
</svg>
    """
    
    # 写入SVG文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content.strip())
    print(f"✓ RAG流程图已保存至: {output_path}")

# 加载示例句子
def load_example_sentences(data_path="data/examples.json"):
    """\加载示例句子列表，优先从文件加载，文件不存在则使用内置示例"""
    # 内置示例句子
    default_examples = [
        "人工智能正在改变我们的生活方式",
        "机器学习是人工智能的一个分支",
        "深度学习在图像识别领域取得了重大突破",
        "自然语言处理让计算机能够理解人类语言",
        "语音识别技术已经广泛应用于各种设备中",
        "计算机视觉帮助机器理解图像内容",
        "数据挖掘是从大量数据中提取有用信息的过程",
        "知识图谱用于表示实体之间的关系",
        "推荐系统可以根据用户兴趣提供个性化内容",
        "强化学习通过试错来学习最优策略"
    ]
    
    # 检查是否有自定义数据文件
    if os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
            print(f"✓ 从文件加载示例句子: {data_path}")
            return examples
        except Exception as e:
            print(f"✗ 加载自定义数据失败: {str(e)}，使用默认示例")
    
    # 使用默认示例
    print("✓ 使用内置示例句子")
    return default_examples

# 计算句子嵌入
def compute_sentence_embeddings(sentences: List[str]):
    """\使用TF-IDF计算句子嵌入向量"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    print("✓ 使用TF-IDF计算句子嵌入向量...")
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(sentences).toarray()
    
    # 对嵌入向量进行单位化（归一化），便于计算余弦相似度和内积
    print("✓ 对嵌入向量进行单位化...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    return normalized_embeddings

# 计算相似度矩阵和排序结果
def compute_similarities_and_rankings(embeddings, query_index=0, top_k=5):
    """
    计算三种相似度度量并获取排序结果
    1. 余弦相似度: 已归一化的向量，等价于点积
    2. 内积(IP): 与余弦相似度相同（对于归一化向量）
    3. 欧氏距离(L2): 距离越小，相似度越高
    """
    import numpy as np
    
    # 选择查询向量
    query_embedding = embeddings[query_index:query_index+1]
    
    # 计算余弦相似度 (对于归一化向量，等价于内积)
    cosine_scores = np.dot(embeddings, query_embedding.T).flatten()
    
    # 计算欧氏距离
    l2_distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # 对相似度进行排序（注意L2是距离，排序方向相反）
    cosine_ranking = np.argsort(-cosine_scores)  # 降序排列
    ip_ranking = cosine_ranking.copy()  # 对于归一化向量，内积等于余弦相似度
    l2_ranking = np.argsort(l2_distances)  # 升序排列 (距离越小越相似)
    
    # 获取Top-K结果
    cosine_top_k = cosine_ranking[:top_k]
    ip_top_k = ip_ranking[:top_k]
    l2_top_k = l2_ranking[:top_k]
    
    results = {
        "cosine": {
            "scores": cosine_scores,
            "ranking": cosine_ranking,
            "top_k": cosine_top_k
        },
        "ip": {
            "scores": cosine_scores.copy(),  # 对于归一化向量，内积等于余弦相似度
            "ranking": ip_ranking,
            "top_k": ip_top_k
        },
        "l2": {
            "scores": -l2_distances,  # 转换为负值以便于排序一致性比较
            "ranking": l2_ranking,
            "top_k": l2_top_k
        }
    }
    
    return results, query_index

# 计算排序一致性指标
def compute_rank_correlations(rankings):
    """\计算Kendall和Spearman排序一致性相关系数"""
    from scipy import stats
    
    # 提取排名数组
    cosine_ranking = rankings["cosine"]["ranking"]
    ip_ranking = rankings["ip"]["ranking"]
    l2_ranking = rankings["l2"]["ranking"]
    
    # 计算Kendall相关系数
    kendall_cosine_ip, _ = stats.kendalltau(cosine_ranking, ip_ranking)
    kendall_cosine_l2, _ = stats.kendalltau(cosine_ranking, l2_ranking)
    kendall_ip_l2, _ = stats.kendalltau(ip_ranking, l2_ranking)
    
    # 计算Spearman相关系数
    spearman_cosine_ip, _ = stats.spearmanr(cosine_ranking, ip_ranking)
    spearman_cosine_l2, _ = stats.spearmanr(cosine_ranking, l2_ranking)
    spearman_ip_l2, _ = stats.spearmanr(ip_ranking, l2_ranking)
    
    correlations = {
        "kendall": {
            "cosine-ip": kendall_cosine_ip,
            "cosine-l2": kendall_cosine_l2,
            "ip-l2": kendall_ip_l2
        },
        "spearman": {
            "cosine-ip": spearman_cosine_ip,
            "cosine-l2": spearman_cosine_l2,
            "ip-l2": spearman_ip_l2
        }
    }
    
    return correlations

# 打印相似度比较结果
def print_similarity_comparison(results, sentences, query_index, top_k=5):
    """打印Top-N相似对的比较结果"""
    print("\n=== 相似度度量比较结果 ===")
    print(f"查询句子: {sentences[query_index]}")
    print("\nTop-{top_k} 相似句子排名:")
    
    # 打印表头
    print("\n{:<5} {:<25} {:<25} {:<25}".format("排名", "余弦相似度", "内积(IP)", "欧氏距离(L2)"))
    print("=" * 85)
    
    # 获取三种度量的Top-K结果
    cosine_top_k = results["cosine"]["top_k"]
    ip_top_k = results["ip"]["top_k"]
    l2_top_k = results["l2"]["top_k"]
    
    # 打印每行结果
    for i in range(top_k):
        # 对于每个排名位置，获取三种度量对应的句子
        cosine_sentence = sentences[cosine_top_k[i]] if i < len(cosine_top_k) else "-"
        ip_sentence = sentences[ip_top_k[i]] if i < len(ip_top_k) else "-"
        l2_sentence = sentences[l2_top_k[i]] if i < len(l2_top_k) else "-"
        
        # 截断过长的句子以便于显示
        cosine_sentence = (cosine_sentence[:22] + '...') if len(cosine_sentence) > 25 else cosine_sentence
        ip_sentence = (ip_sentence[:22] + '...') if len(ip_sentence) > 25 else ip_sentence
        l2_sentence = (l2_sentence[:22] + '...') if len(l2_sentence) > 25 else l2_sentence
        
        print("{:<5} {:<25} {:<25} {:<25}".format(i+1, cosine_sentence, ip_sentence, l2_sentence))

# 打印排序一致性结果
def print_rank_correlations(correlations):
    """打印排序一致性指标（Kendall和Spearman相关系数）"""
    print("\n=== 排序一致性分析 ===")
    print("\nKendall相关系数:")
    print("{:<12} {:.4f}".format("cosine-ip:", correlations["kendall"]["cosine-ip"]))
    print("{:<12} {:.4f}".format("cosine-l2:", correlations["kendall"]["cosine-l2"]))
    print("{:<12} {:.4f}".format("ip-l2:", correlations["kendall"]["ip-l2"]))
    
    print("\nSpearman相关系数:")
    print("{:<12} {:.4f}".format("cosine-ip:", correlations["spearman"]["cosine-ip"]))
    print("{:<12} {:.4f}".format("cosine-l2:", correlations["spearman"]["cosine-l2"]))
    print("{:<12} {:.4f}".format("ip-l2:", correlations["spearman"]["ip-l2"]))
    
    # 简单解释
    print("\n一致性解释:")
    print("- 相关系数越接近1，表示排序结果越一致")
    print("- 对于归一化的嵌入向量，余弦相似度和内积应该具有完全一致的排序")
    print("- L2距离与余弦相似度/内积的一致性取决于数据分布")

# 主函数
def main():
    """主函数，执行完整流程"""
    print("\n===== RAG 和相似度计算快速入门 =====")
    
    # 1. 检查并安装依赖
    check_and_install_dependencies()
    
    # 2. 创建输出目录
    output_dir = create_output_directory()
    
    # 3. 生成RAG流程图
    svg_path = os.path.join(output_dir, "rag_pipeline.svg")
    generate_rag_flow_svg(svg_path)
    
    # 4. 加载示例句子
    sentences = load_example_sentences()
    
    # 5. 计算句子嵌入向量
    embeddings = compute_sentence_embeddings(sentences)
    
    # 6. 计算相似度和排序结果
    results, query_index = compute_similarities_and_rankings(embeddings, query_index=0, top_k=5)
    
    # 7. 计算排序一致性
    correlations = compute_rank_correlations(results)
    
    # 8. 打印相似度比较结果
    print_similarity_comparison(results, sentences, query_index, top_k=5)
    
    # 9. 打印排序一致性结果
    print_rank_correlations(correlations)
    
    print("\n===== 程序执行完成 =====")

# 程序入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)