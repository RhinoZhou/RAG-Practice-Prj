# -*- coding: utf-8 -*-
"""
同义词未归一导致 Recall@10 下降（反事实实验）

用法：
    python counterfactual_synonym_fault.py

目录结构（与脚本同级）：
    ./data/queries.csv
    ./data/corpus.csv
    ./data/synonyms.csv

功能：
    - 使用简化的 TF-IDF + 余弦相似度进行检索排名（无需第三方检索库）
    - 对比两种模式：
        1) 不做同义词归一（no_norm）
        2) 对 query 和 doc 同步做同义词归一（with_norm）
    - 计算每个 query 的 Recall@10，并输出平均值对比
    - 展示若干典型查询的 Top-10 排名对照
"""

import os
import csv
import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple


# ---------- 工具函数 ----------

def read_csv(filepath: str) -> List[Dict[str, str]]:
    rows = []
    # 尝试不同编码格式和分隔符
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    delimiters = [',', '\t']  # 逗号和制表符
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows = [{k: v for k, v in r.items()} for r in reader]
                
                # 检查是否成功解析了列
                if rows and len(rows[0]) > 1:
                    print(f"Successfully read {filepath} with {encoding} encoding and {repr(delimiter)} delimiter")
                    return rows
            except (UnicodeDecodeError, csv.Error):
                continue
    
    # 最后的尝试：手动解析
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                lines = f.readlines()
                if not lines:
                    continue
                
                # 尝试解析第一行作为标题
                headers = lines[0].strip().split('\t')  # 优先尝试Tab分隔
                if len(headers) <= 1:
                    headers = lines[0].strip().split(',')  # 然后尝试逗号分隔
                
                # 如果解析出多个标题，手动构建字典
                if len(headers) > 1:
                    rows = []
                    for line in lines[1:]:
                        values = line.strip().split('\t')
                        if len(values) <= 1:
                            values = line.strip().split(',')
                        if len(values) == len(headers):
                            rows.append(dict(zip(headers, values)))
                    
                    if rows:
                        print(f"Successfully read {filepath} with manual parsing ({encoding} encoding)")
                        return rows
        except UnicodeDecodeError:
            continue
    
    # 如果所有方法都失败，抛出异常
    raise ValueError(f"Could not decode and parse {filepath} with any of the supported methods")


def simple_tokenize(text: str) -> List[str]:
    """
    中文分词函数：
    1. 首先用正则表达式保留中文、字母、数字，其他作为分隔
    2. 然后针对中文文本进行更细致的处理，将常见的药品名称和专业术语作为整体识别
    """
    t = str(text).lower()
    # 保留中文、字母、数字，其他作为分隔
    t = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", " ", t)
    
    # 先进行简单的空格分割
    tokens = [x for x in t.split() if x]
    
    # 针对中文文本进行更细粒度的分词
    # 我们可以采用基于词典的最大正向匹配来改进分词结果
    # 首先定义一个简单的词典，包含一些常见的药品名称和专业术语
    medicine_dict = {
        '对乙酰氨基酚', '乙酰氨基酚', '扑热息痛', 
        '阿司匹林', '乙酰水杨酸', 
        '布洛芬', '芬必得',
        '不良反应', '副作用',
        '适应症', '用途',
        '用法用量', '使用剂量',
        '注意事项', '用药须知'
    }
    
    # 最大正向匹配算法
    result = []
    for token in tokens:
        # 如果不是纯中文，直接加入结果
        if not all('\u4e00' <= char <= '\u9fa5' for char in token):
            result.append(token)
            continue
        
        # 对纯中文词语进行最大正向匹配
        i = 0
        while i < len(token):
            max_len = min(8, len(token) - i)  # 最大匹配长度为8
            found = False
            for j in range(max_len, 0, -1):
                word = token[i:i+j]
                if word in medicine_dict:
                    result.append(word)
                    i += j
                    found = True
                    break
            if not found:
                # 如果没有找到匹配的词，就将单个字符作为一个词
                result.append(token[i])
                i += 1
    
    return result


def build_synonym_maps(syn_rows: List[Dict[str, str]]) -> Tuple[Dict[str, set], Dict[str, str]]:
    """
    构建：
      - lemma -> set(synonyms)
      - synonym -> lemma （含 lemma->lemma 自映射）
    """
    lemma2syns: Dict[str, set] = defaultdict(set)
    syn2lemma: Dict[str, str] = {}
    print(f"读取到的同义词行数量: {len(syn_rows)}")
    if syn_rows:
        print(f"同义词表前几行: {syn_rows[:3]}")
    
    for r in syn_rows:
        lemma = (r.get("lemma") or "").strip()
        syn = (r.get("synonym") or "").strip()
        if not lemma or not syn:
            continue
        lemma2syns[lemma].add(syn)
        syn2lemma[syn] = lemma
        syn2lemma[lemma] = lemma  # 自映射
    
    print(f"构建的同义词映射数量: {len(syn2lemma)}")
    # 打印部分映射用于调试
    sample_size = 5
    print(f"部分同义词映射示例（最多{sample_size}个）:")
    for i, (syn, lemma) in enumerate(syn2lemma.items()):
        if i < sample_size:
            print(f"  {syn} -> {lemma}")
        else:
            break
    
    return dict(lemma2syns), syn2lemma


def normalize_tokens(tokens: List[str], syn2lemma: Dict[str, str]) -> List[str]:
    return [syn2lemma.get(t, t) for t in tokens]


def build_vocab_and_idf(docs_tokens_list: List[List[str]], min_df: int = 1) -> Tuple[Dict[str, int], List[float]]:
    """
    基于文档词频构建词表与 IDF
    """
    df_count: Dict[str, int] = defaultdict(int)
    for tokens in docs_tokens_list:
        for t in set(tokens):
            df_count[t] += 1

    vocab = {}
    for t in sorted(df_count.keys()):
        if df_count[t] >= min_df:
            vocab[t] = len(vocab)

    N = len(docs_tokens_list)
    idf = [0.0] * len(vocab)
    for t, i in vocab.items():
        # 平滑 IDF
        idf[i] = math.log((N + 1) / (df_count[t] + 1)) + 1.0
    return vocab, idf


def vectorize(tokens: List[str], vocab: Dict[str, int], idf: List[float]) -> List[float]:
    vec = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    # tf-idf
    vec = [vec[i] * idf[i] for i in range(len(vec))]
    # L2 归一化
    norm = math.sqrt(sum(x * x for x in vec)) + 1e-12
    return [x / norm for x in vec]


def cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def recall_at_k(ranked_doc_ids: List[str], gold_set: set, k: int = 10) -> float:
    topk = ranked_doc_ids[:k]
    hit = sum(1 for d in topk if d in gold_set)
    return hit / max(1, len(gold_set))


def rank_docs(query_vec: List[float], doc_vecs: List[List[float]], doc_ids: List[str]) -> List[str]:
    scores = [(doc_id, cosine_sim(query_vec, dv)) for doc_id, dv in zip(doc_ids, doc_vecs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scores]


# ---------- 实验主体 ----------

def run_experiment(
    queries_rows: List[Dict[str, str]],
    corpus_rows: List[Dict[str, str]],
    syn2lemma: Dict[str, str],
    mode: str = "no_norm",
    k: int = 10,
):
    """
    mode: 'no_norm' 或 'with_norm'
    返回：列表，每个元素包含 {query_id, recall@10, ranked}
    """
    print(f"\n===== 运行实验模式: {mode} =====")
    # 添加调试信息
    print(f"corpus_rows length: {len(corpus_rows)}")
    if corpus_rows:
        print(f"First row keys: {list(corpus_rows[0].keys())}")
        print(f"First row content: {corpus_rows[0]}")
    
    print(f"queries_rows length: {len(queries_rows)}")
    if queries_rows:
        print(f"First query: {queries_rows[0]}")
    
    # 文档预处理
    doc_tokens_all: List[List[str]] = []
    print(f"\n处理文档示例:")
    for i, r in enumerate(corpus_rows[:2]):  # 只打印前两个文档的处理情况
        text = f"{r.get('title','')} {r.get('content','')}"
        toks = simple_tokenize(text)
        norm_toks = normalize_tokens(toks, syn2lemma) if mode == "with_norm" else toks
        print(f"文档 {r.get('doc_id')}:")
        print(f"  原始文本: {text}")
        print(f"  分词结果: {toks}")
        if mode == "with_norm":
            print(f"  归一化结果: {norm_toks}")
        doc_tokens_all.append(norm_toks)
    
    # 处理剩余文档
    for r in corpus_rows[2:]:
        text = f"{r.get('title','')} {r.get('content','')}"
        toks = simple_tokenize(text)
        if mode == "with_norm":
            toks = normalize_tokens(toks, syn2lemma)
        doc_tokens_all.append(toks)

    vocab, idf = build_vocab_and_idf(doc_tokens_all)
    print(f"\n构建的词表大小: {len(vocab)}")
    # 打印部分词表和IDF值
    sample_size = 5
    print(f"部分词表和IDF值（最多{sample_size}个）:")
    for i, (word, idx) in enumerate(list(vocab.items())[:sample_size]):
        print(f"  {word}: {idf[idx]:.4f}")
    
    doc_vecs = [vectorize(toks, vocab, idf) for toks in doc_tokens_all]
    doc_ids = [r.get("doc_id", "Unknown") for r in corpus_rows]  # 使用get方法避免KeyError
    
    # 检查是否有未知文档ID
    unknown_count = sum(1 for doc_id in doc_ids if doc_id == "Unknown")
    if unknown_count > 0:
        print(f"Warning: {unknown_count} documents have no 'doc_id' field")

    results = []
    print(f"\n处理查询示例:")
    for i, qr in enumerate(queries_rows[:2]):  # 只打印前两个查询的处理情况
        q_text = qr["query_text"]
        gold = set((qr["gold_doc_ids"] or "").split("|")) if qr.get("gold_doc_ids") else set()
        q_tokens = simple_tokenize(q_text)
        norm_q_tokens = normalize_tokens(q_tokens, syn2lemma) if mode == "with_norm" else q_tokens
        print(f"查询 {qr['query_id']}:")
        print(f"  查询文本: {q_text}")
        print(f"  分词结果: {q_tokens}")
        if mode == "with_norm":
            print(f"  归一化结果: {norm_q_tokens}")
        print(f"  金标准文档: {gold}")
        
        q_vec = vectorize(norm_q_tokens, vocab, idf)
        ranked = rank_docs(q_vec, doc_vecs, doc_ids)
        rec = recall_at_k(ranked, gold, k=k)
        print(f"  召回率@10: {rec:.3f}")
        print(f"  Top-5 排名: {ranked[:5]}")
        results.append({"query_id": qr["query_id"], "recall@10": rec, "ranked": ranked[:k]})
    
    # 处理剩余查询
    for qr in queries_rows[2:]:
        q_text = qr["query_text"]
        gold = set((qr["gold_doc_ids"] or "").split("|")) if qr.get("gold_doc_ids") else set()
        q_tokens = simple_tokenize(q_text)
        if mode == "with_norm":
            q_tokens = normalize_tokens(q_tokens, syn2lemma)
        q_vec = vectorize(q_tokens, vocab, idf)
        ranked = rank_docs(q_vec, doc_vecs, doc_ids)
        rec = recall_at_k(ranked, gold, k=k)
        results.append({"query_id": qr["query_id"], "recall@10": rec, "ranked": ranked[:k]})
    
    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    queries_rows = read_csv(os.path.join(data_dir, "queries.csv"))
    corpus_rows = read_csv(os.path.join(data_dir, "corpus.csv"))
    synonyms_rows = read_csv(os.path.join(data_dir, "synonyms.csv"))

    _, syn2lemma = build_synonym_maps(synonyms_rows)

    # 运行两种模式
    res_no = run_experiment(queries_rows, corpus_rows, syn2lemma, mode="no_norm", k=10)
    res_yes = run_experiment(queries_rows, corpus_rows, syn2lemma, mode="with_norm", k=10)

    # 整理对比
    by_id_no = {r["query_id"]: r for r in res_no}
    by_id_yes = {r["query_id"]: r for r in res_yes}
    qids = [r["query_id"] for r in queries_rows]

    print("=" * 80)
    print("同义词归一化前后 Recall@10 对比（每条查询）")
    print("-" * 80)
    header = f"{'QueryID':<6}  {'Recall@10(no_norm)':>18}  {'Recall@10(with_norm)':>20}  {'Delta':>7}"
    print(header)
    print("-" * len(header))

    avg_no = 0.0
    avg_yes = 0.0
    for qid in qids:
        r_no = by_id_no[qid]["recall@10"]
        r_yes = by_id_yes[qid]["recall@10"]
        avg_no += r_no
        avg_yes += r_yes
        print(f"{qid:<6}  {r_no:>18.3f}  {r_yes:>20.3f}  {r_yes - r_no:>7.3f}")

    n = max(1, len(qids))
    avg_no /= n
    avg_yes /= n

    print("-" * len(header))
    print(f"{'AVERAGE':<6}  {avg_no:>18.3f}  {avg_yes:>20.3f}  {avg_yes - avg_no:>7.3f}")
    print("=" * 80)

    # 展示若干典型查询的 Top-10 排名对照
    demos = ["Q1", "Q2", "Q6", "Q7", "Q10"]
    print("\n典型查询 Top-10 排名对照（no_norm vs with_norm）")
    for qid in demos:
        rno = by_id_no[qid]["ranked"]
        ryes = by_id_yes[qid]["ranked"]
        print("-" * 80)
        print(f"Query {qid}")
        print("No Norm  Top-10:", rno)
        print("With Norm Top-10:", ryes)

    # 课堂讲解提示
    print("\n提示：")
    print("1) 未归一：同义词以不同词形出现，信号分散到不同维度，匹配概率下降。")
    print("2) 归一化：query 与 doc 同步映射到 lemma，语义信号汇聚，同义项相互加强。")
    print("3) 事故复盘：将 miss 的样例加入评测集（金标准），形成持续回归监测。")


if __name__ == "__main__":
    main()