#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多级索引与二次检索（Two-Pass Retrieval）演示

功能：
- L1 段落级向量索引进行粗召回
- L2 句子级向量索引进行精排
- 标题/正文/元数据混合加权评分（动态可配）
- 可选倒排索引用于关键词过滤
- 增量更新双缓冲模式（影子索引构建→原子切换）

输入：
- 分块文本（含层级：文档->段落->句子）
- 查询集合
- 增量文档（插入/更新）

输出：
- 检索结果（命中段与句）
- 加权明细
- 更新切换日志

代码结构与功能
该代码实现了一个多级索引与二次检索（Two-Pass Retrieval）系统，主要包含以下核心组件：

1. 基础文本向量化与相似度计算
   - 简化的分词和词频向量化
   - 余弦相似度计算
2. 多层次数据结构
   - 句子（Sentence）：最小文本单元，包含文本、向量和元数据
   - 段落（Paragraph）：包含多个句子，自动分句功能
   - 文档（Document）：包含多个段落和元数据
3. 索引实现
   - 向量索引（VectorIndex）：支持段落级和句子级检索
   - 倒排索引（InvertedIndex）：支持关键词匹配
   - 多级索引（MultiLevelIndex）：组合向量索引和倒排索引
4. 双缓冲增量更新机制
   - 影子索引构建
   - 原子切换
   - 日志记录
5. 两阶段检索管线
   - 第一阶段：段落级粗排
   - 第二阶段：句子级精排
   - 混合评分：段落相似度、句子相似度、关键词匹配
注意：为便于演示，采用轻量化的词频向量与余弦相似度，不依赖外部向量库。
"""

from __future__ import annotations
import re
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict

# -----------------------------
# 基础文本向量化与相似度
# -----------------------------

def tokenize(text: str) -> List[str]:
    # 简化分词，按字母数字与中文字符片段，全部小写
    return re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", text.lower())


def text_to_vector(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    all_keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# -----------------------------
# 数据结构
# -----------------------------

@dataclass
class Sentence:
    text: str
    sent_id: str

    def vector(self) -> Counter:
        return text_to_vector(self.text)


@dataclass
class Paragraph:
    title: str
    body: str
    para_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sentences(self) -> List[Sentence]:
        # 简化的句子切分：按中文句号/英文句号/问号/感叹号
        raw = re.split(r"(?<=[。！？.!?])\s+", self.body.strip())
        result: List[Sentence] = []
        for i, s in enumerate([x for x in raw if x]):
            result.append(Sentence(text=s, sent_id=f"{self.para_id}_s{i+1}"))
        if not result and self.body:
            result.append(Sentence(text=self.body, sent_id=f"{self.para_id}_s1"))
        return result

    def vector_title(self) -> Counter:
        return text_to_vector(self.title)

    def vector_body(self) -> Counter:
        return text_to_vector(self.body)


@dataclass
class Document:
    doc_id: str
    paragraphs: List[Paragraph]
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# 倒排索引（可选）
# -----------------------------

class InvertedIndex:
    def __init__(self) -> None:
        self.term_to_para_ids: Dict[str, set] = defaultdict(set)

    def build(self, documents: List[Document]) -> None:
        self.term_to_para_ids.clear()
        for doc in documents:
            for para in doc.paragraphs:
                tokens = set(tokenize(para.title + " " + para.body))
                for t in tokens:
                    self.term_to_para_ids[t].add(para.para_id)

    def filter_candidates(self, query: str, candidates: List[str]) -> List[str]:
        # 若查询中出现的关键词在倒排中无匹配，则不过滤；若有匹配则保留有命中的候选
        q_terms = set(tokenize(query))
        matched_sets = [self.term_to_para_ids.get(t, set()) for t in q_terms]
        union: set = set().union(*matched_sets) if matched_sets else set()
        if not union:
            return candidates
        candidate_set = set(candidates)
        return [pid for pid in candidates if pid in union and pid in candidate_set]


# -----------------------------
# 多级向量索引（L1 段落级，L2 句子级）
# -----------------------------

@dataclass
class L1IndexItem:
    para_id: str
    title_vec: Counter
    body_vec: Counter
    metadata: Dict[str, Any]


class L1ParagraphIndex:
    def __init__(self, weight_title: float = 0.3, weight_body: float = 0.7, weight_meta: float = 0.1):
        self.items: Dict[str, L1IndexItem] = {}
        self.weight_title = weight_title
        self.weight_body = weight_body
        self.weight_meta = weight_meta

    def build(self, documents: List[Document]) -> None:
        self.items.clear()
        for doc in documents:
            for para in doc.paragraphs:
                self.items[para.para_id] = L1IndexItem(
                    para_id=para.para_id,
                    title_vec=para.vector_title(),
                    body_vec=para.vector_body(),
                    metadata=para.metadata.copy(),
                )

    def score(self, query_vec: Counter, query_terms: List[str], item: L1IndexItem) -> Tuple[float, Dict[str, float]]:
        s_title = cosine_similarity(query_vec, item.title_vec)
        s_body = cosine_similarity(query_vec, item.body_vec)
        # 元数据词匹配简单加成：若query包含的词与metadata的字符串化包含关系
        meta_text = json.dumps(item.metadata, ensure_ascii=False).lower()
        meta_hit = sum(1 for t in query_terms if t in meta_text)
        s_meta = min(1.0, meta_hit * 0.1)
        score = self.weight_title * s_title + self.weight_body * s_body + self.weight_meta * s_meta
        detail = {"title": s_title, "body": s_body, "meta": s_meta, "weighted": score}
        return score, detail

    def search(self, query: str, top_k: int = 5, inverted: Optional[InvertedIndex] = None) -> List[Tuple[str, float, Dict[str, float]]]:
        query_vec = text_to_vector(query)
        q_terms = tokenize(query)
        para_ids = list(self.items.keys())
        if inverted is not None:
            para_ids = inverted.filter_candidates(query, para_ids)
        scored: List[Tuple[str, float, Dict[str, float]]] = []
        for pid in para_ids:
            item = self.items[pid]
            s, detail = self.score(query_vec, q_terms, item)
            scored.append((pid, s, detail))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class L2SentenceIndex:
    def __init__(self) -> None:
        self.sent_vecs: Dict[str, Counter] = {}
        self.sent_text: Dict[str, str] = {}
        self.para_to_sents: Dict[str, List[str]] = defaultdict(list)

    def build_from_documents(self, documents: List[Document]) -> None:
        self.sent_vecs.clear()
        self.sent_text.clear()
        self.para_to_sents.clear()
        for doc in documents:
            for para in doc.paragraphs:
                for sent in para.sentences():
                    self.sent_vecs[sent.sent_id] = sent.vector()
                    self.sent_text[sent.sent_id] = sent.text
                    self.para_to_sents[para.para_id].append(sent.sent_id)

    def rerank_within_paragraphs(self, query: str, para_ids: List[str], top_m_per_para: int = 1) -> List[Tuple[str, str, float]]:
        q_vec = text_to_vector(query)
        results: List[Tuple[str, str, float]] = []
        for pid in para_ids:
            sids = self.para_to_sents.get(pid, [])
            scored = []
            for sid in sids:
                s = cosine_similarity(q_vec, self.sent_vecs.get(sid, Counter()))
                scored.append((pid, sid, s))
            scored.sort(key=lambda x: x[2], reverse=True)
            results.extend(scored[:top_m_per_para])
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# -----------------------------
# 双缓冲增量更新（影子索引→切换）
# -----------------------------

@dataclass
class IndexBundle:
    l1: L1ParagraphIndex
    l2: L2SentenceIndex
    inverted: Optional[InvertedIndex]
    version: str


class DoubleBufferIndexer:
    def __init__(self, use_inverted: bool = True) -> None:
        self.active: Optional[IndexBundle] = None
        self.shadow: Optional[IndexBundle] = None  # type: ignore
        self.use_inverted = use_inverted
        self.logs: List[str] = []

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        print(entry)

    def build_bundle(self, documents: List[Document], version: str) -> IndexBundle:
        l1 = L1ParagraphIndex()
        l1.build(documents)
        l2 = L2SentenceIndex()
        l2.build_from_documents(documents)
        inverted = InvertedIndex() if self.use_inverted else None
        if inverted is not None:
            inverted.build(documents)
        return IndexBundle(l1=l1, l2=l2, inverted=inverted, version=version)

    def initial_build(self, documents: List[Document], version: str = "v1") -> None:
        self.log("启动初始构建...")
        self.active = self.build_bundle(documents, version)
        self.log(f"初始索引已激活：{version}")

    def incremental_update(self, base_documents: List[Document], new_documents: List[Document], new_version: str) -> None:
        self.log("开始增量更新：构建影子索引...")
        merged = self._merge_documents(base_documents, new_documents)
        self.shadow = self.build_bundle(merged, new_version)
        self.log(f"影子索引构建完成：{new_version}")
        self._atomic_switch()

    def _merge_documents(self, base: List[Document], inc: List[Document]) -> List[Document]:
        merged: Dict[str, Document] = {d.doc_id: d for d in base}
        for d in inc:
            merged[d.doc_id] = d  # 覆盖或插入
        return list(merged.values())

    def _atomic_switch(self) -> None:
        if self.shadow is None:
            self.log("影子索引不存在，无法切换")
            return
        new_version = self.shadow.version
        self.active, self.shadow = self.shadow, None
        self.log(f"已原子切换到新版本：{new_version}")


# -----------------------------
# 检索管线
# -----------------------------

@dataclass
class QueryResult:
    query: str
    hits: List[Dict[str, Any]]  # 每个命中包含段落/句子/分数/加权明细


def two_pass_search(bundle: IndexBundle, query: str, top_k_paragraphs: int = 3, top_m_per_para: int = 1) -> QueryResult:
    l1 = bundle.l1
    l2 = bundle.l2
    inverted = bundle.inverted
    # L1 粗召回
    l1_hits = l1.search(query, top_k=top_k_paragraphs, inverted=inverted)
    para_ids = [pid for pid, _, _ in l1_hits]
    # L2 精排
    l2_hits = l2.rerank_within_paragraphs(query, para_ids, top_m_per_para=top_m_per_para)
    # 汇总结果
    detail_map = {pid: detail for pid, _, detail in l1_hits}
    results: List[Dict[str, Any]] = []
    for pid, sid, s in l2_hits:
        results.append({
            "paragraph_id": pid,
            "sentence_id": sid,
            "sentence_score": s,
            "l1_detail": detail_map.get(pid, {}),
        })
    return QueryResult(query=query, hits=results)


# -----------------------------
# 演示数据与主程序
# -----------------------------

def demo_documents() -> List[Document]:
    docs: List[Document] = []
    # 文档A
    paras_a = [
        Paragraph(title="人工智能概述", body="人工智能涵盖机器学习与深度学习。应用包括计算机视觉与自然语言处理。", para_id="A_p1", metadata={"tags": ["AI", "overview"], "lang": "zh"}),
        Paragraph(title="机器学习", body="监督学习与无监督学习常见。回归、分类与聚类是代表任务。", para_id="A_p2", metadata={"tags": ["ML"], "level": "basic"}),
    ]
    docs.append(Document(doc_id="A", paragraphs=paras_a, metadata={"source": "handbook"}))
    # 文档B
    paras_b = [
        Paragraph(title="深度学习", body="卷积神经网络在图像识别中表现优异。循环网络适合序列数据。", para_id="B_p1", metadata={"tags": ["DL"], "domain": "cv"}),
        Paragraph(title="NLP模型", body="Transformer结构推动大语言模型发展，增强文本理解与生成能力。", para_id="B_p2", metadata={"tags": ["NLP", "Transformer"]}),
    ]
    docs.append(Document(doc_id="B", paragraphs=paras_b, metadata={"source": "notes"}))
    return docs


def demo_incremental_documents() -> List[Document]:
    paras_c = [
        Paragraph(title="向量检索", body="倒排索引适合关键词匹配，向量索引用于语义相似度。", para_id="C_p1", metadata={"tags": ["IR"], "engine": "toy"}),
        Paragraph(title="混合加权", body="标题命中、正文语义、元数据标签共同影响最终得分。", para_id="C_p2", metadata={"tags": ["rank"], "note": "mixed"}),
    ]
    return [Document(doc_id="C", paragraphs=paras_c, metadata={"source": "incremental"})]



def main() -> None:
    # 初始构建
    base_docs = demo_documents()
    indexer = DoubleBufferIndexer(use_inverted=True)
    indexer.initial_build(base_docs, version="v1")

    # 查询集合
    queries = [
        "什么是人工智能的主要方向？",
        "Transformer 在 NLP 中的作用",
        "图像识别常用的网络",
        "如何进行关键词与语义的混合检索",
    ]

    # 执行两阶段检索
    for q in queries:
        result = two_pass_search(indexer.active, q, top_k_paragraphs=3, top_m_per_para=1)  # type: ignore
        print("\n查询：", q)
        for hit in result.hits:
            print(json.dumps(hit, ensure_ascii=False))

    # 增量更新 - 双缓冲
    new_docs = demo_incremental_documents()
    indexer.incremental_update(base_documents=base_docs, new_documents=new_docs, new_version="v2")

    # 更新后再检索
    q2 = "向量索引与倒排索引的差异"
    result2 = two_pass_search(indexer.active, q2, top_k_paragraphs=3, top_m_per_para=2)  # type: ignore
    print("\n查询：", q2)
    for hit in result2.hits:
        print(json.dumps(hit, ensure_ascii=False))

    # 打印切换日志
    print("\n=== 切换日志 ===")
    for line in indexer.logs:
        print(line)


if __name__ == "__main__":
    main()
