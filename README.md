# 04-Question-Answering-System
法规RAG

## 02-snomed_standardization
将输入的英文病例中的术语，提取出来，使用SNOMED CT库，转换为标准术语

## 03-Entity-Tag
使用BiLSTM-CRF模型结合法律术语词典进行法律文本实体识别

## 04-article-segmentation
使用基于Transformer的段落分割算法对文档进行智能分割

## 05-relation-graph.py
识别法律条款中的引用关系，构建条款之间的逻辑关联图谱

## 06-article-segmentation-hash
给每一个文档分块生成 生成SHA-256哈希值

## 07-multidimensional-tagging
构建多维度标签体系，通过半监督学习方法对法律条款进行自动分类标注

## 08-legal-vector-db
使用FAISS向量库保存前面的数据，构建HNSW索引

## 09-hybrid-retrieval
混合检索，实现BM25和向量相似度的双通道召回，归一化融合打分后去重合并，再用跨编码器重排，输出可引用的Top-N文段
