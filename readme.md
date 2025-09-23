# VectorDB索引技术实践指南

## 项目概述

本项目是一个向量数据库索引技术的实践指南，包含多种向量检索算法实现、参数调优、多模态搜索和评估工具。旨在帮助用户了解和掌握向量数据库索引技术在实际应用中的使用方法。

## 目录结构

```
├── 01-embed_and_reduce.py      # 向量嵌入与降维示例
├── 02-faiss_hnsw_ivfpq_demo.py # FAISS HNSW和IVF-PQ索引演示
├── 03-hnsw_param_sweep.py      # HNSW索引参数调优脚本
├── 04-ivfpq_param_grid.py      # IVF-PQ索引参数网格搜索
├── 05-hybrid_bm25_ann.py       # BM25与向量检索混合搜索
├── 06-clip_multimodal_search.py # CLIP多模态搜索实现
├── 07-mini_rag_pipeline.py     # 简易RAG流水线实现
├── 08-tiered_routing_sim.py    # 分层路由检索模拟
├── 09-eval_curves_ab.py        # 向量检索性能A/B测试评估
├── captions.json               # 图像标题数据
├── check_embeddings.py         # 嵌入向量检查工具
├── data/                       # 数据存储目录
│   ├── chunks.jsonl            # 文本块数据
│   ├── embeddings_256.npy      # 256维嵌入向量
│   └── embeddings_768.npy      # 768维嵌入向量
├── docs/                       # 文档和语料库
│   ├── corpus.txt              # 示例文本语料库
│   └── queries.txt             # 查询文本集合
├── expand_dataset.py           # 数据集扩充工具
├── expand_dataset_to_10x.py    # 数据集10倍扩充脚本
├── images/                     # 图像文件目录
│   ├── image_1.png             # 示例图像1
│   ├── image_2.png             # 示例图像2
│   ├── ...
│   └── image_8.png             # 示例图像8
├── query_texts.txt             # 查询文本文件
└── result/                     # 实验结果存储目录
    ├── ab_eval.csv             # A/B测试CSV结果
    ├── ab_eval.md              # A/B测试Markdown报告
    ├── clip_search.md          # CLIP搜索结果报告
    ├── hnsw_ivfpq_report.md    # HNSW/IVF-PQ索引报告
    ├── hnsw_param_summary.md   # HNSW参数调优总结
    ├── hnsw_param_sweep.csv    # HNSW参数调优CSV结果
    ├── hybrid_results.md       # 混合搜索结果报告
    ├── ivfpq_grid.csv          # IVF-PQ参数网格搜索结果
    ├── mini_rag_answers.md     # 简易RAG问答结果
    └── tiered_routing_report.md # 分层路由报告
```

## 子目录说明

### data/
存储各种格式的向量数据和文本块数据，包括低维(256维)和高维(768维)的嵌入向量，用于索引构建和检索测试。

### docs/
包含用于训练和测试的文本语料库和查询集合，提供示例数据支持各种检索实验。

### images/
存放用于多模态搜索的示例图像文件，支持CLIP模型的图像-文本跨模态检索功能。

### result/
保存所有实验的输出结果，包括CSV格式的数值结果和Markdown格式的可视化报告。

## 数字开头Python文件说明

### 01-embed_and_reduce.py
**功能**：演示文本向量化和向量降维技术，展示如何将高维文本向量降维到低维空间。

**调用说明**：直接运行脚本，会处理示例文本并输出降维前后的向量维度信息。
```bash
python 01-embed_and_reduce.py
```

### 02-faiss_hnsw_ivfpq_demo.py
**功能**：演示FAISS库中HNSW和IVF-PQ两种重要索引算法的基本使用方法。

**调用说明**：直接运行脚本，会创建两种索引并执行基本的向量检索操作。
```bash
python 02-faiss_hnsw_ivfpq_demo.py
```

### 03-hnsw_param_sweep.py
**功能**：对HNSW索引的关键参数进行扫描调优，寻找最佳参数组合以平衡检索性能和精度。

**调用说明**：直接运行脚本，会自动测试不同参数组合并生成参数调优报告。
```bash
python 03-hnsw_param_sweep.py
```

### 04-ivfpq_param_grid.py
**功能**：通过网格搜索方法对IVF-PQ索引的多个参数进行全面测试，找出最优参数配置。

**调用说明**：直接运行脚本，会执行网格搜索并生成参数配置结果。
```bash
python 04-ivfpq_param_grid.py
```

### 05-hybrid_bm25_ann.py
**功能**：实现BM25文本检索与向量检索的混合搜索策略，结合两种检索方法的优势。

**调用说明**：直接运行脚本，会执行混合搜索并生成性能评估报告。
```bash
python 05-hybrid_bm25_ann.py
```

### 06-clip_multimodal_search.py
**功能**：使用CLIP模型实现文本到图像和图像到文本的跨模态搜索功能。

**调用说明**：直接运行脚本，会加载图像和文本数据，构建多模态索引并执行检索演示。
```bash
python 06-clip_multimodal_search.py
```

### 07-mini_rag_pipeline.py
**功能**：实现一个简易的RAG(检索增强生成)流水线，展示向量检索在LLM应用中的实际应用。

**调用说明**：直接运行脚本，会构建简易RAG系统并生成问答结果。
```bash
python 07-mini_rag_pipeline.py
```

### 08-tiered_routing_sim.py
**功能**：模拟分层路由检索策略，演示如何在大规模向量数据库中实现高效检索。

**调用说明**：直接运行脚本，会模拟分层路由过程并生成性能分析报告。
```bash
python 08-tiered_routing_sim.py
```

### 09-eval_curves_ab.py
**功能**：提供向量检索性能的A/B测试评估工具，支持不同索引算法的性能对比。

**调用说明**：直接运行脚本，会执行A/B测试并生成包含中文的评估报告。
```bash
python 09-eval_curves_ab.py
```

## 辅助工具说明

### check_embeddings.py
**功能**：检查嵌入向量文件的基本信息，包括向量数量、维度和数据格式。

**调用说明**：直接运行脚本，会显示当前嵌入向量的统计信息。
```bash
python check_embeddings.py
```

### expand_dataset.py & expand_dataset_to_10x.py
**功能**：用于扩充现有数据集，增加向量数量以支持大规模索引测试。其中expand_dataset_to_10x.py专门将数据集扩充至10倍大小。

**调用说明**：直接运行脚本，会自动扩充数据集并保存结果。
```bash
python expand_dataset.py
# 或扩充至10倍
python expand_dataset_to_10x.py
```

## 环境配置

项目需要以下核心依赖包：
- faiss-cpu：向量索引库
- numpy：数值计算库
- torch & torchvision：PyTorch深度学习框架
- open_clip_torch：OpenCLIP模型实现
- pillow：图像处理库
- tqdm：进度条显示
- requests：网络请求库

大多数脚本会自动检查并安装所需依赖，如需手动安装可使用：
```bash
pip install faiss-cpu numpy torch torchvision open_clip_torch pillow tqdm requests
```

## 使用指南

1. **环境准备**：确保已安装Python 3.8+和pip包管理器
2. **依赖安装**：通过上述命令安装必要的依赖包
3. **示例运行**：根据需要运行对应的示例脚本
4. **结果查看**：所有实验结果会保存在result/目录下

## 最佳实践建议

1. **索引选择**：根据数据集大小和性能需求选择合适的索引算法
   - 小规模数据集：使用FAISS Flat索引确保最佳精度
   - 中等规模数据集：考虑HNSW索引平衡速度和精度
   - 大规模数据集：使用IVF-PQ索引优先考虑检索效率

2. **参数调优**：使用提供的参数调优工具找到最适合特定数据集的参数配置

3. **混合检索**：在实际应用中考虑结合传统文本检索和向量检索的优势

4. **多模态应用**：利用CLIP等多模态模型实现更丰富的检索功能

## 注意事项

1. 部分脚本需要较大内存，处理大规模数据集时请注意系统资源
2. 对于生产环境应用，建议使用GPU版本的FAISS以提高性能
3. 数据集扩充工具生成的是模拟数据，实际应用中应使用真实数据
4. 实验结果会定期更新，建议定期查看result/目录获取最新结果

## 总结

本项目提供了向量数据库索引技术的全面实践指南，涵盖了从基础索引构建到高级应用的各个方面。通过学习和实践这些示例，用户可以掌握向量检索技术在实际应用中的使用方法，为构建高效的检索系统提供技术支持。