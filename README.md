# 数据准备阶段演示代码集

本目录包含一系列用于演示RAG（检索增强生成）系统中数据准备阶段关键技术的Python脚本。这些脚本涵盖了数据质量分析、清洗、转换、评估和RAG流水线实现的各个方面。

## 脚本功能说明

### 01-counterfactual_synonym_fault.py

**反事实同义词故障分析**
- 演示了文本归一化对检索结果的影响
- 对比了未归一化和归一化（同义词处理）两种情况下的检索性能
- 分析了同义词以不同词形出现时，查询与文档匹配概率的变化
- 提供了查询结果的Top-10排名对照，展示了归一化如何改善检索效果

### 02-compute_quality_matrix.py

**数据质量矩阵计算**
- 实现了数据质量多维度评估框架
- 计算了完整性（completeness）、准确性（accuracy）、标准化（standardization）、新鲜度（freshness_p95）和可治理性（governability）等关键质量指标
- 将质量指标映射到"Good"、"Warn"、"Bad"三个告警等级
- 生成质量矩阵和总体质量摘要报表，输出CSV格式的质量评估结果

### 03-ab_improvement_simulation.py

**A/B测试改进模拟**
- 模拟了不同检索策略的A/B测试过程
- 对比了多个领域（医疗、法律、金融）的检索性能
- 计算了召回率（recall）、NDCG（归一化折损累积增益）和幻觉率（hallucination）等关键指标
- 提供了查询级和域级的详细测试结果和统计显著性检验
- 生成CSV格式的测试结果和摘要报表

### 04-generate_retrieval_payload.py

**检索负载生成**
- 根据预定义的schema生成标准化的检索payload
- 确保生成的payload符合合规性要求，包括引用ID、版本信息等
- 实现了跨版本混合禁止机制，维护文档版本一致性
- 提供了完整的合规性统计报告，确保100%的合规率
- 输出JSONL格式的payload文件，便于集成到检索系统

### 05-data_cleaning_methods.py

**数据清洗方法演示**
- 实现了三种关键的数据清洗技术：
  1. 缺失值填补（KNN填补法与均值填补对比）
  2. 异常值检测（Z-score、IQR和IsolationForest方法）
  3. 单位统一（不同单位间的换算策略对比）
- 提供了数据清洗前后的可视化对比
- 生成多种图表展示清洗效果，并保存到outputs/figures目录
- 输出CSV格式的清洗后数据，便于后续分析和使用

### 06-minimal_rag_pipeline.py

**RAG最小闭环流水线**
- 实现了完整的RAG数据处理最小闭环，包含六个核心环节：
  1. 数据采集：模拟生成医疗领域文档数据
  2. 数据清洗：执行文本标准化和预处理
  3. 向量索引：生成文档向量并构建索引
  4. 检索匹配：基于余弦相似度进行相关文档检索
  5. 效果评测：计算相关性分数和命中率等指标
  6. 告警机制：监控评测指标并进行阈值检查
- 自动保存各个环节的中间结果和最终输出
- 提供了完整的流水线执行日志和统计信息

## 目录结构

```
├── 01-counterfactual_synonym_fault.py  # 反事实同义词故障分析
├── 02-compute_quality_matrix.py        # 质量矩阵计算
├── 03-ab_improvement_simulation.py     # A/B测试改进模拟
├── 04-generate_retrieval_payload.py    # 检索负载生成
├── 05-data_cleaning_methods.py         # 数据清洗方法演示
├── 06-minimal_rag_pipeline.py          # RAG最小闭环流水线
├── config/                             # 配置文件目录
├── data/                               # 示例数据目录
└── outputs/                            # 输出结果目录
    ├── figures/                        # 可视化结果目录
    ├── *.csv                           # CSV格式的结果文件
    └── *.json*/                        # JSON/JSONL格式的结果文件
```

## 运行说明

每个脚本都可以独立运行，使用Python 3.6+环境执行：

```bash
python 01-counterfactual_synonym_fault.py
python 02-compute_quality_matrix.py
# 以此类推...
```

所有脚本都会在执行过程中输出关键信息，并将结果保存到outputs目录中。

## 依赖包

主要依赖包包括：
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- json
- os

## 输出结果

执行脚本后，所有结果文件将保存在outputs目录中，包括：
- CSV格式的表格数据
- JSON/JSONL格式的结构化数据
- PNG格式的可视化图表（保存在figures子目录）

这些输出结果可以用于进一步的数据分析、可视化展示或集成到其他系统中。