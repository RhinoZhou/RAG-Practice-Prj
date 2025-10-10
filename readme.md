# RAG 实践项目目录结构与功能说明

本项目是一个完整的 RAG（检索增强生成）实践代码库，包含从数据处理、检索优化、生成增强到评估监控的全流程实现。

## 目录结构

```
..\14-generation/
├── 01-context_budget_and_windowing_demo.py  # 上下文预算与窗口管理
├── 02-table_to_text_with_anchors.py         # 表格转文本与锚点生成
├── 03-gated_mix_and_projection_retrieval.py # 门控混合与投影检索
├── 04-self_rag_trigger_and_minipatch.py     # 自监督触发与最小修订
├── 05-rrr_minipipeline_with_stopping.py     # RRR轻量流水线
├── 06-multi_source_fusion_resolver.py       # 多源融合与冲突裁决
├── 07-rrr_selfrag_gating_coop_demo.py       # RRR与Self RAG协同
├── 08-perf_cost_microbench.py               # 性能与成本优化演示
├── 09-evaluation_ablation_dashboard.py      # 消融实验与评估报表
├── 10-slo_compliance_security_incident_demo.py # SLO合规与安全响应
├── audit/                                   # 审计日志目录
│   └── 2025-10/                             # 按年月组织的审计记录
├── kb_corpus/                               # 知识库语料目录
│   ├── 01.txt
│   ├── 02.txt
│   ├── 03.txt
│   ├── 04.txt
│   └── 05.txt
├── runs/                                    # 评估运行结果
│   ├── baseline/                            # 基线运行结果
│   ├── full/                                # 全功能运行结果
│   ├── no_read/                             # 无Read组件运行结果
│   └── no_refine/                           # 无Refine组件运行结果
└── [各类配置、输入、输出文件]                 # 项目相关数据文件
```

## 子目录说明

| 目录名 | 作用描述 |
|-------|---------|
| `audit/` | 存储系统审计日志，按年月日组织，记录每次请求的SLO检查、安全扫描和操作建议等信息 |
| `kb_corpus/` | 存储知识库语料文本，用于检索和生成过程中的证据来源 |
| `runs/` | 存储评估运行结果，包含不同配置和变体的性能指标数据 |


### 01-context_budget_and_windowing_demo.py
**功能说明**：上下文预算器与滑窗切分器，为要点/引文/指令分配token预算，并进行语义切分与滑窗重叠。
**内容概述**：读取长文与标题层级，基于句号/标题规则切分；按窗口大小与重叠比生成chunks；对must-cover要点与证据优先级执行预算分配，输出"注入清单+预算统计"。
**调用说明**：直接运行，会处理当前目录下的相关输入文件。

### 02-table_to_text_with_anchors.py
**功能说明**：将CSV表格转结构化JSON与自然语言摘要，附单元格锚点。
**内容概述**：解析CSV，生成结构化表与口径/单位注释；产生"结论型摘要"；对关键单元格生成table_id/cell/desc锚点；导出JSON+Markdown双格式供演示。
**调用说明**：直接运行，会处理当前目录下的table.csv文件。

### 03-gated_mix_and_projection_retrieval.py
**功能说明**：用可调α融合"证据/语言"两路得分，模拟轻量投影检索。
**内容概述**：以词频/TF IDF近似"生成/检索"两路表征，合成分数：score=α*retrieval+(1-α)*generation；使用线性投影矩阵把查询映射到"检索空间"并返回Top K。
**调用说明**：直接运行，支持通过参数调整α值。

### 04-self_rag_trigger_and_minipatch.py
**功能说明**：基于不确定度/覆盖度/矛盾率触发K次检索，执行最小修订。
**内容概述**：以N best分歧度近似不确定度；must cover命中率近似覆盖度；基于规则近似矛盾率；每轮若越阈则检索样例库补证并patch指定段落；R≤2。
**调用说明**：直接运行，会处理draft.md、n_best.json等输入文件。

### 05-rrr_minipipeline_with_stopping.py
**功能说明**：RRR（Rank Read Refine）轻量流水线，以词频召回+特征加权重排、跨度抽取与压缩，按阈值迭代停机。
**内容概述**：多通道召回→特征融合排序→按要点抽取句/短语/单元格→Refine去冗保锚点→若覆盖度<(c_0)或冲突>(r_0)则新一轮，直至提升<(δ)停机。
**调用说明**：直接运行，会处理kb_corpus/目录下的文本文件和must_cover.json配置。

### 06-multi_source_fusion_resolver.py
**功能说明**：按权威度/新鲜度打分，执行语义去重与冲突裁决。
**内容概述**：对片段执行向量相似+编辑距离去重；对来源按"官方>权威媒体>行业媒体>个人"分级；结合时效判定优先；NLI like规则近似一致/矛盾；冲突保留裁决说明与双向引用。
**调用说明**：直接运行，会处理fragments.json输入文件。

### 07-rrr_selfrag_gating_coop_demo.py
**功能说明**：用RRR产出高密度证据，Self RAG自检回流，门控α随相似度调节。
**内容概述**：串联前面4-6的输出：RRR输出证据块→Self RAG检出缺口触发回流→依据相似度调节α（高相似提高证据占比）；展示一次闭环演示日志。
**调用说明**：直接运行，支持通过参数--alpha_base调整基础α值。

### 08-perf_cost_microbench.py
**功能说明**：模拟近邻缓存、批量重排、档位路由对P95延迟与成本的影响。
**内容概述**：构造简易请求流；实现LRU近邻缓存命中统计；批量重排吞吐对比；按"任务风险"将请求路由至不同档位（强/中/小模型占位），测量P50/P95与估算Token成本。
**调用说明**：直接运行，会处理traffic.ndjson和profiles.yaml配置文件。

### 09-evaluation_ablation_dashboard.py
**功能说明**：离线评估召回/覆盖/矛盾与压缩保真；输出消融与A/B报表。
**内容概述**：加载基线与不同变体（无Read/无Refine/全量），计算NDCG@K、覆盖度、冲突率、压缩率与保真度；生成雷达图数据与四象限Markdown报表。
**调用说明**：直接运行，会处理runs/目录下的metrics.json文件。

### 10-slo_compliance_security_incident_demo.py
**功能说明**：按档位核验SLO，输出回滚/回调建议；记录合规日志；模拟安全命中与处置。
**内容概述**：加载档位配置与分桶指标；判定是否达成SLO；触发"回滚/参数回调/灰度"建议；生成可回放审计日志；按安全规则检测PII/红线词/提示注入，套用处置矩阵。
**调用说明**：直接运行，会处理profiles.yaml、metrics.ndjson和security_rules.json配置文件。

## 主要数据文件说明

| 文件名 | 作用描述 |
|-------|---------|
| `budget.json` | 上下文预算配置文件 |
| `corpus.txt` | 原始语料文本 |
| `draft.md` | 草稿文档，用于Self RAG触发与修订 |
| `evaluation_metrics.json` | 评估指标汇总数据 |
| `evidence_store.json` | 证据存储库 |
| `fragments.json` | 多源片段数据 |
| `must_cover.json` | 必须覆盖的要点列表 |
| `n_best.json` | N-best变体结果 |
| `profiles.yaml` | 模型配置与SLO分档 |
| `security_rules.json` | 安全规则配置 |
| `table.csv` | 表格数据 |
| `traffic.ndjson` | 流量模拟数据 |

## 使用说明

1. 所有Python文件均支持直接运行，会自动检查和安装必要的依赖
2. 程序运行后会在当前目录生成相应的输出文件
3. 大部分程序都设计为独立运行，不需要按特定顺序执行
4. 如需修改参数配置，可以编辑相应的输入文件或通过命令行参数调整
