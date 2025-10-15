# RAG系统评估与优化工具箱

## 目录简介

本目录包含一系列用于RAG（检索增强生成）系统评估、诊断和优化的Python工具脚本。这些工具可以帮助开发者和研究人员全面评估RAG系统的性能，识别潜在问题，并实施针对性的优化策略。

当前目录下共有9个主要评估优化工具（以数字开头的Python文件）以及相关的数据文件和输出文件。

## 主要工具说明

### 01-eval_binary_metrics_and_errors.py
**程序名称**：二分类评估与错例回放
**功能说明**：计算 Precision/Recall/F1/Precision@K 等二分类评估指标，并导出错例清单与PR曲线数据点。
**执行流程**：
- 读取CSV（字段：query、doc_id、score、label[0-4]、rank）并按query聚合
- 设置相关性阈值（如 label≥3）计算TP/FP/FN/TN及Precision/Recall/F1
- 计算Precision@K和Recall@K（K可配置，多K列表）
- 生成PR曲线数据点（按score扫描阈值）并输出为CSV
- 导出错例清单（高分却不相关、低分却相关）
**调用说明**：`python 01-eval_binary_metrics_and_errors.py --rel-th 3 --ks 1,3,5,10`
**输出文件**：pr_points.csv、fp_top.csv、fn_low.csv

### 02-eval_ranking_metrics.py
**程序名称**：排序指标批量评估
**功能说明**：计算 NDCG@K、MRR、Hit@K 等排序评估指标，输出随K变化的曲线数据。
**执行流程**：
- 读取CSV（query、rank、score、label、bucket）
- 针对每个bucket，计算NDCG@K（K列表）、MRR、Hit@K
- 汇总均值、分位（P50、P90）、方差
- 导出K-指标的曲线数据CSV（便于外部作图）
- 输出"前列不稳"的query清单（NDCG波动大）
**调用说明**：`python 02-eval_ranking_metrics.py --ks 3,5,10 --bucket-col bucket --rel-th 3`
**输出文件**：ndcg_curve.csv、hit_curve.csv、unstable_queries.csv

### 03-simulate_chunking_recall_tradeoff.py
**程序名称**：文本分块与重叠效果评估
**功能说明**：模拟不同窗口与重叠对召回与前列覆盖的影响（基于关键词近似）。
**执行流程**：
- 读取文档集合（自动生成示例数据），接受窗口与重叠参数
- 对每个query的关键词集进行匹配，记录命中chunk与位置
- 统计不同配置下的"命中chunk比例"和"前K片段命中率"
- 计算成本近似（片段数、总字符数）
- 输出配置-指标-成本的对照表
**调用说明**：`python 03-simulate_chunking_recall_tradeoff.py --win 256 --overlap 0.15 --k 5 --queries queries.csv`
**输出文件**：chunking_grid_summary.csv

### 04-dynamic_topk_policy.py
**程序名称**：基于复杂度的Top-K选择
**功能说明**：根据查询复杂度信号（长度、实体计数、疑问词）动态给出K值与是否触发重排。
**执行流程**：
- 解析查询，提取特征（长度、数字/实体比、疑问词）
- 复杂度打分并映射到K档位（如 3/5/10）
- 应用触发规则（复杂度≥T 或 风险桶→触发重排）
- 估算代价：平均K与重排比例，输出预算对齐检查
- 导出每条query的决策与整体摘要
**调用说明**：`python 04-dynamic_topk_policy.py --rules rules.json --queries queries.csv --risk-bucket medical`
**输出文件**：dynamic_k_decisions.csv

### 05-controlled_rerank_simulator.py
**程序名称**：重排触发与前列增益评估模拟器
**功能说明**：用简单打分器模拟重排触发对Top-N前列命中率的增益与延迟影响。
**执行流程**：
- 读取候选列表数据（包含查询、文档、初始分数和特征）
- 根据设定的触发条件（相似度阈值或风险桶）决定是否触发重排
- 模拟重排函数，应用更精确的打分规则
- 比较重排前后Top-N命中率的变化
- 估算重排带来的延迟成本
- 生成受控重排收益报告
**调用说明**：`python 05-controlled_rerank_simulator.py --threshold 0.7 --topn 5 --features-config weights.json`
**输出文件**：rerank_gain_report.csv

### 06-citation_coverage_checker.py
**程序名称**：must-cover命中与引用三元校验
**功能说明**：校验回答是否覆盖 must-cover 要点，检查引用三元是否完整并可回放。
**执行流程**：
- 读取回答文本、must-cover要点清单和引用列表
- 逐条判断"要点是否被回答命中且引用有效"
- 统计覆盖率与回放成功率
- 列出缺口清单
**调用说明**：`python 06-citation_coverage_checker.py --answers answers.csv --must must_cover.csv`
**输出文件**：cover_gap_actions.csv

### 07-nli_consistency_approx.py
**程序名称**：规则化一致性与矛盾率估计
**功能说明**：用规则/字典近似检测"内部矛盾""与证据矛盾"，估算矛盾率。
**执行流程**：
- 载入反义/否定词表、单位换算映射
- 对回答句子两两比对，检测否定与数值冲突
- 与引用片段进行关键词与数字对齐校验
- 汇总矛盾条数/总句对，计算矛盾率
- 导出矛盾样例与修订建议
**调用说明**：`python 07-nli_consistency_approx.py --answers answers.csv --evidence evidence.csv`
**输出文件**：contradictions_examples.csv

### 08-ab_significance_and_power.py
**程序名称**：A/B检验与样本量计算器
**功能说明**：对两组指标进行统计检验（t检验或非参数检验），比较均值或中位数差异，并根据预期效应量计算所需样本量。
**执行流程**：
- 读取control.csv与treatment.csv（单列指标数据）
- 根据用户选择进行检验类型（t/非参）和假设方向（单/双侧）设置
- 计算统计指标，输出显著性判断
- 根据用户输入的效应量、方差、显著性水平和功效进行样本量估算
- 生成完整报告并提供多重比较校正提醒
**调用说明**：`python 08-ab_significance_and_power.py --test_type ttest --delta 0.03`
**输出文件**：ab_test_report.csv、ab_test_plot.png、control.csv、treatment.csv

### 09-end_to_end_eval_opt_cycle.py
**程序名称**：端到端评估诊断优化复盘
**功能说明**：整合多个评估步骤，展示自动化"评估→诊断→优化→再评估"流程，帮助用户系统地提升RAG系统性能。
**执行流程**：
- 模拟基础评估流程，计算关键指标（Recall、Precision、NDCG等）
- 分析评估结果，识别系统瓶颈（如Chunking策略、检索参数等）
- 执行优化策略（改进Chunking方法、调整动态Top-K参数、应用重排算法）
- 再次评估优化后的系统性能
- 生成详细的优化前后对比报告
- 提供部署建议
**调用说明**：`python 09-end_to_end_eval_opt_cycle.py`
**输出文件**：initial_evaluation.csv、optimized_evaluation.csv、evaluation_comparison.csv、optimization_results.png、optimized_chunking_params.json、optimized_dynamic_k_policy.json、optimized_rerank_config.json

## 数据文件说明

目录中包含多种数据文件，主要分为以下几类：

1. **示例数据文件**：如 `docs.json`、`queries.json`、`sample_docs.json` 等，用于程序测试和演示。
2. **评估输出文件**：如 `pr_points.csv`、`ndcg_curve.csv`、`hit_curve.csv` 等，包含各种评估指标和曲线数据。
3. **诊断分析文件**：如 `fp_top.csv`、`fn_low.csv`、`unstable_queries.csv` 等，记录诊断发现的问题和建议。
4. **优化配置文件**：如 `optimized_chunking_params.json`、`optimized_dynamic_k_policy.json` 等，包含优化后的系统参数配置。

## 使用建议

1. **初学者入门**：建议从 `09-end_to_end_eval_opt_cycle.py` 开始，了解完整的RAG评估优化流程。
2. **针对性评估**：根据具体需求选择相应的评估工具，如二分类评估、排序指标评估等。
3. **优化实施**：先使用诊断工具识别问题，再应用相应的优化策略，最后使用A/B检验验证效果。
4. **自定义扩展**：所有工具都支持参数配置，可以根据实际场景调整参数以获得更准确的评估结果。

## 注意事项

1. 所有程序均支持自动安装依赖包，但建议预先安装必要的库以提高执行效率。
2. 部分程序会自动生成示例数据，如需使用真实数据，请按照程序要求的格式准备数据文件。
3. 输出文件默认保存在当前目录，运行前请确保有写入权限。
4. 可视化图表中的中文显示已配置，但如遇显示问题，请检查matplotlib的字体设置。