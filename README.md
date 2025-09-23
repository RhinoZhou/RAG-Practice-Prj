# RAG系统实践与演示目录

本目录包含一系列RAG（检索增强生成）系统的实践演示程序和分析报告，涵盖从基础组件到高级功能的完整实现。

## 目录结构

```
├── 01-text_to_sql_dryrun.py        # SQL查询生成与安全执行
├── 02_text_to_cypher_safe.py       # Cypher查询生成与安全执行
├── 03-self_query_filter_demo.py    # 自查询过滤器演示
├── 03-self_query_filter_demo_analysis.md  # 自查询过滤器分析报告
├── 04-ivf_two_stage_demo.py        # IVF两阶段检索演示
├── 04-ivf_two_stage_demo_analysis.md      # IVF两阶段检索分析报告
├── 05-multi_recall_fuse_rerank.py  # 多路召回融合与重排序
├── 06-hot_cold_routing_demo.py     # 冷热数据路由演示
├── 06-hot_cold_routing_demo_analysis.md   # 冷热数据路由分析报告
├── 07-hyde_retrieval_minidemo.py   # HyDE检索最小演示
├── 07-hyde_retrieval_minidemo_analysis.md # HyDE检索分析报告
├── 07-hyde_retrieval_results.json  # HyDE检索结果数据
├── 08-query_dag_minimum.py         # 查询DAG最小系统
├── 08-query_dag_minimum_analysis.md       # 查询DAG系统分析报告
├── 08-query_dag_results.json       # 查询DAG执行结果
├── 09-metrics_and_policy_dashboard.py     # 指标监控与策略仪表盘
├── 09-metrics_and_policy_dashboard_analysis.md  # 指标监控分析报告
├── 09-metrics_and_policy_dashboard_analysis_optimized.md  # 指标监控优化分析报告
├── 10-rewrite_cache_dedup_combo.py # 重写、缓存与去重组合系统
├── 10-rewrite_cache_dedup_combo_analysis.md  # 重写缓存去重分析报告
├── ab_test_logs.json               # A/B测试日志数据
├── ivf_demo_visualization.png      # IVF检索可视化图
├── metrics_analysis_results.json   # 指标分析结果数据
├── metrics_comparison.png          # 指标对比图表
├── metrics_dashboard.log           # 指标仪表盘日志
├── mvp/                            # 最小可行产品目录
├── rewrite_cache_dedup.log         # 重写缓存去重系统日志
└── rewrite_cache_dedup_stats.json  # 重写缓存去重统计数据
```

## 子目录说明

### mvp/
最小可行产品（Minimum Viable Product）目录，包含一个完整的RAG系统实现，包括以下子组件：

- **app/**: 核心应用代码，包含查询处理、检索、重排、缓存等功能模块
- **config/**: 配置文件，包括枚举定义和路由规则
- **data/**: 数据目录，包含FAQ、政策和SOP等示例数据
- **index/**: 索引存储目录
- **logs/**: 系统日志文件
- **replay/**: 回放测试数据，用于系统验证
- **scripts/**: 辅助脚本，用于构建索引和准备数据
- **tests/**: 单元测试代码
- **ui/**: 用户界面代码
- **requirements.txt**: 项目依赖文件

## 编号Python文件说明

### 01-text_to_sql_dryrun.py
**功能**: 将自然语言查询转换为SQL查询并执行"dry run"（安全测试运行）

**核心功能**: 
- 实现文本到SQL的转换逻辑
- 添加安全检查机制防止SQL注入
- 支持查询验证和语法检查
- 提供详细的执行日志和错误处理

**调用方式**: `python 01-text_to_sql_dryrun.py`

### 02_text_to_cypher_safe.py
**功能**: 将自然语言查询转换为Cypher查询（图数据库查询语言）并安全执行

**核心功能**: 
- 文本到Cypher查询的转换
- 图数据库查询的安全验证
- 查询结果格式化输出
- 异常处理和日志记录

**调用方式**: `python 02_text_to_cypher_safe.py`

### 03-self_query_filter_demo.py
**功能**: 演示自查询过滤器（Self-Query Filter）的工作原理和应用

**核心功能**: 
- 实现基于元数据的查询过滤
- 支持动态条件构建
- 提供交互式查询示例
- 展示过滤效果统计

**调用方式**: `python 03-self_query_filter_demo.py`

### 04-ivf_two_stage_demo.py
**功能**: 演示IVF（倒排文件索引）两阶段检索算法的实现和性能

**核心功能**: 
- 构建IVF索引
- 实现两阶段检索流程
- 性能对比和可视化
- 参数调优示例

**调用方式**: `python 04-ivf_two_stage_demo.py`

### 05-multi_recall_fuse_rerank.py
**功能**: 实现多路召回融合与重排序机制

**核心功能**: 
- 支持多种召回策略并行执行
- 实现结果融合算法
- 提供多种重排序模型
- 性能评估和对比

**调用方式**: `python 05-multi_recall_fuse_rerank.py`

### 06-hot_cold_routing_demo.py
**功能**: 演示冷热数据路由策略的实现

**核心功能**: 
- 冷热数据识别与分类
- 动态路由决策机制
- 性能优化和资源分配
- 路由效果监控

**调用方式**: `python 06-hot_cold_routing_demo.py`

### 07-hyde_retrieval_minidemo.py
**功能**: 实现HyDE（Hypothetical Document Embeddings）检索的最小演示

**核心功能**: 
- 基于假设文档的嵌入生成
- 倒排索引构建与查询扩展
- TF-IDF计算与相关性排序
- Hit@K评估指标计算

**调用方式**: `python 07-hyde_retrieval_minidemo.py`

### 08-query_dag_minimum.py
**功能**: 实现查询DAG（有向无环图）的最小系统，支持复杂查询执行计划

**核心功能**: 
- 意图分析与节点拆分
- DAG依赖图构建
- 拓扑排序执行引擎
- 结果合并与汇总

**支持节点类型**: self_query, sql, cypher, hybrid_retrieval, merge

**调用方式**: `python 08-query_dag_minimum.py`

### 09-metrics_and_policy_dashboard.py
**功能**: 实现指标监控与策略仪表盘，用于评估和优化检索系统

**核心功能**: 
- A/B测试指标聚合与对比
- 基于阈值的路由决策
- 告警机制和异常检测
- 可视化图表生成
- 执行效率优化配置

**调用方式**: 
- 默认模式: `python 09-metrics_and_policy_dashboard.py`
- 禁用可视化: 可通过修改代码中的`enable_visualization`参数实现

### 10-rewrite_cache_dedup_combo.py
**功能**: 实现查询重写、缓存和去重的组合系统

**核心功能**: 
- 查询规范化与同义词替换
- 编辑距离为1的拼写纠错
- L1缓存机制与命中短路
- MinHash风格签名去重
- 实时统计与性能监控

**调用方式**: `python 10-rewrite_cache_dedup_combo.py`

## 运行环境要求

- Python 3.x
- 主要依赖: numpy, matplotlib（部分程序需要）
- 所有程序均包含自动依赖检查和安装机制

## 使用说明

1. 确保安装了Python 3.x环境
2. 进入当前目录
3. 直接运行各个Python文件，如: `python 01-text_to_sql_dryrun.py`
4. 程序会自动检查并安装必要的依赖
5. 执行结果将显示在控制台，并可能生成相应的结果文件

## 实验结果文件

每个演示程序可能生成相应的结果文件，主要包括:
- **.json**: 包含详细的执行结果和统计数据
- **.md**: 实验分析报告，包含功能说明、执行结果和优化建议
- **.log**: 程序执行日志
- **.png**: 可视化图表（部分程序生成）

## 扩展开发指南

1. 可以基于现有演示程序进行功能扩展
2. mvp目录包含完整的RAG系统实现，可作为开发参考
3. 所有程序均采用模块化设计，便于修改和扩展
4. 建议遵循现有的代码风格和命名规范

## 注意事项

1. 部分程序需要连接数据库或索引服务，请确保相应服务可用
2. 默认配置适用于演示环境，实际应用中可能需要调整参数
3. 程序生成的日志和结果文件保存在当前目录
4. 执行效率优化配置可根据实际需求调整

## 免责声明

本目录中的程序和数据仅供学习和演示使用，不应直接用于生产环境。在实际应用前，请进行充分的测试和验证。