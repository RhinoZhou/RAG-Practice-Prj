# 复杂检索策略与多模态RAG实践目录

## 目录结构概览

本目录包含了一系列用于复杂检索策略、多模态检索增强生成(RAG)、智能调度与监控的Python实现。这些脚本展示了从图搜索融合、拓扑加权、上下文重写到多代理协作等高级RAG技术。

```
├── 01-graph_search_fusion_demo.py  # 图搜索融合演示
├── 02-topology_weighting_compare.py  # 拓扑加权比较
├── 03-context_rewrite_fsm_demo.py  # 上下文重写有限状态机演示
├── 04-table_field_lookup.py  # 表格字段查找
├── 05-multimodal_anchor_ranking.py  # 多模态锚点排序
├── 06-agent_scheduler_signal_policy.py  # 代理调度信号策略
├── 07-heuristic_dp_planner.py  # 启发式动态规划规划器
├── 08-multi_agent_protocol_demo.py  # 多代理协议演示
├── 09-tier_grid_ab_report.py  # 分层网格A/B测试报告
├── 10-mm_guardrail_actions.py  # 多模态护栏动作建议
├── 各种CSV和JSON数据文件  # 测试数据、配置和输出文件
```

## Python脚本功能详解

### 01-graph_search_fusion_demo.py
**功能说明**：演示基于图结构的搜索和结果融合技术，将知识图谱与传统检索方法结合，提高查询结果的准确性。

**调用说明**：直接运行 `python 01-graph_search_fusion_demo.py`

**主要特性**：
- 实现基于图的节点和边权重计算
- 支持最佳答案节点识别（如DDI严重程度指南）
- 融合多源检索结果

### 02-topology_weighting_compare.py
**功能说明**：比较不同拓扑加权算法在知识图谱检索中的性能表现，生成对比报告。

**调用说明**：直接运行 `python 02-topology_weighting_compare.py`

**输出**：`topology_weighting_comparison.csv` - 不同加权策略的性能对比

### 03-context_rewrite_fsm_demo.py
**功能说明**：使用有限状态机(FSM)实现上下文重写功能，优化查询表达以提高检索质量。

**调用说明**：直接运行 `python 03-context_rewrite_fsm_demo.py`

**输出**：`rewrite_evaluation.csv` - 重写质量评估结果

### 04-table_field_lookup.py
**功能说明**：提供结构化表格数据的高效字段查找功能，支持复杂查询和过滤。

**调用说明**：直接运行 `python 04-table_field_lookup.py`

**输入**：支持多种格式的表格数据（CSV等）

### 05-multimodal_anchor_ranking.py
**功能说明**：实现多模态内容（文本、图像、音频等）的锚点排序算法，建立不同模态内容之间的关联。

**调用说明**：直接运行 `python 05-multimodal_anchor_ranking.py`

**主要特性**：
- 多模态特征提取和融合
- 锚点关联强度计算
- 排序结果优化

### 06-agent_scheduler_signal_policy.py
**功能说明**：实现基于信号策略的代理调度系统，根据系统状态动态调整代理行为。

**调用说明**：直接运行 `python 06-agent_scheduler_signal_policy.py`

**输出**：
- `agent_policy_summary.txt` - 代理策略摘要
- `policy_trajectory.json` - 策略执行轨迹
- `session_states.csv` - 会话状态记录

### 07-heuristic_dp_planner.py
**功能说明**：使用启发式动态规划算法进行路径规划，优化多步骤决策过程。

**调用说明**：直接运行 `python 07-heuristic_dp_planner.py`

**输出**：`planning_results.json` - 规划结果

### 08-multi_agent_protocol_demo.py
**功能说明**：演示多代理协作协议，实现检索/校验/整合三代理的协作时序与去重控制。

**调用说明**：`python 08-multi_agent_protocol_demo.py --max-steps 5 --dedup on`

**参数说明**：
- `--max-steps`：最大执行步数
- `--dedup`：是否启用去重（on/off）

**输出**：
- `sequence_log.csv` - 时序日志
- `protocol_metrics.json` - 协议指标

### 09-tier_grid_ab_report.py
**功能说明**：实现分层网格试验与A/B测试显著性报告生成，帮助选择最优策略档位。

**调用说明**：`python 09-tier_grid_ab_report.py --grid grid.json --ab control.csv treatment.csv --alpha 0.05`

**参数说明**：
- `--grid`：网格配置文件路径
- `--ab`：A/B测试数据文件（控制组和处理组）
- `--alpha`：显著性水平（默认0.05）

**输出**：
- `tier_ab_report.csv` - A/B测试结果表格
- `tier_ab_report.md` - 详细的Markdown格式报告

### 10-mm_guardrail_actions.py
**功能说明**：实现跨模态SLO监控、异常检测与动作建议系统，提供降级与回滚策略。

**调用说明**：`python 10-mm_guardrail_actions.py --slo guardrails.json --window 15min`

**参数说明**：
- `--slo`：监控阈值配置文件路径
- `--window`：滑动窗口大小

**输出**：
- `alerts.csv` - 告警信息
- `action_plan.csv` - 动作计划
- `audit_log.csv` - 审计日志

## 数据文件说明

### 配置文件
- `guardrails.json` - SLO监控阈值配置
- `conf.json` - 通用配置
- `schema.json` - 数据结构定义
- `edges.json` - 图结构边信息
- `node_attributes.json` - 图节点属性

### 输入数据文件
- `timeseries.csv` - 时间序列监控数据
- `text.csv` - 文本数据
- `ocr.json` - OCR识别结果
- `asr.json` - 语音识别结果
- `dialog.json` - 对话数据
- `case.json` - 案例数据

### 输出结果文件
- `evidence_paths.csv` - 证据路径
- `protocol_metrics.json` - 协议指标
- `query_results.json` - 查询结果
- `ranked_results.csv` - 排序结果
- `citations.csv` - 引用信息
- `ndcg_compare.csv` - NDCG对比结果

## 使用指南

1. **环境准备**：确保安装了必要的依赖包
   ```
   pip install pandas numpy
   ```

2. **运行示例**：按照各脚本的调用说明直接运行

3. **自定义配置**：可以修改相应的JSON配置文件以适应特定需求

4. **查看结果**：每个脚本会生成相应的输出文件，可用于进一步分析

## 高级功能说明

### 多模态融合技术
- 支持文本、图像、音频等多种模态数据的整合
- 实现模态间的锚点关联和互增强

### 图搜索与推理
- 基于知识图谱的深度搜索算法
- 支持复杂逻辑推理和路径规划

### 智能调度与监控
- 自适应代理调度策略
- 实时性能监控和异常检测
- 自动降级和回滚机制

### A/B测试与优化
- 网格搜索策略优化
- 统计显著性检验
- 自动化报告生成

## 注意事项

- 部分脚本可能需要特定的数据文件，如果缺少会自动生成测试数据
- 建议在运行前查看脚本中的注释以了解详细的参数设置和使用方法
- 所有输出文件均采用UTF-8编码，确保中文显示正常

## 作者信息

主要实现由 Ph.D. Rhino 及团队完成，包含多种高级RAG技术和智能系统设计。