# RAG系统文本分块实践项目

本项目是一个全面的RAG（Retrieval-Augmented Generation）系统文本分块技术实践集合，包含了从基础分块算法到高级策略编排的完整解决方案。

## 📁 目录结构

### 核心实验脚本（数字开头）

#### 基础实验与分析（01-06）
- **01-u_curve_chunksize_f1_latency_experiment.py** - 分块粒度-准确率-延迟倒U曲线实验
- **02_overlap_coverage_vs_index_scale_analysis.py** - 分块重叠-覆盖率-索引规模分析
- **03-min_evalset_builder_multiscenario.py** - 多场景最小验证集生成器
- **04-token_size_band_explorer_f1_ndcg_coverage.py** - Token尺寸区间探索与任务适配分析
- **05-_hierarchical_chunking_overlap_dynamic_boundary_demo.py** - 层次化分块与上下文突破演示
- **06-_answer_span_match_vs_chunk_granularity.py** - 答案跨度匹配度与块粒度关系评估

#### 分块算法实现（07-15）
- **07-_token_fixed_overlap_chunker.py** - 固定重叠Token分块器
- **08-recursive_paragraph_first_splitter.py** - 递归段落优先分割器
- **09-hierarchical_chunking_overlap_dynamic_boundary_demo.py** - 基于Markdown/PDF的结构化分块演示
- **10-pdf_layout_paragraph_table_chunker.py** - PDF版面特征辅助分块器
- **11-dialog_turn_detector.py** - 对话轮次检测器
- **12-markdown_structure_aware_chunker.py** - Markdown结构感知分块器
- **13-topic_cluster_boundary_chunker.py** - 主题聚类驱动边界分块器
- **14-coref_srl_boundary_guard.py** - 指代链与SRL守护边界分块器
- **15-ner_density_adaptive_chunking.py** - 实体密度自适应分块器

#### 高级技术与优化（16-26）
- **16-embed_layout_fusion_clustering_segments.py** - 嵌入与聚类驱动的边界发现演示
- **17-tiktoken_length_distribution_report.py** - Tiktoken精准计数与分布报告生成器
- **18-overlap_coverage_redundancy_evaluator.py** - 重叠率与信息覆盖评估工具
- **19-resource_adaptive_chunking_policy_simulator.py** - 资源自适应分块策略模拟器
- **20-chunk_metadata_schema_generator.py** - 分块元数据模板生成器
- **21-pyramid_index_with_provenance.py** - 分块溯源与层级索引实现
- **22-hybrid_chunking_strategy_orchestrator.py** - 混合分块策略编排器
- **24-multistage_index_two_pass_retrieval.py** - 多级索引与二次检索演示
- **25-strategy_params_to_metrics_mapper.py** - 分块策略参数到指标映射器
- **26-massive_pipeline_simulator_checkpointing.py** - 海量文档处理流水线模拟器（带断点续传）

### 辅助工具与分析脚本
- **analyze_*.py** - 各种分析工具脚本
- **check_*.py** - 检查和验证工具
- **visualize_*.py** - 可视化工具
- **checkpoint_recovery_demo.py** - 检查点恢复演示

### 数据目录
- **📁 data/** - 测试数据文件
  - `billionaires_page-1-5.pdf` - PDF测试文档
  - `medical_ai_dialog.txt` - 医疗AI对话数据
  - `大脑中动脉狭窄与闭塞致脑梗死的影像特点及发病机制的研究.pdf` - 中文医学论文

- **📁 results/** - 实验结果输出目录
  - `analysis/` - 分析结果子目录
  - 各种JSON、CSV、PNG格式的实验结果文件

- **📁 checkpoints/** - 检查点文件存储目录

- **📁 requirements/** - 依赖文件目录

### 配置与模板文件
- **chunk_metadata_schema.json** - 分块元数据模式
- **batch_chunk_metadata_templates.json** - 批量分块元数据模板
- **pyramid_index_example.json** - 金字塔索引示例
- **qa_pairs.json** - 问答对数据
- **corpus.txt** - 语料库文件

## 🚀 快速开始

### 环境准备

1. **安装基础依赖**：
```bash
pip install -r requirements.txt
```

2. **安装可选依赖**（根据需要）：
```bash
# 用于PDF处理
pip install pymupdf

# 用于高质量分词和NER
pip install spacy
python -m spacy download zh_core_web_sm

# 用于精确token计数
pip install tiktoken

# 用于向量化和聚类
pip install scikit-learn sentence-transformers
```

### 基础使用示例

#### 1. 运行基础分块实验
```bash
# 分块粒度实验
python 01-u_curve_chunksize_f1_latency_experiment.py

# 重叠率分析
python 02_overlap_coverage_vs_index_scale_analysis.py
```

#### 2. 生成验证数据集
```bash
# 生成多场景验证集
python 03-min_evalset_builder_multiscenario.py
```

#### 3. 使用特定分块器
```bash
# 使用Token固定重叠分块器
python 07-_token_fixed_overlap_chunker.py

# 使用对话轮次检测器
python 11-dialog_turn_detector.py
```

#### 4. 运行高级策略
```bash
# 混合分块策略编排
python 22-hybrid_chunking_strategy_orchestrator.py

# 海量文档处理模拟
python 26-massive_pipeline_simulator_checkpointing.py
```

## 📊 核心功能说明

### 实验类脚本
这些脚本主要用于验证分块策略的效果和性能：

- **性能评估**：F1分数、nDCG、覆盖率、延迟等指标
- **参数优化**：chunk_size、overlap、top-k等参数的最优配置
- **对比分析**：不同策略在各种场景下的表现对比

### 分块器实现
提供了多种分块算法的完整实现：

- **基础分块**：固定大小、重叠窗口
- **结构化分块**：基于段落、标题、对话轮次
- **语义分块**：基于主题聚类、实体密度、相似度
- **混合策略**：多种信号融合的智能分块

### 评估与优化工具
- **元数据生成**：标准化的分块元数据模板
- **性能监控**：资源使用情况和系统性能监控
- **质量评估**：覆盖率、冗余率、信息保留度评估

## 🔧 调用说明

### 通用调用模式

大多数脚本都遵循以下调用模式：

```python
# 1. 导入模块
from script_name import MainClass

# 2. 创建实例
processor = MainClass(config_params)

# 3. 处理文本
results = processor.process(text_input)

# 4. 保存结果
processor.save_results(results, output_path)
```

### 配置参数说明

常见的配置参数包括：
- `chunk_size`: 分块大小（字符数或token数）
- `overlap`: 重叠大小
- `output_dir`: 输出目录
- `corpus_file`: 语料文件路径
- `qa_file`: 问答对文件路径

### 输出格式

大多数脚本输出以下格式的结果：
- **JSON格式**：结构化的分块结果和元数据
- **CSV格式**：表格形式的统计数据
- **PNG格式**：可视化图表

## 📈 实验结果

所有实验结果都保存在 `results/` 目录下，包括：

- **分块结果**：各种策略的分块输出
- **性能指标**：F1、nDCG、覆盖率等评估结果
- **可视化图表**：性能对比图、分布图等
- **日志文件**：详细的执行日志和错误信息

## 🛠️ 开发指南

### 添加新的分块策略

1. 创建新的Python文件，遵循命名规范
2. 实现标准的分块接口
3. 添加配置类和元数据支持
4. 编写测试用例和文档

### 扩展评估指标

1. 在相应的评估脚本中添加新指标
2. 更新结果输出格式
3. 添加可视化支持

## 📝 注意事项

1. **依赖管理**：某些脚本需要特定的外部库，请根据错误提示安装
2. **数据路径**：确保测试数据文件存在于正确的路径
3. **内存使用**：处理大文件时注意内存使用情况
4. **编码问题**：所有文件都使用UTF-8编码

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。在提交代码前，请确保：

1. 代码符合项目的编码规范
2. 添加了适当的文档和注释
3. 通过了基本的测试

## 📄 许可证

本项目仅供学习和研究使用。

---

*最后更新：2025年1月*