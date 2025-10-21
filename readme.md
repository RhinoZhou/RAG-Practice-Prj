# RAG项目 - Fine-tuning 模块

## 目录结构说明

当前目录是RAG项目的Fine-tuning模块，包含模型训练、量化、推理优化和检索相关的实现。

### 主要目录

- **data/**: 包含各种数据集文件，如FAQ文档、问题变体、性能分析结果等
- **hybrid_retriever_configs/**: 混合检索器的配置文件
- **hybrid_retriever_data/**: 混合检索器的测试数据集
- **hybrid_retriever_results/**: 混合检索器的评估结果和可视化图表
- **lora_configs/**: LoRA微调配置文件
- **quantization_configs/**: 模型量化配置文件
- **quantized_models/**: 量化后的模型文件和性能评估结果
- **trainer_output/**: 训练器输出配置文件
- **trt_configs/**: TensorRT-LLM引擎配置文件
- **trt_data/**: TensorRT-LLM测试数据
- **trt_engines/**: TensorRT-LLM引擎文件
- **trt_results/**: TensorRT-LLM性能分析结果

## Python脚本功能说明

### 数据处理与准备

- **01-dataset_cleaning_demo.py**: 数据集清洗演示脚本，用于清洗和预处理训练数据
  - 功能：处理原始数据，去除噪声，标准化格式
  - 调用：`python 01-dataset_cleaning_demo.py`

- **02-dataset_split_demo.py**: 数据集分割演示脚本
  - 功能：将数据集划分为训练集、验证集和测试集
  - 调用：`python 02-dataset_split_demo.py`

### 模型训练与优化

- **03-lora_matrix_update_demo.py**: LoRA矩阵更新演示脚本
  - 功能：演示LoRA（Low-Rank Adaptation）微调技术中的权重矩阵更新过程
  - 调用：`python 03-lora_matrix_update_demo.py`

- **04-quantization_demo.py**: 模型量化演示脚本
  - 功能：演示模型量化技术，降低模型大小和加速推理
  - 调用：`python 04-quantization_demo.py`

- **05-training_monitor_demo.py**: 训练监控演示脚本
  - 功能：监控模型训练过程中的各种指标变化
  - 调用：`python 05-training_monitor_demo.py`

- **06-metrics_evaluation_demo.py**: 评估指标演示脚本
  - 功能：演示各种模型评估指标的计算和分析
  - 调用：`python 06-metrics_evaluation_demo.py`

- **07-loss_lr_schedule_demo.py**: 损失函数和学习率调度演示脚本
  - 功能：演示不同损失函数和学习率调度策略的效果
  - 调用：`python 07-loss_lr_schedule_demo.py`

### 推理优化

- **08-inference_cache_demo.py**: 推理缓存优化演示脚本
  - 功能：演示使用缓存技术加速模型推理过程
  - 调用：`python 08-inference_cache_demo.py`

- **09-api_service_demo.py**: API服务演示脚本
  - 功能：演示如何将模型部署为API服务
  - 调用：`python 09-api_service_demo.py`

### 数据增强

- **10-generate_negative_examples.py**: 负样本生成脚本
  - 功能：生成训练所需的负样本数据
  - 调用：`python 10-generate_negative_examples.py`

- **11-generate_question_variations.py**: 问题变体生成脚本
  - 功能：为FAQ问题生成多种表述变体，增强数据多样性
  - 调用：`python 11-generate_question_variations.py`

### 模型加载与训练

- **12-load_model_and_tokenizer.py**: 模型和分词器加载脚本
  - 功能：演示如何加载预训练模型和分词器
  - 调用：`python 12-load_model_and_tokenizer.py`

- **13-create_trainer.py**: 训练器创建脚本
  - 功能：创建和配置模型训练器
  - 调用：`python 13-create_trainer.py`

### 量化与优化

- **14-quantize_model_int8.py**: INT8模型量化脚本
  - 功能：将模型量化为INT8精度，进一步减小模型体积
  - 调用：`python 14-quantize_model_int8.py`

- **15-build_tensorrt_llm_engine.py**: TensorRT-LLM引擎构建脚本
  - 功能：构建TensorRT-LLM推理引擎，优化推理性能
  - 调用：`python 15-build_tensorrt_llm_engine.py`

### 检索优化

- **16-HybridRetriever.py**: 混合检索器实现
  - 功能：融合TF-IDF向量检索和BM25关键词检索的混合检索策略
  - 调用：`python 16-HybridRetriever.py`
  - 特点：通过RRF（倒数排名融合）算法综合检索结果，提高检索准确性

## 其他相关文件

- **dataset_cleaned.json**: 清洗后的数据集
- **formatted_qa_data.json**: 格式化的问答数据
- **sample_qa_data.txt**: 样本问答数据
- **train.json**: 训练集数据
- **valid.json**: 验证集数据
- **test.json**: 测试集数据

## 使用说明

1. 按照顺序运行数据处理脚本准备训练数据
2. 配置模型参数和训练参数
3. 运行训练脚本进行模型训练
4. 使用量化和优化脚本提升模型性能
5. 利用混合检索器脚本优化检索效果

所有脚本都包含详细的注释和说明，便于理解和修改。