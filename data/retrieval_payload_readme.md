# 检索Payload数据生成与合规管理

## 项目说明

本模块提供了一套完整的解决方案，用于从清洗后的文档生成符合规范的检索Payload数据，并严格执行"跨版本禁混"和"必须带引用ID/版本"的合规约束。

## 文件组成

1. **retrieval_payload.schema.json** - JSON Schema定义文件
2. **04-generate_retrieval_payload.py** - Python生成脚本
3. **retrieval_payload_readme.md** - 本说明文档

## 核心功能

### 1. JSON Schema定义 (retrieval_payload.schema.json)

该文件定义了检索Payload必须遵循的数据结构规范，包含以下关键字段：

- `doc_id`: 文档唯一标识符
- `title`: 文档标题
- `version`: 文档版本号（格式：vX.Y.Z）
- `creation_time`: 创建时间戳（ISO 8601格式）
- `last_update_time`: 最后更新时间戳
- `units`: 文档中使用的度量单位
- `tags`: 文档分类标签
- `entities`: 从文档中提取的命名实体
- `content`: 文档主要内容
- `source_ref`: 原始来源引用ID
- `compliance_flags`: 合规检查标志

### 2. Python生成脚本 (04-generate_retrieval_payload.py)

该脚本实现了以下核心功能：

- 读取清洗后的文档数据（从corpus.csv）
- 为每个文档生成符合Schema的Payload
- 实现版本控制机制
- 执行合规性检查
- 输出生成的Payload数据（到outputs/retrieval_payloads.jsonl）

## 关键合规约束说明

### 1. 跨版本禁混约束

**目的**：防止不同版本的文档内容被错误地混合使用，确保检索结果的一致性和准确性。

**实现机制**：
- 为每个文档分配唯一的版本号（vX.Y.Z格式）
- 维护文档版本历史记录
- 确保每个文档在生成时都使用正确的版本号
- 通过compliance_flags.cross_version_mixed标志监控跨版本混合情况

### 2. 必须带引用ID/版本约束

**目的**：确保所有文档都有可追溯的来源和明确的版本标识，满足合规性和审计要求。

**实现机制**：
- 自动为每个文档生成source_ref（格式：source_xxxxxxxxxx）
- 严格检查每个文档是否包含有效的version字段
- 通过compliance_flags.has_reference_id和compliance_flags.has_version标志监控合规性
- 统计不合规文档数量并输出报告

## 使用方法

### 环境要求

- Python 3.6+
- jsonschema库

### 安装依赖

```bash
pip install jsonschema
```

### 运行脚本

```bash
python 04-generate_retrieval_payload.py
```

### 输出结果

- 生成的Payload数据保存在`outputs/retrieval_payloads.jsonl`
- 控制台输出合规性统计信息
- 显示生成的Payload示例

## 合规性监控

脚本运行后会输出详细的合规性统计信息，包括：
- 总文档数
- 合规文档数
- 缺失引用ID的文档数
- 缺失版本的文档数
- 跨版本混合的文档数
- 总体合规率

## 数据流转说明

1. **数据输入**：清洗后的文档数据（corpus.csv）
2. **数据处理**：
   - 读取文档内容
   - 提取实体、单位和标签
   - 生成版本号和引用ID
   - 构建符合Schema的Payload
   - 执行合规性检查
3. **数据输出**：生成的Payload数据（JSON Lines格式）
4. **数据应用**：输出的Payload可直接用于检索系统

## 演示重点

在讲解本模块时，建议重点演示以下内容：

1. JSON Schema的结构和字段约束
2. 版本控制机制的实现
3. 合规性检查逻辑
4. 如何处理不同编码的输入文件
5. 生成的Payload数据示例
6. 合规性统计报告的解读

通过这些演示，可以帮助理解如何在实际应用中确保数据的规范性、可追溯性和合规性。