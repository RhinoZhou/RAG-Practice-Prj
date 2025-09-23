# RAG系统查询重放文件格式说明

## 1. 概述

本文档描述了RAG系统查询重放功能使用的JSON文件格式规范，用于批量测试和评估系统性能。

## 2. 基本格式

重放文件必须是一个JSON数组，其中每个元素代表一个要执行的查询。

```json
[
    {
        "query": "查询文本内容",
        "query_id": "唯一的查询ID",
        "user_profile": "用户画像JSON字符串",
        "top_k": 检索结果数量,
        "filters": {"过滤条件": "值"}
    },
    // 更多查询...
]
```

## 3. 字段说明

| 字段名 | 类型 | 是否必需 | 说明 |
|-------|------|----------|------|
| query | 字符串 | 必需 | 要查询的文本内容 |
| query_id | 字符串 | 可选 | 唯一标识查询的ID，如果不提供，系统会自动生成 |
| user_profile | 字符串 | 可选 | 用户画像的JSON字符串，用于个性化搜索 |
| top_k | 整数 | 可选 | 要返回的检索结果数量，默认为5 |
| filters | 对象 | 可选 | 过滤条件，用于限制检索范围 |

## 4. 示例

### 4.1 主样本示例 (main_sample.json)

```json
[
    {
        "query": "什么是RAG技术？",
        "query_id": "main_sample_001",
        "user_profile": "{\"domain\": \"人工智能\", \"expertise_level\": \"初级\", \"interests\": [\"大语言模型\", \"知识库\"], \"history\": [\"embedding技术\"]}",
        "top_k": 3
    },
    {
        "query": "如何构建高性能的向量检索系统？",
        "query_id": "main_sample_002",
        "user_profile": "{\"domain\": \"数据工程\", \"expertise_level\": \"中级\", \"interests\": [\"向量数据库\", \"检索算法\"], \"history\": [\"FAISS\"]}",
        "top_k": 5,
        "filters": {"source_type": "document"}
    }
]
```

### 4.2 失败样本示例 (failure_sample.json)

用于测试系统错误处理能力的示例：

```json
[
    {
        "query": "",
        "query_id": "failure_sample_001",
        "user_profile": "{\"domain\": \"测试\", \"expertise_level\": \"中级\"}"
    },
    {
        "query": "这是一个超长的查询字符串，...",
        "query_id": "failure_sample_002",
        "user_profile": "{\"domain\": \"测试\", \"expertise_level\": \"高级\"}",
        "top_k": 1000
    }
]
```

## 5. 使用方法

1. 将查询文件保存到 `replay/` 目录中
2. 运行以下命令进行批量重放：
   
   ```bash
   python scripts/replay.py --replay-dir replay --output-dir replay_out
   ```

3. 查看 `replay_out/` 目录中的结果文件

## 6. 输出结果

重放完成后，会在指定的输出目录中生成以下文件：

- `[文件名]_results.json`: 包含所有查询的详细结果
- `[文件名]_report.json`: 包含查询统计和性能分析
- `[文件名]/charts/`: 包含可视化图表（如果启用）

## 7. 注意事项

- 确保JSON格式正确，避免语法错误
- 用户画像必须是有效的JSON字符串（需要转义引号）
- 避免包含可能导致安全问题的内容
- 大批量查询时建议增加查询间隔时间