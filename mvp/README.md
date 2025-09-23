# RAG演示系统

一个基于FastAPI和Streamlit的检索增强生成（RAG）演示系统，支持多种检索策略和可观测性。

## 项目结构

```
rag_demo/
  app/                  # 核心应用代码
  config/               # 配置文件
  data/                 # 数据目录
  index/                # 索引存储目录
  scripts/              # 工具脚本
  ui/                   # 用户界面
  tests/                # 测试用例
  requirements.txt      # 项目依赖
  README.md             # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将文档放入以下目录：
- `data/policy/`: 政策文档
- `data/faq/`: 常见问题
- `data/sop/`: 标准操作流程

### 3. 构建索引

```bash
# 准备语料
python -m scripts.prepare_corpus

# 构建BM25索引
python -m scripts.build_bm25

# 构建向量索引
python -m scripts.build_index
```

### 4. 启动API服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API服务启动后，可访问 http://localhost:8000/health 进行健康检查。

### API使用示例

使用curl发送查询请求示例：
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG技术？", "user_profile": "{\"domain\": \"人工智能\", \"expertise_level\": \"初级\"}", "top_k": 3}'
```

成功响应示例：
```json
{
  "answer": "RAG（检索增强生成）技术是一种结合信息检索和生成模型的AI技术...",
  "evidences": [
    {"id": "doc1", "title": "RAG技术概述", "content": "...", "score": 0.95},
    {"id": "doc2", "title": "检索增强生成应用", "content": "...", "score": 0.88}
  ],
  "reranked": [...],
  "query_id": "auto-generated-uuid"
}
```

### 5. 启动UI界面

```bash
streamlit run ui/app.py
```

UI界面启动后，可访问 http://localhost:8501 进行交互。

### UI使用指南（对应文档第39页步骤）

UI界面分为以下几个主要区块，对应演示流程步骤：

1. **功能开关区（左上角）** - 对应步骤1-3
   - 控制检索策略（向量/BM25/混合）
   - 启用/禁用查询重写
   - 启用/禁用结果重排
   - 设置是否使用缓存

2. **参数配置区** - 对应步骤4
   - top_k（检索结果数量）
   - 重排窗口大小
   - 温度参数
   - 最大生成长度

3. **输入区** - 对应步骤5-7
   - 问题输入框（主查询）
   - 用户画像JSON输入框
   - Request ID（可选）
   - "运行端到端管道"按钮

4. **结果展示区** - 对应步骤8-14
   - 原始查询与重写查询对比
   - 最终回答结果
   - 引用证据列表
   - 检索结果分布图表
   - 执行时间统计
   - 思考链展示
   - 错误提示区域

完整演示流程请按UI界面左侧引导步骤操作。

## 功能特点

- 多种检索策略：向量检索、BM25全文检索和混合检索
- 查询重写和自查询解析
- 检索结果重排
- 可观测性日志系统
- 缓存机制
- 完整的API和用户界面

## 环境变量配置

可在项目根目录创建`.env`文件配置环境变量：

```
# 数据和索引路径
DATA_DIR=data
INDEX_DIR=index

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs

# 检索配置
TOP_K=10
```

## 开发指南

### 运行测试

```bash
pytest tests/
```

### 代码规范

- 遵循PEP8规范
- 使用类型注解
- 添加适当的文档字符串

## 常见问题（FAQ）

### 1. 如何在仅CPU环境下运行？

系统支持纯CPU运行，但部分功能会受限：
1. 编辑配置文件 `config/routing_rules.yaml`，将所有模型设置为CPU兼容版本
2. 启动时添加环境变量：`FORCE_CPU=true python start_api.py`
3. 注意：纯CPU环境下，向量检索和生成速度会显著降低

### 2. 模型下载慢的解决办法？

1. 使用国内镜像源：
   ```bash
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```
2. 手动下载模型文件并放置到以下目录：
   ```
   ~/.cache/huggingface/hub
   ```
3. 配置环境变量使用代理：
   ```bash
   export HTTP_PROXY=http://proxy.example.com:port
   export HTTPS_PROXY=http://proxy.example.com:port
   ```

### 3. 如何使用降级开关？

当系统资源紧张或服务不稳定时，可以启用降级模式：

1. 基础降级（禁用重排和高级解析）：
   ```bash
   python start_api.py --simple-mode
   ```

2. 完全降级（仅使用BM25检索和基础生成）：
   编辑 `.env` 文件添加：
   ```
   DEGRADATION_MODE=full
   DISABLE_VECTOR_SEARCH=true
   DISABLE_RERANKER=true
   ```

### 4. 如何导入新文档？

1. 将新文档放入对应的数据目录：
   - 政策文档：`data/policy/`
   - 常见问题：`data/faq/`
   - 操作流程：`data/sop/`

2. 支持的文档格式：txt, md, pdf, docx

3. 重新构建索引：
   ```bash
   python scripts/prepare_corpus.py
   python scripts/build_bm25.py
   python scripts/build_index.py
   ```

4. 无需重启API/UI服务，新文档将立即生效

## License

MIT