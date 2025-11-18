# LawRAG 项目结构与文件功能说明

## 项目概述

LawRAG是一个面向法律领域的检索增强生成（RAG）系统，旨在提供基于法律知识库的智能问答服务。该项目集成了向量检索、语言模型和重排序技术，能够根据用户提问从法律知识库中检索相关信息并生成准确的回答。

## 目录结构

```
d:\rag-project\08-LawRAG\
├── .env.example         # 环境变量示例文件
├── .gitignore          # Git忽略文件配置
├── RAG\                # 核心RAG功能模块
│   ├── TxtEmbedding.py # 嵌入模型实现
│   ├── LLM.py          # 语言模型接口
│   ├── Reranker.py     # 重排序模型实现
│   ├── VecDB.py        # 向量数据库接口
│   ├── __init__.py     # 包初始化文件
│   └── utils.py        # 工具函数集合

├── ChatRobot.py        # 聊天机器人实现
├── example.py          # 基础使用示例
├── example_with_reranker.py # 使用重排序模型的示例
├── paper.md            # 相关论文或文档
├── Questionary.py      # 问题分类器
├── requirements.txt    # 项目依赖列表
├── scripts\            # 脚本工具目录
│   ├── create.py       # 创建知识库脚本
│   └── create_kb.ipynb # 创建知识库的Jupyter Notebook
└── web_demo.py         # Web演示界面
```

## 目录功能说明

### RAG/ 目录

核心功能模块目录，包含RAG系统的基础组件：
- **TxtEmbedding.py**: 实现文本嵌入功能，将文本转换为向量表示
- **LLM.py**: 语言模型接口，负责生成回答
- **Reranker.py**: 重排序模型，对检索结果进行二次排序优化
- **VecDB.py**: 向量数据库接口，负责向量的存储和检索
- **utils.py**: 工具函数集合，包括文件读取、文本分块等功能

### scripts/ 目录

包含用于创建和管理知识库的脚本：
- **create.py**: 知识库创建脚本
- **create_kb.ipynb**: 交互式的知识库创建Jupyter Notebook

## 文件功能详细说明

### 核心配置文件

#### requirements.txt
项目依赖列表，包含所有必要的Python包和版本信息。

#### .env.example
环境变量配置示例文件，用于指导用户配置API密钥等敏感信息。

### RAG模块文件

#### RAG/TxtEmbedding.py
实现文本嵌入功能的核心模块：
- **BaseEmbeddings类**: 嵌入模型的抽象基类，定义了嵌入接口
- **BgeEmbedding类**: 基于BAAI/bge-base-zh-v1.5模型的中文嵌入实现
- **主要方法**:
  - `get_embedding()`: 将文本转换为向量表示
  - `cosine_similarity()`: 计算两个向量之间的余弦相似度

#### RAG/LLM.py
语言模型接口模块：
- **BaseModel类**: 语言模型的抽象基类
- **QwenModelChat类**: 基于阿里云Qwen模型的对话实现
- **主要方法**:
  - `chat()`: 与语言模型进行单轮对话
  - `batch_chat()`: 批量处理多轮对话
  - `generate_prompt()`: 生成适合模型输入的提示词

#### RAG/Reranker.py
重排序模型模块：
- **BaseReranker类**: 重排序模型的抽象基类
- **BgeReranker类**: 基于BAAI/bge-reranker-base模型的重排序实现
- **主要方法**:
  - `rerank()`: 根据查询和候选内容重新排序

#### RAG/VecDB.py
向量数据库接口模块：
- **VectorStore类**: 基于Milvus的向量存储和检索实现
- **ItemModel/ReturnModel类**: 数据模型定义
- **主要方法**:
  - `query()`: 根据查询在指定集合中检索相似内容

#### RAG/utils.py
工具函数集合：
- **ReadFiles类**: 文件读取和处理工具，支持PDF、Markdown、TXT格式
- **Documents类**: 处理JSON格式文档
- **主要功能**:
  - `get_content()`: 读取并处理文件内容
  - `get_chunk()`: 将长文本分割成多个块
  - 文件类型特定的读取方法

### 应用层文件

#### ChatRobot.py
聊天机器人实现：
- **ChatBotGraph类**: 集成了分类器、语言模型、嵌入模型和向量数据库的对话系统
- **主要方法**:
  - `chat_main()`: 处理用户输入并生成回答

#### web_demo.py
Web演示界面实现：
- 使用Streamlit构建的交互式Web界面
- 支持实时对话和历史记录管理
- 显示参考资料来源

#### Questionary.py
问题分类器实现：
- **QuestionClassifier类**: 通过关键词匹配对用户问题进行分类
- **主要方法**:
  - `classify()`: 对问题进行分类
  - `key_words_match_intention()`: 关键词匹配意图

### 示例文件

#### example.py
基础使用示例，展示如何：
- 创建和加载向量数据库
- 使用嵌入模型和语言模型
- 执行基础的检索和问答流程

#### example_with_reranker.py
包含重排序功能的使用示例，展示如何：
- 结合向量检索和重排序模型
- 优化检索结果的质量

## 调用关系说明

### 典型调用流程

1. **知识库创建流程**:
   - 使用`scripts/create_kb.ipynb`或`scripts/create.py`
   - 加载文档 → 文本分块 → 生成嵌入 → 存储到向量数据库

2. **问答流程**:
   - 用户提问 → 问题分类(`Questionary.py`)
   - 根据分类选择知识库 → 向量检索(`VecDB.py`)
   - 可选：重排序(`Reranker.py`)
   - 生成回答(`LLM.py`)

3. **Web交互流程**:
   - `web_demo.py` → `ChatRobot.py` → RAG核心组件

## 程序运行说明

### 1. 环境准备

#### 1.1 系统要求
- Python 3.8 或更高版本
- 建议至少 8GB RAM（处理大型语言模型时）
- 足够的磁盘空间用于存储向量数据库和模型文件

#### 1.2 安装依赖

1. 克隆项目代码：
```bash
git clone <项目仓库地址>
cd d:\rag-project\08-LawRAG
```

2. 创建虚拟环境（可选但推荐）：
```bash
python -m venv venv
# Windows
env\Scripts\activate
# Linux/Mac
# source venv/bin/activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

### 2. 环境变量配置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑`.env`文件，配置以下必要的环境变量：
```
# 嵌入模型相关
EMBEDDING_MODEL_PATH=/path/to/bge-large-zh-v1___5

# 语言模型相关
LLM_API_KEY=your_api_key_here
LLM_API_BASE=your_api_base_url

# Milvus向量数据库配置
MILVUS_URI=/path/to/milvus_law.db

# 重排序模型路径
RERANKER_MODEL_PATH=/path/to/bge-reranker-base
```

### 3. 知识库创建

#### 3.1 使用Jupyter Notebook创建知识库

1. 启动Jupyter Notebook：
```bash
jupyter notebook scripts/create_kb.ipynb
```

2. 按照notebook中的步骤逐步执行：
   - 加载法律模板数据（JSON格式）
   - 生成文本嵌入向量
   - 创建Milvus集合并插入数据

#### 3.2 使用Python脚本创建知识库

1. 准备数据：确保您的数据符合要求的格式（如JSON）

2. 运行创建脚本：
```bash
python scripts/create.py
```

### 4. 运行示例

#### 4.1 基础RAG示例

运行基础检索和问答示例：
```bash
python example.py
```

此示例演示了：
- 如何创建和持久化向量数据库
- 如何使用不同的嵌入模型（JinaEmbedding、ZhipuEmbedding）
- 如何执行检索并生成回答

#### 4.2 带重排序的RAG示例

运行结合重排序模型的示例：
```bash
python example_with_reranker.py
```

此示例演示了：
- 如何集成BgeReranker进行结果优化
- 先检索多个候选结果，再重排序选择最相关的内容
- 使用ZhipuChat生成基于重排序结果的回答

#### 4.3 命令行聊天机器人

运行命令行交互式聊天机器人：
```bash
python ChatRobot.py
```

功能特点：
- 支持问题分类，自动选择相关知识库
- 集成重排序模型优化检索结果
- 交互式命令行界面

#### 4.4 Web演示界面

启动基于Streamlit的Web演示：
```bash
streamlit run web_demo.py
```

Web界面功能：
- 友好的用户对话界面
- 显示参考资料来源
- 支持清空对话历史
- 实时响应和结果展示

### 5. 开发和使用注意事项

#### 5.1 模型路径配置

确保在`.env`文件中正确配置以下模型路径：
- 嵌入模型：BGE系列中文模型
- 重排序模型：BGE-Reranker模型

#### 5.2 向量数据库配置

- Milvus数据库默认使用本地文件路径存储
- 可根据需要配置为远程Milvus服务

#### 5.3 性能优化建议

- 对于大规模知识库，考虑增加Milvus的配置参数
- 在处理大量文档时，推荐使用批量处理方式
- 可调整检索结果数量和重排序阈值以优化性能和准确性

### 6. 常见问题与解决方案

#### 6.1 环境依赖问题

如果安装依赖时遇到问题，尝试更新pip并使用镜像源：
```bash
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 6.2 模型加载失败

确保模型路径正确，且有权限访问。对于大模型，可能需要更多内存。

#### 6.3 知识库检索不准确

- 检查嵌入模型配置是否正确
- 调整检索的top_k参数
- 启用重排序功能提升结果质量

#### 6.4 Web界面启动问题

如遇到Streamlit相关错误，尝试更新Streamlit：
```bash
pip install --upgrade streamlit
```

## 扩展说明

- **支持的模型**：
  - 嵌入模型：BGE系列
  - 语言模型：Qwen系列
  - 重排序模型：BGE-Reranker
  - 向量数据库：Milvus

- **支持的文件格式**：
  - PDF
  - Markdown (.md)
  - 文本文件 (.txt)
  - JSON

- **问题分类类型**：
  - legal_articles: 法律条款相关
  - legal_books: 法律书籍相关
  - legal_templates: 法律文书模板相关
  - legal_cases: 法律案例相关
  - JudicialExamination: 司法考试相关