# 🏥 医疗知识问答系统

## 📖 项目概述

智能医疗助手，支持多知识库管理、多轮对话、普通语义检索和高级多跳推理。本系统通过创新的检索增强生成（RAG）技术，为用户提供准确、全面的医疗信息查询服务。

## 📁 文件结构

```
├── rag.py              # 主程序文件，包含核心功能实现
├── config.py           # 配置文件，存储各种参数设置
├── requirements.txt    # 依赖包列表
├── README.md           # 项目说明文档
├── knowledge_bases/    # 知识库目录
└── temp_outputs/       # 临时输出目录
```

## 🚀 安装部署

### 环境准备
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置设置
编辑 `config.py` 文件，配置以下参数：
- API 密钥和基础 URL
- 模型参数
- 知识库路径
- 其他系统参数

### 启动服务

#### 1. 命令行版本（推荐，稳定可靠）
```bash
python cli_rag_system.py
```

#### 2. Gradio界面版
```bash
python rag_gradio.py
```

**注意**：如果Gradio界面无法正常启动，请使用命令行版本。

## ✨ 系统功能

- 📚 **多知识库管理**：支持创建、删除、切换多个知识库
- 📄 **文件上传**：支持 TXT 和 PDF 格式文件的批量上传
- 🔍 **语义检索**：基于向量的高效语义检索
- 🔄 **多跳推理**：创新的多轮推理机制，深度挖掘知识库信息
- 💬 **多轮对话**：支持连续对话，保持上下文理解
- 📊 **表格展示**：支持以表格形式展示查询结果
- 🌐 **网页搜索**：可选集成网页搜索功能，扩展信息来源

## 🧠 核心算法

### 1. 语义分块

系统采用基于语义的文本分块算法，确保每个文本块的完整性和相关性：

```python
def split_text_semantically(text: str, chunk_size=800, chunk_overlap=20) -> List[dict]:
    """语义化文本分块"""
    # 使用LlamaIndex的SentenceSplitter进行智能分块
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    
    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            "id": i + 1,
            "chunk": node.text,
            "method": "semantic"
        })
    
    return chunks
```

### 2. 多跳推理RAG系统

ReasoningRAG类实现了创新的多跳推理机制，能够深度挖掘知识库中的信息：

```python
class ReasoningRAG:
    def __init__(self, 
                 index_path: str, 
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = AppConfig.llm_model,
                 verbose: bool = False):
        """初始化多跳推理RAG系统"""
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_hops = max_hops
        self.initial_candidates = initial_candidates
        self.refined_candidates = refined_candidates
        self.reasoning_model = reasoning_model
        self.verbose = verbose
        
        self._load_resources()

    def _generate_reasoning(self, 
                           query: str, 
                           retrieved_chunks: List[Dict[str, Any]], 
                           previous_queries: List[str] = None,
                           hop_number: int = 0) -> Dict[str, Any]:
        """
        为检索到的信息生成推理分析并识别信息缺口
        
        返回包含以下字段的字典:
            - analysis: 对当前信息的推理分析
            - missing_info: 已识别的缺失信息
            - follow_up_queries: 填补信息缺口的后续查询列表
            - is_sufficient: 表示信息是否足够的布尔值
        """
        # 实现推理逻辑...

    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """执行多跳检索并生成答案"""
        # 实现多跳检索和回答逻辑...
```

### 3. 向量化与检索

系统使用高效的向量搜索引擎，支持快速准确的语义检索：

```python
def vectorize_query(query, model_name=AppConfig.embedding_model_name, batch_size=AppConfig.batch_size) -> np.ndarray:
    """向量化查询文本"""
    # 实现向量化逻辑...


def vector_search(query, index_path, metadata_path, limit):
    """执行向量搜索"""
    # 实现搜索逻辑...
```

## 🔌 API 集成

系统支持与多种API集成，包括：

1. **嵌入模型API**
   ```python
   # 配置示例
   EMBEDDING_BASE_URL = "https://api.deepseek.com/v1"
   EMBEDDING_API_KEY = "your_api_key"
   EMBEDDING_MODEL_NAME = "deepseek-ai/deepseek-embed-flash-v1"
   ```

2. **LLM API**
   ```python
   # 配置示例
   LLM_BASE_URL = "https://api.deepseek.com/v1"
   LLM_API_KEY = "your_api_key"
   LLM_MODEL = "deepseek-chat"
   ```

## 📖 使用指南

### 1. 知识库管理

- **创建知识库**：在"知识库管理"标签页中输入名称并点击"创建知识库"
- **上传文件**：选择一个知识库，点击"选择文件"上传TXT或PDF文件
- **删除知识库**：从下拉列表中选择知识库，点击"删除知识库"

### 2. 信息查询

- **普通查询**：在"对话"标签页中输入问题，点击"提交"
- **多跳推理**：勾选"启用多跳推理"选项，获取更深度的答案
- **表格展示**：勾选"表格形式展示"选项，以表格形式查看结果

### 3. 高级功能

- **网页搜索**：勾选"启用网页搜索"选项，扩展信息来源
- **查看检索过程**：点击"检索"标签页，查看检索和推理的详细过程

## 💻 代码功能详细解释

### 1. 配置管理 (AppConfig)

AppConfig类提供统一的配置管理，确保系统参数的一致性和可维护性：

```python
class AppConfig:
    # 检索参数
    chunk_size = 800
    chunk_overlap = 20
    batch_size = 128
    top_k = 5
    
    # 生成器参数
    temperature = 0.3
    max_tokens = 2048
    
    # API配置
    llm_base_url = "https://api.deepseek.com/v1"
    llm_api_key = "your_api_key_here"
    llm_model = "deepseek-chat"
    embedding_base_url = "https://api.deepseek.com/v1"
    embedding_api_key = "your_api_key_here"
    embedding_model_name = "deepseek-ai/deepseek-embed-flash-v1"
    
    # 知识库配置
    knowledge_base_root = "knowledge_bases"
    default_knowledge_base = "default"
    temp_output_dir = "temp_outputs"
```

### 2. DeepSeekClient类

DeepSeekClient类封装了与DeepSeek API的交互，提供统一的接口：

```python
class DeepSeekClient:
    def generate_response(self, system_prompt, user_prompt, model=AppConfig.llm_model):
        """
        使用DeepSeek API生成响应
        
        参数:
            system_prompt: 系统提示，定义模型角色和行为
            user_prompt: 用户输入的提示
            model: 使用的模型名称
            
        返回:
            生成的文本响应
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=AppConfig.temperature,
                max_tokens=AppConfig.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API调用错误: {str(e)}")
            return f"生成响应时发生错误: {str(e)}"
```

### 3. 知识库管理功能

系统提供完整的知识库管理功能，包括创建、删除、查询等操作：

```python
def get_knowledge_bases() -> List[str]:
    """获取所有知识库名称"""
    if not os.path.exists(KB_BASE_DIR):
        os.makedirs(KB_BASE_DIR)
    return [d for d in os.listdir(KB_BASE_DIR) if os.path.isdir(os.path.join(KB_BASE_DIR, d))]


def create_knowledge_base(kb_name: str) -> str:
    """创建新的知识库"""
    # 实现创建逻辑...


def delete_knowledge_base(kb_name: str) -> str:
    """删除知识库"""
    # 实现删除逻辑...
```

### 4. 文本处理功能

系统提供丰富的文本处理功能，包括语义分块、向量化、向量检索等：

```python
def split_text_semantically(text: str, chunk_size=800, chunk_overlap=20) -> List[dict]:
    """语义化文本分块"""
    # 实现分块逻辑...


def vectorize_file(data_list, output_file_path, field_name="chunk"):
    """向量化文件内容"""
    # 实现向量化逻辑...


def build_faiss_index(vector_file, index_path, metadata_path):
    """构建FAISS索引"""
    # 实现索引构建逻辑...


def vector_search(query, index_path, metadata_path, limit):
    """执行向量搜索"""
    # 实现搜索逻辑...
```

### 5. ReasoningRAG类

ReasoningRAG类实现了创新的多跳推理机制，能够深度挖掘知识库中的信息：

```python
class ReasoningRAG:
    def __init__(self, 
                 index_path: str, 
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = AppConfig.llm_model,
                 verbose: bool = False):
        """初始化多跳推理RAG系统"""
        # 初始化配置...

    def _generate_reasoning(self, 
                           query: str, 
                           retrieved_chunks: List[Dict[str, Any]], 
                           previous_queries: List[str] = None,
                           hop_number: int = 0) -> Dict[str, Any]:
        """生成推理分析"""
        # 实现推理逻辑...

    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """执行多跳检索并生成答案"""
        # 实现检索和回答逻辑...
```

### 6. Gradio界面

系统使用Gradio构建了友好的Web界面，支持知识库管理和信息查询：

```python
# 创建Gradio界面
with gr.Blocks(title="医疗知识问答系统", css=custom_css) as demo:
    gr.Markdown("""
        # 医疗知识问答系统
        **智能医疗助手，支持多知识库管理、多轮对话、普通语义检索和高级多跳推理**  
        本系统支持创建多个知识库，上传TXT或PDF文件，通过语义向量检索或创新的多跳推理机制提供医疗信息查询服务。
        """)
    
    # 使用State来存储对话历史
    chat_history_state = gr.State([])
    
    # 创建标签页
    with gr.Tabs() as tabs:
        # 知识库管理标签页
        with gr.TabItem("知识库管理"):
            # 实现知识库管理界面...
        
        # 对话标签页
        with gr.TabItem("对话"):
            # 实现对话界面...
        
        # 检索标签页
        with gr.TabItem("检索"):
            # 实现检索界面...

# 启动界面
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

### 7. 主要功能流程

#### 文档上传处理流程

```
1. 用户选择知识库和文件
2. 系统提取文件文本内容
3. 对文本进行语义分块
4. 对分块内容进行向量化
5. 构建FAISS索引
6. 保存索引和元数据
```

#### 问题回答流程

```
1. 用户输入问题
2. 系统对问题进行向量化
3. 执行向量检索获取相关文本块
4. （可选）执行多跳推理深入挖掘信息
5. 使用LLM生成回答
6. 展示回答和检索过程
```

## 许可证

本项目采用 MIT 许可证。详情请见 LICENSE 文件。