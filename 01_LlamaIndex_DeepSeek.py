# 1：准备环境
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 显式设置API密钥（双重保障）
api_key = "sk-XXXXXXXX"         # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
os.environ["DEEPSEEK_API_KEY"] = api_key

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# 创建 Deepseek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 2：加载数据
documents = SimpleDirectoryReader(input_files=["../20-Data/法律文档/中华人民共和国民营经济促进法.txt"]).load_data()

# 3：构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 4：创建问答引擎
query_engine = index.as_query_engine(
    llm=llm
)

# 5: 开始问答
print(query_engine.query("哪些部门负责促进民营经济发展的工作?"))
