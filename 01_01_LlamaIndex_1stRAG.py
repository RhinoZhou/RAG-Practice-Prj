# 导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 需要pip install llama-index-embeddings-huggingface
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek

from llama_index.core import Settings

# 加载环境变量
import os
from dotenv import load_dotenv

# 先确保正确加载环境变量
load_dotenv()  # 这应该放在最前面

# 显式设置API密钥（双重保障）
api_key = "sk-xxx-xxx"         # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
os.environ["DEEPSEEK_API_KEY"] = api_key

# 配置嵌入模型（使用本地模型，避免API问题）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh")

# 确保安装了必要的包
# pip install llama-index-llms-deepseek

# 创建 Deepseek LLM（显式传递API密钥）
llm = DeepSeek(
    model="deepseek-chat",  # 选择要使用的大预言模型
    api_key=api_key  # 直接传递API密钥，而不是依赖环境变量
)

# 加载数据
documents = SimpleDirectoryReader(
    input_files=["../20-Data/法律文档/中华人民共和国民营经济促进法.txt"]
).load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建问答引擎
query_engine = index.as_query_engine(llm=llm)

# 开始问答
try:
    response = query_engine.query("哪些部门负责促进民营经济发展的工作?")
    print(response)
except Exception as e:
    print(f"查询过程中发生错误: {str(e)}")
    # 检查API密钥是否有效
    if "Authentication" in str(e):
        print("请检查您的DeepSeek API密钥是否有效")
