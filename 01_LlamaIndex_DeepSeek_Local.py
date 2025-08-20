'''
    解决Hugging Face连接需要科学上网问题 使用本地已下载的模型
    先手动下载模型到本地，例如：https://huggingface.co/BAAI/bge-small-zh
        pip install modelscope
        modelscope download --model AI-ModelScope/bge-small-zh-v1.5 --local_dir ./00-My1stRAG/11_local_models    #替换为自己的目录
    然后指定本地路径，如："../11_local_models/bge-small-zh"
'''

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os
import time
import pathlib

# 加载环境变量
load_dotenv()

# 显式设置API密钥
api_key = "sk-XXX"                                                                         # 替换为您的有效DeepSeek API密钥
os.environ["DEEPSEEK_API_KEY"] = api_key

# 处理路径
current_dir = pathlib.Path(__file__).parent.resolve()
local_model_path = current_dir / "../11_local_models/bge-small-zh-v1.5"
local_model_path = str(local_model_path)

# 验证模型路径
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"模型路径不存在: {local_model_path}\n请检查模型文件夹是否正确放置")

# 配置嵌入模型 - 移除device_map，改用device参数并处理依赖
try:
    # 尝试不依赖accelerate的配置
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=local_model_path,
        model_kwargs={"device": "cpu"},  # 使用device而非device_map，避免需要accelerate
        trust_remote_code=True,
        tokenizer_kwargs={"local_files_only": True}
    )
except TypeError:
    # 某些版本可能不支持device参数，使用环境变量指定
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用CPU
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=local_model_path,
        trust_remote_code=True,
        tokenizer_kwargs={"local_files_only": True}
    )

# 配置DeepSeek作为LLM
Settings.llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7
)

# 1. 加载数据
start_time = time.time()
data_path = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"
documents = SimpleDirectoryReader(input_files=[str(data_path)]).load_data()
load_time = time.time() - start_time
print(f"1、加载数据时间: {load_time:.4f} 秒")

# 2. 构建索引
start_time = time.time()
index = VectorStoreIndex.from_documents(documents)
index_time = time.time() - start_time
print(f"2、构建索引时间: {index_time:.4f} 秒")

# 3. 创建问答引擎
start_time = time.time()
query_engine = index.as_query_engine()
engine_time = time.time() - start_time
print(f"3、创建问答引擎时间: {engine_time:.4f} 秒")

# 4. 回答问题
start_time = time.time()
query = "哪些部门负责促进民营经济发展的工作?"
response = query_engine.query(query)
answer_time = time.time() - start_time
print(f"4、回答问题时间: {answer_time:.4f} 秒\n")

# 输出结果
print(f"问题: {query}")
print(f"回答: {response}")
