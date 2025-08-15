# 1. 准备文档数据（从本地文件读取）
import os
import re
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从本地文件读取内容
file_path = "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 简单分割文档为段落（可根据实际情况调整分割逻辑）
    # 这里使用空行分割，并过滤掉过短的段落
    docs = [para.strip() for para in re.split(r'\n\s*\n', content) if len(para.strip()) > 50]
    print(f"成功加载文档，分割为 {len(docs)} 个段落")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请检查路径是否正确")
    exit(1)
except Exception as e:
    print(f"读取文件时发生错误：{str(e)}")
    exit(1)

# 2. 设置嵌入模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)
print(f"文档向量维度: {doc_embeddings.shape}")

# 3. 创建向量存储
import faiss  # pip install faiss-cpu
import numpy as np

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"向量数据库中的文档数量: {index.ntotal}")

# 4. 执行相似度检索
question = "哪些部门负责促进民营经济发展的工作？"
query_embedding = model.encode([question])[0]
distances, indices = index.search(
    np.array([query_embedding]).astype('float32'),
    k=3
)
context = [docs[idx] for idx in indices[0]]
print("\n检索到的相关文档:")
for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc[:100]}...")  # 只显示前100字符

# 5. 构建提示词
prompt = f"""根据以下参考信息回答问题，并给出信息源编号。
如果无法从参考信息中找到答案，请说明无法回答。
参考信息:
{chr(10).join(f"[{i + 1}] {doc}" for i, doc in enumerate(context))}
问题: {question}
答案:"""

# 6. 使用DeepSeek生成答案（确保使用正确的API密钥）
from openai import OpenAI

# 获取并验证API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    # 使用备用API密钥
    api_key = "sk-XXXXXXXX"  # 替换为您的有效DeepSeek API密钥, 到 https://platform.deepseek.com/api_keys 申请
    print("警告：从环境变量加载API密钥失败，使用备用密钥")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1024
    )
    print(f"\n生成的答案: {response.choices[0].message.content}")
except Exception as e:
    print(f"调用DeepSeek API时发生错误：{str(e)}")
