import os
import re
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# 记录总耗时
total_start = time.time()

# ==============================================
# 1. 配置路径与验证资源
# ==============================================
start_time = time.time()

current_dir = Path(__file__).resolve().parent
DOC_PATH = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"
LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/all-MiniLM-L6-v2"  # 本地嵌入模型路径

# 验证资源
def validate_resources():
    if not DOC_PATH.exists():
        raise FileNotFoundError(f"文档不存在：{DOC_PATH}")
    if not LOCAL_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"嵌入模型不存在：{LOCAL_EMBEDDING_PATH}\n"
            f"下载命令：modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir 11_local_models/all-MiniLM-L6-v2"
        )

validate_resources()
resource_check_time = time.time() - start_time
print(f"【1/6】资源验证耗时：{resource_check_time:.4f}秒")


# ==============================================
# 2. 加载并分割文档
# ==============================================
start_time = time.time()

try:
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    docs = [para.strip() for para in re.split(r'\n\s*\n', content) if len(para.strip()) > 50]
    print(f"成功加载文档，分割为 {len(docs)} 个段落")
except Exception as e:
    print(f"读取文件错误：{str(e)}")
    exit(1)

doc_load_time = time.time() - start_time
print(f"【2/6】文档加载与分割耗时：{doc_load_time:.4f}秒")


# ==============================================
# 3. 本地嵌入模型初始化与向量生成
# ==============================================
start_time = time.time()

# 加载本地嵌入模型（禁止联网）
embedding_model = SentenceTransformer(
    str(LOCAL_EMBEDDING_PATH),
    local_files_only=True
)
doc_embeddings = embedding_model.encode(docs)
print(f"文档向量维度: {doc_embeddings.shape}")

embed_time = time.time() - start_time
print(f"【3/6】嵌入模型初始化与向量生成耗时：{embed_time:.4f}秒")


# ==============================================
# 4. 创建Faiss向量索引
# ==============================================
start_time = time.time()

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"向量数据库中的文档数量: {index.ntotal}")

faiss_time = time.time() - start_time
print(f"【4/6】Faiss向量索引创建耗时：{faiss_time:.4f}秒")


# ==============================================
# 5. 本地相似度检索
# ==============================================
start_time = time.time()

question = "哪些部门负责促进民营经济发展的工作？"
query_embedding = embedding_model.encode([question])[0]
distances, indices = index.search(
    np.array([query_embedding]).astype('float32'),
    k=3
)

context = [docs[idx] for idx in indices[0]]
print("\n检索到的相关文档:")
for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc[:100]}...")

retrieval_time = time.time() - start_time
print(f"【5/6】本地检索耗时：{retrieval_time:.4f}秒")


# ==============================================
# 6. 使用DeepSeek生成答案（核心保留）
# ==============================================
start_time = time.time()

# 构建提示词
prompt = f"""根据以下参考信息回答问题，并给出信息源编号。
如果无法从参考信息中找到答案，请说明无法回答。
参考信息:
{chr(10).join(f"[{i + 1}] {doc}" for i, doc in enumerate(context))}
问题: {question}
答案:"""

# 初始化DeepSeek客户端
api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXX"
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    print(f"\n生成的答案: {response.choices[0].message.content}")
except Exception as e:
    print(f"调用DeepSeek API时发生错误：{str(e)}")

deepseek_time = time.time() - start_time
print(f"【6/6】DeepSeek生成答案耗时：{deepseek_time:.4f}秒")


# ==============================================
# 总耗时统计
# ==============================================
total_time = time.time() - total_start
print(f"\n===== 全流程总耗时：{total_time:.4f}秒 =====")