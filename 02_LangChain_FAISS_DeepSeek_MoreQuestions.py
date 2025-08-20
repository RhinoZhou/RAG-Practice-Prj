import os
import time
import torch
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from sentence_transformers import SentenceTransformer as ST

def main():
    # 记录总初始化耗时
    start_total = time.time()

    # 加载环境变量
    load_dotenv()

    # 获取当前脚本路径，统一处理绝对路径
    current_dir = Path(__file__).resolve().parent

    # 配置本地模型路径（请确保已下载到这些路径）
    LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"
    # 文档路径
    DOCUMENT_PATH = current_dir / "../20-Data/法律文档/中华人民共和国民营经济促进法.txt"
    # FAISS向量库保存路径
    FAISS_DB_PATH = current_dir / "../23-faiss_db"

    # 验证本地模型是否存在
    if not LOCAL_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"本地嵌入模型不存在于路径: {LOCAL_EMBEDDING_PATH}\n"
            f"请先下载模型到该路径，可使用命令:\n"
            f"modelscope download --model AI-ModelScope/bge-small-zh-v1.5 --local_dir {LOCAL_EMBEDDING_PATH}"
        )

    # 验证文档是否存在
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"文档文件不存在: {DOCUMENT_PATH}")

    # 1. 加载文档
    start_time = time.time()
    loader = TextLoader(
        file_path=str(DOCUMENT_PATH),
        encoding='utf-8'  # 确保中文正常加载
    )
    docs = loader.load()
    load_time = time.time() - start_time
    print(f"1、文档加载耗时：{load_time:.4f}秒")

    # 2. 文档分块
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    all_splits = text_splitter.split_documents(docs)
    split_time = time.time() - start_time
    print(f"2、文档分块耗时：{split_time:.4f}秒，共分{len(all_splits)}块")

    # 3. 初始化本地嵌入模型
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model=str(LOCAL_EMBEDDING_PATH),
        model_kwargs={
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'local_files_only': True
        },
        encode_kwargs={
            'batch_size': 32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'normalize_embeddings': True
        }
    )

    # 获取设备信息（用于显示）
    temp_model = ST(str(LOCAL_EMBEDDING_PATH))
    device = temp_model.device
    del temp_model  # 释放资源

    embed_time = time.time() - start_time
    print(f"3、本地嵌入模型初始化耗时：{embed_time:.4f}秒，使用设备：{device}")

    # 4. 创建或加载FAISS向量存储
    start_time = time.time()
    if FAISS_DB_PATH.exists() and len(os.listdir(FAISS_DB_PATH)) > 0:
        vector_store = FAISS.load_local(
            str(FAISS_DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"加载已存在的FAISS向量库，路径：{FAISS_DB_PATH}")
    else:
        vector_store = FAISS.from_documents(all_splits, embeddings)
        FAISS_DB_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(FAISS_DB_PATH))
        print(f"新建并保存FAISS向量库，路径：{FAISS_DB_PATH}")

    vector_time = time.time() - start_time
    print(f"4、FAISS向量存储处理耗时：{vector_time:.4f}秒")

    # 5. 创建LLM模型
    start_time = time.time()
    api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXXX"

    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=2048,
        api_key=api_key
    )
    llm_time = time.time() - start_time
    print(f"5、创建LLM耗时：{llm_time:.4f}秒")

    # 6. 构建提示模板
    prompt = ChatPromptTemplate.from_template("""
    基于以下上下文，用中文简洁准确地回答问题。如果上下文中没有相关信息，
    请说"我无法从提供的上下文中找到相关信息"。
    上下文: {context}
    问题: {question}
    回答:"""
                                              )

    # 初始化完成提示
    total_init_time = time.time() - start_total
    print(f"\n===== 系统初始化完成（总耗时：{total_init_time:.4f}秒） =====")
    print("请输入问题进行咨询（输入'q'或'quit'退出程序）：")

    # 7. 循环处理用户输入
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入问题：").strip()

            # 检查退出条件
            if question.lower() in ['q', 'quit']:
                print("程序已退出，感谢使用！")
                break

            # 忽略空输入
            if not question:
                print("输入不能为空，请重新输入问题")
                continue

            # 记录单次查询耗时
            query_start = time.time()

            # 检索相关文档
            start_time = time.time()
            retrieved_docs = vector_store.similarity_search(question, k=3)
            retrieve_time = time.time() - start_time
            print(f"文档检索耗时：{retrieve_time:.4f}秒，返回{len(retrieved_docs)}条相关结果")

            # 准备上下文内容
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # 调用LLM生成答案
            start_time = time.time()
            answer = llm.invoke(prompt.format(question=question, context=docs_content))
            llm_time = time.time() - start_time
            print(f"答案生成耗时：{llm_time:.4f}秒")

            # 输出结果
            print("\n===== 回答结果 =====")
            print(f"问题：{question}")
            print(f"回答：{answer.content}")

            # 显示单次查询总耗时
            query_total_time = time.time() - query_start
            print(f"\n本次查询总耗时：{query_total_time:.4f}秒")

        except Exception as e:
            print(f"处理过程中发生错误：{str(e)}")
            print("请尝试重新输入问题")

if __name__ == "__main__":
    main()
