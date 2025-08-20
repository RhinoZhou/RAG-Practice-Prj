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

# 记录总耗时
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
print(f"1、文档加载耗时：{load_time:.4f}秒")                                                                               # 文档加载耗时：0.0003秒

# 2. 文档分块
'''
 使用 递归字符分割技术，这种技术是处理非结构化文本（如文档、文章等）时最常用的分块策略之一，尤其适合中文等语言的文本分割。
 递归字符分割的核心思想是：
    按优先级使用分隔符：先尝试用优先级高的分隔符（如段落分隔）分割文本，若分割后的块大小仍超过设定阈值，则递归使用优先级低的分隔符（如句子分隔、短语分隔）继续分割。
    平衡块大小与语义完整性：通过保留自然语言中的语义单元（段落、句子、短语），在控制块大小的同时，最大限度减少对文本语义的破坏。
    （1）separators：分隔符列表（最关键参数）
    代码中配置了中文场景下的分隔符优先级：
        第一优先级：\n\n（段落分隔）→ 优先按段落分割，保留完整段落语义。
        第二优先级：\n（换行）→ 段落内按换行分割。
        第三优先级：。、！、？（中文句尾标点）→ 按句子分割。
        第四优先级：，、 （中文句中标点和空格）→ 按短语或词语分割。
    作用：递归使用分隔符，确保分块尽可能遵循自然语言的语义边界，减少 “切断句子”“拆分短语” 等破坏语义的情况。
    （2）chunk_size：块大小
        设置每个分块的最大字符数（这里是 1000）。超过这个阈值的文本会被进一步分割。
    （3）chunk_overlap：块重叠度
        设置相邻块之间的重叠字符数（这里是 200）。重叠的作用是：       
            避免关键信息被 “切断” 在两个块的边界上（如一个句子被拆分成两个块）。
            增强分块之间的上下文关联性，提升后续检索的准确性
'''
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,                                                                                                    # 每个块的最大字符数
    chunk_overlap=200,                                                                                                  # 块之间的重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]                                                          # 中文优先分割符，解决了中文与英文在标点和语义单元上的差异：
)
all_splits = text_splitter.split_documents(docs)                                                                        # 执行分块
split_time = time.time() - start_time
print(f"2、文档分块耗时：{split_time:.4f}秒，共分{len(all_splits)}块")                                                       # 文档分块耗时：0.0004秒，共分14块

# 3. 初始化本地嵌入模型，将Embedding模型，加载到内存中，并且设置好各种参数
start_time = time.time()

#初始化Embedding模型
'''
这段代码用于初始化一个基于 Hugging Face 模型的嵌入器（Embeddings），主要作用是将文本转换为向量表示：
1. 核心作用
    HuggingFaceEmbeddings 是 LangChain 提供的一个封装类，用于连接 Hugging Face 生态的预训练模型，将文本（如文档片段、问题等）转换为数值向量（嵌入向量）。这些向量能够反映文本的语义信息，后续可用于计算文本相似度（如检索相关文档）。
2. 参数详解
    （1）model=str(LOCAL_EMBEDDING_PATH)
        作用：指定用于生成嵌入向量的预训练模型路径（或模型名称）。
        这里使用的是本地模型路径（LOCAL_EMBEDDING_PATH），即提前下载到本地的 bge-small-zh-v1.5 模型（一个中文优化的嵌入模型）。
        若使用公开模型（如未下载到本地），可直接传入模型名称字符串（如 model="BAAI/bge-small-zh-v1.5"），程序会自动从 Hugging Face Hub 下载。
    （2）model_kwargs：模型初始化参数
        这是一个字典，用于配置模型加载时的参数，会传递给底层的 SentenceTransformer 模型（文本嵌入模型的核心实现）。
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        作用：指定模型运行的设备。
        优先使用 GPU（cuda）加速计算，若没有 GPU 则自动切换到 CPU（cpu）。
        嵌入模型的计算（尤其是批量处理时）在 GPU 上会显著更快。
        'local_files_only': True
        作用：限制模型仅从本地加载，不尝试从网络下载。
        避免因网络问题导致的加载失败，确保使用的是预先下载到 LOCAL_EMBEDDING_PATH 的模型文件。
    （3）encode_kwargs：编码（文本转向量）时的参数
        这是一个字典，用于配置将文本转换为向量时的计算参数。
        
        'batch_size': 32
        作用：指定批量处理的文本数量。
        批量处理能提高效率（尤其是 GPU 上），32 是一个兼顾速度和内存的常见值，可根据硬件配置调整（如内存大的设备可设为 64）。
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        作用：指定编码计算时使用的设备，与模型加载的设备保持一致即可。
        'normalize_embeddings': True
        作用：将生成的向量归一化（L2 归一化）。
        归一化后，向量的模长为 1，此时计算向量间的余弦相似度可简化为点积运算，提高检索效率。
3. 底层工作流程
    模型加载：根据 model 参数指定的路径，加载本地的 bge-small-zh-v1.5 模型到内存，并根据 model_kwargs 配置运行设备（CPU/GPU）。
    文本编码：当需要将文本转换为向量时（如处理文档分块或用户问题时），会使用 encode_kwargs 中的参数（如批量大小、归一化等）进行计算。
    向量输出：最终生成的嵌入向量将用于构建向量数据库（如 FAISS），或用于检索时的相似度计算。
'''
embeddings = HuggingFaceEmbeddings(
    model=str(LOCAL_EMBEDDING_PATH),                                                                                    # 使用model参数指定模型路径
    model_kwargs={                                                                                                      # 模型初始化参数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',                                                       # 指定模型运行的设备，优先使用 GPU（cuda）加速计算，若没有 GPU 则自动切换到 CPU（cpu）
        'local_files_only': True                                                                                        # 限制模型仅从本地加载，不尝试从网络下载
    },
    encode_kwargs={                                                                                                     # 编码（文本转向量）时的参数
        'batch_size': 32,                                                                                               # 批量处理的文本数量
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',                                                       # 指定编码计算时使用的设备
        'normalize_embeddings': True                                                                                    # 将生成的向量归一化
    }
)

# 获取设备信息（用于显示）
temp_model = ST(str(LOCAL_EMBEDDING_PATH))
device = temp_model.device
del temp_model  # 释放资源

embed_time = time.time() - start_time
print(f"3、本地嵌入模型初始化耗时：{embed_time:.4f}秒，使用设备：{device}")                                                     # 本地嵌入模型初始化耗时：7.9757秒，使用设备：cpu

# 4. 创建或加载FAISS向量存储
'''
1. 核心作用
    FAISS（Facebook AI Similarity Search）是一个高效的向量检索库，能够快速地从海量向量中找到与目标向量最相似的结果。这段代码的作用是：   
    检查是否已存在预处理好的 FAISS 向量库。
    若存在，则直接加载复用（节省时间）。
    若不存在，则新建向量库并保存到本地（供下次使用）。
2. 为什么这样设计？
    节省时间：文档转换为向量（尤其是大量文档）是耗时操作，复用已有向量库可显著减少每次运行的时间。
    一致性：确保每次运行使用相同的向量库，避免因重复处理导致的微小差异（如浮点计算误差）。
    可扩展性：当文档更新时，可修改逻辑为 “加载已有库→新增文档向量→保存更新”，实现增量更新。
3. 注意事项
    嵌入器一致性：加载向量库时使用的embeddings必须与创建时完全相同（模型、参数一致），否则向量维度或语义空间不匹配，检索结果会出错。
    安全性：allow_dangerous_deserialization=True存在潜在风险，生产环境中需确保向量库文件未被篡改。
    存储占用：向量库会占用磁盘空间（取决于文档数量和向量维度），若文档频繁更新，需定期清理旧库。
    
  *这段代码通过 “检查→加载 / 创建→保存” 的逻辑，高效管理 FAISS 向量库，既避免了重复计算的资源浪费，又保证了系统的可复用性和一致性，是 RAG 系统中提升效率的关键环节。
'''
start_time = time.time()
if FAISS_DB_PATH.exists() and len(os.listdir(FAISS_DB_PATH)) > 0:                                                       # 判断向量库是否已存在，0.0337秒秒
    vector_store = FAISS.load_local(                                                                                    # 如果存在：加载已存在的 FAISS 向量库
        str(FAISS_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True                                                                            # 由于向量库包含二进制文件，加载时需要允许反序列化
    )
    print(f"加载已存在的FAISS向量库，路径：{FAISS_DB_PATH}")
else:                                                                                                                   # 不存在，创建本地向量库，保存到硬盘上，耗时1.4600秒
    vector_store = FAISS.from_documents(all_splits, embeddings)                                                         # 将文档分块（all_splits）通过嵌入器（embeddings）转换为向量，并构建 FAISS 向量库。
    FAISS_DB_PATH.mkdir(parents=True, exist_ok=True)                                                                    # 创建保存目录（若不存在）
    vector_store.save_local(str(FAISS_DB_PATH))                                                                         # 保存向量库到本地，下次运行程序时，可通过load_local直接加载，无需重新处理文档
    print(f"新建并保存FAISS向量库，路径：{FAISS_DB_PATH}")

vector_time = time.time() - start_time
print(f"4、FAISS向量存储处理耗时：{vector_time:.4f}秒")

# 5. 检索相关文档
'''
1、底层工作流程
    1）问题向量化：
        首先，系统会使用与构建向量库时相同的嵌入器（embeddings），将用户的问题（question）转换为对应的向量表示（与文档向量维度一致）。
    2）相似度计算：
        FAISS 向量库会计算 “问题向量” 与库中所有 “文档分块向量” 的语义相似度（默认使用余弦相似度）。
        相似度越高，说明文档内容与问题的相关性越强。
    3）筛选与返回：
        按照相似度从高到低排序，取前k=3个文档分块，作为最相关的结果返回给 retrieved_docs。
2、 返回结果（retrieved_docs）
    这是一个包含 3 个文档对象的列表，每个对象对应一个相关的文档分块，包含以下关键信息：   
        page_content：文档分块的具体文本内容（最核心的部分，后续会作为上下文传给 LLM）。例如，retrieved_docs[0].page_content 即可获取与问题最相关的文档文本。
        metadata：文档的元数据（如来源文件路径、页码等，视初始化时的配置而定）。
3、 为什么这样设计？
    精准定位：通过语义相似度而非关键词匹配，能理解问题和文档的深层含义（例如 “促进民营经济” 与 “支持民营企业发展” 是语义相关的）。
    效率优先：FAISS 库针对向量检索做了优化，即使处理百万级向量也能快速返回结果。
    控制成本：只返回最相关的k个文档，避免将过多无关信息传给 LLM，既节省计算资源，又提高回答准确性。
'''
question = "哪些部门负责促进民营经济发展的工作？"                                                                              # 问题
start_time = time.time()
retrieved_docs = vector_store.similarity_search(question, k=3)                                                          # 语义相似度搜索，从已构建的向量库（vector_store）中筛选出与用户问题（question）最相关的前 3 个文档片段（k=3），并将结果存储在 retrieved_docs 中。
retrieve_time = time.time() - start_time
print(f"5、文档检索耗时：{retrieve_time:.4f}秒，返回{len(retrieved_docs)}条结果")                                             # 文档检索耗时：0.0431秒，返回3条结果

# 准备上下文内容
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. 构建提示模板
'''
语法与参数解析
    ChatPromptTemplate：LangChain 提供的用于构建对话提示的类，专门用于与聊天模型（如 ChatDeepSeek）交互。
    from_template(...)：通过一个字符串模板创建提示模板的便捷方法。模板中用 {变量名} 表示需要动态替换的内容。
指令部分（前两行）： 告诉 LLM 回答的规则：
    必须基于提供的 “上下文” 内容（不能编造外部信息）
    用中文回答，要求简洁、准确
    若上下文没有相关信息，必须返回固定话术（避免幻觉回答） ***
{context}：
    占位符，后续会被替换为从向量库中检索到的相关文档内容（即 docs_content，由 retrieved_docs 拼接而成）。这是 LLM 生成答案的 “知识来源”。
{question}：
    占位符，后续会被替换为用户的原始问题（如 哪些部门负责促进民营经济发展的工作？）。
回答:：
    引导 LLM 在这个位置开始生成答案，确保输出格式的一致性。
'''
prompt = ChatPromptTemplate.from_template("""                                                                   
基于以下上下文，用中文简洁准确地回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:"""
                                         )

# 7. 创建LLM模型
'''
初始化一个 DeepSeek 大语言模型的客户端实例，用于基于检索到的上下文生成最终回答。
1. 核心作用
    通过 ChatDeepSeek 类，创建一个大语言模型（LLM）实例 llm，后续可通过这个实例调用 DeepSeek 模型的 API，传入提示词（含上下文和问题）并获取生成的答案。
2. 参数详解
    （1）model="deepseek-chat"
        指定要使用的 DeepSeek 模型版本。
        "deepseek-chat" 是 DeepSeek 官方提供的对话模型，专门优化了多轮对话和问答能力，适合处理基于上下文的问题。
        其他可选模型可能包括 deepseek-vl（多模态模型）等，具体取决于 DeepSeek 提供的 API 支持。
    （2）temperature=0.0
        控制模型生成答案的 “随机性” 或 “创造性”：
        取值范围通常为 0.0~1.0（部分模型支持更高值）。
        temperature=0.0 表示生成结果完全确定（最保守模式），每次输入相同内容时，输出几乎完全一致。
        若调高热值（如 0.7），模型会生成更多样化的结果，但可能引入不准确信息。
        此处设置为 0.0，因为 RAG 系统需要基于固定上下文生成事实性回答，优先保证准确性而非多样性。
    （3）max_tokens=2048
        限制模型生成答案的最大 token 数量（1 个汉字约等于 1 个 token，英文单词可能拆分为多个 token）。
        2048 表示答案长度最多不超过 2048 个 token，足够覆盖大多数问题的详细回答。
        若答案超过此限制，模型会在达到阈值时截断输出，因此需根据问题复杂度合理设置。
    （4）api_key=api_key
        传入访问 DeepSeek API 的密钥（从环境变量或默认值获取）。
        API 密钥是调用第三方模型接口的身份凭证，确保只有授权用户能使用服务（避免滥用）。
        代码中通过 os.getenv("DEEPSEEK_API_KEY") 从环境变量读取，若未设置则使用默认值（实际使用时需替换为个人有效密钥）。
3. 底层工作原理
    初始化客户端：通过上述参数配置，ChatDeepSeek 类会创建一个与 DeepSeek API 通信的客户端。
    接口调用准备：客户端内部处理 API 请求的格式、认证（通过 api_key）、超时设置等底层细节。
    生成答案：后续通过 llm.invoke(prompt) 调用时，客户端会将填充后的提示词（含上下文和问题）发送到 DeepSeek 服务器，获取模型生成的答案并返回。
4. 为什么这样配置？
    模型选择：deepseek-chat 适合对话场景，能更好地理解 “基于上下文回答” 的指令。
    确定性优先：temperature=0.0 确保对于相同的上下文和问题，生成的答案稳定一致，适合事实性问答（如法律条文解读）。
    长度控制：max_tokens=2048 在保证回答完整性的同时，避免生成过长内容导致的资源浪费。
    安全性：通过 api_key 认证，确保 API 调用的合法性和安全性。
'''
start_time = time.time()
api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-XXXX"

llm = ChatDeepSeek(
    model="deepseek-chat",                                                                                              # deepseek-chat model points to DeepSeek-V3-0324. 使用10.4504秒
    #model="deepseek-reasoner",                                                                                          # deepseek-reasoner model points to DeepSeek-R1-0528. 使用24.0067秒
    temperature=0.0,
    max_tokens=2048,
    api_key=api_key
)
llm_time = time.time() - start_time
print(f"6、创建LLM耗时：{llm_time:.4f}秒")                                                                                 # 创建LLM耗时：2.3727秒

# 8. 调用LLM生成答案
start_time = time.time()
answer = llm.invoke(prompt.format(question=question, context=docs_content))                                             # 调用大语言模型生成答案
llm_time = time.time() - start_time
print(f"7、生成答案耗时：{llm_time:.4f}秒")                                                                                 # 生成答案耗时：7.9308秒

# 输出结果
print("\n===== 问答结果 =====")
print(f"问题：{question}")
print(f"回答：{answer.content}")

# 总耗时统计
total_time = time.time() - start_total
print(f"\n总流程耗时：{total_time:.4f}秒")
