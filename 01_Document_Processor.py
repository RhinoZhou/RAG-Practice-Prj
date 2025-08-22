import os                                                                                          # 操作系统接口模块，用于文件和目录操作
import time                                                                                        # 时间模块，用于性能计时和延迟控制
import torch                                                                                       # PyTorch深度学习框架，用于GPU加速和模型管理
import pythoncom                                                                                   # Windows COM接口初始化，用于Office文档处理
from pathlib import Path                                                                           # 面向对象的路径操作，提供跨平台路径处理
from dotenv import load_dotenv                                                                     # 环境变量加载，用于配置管理
from langchain_text_splitters import RecursiveCharacterTextSplitter                                # LangChain文本分割器，按字符递归分割长文档
from langchain_huggingface import HuggingFaceEmbeddings                                            # HuggingFace嵌入模型封装，用于文本向量化
from langchain_community.vectorstores import FAISS                                                 # Facebook AI相似度搜索库，用于高效向量检索
from modelscope.hub.snapshot_download import snapshot_download                                     # ModelScope模型下载器，从阿里云平台获取预训练模型
from sentence_transformers import SentenceTransformer as ST                                        # 句子转换器，用于生成文本嵌入向量

# 文档加载器 - 基础依赖
from langchain_community.document_loaders import (                                                  # LangChain社区文档加载器集合
    TextLoader,                                                                                    # 纯文本文件加载器，支持.txt文件
    PyPDFLoader,                                                                                   # PDF文档加载器，基于PyPDF2库
    UnstructuredExcelLoader,                                                                       # Excel文件加载器，支持.xls和.xlsx格式
    UnstructuredPowerPointLoader                                                                   # PowerPoint加载器，支持.ppt和.pptx格式
)
from langchain_core.documents import Document as LangDoc                                           # LangChain文档对象，统一文档格式标准


# -------------------------- 自定义加载器 --------------------------
class DocxLoader:                                                                                       # DOCX文档专用加载器类
    """处理.docx文件（基于python-docx）"""                                                                # 类说明：使用python-docx库处理Word文档

    def __init__(self, file_path):                                                                      # 构造函数，接收文件路径参数
        self.file_path = file_path                                                                      # 存储文件路径到实例变量

    def load(self):                                                                                 # 加载文档的主要方法
        try:                                                                                        # 异常处理开始
            from docx import Document                                                               # 动态导入python-docx库
            doc = Document(self.file_path)                                                          # 创建Document对象并打开文件
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]                 # 提取所有非空段落文本
            if not full_text:                                                                       # 检查是否提取到有效内容
                return []                                                                            # 无内容返回空列表
            return [LangDoc(                                                                         # 创建LangChain文档对象
                page_content='\n'.join(full_text),                                                  # 合并段落文本作为页面内容
                metadata={"source": str(self.file_path)}                                             # 添加源文件路径元数据
            )]
        except Exception as e:                                                                      # 捕获所有异常
            print(f"[DOCX失败] {os.path.basename(self.file_path)}: {str(e)}")                        # 打印错误信息和文件名
            return []                                                                               # 异常时返回空列表


class DocLoader:                                                                                   # DOC文档专用加载器类（老版本Word格式）
    """处理.doc文件（基于win32com，修复连接问题）"""                                              # 类说明：使用Windows COM接口处理.doc文件

    def __init__(self, file_path):                                                                 # 构造函数，接收文件路径参数
        self.file_path = file_path                                                                  # 存储文件路径到实例变量
        self.word = None  # Word实例                                                               # 初始化Word应用实例为None

    def _init_word(self):                                                                          # 初始化Word应用的私有方法
        """初始化Word实例，确保每次都是新连接"""                                                   # 方法说明：创建新的Word进程避免冲突
        try:                                                                                        # 异常处理开始
            if self.word:                                                                           # 检查是否已有Word实例
                self.word.Quit()                                                                    # 关闭现有Word实例
        except:                                                                                     # 忽略关闭异常
            pass                                                                                    # 静默处理

        import win32com.client                                                                      # 动态导入win32com库
        self.word = win32com.client.Dispatch("Word.Application")                                   # 创建Word应用COM对象
        self.word.Visible = False                                                                   # 设置Word不可见运行
        self.word.DisplayAlerts = 0                                                                 # 禁用警告对话框
        return self.word                                                                            # 返回Word实例

    def load(self):                                                                                # 加载文档的主要方法
        try:                                                                                        # 异常处理开始
            pythoncom.CoInitialize()                                                                # 初始化COM库
            word = self._init_word()                                                                # 获取Word实例
            doc = word.Documents.Open(str(self.file_path))                                          # 打开Word文档
            text = doc.Content.Text.strip()                                                         # 提取文档纯文本内容
            doc.Close(SaveChanges=0)                                                                # 关闭文档不保存
            self.word.Quit()                                                                        # 退出Word应用
            self.word = None                                                                        # 清空Word实例

            if not text:                                                                            # 检查是否提取到有效内容
                return []                                                                            # 无内容返回空列表
            return [LangDoc(                                                                         # 创建LangChain文档对象
                page_content=text,                                                                   # 设置文档内容
                metadata={"source": str(self.file_path)}                                             # 添加源文件路径元数据
            )]
        except Exception as e:                                                                      # 捕获所有异常
            print(f"[DOC失败] {os.path.basename(self.file_path)}: {str(e)}")                         # 打印错误信息和文件名
            if self.word:                                                                           # 检查Word实例是否存在
                try:                                                                                # 尝试清理
                    self.word.Quit()                                                                # 强制退出Word
                except:                                                                             # 忽略退出异常
                    pass                                                                            # 静默处理
                self.word = None                                                                    # 清空Word实例
            return []                                                                               # 异常时返回空列表
        finally:                                                                                     # 最终清理
            pythoncom.CoUninitialize()                                                              # 反初始化COM库


class PptLoader:                                                                                  # PPT文档专用加载器类
    """处理.ppt文件（基于win32com调用PowerPoint）"""                                              # 类说明：使用Windows COM接口处理PPT文件

    def __init__(self, file_path):                                                                 # 构造函数，接收文件路径参数
        self.file_path = file_path                                                                  # 存储文件路径到实例变量
        self.powerpoint = None                                                                      # 初始化PowerPoint实例

    def _init_ppt(self):                                                                           # 初始化PowerPoint应用的私有方法
        """初始化PowerPoint实例"""                                                               # 方法说明：创建新的PowerPoint进程
        if self.powerpoint:                                                                        # 检查是否已有PowerPoint实例
            try:                                                                                   # 异常处理开始
                self.powerpoint.Quit()                                                             # 关闭现有PowerPoint实例
            except:                                                                                # 忽略关闭异常
                pass                                                                               # 静默处理

        import win32com.client                                                                     # 动态导入win32com库
        self.powerpoint = win32com.client.Dispatch("PowerPoint.Application")                        # 创建PowerPoint应用COM对象
        self.powerpoint.Visible = True                                                             # 设置PowerPoint可见（某些操作需要）
        return self.powerpoint                                                                     # 返回PowerPoint实例

    def load(self):                                                                                # 加载文档的主要方法
        try:                                                                                        # 异常处理开始
            pythoncom.CoInitialize()                                                                # 初始化COM库
            ppt = self._init_ppt()                                                                  # 获取PowerPoint实例
            presentation = ppt.Presentations.Open(str(self.file_path))                             # 打开PPT演示文稿
            full_text = []                                                                          # 初始化文本收集列表
            for slide in presentation.Slides:                                                       # 遍历所有幻灯片
                for shape in slide.Shapes:                                                          # 遍历幻灯片内所有形状
                    if shape.HasTextFrame and shape.TextFrame.HasText:                             # 检查形状是否包含文本
                        full_text.append(shape.TextFrame.TextRange.Text)                           # 提取文本内容
            presentation.Close()                                                                    # 关闭演示文稿
            self.powerpoint.Quit()                                                                    # 退出PowerPoint应用
            self.powerpoint = None                                                                  # 清空PowerPoint实例

            text = '\n'.join(full_text).strip()                                                     # 合并所有文本并清理空白
            if not text:                                                                            # 检查是否提取到有效内容
                return []                                                                            # 无内容返回空列表
            return [LangDoc(                                                                         # 创建LangChain文档对象
                page_content=text,                                                                   # 设置文档内容
                metadata={"source": str(self.file_path)}                                             # 添加源文件路径元数据
            )]
        except Exception as e:                                                                      # 捕获所有异常
            print(f"[PPT失败] {os.path.basename(self.file_path)}: {str(e)}")                        # 打印错误信息和文件名
            if self.powerpoint:                                                                     # 检查PowerPoint实例是否存在
                try:                                                                                # 尝试清理
                    self.powerpoint.Quit()                                                          # 强制退出PowerPoint
                except:                                                                             # 忽略退出异常
                    pass                                                                            # 静默处理
                self.powerpoint = None                                                              # 清空PowerPoint实例
            return []                                                                               # 异常时返回空列表
        finally:                                                                                     # 最终清理
            pythoncom.CoUninitialize()                                                              # 反初始化COM库


# -------------------------- 主逻辑 --------------------------
def create_vector_db():                                                                           # 主函数：创建向量数据库
    start_total = time.time()                                                                     # 记录总开始时间
    current_dir = Path(__file__).resolve().parent                                                 # 获取当前脚本所在目录的绝对路径

    # 路径配置
    LOCAL_EMBEDDING_PATH = current_dir / "../11_local_models/bge-small-zh-v1.5"                 # 本地嵌入模型存储路径
    DOCUMENTS_DIR = current_dir / "../20-Data/01-金融"                                              # 待处理文档目录路径
    FAISS_DB_PATH = current_dir / "../23-faiss_db"                                                # FAISS向量数据库输出路径

    # 初始化嵌入模型
    load_dotenv()                                                                                 # 加载环境变量配置
    print("本地嵌入模型已存在，无需下载")                                                           # 提示信息：模型已存在

    # 验证本地嵌入模型是否存在
    if not LOCAL_EMBEDDING_PATH.exists() or len(os.listdir(LOCAL_EMBEDDING_PATH)) == 0:           # 检查模型目录是否存在且非空
        print(f"本地嵌入模型不存在于路径: {LOCAL_EMBEDDING_PATH}")                                   # 提示模型不存在
        print("开始自动下载模型...")                                                               # 开始下载提示

        LOCAL_EMBEDDING_PATH.mkdir(parents=True, exist_ok=True)                                   # 创建模型存储目录

        try:                                                                                        # 异常处理开始
            snapshot_download(                                                                      # 从ModelScope下载模型
                model_id="AI-ModelScope/bge-small-zh-v1.5",                                           # 模型标识符：bge-small中文v1.5
                local_dir=str(LOCAL_EMBEDDING_PATH)                                                 # 本地存储目录
            )
        except Exception as e:                                                                      # 捕获下载异常
            print(f"模型下载失败: {str(e)}")                                                        # 打印下载错误信息
            print("请检查网络连接或手动下载模型")                                                   # 提供解决建议
            raise                                                                                   # 重新抛出异常

    embeddings = HuggingFaceEmbeddings(                                                             # 创建HuggingFace嵌入模型实例
        model=str(LOCAL_EMBEDDING_PATH),                                                            # 指定本地模型路径
        model_kwargs={'device': 'cpu', 'local_files_only': True},                                   # 模型参数：使用CPU，仅本地文件
        encode_kwargs={'batch_size': 32, 'device': 'cpu', 'normalize_embeddings': True}             # 编码参数：批大小32，标准化向量
    )
    print("嵌入模型初始化完成")                                                                       # 提示模型初始化完成

    # 加载文档
    load_start = time.time()                                                                        # 记录文档加载开始时间
    loaders = {                                                                                     # 定义文件类型到加载器的映射字典
        '.txt': TextLoader,                                                                         # 文本文件使用TextLoader
        '.doc': DocLoader,                                                                          # DOC文件使用自定义DocLoader
        '.docx': DocxLoader,                                                                        # DOCX文件使用自定义DocxLoader
        '.xls': UnstructuredExcelLoader,                                                            # 老版本Excel使用UnstructuredExcelLoader
        '.xlsx': UnstructuredExcelLoader,                                                           # 新版本Excel使用UnstructuredExcelLoader
        '.pdf': PyPDFLoader,                                                                        # PDF文件使用PyPDFLoader
        '.ppt': PptLoader,                                                                          # PPT文件使用自定义PptLoader
        '.pptx': UnstructuredPowerPointLoader                                                       # PPTX文件使用UnstructuredPowerPointLoader
    }

    all_docs = []                                                                                   # 初始化文档收集列表
    # 统计信息字典
    stats = {                                                                                       # 初始化统计信息结构
        'total': {'found': 0, 'success': 0, 'failed': 0},                                           # 总体统计：找到、成功、失败数量
        'types': {}  # 按文件类型统计                                                             # 按扩展名分类统计
    }
    # 记录失败的文件
    failed_files = []                                                                               # 初始化失败文件列表

    # 初始化各文件类型的统计
    for ext in loaders.keys():                                                                      # 遍历所有支持的文件扩展名
        stats['types'][ext] = {'found': 0, 'success': 0, 'failed': 0}                             # 为每种类型初始化统计计数器

    for ext, loader_cls in loaders.items():                                                           # 遍历每种文件类型和对应的加载器
        file_paths = list(DOCUMENTS_DIR.glob(f"**/*{ext}"))                                         # 递归查找该类型的所有文件
        file_count = len(file_paths)                                                                # 统计文件数量

        # 更新统计
        stats['total']['found'] += file_count                                                       # 累加到总体找到数量
        stats['types'][ext]['found'] = file_count                                                   # 记录该类型的找到数量

        if file_count == 0:                                                                         # 检查是否找到文件
            print(f"未找到{ext}文件")                                                               # 提示未找到该类型文件
            continue                                                                                # 跳过该类型继续处理
        print(f"开始加载{file_count}个{ext}文件...")                                               # 提示开始加载该类型文件

        for file_path in file_paths:                                                                # 遍历每个找到的文件
            try:                                                                                    # 异常处理开始
                if ext in ['.doc', '.docx', '.ppt']:                                                # 检查是否为自定义加载器类型
                    loader = loader_cls(file_path)                                                  # 创建自定义加载器实例
                else:                                                                               # 其他类型使用标准加载器
                    loader = loader_cls(str(file_path))                                             # 创建标准加载器实例

                docs = loader.load()                                                                # 执行文档加载
                if docs:                                                                            # 检查是否成功加载文档
                    all_docs.extend(docs)                                                           # 将文档添加到总列表
                    # 更新成功统计
                    stats['total']['success'] += 1                                                  # 累加总体成功数量
                    stats['types'][ext]['success'] += 1                                             # 累加该类型成功数量
                    print(f"[成功] {os.path.basename(file_path)}")                                # 打印成功加载提示
                else:                                                                               # 文档内容为空
                    # 空内容也算失败
                    raise Exception("文件内容为空")                                                 # 抛出异常标记为失败
            except Exception as e:                                                                  # 捕获所有加载异常
                # 更新失败统计
                stats['total']['failed'] += 1                                                     # 累加总体失败数量
                stats['types'][ext]['failed'] += 1                                                # 累加该类型失败数量
                failed_files.append({                                                               # 记录失败文件信息
                    'path': str(file_path),                                                         # 记录完整文件路径
                    'name': os.path.basename(file_path),                                            # 记录文件名
                    'ext': ext,                                                                     # 记录文件扩展名
                    'error': str(e)                                                                 # 记录错误信息
                })
                print(f"[加载失败] {os.path.basename(file_path)}: {str(e)}")                       # 打印失败信息

    # -------------------------- 文档处理完成后的统计与向量数据库创建 --------------------------
    if stats['total']['success'] == 0:                                                              # 检查是否有成功加载的文档
        raise ValueError("未加载到任何文件")                                                        # 无文档则抛出异常终止程序

    # 打印详细统计信息
    load_time = time.time() - load_start                                                            # 计算文档加载总耗时
    print(f"\n1、文档加载完成，总耗时：{load_time:.4f}秒")                                            # 打印加载耗时
    print(                                                                                          # 打印总体统计信息
        f"   总文件数：{stats['total']['found']}个，成功加载：{stats['total']['success']}个，加载失败：{stats['total']['failed']}个")
    print("   按文件类型统计：")                                                                      # 打印文件类型统计标题
    for ext, ext_stats in stats['types'].items():                                                  # 遍历每种文件类型统计
        if ext_stats['found'] > 0:                                                                # 检查该类型是否找到文件
            print(f"   - {ext}文件：共{ext_stats['found']}个，成功{ext_stats['success']}个，失败{ext_stats['failed']}个")  # 打印类型统计详情

    # 打印失败文件列表
    if failed_files:                                                                                # 检查是否有加载失败的文件
        print("\n2、加载失败的文件列表：")                                                           # 打印失败文件列表标题
        for i, file in enumerate(failed_files, 1):                                                  # 遍历失败文件列表并编号
            print(f"   {i}. {file['name']}（{file['ext']}）：{file['error'][:100]}")  # 打印失败文件详情（限制错误信息长度）

    # 文档分块处理
    split_start = time.time()                                                                       # 记录分块开始时间
    text_splitter = RecursiveCharacterTextSplitter(                                                 # 创建文本分割器实例
        chunk_size=1000, chunk_overlap=200,                                                         # 设置块大小1000字符，重叠200字符
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]                 # 设置中文文本分隔符
    )
    all_splits = text_splitter.split_documents(all_docs)                                            # 对所有文档进行分块处理
    split_time = time.time() - split_start                                                          # 计算分块耗时
    print(f"\n3、文档分块完成，共{len(all_splits)}块，耗时：{split_time:.4f}秒")                     # 打印分块结果和耗时

    # 创建向量数据库
    create_start = time.time()                                                                      # 记录向量库创建开始时间
    vector_store = FAISS.from_documents(all_splits, embeddings)                                     # 使用FAISS创建向量数据库
    FAISS_DB_PATH.mkdir(parents=True, exist_ok=True)                                                # 确保输出目录存在
    vector_store.save_local(str(FAISS_DB_PATH))                                                    # 将向量数据库保存到本地
    vector_time = time.time() - create_start                                                        # 计算向量库创建耗时
    print(f"4、向量库创建完成，耗时：{vector_time:.4f}秒")                                             # 打印向量库创建完成信息

    # 打印最终汇总信息
    total_time = time.time() - start_total                                                        # 计算整个处理流程总耗时
    print(f"\n===== 总耗时：{total_time:.4f}秒，处理{stats['total']['success']}个文件 → {len(all_splits)}块 =====")  # 打印最终汇总


if __name__ == "__main__":
    create_vector_db()
