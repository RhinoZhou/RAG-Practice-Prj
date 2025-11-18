# -*- coding: utf-8 -*-
'''
RAG系统工具集模块，提供文件读取、文本处理、文档分块和性能分析等核心功能。
本模块是RAG系统的基础工具组件，主要特性包括：
1. 多种格式文件读取（PDF、Markdown、TXT）
2. 智能文本分块处理（支持Token级别分块和重叠区域）
3. 文档集合管理
4. 性能监控和计时工具
5. 自动依赖检查和安装
6. 统一的文件处理接口抽象
7. 错误处理和异常管理
8. 支持批处理和并行操作
'''

import os
import sys
import time
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from abc import ABC, abstractmethod
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed


# 自动安装依赖
def setup_dependencies():
    """
    自动安装所需依赖库
    """
    try:
        import pip
        
        required_packages = [
            'PyPDF2>=3.0.0',
            'markdown>=3.4.0',
            'beautifulsoup4>=4.11.0',
            'tiktoken>=0.5.0',
            'tqdm>=4.65.0',
            'html2text>=2020.1.16'
        ]
        
        print("正在检查并安装必要的依赖...")
        for package in required_packages:
            try:
                # 尝试导入包
                if package.split('>=')[0].replace('-', '_') == 'PyPDF2':
                    import PyPDF2
                elif package.split('>=')[0].replace('-', '_') == 'markdown':
                    import markdown
                elif package.split('>=')[0].replace('-', '_') == 'beautifulsoup4':
                    import bs4
                elif package.split('>=')[0].replace('-', '_') == 'tiktoken':
                    import tiktoken
                elif package.split('>=')[0].replace('-', '_') == 'tqdm':
                    import tqdm
                elif package.split('>=')[0].replace('-', '_') == 'html2text':
                    import html2text
            except ImportError:
                print(f"安装 {package}...")
                pip.main(['install', package])
        
        print("✓ 依赖安装完成")
    except Exception as e:
        print(f"警告: 自动安装依赖时出错: {str(e)}")
        print("请手动安装所需依赖")


# 自动安装依赖
setup_dependencies()


# 导入必要的库
import PyPDF2
import markdown
import html2text
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup


# 初始化token编码器
def get_token_encoder() -> tiktoken.Encoding:
    """
    获取或创建token编码器
    
    Returns:
        token编码器实例
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        print("警告: 无法加载cl100k_base编码器，使用gpt2作为备选")
        return tiktoken.get_encoding("gpt2")


# 创建token编码器实例
enc = get_token_encoder()


# 性能监控装饰器
def timer_decorator(func):
    """
    函数执行时间监控装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数，会在执行时记录并打印执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper


# 错误处理装饰器
def error_handler(func):
    """
    函数错误处理装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数，会捕获并记录异常，返回None或默认值
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"错误: {func.__name__} 执行失败: {str(e)}")
            return None
    return wrapper


# 文件读取基类
class BaseRead(ABC):
    """
    文件读取抽象基类
    定义了所有文件读取类必须实现的接口
    """
    
    @abstractmethod
    def read(self) -> str:
        """
        读取文件内容
        
        Returns:
            文件内容字符串
        """
        pass


# 文件读取器类
class ReadFiles:
    """
    多格式文件读取器类
    支持读取PDF、Markdown和文本文件，提供文件遍历、内容提取和文本分块功能
    """
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = ['.pdf', '.md', '.txt', '.markdown']
    
    def __init__(self, path: str) -> None:
        """
        初始化文件读取器
        
        Args:
            path: 要读取的文件或目录路径
        """
        self._path = path
        self.file_list = []
        
        # 初始化时获取文件列表
        self.file_list = self.get_files()
        print(f"找到 {len(self.file_list)} 个支持的文件")
    
    def get_files(self) -> List[str]:
        """
        递归获取指定目录下的所有支持的文件
        
        Returns:
            文件路径列表
        """
        file_list = []
        
        try:
            # 检查路径是否存在
            if not os.path.exists(self._path):
                print(f"警告: 路径不存在: {self._path}")
                return file_list
                
            # 如果是单个文件
            if os.path.isfile(self._path):
                # 检查文件扩展名是否支持
                if any(self._path.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    file_list.append(os.path.abspath(self._path))
                else:
                    print(f"警告: 不支持的文件类型: {self._path}")
                return file_list
                
            # 如果是目录，递归遍历
            print(f"正在扫描目录: {self._path}")
            for root, dirs, files in os.walk(self._path):
                for filename in files:
                    # 检查文件扩展名
                    if any(filename.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                        file_path = os.path.join(root, filename)
                        file_list.append(os.path.abspath(file_path))
                        
            print(f"成功扫描到 {len(file_list)} 个文件")
            return file_list
            
        except Exception as e:
            print(f"错误: 扫描文件时出错: {str(e)}")
            return file_list
    
    def get_content(self, max_token_len: int = 600, cover_content: int = 150, 
                   parallel: bool = False, max_workers: int = 4) -> List[str]:
        """
        获取所有文件的内容并分块
        
        Args:
            max_token_len: 每个分块的最大token长度
            cover_content: 分块之间的重叠内容长度（token）
            parallel: 是否使用并行处理
            max_workers: 并行处理的最大工作线程数
            
        Returns:
            分块后的文档内容列表
        """
        docs = []
        
        if not self.file_list:
            print("警告: 没有可处理的文件")
            return docs
        
        print(f"开始读取 {len(self.file_list)} 个文件")
        start_time = time.time()
        
        # 单个文件处理函数
        def process_file(file_path: str) -> List[str]:
            try:
                content = self.read_file_content(file_path)
                if content:
                    chunk_content = self.get_chunk(
                        content, max_token_len=max_token_len, cover_content=cover_content
                    )
                    return chunk_content
                return []
            except Exception as e:
                print(f"错误: 处理文件 {file_path} 时出错: {str(e)}")
                return []
        
        # 根据是否并行选择处理方式
        if parallel and len(self.file_list) > 1:
            # 并行处理文件
            print(f"使用并行模式处理，最大工作线程: {max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_file = {executor.submit(process_file, file): file for file in self.file_list}
                
                # 使用tqdm显示进度
                with tqdm(total=len(self.file_list), desc="处理文件") as pbar:
                    # 处理完成的任务
                    for future in as_completed(future_to_file):
                        file = future_to_file[future]
                        try:
                            chunks = future.result()
                            docs.extend(chunks)
                        except Exception as e:
                            print(f"错误: 处理文件 {file} 时出错: {str(e)}")
                        pbar.update(1)
        else:
            # 串行处理文件
            with tqdm(self.file_list, desc="处理文件") as pbar:
                for file in pbar:
                    chunks = process_file(file)
                    docs.extend(chunks)
        
        # 计算总耗时
        end_time = time.time()
        print(f"文件处理完成，共生成 {len(docs)} 个分块，耗时: {end_time - start_time:.4f}秒")
        
        return docs
    
    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150) -> List[str]:
        """
        将文本分块，支持token级别分块和重叠区域
        
        Args:
            text: 要分块的文本
            max_token_len: 每个分块的最大token长度
            cover_content: 分块之间的重叠内容长度（token）
            
        Returns:
            分块后的文本列表
        """
        try:
            if not text or not isinstance(text, str):
                return []
                
            chunk_text = []
            curr_len = 0
            curr_chunk = ''

            # 计算实际可用的token长度（减去重叠部分）
            token_len = max_token_len - cover_content
            if token_len <= 0:
                token_len = 100  # 确保有一个合理的最小值
            
            # 按行分割文本
            lines = text.splitlines()
            
            # 处理每一行
            for line in lines:
                # 去除多余空格
                line = line.strip()
                if not line:
                    continue
                
                # 计算当前行的token长度
                line_len = len(enc.encode(line))
                
                # 如果单行长度就超过限制，需要特殊处理
                if line_len > max_token_len:
                    print(f"警告: 单行长度超过限制，需要分割: {line_len} tokens")
                    # 将超长行分割成多个块
                    num_chunks = (line_len + token_len - 1) // token_len
                    for i in range(num_chunks):
                        # 计算当前块的字符起始位置（近似）
                        # 注意：这只是一个近似，因为token和字符不是一一对应的
                        # 对于中文来说，我们可以尝试使用字符级别的分割
                        avg_token_len = len(line) / line_len if line_len > 0 else 1
                        start_char = int(i * token_len * avg_token_len)
                        end_char = int((i + 1) * token_len * avg_token_len)
                        
                        # 确保不越界
                        if end_char > len(line):
                            end_char = len(line)
                        
                        # 尝试在词边界分割（对于中文，这可能不太适用，但我们尽量尝试）
                        # 对于中文，我们可以尝试在标点符号处分割
                        if i < num_chunks - 1:
                            # 寻找合适的分割点
                            split_point = end_char
                            for j in range(end_char, max(end_char - 50, start_char), -1):
                                if j < len(line) and line[j] in '，。；！？.,;!?:""':
                                    split_point = j + 1
                                    break
                            end_char = split_point
                        
                        # 获取当前块内容
                        chunk = line[start_char:end_char]
                        
                        # 添加到结果列表
                        chunk_text.append(chunk)
                        
                        # 更新当前块和长度
                        curr_chunk = chunk[-cover_content:] if len(chunk) > cover_content else chunk
                        curr_len = len(enc.encode(curr_chunk))
                    
                    continue
                
                # 常规行处理
                if curr_len + line_len + 1 <= token_len:  # +1 是为换行符预留的token
                    curr_chunk += line + '\n'
                    curr_len += line_len + 1
                else:
                    # 当前块已满，添加到结果
                    if curr_chunk.strip():
                        chunk_text.append(curr_chunk.strip())
                    
                    # 开始新块，包含重叠内容
                    curr_chunk = curr_chunk[-cover_content:] + line + '\n' if len(curr_chunk) > cover_content else line + '\n'
                    curr_len = len(enc.encode(curr_chunk))
            
            # 添加最后一个块
            if curr_chunk.strip():
                chunk_text.append(curr_chunk.strip())
            
            return chunk_text
            
        except Exception as e:
            print(f"错误: 文本分块时出错: {str(e)}")
            return [text]  # 出错时返回原始文本
    
    @classmethod
    @error_handler
    def read_file_content(cls, file_path: str) -> str:
        """
        根据文件扩展名选择合适的读取方法
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
            
        Raises:
            ValueError: 不支持的文件类型
        """
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return ""
        
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md') or file_path.endswith('.markdown'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
    
    @classmethod
    @error_handler
    def read_pdf(cls, file_path: str) -> str:
        """
        读取PDF文件内容
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            PDF文件的文本内容
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                # 使用PyPDF2读取PDF
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                print(f"正在读取PDF文件: {file_path} ({num_pages} 页)")
                
                # 使用tqdm显示进度
                for page_num in tqdm(range(num_pages), desc="读取PDF页面"):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n\n'
                    except Exception as e:
                        print(f"警告: 读取第 {page_num + 1} 页时出错: {str(e)}")
                        continue
            
            # 清理文本
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # 将多个空白字符替换为单个空格
            
            print(f"成功读取PDF文件，提取了 {len(text)} 个字符")
            return text
            
        except Exception as e:
            print(f"错误: 读取PDF文件时出错: {str(e)}")
            return ""
    
    @classmethod
    @error_handler
    def read_markdown(cls, file_path: str) -> str:
        """
        读取Markdown文件并转换为纯文本
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            处理后的纯文本内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
                
            # 将Markdown转换为HTML
            html_text = markdown.markdown(md_text)
            
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            
            # 清理文本
            plain_text = re.sub(r'\s+', ' ', plain_text)  # 将多个空白字符替换为单个空格
            plain_text = re.sub(r'http\S+', '', plain_text)  # 移除URL链接
            plain_text = plain_text.strip()
            
            print(f"成功读取Markdown文件: {file_path}")
            return plain_text
            
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as file:
                    md_text = file.read()
                    # 后续处理同上
                    html_text = markdown.markdown(md_text)
                    soup = BeautifulSoup(html_text, 'html.parser')
                    plain_text = soup.get_text()
                    plain_text = re.sub(r'\s+', ' ', plain_text)
                    plain_text = re.sub(r'http\S+', '', plain_text)
                    plain_text = plain_text.strip()
                    return plain_text
            except Exception as e:
                print(f"错误: 读取Markdown文件时编码错误: {str(e)}")
                return ""
        except Exception as e:
            print(f"错误: 读取Markdown文件时出错: {str(e)}")
            return ""
    
    @classmethod
    @error_handler
    def read_text(cls, file_path: str) -> str:
        """
        读取文本文件内容
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            文件的文本内容
        """
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'latin-1']
            text = ""
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    print(f"成功读取文本文件 ({encoding}): {file_path}")
                    break
                except UnicodeDecodeError:
                    continue
            
            # 清理文本
            if text:
                text = re.sub(r'\s+', ' ', text)  # 将多个空白字符替换为单个空格
                text = text.strip()
            
            return text
            
        except Exception as e:
            print(f"错误: 读取文本文件时出错: {str(e)}")
            return ""
    
    def export_chunks(self, chunks: List[str], output_path: str = './chunks.json') -> bool:
        """
        将分块后的文档导出到JSON文件
        
        Args:
            chunks: 分块后的文档列表
            output_path: 输出文件路径
            
        Returns:
            是否成功导出
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 导出到JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            print(f"成功导出 {len(chunks)} 个文档分块到: {output_path}")
            return True
            
        except Exception as e:
            print(f"错误: 导出文档分块时出错: {str(e)}")
            return False


# JSON文档读取类
class Documents:
    """
    JSON格式文档管理类
    用于读取和管理已分类的JSON格式文档集合
    """
    
    def __init__(self, path: str = '') -> None:
        """
        初始化文档管理器
        
        Args:
            path: JSON文档文件路径
        """
        self.path = path
        self.content = None
    
    def get_content(self) -> Dict[str, Any]:
        """
        读取JSON格式文档内容
        
        Returns:
            JSON解析后的字典对象
        """
        try:
            if not self.path:
                print("警告: 文档路径未设置")
                return {}
            
            if not os.path.exists(self.path):
                print(f"警告: 文档文件不存在: {self.path}")
                return {}
            
            print(f"正在读取文档文件: {self.path}")
            
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'latin-1']
            content = {}
            
            for encoding in encodings:
                try:
                    with open(self.path, mode='r', encoding=encoding) as f:
                        content = json.load(f)
                    print(f"成功读取文档文件 ({encoding})")
                    self.content = content
                    return content
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            print(f"错误: 无法读取或解析文档文件: {self.path}")
            return {}
            
        except Exception as e:
            print(f"错误: 读取文档内容时出错: {str(e)}")
            return {}
    
    def save_content(self, content: Dict[str, Any], path: str = None) -> bool:
        """
        保存文档内容到JSON文件
        
        Args:
            content: 要保存的内容
            path: 保存路径，如果为None则使用初始化时的路径
            
        Returns:
            是否成功保存
        """
        try:
            save_path = path if path else self.path
            if not save_path:
                print("错误: 保存路径未设置")
                return False
            
            # 确保输出目录存在
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存到JSON文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            
            print(f"成功保存文档内容到: {save_path}")
            self.content = content
            return True
            
        except Exception as e:
            print(f"错误: 保存文档内容时出错: {str(e)}")
            return False


# 文本预处理工具类
class TextProcessor:
    """
    文本预处理工具类
    提供文本清洗、规范化和增强等功能
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，去除多余字符和格式
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 替换多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空白
        text = text.strip()
        
        # 移除特殊控制字符（保留换行符）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        规范化空白字符
        
        Args:
            text: 原始文本
            
        Returns:
            规范化后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 将所有空白字符替换为空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        移除文本中的URL链接
        
        Args:
            text: 原始文本
            
        Returns:
            移除URL后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 匹配常见URL格式
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        return text
    
    @staticmethod
    def tokenize(text: str, encoding: tiktoken.Encoding = None) -> List[str]:
        """
        将文本分词为tokens
        
        Args:
            text: 原始文本
            encoding: token编码器，默认使用cl100k_base
            
        Returns:
            token列表
        """
        if not text or not isinstance(text, str):
            return []
        
        token_encoder = encoding if encoding else enc
        return token_encoder.encode(text)
    
    @staticmethod
    def count_tokens(text: str, encoding: tiktoken.Encoding = None) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 原始文本
            encoding: token编码器，默认使用cl100k_base
            
        Returns:
            token数量
        """
        if not text or not isinstance(text, str):
            return 0
        
        token_encoder = encoding if encoding else enc
        return len(token_encoder.encode(text))


# 文件操作工具函数
def ensure_directory(directory: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        是否成功确保目录存在
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        return True
    except Exception as e:
        print(f"错误: 创建目录时出错: {str(e)}")
        return False


def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小（字节），如果出错则返回-1
    """
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return os.path.getsize(file_path)
        return -1
    except Exception as e:
        print(f"错误: 获取文件大小时出错: {str(e)}")
        return -1


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读的格式
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化后的大小字符串
    """
    if size_bytes < 0:
        return "未知"
    
    # 定义单位
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    # 计算单位
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # 格式化输出
    return f"{size:.2f} {units[unit_index]}"


# 示例用法
if __name__ == "__main__":
    try:
        print("=== 测试工具集模块 ===")
        
        # 示例1: 读取文件并分块
        print("\n1. 测试文件读取和分块:")
        
        # 创建一个临时测试文件
        test_dir = "./test_docs"
        ensure_directory(test_dir)
        
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文件\n用于测试文件读取功能\n\n" * 5)
        
        print(f"创建测试文件: {test_file}")
        
        # 测试ReadFiles类
        reader = ReadFiles(test_dir)
        chunks = reader.get_content(max_token_len=50, cover_content=10)
        
        print(f"生成的文档分块数量: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"\n分块 {i+1}:")
            print(f"内容: {chunk}")
            print(f"Token数量: {TextProcessor.count_tokens(chunk)}")
        
        # 示例2: 测试TextProcessor
        print("\n2. 测试文本处理:")
        test_text = "  这是一个测试文本 包含多个  空格和\n换行符。还有一个链接 https://example.com 。  "
        
        print(f"原始文本: '{test_text}'")
        print(f"清理后: '{TextProcessor.clean_text(test_text)}'")
        print(f"移除URL后: '{TextProcessor.remove_urls(test_text)}'")
        print(f"Token数量: {TextProcessor.count_tokens(test_text)}")
        
        # 示例3: 测试文件操作工具
        print("\n3. 测试文件操作工具:")
        if os.path.exists(test_file):
            file_size = get_file_size(test_file)
            print(f"测试文件大小: {format_file_size(file_size)}")
        
        # 示例4: 测试Documents类
        print("\n4. 测试JSON文档管理:")
        test_json = os.path.join(test_dir, "test_docs.json")
        
        # 创建测试JSON
        test_data = {"documents": [{"id": 1, "content": "测试文档1"}, {"id": 2, "content": "测试文档2"}]}
        
        doc_manager = Documents(test_json)
        doc_manager.save_content(test_data)
        
        # 读取并验证
        loaded_data = doc_manager.get_content()
        print(f"成功读取JSON文档，包含 {len(loaded_data.get('documents', []))} 个文档")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
