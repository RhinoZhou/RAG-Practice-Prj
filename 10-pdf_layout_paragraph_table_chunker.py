# -*- coding: utf-8 -*-
"""PDF版面特征辅助分块器

实现基于PDF版面特征（行距、缩进）的段落重组，识别表格块并独立输出。
支持文本行聚合、表格抽取、公式/表格保护等功能。
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re

# 尝试导入PyMuPDF库用于PDF处理
try:
    import fitz
    HAS_FITZ = True
    print("成功导入PyMuPDF库")
except ImportError:
    HAS_FITZ = False
    print("未找到PyMuPDF库，请使用pip install pymupdf安装")

# 尝试导入tiktoken库用于token计算
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    print("未找到tiktoken库，将使用简单的字符tokenizer")

@dataclass
class PDFChunk:
    """表示一个PDF分块及其元数据
    
    属性:
        text: 块文本内容
        page: 页码
        type: 块类型 ("paragraph", "table", "header", "footer")
        bbox: 边界框坐标 [x0, y0, x1, y1]
        chunk_id: 块ID
        token_count: token数量
    """
    text: str
    page: int
    type: str
    bbox: Optional[List[float]] = None
    chunk_id: int = 0
    token_count: int = 0

class SimpleCharacterTokenizer:
    """简单的字符tokenizer实现，用于在tiktoken不可用时作为替代"""
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分割为字符级别token"""
        return list(text)
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenize(text))

class PDFLayoutChunker:
    """PDF版面特征辅助分块器"""
    
    def __init__(self, tokenizer_name: str = "cl100k_base", use_simple_tokenizer: bool = False):
        """初始化分块器
        
        参数:
            tokenizer_name: 使用的tokenizer名称
            use_simple_tokenizer: 是否强制使用简单的字符tokenizer
        """
        self.tokenizer_name = tokenizer_name
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        self.tokenizer = self._load_tokenizer()
        
    def _load_tokenizer(self):
        """加载tokenizer"""
        if self.use_simple_tokenizer or not HAS_TIKTOKEN:
            return SimpleCharacterTokenizer()
        else:
            try:
                return tiktoken.get_encoding(self.tokenizer_name)
            except Exception as e:
                print(f"加载tiktoken失败: {e}，将使用简单的字符tokenizer")
                return SimpleCharacterTokenizer()
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        参数:
            text: 输入文本
            
        返回:
            token数量
        """
        if isinstance(self.tokenizer, SimpleCharacterTokenizer):
            return self.tokenizer.count_tokens(text)
        else:
            return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def extract_text_and_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF中提取文本和表格
        
        参数:
            pdf_path: PDF文件路径
            
        返回:
            页面内容列表，每个页面包含文本行和表格信息
        """
        if not HAS_FITZ:
            raise ImportError("需要PyMuPDF库来处理PDF文件，请使用pip install pymupdf安装")
        
        result = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = {
                    "page": page_num + 1,  # 页码从1开始
                    "text_lines": [],
                    "tables": []
                }
                
                # 提取文本和文本块
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block["type"] == 0:  # 文本块
                        for line in block["lines"]:
                            # 提取行文本和位置信息
                            line_text = "".join([span["text"] for span in line["spans"]])
                            if line_text.strip():
                                # 计算行的边界框
                                x0 = min([span["bbox"][0] for span in line["spans"]])
                                y0 = min([span["bbox"][1] for span in line["spans"]])
                                x1 = max([span["bbox"][2] for span in line["spans"]])
                                y1 = max([span["bbox"][3] for span in line["spans"]])
                                
                                # 提取字体大小信息（取第一个span的字体大小作为行的字体大小）
                                font_size = line["spans"][0]["size"] if line["spans"] else 0
                                
                                page_data["text_lines"].append({
                                    "text": line_text,
                                    "bbox": [x0, y0, x1, y1],
                                    "font_size": font_size,
                                    "block_no": block["number"]
                                })
                    elif block["type"] == 1:  # 图像块（可能包含表格）
                        # PyMuPDF默认不直接识别表格，这里做简单的表格检测
                        # 更复杂的表格提取可能需要使用专门的表格识别库
                        pass
                
                # 对文本行按y坐标排序
                page_data["text_lines"].sort(key=lambda x: x["bbox"][1])
                
                # 简单的表格检测（基于水平线和垂直线）
                # 这里使用一种简单的启发式方法来检测表格
                page_data["tables"] = self._detect_tables(page)
                
                result.append(page_data)
            
            doc.close()
        except Exception as e:
            print(f"处理PDF文件时出错: {e}")
            raise
        
        return result
    
    def _detect_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """检测页面中的表格
        
        参数:
            page: PyMuPDF的Page对象
            
        返回:
            检测到的表格列表
        """
        tables = []
        
        try:
            # 尝试1: 使用PyMuPDF的矩形检测功能
            # 获取页面中的所有文本块
            text_blocks = page.get_text("blocks")
            
            # 过滤出文本块（类型为0）
            text_blocks = [b for b in text_blocks if b[6] == 0]
            
            # 检测潜在的表格区域：寻找有规律的文本块排列
            if text_blocks:
                # 按y坐标和x坐标排序文本块，以便更好地分析布局
                text_blocks.sort(key=lambda b: (b[1], b[0]))
                
                # 计算文本块的平均宽度、高度和间距
                widths = [b[2] - b[0] for b in text_blocks]
                heights = [b[3] - b[1] for b in text_blocks]
                avg_width = sum(widths) / len(widths) if widths else 0
                avg_height = sum(heights) / len(heights) if heights else 0
                
                # 尝试2: 基于行列排列检测表格
                # 找出可能的行（y坐标相近的块）
                rows = []
                current_row = [text_blocks[0]]
                
                for i in range(1, len(text_blocks)):
                    block = text_blocks[i]
                    prev_block = text_blocks[i-1]
                    
                    # 如果垂直间距小于2倍平均高度，认为是同一行
                    if block[1] - prev_block[3] < 2 * avg_height:
                        current_row.append(block)
                    else:
                        rows.append(current_row)
                        current_row = [block]
                rows.append(current_row)
                
                # 检查是否有至少2行，每行至少有2个元素，且列对齐
                if len(rows) >= 2:
                    # 检查每行的元素数量是否相似
                    row_lengths = [len(row) for row in rows if len(row) >= 2]
                    
                    if row_lengths and len(row_lengths) >= 2:
                        # 计算行元素数量的平均值
                        avg_cells_per_row = sum(row_lengths) / len(row_lengths)
                        
                        # 如果大多数行的元素数量接近平均值，可能是表格
                        if sum(1 for rl in row_lengths if abs(rl - avg_cells_per_row) <= 1) > 0.7 * len(row_lengths):
                            # 合并相邻行构建表格文本
                            table_text = ""
                            table_x0 = min(b[0] for row in rows for b in row)
                            table_y0 = min(b[1] for row in rows for b in row)
                            table_x1 = max(b[2] for row in rows for b in row)
                            table_y1 = max(b[3] for row in rows for b in row)
                            
                            # 对每行的文本块按x坐标排序
                            for row in rows:
                                row.sort(key=lambda b: b[0])
                                # 用制表符分隔单元格
                                row_text = "\t".join([b[4].strip() for b in row])
                                table_text += row_text + "\n"
                            
                            tables.append({
                                "text": table_text.strip(),
                                "bbox": [table_x0, table_y0, table_x1, table_y1]
                            })
                            
            # 如果上述方法没有检测到表格，使用原始的基于分隔符的检测方法
            if not tables:
                # 原始的基于分隔符的检测方法
                for block in page.get_text("blocks"):
                    if block[6] == 0:  # 文本块
                        x0, y0, x1, y1 = block[:4]
                        text = block[4]
                        
                        # 判断：如果文本包含多个由制表符或多个空格分隔的值，可能是表格
                        if re.search(r'\t|\s{3,}', text) and len(text.split()) > 3:
                            lines = text.strip().split('\n')
                            
                            # 如果有多行且每行的结构相似，更可能是表格
                            if len(lines) >= 2 and all(re.search(r'\t|\s{3,}', line) for line in lines[:min(3, len(lines))]):
                                tables.append({
                                    "text": text,
                                    "bbox": [x0, y0, x1, y1]
                                })
        except Exception as e:
            print(f"表格检测出错: {e}")
            # 出错时返回空列表，不影响整体处理
            
        return tables
    
    def group_lines_into_paragraphs(self, text_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于行距和缩进将文本行聚合为段落
        
        参数:
            text_lines: 文本行列表
            
        返回:
            段落列表
        """
        if not text_lines:
            return []
        
        paragraphs = []
        current_paragraph = {
            "text_lines": [text_lines[0]],
            "bbox": text_lines[0]["bbox"].copy()
        }
        
        # 计算平均行高和字体大小，用于确定段落边界
        line_heights = [line["bbox"][3] - line["bbox"][1] for line in text_lines]
        avg_line_height = sum(line_heights) / len(line_heights)
        
        font_sizes = [line["font_size"] for line in text_lines]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        # 段落阈值：行间距大于1.5倍平均行高时认为是新段落
        paragraph_threshold = avg_line_height * 1.5
        
        for i in range(1, len(text_lines)):
            current_line = text_lines[i]
            prev_line = text_lines[i-1]
            
            # 计算当前行与前一行的垂直间距
            vertical_gap = current_line["bbox"][1] - prev_line["bbox"][3]
            
            # 判断是否为新段落的条件：
            # 1. 垂直间距大于阈值
            # 2. 当前行的缩进与前一行明显不同
            # 3. 当前行的字体大小与前一行明显不同（可能是标题）
            is_new_paragraph = False
            
            if vertical_gap > paragraph_threshold:
                is_new_paragraph = True
            elif abs(current_line["bbox"][0] - prev_line["bbox"][0]) > 5:  # 缩进差异
                is_new_paragraph = True
            elif abs(current_line["font_size"] - prev_line["font_size"]) > avg_font_size * 0.2:  # 字体大小差异
                is_new_paragraph = True
            
            if is_new_paragraph:
                # 完成当前段落
                paragraphs.append(current_paragraph)
                # 开始新段落
                current_paragraph = {
                    "text_lines": [current_line],
                    "bbox": current_line["bbox"].copy()
                }
            else:
                # 扩展当前段落
                current_paragraph["text_lines"].append(current_line)
                # 更新段落的边界框
                current_paragraph["bbox"][0] = min(current_paragraph["bbox"][0], current_line["bbox"][0])
                current_paragraph["bbox"][1] = min(current_paragraph["bbox"][1], current_line["bbox"][1])
                current_paragraph["bbox"][2] = max(current_paragraph["bbox"][2], current_line["bbox"][2])
                current_paragraph["bbox"][3] = max(current_paragraph["bbox"][3], current_line["bbox"][3])
        
        # 添加最后一个段落
        paragraphs.append(current_paragraph)
        
        # 为每个段落生成完整文本
        for para in paragraphs:
            para["text"] = " ".join([line["text"] for line in para["text_lines"]])
            # 移除多余的空格
            para["text"] = re.sub(r'\s+', ' ', para["text"]).strip()
        
        return paragraphs
    
    def split_paragraph_into_chunks(self, paragraph: Dict[str, Any], page_num: int, 
                                  max_tokens: int, overlap_ratio: float, 
                                  base_chunk_id: int) -> List[PDFChunk]:
        """将段落按tokens分割成小块
        
        参数:
            paragraph: 段落数据
            page_num: 页码
            max_tokens: 最大token数
            overlap_ratio: 重叠比例
            base_chunk_id: 基础块ID
            
        返回:
            PDFChunk对象列表
        """
        chunks = []
        text = paragraph["text"]
        token_count = self.count_tokens(text)
        
        if token_count <= max_tokens:
            # 如果段落token数不超过最大限制，直接作为一个块
            chunks.append(PDFChunk(
                text=text,
                page=page_num,
                type="paragraph",
                bbox=paragraph["bbox"],
                chunk_id=base_chunk_id,
                token_count=token_count
            ))
        else:
            # 对于长段落，我们需要按tokens分割
            # 这里使用一种简单的基于字符的近似分割方法
            # 对于长段落，我们需要按tokens分割
            # 这里使用一种简单的基于字符的近似分割方法
            avg_chars_per_token = len(text) / token_count
            chunk_size = int(max_tokens * avg_chars_per_token)
            overlap_size = int(chunk_size * overlap_ratio)
            
            start_idx = 0
            chunk_id = base_chunk_id
            max_iterations = token_count // max_tokens + 10  # 最大迭代次数，防止死循环
            iteration = 0
            
            while start_idx < len(text) and iteration < max_iterations:
                iteration += 1
                end_idx = min(start_idx + chunk_size, len(text))
                
                # 尝试在句子边界处分割
                if end_idx < len(text):
                    # 寻找最近的句号、问号或感叹号
                    punctuation_pos = max(
                        text.rfind('.', start_idx, end_idx),
                        text.rfind('?', start_idx, end_idx),
                        text.rfind('!', start_idx, end_idx)
                    )
                    
                    if punctuation_pos > start_idx + chunk_size * 0.5:  # 确保分割点不是太靠近开始
                        end_idx = punctuation_pos + 1
                
                chunk_text = text[start_idx:end_idx].strip()
                if chunk_text:
                    # 近似计算chunk的边界框
                    # 这是一个简化的实现，实际应用中可能需要更精确的映射
                    chunk_bbox = paragraph["bbox"].copy()
                    chunk_bbox[1] = paragraph["bbox"][1] + (start_idx / len(text)) * (paragraph["bbox"][3] - paragraph["bbox"][1])
                    chunk_bbox[3] = paragraph["bbox"][1] + (end_idx / len(text)) * (paragraph["bbox"][3] - paragraph["bbox"][1])
                    
                    chunks.append(PDFChunk(
                        text=chunk_text,
                        page=page_num,
                        type="paragraph",
                        bbox=chunk_bbox,
                        chunk_id=chunk_id,
                        token_count=self.count_tokens(chunk_text)
                    ))
                    
                    chunk_id += 1
                
                # 移动到下一个块，考虑重叠
                start_idx = end_idx - overlap_size
                
                # 防止死循环
                if start_idx >= end_idx or start_idx >= len(text):
                    break
        
        return chunks
    
    def split_pdf(self, pdf_path: str, max_tokens: int = 200, overlap_ratio: float = 0.2) -> List[PDFChunk]:
        """PDF分块的主方法
        
        参数:
            pdf_path: PDF文件路径
            max_tokens: 每个块的最大token数
            overlap_ratio: 重叠比例
            
        返回:
            PDFChunk对象列表
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 提取文本和表格
        pdf_content = self.extract_text_and_tables(pdf_path)
        
        chunks = []
        chunk_id = 0
        
        for idx, page_data in enumerate(pdf_content):
            page_num = page_data["page"]
            print(f"处理第{page_num}页 ({idx+1}/{len(pdf_content)})...")
            
            # 首先处理表格，将其作为独立块
            table_count = len(page_data['tables'])
            print(f"发现{table_count}个表格")
            
            # 确保表格块只被添加一次
            tables_added = 0
            for table in page_data["tables"]:
                token_count = self.count_tokens(table["text"])
                table_chunk = PDFChunk(
                    text=table["text"],
                    page=page_num,
                    type="table",
                    bbox=table["bbox"],
                    chunk_id=chunk_id,
                    token_count=token_count
                )
                print(f"添加表格块 {tables_added+1}: id={chunk_id}, 文本长度={len(table['text'])}, 位置={table['bbox'][:4]}")
                chunks.append(table_chunk)
                chunk_id += 1
                tables_added += 1
            
            # 然后处理文本行，聚合成段落并分割
            paragraphs = self.group_lines_into_paragraphs(page_data["text_lines"])
            
            for paragraph in paragraphs:
                # 判断是否为标题（基于字体大小和位置）
                is_header = False
                avg_font_size = sum([line["font_size"] for line in paragraph["text_lines"]]) / len(paragraph["text_lines"])
                
                # 简单判断：字体较大且位于页面顶部的可能是标题
                if avg_font_size > 12 and paragraph["bbox"][1] < 200:  # 假设页面顶部200像素内
                    is_header = True
                
                para_chunks = self.split_paragraph_into_chunks(
                    paragraph=paragraph,
                    page_num=page_num,
                    max_tokens=max_tokens,
                    overlap_ratio=overlap_ratio,
                    base_chunk_id=chunk_id
                )
                
                # 更新块类型
                for chunk in para_chunks:
                    if is_header:
                        chunk.type = "header"
                
                chunks.extend(para_chunks)
                chunk_id += len(para_chunks)
            
            print(f"  第{page_num}页处理完成，当前总块数: {len(chunks)}")
        
        return chunks
    
    def convert_to_json(self, chunks: List[PDFChunk]) -> List[Dict]:
        """将PDFChunk对象列表转换为JSON可序列化的字典列表
        
        参数:
            chunks: PDFChunk对象列表
            
        返回:
            可序列化的字典列表
        """
        return [{k: v for k, v in chunk.__dict__.items() if v is not None} for chunk in chunks]
    
    def save_chunks_to_json(self, chunks: List[PDFChunk], output_path: str) -> None:
        """将块保存为JSON文件
        
        参数:
            chunks: PDFChunk对象列表
            output_path: 输出文件路径
        """
        # 调试：检查chunks列表中是否有table类型的块
        table_chunks = [chunk for chunk in chunks if chunk.type == "table"]
        print(f"准备保存到JSON：发现{len(table_chunks)}个table类型的块")
        for i, table_chunk in enumerate(table_chunks):
            print(f"  table块{i+1}: id={table_chunk.chunk_id}, 文本长度={len(table_chunk.text)}, 页码={table_chunk.page}")
        
        json_data = self.convert_to_json(chunks)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        print(f"分块结果已保存到: {output_path}")
    
    def print_chunk_summary(self, chunks: List[PDFChunk]) -> None:
        """打印块的摘要信息
        
        参数:
            chunks: PDFChunk对象列表
        """
        print(f"总块数: {len(chunks)}")
        
        # 按类型统计块数
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.type] = type_counts.get(chunk.type, 0) + 1
        print("块类型统计:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}个")
        
        # 按页码统计块数
        page_counts = {}
        for chunk in chunks:
            page_counts[chunk.page] = page_counts.get(chunk.page, 0) + 1
        print("按页码统计:")
        for page_num, count in sorted(page_counts.items()):
            print(f"  第{page_num}页: {count}个块")

# 安装必要的依赖
def install_dependencies():
    """安装必要的依赖库"""
    import subprocess
    import sys
    
    packages = ["pymupdf"]
    if not HAS_TIKTOKEN:
        packages.append("tiktoken")
    
    for package in packages:
        try:
            # 检查是否已安装
            __import__(package)
            print(f"{package} 已安装")
        except ImportError:
            print(f"安装 {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], timeout=30)
            except subprocess.TimeoutExpired:
                print(f"安装 {package}超时，可能需要手动安装")
            except Exception as e:
                print(f"安装 {package}失败: {e}")


def main():
    """主函数，演示PDF版面特征辅助分块功能"""
    print("PDF版面特征辅助分块演示")
    print("=" * 50)
    
    # 安装必要的依赖
    install_dependencies()
    
    # 确保PyMuPDF库已安装
    # 直接在函数内部尝试导入，而不是依赖全局变量
    try:
        import fitz
        has_fitz = True
    except ImportError:
        print("错误: 无法导入PyMuPDF库，请手动安装")
        return
    
    # 如果之前没有导入成功，现在更新全局变量
    global HAS_FITZ
    HAS_FITZ = has_fitz
    
    # PDF文件路径
    pdf_path = "data/billionaires_page-1-5.pdf"
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        print("请确保文件路径正确")
        return
    
    # 初始化分块器
    use_simple = not HAS_TIKTOKEN
    print(f"使用{'简单的字符tokenizer' if use_simple else 'tiktoken'}")
    chunker = PDFLayoutChunker(use_simple_tokenizer=use_simple)
    
    # 分块参数
    max_tokens = 200
    overlap_ratio = 0.2
    print(f"使用最大token数={max_tokens}, 重叠率={overlap_ratio}进行分块...")
    
    # 执行分块
    chunks = chunker.split_pdf(pdf_path, max_tokens, overlap_ratio)
    
    # 打印摘要
    chunker.print_chunk_summary(chunks)
    
    # 保存结果
    output_path = "results/pdf_layout_chunks.json"
    chunker.save_chunks_to_json(chunks, output_path)
    
    print("\nPDF分块演示完成！")

if __name__ == "__main__":
    main()