#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表格抽取工具
功能：从PDF文件中抽取表格，并导出为JSON和Markdown格式
输入：PDF页面图/文本（特别是data子目录下的就业指导考试表格.pdf）
处理：使用pdfplumber抽取表格，进行列对齐率校验、合并单元格拆解
输出：表格JSON（标准schema）+ Markdown（可索引）

使用说明：
1. 安装依赖：pip install pdfplumber pandas
2. 如需Excel导出功能：pip install openpyxl
3. 运行示例：python 08-table_extract.py --pdf_path data/就业指导考试表格.pdf

注：程序会自动在输出目录创建表格文件，并在终端显示处理进度和结果。
"""

import os
import json
import logging
import argparse
import pandas as pd
import pdfplumber
from typing import List, Dict, Any, Optional, Tuple

# 配置日志 - 同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("table_extract_log.txt"),  # 日志文件
        logging.StreamHandler()  # 控制台输出
    ]
)
logger = logging.getLogger('TableExtractor')  # 创建日志记录器

class TableExtractor:
    """表格抽取器，负责从PDF中抽取表格并进行处理"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化表格抽取器
        
        Args:
            config: 配置参数
        """
        # 默认配置
        self.default_config = {
            'output_dir': './table_output',  # 输出目录
            'min_row_count': 2,              # 最小行数
            'min_col_count': 2,              # 最小列数
            'vertical_alignment_threshold': 0.4,  # 列对齐率阈值（降低要求以便捕获更多表格）
            'edge_tolerance': 1.0,           # 边缘容忍度
            'snap_tolerance': 3.0,           # 捕捉容忍度
            'extract_kwargs': {
                'vertical_strategy': 'lines',  # 垂直线检测策略
                'horizontal_strategy': 'lines',  # 水平线检测策略
                'snap_x_tolerance': 3.0,       # X轴捕捉容忍度
                'snap_y_tolerance': 3.0,       # Y轴捕捉容忍度
            }
        }
        
        # 合并用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            # 合并extract_kwargs
            if 'extract_kwargs' in config:
                self.config['extract_kwargs'].update(config['extract_kwargs'])
        
        # 确保输出目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        logger.info(f"表格抽取器初始化完成，配置: {self.config}")
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        从PDF文件中抽取所有表格
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            包含所有抽取表格信息的列表
        """
        if not os.path.exists(pdf_path):
            logger.error(f"文件不存在: {pdf_path}")
            return []
        
        tables_info = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 获取PDF文件名，不包含扩展名
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"正在处理第{page_num}页")
                    
                    try:
                        # 尝试直接提取表格（不使用额外参数）
                        tables = page.extract_tables()
                    except Exception as e:
                        logger.error(f"提取表格时出错: {e}")
                        tables = []
                    
                    if not tables:
                        logger.info(f"第{page_num}页未发现表格")
                        continue
                    
                    logger.info(f"第{page_num}页发现{len(tables)}个表格")
                    
                    # 获取页面尺寸，用于坐标计算
                    page_width = page.width
                    page_height = page.height
                    
                    # 处理每个表格
                    for table_idx, table in enumerate(tables):
                        # 跳过空表格或行数/列数过少的表格
                        if not table or len(table) < self.config['min_row_count']:
                            logger.warning(f"跳过第{page_num}页第{table_idx+1}个表格（行数不足）")
                            continue
                        
                        # 获取表格的边界框
                        table_bbox = self._get_table_bbox(page, table_idx)
                        
                        # 校验列对齐率
                        alignment_score = self._check_column_alignment(table)
                        
                        # 过滤对齐率低的表格
                        if alignment_score < self.config['vertical_alignment_threshold']:
                            logger.warning(f"跳过第{page_num}页第{table_idx+1}个表格（列对齐率低: {alignment_score:.2f}）")
                            continue
                        
                        # 处理合并单元格
                        processed_table = self._handle_merged_cells(table)
                        
                        # 生成表格ID
                        table_id = f"{pdf_name}_page{page_num}_table{table_idx+1}"
                        
                        # 转换为标准schema
                        table_info = {
                            'table_id': table_id,
                            'pdf_path': pdf_path,
                            'page_num': page_num,
                            'table_index': table_idx + 1,
                            'bbox': table_bbox,
                            'page_dimensions': {
                                'width': page_width,
                                'height': page_height
                            },
                            'row_count': len(processed_table),
                            'col_count': len(processed_table[0]) if processed_table else 0,
                            'alignment_score': alignment_score,
                            'data': processed_table,
                            'metadata': {
                                'extraction_method': 'pdfplumber',
                                'config_used': self.config['extract_kwargs']
                            }
                        }
                        
                        tables_info.append(table_info)
                        logger.info(f"成功处理第{page_num}页第{table_idx+1}个表格")
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {e}")
        
        return tables_info
    
    def _get_table_bbox(self, page: Any, table_idx: int) -> Dict[str, float]:
        """
        获取表格的边界框
        
        Args:
            page: PDF页面对象
            table_idx: 表格索引
        
        Returns:
            包含表格边界坐标的字典
        """
        try:
            # 尝试使用更简单的方式获取表格边界
            # 直接返回页面边界作为表格边界（简化实现）
            return {
                'x0': 0,
                'top': 0,
                'x1': page.width,
                'bottom': page.height
            }
        except Exception as e:
            logger.warning(f"获取表格边界框时出错: {e}")
        
        # 如果出错，返回页面边界
        return {
            'x0': 0,
            'top': 0,
            'x1': page.width,
            'bottom': page.height
        }
    
    def _check_column_alignment(self, table: List[List[str]]) -> float:
        """
        检查表格的列对齐率
        
        Args:
            table: 表格数据
        
        Returns:
            对齐率分数 (0-1)
        
        说明：
        - 计算每列的非空值比例，用于评估表格的结构完整性
        - 值越高表示表格结构越完整，列对齐越好
        - 当前阈值设置为0.4，低于此值的表格将被跳过
        "
        """
        if not table or len(table) < 2:
            return 0.0
        
        # 计算每列的非空值比例（优化算法）
        col_count = max(len(row) for row in table)
        valid_col_ratios = []
        
        for col_idx in range(col_count):
            non_empty_count = 0
            total_count = 0
            
            for row in table:
                if col_idx < len(row):
                    cell_content = str(row[col_idx]).strip() if row[col_idx] else ''
                    if cell_content:
                        non_empty_count += 1
                    total_count += 1
            
            # 计算该列的有效比例
            if total_count > 0:
                valid_ratio = non_empty_count / total_count
                valid_col_ratios.append(valid_ratio)
        
        # 返回平均有效比例
        return sum(valid_col_ratios) / len(valid_col_ratios) if valid_col_ratios else 0.0
    
    def _handle_merged_cells(self, table: List[List[str]]) -> List[List[str]]:
        """
        处理合并单元格，尽可能还原原始表格结构
        
        Args:
            table: 原始表格数据
        
        Returns:
            处理后的表格数据
        
        说明：
        - 首先统一每行的列数，确保表格结构一致
        - 然后检测并标记可能的合并单元格，通过在值后添加"(merged)"标识
        - 这是一个基于内容的启发式方法，对于复杂表格可能需要更高级的检测算法
        """
        if not table:
            return []
        
        # 统一每行的列数
        max_cols = max(len(row) for row in table)
        processed_table = []
        
        for row in table:
            # 补充空字符串使每行列数一致
            processed_row = row + [''] * (max_cols - len(row))
            processed_table.append(processed_row)
        
        # 简单的合并单元格检测：检查是否有相邻行的相同内容
        # 这是一个简化实现，更复杂的合并单元格检测需要分析PDF的底层结构
        for i in range(len(processed_table) - 1):
            for j in range(len(processed_table[i])):
                current_cell = processed_table[i][j]
                next_cell = processed_table[i+1][j]
                
                # 如果当前单元格不为空，而下一行同一列的单元格为空，可能是纵向合并
                if current_cell and not next_cell:
                    processed_table[i+1][j] = f"{current_cell} (merged)"
        
        return processed_table
    
    def export_tables_to_json(self, tables_info: List[Dict[str, Any]], output_path: str) -> bool:
        """
        将表格数据导出为JSON格式
        
        Args:
            tables_info: 表格信息列表
            output_path: 输出文件路径
        
        Returns:
            是否导出成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tables_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功导出{len(tables_info)}个表格到JSON文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出JSON文件时出错: {e}")
            return False
    
    def export_tables_to_markdown(self, tables_info: List[Dict[str, Any]], output_path: str) -> bool:
        """
        将表格数据导出为Markdown格式
        
        Args:
            tables_info: 表格信息列表
            output_path: 输出文件路径
        
        Returns:
            是否导出成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for table_info in tables_info:
                    # 写入表格元信息
                    f.write(f"# 表格ID: {table_info['table_id']}\n")
                    f.write(f"- 来源文件: {os.path.basename(table_info['pdf_path'])}\n")
                    f.write(f"- 页码: {table_info['page_num']}\n")
                    f.write(f"- 表格索引: {table_info['table_index']}\n")
                    f.write(f"- 坐标: x0={table_info['bbox']['x0']:.2f}, top={table_info['bbox']['top']:.2f}, ")
                    f.write(f"x1={table_info['bbox']['x1']:.2f}, bottom={table_info['bbox']['bottom']:.2f}\n")
                    f.write(f"- 行列数: {table_info['row_count']}行 × {table_info['col_count']}列\n")
                    f.write(f"- 列对齐率: {table_info['alignment_score']:.2f}\n\n")
                    
                    # 写入Markdown表格
                    table_data = table_info['data']
                    if not table_data:
                        f.write("表格数据为空\n\n")
                        continue
                    
                    # 写入表头分隔线
                    headers = table_data[0]
                    separator = ['---'] * len(headers)
                    
                    # 写入表格内容
                    f.write('| ' + ' | '.join(headers) + ' |\n')
                    f.write('| ' + ' | '.join(separator) + ' |\n')
                    
                    for row in table_data[1:]:
                        # 确保行的列数与表头一致
                        row_data = row + [''] * (len(headers) - len(row))
                        f.write('| ' + ' | '.join(str(cell).strip() for cell in row_data) + ' |\n')
                    
                    f.write('\n\n')
                    f.write('---\n\n')  # 分隔符
            
            logger.info(f"成功导出{len(tables_info)}个表格到Markdown文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出Markdown文件时出错: {e}")
            return False
    
    def export_tables_to_excel(self, tables_info: List[Dict[str, Any]], output_path: str) -> bool:
        """
        将表格数据导出为Excel格式（额外功能）
        
        Args:
            tables_info: 表格信息列表
            output_path: 输出文件路径
        
        Returns:
            是否导出成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for table_info in tables_info:
                    # 为工作表名称创建一个简短的标识符
                    sheet_name = f"P{table_info['page_num']}_T{table_info['table_index']}"
                    # Excel工作表名称不能超过31个字符
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:28] + '...'
                    
                    # 创建DataFrame
                    df = pd.DataFrame(table_info['data'])
                    
                    # 写入工作表
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    
                    # 添加表格元信息到工作表顶部
                    worksheet = writer.sheets[sheet_name]
                    worksheet['A1'] = f"表格ID: {table_info['table_id']}"
                    worksheet['A2'] = f"来源文件: {os.path.basename(table_info['pdf_path'])}"
                    worksheet['A3'] = f"页码: {table_info['page_num']}"
                    
                    # 调整数据区域的起始位置
                    if not df.empty:
                        # 移动数据到第5行开始
                        for row_idx, row in enumerate(df.itertuples(index=False, name=None), 5):
                            for col_idx, value in enumerate(row):
                                worksheet.cell(row=row_idx, column=col_idx+1, value=value)
            
            logger.info(f"成功导出{len(tables_info)}个表格到Excel文件: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出Excel文件时出错: {e}")
            return False


def main():
    """
    主函数 - 程序入口点
    
    功能流程：
    1. 解析命令行参数
    2. 初始化表格抽取器
    3. 从PDF文件中抽取表格
    4. 将表格导出为JSON、Markdown格式（可选导出Excel）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF表格抽取工具')
    parser.add_argument('--pdf_path', type=str, 
                        default='data/就业指导考试表格.pdf',
                        help='PDF文件路径，默认为data/就业指导考试表格.pdf')
    parser.add_argument('--output_dir', type=str, 
                        default='./table_output',
                        help='输出目录，默认为./table_output')
    parser.add_argument('--skip_excel', action='store_true', 
                        help='跳过Excel导出')
    args = parser.parse_args()
    
    # 初始化表格抽取器
    extractor = TableExtractor({
        'output_dir': args.output_dir
    })
    
    # 抽取表格
    print("===== 开始表格抽取任务 =====")
    logger.info(f"开始处理文件: {args.pdf_path}")
    tables_info = extractor.extract_tables_from_pdf(args.pdf_path)
    
    if not tables_info:
        logger.warning("未提取到任何表格")
        print("⚠️  警告：未从PDF文件中提取到任何表格。可能原因：")
        print("  - PDF文件中可能没有结构化表格")
        print("  - 表格格式可能不符合当前检测算法")
        print("  - 可尝试调整vertical_alignment_threshold参数值")
        return
    
    # 获取输出文件名
    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
    
    # 导出JSON
    json_output_path = os.path.join(args.output_dir, f"{pdf_name}_tables.json")
    json_success = extractor.export_tables_to_json(tables_info, json_output_path)
    if json_success:
        print(f"✅  已成功导出JSON格式表格：{json_output_path}")
    
    # 导出Markdown
    md_output_path = os.path.join(args.output_dir, f"{pdf_name}_tables.md")
    md_success = extractor.export_tables_to_markdown(tables_info, md_output_path)
    if md_success:
        print(f"✅  已成功导出Markdown格式表格：{md_output_path}")
    
    # 导出Excel（可选）
    if not args.skip_excel:
        print("📊  正在尝试导出Excel格式...")
        excel_output_path = os.path.join(args.output_dir, f"{pdf_name}_tables.xlsx")
        excel_success = extractor.export_tables_to_excel(tables_info, excel_output_path)
        if not excel_success:
            print("⚠️  提示：Excel导出需要openpyxl库支持。")
            print("  可通过以下命令安装：pip install openpyxl")
            print("  或使用--skip_excel参数跳过Excel导出")
    
    print(f"\n🎉  表格抽取任务已完成！成功提取了{len(tables_info)}个表格")
    print(f"📁  所有输出文件已保存至：{args.output_dir}")
    print("===== 任务完成 =====")
    logger.info("表格抽取和导出任务已完成")


if __name__ == '__main__':
    main()