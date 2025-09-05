#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF解析路由选择器

功能说明：
- 自动检测PDF文件是否有文本层及文本层质量
- 评估PDF版面复杂度和结构特征
- 根据检测结果智能选择最佳解析路线：pdfplumber/版面重建/OCR
- 输出路由选择结果、解析产物和详细决策日志
- 支持快速探测，避免"全上OCR"的资源浪费

使用场景：
- 大规模PDF文档处理系统
- 自动化文档转换流程
- 需要智能选择PDF解析策略的应用
- 混合类型PDF文档的批量处理

设计原则：
- 快速探测，避免不必要的完整解析
- 优先使用轻量级解析方法，降低资源消耗
- 详细日志记录，便于问题追溯和优化
- 模块化设计，易于扩展和维护

依赖说明：
- PyMuPDF (fitz): PDF文件处理核心库
- pdfplumber: 高质量文本提取
- pytesseract: OCR文本识别
- OpenCV (cv2): 图像处理
- NumPy: 数值计算
- PIL (Pillow): 图像处理
"""

import os
import sys
import time
import json
import logging
import subprocess
import platform
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io
import pytesseract

# 确保中文显示正常（Windows环境下防止输出乱码）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_tesseract_installed() -> bool:
    """
    检查Tesseract OCR是否已安装并可用，优先使用绝对路径
    
    Returns:
        bool: Tesseract是否已安装并可用
    """
    # Windows系统默认安装路径
    DEFAULT_TESSERACT_PATH = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    try:
        # 首先尝试使用绝对路径
        if os.path.exists(DEFAULT_TESSERACT_PATH):
            result = subprocess.run(
                [DEFAULT_TESSERACT_PATH, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=5  # 添加超时防止命令挂起
            )
            if result.returncode == 0:
                # 检查中文语言包是否存在
                tessdata_path = os.path.join(os.path.dirname(DEFAULT_TESSERACT_PATH), 'tessdata')
                chi_sim_path = os.path.join(tessdata_path, 'chi_sim.traineddata')
                if os.path.exists(chi_sim_path):
                    logger.info("检测到Tesseract中文语言包已安装")
                else:
                    logger.warning(f"Tesseract已安装，但未找到中文语言包(chi_sim.traineddata)，请从以下地址下载:\n"+
                                  f"https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata\n"+
                                  f"并放置到: {tessdata_path}")
                return True
        
        # 然后尝试在PATH中查找
        result = subprocess.run(
            ['tesseract', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5
        )
        # 如果命令成功运行，返回True
        if result.returncode == 0:
            # 尝试获取tesseract的数据目录
            try:
                import shutil
                tess_config = subprocess.run(
                    ['tesseract', '--print-parameters', 'tessdata_dir'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                # 检查中文语言包
                tessdata_path = os.environ.get('TESSDATA_PREFIX', '')
                if not tessdata_path:
                    # 尝试默认路径
                    tessdata_path = os.path.join(os.path.dirname(shutil.which('tesseract')), 'tessdata')
                
                chi_sim_path = os.path.join(tessdata_path, 'chi_sim.traineddata')
                if os.path.exists(chi_sim_path):
                    logger.info("检测到Tesseract中文语言包已安装")
                else:
                    logger.warning(f"Tesseract已安装，但未找到中文语言包(chi_sim.traineddata)，请从以下地址下载:\n"+
                                  f"https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata\n"+
                                  f"并放置到适当的tessdata目录")
            except:
                logger.warning("无法确定Tesseract数据目录位置，请确保中文语言包已安装")
            return True
        return False
    except FileNotFoundError:
        # 命令未找到，检查Windows常见安装路径
        if platform.system().lower() == 'windows':
            # 检查Windows常见安装路径
            common_paths = [
                DEFAULT_TESSERACT_PATH,
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Tesseract-OCR', 'tesseract.exe')
            ]
            for path in common_paths:
                if os.path.exists(path):
                    # 检查中文语言包
                    tessdata_path = os.path.join(os.path.dirname(path), 'tessdata')
                    chi_sim_path = os.path.join(tessdata_path, 'chi_sim.traineddata')
                    if os.path.exists(chi_sim_path):
                        logger.info("检测到Tesseract中文语言包已安装")
                    else:
                        logger.warning(f"检测到Tesseract安装在{path}，但未找到中文语言包(chi_sim.traineddata)，请从以下地址下载:\n"+
                                      f"https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata\n"+
                                      f"并放置到: {tessdata_path}")
                    logger.warning(f"检测到Tesseract安装在{path}，但未添加到系统PATH")
                    return True  # 返回True，因为我们可以使用绝对路径
        return False
    except Exception as e:
        logger.error(f"检查Tesseract安装状态时出错: {e}")
        return False

def install_tesseract() -> bool:
    """
    根据操作系统自动安装Tesseract OCR
    
    Returns:
        bool: 安装是否成功
    """
    os_type = platform.system().lower()
    
    if 'windows' in os_type:
        try:
            logger.info("Windows系统: 正在尝试安装/配置Tesseract OCR...")
            
            # 检查是否已安装
            common_paths = [
                os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'), 'Tesseract-OCR', 'tesseract.exe'),
                os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'), 'Tesseract-OCR', 'tesseract.exe')
            ]
            
            installed_path = None
            for path in common_paths:
                if os.path.exists(path):
                    installed_path = path
                    break
            
            if installed_path:
                # 已安装但可能未添加到PATH
                tesseract_dir = os.path.dirname(installed_path)
                path_env = os.environ.get('PATH', '')
                
                if tesseract_dir not in path_env:
                    logger.warning(f"检测到Tesseract已安装在{installed_path}，但未添加到系统PATH")
                    logger.warning(f"请手动将{tesseract_dir}添加到系统环境变量PATH中")
                    logger.warning("或者可以在代码中临时设置环境变量:")
                    logger.warning(f"import os; os.environ['PATH'] += ';' + '{tesseract_dir}'")
                return False
            else:
                # 未安装，提供详细安装指南
                logger.info("请从以下地址下载并安装Tesseract OCR:")
                logger.info("https://github.com/UB-Mannheim/tesseract/wiki")
                logger.info("安装时请记住安装路径，并将其添加到系统环境变量PATH中")
                logger.info("建议同时安装中文语言包")
                return False
        except Exception as e:
            logger.error(f"安装/检查Tesseract时出错: {e}")
            return False
    elif 'linux' in os_type:
        try:
            logger.info("Linux系统: 正在尝试安装Tesseract OCR...")
            # 在Ubuntu/Debian系统上安装Tesseract
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr', 'tesseract-ocr-chi-sim'], check=True)
            logger.info("Tesseract OCR已成功安装")
            return True
        except Exception as e:
            logger.error(f"安装Tesseract时出错: {e}")
            return False
    elif 'darwin' in os_type:
        try:
            logger.info("macOS系统: 正在尝试安装Tesseract OCR...")
            # 在macOS上使用Homebrew安装Tesseract
            subprocess.run(['brew', 'install', 'tesseract', 'tesseract-lang'], check=True)
            logger.info("Tesseract OCR已成功安装")
            return True
        except Exception as e:
            logger.error(f"安装Tesseract时出错: {e}")
            return False
    else:
        logger.error(f"不支持的操作系统: {os_type}")
        return False

# 全局常量定义
# 路由类型常量
ROUTE_PDFPLUMBER = 'pdfplumber'  # 轻量级文本提取
ROUTE_LAYOUT = 'layout'          # 版面重建解析
ROUTE_OCR = 'ocr'                # 光学字符识别

# 文本质量评级
TEXT_QUALITY_GOOD = 'good'       # 高质量文本层
TEXT_QUALITY_POOR = 'poor'       # 低质量文本层
TEXT_QUALITY_NONE = 'none'       # 无文本层

# 版面复杂度评级
LAYOUT_SIMPLE = 'simple'         # 简单版面
LAYOUT_MODERATE = 'moderate'     # 中等复杂度版面
LAYOUT_COMPLEX = 'complex'       # 复杂版面

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_route_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PDFRouteSelector')

class PDFRouteSelector:
    """PDF解析路由选择器，负责自动选择最佳的PDF解析策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化PDF路由选择器
        
        Args:
            config: 用户自定义配置参数，可覆盖默认配置
        """
        # 默认配置参数
        self.default_config = {
            'text_density_threshold': 0.1,  # 文本密度阈值，低于此值认为文本层质量低
            'image_page_ratio_threshold': 0.5,  # 图像页面比例阈值
            'complex_layout_threshold': 0.6,  # 复杂版面阈值
            'sample_pages': 3,  # 用于采样分析的页面数量
            'output_dir': './pdf_output',  # 解析产物输出目录
            'temp_dir': './temp',           # 临时文件目录
            'tesseract_path': r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Tesseract绝对路径
        }
        
        # 合并用户配置与默认配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 确保必要的目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        
        # 初始化运行统计信息
        self.stats = {
            'total_processed': 0,        # 处理的文件总数
            'pdfplumber_routes': 0,      # PDFPlumber路由选择次数
            'layout_routes': 0,          # 版面重建路由选择次数
            'ocr_routes': 0,             # OCR路由选择次数
            'total_processing_time': 0   # 总处理时间(秒)
        }
        
        # 检查Tesseract OCR是否安装
        self.tesseract_available = check_tesseract_installed()
        if not self.tesseract_available:
            logger.warning("未检测到Tesseract OCR")
            # 尝试自动安装Tesseract（仅在非Windows系统上可能成功）
            install_success = install_tesseract()
            if install_success:
                self.tesseract_available = True
                logger.info("Tesseract OCR安装成功，可以使用OCR功能")
            else:
                logger.warning("Tesseract OCR安装失败或未配置，OCR功能不可用")
                logger.warning("对于Windows用户，请手动安装Tesseract并添加到系统PATH环境变量")
        else:
            logger.info("检测到Tesseract OCR已安装")
            
        # 在实例中保存Tesseract状态供路由选择时使用
        self.config['tesseract_available'] = self.tesseract_available
        
        logger.info(f"PDF路由选择器初始化完成，配置: {self.config}")
    
    def detect_text_layer(self, pdf_path: str) -> Dict[str, Any]:
        """
        检测PDF是否有文本层，以及文本层的质量
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含文本层检测结果的字典
        """
        result = {
            'has_text_layer': False,
            'text_density': 0.0,
            'text_pages_ratio': 0.0,
            'estimated_text_quality': 'unknown'
        }
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            text_pages = 0
            total_chars = 0
            
            # 采样分析
            sample_size = min(self.config['sample_pages'], total_pages)
            sample_indices = np.linspace(0, total_pages - 1, sample_size, dtype=int)
            
            for page_idx in sample_indices:
                page = doc.load_page(page_idx)
                text = page.get_text()
                
                # 计算页面文本密度
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                
                if text.strip():
                    text_pages += 1
                    total_chars += len(text)
                    
            # 计算统计指标
            if sample_size > 0:
                result['has_text_layer'] = text_pages > 0
                result['text_pages_ratio'] = text_pages / sample_size
                
                # 估算文本密度（字符数/页面面积）
                avg_chars_per_page = total_chars / sample_size if sample_size > 0 else 0
                avg_page_area = doc[0].rect.width * doc[0].rect.height if total_pages > 0 else 1
                result['text_density'] = avg_chars_per_page / avg_page_area
                
                # 评估文本质量
                if result['text_density'] > self.config['text_density_threshold']:
                    result['estimated_text_quality'] = 'good'
                elif result['has_text_layer']:
                    result['estimated_text_quality'] = 'poor'
                else:
                    result['estimated_text_quality'] = 'none'
            
            doc.close()
            
        except Exception as e:
            logger.error(f"检测文本层时出错: {e}")
            result['error'] = str(e)
        
        logger.info(f"文本层检测结果: {result}")
        return result
    
    def assess_layout_complexity(self, pdf_path: str) -> Dict[str, Any]:
        """
        评估PDF版面复杂度
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含版面复杂度评估结果的字典
        """
        result = {
            'is_complex': False,
            'complexity_score': 0.0,
            'estimated_layout_type': 'unknown'
        }
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            complexity_scores = []
            
            # 采样分析
            sample_size = min(self.config['sample_pages'], total_pages)
            sample_indices = np.linspace(0, total_pages - 1, sample_size, dtype=int)
            
            for page_idx in sample_indices:
                page = doc.load_page(page_idx)
                
                # 获取页面中的文本块、图像和表格数量
                blocks = page.get_text("dict")['blocks']
                text_blocks = 0
                image_blocks = 0
                
                for block in blocks:
                    if block['type'] == 0:  # 文本块
                        text_blocks += 1
                    elif block['type'] == 1:  # 图像块
                        image_blocks += 1
                
                # 计算页面复杂度分数
                # 文本块数量多、图像块比例高，认为版面复杂
                page_complexity = 0
                if blocks:
                    text_block_ratio = text_blocks / len(blocks)
                    image_block_ratio = image_blocks / len(blocks)
                    
                    # 复杂版面特征：文本块多且分散，或包含大量图像
                    page_complexity = min(1.0, 
                                         (min(text_blocks / 10, 1.0) * 0.5 + 
                                          min(image_block_ratio * 2, 1.0) * 0.5))
                
                complexity_scores.append(page_complexity)
            
            # 计算平均复杂度
            if complexity_scores:
                avg_complexity = sum(complexity_scores) / len(complexity_scores)
                result['complexity_score'] = avg_complexity
                result['is_complex'] = avg_complexity > self.config['complex_layout_threshold']
                
                # 估计版面类型
                if avg_complexity < 0.2:
                    result['estimated_layout_type'] = 'simple'
                elif avg_complexity < 0.5:
                    result['estimated_layout_type'] = 'moderate'
                else:
                    result['estimated_layout_type'] = 'complex'
            
            doc.close()
            
        except Exception as e:
            logger.error(f"评估版面复杂度时出错: {e}")
            result['error'] = str(e)
        
        logger.info(f"版面复杂度评估结果: {result}")
        return result
    
    def select_route(self, pdf_path: str) -> Dict[str, Any]:
        """
        根据PDF特征选择最佳解析路线
        
        核心路由决策函数，基于文本层质量、版面复杂度评估结果以及Tesseract可用性，
        智能选择最适合的解析策略，遵循"能轻则轻，必须才重"的原则。
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含路由选择结果的完整报告
        """
        start_time = time.time()
        
        # 收集基本文件信息
        file_name = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        
        # 第一阶段：检测文本层特征
        text_layer_result = self.detect_text_layer(pdf_path)
        
        # 第二阶段：评估版面复杂度
        layout_result = self.assess_layout_complexity(pdf_path)
        
        # 获取Tesseract可用性状态
        tesseract_available = self.tesseract_available
        
        # 路由决策初始化
        route_decision = {
            'route': 'unknown',
            'reason': [],
            'confidence': 0.0
        }
        
        # 核心决策逻辑：基于文本层质量和版面复杂度的组合策略
        # 策略1：有高质量文本层的PDF
        if text_layer_result['has_text_layer'] and text_layer_result['estimated_text_quality'] == TEXT_QUALITY_GOOD:
            if not layout_result['is_complex']:
                # 简单版面，优先使用轻量级的pdfplumber
                route_decision = {
                    'route': ROUTE_PDFPLUMBER,
                    'reason': ['有高质量文本层', '版面简单'],
                    'confidence': 0.95
                }
            else:
                # 复杂版面，需要进行版面重建以保持文本位置关系
                route_decision = {
                    'route': ROUTE_LAYOUT,
                    'reason': ['有高质量文本层', '版面复杂需要重建'],
                    'confidence': 0.9
                }
        
        # 策略2：有文本层但质量较差的PDF
        elif text_layer_result['has_text_layer'] and text_layer_result['estimated_text_quality'] == TEXT_QUALITY_POOR:
            if layout_result['is_complex']:
                # 复杂版面+低质量文本，优先使用OCR，但考虑Tesseract可用性
                if tesseract_available:
                    route_decision = {
                        'route': ROUTE_OCR,
                        'reason': ['文本层质量差', '版面复杂', 'Tesseract可用'],
                        'confidence': 0.85
                    }
                else:
                    # Tesseract不可用，降级使用版面重建
                    route_decision = {
                        'route': ROUTE_LAYOUT,
                        'reason': ['文本层质量差', '版面复杂', 'Tesseract不可用，使用版面重建作为备选'],
                        'confidence': 0.6
                    }
            else:
                # 简单版面，尝试使用pdfplumber并进行后处理
                route_decision = {
                    'route': ROUTE_PDFPLUMBER,
                    'reason': ['有文本层但质量较差', '版面简单'],
                    'confidence': 0.8
                }
        
        # 策略3：没有文本层的PDF（图像PDF）
        else:
            # 无文本层，优先使用OCR，但考虑Tesseract可用性
            if tesseract_available:
                route_decision = {
                    'route': ROUTE_OCR,
                    'reason': ['无文本层，需要OCR识别', 'Tesseract可用'],
                    'confidence': 0.95
                }
            else:
                # Tesseract不可用，无法进行OCR识别
                route_decision = {
                    'route': 'no_text_layer_fallback',
                    'reason': ['无文本层，需要OCR识别', 'Tesseract不可用，无法进行OCR识别'],
                    'confidence': 0.0
                }
                logger.warning(f"文件 {file_name} 无文本层，且Tesseract OCR不可用，无法提取文本内容")
                logger.warning("请安装Tesseract OCR并添加到系统环境变量PATH中以启用OCR功能")
        
        # 生成完整的路由报告
        route_report = {
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'name': file_name,
                'path': pdf_path,
                'size_mb': round(file_size, 2)
            },
            'text_layer_analysis': text_layer_result,
            'layout_analysis': layout_result,
            'route_decision': route_decision,
            'processing_time_seconds': round(time.time() - start_time, 3)
        }
        
        # 更新统计信息
        self.stats['total_processed'] += 1
        self.stats['total_processing_time'] += route_report['processing_time_seconds']
        
        if route_decision['route'] == 'pdfplumber':
            self.stats['pdfplumber_routes'] += 1
        elif route_decision['route'] == 'layout':
            self.stats['layout_routes'] += 1
        elif route_decision['route'] == 'ocr':
            self.stats['ocr_routes'] += 1
        
        # 保存路由日志
        self._save_route_log(route_report)
        
        logger.info(f"路由选择完成: {file_name} -> {route_decision['route']}, 置信度: {route_decision['confidence']}")
        
        return route_report
    
    def _execute_no_text_layer_fallback(self, pdf_path: str) -> Dict[str, Any]:
        """
        当PDF无文本层且Tesseract不可用时的备选处理方案
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理结果
        """
        try:
            # 生成输出文件名
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            info_output_path = os.path.join(self.config['output_dir'], f"{name_without_ext}_info.txt")
            
            # 提取基本信息并保存
            with open(info_output_path, 'w', encoding='utf-8') as f:
                f.write(f"文件: {base_name}\n")
                f.write(f"状态: 无文本层，且Tesseract OCR不可用\n")
                f.write(f"建议: 请安装Tesseract OCR并添加到系统环境变量PATH中以启用OCR功能\n")
                f.write(f"Tesseract下载地址: https://github.com/UB-Mannheim/tesseract/wiki\n")
                
            logger.info(f"已保存文档信息至: {info_output_path}")
            
            return {
                'success': False,
                'method': 'info_only',
                'output_files': [info_output_path],
                'message': '无文本层，且Tesseract OCR不可用，无法提取文本内容'
            }
        except Exception as e:
            logger.error(f"保存文档信息时出错: {e}")
            return {'success': False, 'error': str(e)}
            
    def execute_route(self, pdf_path: str, route_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行选择的解析路线
        
        Args:
            pdf_path: PDF文件路径
            route_report: 路由报告
            
        Returns:
            包含解析结果的字典
        """
        route = route_report['route_decision']['route']
        start_time = time.time()
        
        # 根据选择的路由执行解析
        if route == 'pdfplumber':
            result = self._execute_pdfplumber_route(pdf_path)
        elif route == 'layout':
            result = self._execute_layout_route(pdf_path)
        elif route == 'ocr':
            result = self._execute_ocr_route(pdf_path)
        elif route == 'no_text_layer_fallback':
            # 处理无文本层且Tesseract不可用的情况
            result = self._execute_no_text_layer_fallback(pdf_path)
        else:
            result = {'error': f'未知的路由: {route}'}
        
        # 添加执行信息
        execution_info = {
            'executed_route': route,
            'execution_time_seconds': round(time.time() - start_time, 3),
            'result': result
        }
        
        # 更新路由报告
        route_report['execution'] = execution_info
        
        # 保存完整报告
        self._save_full_report(route_report)
        
        return route_report
    
    def _execute_pdfplumber_route(self, pdf_path: str) -> Dict[str, Any]:
        """执行pdfplumber解析路线"""
        try:
            # 导入pdfplumber
            import pdfplumber
            
            # 生成输出文件名
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            text_output_path = os.path.join(self.config['output_dir'], f"{name_without_ext}_pdfplumber.txt")
            
            # 使用pdfplumber提取文本
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text_content.append(f"==== 第{page_num}页 ====\n{page_text}\n")
            
            # 保存提取的文本
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            
            logger.info(f"pdfplumber解析完成，已保存至: {text_output_path}")
            
            return {
                'success': True,
                'method': 'pdfplumber',
                'output_files': [text_output_path],
                'page_count': len(text_content)
            }
        except ImportError:
            logger.error("pdfplumber库未安装，请运行: pip install pdfplumber")
            return {'success': False, 'error': 'pdfplumber库未安装'}
        except Exception as e:
            logger.error(f"pdfplumber解析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_layout_route(self, pdf_path: str) -> Dict[str, Any]:
        """
        执行版面重建解析路线
        
        针对复杂版面的PDF文档，通过提取文本块并按空间位置排序，
        尽可能保持原始文档的排版结构和阅读顺序。
        
        注意：这是一个简化实现，实际应用中可以集成更专业的版面分析库。
        """
        try:
            # 这里简化实现，实际应用中可能需要使用更复杂的版面分析库
            # 例如LayoutLM、DocTr等
            
            # 生成输出文件名
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            text_output_path = os.path.join(self.config['output_dir'], f"{name_without_ext}_layout.txt")
            
            # 使用PyMuPDF进行更精细的文本提取
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num, page in enumerate(doc, 1):
                # 使用更精细的文本提取模式，获取所有文本块
                blocks = page.get_text("blocks")
                # 按空间位置排序文本块：先按y坐标(垂直方向)，再按x坐标(水平方向)
                blocks.sort(key=lambda b: (b[1], b[0]))  # 先按y坐标，再按x坐标
                
                page_text = []
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        page_text.append(text)
                
                # 按顺序拼接页面文本
                text_content.append(f"==== 第{page_num}页 ====\n{chr(10).join(page_text)}\n")
            
            # 保存提取的文本
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            
            doc.close()
            
            logger.info(f"版面重建解析完成，已保存至: {text_output_path}")
            
            return {
                'success': True,
                'method': 'layout_reconstruction',
                'output_files': [text_output_path],
                'page_count': len(text_content)
            }
        except Exception as e:
            logger.error(f"版面重建解析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_ocr_route(self, pdf_path: str) -> Dict[str, Any]:
        """
        执行OCR解析路线
        
        针对无文本层或文本层质量极差的PDF文档，通过以下步骤处理：
        1. 将PDF页面转换为高质量图像
        2. 使用Tesseract OCR引擎识别图像中的文字
        3. 保存识别结果
        
        注意：OCR识别需要安装Tesseract OCR引擎，并确保中文语言包可用。
        """
        try:
            # 首先检查Tesseract是否可用
            if not self.tesseract_available:
                logger.warning("Tesseract OCR不可用，无法执行OCR识别")
                
                # 尝试一次重新检查
                self.tesseract_available = check_tesseract_installed()
                if not self.tesseract_available:
                    logger.warning("请按照以下步骤解决Tesseract OCR不可用问题:")
                    logger.warning("1. 安装Tesseract OCR引擎")
                    logger.warning("   - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装")
                    logger.warning("   - Linux: 运行 'sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim'")
                    logger.warning("   - macOS: 运行 'brew install tesseract tesseract-lang'")
                    logger.warning("2. 确保将Tesseract安装目录添加到系统环境变量PATH中")
                    logger.warning("3. 安装中文语言包以支持中文识别")
                    
                    return {
                        'success': False,
                        'error': 'Tesseract OCR不可用',
                        'details': '请安装Tesseract OCR并配置环境变量PATH'
                    }
            
            # 生成输出文件名
            base_name = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(base_name)[0]
            text_output_path = os.path.join(self.config['output_dir'], f"{name_without_ext}_ocr.txt")
            
            # 使用PyMuPDF和Tesseract OCR提取文本
            doc = fitz.open(pdf_path)
            text_content = []
            total_pages = len(doc)
            
            logger.info(f"开始OCR处理文档，共{total_pages}页")
            
            for page_num, page in enumerate(doc, 1):
                # 进度日志
                if page_num % 10 == 0 or page_num == total_pages:
                    logger.info(f"OCR处理进度: {page_num}/{total_pages}页 ({page_num/total_pages*100:.1f}%)")
                
                # 将页面转换为高质量图像（400dpi提高识别精度）
                pix = page.get_pixmap(dpi=400)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # 使用Tesseract OCR识别中文文本
                try:
                    # 再次检查Tesseract是否可用
                    if not check_tesseract_installed():
                        logger.warning("Tesseract OCR突然不可用，跳过OCR识别")
                        page_text = ""  # Tesseract不可用时使用空文本
                        self.tesseract_available = False
                    else:
                        # 配置Tesseract识别中文和英文，并设置绝对路径
                        # 使用原始字符串r前缀避免转义序列问题
                        CUSTOM_TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                        
                        # 配置tesseract_cmd使用绝对路径
                        if os.path.exists(CUSTOM_TESSERACT_PATH):
                            pytesseract.pytesseract.tesseract_cmd = CUSTOM_TESSERACT_PATH
                            logger.info(f"使用Tesseract绝对路径: {CUSTOM_TESSERACT_PATH}")
                        
                        # 高级图像预处理以提高中文识别质量
                        # 1. 转换为灰度图
                        img = img.convert('L')
                        
                        # 2. 应用高斯模糊去噪
                        # 降低图像噪声，有助于提高OCR准确性
                        from PIL import ImageFilter
                        img = img.filter(ImageFilter.GaussianBlur(radius=1))
                        
                        # 3. 应用自适应阈值处理增强对比度
                        # 尝试不同的阈值来找到最佳效果
                        img = img.point(lambda x: 0 if x < 160 else 255, '1')
                        
                        # 4. 锐化图像以增强文本边缘
                        img = img.filter(ImageFilter.SHARPEN)
                        
                        # 5. 优化OCR配置以提高中文识别质量
                        # 调整页面分割模式和引擎模式
                        
                        # 配置1: 适合中文文档的默认配置
                        config_options = [
                            r'--oem 3 --psm 6 --dpi 400 -l chi_sim',  # 单一文本块，仅中文
                            r'--oem 3 --psm 3 --dpi 400 -l chi_sim+eng',  # 自动页面分割，中英文
                            r'--oem 1 --psm 6 --dpi 400 -l chi_sim',  # LSTM引擎，仅中文
                            r'--oem 1 --psm 3 --dpi 400 -l chi_sim+eng',  # LSTM引擎，中英文
                            r'--oem 3 --psm 11 --dpi 400 -l chi_sim+eng',  # 稀疏文本，中英文
                            r'--oem 3 --psm 12 --dpi 400 -l chi_sim',  # 稀疏文本，仅中文
                        ]
                        
                        page_text = ""
                        
                        # 尝试多种配置直到获得有效结果
                        for i, config in enumerate(config_options):
                            try:
                                logger.info(f"尝试配置{i+1}: {config}")
                                temp_text = pytesseract.image_to_string(img, config=config)
                                # 检查识别结果是否有效（不是全乱码）
                                if temp_text.strip() and len(temp_text.strip()) > 10:
                                    # 简单检测中文：检查是否包含中文字符
                                    chinese_chars = sum(1 for char in temp_text if '\u4e00' <= char <= '\u9fff')
                                    total_chars = len(temp_text)
                                    if total_chars > 0:
                                            chinese_ratio = chinese_chars / total_chars * 100
                                            if chinese_ratio > 30 or any('\u4e00' <= char <= '\u9fff' for char in temp_text[:50]):
                                                page_text = temp_text
                                                logger.info(f"OCR处理第{page_num}页成功，使用配置{i+1}，中文占比: {chinese_ratio:.1f}%")
                                                break
                            except pytesseract.TesseractError as te:
                                logger.warning(f"Tesseract识别错误(第{page_num}页, 配置{i+1}): {te}")
                                if 'Failed loading language \'chi_sim\'' in str(te) or 'Could not initialize tesseract' in str(te):
                                    logger.warning(f"错误可能是由于中文语言包(chi_sim.traineddata)缺失导致的")
                                    logger.warning(f"请从以下地址下载中文语言包:")
                                    logger.warning(f"https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata")
                                    logger.warning(f"并将其放置到Tesseract安装目录的tessdata文件夹中，如: C:\\Program Files\\Tesseract-OCR\\tessdata")
                        
                        # 如果所有配置都失败，尝试调整预处理参数
                        if not page_text.strip():
                            logger.warning(f"第{page_num}页所有配置都失败，尝试调整预处理")
                            # 重新加载图像并尝试不同的预处理
                            img = Image.open(io.BytesIO(img_data))
                            
                            # 尝试不同的预处理方案
                            # 方案1: 自适应阈值 + 边缘增强
                            try:
                                img_processed = img.convert('L')
                                img_processed = img_processed.point(lambda x: 0 if x < 130 else 255, '1')
                                img_processed = img_processed.filter(ImageFilter.EDGE_ENHANCE_MORE)
                                
                                alt_config = r'--oem 3 --psm 6 --dpi 400 -l chi_sim'
                                page_text = pytesseract.image_to_string(img_processed, config=alt_config)
                                logger.info(f"已尝试边缘增强处理第{page_num}页")
                            except Exception as e:
                                logger.error(f"边缘增强预处理失败: {e}")
                            
                            # 如果仍失败，尝试方案2: 中值滤波去噪
                            if not page_text.strip():
                                try:
                                    img_processed = img.convert('L')
                                    img_processed = img_processed.filter(ImageFilter.MedianFilter(size=3))
                                    img_processed = img_processed.point(lambda x: 0 if x < 150 else 255, '1')
                                    
                                    alt_config = r'--oem 1 --psm 3 --dpi 400 -l chi_sim+eng'
                                    page_text = pytesseract.image_to_string(img_processed, config=alt_config)
                                    logger.info(f"已尝试中值滤波处理第{page_num}页")
                                except Exception as e:
                                    logger.error(f"中值滤波预处理失败: {e}")
                        
                        # 记录识别结果统计
                        if page_text:
                            chinese_chars = sum(1 for char in page_text if '\u4e00' <= char <= '\u9fff')
                            total_chars = len(page_text)
                            if total_chars > 0:
                                chinese_ratio = chinese_chars / total_chars * 100
                                logger.info(f"第{page_num}页中文识别率估计: {chinese_ratio:.1f}%")
                                
                                # 检测是否主要为英文/乱码结果
                                if chinese_ratio < 10:
                                    logger.warning(f"第{page_num}页中文识别率较低({chinese_ratio:.1f}%)，可能是语言包问题或图像质量问题")
                                    logger.warning(f"如果尚未安装中文语言包，请从以下地址下载:")
                                    logger.warning(f"https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata")
                                    logger.warning(f"并将其放置到Tesseract安装目录的tessdata文件夹中")
                        else:
                            logger.error(f"第{page_num}页OCR识别失败，未获取到任何文本内容")
                        
                        # 如果识别结果为空，尝试更激进的预处理
                        if not page_text.strip():
                            logger.warning(f"第{page_num}页识别结果为空，尝试增强对比度")
                            # 重新加载图像并应用不同的阈值处理
                            img = Image.open(io.BytesIO(img_data)).convert('L')
                            # 尝试不同的阈值（降低阈值以增强深色文本）
                            img = img.point(lambda x: 0 if x < 180 else 255, '1')
                            
                            # 使用更适合低质量图像的配置
                            aggressive_config = r'--oem 3 --psm 12 -l chi_sim+eng'
                            page_text = pytesseract.image_to_string(img, config=aggressive_config)
                            logger.info(f"已尝试增强对比度处理第{page_num}页")
                except Exception as ocr_err:
                    logger.warning(f"OCR处理第{page_num}页失败: {ocr_err}")
                    # 提供更友好的错误提示
                    if "not installed or it's not in your PATH" in str(ocr_err):
                        logger.warning("请安装Tesseract OCR并添加到系统PATH环境变量")
                        self.tesseract_available = False
                    page_text = ""  # 识别失败时使用空文本
                
                text_content.append(f"==== 第{page_num}页 ====\n{page_text}\n")
            
            # 保存提取的文本
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            
            doc.close()
            
            logger.info(f"OCR解析完成，已保存至: {text_output_path}")
            
            return {
                'success': True,
                'method': 'ocr',
                'output_files': [text_output_path],
                'page_count': len(text_content),
                'tesseract_available': check_tesseract_installed()
            }
        except ImportError:
            logger.error("pytesseract库未安装，请运行: pip install pytesseract")
            # 尝试自动安装pytesseract
            try:
                logger.info("正在尝试自动安装pytesseract库...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytesseract'], check=True)
                logger.info("pytesseract库安装成功")
                return {'success': False, 'error': 'pytesseract库已安装，请重新运行程序'}
            except Exception as e:
                logger.error(f"安装pytesseract库失败: {e}")
            return {'success': False, 'error': 'pytesseract库未安装'}
        except Exception as e:
            logger.error(f"OCR解析失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_route_log(self, route_report: Dict[str, Any]):
        """保存路由选择日志"""
        try:
            log_file = os.path.join(self.config['output_dir'], 'pdf_route_log.jsonl')
            with open(log_file, 'a', encoding='utf-8') as f:
                # 只保存关键信息到日志
                log_entry = {
                    'timestamp': route_report['timestamp'],
                    'file_name': route_report['file_info']['name'],
                    'file_size_mb': route_report['file_info']['size_mb'],
                    'selected_route': route_report['route_decision']['route'],
                    'confidence': route_report['route_decision']['confidence'],
                    'reasons': route_report['route_decision']['reason'],
                    'processing_time_seconds': route_report['processing_time_seconds']
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存路由日志失败: {e}")
    
    def _save_full_report(self, route_report: Dict[str, Any]):
        """保存完整的解析报告"""
        try:
            file_name = route_report['file_info']['name']
            report_file = os.path.join(self.config['output_dir'], 
                                     f"{os.path.splitext(file_name)[0]}_report.json")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(route_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"完整报告已保存至: {report_file}")
        except Exception as e:
            logger.error(f"保存完整报告失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def print_summary(self, route_report: Dict[str, Any]):
        """打印解析路线的摘要信息"""
        print("="*80)
        print(f"PDF解析路由选择结果")
        print("="*80)
        
        # 文件信息
        print(f"文件信息:")
        print(f"  文件名: {route_report['file_info']['name']}")
        print(f"  文件大小: {route_report['file_info']['size_mb']} MB")
        
        # 文本层分析
        print(f"\n文本层分析:")
        text_layer = route_report['text_layer_analysis']
        print(f"  是否有文本层: {'是' if text_layer['has_text_layer'] else '否'}")
        if text_layer['has_text_layer']:
            print(f"  文本密度: {text_layer['text_density']:.6f}")
            print(f"  有文本页面比例: {text_layer['text_pages_ratio']:.2f}")
            print(f"  文本质量评估: {text_layer['estimated_text_quality']}")
        
        # 版面复杂度分析
        print(f"\n版面复杂度分析:")
        layout = route_report['layout_analysis']
        print(f"  复杂度分数: {layout['complexity_score']:.2f}")
        print(f"  是否复杂版面: {'是' if layout['is_complex'] else '否'}")
        print(f"  版面类型评估: {layout['estimated_layout_type']}")
        
        # 路由决策
        print(f"\n路由决策:")
        decision = route_report['route_decision']
        print(f"  选择的路由: {decision['route']}")
        print(f"  决策理由: {', '.join(decision['reason'])}")
        print(f"  置信度: {decision['confidence']:.2f}")
        
        # 执行信息
        if 'execution' in route_report:
            execution = route_report['execution']
            print(f"\n执行信息:")
            print(f"  执行的路由: {execution['executed_route']}")
            print(f"  执行时间: {execution['execution_time_seconds']:.3f} 秒")
            print(f"  执行结果: {'成功' if execution['result'].get('success', False) else '失败'}")
            
            if execution['result'].get('output_files'):
                print(f"  输出文件:")
                for file_path in execution['result']['output_files']:
                    print(f"    - {file_path}")
        
        # 总处理时间
        total_time = route_report['processing_time_seconds']
        if 'execution' in route_report:
            total_time += route_report['execution']['execution_time_seconds']
        
        print(f"\n总处理时间: {total_time:.3f} 秒")
        print("="*80)

def main():
    """
    主函数，演示PDF路由选择器的完整工作流程
    
    工作流程：
    1. 解析命令行参数，获取PDF文件路径
    2. 初始化PDF路由选择器
    3. 分析PDF特征并选择最佳解析路线
    4. 执行选定的解析路线
    5. 输出解析结果摘要和统计信息
    """
    # 检查命令行参数
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # 默认处理指定的PDF文件
        pdf_path = os.path.join("data", "大脑中动脉狭窄与闭塞致脑梗死的影像特点及发病机制的研究.pdf")
        
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 '{pdf_path}'")
        print("请确保PDF文件位于正确的路径，或通过命令行参数指定文件路径")
        sys.exit(1)
    
    # 初始化路由选择器
    selector = PDFRouteSelector()
    
    print(f"开始处理文件: {pdf_path}")
    
    # 选择解析路线
    print("\n1. 正在分析PDF特征并选择最佳解析路线...")
    route_report = selector.select_route(pdf_path)
    
    # 执行选择的路线
    print(f"\n2. 正在执行解析路线: {route_report['route_decision']['route']}")
    full_report = selector.execute_route(pdf_path, route_report)
    
    # 打印摘要
    print("\n3. 解析完成，生成结果摘要:")
    selector.print_summary(full_report)
    
    # 显示统计信息
    stats = selector.get_stats()
    print(f"\n4. 全局统计信息:")
    print(f"   - 总处理文件数: {stats['total_processed']}")
    print(f"   - PDFPlumber路线: {stats['pdfplumber_routes']}")
    print(f"   - 版面重建路线: {stats['layout_routes']}")
    print(f"   - OCR路线: {stats['ocr_routes']}")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()