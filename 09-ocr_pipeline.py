#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR与版面对齐工具
功能：对PDF或图像文件进行OCR识别，并进行版面对齐
流程：前处理（去噪/旋转/二值化）→ PaddleOCR → 段落合并 → 版面对齐（图文框对齐）
输出：段落文本 + 版面元素坐标

使用说明：
1. 安装依赖：pip install paddlepaddle paddleocr opencv-python pillow numpy
2. 运行示例：python 09-ocr_pipeline.py --input_file data/your_file.pdf --output_dir ./ocr_output

关键点：
- 字符准确率评估：计算OCR识别结果的置信度
- 倾斜角校正：自动检测并校正图像倾斜
- 版面元素命名：为不同类型的版面元素（标题、正文、图片、表格等）分配唯一标识符
"""

# 导入基础库
import os
import json
import logging
import argparse
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional

# 尝试导入主要依赖，如果失败则提供友好的错误信息
packages_available = {
    'cv2': False,
    'numpy': False,
    'PIL': False,
    'paddleocr': False,
    'pdfplumber': False
}

def check_dependencies():
    """检查并尝试安装必要的依赖包"""
    try:
        import cv2
        packages_available['cv2'] = True
    except ImportError:
        pass
        
    try:
        import numpy
        packages_available['numpy'] = True
    except ImportError:
        pass
        
    try:
        from PIL import Image
        packages_available['PIL'] = True
    except ImportError:
        pass
        
    try:
        import paddleocr
        packages_available['paddleocr'] = True
    except ImportError as e:
        print(f"警告：无法导入paddleocr: {e}")
        print("这可能是由于Python版本与PyTorch不兼容导致的")
        print("建议使用Python 3.8-3.11版本，这些版本与PyTorch有更好的兼容性")
        
    try:
        import pdfplumber
        packages_available['pdfplumber'] = True
    except ImportError:
        pass

    # 安装缺失的依赖
    required_packages = []
    if not packages_available['cv2']:
        required_packages.append('opencv-python>=4.8.0')
    if not packages_available['numpy']:
        required_packages.append('numpy>=1.24.0')
    if not packages_available['PIL']:
        required_packages.append('pillow>=9.5.0')
    if not packages_available['pdfplumber']:
        required_packages.append('pdfplumber>=0.9.0')
    
    # 注意：由于PaddleOCR依赖问题，我们不自动安装它
    if required_packages:
        print(f"检测到缺少以下基础依赖包：{required_packages}")
        print("正在自动安装基础依赖包，请稍候...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *required_packages])
            print("基础依赖包安装成功！")
            # 重新导入
            check_dependencies()
        except subprocess.CalledProcessError:
            print("警告：自动安装基础依赖包失败，请手动运行以下命令安装：")
            print(f"pip install {' '.join(required_packages)}")
            print("程序可能无法正常运行。")

# 检查依赖
check_dependencies()

# 尝试导入其余必要的库
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pdfplumber
    # 只在paddleocr可用时导入
    if packages_available['paddleocr']:
        import paddleocr
    else:
        paddleocr = None
except ImportError as e:
    print(f"导入必要的库失败: {e}")

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别以查看详细的调试信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_pipeline_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OCRPipeline')

class ImagePreprocessor:
    """图像前处理器，负责去噪、旋转校正和二值化等操作"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化图像前处理器"""
        self.default_config = {
            'denoise_strength': 1.0,
            'blur_kernel_size': (5, 5),
            'binary_threshold': 127,
            'max_rotation_angle': 15.0
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """加载图像文件或PDF文件的第一页为图像"""
        try:
            if file_path.lower().endswith('.pdf'):
                # 处理PDF文件，仅提取第一页
                print(f"📄  检测到PDF文件: {file_path}")
                print("🔍  使用pdfplumber加载PDF并提取第一页...")
                with pdfplumber.open(file_path) as pdf:
                    if not pdf.pages:
                        logger.error(f"PDF文件为空: {file_path}")
                        print("❌  PDF文件为空，无法提取内容")
                        return None
                    print(f"📑  PDF包含{len(pdf.pages)}页，将处理第一页")
                    first_page = pdf.pages[0]
                    print("📸  将PDF页面转换为图像，分辨率设置为300DPI...")
                    img = np.array(first_page.to_image(resolution=300).original)
                    print("✅  PDF页面转换成功")
                    return img
            else:
                # 处理图像文件
                print(f"🖼️  检测到图像文件: {file_path}")
                img = cv2.imread(file_path)
                if img is None:
                    logger.error(f"无法加载图像文件: {file_path}")
                    print(f"❌  无法加载图像文件: {file_path}")
                    return None
                print(f"✅  图像加载成功，尺寸: {img.shape[1]}x{img.shape[0]}")
                return img
        except Exception as e:
            logger.error(f"加载文件时出错: {e}")
            print(f"⚠️  加载文件时出错: {type(e).__name__}: {e}")
            return None
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """图像去噪处理"""
        # 使用高斯模糊去噪
        blurred = cv2.GaussianBlur(image, self.config['blur_kernel_size'], 0)
        return blurred
    
    def detect_rotation_angle(self, image: np.ndarray) -> float:
        """检测图像倾斜角度"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 二值化
        _, binary = cv2.threshold(gray, self.config['binary_threshold'], 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 计算最小外接矩形，找出旋转角度
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                # 调整角度范围
                if angle < -45:
                    angle += 90
                angles.append(angle)
        
        # 返回平均角度
        return np.mean(angles) if angles else 0.0
    
    def correct_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """校正图像倾斜"""
        if abs(angle) < 0.5:  # 角度太小，不需要校正
            return image
        
        # 限制最大旋转角度
        angle = max(-self.config['max_rotation_angle'], min(self.config['max_rotation_angle'], angle))
        
        # 获取图像中心点
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 执行旋转，保持图像尺寸不变
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """图像二值化处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        return binary
    
    def preprocess(self, file_path: str) -> Tuple[Optional[np.ndarray], float]:
        """完整的预处理流程"""
        # 加载图像
        image = self.load_image(file_path)
        if image is None:
            return None, 0.0
        
        # 去噪
        denoised = self.denoise(image)
        
        # 检测倾斜角度
        rotation_angle = self.detect_rotation_angle(denoised)
        
        # 校正倾斜
        rotated = self.correct_rotation(denoised, rotation_angle)
        
        # 二值化
        binary = self.binarize(rotated)
        
        return binary, rotation_angle

class OCRProcessor:
    """OCR处理器，使用PaddleOCR进行文本识别"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化OCR处理器"""
        self.default_config = {
            'lang': 'ch',  # 语言，'ch'表示中文
            'text_det_thresh': 0.3,  # 文本检测阈值（新版参数名）
            'text_recognition_batch_size': 6  # 识别批次大小（新版参数名）
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化PaddleOCR
        self.ocr = None
        if paddleocr is not None:
            try:
                # 使用新版PaddleOCR的参数名
                ocr_args = {
                    'lang': self.config['lang']
                }
                
                # 检测阈值参数
                if 'text_det_thresh' in self.config:
                    ocr_args['text_det_thresh'] = self.config['text_det_thresh']
                
                # 识别批次大小参数
                if 'text_recognition_batch_size' in self.config:
                    ocr_args['text_recognition_batch_size'] = self.config['text_recognition_batch_size']
                
                # 注意：新版PaddleOCR已经不需要显式设置use_gpu参数
                
                self.ocr = paddleocr.PaddleOCR(**ocr_args)
                logger.info("PaddleOCR初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                self.ocr = None
        else:
            logger.warning("PaddleOCR不可用，无法进行OCR识别")
    
    def recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用PaddleOCR进行文本识别，完全适配新版PaddleOCR的返回格式
        
        Args:
            image: 预处理后的图像数据（numpy数组格式）
            
        Returns:
            识别到的文本信息列表，每个元素包含文本内容、置信度和位置信息
        """
        if self.ocr is None:
            logger.error("OCR引擎未初始化，无法进行识别")
            return []
        
        recognized_texts = []
        
        try:
            # 步骤1: 检查图像格式并确保是RGB格式
            print("🔍  开始检查图像格式...")
            if len(image.shape) == 2:
                # 二值化图像或灰度图像，转换为RGB
                print("📷  检测到二值化/灰度图像，转换为RGB格式...")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA图像，转换为RGB
                print("📷  检测到RGBA图像，转换为RGB格式...")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                print("✅  图像已是RGB格式，无需转换")
            print(f"📐  处理后图像尺寸: {image.shape[1]}x{image.shape[0]}")
            
            # 步骤2: 执行OCR识别，使用最新的predict方法
            print("🤖  执行OCR识别，调用predict方法...")
            result = self.ocr.predict(image)
            print("✅  OCR识别执行完成")
            
            # 步骤3: 基础类型检查
            if result is None:
                logger.warning("OCR返回结果为空")
                print("⚠️  OCR返回结果为空")
                return recognized_texts
            
            print(f"===== OCR调试 - 结果类型: {type(result)} ====")
            
            # 步骤4: 适配新版PaddleOCR的返回格式
            # 从测试结果可以看出，新版PaddleOCR返回的是一个包含字典的列表
            print("🔍  开始解析OCR结果...")
            if isinstance(result, list) and len(result) > 0:
                # 获取第一个元素（通常是主要的OCR结果）
                ocr_result = result[0]
                
                # 检查是否包含识别的文本和置信度
                if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result and 'rec_scores' in ocr_result and 'dt_polys' in ocr_result:
                    print(f"✅  检测到新版PaddleOCR格式，包含rec_texts, rec_scores和dt_polys")
                    
                    # 获取识别的文本、置信度和边界框
                    texts = ocr_result['rec_texts']
                    scores = ocr_result['rec_scores']
                    bboxes = ocr_result['dt_polys']
                    
                    # 确保三个列表长度一致
                    min_len = min(len(texts), len(scores), len(bboxes))
                    print(f"📊  识别到{min_len}个文本区域")
                    
                    # 处理每个识别的文本区域
                    print("🔧  开始处理识别到的文本区域...")
                    for i in range(min_len):
                        try:
                            text = texts[i]
                            confidence = scores[i]
                            bbox = bboxes[i]
                            
                            # 检查并处理边界框坐标
                            if isinstance(bbox, np.ndarray):
                                # 将numpy数组转换为列表
                                bbox_list = bbox.tolist()
                            elif isinstance(bbox, (list, tuple)):
                                bbox_list = list(bbox)
                            else:
                                print(f"⚠️  跳过无效的边界框类型: {type(bbox)}")
                                continue
                            
                            # 计算边界框坐标
                            try:
                                x_coords = []
                                y_coords = []
                                
                                # 处理边界框点（可能是numpy数组）
                                for point in bbox_list:
                                    if isinstance(point, np.ndarray):
                                        # 对于numpy数组格式的点
                                        if len(point) >= 2:
                                            x_coords.append(float(point[0]))
                                            y_coords.append(float(point[1]))
                                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                                        # 对于列表或元组格式的点
                                        x_coords.append(float(point[0]))
                                        y_coords.append(float(point[1]))
                                
                                if x_coords and y_coords:
                                    # 计算中心点
                                    center_x = sum(x_coords) / len(x_coords)
                                    center_y = sum(y_coords) / len(y_coords)
                                    
                                    # 添加识别的文本信息
                                    recognized_texts.append({
                                        'bbox': {
                                            'x0': min(x_coords),
                                            'y0': min(y_coords),
                                            'x1': max(x_coords),
                                            'y1': max(y_coords),
                                            'points': bbox_list
                                        },
                                        'center': {
                                            'x': center_x,
                                            'y': center_y
                                        },
                                        'text': str(text) if text else "",
                                        'confidence': float(confidence) if confidence else 0.0
                                    })
                            except Exception as coord_err:
                                print(f"⚠️  解析边界框坐标出错: {coord_err}")
                                continue
                        except Exception as item_err:
                            print(f"⚠️  解析单个OCR项目出错: {item_err}")
                            continue
                else:
                    print("❌  OCR结果不包含预期的键，尝试其他解析方式")
                    # 打印结果的键，帮助调试
                    if isinstance(ocr_result, dict):
                        print(f"   可用键: {ocr_result.keys()}")
            else:
                print(f"❌  OCR返回的不是预期的列表格式，结果: {str(result)[:200]}")
            
            print(f"✅  文本块解析完成，成功解析{len(recognized_texts)}个文本块")
            
            # 显示前几个识别结果作为预览
            if recognized_texts:
                preview_count = min(3, len(recognized_texts))
                print(f"🔍  前{preview_count}个识别结果预览:")
                for i in range(preview_count):
                    text_info = recognized_texts[i]
                    print(f"   [{i+1}] '{text_info['text'][:30]}{'...' if len(text_info['text']) > 30 else ''}' (置信度: {text_info['confidence']:.4f})")
            
            # 如果没有解析到任何文本，记录详细的调试信息
            if not recognized_texts:
                print(f"❌  未解析到任何文本，完整OCR结果结构:")
                print(f"   类型: {type(result)}")
                if isinstance(result, list):
                    print(f"   列表长度: {len(result)}")
                    for i, item in enumerate(result):
                        print(f"   元素{i}: 类型={type(item)}")
                        if isinstance(item, dict):
                            print(f"     键: {item.keys()}")
            
            return recognized_texts
        
        except Exception as e:
            print(f"❌  OCR识别过程出错: {type(e).__name__}: {e}")
            print("💡  错误排查建议:")
            print("   1. 检查PaddleOCR版本是否兼容")
            print("   2. 确认图像格式是否正确")
            print("   3. 检查系统资源是否充足")
            logger.error(f"OCR识别出错: {e}")
            
            # 即使出错，也要返回已成功解析的文本块（如果有）
            return recognized_texts
    
    def evaluate_accuracy(self, recognized_texts: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估OCR识别准确率"""
        if not recognized_texts:
            return {
                'average_confidence': 0.0,
                'high_confidence_rate': 0.0,
                'low_confidence_count': 0
            }
        
        # 计算平均置信度
        total_confidence = sum(text['confidence'] for text in recognized_texts)
        average_confidence = total_confidence / len(recognized_texts)
        
        # 计算高置信度（>0.9）文本比例
        high_confidence_count = sum(1 for text in recognized_texts if text['confidence'] > 0.9)
        high_confidence_rate = high_confidence_count / len(recognized_texts)
        
        # 统计低置信度（<0.5）文本数量
        low_confidence_count = sum(1 for text in recognized_texts if text['confidence'] < 0.5)
        
        return {
            'average_confidence': average_confidence,
            'high_confidence_rate': high_confidence_rate,
            'low_confidence_count': low_confidence_count
        }

class LayoutAligner:
    """版面对齐器，负责段落合并和版面对齐"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化版面对齐器
        
        Args:
            config: 版面对齐配置参数
        """
        self.default_config = {
            'vertical_merge_threshold': 10.0,  # 垂直方向合并阈值（像素）
            'horizontal_merge_threshold': 20.0,  # 水平方向合并阈值（像素）
            'paragraph_min_words': 5,  # 段落最小字数
            'element_types': ['text', 'title', 'image', 'table', 'figure']  # 支持的版面元素类型
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
    
    def merge_paragraphs(self, recognized_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并相邻的文本块为段落
        
        Args:
            recognized_texts: 识别到的文本块列表
            
        Returns:
            合并后的段落列表
        """
        if not recognized_texts:
            print("❌  没有可合并的文本块")
            return []
        
        # 按照Y坐标（从上到下）和X坐标（从左到右）排序文本块
        print(f"🔍  开始合并文本块，共{len(recognized_texts)}个文本块")
        print("📝  按页面位置排序文本块...")
        sorted_texts = sorted(recognized_texts, key=lambda x: (x['center']['y'], x['center']['x']))
        
        paragraphs = []
        current_paragraph = sorted_texts[0].copy()
        current_paragraph['texts'] = [current_paragraph.pop('text')]
        
        merge_count = 0  # 合并计数
        
        for text_block in sorted_texts[1:]:
            # 计算垂直距离（判断是否在同一垂直区域）
            vertical_distance = text_block['center']['y'] - current_paragraph['center']['y']
            
            # 计算水平重叠（判断是否在同一水平区域）
            horizontal_overlap = max(0, min(text_block['bbox']['x1'], current_paragraph['bbox']['x1']) - 
                                     max(text_block['bbox']['x0'], current_paragraph['bbox']['x0']))
            
            # 如果垂直距离小于阈值且有水平重叠，则合并为同一段落
            if (vertical_distance < self.config['vertical_merge_threshold'] and 
                horizontal_overlap > self.config['horizontal_merge_threshold']):
                # 更新段落边界框
                current_paragraph['bbox']['x0'] = min(current_paragraph['bbox']['x0'], text_block['bbox']['x0'])
                current_paragraph['bbox']['y0'] = min(current_paragraph['bbox']['y0'], text_block['bbox']['y0'])
                current_paragraph['bbox']['x1'] = max(current_paragraph['bbox']['x1'], text_block['bbox']['x1'])
                current_paragraph['bbox']['y1'] = max(current_paragraph['bbox']['y1'], text_block['bbox']['y1'])
                
                # 更新中心点
                current_paragraph['center']['x'] = (current_paragraph['center']['x'] + text_block['center']['x']) / 2
                current_paragraph['center']['y'] = (current_paragraph['center']['y'] + text_block['center']['y']) / 2
                
                # 合并文本
                current_paragraph['texts'].append(text_block['text'])
                
                # 更新置信度（取平均值）
                current_paragraph['confidence'] = (current_paragraph['confidence'] + text_block['confidence']) / 2
                
                merge_count += 1
            else:
                # 保存当前段落并开始新段落
                current_paragraph['text'] = ' '.join(current_paragraph['texts'])
                paragraphs.append(current_paragraph)
                
                new_paragraph = text_block.copy()
                new_paragraph['texts'] = [new_paragraph.pop('text')]
                current_paragraph = new_paragraph
        
        # 添加最后一个段落
        current_paragraph['text'] = ' '.join(current_paragraph['texts'])
        paragraphs.append(current_paragraph)
        
        # 过滤字数过少的段落
        print(f"📋  合并完成，初步得到{len(paragraphs)}个段落")
        print(f"🔍  过滤字数少于{self.config['paragraph_min_words']}的段落...")
        filtered_paragraphs = [p for p in paragraphs if len(p['text']) >= self.config['paragraph_min_words']]
        
        # 为段落添加唯一标识符
        for i, paragraph in enumerate(filtered_paragraphs):
            paragraph['paragraph_id'] = f"para_{i+1}"
        
        print(f"✅  段落合并完成，最终得到{len(filtered_paragraphs)}个有效段落")
        print(f"🔍  共合并了{merge_count}次文本块")
        
        # 显示前几个段落作为预览
        if filtered_paragraphs:
            preview_count = min(2, len(filtered_paragraphs))
            print(f"📄  前{preview_count}个段落预览:")
            for i in range(preview_count):
                para = filtered_paragraphs[i]
                preview_text = para['text'][:50] + '...' if len(para['text']) > 50 else para['text']
                print(f"   [{para['paragraph_id']}] {preview_text} (字数: {len(para['text'])}, 置信度: {para['confidence']:.4f})")
        
        return filtered_paragraphs
    
    def identify_layout_elements(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别版面元素类型（如标题、正文等）
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            添加了元素类型的段落列表
        """
        print("🔍  开始识别版面元素类型...")
        
        # 统计各类元素的数量
        element_counts = {'title': 0, 'text': 0}
        
        # 这是一个简化的实现，基于文本特征初步判断元素类型
        for i, paragraph in enumerate(paragraphs):
            # 根据文本特征初步判断元素类型
            text_length = len(paragraph['text'])
            text_lines = paragraph['text'].count('\n') + 1
            
            # 简单规则：文本行少但每行文字少，可能是标题
            if text_lines <= 3 and text_length < 100 and paragraph['confidence'] > 0.8:
                element_type = 'title'
            # 文本行数多，可能是正文
            elif text_lines > 2 and text_length > 50:
                element_type = 'text'
            # 其他情况作为普通文本
            else:
                element_type = 'text'
            
            # 更新计数
            element_counts[element_type] += 1
            
            # 为元素添加唯一标识符
            paragraph['element_id'] = f"{element_type}_{i+1}"
            paragraph['element_type'] = element_type
        
        print(f"✅  版面元素识别完成，共识别{len(paragraphs)}个元素")
        print(f"📊  元素类型统计: 标题 {element_counts['title']}个, 正文 {element_counts['text']}个")
        
        return paragraphs
    
    def align_layout(self, paragraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行版面对齐，生成结构化的版面信息
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            结构化的版面信息字典
        """
        if not paragraphs:
            print("❌  没有段落可进行版面对齐")
            return {}
            
        # 识别版面元素类型
        print("📐  开始执行版面对齐...")
        layout_elements = self.identify_layout_elements(paragraphs)
        
        # 计算整体边界框
        if layout_elements:
            x0 = min(e['bbox']['x0'] for e in layout_elements)
            y0 = min(e['bbox']['y0'] for e in layout_elements)
            x1 = max(e['bbox']['x1'] for e in layout_elements)
            y1 = max(e['bbox']['y1'] for e in layout_elements)
        else:
            x0, y0, x1, y1 = 0, 0, 0, 0
        
        print(f"📏  计算整体版面边界: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        
        # 生成版面信息
        layout_info = {
            'layout_bbox': {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
            'element_count': len(layout_elements),
            'elements': layout_elements,
            'element_types_count': {
                'title': sum(1 for e in layout_elements if e['element_type'] == 'title'),
                'text': sum(1 for e in layout_elements if e['element_type'] == 'text')
            }
        }
        
        print(f"✅  版面对齐完成")
        
        return layout_info

class OCRPipeline:
    """OCR完整处理流水线"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化OCR处理流水线"""
        self.default_config = {
            'output_dir': './ocr_output',
            'image_preprocess': {},
            'ocr_recognize': {},
            'layout_align': {}
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 确保输出目录存在
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 初始化各个组件
        self.preprocessor = ImagePreprocessor(self.config['image_preprocess'])
        
        # 检查paddleocr是否可用
        if paddleocr is None:
            logger.warning("PaddleOCR库不可用，OCR功能将无法使用")
            self.ocr_processor = None
        else:
            self.ocr_processor = OCRProcessor(self.config['ocr_recognize'])
        
        self.layout_aligner = LayoutAligner(self.config['layout_align'])
        
        logger.info(f"OCR处理流水线初始化完成，配置: {self.config}")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理单个文件的完整流程"""
        logger.info(f"开始处理文件: {file_path}")
        
        # 1. 图像预处理
        logger.info("执行图像预处理...")
        print("🔧  开始图像预处理...")
        preprocessed_image, rotation_angle = self.preprocessor.preprocess(file_path)
        if preprocessed_image is None:
            logger.error("图像预处理失败，无法继续处理")
            print("❌  图像预处理失败，无法继续处理")
            return {}
        
        logger.info(f"预处理完成，校正倾斜角度: {rotation_angle:.2f}度")
        print(f"✅  图像预处理完成，校正倾斜角度: {rotation_angle:.2f}度")
        print(f"🔍  预处理后图像尺寸: {preprocessed_image.shape[1]}x{preprocessed_image.shape[0]}")
        
        # 2. 检查OCR处理器是否可用
        if self.ocr_processor is None or self.ocr_processor.ocr is None:
            logger.error("OCR处理器不可用，无法进行文本识别")
            print("⚠️  OCR功能无法使用，这可能是由于以下原因：")
            print("1. PaddleOCR库未正确安装")
            print("2. Python版本(3.13)与PyTorch不兼容")
            print("3. 缺少必要的系统依赖")
            print("建议：")
            print("- 使用Python 3.8-3.11版本重新运行")
            print("- 手动安装PaddleOCR及其依赖")
            print("- 检查系统是否安装了所有必要的CUDA组件（如使用GPU）")
            return {}
        
        # 3. OCR识别
        logger.info("执行OCR识别...")
        print("🔍  开始OCR文本识别...")
        recognized_texts = self.ocr_processor.recognize(preprocessed_image)
        if not recognized_texts:
            logger.warning("未识别到任何文本")
            print("❌  未识别到任何文本")
            return {}
        
        logger.info(f"OCR识别完成，识别到{len(recognized_texts)}个文本块")
        print(f"✅  OCR识别完成，识别到{len(recognized_texts)}个文本块")
        
        # 4. 评估识别准确率
        print("📊  评估OCR识别准确率...")
        accuracy_metrics = self.ocr_processor.evaluate_accuracy(recognized_texts)
        logger.info(f"OCR识别准确率评估: 平均置信度={accuracy_metrics['average_confidence']:.4f}, \
                    高置信度比例={accuracy_metrics['high_confidence_rate']:.4f}, \
                    低置信度数量={accuracy_metrics['low_confidence_count']}")
        print(f"✅  准确率评估完成，平均置信度: {accuracy_metrics['average_confidence']:.4f}")
        
        # 5. 段落合并
        logger.info("执行段落合并...")
        print("📝  执行段落合并...")
        paragraphs = self.layout_aligner.merge_paragraphs(recognized_texts)
        logger.info(f"段落合并完成，合并为{len(paragraphs)}个段落")
        print(f"✅  段落合并完成，合并为{len(paragraphs)}个段落")
        
        # 6. 版面对齐
        logger.info("执行版面对齐...")
        print("📐  执行版面对齐...")
        layout_info = self.layout_aligner.align_layout(paragraphs)
        print(f"✅  版面对齐完成")
        
        # 7. 生成完整结果
        print("📋  生成完整处理结果...")
        try:
            import pandas as pd
            timestamp = pd.Timestamp.now().isoformat()
        except ImportError:
            # 如果没有pandas，使用基本的时间戳
            import datetime
            timestamp = datetime.datetime.now().isoformat()
        
        result = {
            'file_path': file_path,
            'preprocessing_info': {
                'rotation_angle': rotation_angle
            },
            'ocr_metrics': accuracy_metrics,
            'layout_info': layout_info,
            'recognized_texts': recognized_texts,
            'timestamp': timestamp
        }
        
        print("✅  完整结果生成成功")
        return result
    
    def save_results(self, result: Dict[str, Any], output_dir: str = None) -> Dict[str, str]:
        """保存处理结果到文件"""
        if not result:
            logger.warning("没有结果可以保存")
            return {}
        
        # 使用指定的输出目录或默认目录
        save_dir = output_dir or self.config['output_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取文件名（不包含扩展名）
        file_name = os.path.splitext(os.path.basename(result['file_path']))[0]
        
        # 保存完整结果为JSON
        json_path = os.path.join(save_dir, f"{file_name}_ocr_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本结果
        text_path = os.path.join(save_dir, f"{file_name}_ocr_text.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            if 'layout_info' in result and 'elements' in result['layout_info']:
                for element in result['layout_info']['elements']:
                    if 'text' in element:
                        f.write(element['text'] + '\n\n')
        
        # 保存版面元素信息
        layout_path = os.path.join(save_dir, f"{file_name}_layout_info.json")
        if 'layout_info' in result:
            with open(layout_path, 'w', encoding='utf-8') as f:
                json.dump(result['layout_info'], f, ensure_ascii=False, indent=2)
        
        return {
            'json_result': json_path,
            'text_result': text_path,
            'layout_info': layout_path
        }

def main():
    """主函数 - OCR处理流水线入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OCR与版面对齐工具')
    parser.add_argument('--input_file', type=str, 
                        default='data\百年IBM的24个瞬间：从制表机到超级计算机.pdf',
                        help='输入文件路径，支持PDF和图像文件')
    parser.add_argument('--output_dir', type=str, 
                        default='./ocr_output',
                        help='输出目录')
    # 注意：新版PaddleOCR已自动处理GPU支持，无需显式设置
    args = parser.parse_args()
    
    # 显示当前任务信息
    print("=" * 60)
    print(f"📝  待处理文件: {args.input_file}")
    print(f"📂  输出目录: {args.output_dir}")
    print(f"⚙️  PaddleOCR版本: 新版(自动GPU支持)")
    print("=" * 60)
    
    # 初始化OCR处理流水线
    config = {
        'output_dir': args.output_dir
    }
    print("🔄  初始化OCR处理流水线...")
    pipeline = OCRPipeline(config)
    print("✅  流水线初始化完成")
    
    # 处理文件
    print("\n===== 开始OCR处理任务 =====")
    print("📋  处理流程: 图像预处理 → 文本识别 → 准确率评估 → 段落合并 → 版面对齐 → 结果保存")
    result = pipeline.process_file(args.input_file)
    
    if not result:
        print("⚠️  处理失败，未生成任何结果")
        return
    
    # 保存结果
    print("💾  正在保存处理结果...")
    saved_paths = pipeline.save_results(result)
    
    # 显示详细结果信息
    print(f"\n🎉  OCR处理任务已完成！")
    print(f"📊  处理统计信息:")
    print(f"  ├── 识别文本块数量: {len(result.get('recognized_texts', []))}")
    if 'layout_info' in result:
        print(f"  ├── 合并段落数量: {len(result['layout_info'].get('elements', []))}")
        print(f"  │   ├── 标题数量: {result['layout_info'].get('element_types_count', {}).get('title', 0)}")
        print(f"  │   └── 正文数量: {result['layout_info'].get('element_types_count', {}).get('text', 0)}")
    if 'ocr_metrics' in result:
        print(f"  └── 平均置信度: {result['ocr_metrics']['average_confidence']:.4f}")
    print(f"\n📁  输出文件:")
    for name, path in saved_paths.items():
        print(f"  └── {name}: {path}")
    print("=" * 60)
    print("✅  任务完成")

if __name__ == '__main__':
    main()