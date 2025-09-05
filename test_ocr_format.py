#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试PaddleOCR的版本和predict方法返回格式
"""

import sys
import cv2
import numpy as np
from PIL import Image
import os

# 尝试导入PaddleOCR
try:
    import paddleocr
    print(f"✅ PaddleOCR库已成功导入")
    print(f"PaddleOCR版本: {paddleocr.__version__}")
except ImportError as e:
    print(f"❌ 无法导入PaddleOCR: {e}")
    print("这可能是由于Python版本与PyTorch不兼容导致的")
    print(f"当前Python版本: {sys.version}")
    sys.exit(1)

# 创建一个简单的测试图像
def create_test_image():
    # 创建一个白色背景的图像
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加一些中文文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, '测试文本', (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'PaddleOCR', (50, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return image

# 保存测试图像
def save_test_image(image, path='test_image.png'):
    cv2.imwrite(path, image)
    print(f"测试图像已保存到: {path}")
    return path

# 初始化OCR并测试
def test_ocr():
    try:
        # 初始化OCR，使用中英文模型
        print("\n正在初始化PaddleOCR...")
        # 注意：使用简单配置，避免可能的参数问题
        ocr = paddleocr.PaddleOCR(lang='ch', use_angle_cls=False)
        print("✅ PaddleOCR初始化成功")
        
        # 创建并保存测试图像
        test_image = create_test_image()
        image_path = save_test_image(test_image)
        
        # 测试predict方法
        print("\n正在调用predict方法...")
        result = ocr.predict(image_path)
        
        # 打印结果信息
        print(f"\n=== OCR结果信息 ===")
        print(f"结果类型: {type(result)}")
        print(f"结果长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        print(f"结果内容: {result}")
        
        # 如果结果是列表，打印每个元素的详细信息
        if isinstance(result, list):
            print(f"\n=== 列表元素详细信息 ===")
            for i, item in enumerate(result):
                print(f"元素 {i+1}:")
                print(f"  类型: {type(item)}")
                print(f"  长度: {len(item) if hasattr(item, '__len__') else 'N/A'}")
                print(f"  内容: {item}")
                
                # 如果元素是列表或元组，继续展开
                if isinstance(item, (list, tuple)):
                    for j, sub_item in enumerate(item):
                        print(f"    子元素 {j+1}:")
                        print(f"      类型: {type(sub_item)}")
                        print(f"      长度: {len(sub_item) if hasattr(sub_item, '__len__') else 'N/A'}")
                        print(f"      内容: {sub_item}")
        
        # 测试直接传入图像数组
        print("\n正在测试直接传入图像数组...")
        result_array = ocr.predict(test_image)
        print(f"直接传入数组的结果类型: {type(result_array)}")
        print(f"直接传入数组的结果: {result_array}")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("===== PaddleOCR格式测试工具 =====")
    test_ocr()
    print("\n===== 测试完成 =====")