#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试Gradio导入问题的脚本
"""

import sys
import time
import os
import traceback
import logging

# 配置详细日志
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='gradio_import_debug.log',
                    filemode='w',
                    encoding='utf-8')

def log(message):
    """打印日志到控制台和文件"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    logging.info(message)

def debug_log(message):
    """打印调试信息到文件"""
    logging.debug(message)

# 主测试函数
def main():
    log("开始测试Gradio导入问题")
    log(f"Python版本: {sys.version}")
    log(f"当前目录: {os.getcwd()}")
    log(f"sys.path: {sys.path}")
    
    # 检查是否安装了Gradio
    try:
        import importlib.metadata
        gradio_version = importlib.metadata.version('gradio')
        log(f"已安装的Gradio版本: {gradio_version}")
    except importlib.metadata.PackageNotFoundError:
        log("未找到Gradio包")
    except Exception as e:
        log(f"检查Gradio版本时出错: {type(e).__name__}: {e}")
        debug_log(traceback.format_exc())
    
    # 尝试导入Gradio，逐模块导入以找出问题所在
    modules_to_test = [
        'gradio',
        'gradio.themes',
        'gradio.themes.Soft',
        'gradio.Blocks',
        'gradio.Markdown',
        'gradio.Textbox',
        'gradio.Button'
    ]
    
    for module_path in modules_to_test:
        log(f"\n尝试导入: {module_path}")
        try:
            # 构建导入语句
            if '.' in module_path:
                parts = module_path.split('.')
                import_statement = f"from {'.'.join(parts[:-1])} import {parts[-1]}"
                log(f"执行导入: {import_statement}")
                
                # 执行导入
                exec(import_statement)
                
                # 获取导入的对象
                obj = eval(parts[-1])
                log(f"✓ 成功导入: {module_path} -> {type(obj).__name__}")
            else:
                # 直接导入模块
                module = __import__(module_path)
                log(f"✓ 成功导入: {module_path} -> {module.__name__}")
                
        except Exception as e:
            log(f"✗ 导入失败: {type(e).__name__}: {e}")
            debug_log(traceback.format_exc())
            
            # 尝试导入时捕获更详细的信息
            log("尝试捕获更详细的错误信息...")
            try:
                import faulthandler
                faulthandler.enable()
                if '.' in module_path:
                    parts = module_path.split('.')
                    exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
                else:
                    __import__(module_path)
            except Exception as e2:
                log(f"使用faulthandler捕获的错误: {type(e2).__name__}: {e2}")
                debug_log(traceback.format_exc())
            
            break
    
    log("\n测试完成！")

if __name__ == "__main__":
    main()