#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量删除程序文件中注释的作者和日期信息
"""

import os
import re
import glob

# 定义要处理的文件类型
FILE_TYPES = ["*.py", "*.txt"]

# 定义要删除的模式
AUTHOR_PATTERNS = [
    r'    r'    r'    r']

DATE_PATTERNS = [
    r'    r'    r'    r'    r'    r']

def remove_author_date_comments(file_path):
    """删除文件中注释的作者和日期信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有作者模式
        for pattern in AUTHOR_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # 应用所有日期模式
        for pattern in DATE_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # 如果内容有变化，保存文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"已处理: {file_path}"
        else:
            return False, f"未修改: {file_path}"
            
    except Exception as e:
        return False, f"处理失败 {file_path}: {str(e)}"

def main():
    """主函数"""
    print("开始批量删除作者和日期注释...")
    
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"处理目录: {current_dir}")
    
    # 统计信息
    total_files = 0
    modified_files = 0
    failed_files = 0
    
    # 遍历所有文件类型
    for file_type in FILE_TYPES:
        # 获取匹配的文件
        files = glob.glob(os.path.join(current_dir, "**", file_type), recursive=True)
        
        for file_path in files:
            total_files += 1
            success, message = remove_author_date_comments(file_path)
            print(message)
            
            if success:
                modified_files += 1
            elif "失败" in message:
                failed_files += 1
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"修改文件数: {modified_files}")
    print(f"失败文件数: {failed_files}")
    print("=" * 50)

if __name__ == "__main__":
    main()