#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_docx.py

功能：读取Word文档(.docx)内容并输出为纯文本

"""
import docx

def read_docx_file(filename):
    """
    读取Word文档内容
    
    参数：
        filename (str): Word文档路径
    
    返回：
        str: 文档内容的纯文本形式
    """
    doc = docx.Document(filename)
    content = []
    
    # 遍历文档中的所有段落
    for para in doc.paragraphs:
        content.append(para.text)
    
    return '\n'.join(content)

if __name__ == "__main__":
    # 示例用法：读取代码功能详细解释文档并保存为纯文本
    content = read_docx_file('代码功能详细解释.docx')
    
    # 将内容保存到文本文件
    with open('code_explanation.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("内容已保存到code_explanation.txt文件")