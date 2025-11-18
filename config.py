#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

功能：配置文件，定义系统的所有参数设置

"""
class AppConfig():

    # 检索相关参数
    top_doc_count = 3    # 召回文档数量
    top_text_count = 6    # 召回文本片段数量
    max_text_length = 128  # 召回文本片段长度
    top_keyword_count = 5    # 查询召回关键词数量
    embedding_model_path = '/workspace/model/embedding/tao-8k'
    retrieval_method = 'embed'  # 召回方式 ,keyword,embed

    # 生成器参数
    max_input_length = 767  # 输入最大长度
    max_output_length = 256  # 生成最大长度
    model_max_sequence_length = 1024  # 序列最大长度
    
    # 嵌入 API 参数 - 用于 text2vec.py
    use_embedding_api = True  # 是否使用API而非本地模型
    embedding_api_key = "sk-xx"
    embedding_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model_name = "text-embedding-v3"
    embedding_dimensions = 1024
    embedding_batch_size = 10
    
    # LLM API 参数 - 用于 rag.py
    llm_api_key = "sk-xx"  # 与embedding共用同一个key
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 与embedding共用同一个URL
    llm_model = "qwen-plus"  # 默认使用的LLM模型
    
    # 知识库配置
    knowledge_base_root = "knowledge_bases"  # 知识库根目录
    default_knowledge_base = "default"  # 默认知识库名称
    
    # 输出目录配置 - 现在用作临时文件目录
    temp_output_dir = "output_files"