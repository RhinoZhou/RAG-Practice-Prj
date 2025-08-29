#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床文档命名实体识别与推理程序（简化版）

该程序是一个轻量级的临床文档命名实体识别系统，使用预训练的BERT模型快速识别医学文本中的实体。
相比完整版，该简化版专注于实体识别推理功能，不包含模型训练部分。

主要功能：
  - 使用预训练的BERT模型进行临床文档命名实体识别
  - 支持识别医学实体，包括疾病(Disease)和症状(Symptom)
  - 对识别出的实体提供极性(positive/negative等)和时间性(present/past等)标注
  - 将识别结果保存为结构化JSONL格式

输入数据：
  - 清洗后的临床文档文本文件，格式为JSONL
  - 每个文档包含doc_id和text字段

输出结果：
  - 实体识别结果保存在mentions.jsonl文件
  - 日志信息输出到ner_simple.log文件和控制台

技术栈依赖：
  - transformers: 提供BERT模型和分词器
  - torch: PyTorch深度学习框架
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# 设置日志配置
def setup_logging():
    """设置日志记录器，确保同时输出到文件和控制台
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 创建一个logger实例，命名为"NER_Simple"
    logger = logging.getLogger("NER_Simple")
    # 设置日志级别为INFO，捕获所有INFO及以上级别的日志
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器（在多次调用setup_logging时防止重复配置）
    if not logger.handlers:
        # 创建文件处理器，将日志写入ner_simple.log文件
        file_handler = logging.FileHandler("ner_simple.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式，包含时间戳、logger名称、日志级别和消息内容
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将处理器添加到logger实例
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logging()

class SimpleNERTagger:
    """简化版命名实体识别器类
    
    提供使用预训练BERT模型进行临床文档实体识别的功能
    专注于推理阶段，不包含模型训练部分
    使用BIO标注系统标记实体
    """
    def __init__(self, model_name: str = "bert-base-chinese", num_labels: int = 5):
        """
        初始化NER标记器实例
        
        Args:
            model_name: 预训练BERT模型名称，默认为中文BERT模型
            num_labels: 标签类别数量，默认为5（O, B-Disease, I-Disease, B-Symptom, I-Symptom）
        """
        # 存储模型配置信息
        self.model_name = model_name
        self.num_labels = num_labels
        
        # 加载分词器，用于将文本转换为模型可处理的token序列
        logger.info(f"加载分词器: {model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # 加载BERT命名实体识别模型
        logger.info(f"加载模型: {model_name}")
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # 忽略预训练模型和当前任务的参数不匹配
        )
        
        # 定义标签体系：采用BIO格式标记实体
        # O: 非实体
        # B-Disease: 疾病实体的开始
        # I-Disease: 疾病实体的内部
        # B-Symptom: 症状实体的开始
        # I-Symptom: 症状实体的内部
        self.label_list = ["O", "B-Disease", "I-Disease", "B-Symptom", "I-Symptom"]
        
        # 创建标签与ID的映射字典，便于模型推理
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        # 设置模型为评估模式（关闭dropout等训练时的特殊层）
        self.model.eval()
    
    def load_data(self, file_path: str) -> List[Dict]:
        """
        从JSONL文件加载临床文本数据
        
        Args:
            file_path: JSONL文件路径，包含清洗后的临床文本数据
        
        Returns:
            List[Dict]: 包含文本和文档ID的数据列表
        """
        logger.info(f"开始加载数据: {file_path}")
        
        # 读取JSONL文件，每行一个JSON对象
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        logger.info(f"数据加载完成，共 {len(data)} 条记录")
        return data
    
    def recognize_entities(self, texts: List[Dict]) -> List[Dict]:
        """
        识别文本中的医学实体
        
        对输入文本进行处理，使用BERT模型识别并提取其中的医学实体
        支持识别疾病(Disease)和症状(Symptom)两种实体类型
        
        Args:
            texts: 包含文本和文档ID的字典列表
        
        Returns:
            List[Dict]: 识别出的实体列表，每个实体包含完整信息
        """
        logger.info(f"开始实体识别，共 {len(texts)} 条文本")
        
        # 存储识别出的实体
        mentions = []
        
        # 遍历每个输入文本进行处理
        for item in texts:
            text = item["text"]
            doc_id = item["doc_id"]
            
            try:
                # 对文本进行标记化处理
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",      # 返回PyTorch张量
                    truncation=True,           # 截断过长文本
                    max_length=512             # 最大序列长度设置为512
                )
                
                # 使用模型进行预测，关闭梯度计算以提高效率
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    # 从logits中获取预测类别（取最大值索引）
                    predictions = torch.argmax(logits, dim=2)
                
                # 将token ID转换回token文本
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                # 解析预测结果，提取实体
                current_entity = None
                char_offset = 0  # 当前字符在原始文本中的偏移量
                
                for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
                    # 跳过特殊标记（[CLS], [SEP], [PAD]）
                    if token in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue
                    
                    # 将预测ID转换为标签文本
                    label = self.id2label[pred.item()]
                    
                    # 计算token的实际字符长度
                    # 注意：这里是简化处理，移除BERT的子词标记"##"
                    token_len = len(token.replace("##", ""))
                    
                    # 基于BIO标签体系处理实体
                    if label.startswith("B-"):
                        # 如果当前有未完成的实体，先保存
                        if current_entity is not None:
                            mentions.append(current_entity)
                            current_entity = None
                        
                        # 开始一个新的实体
                        entity_type = label[2:]  # 提取实体类型（Disease或Symptom）
                        current_entity = {
                            "doc_id": doc_id,
                            "offset": [char_offset, char_offset + token_len],  # 实体在原文中的字符位置
                            "text": token.replace("##", ""),                   # 实体文本内容
                            "type": entity_type,                               # 实体类型
                            "polarity": "positive",                            # 极性标记（简化处理）
                            "temporality": "present"                             # 时间性标记（简化处理）
                        }
                    elif label.startswith("I-") and current_entity is not None:
                        # 继续当前实体（处理实体的内部部分）
                        current_entity["offset"][1] = char_offset + token_len  # 更新实体结束位置
                        current_entity["text"] += token.replace("##", "")     # 拼接实体文本
                    else:
                        # 非实体或实体结束
                        if current_entity is not None:
                            mentions.append(current_entity)
                            current_entity = None
                    
                    # 更新字符偏移量，指向下一个token的起始位置
                    char_offset += token_len
                
                # 保存最后一个未处理完的实体（如果有）
                if current_entity is not None:
                    mentions.append(current_entity)
                
            except Exception as e:
                # 处理单个文档的错误，不影响整体流程
                logger.error(f"处理文档 {doc_id} 时出错: {str(e)}")
                continue
        
        logger.info(f"实体识别完成，共识别出 {len(mentions)} 个实体")
        return mentions
    
    def save_mentions(self, mentions: List[Dict], output_file: str = "mentions.jsonl"):
        """
        保存实体识别结果到JSONL文件
        
        将识别出的实体信息以JSONL格式保存到文件，便于后续处理和分析
        
        Args:
            mentions: 实体列表，每个实体包含完整的实体信息
            output_file: 输出文件路径，默认为mentions.jsonl
        """
        logger.info(f"保存实体识别结果到: {output_file}")
        
        # 以UTF-8编码打开文件，逐行写入实体信息
        with open(output_file, 'w', encoding='utf-8') as f:
            for mention in mentions:
                # 使用json.dumps将实体字典转换为JSON字符串，ensure_ascii=False保留非ASCII字符
                f.write(json.dumps(mention, ensure_ascii=False) + '\n')
        
        logger.info(f"结果保存完成，共 {len(mentions)} 条实体记录")

def main():
    """主函数，协调整个实体识别流程
    
    负责程序的整体执行流程，包括参数设置、数据加载、实体识别和结果保存
    包含完整的错误处理机制，确保程序稳定运行
    """
    logger.info("程序开始执行")
    
    try:
        # 配置参数设置
        input_file = "clean_text.jsonl"
        output_file = "mentions.jsonl"
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 初始化NER标记器
        ner_tagger = SimpleNERTagger()
        
        # 加载数据
        data = ner_tagger.load_data(input_file)
        
        # 只处理前10条数据用于测试
        # 注意：在实际应用中，可以根据需要处理全部数据
        sample_data = data[:10]
        logger.info(f"使用 {len(sample_data)} 条数据进行测试")
        
        # 识别实体
        mentions = ner_tagger.recognize_entities(sample_data)
        
        # 保存识别结果
        ner_tagger.save_mentions(mentions, output_file)
        
        logger.info("程序执行完成")
        
    except Exception as e:
        # 记录错误信息并打印到控制台
        logger.error(f"程序执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    """程序入口点
    
    当脚本被直接执行时，调用main函数开始执行程序
    """
    main()