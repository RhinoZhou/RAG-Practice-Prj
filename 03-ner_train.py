#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床文档命名实体识别与推理程序

该程序实现了一个完整的临床文档命名实体识别(NER)系统，使用BERT模型对医疗文本进行实体识别和分析。

主要功能：
  - 使用预训练的BERT模型进行命名实体识别模型的训练
  - 支持识别医学实体，包括疾病(Disease)和症状(Symptom)
  - 对识别出的实体进行极性(positive/negative等)和时间性(present/past等)标注
  - 提供完整的模型训练、保存和推理流程
  - 保存实体识别结果为结构化数据格式

输入数据：
  - 清洗后的临床文档文本文件，格式为JSONL
  - 每个文档包含doc_id和text字段

输出结果：
  - 训练好的模型保存在ner_model/目录
  - 实体识别结果保存在mentions.jsonl文件
  - 日志信息输出到ner_train.log文件和控制台

技术栈依赖：
  - transformers: 提供BERT模型、分词器和训练工具
  - datasets: 高效的数据处理和管理
  - seqeval: 序列标注任务的评估指标计算
  - torch: PyTorch深度学习框架
  - numpy: 数值计算库
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import torch
from transformers import (
    BertTokenizerFast, 
    BertForTokenClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
import evaluate
import numpy as np

# 设置日志配置
def setup_logging():
    """设置日志记录器，确保同时输出到文件和控制台
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 创建一个logger实例，命名为"NER_Trainer"
    logger = logging.getLogger("NER_Trainer")
    # 设置日志级别为INFO，捕获所有INFO及以上级别的日志
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器（在多次调用setup_logging时防止重复配置）
    if not logger.handlers:
        # 创建文件处理器，将日志写入ner_train.log文件
        file_handler = logging.FileHandler("ner_train.log", encoding='utf-8')
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

class NERTrainer:
    """命名实体识别训练器类
    
    负责BERT模型的初始化、训练、推理和结果保存的完整流程
    使用BIO标注系统（Begin-Inside-Outside）来标记实体
    """
    def __init__(self, model_name: str = "bert-base-chinese", num_labels: int = 5):
        """
        初始化NER训练器实例
        
        Args:
            model_name: 预训练BERT模型名称，默认为中文BERT模型
            num_labels: 标签类别数量，默认为5（O, B-Disease, I-Disease, B-Symptom, I-Symptom）
        """
        # 存储模型配置信息
        self.model_name = model_name
        self.num_labels = num_labels
        
        # 初始化分词器，用于将文本转换为模型可处理的token序列
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # 初始化BERT命名实体识别模型
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
        
        # 创建标签与ID的映射字典，便于模型训练和推理
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
    def load_data(self, file_path: str) -> DatasetDict:
        """
        从JSONL文件加载数据并处理为DatasetDict格式
        
        Args:
            file_path: JSONL文件路径，包含清洗后的文本数据
        
        Returns:
            DatasetDict: 包含训练集和测试集的数据字典
        """
        logger.info(f"开始加载数据: {file_path}")
        
        # 读取JSONL文件，每行一个JSON对象
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # 注意：以下代码为演示用，使用模拟标注数据
        # 在实际应用中，应使用真实标注的标签数据
        processed_data = []
        for item in data:
            # 将文本拆分为字符列表作为tokens
            tokens = list(item["text"])
            # 初始化标签列表，全部标记为"O"(非实体)
            labels = [0] * len(tokens)  # 0 对应 "O"标签
            
            # 模拟一些实体标注数据
            if len(tokens) > 5:
                # 在文本中随机位置标记一些示例实体
                labels[2] = 1  # 标记为B-Disease(疾病实体开始)
                for i in range(3, min(5, len(tokens))):
                    labels[i] = 2  # 标记为I-Disease(疾病实体内)
                
                if len(tokens) > 10:
                    labels[7] = 3  # 标记为B-Symptom(症状实体开始)
                    for i in range(8, min(10, len(tokens))):
                        labels[i] = 4  # 标记为I-Symptom(症状实体内)
            
            # 将处理后的数据添加到列表中
            processed_data.append({
                "text": "".join(tokens),       # 原始文本
                "tokens": tokens,              # 字符级tokens
                "labels": labels,              # 标签列表
                "doc_id": item["doc_id"]     # 文档ID
            })
        
        # 使用datasets库创建Dataset对象
        dataset = Dataset.from_list(processed_data)
        # 按8:2的比例分割训练集和测试集
        dataset_dict = dataset.train_test_split(test_size=0.2)
        
        # 设置标签列的格式为ClassLabel序列，便于后续处理
        dataset_dict = dataset_dict.cast_column(
            "labels", 
            Sequence(feature=ClassLabel(names=self.label_list))
        )
        
        logger.info(f"数据加载完成，训练集大小: {len(dataset_dict['train'])}, 测试集大小: {len(dataset_dict['test'])}")
        return dataset_dict
    
    def tokenize_and_align_labels(self, examples):
        """
        标记化文本并对齐标签，处理tokenizer可能导致的字符拆分问题
        
        在BERT等模型中，分词器可能会将单个字符或词拆分为多个token
        该方法确保标签与token正确对齐，特别是处理特殊标记和subword情况
        
        Args:
            examples: 包含文本和标签的批处理数据
        
        Returns:
            dict: 标记化并对齐后的数据，包含输入特征和对齐的标签
        """
        # 对输入文本进行标记化处理
        tokenized_inputs = self.tokenizer(
            examples["tokens"],             # 输入文本（已拆分为字符列表）
            is_split_into_words=True,        # 指示输入已拆分为单词/字符
            truncation=True,                 # 截断过长文本
            padding="max_length",           # 填充到最大长度
            max_length=128                   # 最大序列长度
        )
        
        # 处理标签对齐问题
        labels = []
        for i, label in enumerate(examples["labels"]):
            # 获取每个token对应的原始单词索引
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            # 为每个token确定对应的标签
            for word_idx in word_ids:
                # 对于特殊标记（[CLS], [SEP], [PAD]），标签设为-100（会被PyTorch忽略）
                if word_idx is None:
                    label_ids.append(-100)
                # 对于每个词的第一个token，使用原始标签
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # 对于同一个词的后续token，沿用当前标签
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        # 将对齐后的标签添加到标记化输入中
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_preds):
        """
        计算评估指标，包括准确率、精确率、召回率和F1分数
        
        使用seqeval库计算序列标注任务的标准评估指标
        
        Args:
            eval_preds: 包含模型预测logits和真实标签的元组
        
        Returns:
            dict: 包含各项评估指标的字典
        """
        # 解包预测结果和真实标签
        logits, labels = eval_preds
        # 从logits中获取预测类别（取最大值索引）
        predictions = np.argmax(logits, axis=-1)
        
        # 将-100的标签（对应特殊标记）替换为0（O标签），以便评估
        labels = np.where(labels != -100, labels, 0)
        
        # 加载seqeval评估指标
        seqeval = evaluate.load("seqeval")
        
        # 将标签ID转换回标签文本
        true_labels = [[self.id2label[label] for label in label_seq] for label_seq in labels]
        true_predictions = [[self.id2label[pred] for pred in pred_seq] for pred_seq in predictions]
        
        # 计算评估指标
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        
        # 返回格式化的评估指标结果
        return {
            "precision": results["overall_precision"],  # 精确率
            "recall": results["overall_recall"],        # 召回率
            "f1": results["overall_f1"],                # F1分数
            "accuracy": results["overall_accuracy"]     # 准确率
        }
    
    def train(self, dataset_dict: DatasetDict, output_dir: str = "ner_model"):
        """
        训练BERT命名实体识别模型
        
        使用transformers库的Trainer API进行模型训练，并保存训练好的模型
        
        Args:
            dataset_dict: 包含训练集和测试集的数据字典
            output_dir: 模型保存目录路径
        """
        logger.info(f"开始训练模型，输出目录: {output_dir}")
        
        # 对训练集和测试集进行标记化处理和标签对齐
        # 使用map函数批量处理数据
        tokenized_datasets = dataset_dict.map(
            self.tokenize_and_align_labels,     # 应用标签对齐函数
            batched=True,                       # 批量处理以提高效率
            remove_columns=dataset_dict["train"].column_names  # 移除原始列
        )
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,              # 模型输出目录
            evaluation_strategy="epoch",        # 每个epoch结束时评估一次
            learning_rate=2e-5,                 # 学习率（BERT模型常用的学习率）
            per_device_train_batch_size=16,     # 每个设备的训练批次大小
            per_device_eval_batch_size=16,      # 每个设备的评估批次大小
            num_train_epochs=3,                 # 训练轮数
            weight_decay=0.01,                  # 权重衰减（L2正则化）
            logging_dir="./logs",              # 日志保存目录
            logging_steps=10,                   # 每10步记录一次日志
            save_strategy="epoch",              # 每个epoch结束时保存模型
            load_best_model_at_end=True         # 训练结束时加载表现最好的模型
        )
        
        # 创建数据收集器，负责批量处理数据（填充等）
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # 创建Trainer实例
        trainer = Trainer(
            model=self.model,                   # 待训练的模型
            args=training_args,                 # 训练参数
            train_dataset=tokenized_datasets["train"],  # 训练数据集
            eval_dataset=tokenized_datasets["test"],   # 评估数据集
            tokenizer=self.tokenizer,           # 分词器
            data_collator=data_collator,        # 数据收集器
            compute_metrics=self.compute_metrics  # 评估指标计算函数
        )
        
        # 执行模型训练
        trainer.train()
        
        # 保存训练好的模型和分词器
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("模型训练完成并保存成功")
    
    def predict(self, texts: List[Dict], model_dir: str = "ner_model") -> List[Dict]:
        """
        使用训练好的模型进行实体识别推理
        
        对输入文本进行处理，识别并提取其中的医学实体
        
        Args:
            texts: 包含文本和文档ID的字典列表
            model_dir: 训练好的模型目录路径
        
        Returns:
            List[Dict]: 识别出的实体列表，每个实体包含doc_id, offset, text, type, polarity, temporality
        """
        logger.info(f"开始推理，使用模型: {model_dir}")
        
        # 加载训练好的模型，如果存在则加载本地模型，否则使用当前模型
        if os.path.exists(model_dir):
            logger.info(f"加载本地模型: {model_dir}")
            model = BertForTokenClassification.from_pretrained(model_dir)
            tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        else:
            logger.warning(f"未找到本地模型，使用预训练模型进行推理")
            model = self.model
            tokenizer = self.tokenizer
        
        # 设置模型为评估模式（关闭dropout等训练时的特殊层）
        model.eval()
        
        # 存储识别出的实体
        mentions = []
        
        # 遍历每个输入文本进行处理
        for item in texts:
            text = item["text"]
            doc_id = item["doc_id"]
            
            # 对文本进行标记化处理
            inputs = tokenizer(
                text,
                return_tensors="pt",          # 返回PyTorch张量
                truncation=True,               # 截断过长文本
                padding="max_length",         # 填充到最大长度
                max_length=128                 # 最大序列长度
            )
            
            # 进行预测，使用no_grad()上下文管理器关闭梯度计算以提高效率
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 从模型输出中获取预测结果（取概率最大的类别）
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # 将token ID转换回token文本
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # 解析预测结果，提取实体
            current_entity = None
            for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
                # 跳过特殊标记（[CLS], [SEP], [PAD]）
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                
                # 将预测ID转换为标签文本
                label = self.id2label[pred.item()]
                
                # 基于BIO标签体系处理实体
                if label.startswith("B-"):
                    # 如果当前有未完成的实体，先保存
                    if current_entity is not None:
                        mentions.append(current_entity)
                        current_entity = None
                    
                    # 开始一个新的实体
                    entity_type = label[2:]  # 提取实体类型（如Disease, Symptom）
                    current_entity = {
                        "doc_id": doc_id,
                        "offset": [i, i],    # 记录实体的起始和结束位置（token级别）
                        "text": token,       # 实体文本内容
                        "type": entity_type, # 实体类型
                        "polarity": "positive",  # 极性标记（简化处理）
                        "temporality": "present"  # 时间性标记（简化处理）
                    }
                elif label.startswith("I-") and current_entity is not None:
                    # 继续当前实体（处理实体的内部部分）
                    current_entity["offset"][1] = i  # 更新实体结束位置
                    current_entity["text"] += token  # 拼接实体文本
                else:
                    # 非实体或实体结束
                    if current_entity is not None:
                        mentions.append(current_entity)
                        current_entity = None
            
            # 保存最后一个未处理完的实体（如果有）
            if current_entity is not None:
                mentions.append(current_entity)
        
        logger.info(f"推理完成，共识别出 {len(mentions)} 个实体")
        return mentions
    
    def save_mentions(self, mentions: List[Dict], output_file: str = "mentions.jsonl"):
        """
        保存实体识别结果到JSONL文件
        
        将识别出的实体信息以JSONL格式保存到文件，便于后续处理和分析
        
        Args:
            mentions: 实体列表，每个实体包含完整的实体信息
            output_file: 输出文件路径
        """
        logger.info(f"保存实体识别结果到: {output_file}")
        
        # 以UTF-8编码打开文件，逐行写入实体信息
        with open(output_file, 'w', encoding='utf-8') as f:
            for mention in mentions:
                # 使用json.dumps将实体字典转换为JSON字符串，ensure_ascii=False保留非ASCII字符
                f.write(json.dumps(mention, ensure_ascii=False) + '\n')
        
        logger.info(f"结果保存完成")

def main():
    """主函数，协调整个实体识别和推理流程
    
    负责程序的整体执行流程，包括参数设置、数据加载、模型训练、推理和结果保存
    包含完整的错误处理机制，确保程序稳定运行
    """
    logger.info("程序开始执行")
    
    try:
        # 配置参数设置
        input_file = os.path.abspath("d:/rag-project/05-rag-practice/05-knowledge_graph_construction/clean_text.jsonl")
        output_dir = "ner_model"
        mentions_file = "mentions.jsonl"
        
        # 记录配置信息到日志
        logger.info(f"配置参数 - 输入文件: {input_file}")
        logger.info(f"配置参数 - 输出目录: {output_dir}")
        logger.info(f"配置参数 - 实体结果文件: {mentions_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        else:
            logger.info(f"输入文件已找到: {input_file}")
        
        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir):
            logger.info(f"创建输出目录: {output_dir}")
            os.makedirs(output_dir)
        else:
            logger.info(f"输出目录已存在: {output_dir}")
        
        # 初始化NER训练器
        logger.info("初始化NER训练器")
        ner_trainer = NERTrainer()
        
        # 加载和处理数据
        logger.info("开始加载数据")
        dataset_dict = ner_trainer.load_data(input_file)
        logger.info(f"数据加载完成，训练集大小: {len(dataset_dict['train'])}, 测试集大小: {len(dataset_dict['test'])}")
        
        # 训练模型
        logger.info("开始训练模型")
        ner_trainer.train(dataset_dict, output_dir)
        logger.info("模型训练完成")
        
        # 读取原始数据用于推理
        logger.info("读取原始数据用于推理")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
        logger.info(f"读取到 {len(raw_data)} 条原始数据")
        
        # 进行推理（仅使用前10条数据进行测试）
        logger.info("开始推理过程")
        mentions = ner_trainer.predict(raw_data[:10], output_dir)  # 仅处理前10条数据
        logger.info(f"推理完成，共识别出 {len(mentions)} 个实体")
        
        # 保存实体识别结果
        logger.info(f"保存实体识别结果到: {mentions_file}")
        ner_trainer.save_mentions(mentions, mentions_file)
        
        logger.info("实体识别和推理流程已完成")
        
    except FileNotFoundError as e:
        # 处理文件不存在错误
        logger.error(f"文件错误: {str(e)}")
        print(f"错误: {str(e)}")
        raise
    except ImportError as e:
        # 处理依赖包导入错误
        logger.error(f"导入错误: {str(e)}")
        print(f"错误: 缺少必要的依赖包，请运行 pip install -r requirements.txt\n{str(e)}")
        raise
    except Exception as e:
        # 处理其他未预期的错误
        logger.error(f"程序执行出错: {str(e)}")
        print(f"错误: 程序执行出错，请查看ner_train.log文件获取详细信息\n{str(e)}")
        raise

if __name__ == "__main__":
    """程序入口点
    
    当脚本被直接执行时，调用main函数开始执行程序
    """
    main()