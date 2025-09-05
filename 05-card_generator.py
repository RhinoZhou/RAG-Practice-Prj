#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
结构化数据向量化与检索策略实现

功能说明：
- 处理行记录或多表join结果，生成实体卡片或指标卡片
- 支持多语言模板（中文/英文）
- 实现单位标准化、字段空值回退
- 提供文本嵌入和向量检索功能
- 输出包含文本卡片和元数据的完整结果

使用说明：
- 主要类：CardGenerator（卡片生成器）、UnitConverter（单位转换器）
- 支持模拟嵌入模型，无需API密钥也可运行
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os
import random
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 记录程序开始时间
program_start_time = time.time()

class MockEmbeddings:
    """模拟嵌入模型，用于演示目的"""
    
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        # 设置随机种子以确保结果可复现
        random.seed(42)
        # 为了兼容LangChain的调用方式
        self.embed_query = self._embed_query
        self.embed_documents = self._embed_documents
    
    def _embed_query(self, text: str) -> List[float]:
        """生成模拟的嵌入向量"""
        # 基于文本内容生成确定性的随机向量
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
        random.seed(hash_val)
        return [random.random() for _ in range(self.embedding_dim)]
    
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为多个文档生成模拟的嵌入向量"""
        return [self._embed_query(text) for text in texts]
        
    def __call__(self, text: str) -> List[float]:
        """使对象可调用，兼容FAISS的调用方式"""
        return self._embed_query(text)

class CardGenerator:
    """结构化数据向量化与检索策略实现类"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", 
                 embedding_dim: int = 1536, 
                 default_language: str = "zh",
                 cache_dir: str = "./vector_cache",
                 use_mock_embeddings: bool = False,
                 openai_api_key: Optional[str] = None):
        """
        初始化卡片生成器
        
        Args:
            embedding_model: 嵌入模型名称
            embedding_dim: 嵌入维度
            default_language: 默认语言
            cache_dir: 向量缓存目录
            use_mock_embeddings: 是否使用模拟嵌入（无需API密钥）
            openai_api_key: OpenAI API密钥（如果不使用环境变量）
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.default_language = default_language
        self.cache_dir = cache_dir
        
        # 初始化嵌入模型
        if use_mock_embeddings:
            # 使用模拟嵌入模型
            self.embeddings = MockEmbeddings(embedding_dim)
            self.use_mock_embeddings = True
        else:
            # 尝试导入OpenAIEmbeddings并初始化
            try:
                from langchain_openai import OpenAIEmbeddings
                
                # 配置参数
                client_params = {}
                if openai_api_key:
                    client_params["api_key"] = openai_api_key
                
                self.embeddings = OpenAIEmbeddings(model=embedding_model, **client_params)
                self.use_mock_embeddings = False
            except ImportError:
                print("警告: 无法导入OpenAIEmbeddings，将使用模拟嵌入模型")
                self.embeddings = MockEmbeddings(embedding_dim)
                self.use_mock_embeddings = True
            except Exception as e:
                print(f"警告: 初始化OpenAIEmbeddings失败: {str(e)}，将使用模拟嵌入模型")
                self.embeddings = MockEmbeddings(embedding_dim)
                self.use_mock_embeddings = True
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化单位转换器
        self.unit_converter = UnitConverter()
        
        # 初始化向量存储
        self.vector_store = None
        
        # 定义多语言模板
        self.templates = {
            "zh": {
                "entity": {
                    "product": "产品卡片\n产品ID: {product_id}\n产品名称: {product_name}\n类别: {category}\n价格: {price}\n库存: {stock}\n描述: {description}\n上架日期: {launch_date}",
                    "customer": "客户卡片\n客户ID: {customer_id}\n客户姓名: {customer_name}\n性别: {gender}\n年龄: {age}\n地区: {region}\n会员等级: {membership_level}\n注册日期: {registration_date}"
                },
                "metric": {
                    "sales": "销售指标卡片\n日期: {date}\n产品ID: {product_id}\n销售额: {sales_amount}\n销售量: {sales_volume}\n客单价: {avg_order_value}\n同比增长: {yoy_growth}\n环比增长: {mom_growth}",
                    "inventory": "库存指标卡片\n日期: {date}\n产品ID: {product_id}\n当前库存: {current_stock}\n安全库存: {safety_stock}\n周转率: {turnover_rate}\n缺货次数: {stockout_count}\n库存周转天数: {stock_days}"
                }
            },
            "en": {
                "entity": {
                    "product": "Product Card\nProduct ID: {product_id}\nProduct Name: {product_name}\nCategory: {category}\nPrice: {price}\nStock: {stock}\nDescription: {description}\nLaunch Date: {launch_date}",
                    "customer": "Customer Card\nCustomer ID: {customer_id}\nCustomer Name: {customer_name}\nGender: {gender}\nAge: {age}\nRegion: {region}\nMembership Level: {membership_level}\nRegistration Date: {registration_date}"
                },
                "metric": {
                    "sales": "Sales Metric Card\nDate: {date}\nProduct ID: {product_id}\nSales Amount: {sales_amount}\nSales Volume: {sales_volume}\nAvg. Order Value: {avg_order_value}\nYoY Growth: {yoy_growth}\nMoM Growth: {mom_growth}",
                    "inventory": "Inventory Metric Card\nDate: {date}\nProduct ID: {product_id}\nCurrent Stock: {current_stock}\nSafety Stock: {safety_stock}\nTurnover Rate: {turnover_rate}\nStockout Count: {stockout_count}\nStock Turnover Days: {stock_days}"
                }
            }
        }
        
        # 定义空值回退字典
        self.null_fallback = {
            "zh": "暂无数据",
            "en": "No data available"
        }
    
    def _generate_card_id(self, data: Dict[str, Any], card_type: str, subtype: str) -> str:
        """生成卡片唯一ID"""
        # 提取主键信息作为ID生成的基础
        key_fields = []
        if card_type == "entity":
            if subtype == "product":
                key_fields.append(f"product_{data.get('product_id', 'unknown')}")
            elif subtype == "customer":
                key_fields.append(f"customer_{data.get('customer_id', 'unknown')}")
        elif card_type == "metric":
            key_fields.append(f"{subtype}_{data.get('date', 'unknown')}")
            if subtype in ["sales", "inventory"]:
                key_fields.append(f"product_{data.get('product_id', 'unknown')}")
        
        # 添加时间戳确保唯一性
        key_fields.append(str(int(time.time() * 1000)))
        
        # 生成MD5哈希作为ID
        key_str = "_".join(key_fields)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _process_data(self, data: Dict[str, Any], card_type: str, subtype: str, language: str = "zh") -> Dict[str, Any]:
        """处理数据，包括单位标准化和空值回退"""
        processed_data = data.copy()
        
        # 确保语言可用
        if language not in self.templates:
            language = self.default_language
            
        # 根据卡片类型和子类型处理不同的字段
        if card_type == "entity":
            if subtype == "product":
                # 标准化价格单位
                if "price" in processed_data:
                    processed_data["price"] = self.unit_converter.convert_currency(
                        processed_data["price"], target_unit="CNY", language=language)
                
                # 标准化库存单位
                if "stock" in processed_data:
                    processed_data["stock"] = self.unit_converter.convert_quantity(
                        processed_data["stock"], target_unit="件", language=language)
            elif subtype == "customer":
                # 处理年龄空值
                if "age" not in processed_data or processed_data["age"] is None:
                    processed_data["age"] = self.null_fallback[language]
        elif card_type == "metric":
            if subtype == "sales":
                # 标准化金额单位
                if "sales_amount" in processed_data:
                    processed_data["sales_amount"] = self.unit_converter.convert_currency(
                        processed_data["sales_amount"], target_unit="CNY", language=language)
                
                # 标准化百分比
                for field in ["yoy_growth", "mom_growth"]:
                    if field in processed_data:
                        processed_data[field] = self.unit_converter.convert_percentage(
                            processed_data[field], language=language)
            elif subtype == "inventory":
                # 标准化库存单位
                for field in ["current_stock", "safety_stock"]:
                    if field in processed_data:
                        processed_data[field] = self.unit_converter.convert_quantity(
                            processed_data[field], target_unit="件", language=language)
                
                # 标准化周转率
                if "turnover_rate" in processed_data:
                    processed_data["turnover_rate"] = self.unit_converter.convert_percentage(
                        processed_data["turnover_rate"], language=language)
        
        # 处理所有空值字段
        for key, value in processed_data.items():
            if value is None or value == "":
                processed_data[key] = self.null_fallback[language]
        
        return processed_data
    
    def generate_card(self, data: Dict[str, Any], card_type: str, subtype: str, 
                      language: str = "zh", 
                      validity_days: int = 7, 
                      verbose: bool = False) -> Dict[str, Any]:
        """
        生成实体卡片或指标卡片
        
        Args:
            data: 输入数据（行记录或多表join结果）
            card_type: 卡片类型（entity或metric）
            subtype: 卡片子类型（如product、customer、sales、inventory等）
            language: 语言（zh或en）
            validity_days: 卡片有效期（天）
            verbose: 是否输出详细信息
            
        Returns:
            包含文本卡片和元数据的字典
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始生成{card_type}类型-{subtype}卡片...")
            print(f"输入数据: {json.dumps({k: v for k, v in data.items() if v is not None}, ensure_ascii=False, indent=2)}")
        
        # 验证卡片类型和子类型
        if card_type not in self.templates.get(language, self.templates[self.default_language]):
            raise ValueError(f"不支持的卡片类型: {card_type}")
        
        if subtype not in self.templates.get(language, self.templates[self.default_language])[card_type]:
            raise ValueError(f"不支持的卡片子类型: {subtype}")
        
        # 确保语言可用
        if language not in self.templates:
            language = self.default_language
        
        # 处理数据
        processed_data = self._process_data(data, card_type, subtype, language)
        
        if verbose:
            print(f"数据处理完成: {json.dumps({k: v for k, v in processed_data.items() if v is not None}, ensure_ascii=False, indent=2)}")
        
        # 生成卡片ID
        card_id = self._generate_card_id(processed_data, card_type, subtype)
        
        # 计算有效期
        valid_from = datetime.now().isoformat()
        valid_until = (datetime.now() + timedelta(days=validity_days)).isoformat()
        
        # 生成卡片文本
        template = self.templates[language][card_type][subtype]
        card_text = template.format(**processed_data)
        
        # 构建返回结果
        result = {
            "card_id": card_id,
            "card_type": card_type,
            "subtype": subtype,
            "language": language,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "content": card_text,
            "metadata": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "source_data": {k: v for k, v in data.items() if v is not None},
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        }
        
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 卡片生成完成，耗时: {(time.time() - start_time)*1000:.2f}ms")
        
        return result
    
    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的嵌入向量"""
        return self.embeddings.embed_query(text)
    
    def create_document(self, card: Dict[str, Any]) -> Document:
        """将卡片转换为LangChain的Document对象"""
        return Document(
            page_content=card["content"],
            metadata={
                "card_id": card["card_id"],
                "card_type": card["card_type"],
                "subtype": card["subtype"],
                "language": card["language"],
                "valid_from": card["valid_from"],
                "valid_until": card["valid_until"],
                **card["metadata"]
            }
        )
    
    def init_vector_store(self, documents: List[Document] = None, verbose: bool = False):
        """初始化向量存储"""
        start_time = time.time()
        
        if verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 初始化向量存储...")
            print(f"文档数量: {len(documents) if documents else 0}")
        
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # 创建空的向量存储
            self.vector_store = FAISS.from_texts(["dummy"], self.embeddings)
            # 移除dummy文档
            self.vector_store.delete(["""d9e6762dd1c8eaf6d61b3c6192fc408d"""])  # dummy文档的ID
        
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 向量存储初始化完成，耗时: {(time.time() - start_time)*1000:.2f}ms")
    
    def add_to_vector_store(self, documents: List[Document]):
        """向向量存储添加文档"""
        if self.vector_store is None:
            self.init_vector_store()
        self.vector_store.add_documents(documents)
    
    def search_similar_cards(self, query: str, k: int = 5, 
                             filter: Optional[Dict[str, Any]] = None, 
                             verbose: bool = False) -> List[Tuple[Document, float]]:
        """搜索相似卡片"""
        start_time = time.time()
        
        if verbose:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始相似卡片搜索...")
            print(f"搜索查询: '{query}'")
            if filter:
                print(f"搜索过滤器: {json.dumps(filter, ensure_ascii=False, indent=2)}")
        
        if self.vector_store is None:
            raise ValueError("向量存储尚未初始化，请先调用init_vector_store或add_to_vector_store方法")
        
        # 如果提供了过滤器，使用带过滤的搜索
        if filter:
            # 注意：FAISS的过滤器实现有限，这里简化处理
            # 在实际应用中，可能需要使用其他支持复杂过滤的向量存储
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # 手动过滤结果
            filtered_results = []
            for doc, score in results:
                match = True
                for key, value in filter.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append((doc, score))
            results = filtered_results[:k]  # 确保返回不超过k个结果
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 搜索完成，找到{len(results)}个相似结果，耗时: {(time.time() - start_time)*1000:.2f}ms")
        
        return results
    
    def save_vector_store(self, filename: str = "vector_store"):
        """保存向量存储到文件"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未初始化")
        
        file_path = os.path.join(self.cache_dir, filename)
        self.vector_store.save_local(file_path)
        return file_path
    
    def load_vector_store(self, filename: str = "vector_store"):
        """从文件加载向量存储"""
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"向量存储文件不存在: {file_path}")
        
        self.vector_store = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)
        return self.vector_store

class UnitConverter:
    """单位转换器，处理单位标准化"""
    
    def __init__(self):
        # 定义货币转换率（相对于CNY）
        self.currency_rates = {
            "CNY": 1.0,
            "USD": 0.138,
            "EUR": 0.127,
            "JPY": 21.23,
            "GBP": 0.109
        }
        
        # 定义语言特定的单位名称
        self.unit_names = {
            "zh": {
                "CNY": "元",
                "USD": "美元",
                "EUR": "欧元",
                "JPY": "日元",
                "GBP": "英镑",
                "quantity": {
                    "件": "件",
                    "个": "个",
                    "箱": "箱",
                    "包": "包"
                }
            },
            "en": {
                "CNY": "CNY",
                "USD": "USD",
                "EUR": "EUR",
                "JPY": "JPY",
                "GBP": "GBP",
                "quantity": {
                    "件": "pcs",
                    "个": "units",
                    "箱": "boxes",
                    "包": "packages"
                }
            }
        }
    
    def convert_currency(self, amount: Union[float, str], target_unit: str = "CNY", 
                        source_unit: Optional[str] = None, language: str = "zh") -> str:
        """转换货币单位"""
        # 处理字符串类型的金额
        if isinstance(amount, str):
            # 尝试从字符串中提取金额和单位
            try:
                # 简单处理，假设格式为"123.45 元"或"$123.45"
                import re
                if language == "zh" and "元" in amount:
                    numeric_part = re.search(r'[\d.]+', amount).group()
                    amount = float(numeric_part)
                    source_unit = "CNY"
                elif "$" in amount:
                    numeric_part = re.search(r'[\d.]+', amount).group()
                    amount = float(numeric_part)
                    source_unit = "USD"
                elif "€" in amount:
                    numeric_part = re.search(r'[\d.]+', amount).group()
                    amount = float(numeric_part)
                    source_unit = "EUR"
                else:
                    # 假设是纯数字字符串
                    amount = float(amount)
            except:
                # 如果无法解析，返回原始值
                return amount
        
        # 如果未指定源单位，假设是目标单位
        if source_unit is None:
            source_unit = target_unit
        
        # 转换金额
        try:
            if source_unit != target_unit and source_unit in self.currency_rates and target_unit in self.currency_rates:
                converted_amount = amount * (self.currency_rates[target_unit] / self.currency_rates[source_unit])
            else:
                converted_amount = amount
            
            # 格式化输出
            if language == "zh":
                return f"{converted_amount:.2f} {self.unit_names[language][target_unit]}"
            else:
                return f"{self.unit_names[language][target_unit]} {converted_amount:.2f}"
        except:
            # 如果转换失败，返回原始值
            return str(amount)
    
    def convert_quantity(self, quantity: Union[int, float, str], target_unit: str = "件", 
                        source_unit: Optional[str] = None, language: str = "zh") -> str:
        """转换数量单位"""
        # 处理字符串类型的数量
        if isinstance(quantity, str):
            try:
                # 简单处理，提取数字部分
                import re
                numeric_part = re.search(r'[\d.]+', quantity).group()
                quantity = float(numeric_part)
            except:
                # 如果无法解析，返回原始值
                return quantity
        
        # 格式化输出
        try:
            unit_name = self.unit_names[language]["quantity"].get(target_unit, target_unit)
            if int(quantity) == quantity:
                return f"{int(quantity)} {unit_name}"
            else:
                return f"{quantity:.2f} {unit_name}"
        except:
            # 如果格式化失败，返回原始值
            return str(quantity)
    
    def convert_percentage(self, value: Union[float, str], decimal_places: int = 2, 
                          language: str = "zh") -> str:
        """转换为百分比格式"""
        # 处理字符串类型的值
        if isinstance(value, str):
            try:
                # 尝试从字符串中提取数值
                if "%" in value:
                    # 如果已经是百分比格式，先去掉百分号
                    value = float(value.replace("%", "")) / 100
                else:
                    # 假设是小数或整数
                    value = float(value)
            except:
                # 如果无法解析，返回原始值
                return value
        
        # 格式化输出
        try:
            if language == "zh":
                return f"{value * 100:.{decimal_places}f}%"
            else:
                return f"{value * 100:.{decimal_places}f} percent"
        except:
            # 如果格式化失败，返回原始值
            return str(value)

# 示例用法
if __name__ == "__main__":
    print("="*80)
    print("结构化数据向量化与检索策略演示程序")
    print("="*80)
    
    # 初始化卡片生成器，使用模拟嵌入以便在没有API密钥的情况下也能运行
    print("\n[初始化阶段]")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 初始化卡片生成器...")
    card_gen = CardGenerator(use_mock_embeddings=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 卡片生成器初始化完成，使用{'模拟嵌入模型' if card_gen.use_mock_embeddings else 'OpenAI嵌入模型'}")
    
    # 示例1：生成产品实体卡片
    print("\n" + "-"*80)
    print("示例1：生成产品实体卡片")
    print("-"*80)
    product_data = {
        "product_id": "P12345",
        "product_name": "高性能笔记本电脑",
        "category": "电子产品",
        "price": 6999.99,
        "stock": 150,
        "description": "配备最新处理器和高性能显卡，适合游戏和专业设计工作",
        "launch_date": "2023-06-15"
    }
    
    product_card = card_gen.generate_card(product_data, "entity", "product", verbose=True)
    print("\n=== 产品实体卡片 ===")
    print(f"卡片ID: {product_card['card_id']}")
    print(f"有效期: {product_card['valid_from']} 至 {product_card['valid_until']}")
    print(f"处理时间: {product_card['metadata'].get('processing_time_ms', 0)}ms")
    print(f"\n卡片内容:\n{product_card['content']}")
    print(f"\n源数据: {json.dumps(product_card['metadata']['source_data'], ensure_ascii=False, indent=2)}")
    
    # 示例2：生成销售指标卡片
    print("\n" + "-"*80)
    print("示例2：生成销售指标卡片")
    print("-"*80)
    sales_data = {
        "date": "2023-07-01",
        "product_id": "P12345",
        "sales_amount": 125000.75,
        "sales_volume": 18,
        "avg_order_value": 6944.48,
        "yoy_growth": 0.235,
        "mom_growth": 0.087
    }
    
    sales_card = card_gen.generate_card(sales_data, "metric", "sales", verbose=True)
    print("\n=== 销售指标卡片 ===")
    print(f"卡片ID: {sales_card['card_id']}")
    print(f"有效期: {sales_card['valid_from']} 至 {sales_card['valid_until']}")
    print(f"处理时间: {sales_card['metadata'].get('processing_time_ms', 0)}ms")
    print(f"\n卡片内容:\n{sales_card['content']}")
    
    # 示例3：生成英文卡片
    print("\n" + "-"*80)
    print("示例3：生成英文卡片")
    print("-"*80)
    en_product_card = card_gen.generate_card(product_data, "entity", "product", language="en", verbose=True)
    print("\n=== English Product Card ===")
    print(f"Card ID: {en_product_card['card_id']}")
    print(f"Validity: {en_product_card['valid_from']} to {en_product_card['valid_until']}")
    print(f"Processing Time: {en_product_card['metadata'].get('processing_time_ms', 0)}ms")
    print(f"\nCard Content:\n{en_product_card['content']}")
    
    # 示例4：处理空值和货币转换
    print("\n" + "-"*80)
    print("示例4：处理空值和货币转换")
    print("-"*80)
    incomplete_data = {
        "product_id": "P54321",
        "product_name": "无线耳机",
        "category": None,
        "price": "$199.99",  # 测试货币转换
        "stock": 0,
        "description": "",
        "launch_date": "2023-05-20"
    }
    
    incomplete_card = card_gen.generate_card(incomplete_data, "entity", "product", verbose=True)
    print("\n=== 处理空值的卡片 ===")
    print(f"卡片ID: {incomplete_card['card_id']}")
    print(f"处理时间: {incomplete_card['metadata'].get('processing_time_ms', 0)}ms")
    print(f"\n卡片内容:\n{incomplete_card['content']}")
    
    # 示例5：创建向量存储和搜索
    print("\n" + "-"*80)
    print("示例5：创建向量存储和搜索")
    print("-"*80)
    print("\n=== 向量检索示例 ===")
    # 创建文档
    docs = [
        card_gen.create_document(product_card),
        card_gen.create_document(sales_card),
        card_gen.create_document(en_product_card),
        card_gen.create_document(incomplete_card)
    ]
    
    # 初始化向量存储
    card_gen.init_vector_store(docs, verbose=True)
    
    # 搜索相似卡片 - 示例1
    query = "高性能电脑"
    results = card_gen.search_similar_cards(query, k=2, verbose=True)
    
    print(f"\n搜索结果详情:")
    for i, (doc, score) in enumerate(results):
        print(f"\n相似度排名 #{i+1} (分数: {score:.4f})")
        print(f"卡片类型: {doc.metadata['card_type']} - {doc.metadata['subtype']}")
        print(f"语言: {doc.metadata['language']}")
        print(f"生成时间: {doc.metadata.get('generated_at', 'N/A')}")
        print(f"卡片内容预览: {doc.page_content[:100]}...")
    
    # 搜索相似卡片 - 示例2（带过滤）
    print("\n" + "="*60)
    print("示例6：带过滤条件的向量检索")
    print("="*60)
    query_filter = "电子产品"
    filter_condition = {"language": "zh", "subtype": "product"}
    filtered_results = card_gen.search_similar_cards(query_filter, k=2, filter=filter_condition, verbose=True)
    
    print(f"\n带过滤条件的搜索结果:")
    for i, (doc, score) in enumerate(filtered_results):
        print(f"\n相似度排名 #{i+1} (分数: {score:.4f})")
        print(f"卡片类型: {doc.metadata['card_type']} - {doc.metadata['subtype']}")
        print(f"语言: {doc.metadata['language']}")
        print(f"卡片内容预览: {doc.page_content[:100]}...")
    
    # 保存向量存储
    print("\n" + "="*60)
    print("示例7：保存向量存储")
    print("="*60)
    store_path = card_gen.save_vector_store("demo_vector_store")
    print(f"\n向量存储已成功保存至: {store_path}")
    
    # 计算程序总运行时间
    total_time = time.time() - program_start_time
    print("\n" + "="*80)
    print(f"程序执行完成")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"生成卡片总数: 4")
    print(f"创建向量存储: 1 (包含{len(docs)}个文档)")
    print(f"执行检索操作: 2")
    print("="*80)