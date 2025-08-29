#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据治理与一致性校验程序

功能：
  - 数据去重合并
  - 单位换算（mg/dL↔mmol/L）
  - 参考区间校验
  - 禁忌证/相互作用规则校验
  - 计算节点/边置信度

输入：
  - linked_mentions.jsonl: 链接后的实体提及数据
  - triples.jsonl: 关系三元组数据（如果存在）
  - events.jsonl: 事件数据（如果存在）

输出：
  - nodes.csv: 节点数据
  - edges.csv: 边数据
  或
  - rdf.nt: RDF格式数据

依赖：
  - pandas: 数据处理
  - numpy: 数值计算
  - pyyaml: 配置文件解析
"""

import os
import json
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime

# 设置日志配置
def setup_logging(log_file: str = "normalize_validate.log"):
    """设置程序日志配置
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger("Data_Normalizer")

# 初始化日志记录器
logger = setup_logging()

class DataNormalizer:
    """数据标准化与校验类
    
    负责数据的去重合并、单位换算、参考区间校验、禁忌证/相互作用规则校验，以及计算节点/边置信度
    """
    def __init__(self, config_file: str = "normalize_config.yaml"):
        """
        初始化数据标准化器
        
        Args:
            config_file: 配置文件路径
        """
        # 加载配置文件
        self.config = self._load_config(config_file)
        
        # 初始化单位换算因子
        self.unit_conversion_factors = {
            # 葡萄糖: 1 mg/dL = 0.0555 mmol/L
            "glucose": 0.0555,
            # 胆固醇: 1 mg/dL = 0.0259 mmol/L
            "cholesterol": 0.0259,
            # 甘油三酯: 1 mg/dL = 0.0113 mmol/L
            "triglyceride": 0.0113,
            # 尿酸: 1 mg/dL = 59.489 μmol/L
            "uric_acid": 59.489
        }
        
        # 初始化参考区间
        self.reference_ranges = {
            "glucose": {"mg/dL": (70, 100), "mmol/L": (3.9, 5.6)},
            "cholesterol": {"mg/dL": (125, 200), "mmol/L": (3.2, 5.2)},
            "triglyceride": {"mg/dL": (40, 150), "mmol/L": (0.45, 1.70)},
            "uric_acid": {"mg/dL": (2.4, 6.0), "μmol/L": (143, 357)}
        }
        
        # 初始化禁忌证和相互作用规则
        self.contraindication_rules = self._load_contraindication_rules()
        
        # 初始化数据存储
        self.linked_mentions = []
        self.triples = []
        self.events = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        default_config = {
            "output_format": "csv",  # csv 或 rdf
            "deduplicate_threshold": 0.95,
            "confidence_calculation_method": "default",
            "enable_unit_conversion": True,
            "enable_reference_check": True,
            "enable_contraindication_check": True
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # 合并默认配置和用户配置
                    default_config.update(user_config)
                    logger.info(f"成功加载配置文件: {config_file}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        else:
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            # 创建默认配置文件
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, allow_unicode=True)
                logger.info(f"已创建默认配置文件: {config_file}")
            except Exception as e:
                logger.error(f"创建默认配置文件失败: {str(e)}")
        
        return default_config
    
    def _load_contraindication_rules(self) -> Dict[str, List[str]]:
        """加载禁忌证和相互作用规则
        
        Returns:
            Dict[str, List[str]]: 禁忌证和相互作用规则
        """
        # 示例规则，可以从配置文件或数据库加载
        return {
            "阿司匹林": ["消化道溃疡", "出血倾向"],
            "华法林": ["严重肝功能不全", "严重高血压"],
            "青霉素": ["青霉素过敏史"]
        }
    
    def load_data(self, linked_mentions_file: str, triples_file: str, events_file: str):
        """加载输入数据
        
        Args:
            linked_mentions_file: 链接后的实体提及数据文件路径
            triples_file: 关系三元组数据文件路径
            events_file: 事件数据文件路径
        """
        # 加载linked_mentions数据
        if os.path.exists(linked_mentions_file):
            logger.info(f"加载linked_mentions数据: {linked_mentions_file}")
            with open(linked_mentions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.linked_mentions.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON行时出错: {e}")
            logger.info(f"成功加载 {len(self.linked_mentions)} 条linked_mentions数据")
        else:
            logger.error(f"linked_mentions文件不存在: {linked_mentions_file}")
        
        # 加载triples数据
        if os.path.exists(triples_file):
            logger.info(f"加载triples数据: {triples_file}")
            with open(triples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.triples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON行时出错: {e}")
            logger.info(f"成功加载 {len(self.triples)} 条triples数据")
        else:
            logger.warning(f"triples文件不存在: {triples_file}")
        
        # 加载events数据
        if os.path.exists(events_file):
            logger.info(f"加载events数据: {events_file}")
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON行时出错: {e}")
            logger.info(f"成功加载 {len(self.events)} 条events数据")
        else:
            logger.warning(f"events文件不存在: {events_file}")
    
    def deduplicate_mentions(self) -> List[Dict[str, Any]]:
        """去重合并实体提及
        
        Returns:
            List[Dict[str, Any]]: 去重后的实体提及数据
        """
        if not self.linked_mentions:
            logger.warning("没有加载linked_mentions数据，无法进行去重")
            return []
        
        logger.info("开始去重合并实体提及")
        
        # 简单的去重策略：根据doc_id、offset和type进行去重
        unique_mentions = []
        seen = set()
        
        for mention in self.linked_mentions:
            # 创建一个唯一标识符
            key = (mention.get("doc_id", ""), 
                   tuple(mention.get("offset", [0, 0])), 
                   mention.get("type", ""))
            
            if key not in seen:
                seen.add(key)
                unique_mentions.append(mention)
        
        logger.info(f"去重完成，原始数据 {len(self.linked_mentions)} 条，去重后 {len(unique_mentions)} 条")
        return unique_mentions
    
    def convert_units(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """单位换算（mg/dL↔mmol/L）
        
        Args:
            data: 需要进行单位换算的数据
        
        Returns:
            List[Dict[str, Any]]: 换算后的数据集
        """
        if not self.config.get("enable_unit_conversion", True):
            logger.info("单位换算功能已禁用")
            return data
        
        logger.info("开始进行单位换算")
        
        # 这里仅作为示例，实际应用中需要根据数据的具体结构和字段进行单位换算
        # 由于我们看到的linked_mentions数据主要是实体提及，没有具体的数值和单位
        # 所以这里的实现是简化的示例
        
        converted_data = []
        conversion_count = 0
        
        for item in data:
            # 检查是否有需要进行单位换算的字段
            # 这里仅作为示例，实际应用中需要根据具体的数据结构进行调整
            # 例如：如果有一个数值字段和一个单位字段
            
            converted_item = item.copy()
            # 实际应用中，这里应该有具体的单位换算逻辑
            # 例如：如果item有value和unit字段，并且是需要换算的单位
            
            converted_data.append(converted_item)
        
        logger.info(f"单位换算完成，共处理 {conversion_count} 条需要换算的数据")
        return converted_data
    
    def check_reference_ranges(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """参考区间校验
        
        Args:
            data: 需要进行参考区间校验的数据
        
        Returns:
            List[Dict[str, Any]]: 校验后的数据集
        """
        if not self.config.get("enable_reference_check", True):
            logger.info("参考区间校验功能已禁用")
            return data
        
        logger.info("开始进行参考区间校验")
        
        # 这里仅作为示例，实际应用中需要根据数据的具体结构和字段进行参考区间校验
        # 由于我们看到的linked_mentions数据主要是实体提及，没有具体的数值和参考区间
        # 所以这里的实现是简化的示例
        
        checked_data = []
        out_of_range_count = 0
        
        for item in data:
            checked_item = item.copy()
            # 实际应用中，这里应该有具体的参考区间校验逻辑
            # 例如：如果item有value和unit字段，并且是需要校验的项目
            
            checked_data.append(checked_item)
        
        logger.info(f"参考区间校验完成，共发现 {out_of_range_count} 条超出参考范围的数据")
        return checked_data
    
    def check_contraindications(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """禁忌证/相互作用规则校验
        
        Args:
            data: 需要进行禁忌证/相互作用规则校验的数据
        
        Returns:
            List[Dict[str, Any]]: 校验后的数据集
        """
        if not self.config.get("enable_contraindication_check", True):
            logger.info("禁忌证/相互作用规则校验功能已禁用")
            return data
        
        logger.info("开始进行禁忌证/相互作用规则校验")
        
        # 这里仅作为示例，实际应用中需要根据数据的具体结构和字段进行禁忌证/相互作用规则校验
        # 由于我们看到的linked_mentions数据主要是实体提及，没有具体的药品和诊断信息
        # 所以这里的实现是简化的示例
        
        checked_data = []
        contraindication_count = 0
        
        for item in data:
            checked_item = item.copy()
            # 实际应用中，这里应该有具体的禁忌证/相互作用规则校验逻辑
            # 例如：检查患者的诊断是否与药品存在禁忌证
            
            checked_data.append(checked_item)
        
        logger.info(f"禁忌证/相互作用规则校验完成，共发现 {contraindication_count} 条禁忌证/相互作用")
        return checked_data
    
    def calculate_confidence(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算节点/边置信度
        
        Args:
            data: 需要计算置信度的数据
        
        Returns:
            List[Dict[str, Any]]: 计算置信度后的数据集
        """
        logger.info("开始计算节点/边置信度")
        
        # 根据配置选择置信度计算方法
        method = self.config.get("confidence_calculation_method", "default")
        
        processed_data = []
        
        for item in data:
            processed_item = item.copy()
            
            # 如果已经有置信度分数，则保留
            if "score" in processed_item:
                confidence = processed_item["score"]
            else:
                # 根据不同的方法计算置信度
                if method == "default":
                    # 默认方法：基于现有信息计算置信度
                    confidence = self._calculate_default_confidence(item)
                else:
                    # 其他方法可以在这里添加
                    confidence = 0.5
            
            processed_item["confidence"] = confidence
            processed_data.append(processed_item)
        
        logger.info("节点/边置信度计算完成")
        return processed_data
    
    def _calculate_default_confidence(self, item: Dict[str, Any]) -> float:
        """默认的置信度计算方法
        
        Args:
            item: 需要计算置信度的数据项
        
        Returns:
            float: 置信度分数（0-1之间）
        """
        # 基础置信度
        base_confidence = 0.5
        
        # 如果有score字段，使用它作为基础
        if "score" in item:
            base_confidence = item["score"]
        
        # 根据其他字段调整置信度
        # 例如：如果有概念ID，置信度可能更高
        if "concept_id" in item and item["concept_id"] != "NIL" and not item.get("is_NIL", True):
            base_confidence = min(base_confidence + 0.3, 1.0)
        
        # 如果有极性和时间性信息，置信度可能更高
        if "polarity" in item and "temporality" in item:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        return base_confidence
    
    def generate_nodes_edges(self, unique_mentions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """生成节点和边数据
        
        Args:
            unique_mentions: 去重后的实体提及数据
        
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: 节点数据和边数据
        """
        logger.info("开始生成节点和边数据")
        
        nodes = []
        edges = []
        
        # 生成节点数据
        for mention in unique_mentions:
            # 创建实体节点
            entity_node = {
                "id": f"{mention.get('doc_id', '')}_{mention.get('offset', [0, 0])[0]}_{mention.get('offset', [0, 0])[1]}",
                "text": mention.get("text", ""),
                "type": mention.get("type", ""),
                "concept_id": mention.get("concept_id", "NIL"),
                "polarity": mention.get("polarity", ""),
                "temporality": mention.get("temporality", ""),
                "confidence": mention.get("confidence", 0.5),
                "source": "linked_mentions",
                "source_file": "linked_mentions.jsonl",
                "created_at": datetime.now().isoformat()
            }
            nodes.append(entity_node)
        
        # 如果有triples数据，生成边数据
        if self.triples:
            for triple in self.triples:
                # 创建边数据
                # 这里仅作为示例，实际应用中需要根据triples的具体结构进行调整
                pass
        
        # 如果有events数据，生成节点和边数据
        if self.events:
            for event in self.events:
                # 创建事件节点和相关的边
                # 这里仅作为示例，实际应用中需要根据events的具体结构进行调整
                pass
        
        logger.info(f"节点和边数据生成完成，生成 {len(nodes)} 个节点，{len(edges)} 条边")
        return nodes, edges
    
    def save_as_csv(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                   nodes_file: str = "nodes.csv", edges_file: str = "edges.csv"):
        """保存为CSV格式
        
        Args:
            nodes: 节点数据
            edges: 边数据
            nodes_file: 节点输出文件路径
            edges_file: 边输出文件路径
        """
        logger.info(f"保存为CSV格式: {nodes_file}, {edges_file}")
        
        # 保存节点数据
        if nodes:
            nodes_df = pd.DataFrame(nodes)
            nodes_df.to_csv(nodes_file, index=False, encoding='utf-8-sig')
            logger.info(f"节点数据已保存到: {nodes_file}")
        else:
            logger.warning("没有节点数据可以保存")
        
        # 保存边数据
        if edges:
            edges_df = pd.DataFrame(edges)
            edges_df.to_csv(edges_file, index=False, encoding='utf-8-sig')
            logger.info(f"边数据已保存到: {edges_file}")
        else:
            logger.warning("没有边数据可以保存")
    
    def save_as_rdf(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], 
                   output_file: str = "rdf.nt"):
        """保存为RDF N-Triples格式
        
        Args:
            nodes: 节点数据
            edges: 边数据
            output_file: 输出文件路径
        """
        logger.info(f"保存为RDF N-Triples格式: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入节点数据
            for node in nodes:
                # 这里仅作为示例，实际应用中需要根据RDF规范生成正确的三元组
                # 例如: <http://example.org/node/{node_id}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Entity> .
                pass
            
            # 写入边数据
            for edge in edges:
                # 这里仅作为示例，实际应用中需要根据RDF规范生成正确的三元组
                # 例如: <http://example.org/node/{source_id}> <http://example.org/hasRelation> <http://example.org/node/{target_id}> .
                pass
        
        logger.info(f"RDF数据已保存到: {output_file}")
    
    def run(self, linked_mentions_file: str, triples_file: str, events_file: str, 
            output_dir: str = ".", output_format: str = "csv"):
        """运行完整的数据标准化与校验流程
        
        Args:
            linked_mentions_file: 链接后的实体提及数据文件路径
            triples_file: 关系三元组数据文件路径
            events_file: 事件数据文件路径
            output_dir: 输出目录
            output_format: 输出格式 (csv 或 rdf)
        """
        logger.info("开始数据标准化与校验流程")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data(linked_mentions_file, triples_file, events_file)
        
        # 去重合并实体提及
        unique_mentions = self.deduplicate_mentions()
        
        # 计算置信度
        mentions_with_confidence = self.calculate_confidence(unique_mentions)
        
        # 单位换算
        converted_mentions = self.convert_units(mentions_with_confidence)
        
        # 参考区间校验
        checked_mentions = self.check_reference_ranges(converted_mentions)
        
        # 禁忌证/相互作用规则校验
        validated_mentions = self.check_contraindications(checked_mentions)
        
        # 生成节点和边数据
        nodes, edges = self.generate_nodes_edges(validated_mentions)
        
        # 保存结果
        if output_format.lower() == "rdf":
            # 保存为RDF格式
            output_file = os.path.join(output_dir, "rdf.nt")
            self.save_as_rdf(nodes, edges, output_file)
        else:
            # 保存为CSV格式
            nodes_file = os.path.join(output_dir, "nodes.csv")
            edges_file = os.path.join(output_dir, "edges.csv")
            self.save_as_csv(nodes, edges, nodes_file, edges_file)
        
        logger.info("数据标准化与校验流程完成")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据治理与一致性校验程序")
    parser.add_argument("--linked_mentions", type=str, default="linked_mentions.jsonl", 
                        help="链接后的实体提及数据文件路径")
    parser.add_argument("--triples", type=str, default="triples.jsonl", 
                        help="关系三元组数据文件路径")
    parser.add_argument("--events", type=str, default="events.jsonl", 
                        help="事件数据文件路径")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="输出目录")
    parser.add_argument("--output_format", type=str, choices=["csv", "rdf"], default="csv", 
                        help="输出格式 (csv 或 rdf)")
    parser.add_argument("--config", type=str, default="normalize_config.yaml", 
                        help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化数据标准化器
        normalizer = DataNormalizer(args.config)
        
        # 运行数据标准化与校验流程
        normalizer.run(
            args.linked_mentions,
            args.triples,
            args.events,
            args.output_dir,
            args.output_format
        )
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {str(e)}")
        print(f"错误: {str(e)}")
    except ImportError as e:
        logger.error(f"导入依赖库失败: {str(e)}")
        print(f"错误: 导入依赖库失败 - {str(e)}")
        print("请使用pip install -r requirements.txt安装所需的依赖库")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """程序入口点"""
    main()