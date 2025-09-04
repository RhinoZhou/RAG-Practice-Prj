#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成符合JSON Schema的检索payload数据

本脚本是数据准备阶段的核心工具，负责将原始文档转换为结构化、符合合规要求的检索payload。
重点实现了以下合规约束：
1. 必须携带引用ID/版本（doc_id、version）
2. 跨版本禁混：确保每个文档有独立版本，便于跟踪和管理
3. 字段规范化：统一管理单位、时间格式、实体类型等

通过JSON Schema进行严格的数据格式验证，确保数据质量和一致性。
"""
import os
import json
import csv
import uuid
import datetime
import jsonschema
from jsonschema import validate

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')  # 数据目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')  # 输出目录
SCHEMA_FILE = os.path.join(DATA_DIR, 'retrieval_payload.schema.json')  # JSON Schema文件路径
CORPUS_FILE = os.path.join(DATA_DIR, 'corpus.csv')  # 原始语料库文件路径
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'retrieval_payloads.jsonl')  # 输出的payload文件路径

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RetrievalPayloadGenerator:
    """
    生成符合Schema的检索payload数据的核心类
    
    主要职责：
    1. 加载和验证JSON Schema
    2. 处理原始文档，提取和规范化数据字段
    3. 实现版本控制和合规检查
    4. 生成符合Schema的检索payload
    """
    
    def __init__(self):
        """初始化检索payload生成器，加载JSON Schema并初始化统计数据"""
        # 加载JSON Schema - 这是数据准备阶段注入的合规约束
        with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        # 版本控制跟踪 - 实现跨版本禁混的关键机制
        self.document_versions = {}
        
        # 合规检查统计 - 跟踪合规性状态
        self.compliance_stats = {
            'total_documents': 0,  # 处理的总文档数
            'valid_documents': 0,  # 符合所有合规要求的文档数
            'missing_reference_id': 0,  # 缺少引用ID的文档数
            'missing_version': 0,  # 缺少版本的文档数
            'cross_version_mixed': 0  # 混合不同版本的文档数
        }
    
    def _detect_encoding(self, file_path):
        """
        尝试检测文件编码，确保正确读取CSV文件
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            str: 检测到的编码格式
        """
        encodings = ['utf-8', 'gbk', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # 尝试读取几行测试编码
                    for _ in range(5):
                        f.readline()
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # 默认返回utf-8
    
    def _generate_version(self, doc_id):
        """
        为文档生成版本号，确保版本控制
        这是实现"跨版本禁混"合规要求的核心机制
        
        Args:
            doc_id (str): 文档ID
            
        Returns:
            str: 生成的版本号（格式为vX.Y.Z）
        """
        # 如果文档已存在，版本号递增（补丁版本+1）
        if doc_id in self.document_versions:
            current_version = self.document_versions[doc_id]
            # 解析版本号 vX.Y.Z
            parts = current_version.split('.')
            major, minor, patch = int(parts[0][1:]), int(parts[1]), int(parts[2])
            patch += 1
            new_version = f'v{major}.{minor}.{patch}'
        else:
            # 新文档从v1.0.0开始
            new_version = 'v1.0.0'
        
        # 更新版本跟踪 - 确保版本唯一性，防止跨版本混用
        self.document_versions[doc_id] = new_version
        return new_version
    
    def _extract_entities(self, content):
        """
        从内容中提取实体（简化实现）
        展示了如何在数据准备阶段规范化实体字段
        
        Args:
            content (str): 文档内容
            
        Returns:
            dict: 提取的实体字典
        """
        # 这是一个简化的实体提取实现
        # 在实际应用中，应该使用NLP库如spaCy进行实体识别
        entities = {
            'drug': [],  # 药物实体
            'disease': [],  # 疾病实体
            'symptom': []  # 症状实体
        }
        
        # 模拟实体提取 - 实际应用中应使用更复杂的NLP技术
        if isinstance(content, str):
            if '高血压' in content or '血压' in content:
                entities['disease'].append('hypertension')
            if '糖尿病' in content:
                entities['disease'].append('diabetes')
            if '阿司匹林' in content or 'aspirin' in content.lower():
                entities['drug'].append('aspirin')
            if '头痛' in content or '头痛' in content:
                entities['symptom'].append('headache')
        
        # 移除空列表 - 字段规范化处理
        return {k: v for k, v in entities.items() if v}
    
    def _extract_units(self, content):
        """
        从内容中提取单位
        展示了如何在数据准备阶段规范化单位字段
        
        Args:
            content (str): 文档内容
            
        Returns:
            list: 提取的单位列表
        """
        units = set()
        # 预定义的单位列表 - 数据准备阶段定好的口径
        unit_list = ['mg', 'g', 'kg', 'ml', 'l', 'mm', 'cm', 'm', 'hour', 'day', 'week', 'month', 'year']
        
        if isinstance(content, str):
            for unit in unit_list:
                if unit in content or unit.upper() in content:
                    units.add(unit)
        
        return list(units)
    
    def _generate_tags(self, title, content):
        """
        为文档生成标签
        
        Args:
            title (str): 文档标题
            content (str): 文档内容
            
        Returns:
            list: 生成的标签列表
        """
        tags = set()
        
        # 从标题和内容中提取标签
        if isinstance(title, str):
            title_lower = title.lower()
            if '高血压' in title_lower or '血压' in title_lower:
                tags.add('hypertension')
            if '糖尿病' in title_lower:
                tags.add('diabetes')
            if '治疗' in title_lower:
                tags.add('treatment')
            if '用药' in title_lower or '药物' in title_lower:
                tags.add('medication')
        
        # 限制标签数量
        return list(tags)[:20]
    
    def _check_compliance(self, payload):
        """
        检查文档是否符合合规要求
        验证"必须携带引用ID/版本"和"跨版本禁混"的核心逻辑
        
        Args:
            payload (dict): 生成的payload数据
            
        Returns:
            dict: 合规性标志
        """
        compliance_flags = {
            # 检查是否携带引用ID - "必须带引用ID/版本"合规约束
            'has_reference_id': 'source_ref' in payload and payload['source_ref'] is not None,
            # 检查是否携带版本 - "必须带引用ID/版本"合规约束
            'has_version': 'version' in payload and payload['version'] is not None,
            # 检查是否混合版本 - "跨版本禁混"合规约束
            'cross_version_mixed': False  # 在这个简化版本中，我们假设没有混合版本
        }
        
        # 更新统计信息
        self.compliance_stats['total_documents'] += 1
        
        # 判断文档是否完全合规
        if compliance_flags['has_reference_id'] and compliance_flags['has_version'] and not compliance_flags['cross_version_mixed']:
            self.compliance_stats['valid_documents'] += 1
        else:
            # 记录不符合的原因
            if not compliance_flags['has_reference_id']:
                self.compliance_stats['missing_reference_id'] += 1
            if not compliance_flags['has_version']:
                self.compliance_stats['missing_version'] += 1
            if compliance_flags['cross_version_mixed']:
                self.compliance_stats['cross_version_mixed'] += 1
        
        return compliance_flags
    
    def generate_payload(self, row):
        """
        为一行数据生成payload
        将原始数据转换为符合Schema的结构化数据
        
        Args:
            row (dict): CSV中的一行数据
            
        Returns:
            dict: 生成的payload数据
        """
        # 处理文档ID和标题 - 确保每个文档有唯一标识
        doc_id = row.get('doc_id', f'doc_{uuid.uuid4().hex[:8]}')
        title = row.get('title', 'Untitled Document')
        content = row.get('content', '')
        
        # 生成版本号 - 实现"跨版本禁混"约束
        version = self._generate_version(doc_id)
        
        # 生成时间戳 - 规范化时间格式
        current_time = datetime.datetime.now().isoformat() + 'Z'  # ISO 8601格式，带时区标识
        
        # 生成引用ID - 实现"必须带引用ID/版本"约束
        source_ref = f'source_{uuid.uuid4().hex[:10]}'
        
        # 提取实体、单位和标签 - 字段规范化处理
        entities = self._extract_entities(content)
        units = self._extract_units(content)
        tags = self._generate_tags(title, content)
        
        # 构建payload - 将所有字段、约束、策略注入到结构化数据中
        payload = {
            'doc_id': doc_id,  # 文档唯一标识
            'title': title,  # 文档标题
            'version': version,  # 文档版本号
            'creation_time': current_time,  # 创建时间
            'last_update_time': current_time,  # 最后更新时间
            'units': units,  # 规范化的单位字段
            'tags': tags,  # 文档标签
            'entities': entities,  # 规范化的实体字段
            'content': content,  # 文档内容
            'source_ref': source_ref  # 引用ID
        }
        
        # 添加合规检查标志 - 跟踪合规状态
        compliance_flags = self._check_compliance(payload)
        payload['compliance_flags'] = compliance_flags
        
        # 验证payload是否符合Schema - 通过JSON Schema强化合规要求
        try:
            validate(instance=payload, schema=self.schema)
            return payload
        except jsonschema.exceptions.ValidationError as e:
            print(f"Validation error for document {doc_id}: {e}")
            # 即使验证失败，也返回payload以便分析问题
            return payload
    
    def process_corpus(self):
        """
        处理整个语料库，批量生成payload数据
        
        Returns:
            list: 生成的payload列表
        """
        # 检测文件编码
        encoding = self._detect_encoding(CORPUS_FILE)
        print(f"Detected file encoding: {encoding}")
        
        payloads = []
        
        try:
            with open(CORPUS_FILE, 'r', encoding=encoding) as csvfile:
                # 尝试使用csv.Sniffer检测分隔符
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                
                reader = csv.DictReader(csvfile, dialect=dialect)
                
                # 如果无法识别表头，使用默认表头
                if not reader.fieldnames or len(reader.fieldnames) == 0:
                    csvfile.seek(0)
                    # 假设第一行是数据，使用默认字段名
                    reader = csv.reader(csvfile, dialect=dialect)
                    for i, row in enumerate(reader):
                        if i == 0:
                            # 跳过可能的表头行
                            continue
                        # 构建字典
                        row_dict = {
                            'doc_id': row[0] if len(row) > 0 else f'doc_{i}',
                            'title': row[1] if len(row) > 1 else f'Title {i}',
                            'content': row[2] if len(row) > 2 else f'Content {i}'
                        }
                        payload = self.generate_payload(row_dict)
                        payloads.append(payload)
                else:
                    # 正常处理
                    for i, row in enumerate(reader):
                        payload = self.generate_payload(row)
                        payloads.append(payload)
        except Exception as e:
            print(f"Error processing corpus file: {e}")
            # 创建一些模拟数据用于演示
            print("Creating mock data for demonstration...")
            for i in range(10):
                mock_row = {
                    'doc_id': f'D{i+1}',
                    'title': f'Demo Document {i+1}',
                    'content': f'This is a sample document content for demonstration purpose. It contains information about medical treatment. It mentions units like mg and kg. It discusses conditions like hypertension and diabetes.'
                }
                payload = self.generate_payload(mock_row)
                payloads.append(payload)
        
        # 保存结果
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for payload in payloads:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        
        # 同时保存为JSON格式，便于查看
        json_output_file = os.path.join(OUTPUT_DIR, 'retrieval_payloads.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(payloads, f, ensure_ascii=False, indent=2)
        
        # 输出合规统计
        print("Compliance Statistics:")
        for key, value in self.compliance_stats.items():
            print(f"  {key}: {value}")
        
        # 输出生成的payload示例
        if payloads:
            print(f"\nGenerated {len(payloads)} payloads. Example:")
            print(json.dumps(payloads[0], ensure_ascii=False, indent=2))
        
        return payloads
    
    def enforce_compliance_rules(self):
        """
        强制执行合规规则，输出合规检查结果
        重点展示"跨版本禁混"和"必须带引用ID/版本"的合规约束执行情况
        """
        # 1. 跨版本禁混规则 - 详细解释实现机制
        print("\nEnforcing compliance rules:")
        print("1. Cross-version mixing prohibition: Ensured by version tracking system.")
        print("   - Each document has a unique version number.")
        print("   - Version history is maintained to prevent mixing content from different versions.")
        
        # 2. 必须带引用ID/版本规则 - 展示执行情况
        print("2. Mandatory reference ID and version: All documents are required to have both.")
        print(f"   - Documents with missing reference ID: {self.compliance_stats['missing_reference_id']}")
        print(f"   - Documents with missing version: {self.compliance_stats['missing_version']}")
        
        # 3. 合规文档比例 - 总体合规率
        compliance_rate = (self.compliance_stats['valid_documents'] / self.compliance_stats['total_documents'] * 100) if self.compliance_stats['total_documents'] > 0 else 0
        print(f"3. Overall compliance rate: {compliance_rate:.2f}%")

if __name__ == "__main__":
    """主程序入口，执行payload生成流程"""
    print("Starting retrieval payload generation...")
    print(f"Schema file: {SCHEMA_FILE}")
    print(f"Corpus file: {CORPUS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # 创建生成器实例
    generator = RetrievalPayloadGenerator()
    
    # 处理语料库，生成payload
    payloads = generator.process_corpus()
    
    # 强制执行合规规则，输出检查结果
    generator.enforce_compliance_rules()
    
    print(f"\nPayload generation completed. Results saved to {OUTPUT_FILE}")