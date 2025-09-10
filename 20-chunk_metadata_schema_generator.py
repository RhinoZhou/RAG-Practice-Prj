#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块元数据模板生成器

功能：
- 生成标准的分块JSON模板，字段包含角色、实体、时序、结构路径等
- 支持多维元数据标注体系，包括语义角色、知识图谱和时序信息
- 提供灵活的字段开关，可根据需求生成不同类型的元数据模板

使用示例：
```python
from chunk_metadata_schema_generator import ChunkMetadataGenerator

# 创建生成器实例
generator = ChunkMetadataGenerator()

# 生成完整的元数据模板
full_template = generator.generate_template(
    include_semantic_roles=True,
    include_knowledge_graph=True,
    include_temporal_info=True,
    include_structural_info=True
)

# 保存模板到文件
generator.save_template(full_template, "chunk_metadata_template.json")

# 生成简化的元数据模板
simple_template = generator.generate_template(
    include_semantic_roles=False,
    include_knowledge_graph=False,
    include_temporal_info=True,
    include_structural_info=True
)
```

依赖：
- Python 3.7+
- typing (标准库)
- json (标准库)
"""

import json
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from datetime import datetime


class ChunkMetadataGenerator:
    """分块元数据模板生成器
    
    该类用于生成标准化的文本分块元数据模板，支持多维元数据标注体系，
    包括语义角色标注、知识图谱嵌入、时序版本控制和结构路径信息等。
    """
    
    def __init__(self):
        """初始化分块元数据模板生成器"""
        # 设置默认值和示例数据
        self.default_values = {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "document_type": "legal_case",  # 法律案例、医学报告、技术文档等
            "chunk_text": "在本案例中，原告张三指控被告李四侵犯其知识产权...",
            "token_count": 128,
            "start_position": 0,
            "end_position": 128,
            "version": "v1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # 预定义的示例语义角色列表
        self.example_semantic_roles = [
            {"role_type": "原告", "role_value": "张三", "position": [10, 12]},
            {"role_type": "被告", "role_value": "李四", "position": [18, 20]},
            {"role_type": "行为", "role_value": "侵犯", "position": [22, 24]},
            {"role_type": "对象", "role_value": "知识产权", "position": [25, 29]},
            {"role_type": "时间", "role_value": "2023年10月15日", "position": [35, 45]},
            {"role_type": "地点", "role_value": "北京市海淀区人民法院", "position": [50, 63]},
        ]
        
        # 预定义的示例知识图谱关系
        self.example_knowledge_graph = {
            "entities": [
                {"entity_id": "e1", "entity_type": "自然人", "entity_value": "张三"},
                {"entity_id": "e2", "entity_type": "自然人", "entity_value": "李四"},
                {"entity_id": "e3", "entity_type": "知识产权", "entity_value": "专利A"}
            ],
            "relations": [
                {"source_id": "e1", "target_id": "e3", "relation_type": "拥有"},
                {"source_id": "e2", "target_id": "e3", "relation_type": "侵犯"}
            ]
        }
        
        # 预定义的示例时序信息
        self.example_temporal_info = {
            "effective_time": "2023-10-15T00:00:00Z",
            "expiry_time": "2033-10-15T00:00:00Z",
            "revision_history": [
                {"revision_id": "rev_001", "version": "v1.0", "timestamp": "2023-10-15T10:00:00Z", "author": "system"},
                {"revision_id": "rev_002", "version": "v1.1", "timestamp": "2023-10-16T15:30:00Z", "author": "editor", "changes": "修正了时间表述"}
            ],
            "time_expressions": [
                {"text": "2023年10月15日", "type": "日期", "value": "2023-10-15"},
                {"text": "三个月内", "type": "时间段", "value": "P3M"}
            ]
        }
        
        # 预定义的示例结构路径信息
        self.example_structural_info = {
            "document_structure": "root.section_1.subsection_1.paragraph_2",
            "hierarchy_level": 3,  # 层级深度
            "parent_chunk_id": "chunk_000",
            "child_chunk_ids": ["chunk_002", "chunk_003"],
            "sibling_order": 1,  # 在同级块中的顺序
            "layout_info": {"page_number": 5, "region": "main_content", "block_type": "paragraph"}
        }
        
        # 预定义的示例元数据标签
        self.example_metadata_tags = [
            {"tag_name": "法律领域", "tag_value": "知识产权", "confidence": 0.95},
            {"tag_name": "案件类型", "tag_value": "侵权诉讼", "confidence": 0.90},
            {"tag_name": "重要程度", "tag_value": "高", "confidence": 0.85},
            {"tag_name": "情感倾向", "tag_value": "中性", "confidence": 0.92}
        ]
    
    def generate_template(self, 
                         include_semantic_roles: bool = True, 
                         include_knowledge_graph: bool = True, 
                         include_temporal_info: bool = True, 
                         include_structural_info: bool = True, 
                         include_metadata_tags: bool = True, 
                         custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成分块元数据模板
        
        参数:
            include_semantic_roles: 是否包含语义角色标注
            include_knowledge_graph: 是否包含知识图谱嵌入
            include_temporal_info: 是否包含时序版本控制
            include_structural_info: 是否包含结构路径信息
            include_metadata_tags: 是否包含元数据标签
            custom_fields: 自定义字段，用于添加额外的元数据字段
            
        返回:
            包含指定元数据字段的分块元数据模板
        """
        # 创建基础模板
        template = self.default_values.copy()
        
        # 添加语义角色标注
        if include_semantic_roles:
            template["semantic_roles"] = self.example_semantic_roles
        
        # 添加知识图谱嵌入
        if include_knowledge_graph:
            template["knowledge_graph"] = self.example_knowledge_graph
        
        # 添加时序版本控制
        if include_temporal_info:
            template["temporal_info"] = self.example_temporal_info
        
        # 添加结构路径信息
        if include_structural_info:
            template["structural_info"] = self.example_structural_info
        
        # 添加元数据标签
        if include_metadata_tags:
            template["metadata_tags"] = self.example_metadata_tags
        
        # 添加自定义字段
        if custom_fields:
            template.update(custom_fields)
        
        return template
    
    def save_template(self, template: Dict[str, Any], file_path: str, indent: int = 2) -> None:
        """
        保存元数据模板到文件
        
        参数:
            template: 要保存的元数据模板
            file_path: 保存文件路径
            indent: JSON格式化缩进
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=indent)
            print(f"元数据模板已成功保存到: {file_path}")
        except Exception as e:
            print(f"保存元数据模板时出错: {e}")
    
    def batch_generate_templates(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """
        批量生成多个元数据模板
        
        参数:
            count: 要生成的模板数量
            **kwargs: 传递给generate_template的参数
            
        返回:
            元数据模板列表
        """
        templates = []
        for i in range(count):
            # 为每个模板生成唯一ID
            custom_fields = {
                "chunk_id": f"chunk_{i+1:03d}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            # 如果用户提供了自定义字段，合并它们
            if kwargs.get("custom_fields"):
                custom_fields.update(kwargs["custom_fields"])
            kwargs["custom_fields"] = custom_fields
            
            # 生成模板
            template = self.generate_template(**kwargs)
            templates.append(template)
        
        return templates
    
    def validate_template(self, template: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证元数据模板的基本结构是否符合要求
        
        参数:
            template: 要验证的元数据模板
            
        返回:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必需字段
        required_fields = ["chunk_id", "document_id", "chunk_text", "token_count", "start_position", "end_position"]
        for field in required_fields:
            if field not in template:
                errors.append(f"缺少必需字段: {field}")
        
        # 检查字段类型
        if "token_count" in template and not isinstance(template["token_count"], int):
            errors.append("token_count必须是整数类型")
        
        if "start_position" in template and not isinstance(template["start_position"], int):
            errors.append("start_position必须是整数类型")
        
        if "end_position" in template and not isinstance(template["end_position"], int):
            errors.append("end_position必须是整数类型")
        
        # 检查位置逻辑
        if "start_position" in template and "end_position" in template:
            if template["start_position"] < 0:
                errors.append("start_position不能为负数")
            if template["end_position"] <= template["start_position"]:
                errors.append("end_position必须大于start_position")
        
        # 检查语义角色格式
        if "semantic_roles" in template:
            if not isinstance(template["semantic_roles"], list):
                errors.append("semantic_roles必须是列表类型")
            else:
                for role in template["semantic_roles"]:
                    if not isinstance(role, dict):
                        errors.append("语义角色项必须是字典类型")
                    elif "role_type" not in role or "role_value" not in role:
                        errors.append("语义角色项缺少必需字段: role_type或role_value")
        
        return len(errors) == 0, errors
    
    def generate_json_schema(self) -> Dict[str, Any]:
        """
        生成元数据模板的JSON Schema定义，用于验证和文档化
        
        返回:
            JSON Schema定义
        """
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "分块元数据模板",
            "description": "用于描述文本分块的多维元数据结构",
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "分块唯一标识符"
                },
                "document_id": {
                    "type": "string",
                    "description": "所属文档唯一标识符"
                },
                "document_type": {
                    "type": "string",
                    "description": "文档类型，如法律案例、医学报告等"
                },
                "chunk_text": {
                    "type": "string",
                    "description": "分块的文本内容"
                },
                "token_count": {
                    "type": "integer",
                    "description": "分块的token数量",
                    "minimum": 1
                },
                "start_position": {
                    "type": "integer",
                    "description": "分块在原始文档中的起始位置",
                    "minimum": 0
                },
                "end_position": {
                    "type": "integer",
                    "description": "分块在原始文档中的结束位置",
                    "minimum": 1
                },
                "version": {
                    "type": "string",
                    "description": "分块版本号"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "创建时间"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "更新时间"
                },
                "semantic_roles": {
                    "type": "array",
                    "description": "语义角色标注列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role_type": {
                                "type": "string",
                                "description": "角色类型，如原告、被告、时间、地点等"
                            },
                            "role_value": {
                                "type": "string",
                                "description": "角色值"
                            },
                            "position": {
                                "type": "array",
                                "description": "角色在文本中的位置范围",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "required": ["role_type", "role_value"]
                    }
                },
                "knowledge_graph": {
                    "type": "object",
                    "description": "知识图谱嵌入信息",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "description": "实体列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity_id": {
                                        "type": "string",
                                        "description": "实体唯一标识符"
                                    },
                                    "entity_type": {
                                        "type": "string",
                                        "description": "实体类型，如自然人、知识产权等"
                                    },
                                    "entity_value": {
                                        "type": "string",
                                        "description": "实体值"
                                    }
                                },
                                "required": ["entity_id", "entity_type", "entity_value"]
                            }
                        },
                        "relations": {
                            "type": "array",
                            "description": "实体关系列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {
                                        "type": "string",
                                        "description": "源实体ID"
                                    },
                                    "target_id": {
                                        "type": "string",
                                        "description": "目标实体ID"
                                    },
                                    "relation_type": {
                                        "type": "string",
                                        "description": "关系类型，如拥有、侵犯等"
                                    }
                                },
                                "required": ["source_id", "target_id", "relation_type"]
                            }
                        }
                    },
                    "required": ["entities", "relations"]
                },
                "temporal_info": {
                    "type": "object",
                    "description": "时序版本控制信息",
                    "properties": {
                        "effective_time": {
                            "type": "string",
                            "format": "date-time",
                            "description": "生效时间"
                        },
                        "expiry_time": {
                            "type": "string",
                            "format": "date-time",
                            "description": "过期时间"
                        },
                        "revision_history": {
                            "type": "array",
                            "description": "修订历史",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "revision_id": {
                                        "type": "string",
                                        "description": "修订唯一标识符"
                                    },
                                    "version": {
                                        "type": "string",
                                        "description": "版本号"
                                    },
                                    "timestamp": {
                                        "type": "string",
                                        "format": "date-time",
                                        "description": "修订时间戳"
                                    },
                                    "author": {
                                        "type": "string",
                                        "description": "修订作者"
                                    },
                                    "changes": {
                                        "type": "string",
                                        "description": "变更说明"
                                    }
                                },
                                "required": ["revision_id", "version", "timestamp"]
                            }
                        },
                        "time_expressions": {
                            "type": "array",
                            "description": "文本中的时间表达式",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "时间表达式文本"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "时间表达式类型，如日期、时间段"
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "时间表达式标准化值"
                                    }
                                },
                                "required": ["text", "type", "value"]
                            }
                        }
                    }
                },
                "structural_info": {
                    "type": "object",
                    "description": "结构路径信息",
                    "properties": {
                        "document_structure": {
                            "type": "string",
                            "description": "文档结构路径"
                        },
                        "hierarchy_level": {
                            "type": "integer",
                            "description": "层级深度",
                            "minimum": 0
                        },
                        "parent_chunk_id": {
                            "type": "string",
                            "description": "父分块ID"
                        },
                        "child_chunk_ids": {
                            "type": "array",
                            "description": "子分块ID列表",
                            "items": {"type": "string"}
                        },
                        "sibling_order": {
                            "type": "integer",
                            "description": "在同级块中的顺序",
                            "minimum": 0
                        },
                        "layout_info": {
                            "type": "object",
                            "description": "布局信息",
                            "properties": {
                                "page_number": {
                                    "type": "integer",
                                    "description": "页码",
                                    "minimum": 1
                                },
                                "region": {
                                    "type": "string",
                                    "description": "区域类型"
                                },
                                "block_type": {
                                    "type": "string",
                                    "description": "块类型，如paragraph、table等"
                                }
                            }
                        }
                    }
                },
                "metadata_tags": {
                    "type": "array",
                    "description": "元数据标签列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tag_name": {
                                "type": "string",
                                "description": "标签名称"
                            },
                            "tag_value": {
                                "type": "string",
                                "description": "标签值"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "置信度",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["tag_name", "tag_value"]
                    }
                }
            },
            "required": ["chunk_id", "document_id", "chunk_text", "token_count", "start_position", "end_position"]
        }
        
        return schema


# 示例用法
if __name__ == "__main__":
    # 创建生成器实例
    generator = ChunkMetadataGenerator()
    
    print("===== 分块元数据模板生成器 =====")
    
    # 生成完整的元数据模板
    print("\n1. 生成完整的元数据模板...")
    full_template = generator.generate_template(
        include_semantic_roles=True,
        include_knowledge_graph=True,
        include_temporal_info=True,
        include_structural_info=True,
        include_metadata_tags=True
    )
    
    # 保存完整模板到文件
    generator.save_template(full_template, "full_chunk_metadata_template.json")
    
    # 生成简化的元数据模板
    print("\n2. 生成简化的元数据模板...")
    simple_template = generator.generate_template(
        include_semantic_roles=False,
        include_knowledge_graph=False,
        include_temporal_info=True,
        include_structural_info=True,
        include_metadata_tags=False,
        custom_fields={"project_id": "proj_001"}
    )
    
    # 保存简化模板到文件
    generator.save_template(simple_template, "simple_chunk_metadata_template.json")
    
    # 批量生成多个模板
    print("\n3. 批量生成5个元数据模板...")
    batch_templates = generator.batch_generate_templates(
        count=5,
        include_semantic_roles=True,
        include_knowledge_graph=False,
        include_temporal_info=True,
        include_structural_info=True,
        include_metadata_tags=True
    )
    
    # 保存批量模板到文件
    generator.save_template(batch_templates, "batch_chunk_metadata_templates.json")
    
    # 生成JSON Schema
    print("\n4. 生成JSON Schema定义...")
    json_schema = generator.generate_json_schema()
    generator.save_template(json_schema, "chunk_metadata_schema.json")
    
    # 验证模板
    print("\n5. 验证生成的模板...")
    is_valid, errors = generator.validate_template(full_template)
    if is_valid:
        print("  模板验证通过！")
    else:
        print(f"  模板验证失败：{errors}")
    
    print("\n===== 任务完成 =====")
    print("生成的文件：")
    print("- full_chunk_metadata_template.json: 完整的元数据模板")
    print("- simple_chunk_metadata_template.json: 简化的元数据模板")
    print("- batch_chunk_metadata_templates.json: 批量生成的元数据模板")
    print("- chunk_metadata_schema.json: 元数据模板的JSON Schema定义")