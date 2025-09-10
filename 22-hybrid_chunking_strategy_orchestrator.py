#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合分块策略编排器

功能：
- 将滑动窗口、结构边界、主题检测、实体密度等信号编排为可配置策略
- 按文档类型套用不同的分块模板（合同/论文/财报/客服）
- 实现多信号融合优先级与冲突解决机制
- 输出统一的块结构与策略执行日志

使用示例：
```python
from hybrid_chunking_strategy_orchestrator import HybridChunkingOrchestrator

# 创建编排器实例
orchestrator = HybridChunkingOrchestrator()

# 分块处理文档
text = "你的文档内容..."
result = orchestrator.orchestrate(text, doc_type="论文")

# 输出结果
print("分块结果:", result['chunks'])
print("策略日志:", result['strategy_log'])
```

依赖：
- Python 3.7+
- typing (标准库)
- re (标准库)
- json (标准库)
- collections (标准库)
- datetime (标准库)
- concurrent.futures (标准库)
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class ChunkMetadata:
    """分块元数据类"""
    def __init__(self, chunk_id: str, start_pos: int, end_pos: int):
        self.chunk_id = chunk_id            # 块唯一ID
        self.start_pos = start_pos          # 起始位置
        self.end_pos = end_pos              # 结束位置
        self.content: str = ""              # 块内容
        self.chunk_type: str = "general"    # 块类型
        self.confidence: float = 0.0        # 分块置信度
        self.boundary_signals: Dict[str, float] = {}  # 边界信号来源
        self.structure_path: List[str] = [] # 结构路径
        self.entity_chain: List[str] = []   # 实体链
        self.topic_tags: List[str] = []     # 主题标签
        self.created_at = datetime.now().isoformat()  # 创建时间
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "chunk_id": self.chunk_id,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "confidence": self.confidence,
            "boundary_signals": self.boundary_signals,
            "structure_path": self.structure_path,
            "entity_chain": self.entity_chain,
            "topic_tags": self.topic_tags,
            "created_at": self.created_at
        }


class StrategyTemplate:
    """策略模板类，定义不同文档类型的分块策略"""
    def __init__(self, template_name: str):
        self.template_name = template_name  # 模板名称
        self.params: Dict[str, Any] = {}  # 模板参数
        self.boundary_priority: List[str] = []  # 边界优先级
        self.conflict_resolution: Dict[str, str] = {}  # 冲突解决规则
        self.signal_processors: Dict[str, Dict[str, Any]] = {}  # 信号处理器配置
        
        # 根据模板名称加载预设配置
        self._load_preset_config()
    
    def _load_preset_config(self) -> None:
        """加载预设配置"""
        if self.template_name == "合同":
            self.params = {
                "base_chunk_size": 300,
                "min_chunk_size": 150,
                "max_chunk_size": 500,
                "overlap": 50,
                "sentence_separators": ["。", "！", "？", ".", "!", "?"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"]
            }
            self.boundary_priority = ["contract_section", "paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "contract_section_vs_paragraph": "contract_section",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "contract_section": {"enabled": True, "regex": r'第[一二三四五六七八九十]+条'},  # 合同条款模式
                "paragraph": {"enabled": True},  # 段落边界
                "sentence": {"enabled": True},  # 句子边界
                "punctuation": {"enabled": True},  # 标点符号
                "token_count": {"enabled": True}  # 令牌计数
            }
        
        elif self.template_name == "论文":
            self.params = {
                "base_chunk_size": 400,
                "min_chunk_size": 200,
                "max_chunk_size": 600,
                "overlap": 60,
                "sentence_separators": ["。", "！", "？", ".", "!", "?"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"],
                "heading_patterns": ["#+", "第.+章", "1\\.", "2\\."]
            }
            self.boundary_priority = ["heading", "paragraph", "sentence", "token_count"]
            self.conflict_resolution = {
                "heading_vs_paragraph": "heading",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "heading": {"enabled": True, "patterns": self.params["heading_patterns"]},  # 标题检测
                "paragraph": {"enabled": True},  # 段落边界
                "sentence": {"enabled": True},  # 句子边界
                "token_count": {"enabled": True}  # 令牌计数
            }
        
        elif self.template_name == "财报":
            self.params = {
                "base_chunk_size": 250,
                "min_chunk_size": 100,
                "max_chunk_size": 400,
                "overlap": 40,
                "sentence_separators": ["。", "！", "？", ".", "!", "?", ";"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"],
                "table_patterns": ["(\\d+\\.\\d+)", "(\\d+,\\d+)"]
            }
            self.boundary_priority = ["table", "paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "table_vs_paragraph": "table",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "table": {"enabled": True, "patterns": self.params["table_patterns"]},  # 表格检测
                "paragraph": {"enabled": True},  # 段落边界
                "sentence": {"enabled": True},  # 句子边界
                "punctuation": {"enabled": True},  # 标点符号
                "token_count": {"enabled": True}  # 令牌计数
            }
        
        elif self.template_name == "客服":
            self.params = {
                "base_chunk_size": 200,
                "min_chunk_size": 50,
                "max_chunk_size": 300,
                "overlap": 30,
                "dialog_turn_patterns": ["用户:", "客服:", "顾客:", "员工:", "提问:", "回答:"]
            }
            self.boundary_priority = ["dialog_turn", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "dialog_turn_vs_sentence": "dialog_turn"
            }
            self.signal_processors = {
                "dialog_turn": {"enabled": True, "patterns": self.params["dialog_turn_patterns"]},  # 对话轮次检测
                "sentence": {"enabled": True},  # 句子边界
                "punctuation": {"enabled": True},  # 标点符号
                "token_count": {"enabled": True}  # 令牌计数
            }
        
        else:  # 默认通用模板
            self.params = {
                "base_chunk_size": 350,
                "min_chunk_size": 150,
                "max_chunk_size": 550,
                "overlap": 50,
                "sentence_separators": ["。", "！", "？", ".", "!", "?"]
            }
            self.boundary_priority = ["paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "paragraph": {"enabled": True},  # 段落边界
                "sentence": {"enabled": True},  # 句子边界
                "punctuation": {"enabled": True},  # 标点符号
                "token_count": {"enabled": True}  # 令牌计数
            }
    
    def override_params(self, params: Dict[str, Any]) -> None:
        """覆盖模板参数"""
        self.params.update(params)
        # 如果有新的特殊参数，更新信号处理器配置
        if "heading_patterns" in params and self.template_name == "论文":
            self.signal_processors["heading"]["patterns"] = params["heading_patterns"]
        if "dialog_turn_patterns" in params and self.template_name == "客服":
            self.signal_processors["dialog_turn"]["patterns"] = params["dialog_turn_patterns"]


class BoundarySignalDetector:
    """边界信号检测器"""
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 策略模板
        
    def detect_contract_sections(self, text: str) -> List[Tuple[int, int, float]]:
        """检测合同条款边界"""
        if not self.template.signal_processors.get("contract_section", {}).get("enabled", False):
            return []
        
        pattern = self.template.signal_processors["contract_section"].get("regex", r'第[一二三四五六七八九十]+条')
        matches = re.finditer(pattern, text)
        boundaries = []
        
        for match in matches:
            start = match.start()
            # 通常条款开始于换行后，尝试找到更准确的边界
            prev_newline = text.rfind('\n', 0, start)
            if prev_newline != -1:
                start = prev_newline + 1
            boundaries.append((start, start, 0.95))  # 高置信度
        
        return boundaries
    
    def detect_headings(self, text: str) -> List[Tuple[int, int, float]]:
        """检测标题边界"""
        if not self.template.signal_processors.get("heading", {}).get("enabled", False):
            return []
        
        patterns = self.template.signal_processors["heading"].get("patterns", ["#+"])
        boundaries = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = match.start()
                # 标题通常单独成行
                prev_newline = text.rfind('\n', 0, start)
                next_newline = text.find('\n', match.end())
                if prev_newline != -1:
                    start = prev_newline + 1
                boundaries.append((start, start, 0.90))  # 高置信度
        
        return boundaries
    
    def detect_paragraphs(self, text: str) -> List[Tuple[int, int, float]]:
        """检测段落边界"""
        if not self.template.signal_processors.get("paragraph", {}).get("enabled", False):
            return []
        
        separators = self.template.params.get("paragraph_separators", ["\n\n"])
        boundaries = [(0, 0, 0.0)]  # 文档开始
        
        for sep in separators:
            pos = 0
            while True:
                pos = text.find(sep, pos)
                if pos == -1:
                    break
                # 段落结束于分隔符前，新段落开始于分隔符后
                boundaries.append((pos, pos, 0.85))  # 高置信度
                pos += len(sep)
        
        # 添加文档结束边界
        boundaries.append((len(text), len(text), 0.0))
        
        # 去重并排序
        boundaries = list(set(boundaries))
        boundaries.sort()
        
        return boundaries
    
    def detect_sentences(self, text: str) -> List[Tuple[int, int, float]]:
        """检测句子边界"""
        if not self.template.signal_processors.get("sentence", {}).get("enabled", False):
            return []
        
        separators = self.template.params.get("sentence_separators", ["。", "！", "？", ".", "!", "?"])
        # 构建句子分割正则表达式
        pattern = '([' + ''.join(re.escape(sep) for sep in separators) + '])'
        
        # 使用正则表达式分割文本
        matches = re.finditer(pattern, text)
        boundaries = []
        
        for match in matches:
            end = match.end()
            boundaries.append((end, end, 0.80))  # 中等置信度
        
        return boundaries
    
    def detect_dialog_turns(self, text: str) -> List[Tuple[int, int, float]]:
        """检测对话轮次边界"""
        if not self.template.signal_processors.get("dialog_turn", {}).get("enabled", False):
            return []
        
        patterns = self.template.signal_processors["dialog_turn"].get("patterns", ["用户:", "客服:"])
        boundaries = []
        
        for pattern in patterns:
            matches = re.finditer(re.escape(pattern), text)
            for match in matches:
                start = match.start()
                # 对话轮次开始于模式出现处
                prev_newline = text.rfind('\n', 0, start)
                if prev_newline != -1:
                    start = prev_newline + 1
                boundaries.append((start, start, 0.95))  # 高置信度
        
        return boundaries
    
    def detect_token_based_boundaries(self, text: str) -> List[Tuple[int, int, float]]:
        """基于令牌计数的边界检测（滑动窗口）"""
        if not self.template.signal_processors.get("token_count", {}).get("enabled", False):
            return []
        
        base_chunk_size = self.template.params.get("base_chunk_size", 350)
        min_chunk_size = self.template.params.get("min_chunk_size", 150)
        overlap = self.template.params.get("overlap", 50)
        
        boundaries = []
        pos = 0
        chunk_index = 0
        
        # 简单地按照字符数分割（实际应用中应使用tokenizer）
        while pos < len(text):
            chunk_end = min(pos + base_chunk_size, len(text))
            boundaries.append((chunk_end, chunk_end, 0.70))  # 中等置信度
            pos = chunk_end - overlap
            chunk_index += 1
        
        return boundaries
    
    def detect_all_signals(self, text: str) -> Dict[str, List[Tuple[int, int, float]]]:
        """并行检测所有边界信号"""
        signals = {}
        
        # 并行执行各信号检测器
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 根据模板名称确定需要执行的检测器
            futures = {}
            
            if self.template.template_name == "合同" and self.template.signal_processors.get("contract_section", {}).get("enabled", False):
                futures["contract_section"] = executor.submit(self.detect_contract_sections, text)
            
            if self.template.template_name == "论文" and self.template.signal_processors.get("heading", {}).get("enabled", False):
                futures["heading"] = executor.submit(self.detect_headings, text)
            
            if self.template.signal_processors.get("paragraph", {}).get("enabled", False):
                futures["paragraph"] = executor.submit(self.detect_paragraphs, text)
            
            if self.template.signal_processors.get("sentence", {}).get("enabled", False):
                futures["sentence"] = executor.submit(self.detect_sentences, text)
            
            if self.template.template_name == "客服" and self.template.signal_processors.get("dialog_turn", {}).get("enabled", False):
                futures["dialog_turn"] = executor.submit(self.detect_dialog_turns, text)
            
            if self.template.signal_processors.get("token_count", {}).get("enabled", False):
                futures["token_count"] = executor.submit(self.detect_token_based_boundaries, text)
            
            # 收集结果
            for signal_type, future in futures.items():
                try:
                    signals[signal_type] = future.result()
                except Exception as e:
                    print(f"检测{signal_type}信号时出错: {e}")
                    signals[signal_type] = []
        
        return signals


class SignalFusionEngine:
    """
    信号融合引擎
    负责融合多种边界信号，应用优先级规则和冲突解决机制
    """
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 策略模板
    
    def fuse_signals(self, signals: Dict[str, List[Tuple[int, int, float]]]) -> List[Tuple[int, int, Dict[str, float]]]:
        """融合多种边界信号"""
        # 1. 收集所有信号的位置点
        all_boundaries = defaultdict(list)
        
        for signal_type, signal_boundaries in signals.items():
            for start, end, confidence in signal_boundaries:
                # 位置点统一使用start（因为大多数边界都是点）
                pos = start
                all_boundaries[pos].append((signal_type, confidence))
        
        # 2. 对位置点进行排序
        sorted_positions = sorted(all_boundaries.keys())
        
        # 3. 应用优先级规则融合信号
        fused_boundaries = []
        last_pos = 0
        
        for pos in sorted_positions:
            # 跳过文档开始位置
            if pos == 0:
                continue
            
            # 跳过与上一个位置太近的点（避免过于密集的边界）
            if pos - last_pos < self.template.params.get("min_chunk_size", 150) // 2:
                continue
            
            # 获取该位置的所有信号
            position_signals = all_boundaries[pos]
            
            # 根据信号优先级和冲突解决规则，选择最可信的边界
            selected_signals = self._resolve_conflicts(position_signals)
            
            # 只有当存在有效信号时才添加边界
            if selected_signals:
                # 计算综合置信度
                confidence_sum = sum(conf for _, conf in selected_signals)
                avg_confidence = confidence_sum / len(selected_signals) if selected_signals else 0
                
                # 创建边界字典，记录各信号的贡献
                boundary_signals = {signal_type: conf for signal_type, conf in selected_signals}
                
                # 添加文档结束边界
                if pos == sorted_positions[-1] and pos < len(self.template.params.get("original_text", "")):
                    pos = len(self.template.params.get("original_text", ""))
                
                fused_boundaries.append((last_pos, pos, boundary_signals))
                last_pos = pos
        
        # 确保文档结束
        if last_pos < len(self.template.params.get("original_text", "")):
            fused_boundaries.append((last_pos, len(self.template.params.get("original_text", "")), {"end_of_doc": 1.0}))
        
        return fused_boundaries
    
    def _resolve_conflicts(self, signals: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """解决信号冲突"""
        if not signals:
            return []
        
        # 按照模板中的边界优先级排序信号
        signals.sort(key=lambda x: self.template.boundary_priority.index(x[0]) if x[0] in self.template.boundary_priority else len(self.template.boundary_priority))
        
        # 如果只有一种信号类型或没有冲突，直接返回
        signal_types = {signal_type for signal_type, _ in signals}
        if len(signal_types) <= 1:
            return signals
        
        # 应用冲突解决规则
        resolved_signals = []
        high_priority_signal = signals[0]  # 最高优先级信号
        resolved_signals.append(high_priority_signal)
        
        # 对于其他信号，如果它们不与最高优先级信号冲突，则保留
        for signal_type, confidence in signals[1:]:
            conflict_key = f"{high_priority_signal[0]}_vs_{signal_type}"
            if conflict_key in self.template.conflict_resolution:
                # 根据规则决定是否保留
                if self.template.conflict_resolution[conflict_key] == signal_type:
                    resolved_signals.append((signal_type, confidence))
            else:
                # 默认保留高置信度信号
                if confidence > 0.7:
                    resolved_signals.append((signal_type, confidence))
        
        return resolved_signals


class ChunkGenerator:
    """
    分块生成器
    根据融合后的边界信息生成实际的分块，并添加元数据
    """
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 策略模板
    
    def generate_chunks(self, text: str, boundaries: List[Tuple[int, int, Dict[str, float]]]) -> List[ChunkMetadata]:
        """根据边界生成分块"""
        chunks = []
        chunk_index = 1
        
        for start, end, boundary_signals in boundaries:
            # 提取块内容
            content = text[start:end].strip()
            
            # 跳过空块
            if not content:
                continue
            
            # 创建块元数据
            chunk_id = f"chunk_{chunk_index:04d}"
            chunk = ChunkMetadata(chunk_id, start, end)
            chunk.content = content
            chunk.boundary_signals = boundary_signals
            
            # 确定块类型
            chunk.chunk_type = self._determine_chunk_type(boundary_signals)
            
            # 计算置信度
            chunk.confidence = self._calculate_confidence(boundary_signals)
            
            # 添加结构路径（简化版）
            chunk.structure_path = self._generate_structure_path(start, end, text)
            
            # 添加主题标签（简化版）
            chunk.topic_tags = self._extract_topic_tags(content)
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def _determine_chunk_type(self, boundary_signals: Dict[str, float]) -> str:
        """确定块类型"""
        if not boundary_signals:
            return "general"
        
        # 根据主要信号类型确定块类型
        max_signal = max(boundary_signals.items(), key=lambda x: x[1])
        
        type_mapping = {
            "contract_section": "contract_section",
            "heading": "heading",
            "paragraph": "paragraph",
            "sentence": "sentence",
            "dialog_turn": "dialog_turn",
            "token_count": "token_based",
            "end_of_doc": "general"
        }
        
        return type_mapping.get(max_signal[0], "general")
    
    def _calculate_confidence(self, boundary_signals: Dict[str, float]) -> float:
        """计算块边界的置信度"""
        if not boundary_signals:
            return 0.5  # 默认中等置信度
        
        # 计算平均置信度
        avg_confidence = sum(boundary_signals.values()) / len(boundary_signals)
        
        return avg_confidence
    
    def _generate_structure_path(self, start: int, end: int, text: str) -> List[str]:
        """生成结构路径（简化版）"""
        # 在实际应用中，这里应该根据文档的实际结构生成路径
        path = []
        
        # 简单示例：查找块所在的段落
        prev_newline = text.rfind('\n', 0, start)
        if prev_newline != -1:
            # 查找段落开头的潜在标题
            para_start = prev_newline + 1
            para_text = text[para_start:para_start + 50]  # 查看前50个字符
            
            # 检查是否是标题格式
            if re.match(r'^#|第.*章|\\d+\\.', para_text):
                path.append(f"标题:{para_text[:20]}")
        
        path.append(f"段落:{start}-{end}")
        
        return path
    
    def _extract_topic_tags(self, content: str) -> List[str]:
        """提取主题标签（简化版）"""
        # 在实际应用中，这里应该使用NLP模型提取关键词
        tags = []
        
        # 简单示例：查找常见关键词
        common_topics = {
            "合同": ["条款", "责任", "义务", "权利", "违约", "赔偿"],
            "论文": ["研究", "实验", "结论", "方法", "分析", "结果"],
            "财报": ["收入", "利润", "成本", "增长", "投资", "支出"],
            "客服": ["问题", "解决", "建议", "服务", "满意", "反馈"]
        }
        
        template_topics = common_topics.get(self.template.template_name, [])
        for topic in template_topics:
            if topic in content:
                tags.append(topic)
        
        # 限制标签数量
        return tags[:5] if len(tags) > 5 else tags


class StrategyLogger:
    """
    策略执行日志记录器
    记录分块过程中的各种信息，包括文档信息、策略参数、信号使用情况等
    """
    def __init__(self):
        self.log = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_ms": None,
            "doc_info": {},
            "strategy_params": {},
            "signal_statistics": defaultdict(int),
            "coverage_summary": {
                "total_chunks": 0,
                "chunk_type_distribution": defaultdict(int),
                "avg_confidence": 0.0,
                "signal_coverage": defaultdict(float)
            },
            "warnings": [],
            "errors": []
        }
    
    def update_doc_info(self, doc_info: Dict[str, Any]) -> None:
        """更新文档信息"""
        self.log["doc_info"] = doc_info
    
    def update_strategy_params(self, params: Dict[str, Any]) -> None:
        """更新策略参数"""
        self.log["strategy_params"] = params
    
    def record_signal_usage(self, signal_type: str) -> None:
        """记录信号使用情况"""
        self.log["signal_statistics"][signal_type] += 1
    
    def update_coverage_summary(self, chunks: List[ChunkMetadata]) -> None:
        """更新覆盖率摘要"""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return
        
        # 统计块类型分布
        type_counter = Counter()
        total_confidence = 0.0
        signal_contributions = defaultdict(int)
        
        for chunk in chunks:
            type_counter[chunk.chunk_type] += 1
            total_confidence += chunk.confidence
            
            # 统计各信号的贡献
            for signal_type in chunk.boundary_signals:
                signal_contributions[signal_type] += 1
        
        # 计算平均置信度
        avg_confidence = total_confidence / total_chunks if total_chunks > 0 else 0.0
        
        # 计算各信号的覆盖率
        signal_coverage = {}
        for signal_type, count in signal_contributions.items():
            signal_coverage[signal_type] = count / total_chunks if total_chunks > 0 else 0.0
        
        # 更新覆盖率摘要
        self.log["coverage_summary"] = {
            "total_chunks": total_chunks,
            "chunk_type_distribution": dict(type_counter),
            "avg_confidence": avg_confidence,
            "signal_coverage": dict(signal_coverage)
        }
    
    def add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.log["warnings"].append({
            "time": datetime.now().isoformat(),
            "message": message
        })
    
    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.log["errors"].append({
            "time": datetime.now().isoformat(),
            "message": message
        })
    
    def finalize(self) -> Dict[str, Any]:
        """最终化日志"""
        self.log["end_time"] = datetime.now().isoformat()
        # 计算持续时间（毫秒）
        start_dt = datetime.fromisoformat(self.log["start_time"])
        end_dt = datetime.fromisoformat(self.log["end_time"])
        self.log["duration_ms"] = (end_dt - start_dt).total_seconds() * 1000
        
        # 转换defaultdict为普通dict
        self.log["signal_statistics"] = dict(self.log["signal_statistics"])
        
        return self.log


class HybridChunkingOrchestrator:
    """
    混合分块策略编排器
    协调整个分块流程，从文档元信息提取、策略模板选择、信号检测与融合，到最终生成分块并输出结果
    """
    def __init__(self):
        self.logger = None  # 策略执行日志记录器
        
    def orchestrate(self, text: str, doc_type: str = "通用", 
                   template_name: Optional[str] = None, 
                   params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行混合分块策略编排
        
        参数:
            text: 输入文本
            doc_type: 文档类型（合同/论文/财报/客服/通用）
            template_name: 可选的模板名称（如果提供，将覆盖doc_type）
            params_override: 可选的参数覆写
            
        返回:
            包含分块结果和策略执行日志的字典
        """
        # 初始化日志记录器
        self.logger = StrategyLogger()
        
        try:
            # 开始计时
            start_time = time.time()
            
            # 1. 读取文档元信息
            doc_info = self._extract_doc_info(text, doc_type)
            self.logger.update_doc_info(doc_info)
            
            # 2. 选择策略模板
            template = self._select_strategy_template(doc_type, template_name)
            
            # 3. 应用参数覆写
            if params_override:
                template.override_params(params_override)
                self.logger.update_strategy_params(template.params)
            
            # 保存原文到模板参数，供后续使用
            template.params["original_text"] = text
            
            # 4. 并行运行结构/语义/实体检测
            detector = BoundarySignalDetector(template)
            signals = detector.detect_all_signals(text)
            
            # 记录信号统计
            for signal_type, signal_boundaries in signals.items():
                self.logger.record_signal_usage(signal_type)
            
            # 5. 使用优先级规则融合边界
            fusion_engine = SignalFusionEngine(template)
            fused_boundaries = fusion_engine.fuse_signals(signals)
            
            # 6. 生成块并附加元数据
            chunk_generator = ChunkGenerator(template)
            chunks = chunk_generator.generate_chunks(text, fused_boundaries)
            
            # 7. 更新覆盖率摘要
            self.logger.update_coverage_summary(chunks)
            
            # 转换块为字典格式
            chunks_dict = [chunk.to_dict() for chunk in chunks]
            
            # 结束计时
            duration_ms = (time.time() - start_time) * 1000
            
            # 8. 输出块与策略执行日志
            strategy_log = self.logger.finalize()
            
            return {
                "chunks": chunks_dict,
                "strategy_log": strategy_log,
                "summary": {
                    "total_chunks": len(chunks),
                    "duration_ms": duration_ms,
                    "template_used": template.template_name,
                    "doc_type": doc_type
                }
            }
            
        except Exception as e:
            if self.logger:
                self.logger.add_error(f"分块过程中发生错误: {str(e)}")
                strategy_log = self.logger.finalize()
            else:
                strategy_log = {"error": str(e)}
            
            return {
                "chunks": [],
                "strategy_log": strategy_log,
                "summary": {"error": str(e)}
            }
    
    def _extract_doc_info(self, text: str, doc_type: str) -> Dict[str, Any]:
        """提取文档元信息"""
        # 简单的文档信息提取
        return {
            "doc_type": doc_type,
            "length_chars": len(text),
            "length_words": len(text.split()),
            "language": "zh" if re.search(r'[\u4e00-\u9fa5]', text) else "en",
            "has_tables": bool(re.search(r'(\d+\.\d+\s+){3,}', text)),
            "has_lists": bool(re.search(r'(\d+\.|\*|\-)\s+', text)),
            "paragraph_count": text.count('\n\n') + 1
        }
    
    def _select_strategy_template(self, doc_type: str, template_name: Optional[str]) -> StrategyTemplate:
        """选择策略模板"""
        # 如果提供了template_name，使用它；否则使用doc_type
        template_to_use = template_name if template_name else doc_type
        
        # 验证模板名称
        valid_templates = ["合同", "论文", "财报", "客服", "通用"]
        if template_to_use not in valid_templates:
            self.logger.add_warning(f"未知的模板名称: {template_to_use}，使用默认模板")
            template_to_use = "通用"
        
        return StrategyTemplate(template_to_use)
    
    def save_results(self, result: Dict[str, Any], chunks_file: str = "chunks_result.json", 
                    log_file: str = "strategy_log.json") -> None:
        """保存分块结果和策略日志到文件"""
        try:
            # 保存分块结果
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(result["chunks"], f, ensure_ascii=False, indent=2)
              
            # 保存策略日志
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(result["strategy_log"], f, ensure_ascii=False, indent=2)
              
            print(f"分块结果已保存到: {chunks_file}")
            print(f"策略日志已保存到: {log_file}")
              
        except Exception as e:
            print(f"保存结果时出错: {e}")


# 示例用法
if __name__ == "__main__":
    # 示例文本 - 不同类型的文档
    sample_texts = {
        "合同": """
        合同编号：2023-001
        
        甲方：科技有限公司
        乙方：创新发展中心
        
        第一条 合作内容
        甲乙双方就人工智能技术研发达成合作，甲方提供技术支持，乙方提供研发场地和资金。
        
        第二条 权利与义务
        甲方有权获取项目研发成果的使用权，乙方有权获得技术转让费。双方应保守项目相关的商业秘密。
        
        第三条 违约责任
        若一方违反本合同约定，应向另一方支付违约金，并赔偿因此造成的损失。
        """,
        
        "论文": """
        # 人工智能在医疗领域的应用研究
        
        ## 摘要
        本研究探讨了人工智能技术在医疗诊断、治疗方案制定和患者管理方面的应用现状和前景。
        
        ## 引言
        随着深度学习技术的快速发展，人工智能在医疗领域的应用日益广泛。本文综述了近年来的相关研究成果。
        
        ## 研究方法
        本研究采用文献综述和案例分析相结合的方法，收集了2018-2023年间的相关研究文献。
        
        ## 实验结果
        实验结果表明，基于深度学习的医疗影像诊断系统在某些领域已达到或超过人类专家水平。
        """,
        
        "财报": """
        2023年第三季度财务报告
        
        一、经营业绩
        本季度实现营业收入10.5亿元，同比增长15.3%；净利润2.1亿元，同比增长22.7%。
        
        二、主要财务指标
        毛利率：42.5%（环比+1.2%）
        净利率：20.0%（环比+2.5%）
        经营现金流：3.2亿元
        
        三、业务分析
        智能硬件业务收入占比45%，同比增长20%；软件服务收入占比35%，同比增长30%；其他业务收入占比20%。
        """,
        
        "客服": """
        用户：你好，我想咨询一下你们的产品保修政策。
        客服：您好！很高兴为您服务。我们的产品提供一年免费保修服务，保修期从购买之日起计算。
        用户：如果产品是人为损坏的，还能保修吗？
        客服：如果经检测确认是人为损坏，将不在免费保修范围内，但我们可以提供有偿维修服务。
        用户：那维修费用大概是多少呢？
        客服：维修费用根据损坏程度和所需更换的零部件不同而有所差异，建议您将产品送到我们的服务中心进行检测评估。
        """
    }
    
    print("===== 混合分块策略编排器示例 =====")
    
    # 创建编排器实例
    orchestrator = HybridChunkingOrchestrator()
    
    # 选择一个示例文档类型
    doc_type = "合同"  # 可以更改为 "论文"、"财报"、"客服"
    sample_text = sample_texts[doc_type]
    
    print(f"\n处理 {doc_type} 类型文档...")
    
    # 执行分块策略编排
    result = orchestrator.orchestrate(sample_text, doc_type=doc_type)
    
    # 打印摘要信息
    print("\n分块结果摘要:")
    print(f"  总块数: {result['summary']['total_chunks']}")
    print(f"  使用模板: {result['summary']['template_used']}")
    print(f"  处理时间: {result['summary']['duration_ms']:.2f} ms")
    
    # 打印详细分块信息
    print("\n详细分块信息:")
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"\n块 {i}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  类型: {chunk['chunk_type']}")
        print(f"  位置: {chunk['start_pos']}-{chunk['end_pos']}")
        print(f"  置信度: {chunk['confidence']:.2f}")
        print(f"  边界信号: {chunk['boundary_signals']}")
        print(f"  结构路径: {chunk['structure_path']}")
        print(f"  主题标签: {chunk['topic_tags']}")
        print(f"  内容: {chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}")
    
    # 打印策略日志摘要
    print("\n策略执行日志摘要:")
    log = result['strategy_log']
    print(f"  信号统计: {log['signal_statistics']}")
    print(f"  块类型分布: {log['coverage_summary']['chunk_type_distribution']}")
    print(f"  平均置信度: {log['coverage_summary']['avg_confidence']:.2f}")
    print(f"  信号覆盖率: {log['coverage_summary']['signal_coverage']}")
    
    # 保存结果到文件
    print("\n保存结果到文件...")
    orchestrator.save_results(result, f"chunks_{doc_type}.json", f"log_{doc_type}.json")
    
    # 添加调试信息
    print("\n检查文件是否保存成功：")
    import os
    if os.path.exists(f"chunks_{doc_type}.json"):
        print(f"✅ 分块结果文件 'chunks_{doc_type}.json' 保存成功")
        print(f"   文件大小: {os.path.getsize(f'chunks_{doc_type}.json')} 字节")
    else:
        print(f"❌ 分块结果文件 'chunks_{doc_type}.json' 不存在")
    
    if os.path.exists(f"log_{doc_type}.json"):
        print(f"✅ 日志文件 'log_{doc_type}.json' 保存成功")
        print(f"   文件大小: {os.path.getsize(f'log_{doc_type}.json')} 字节")
    else:
        print(f"❌ 日志文件 'log_{doc_type}.json' 不存在")
        
    # 打印当前工作目录
    print(f"\n当前工作目录: {os.getcwd()}")
    
    print("\n===== 示例完成 =====")