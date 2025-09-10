#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指代链与SRL守护边界分块器

该模块实现了一个基于指代链和语义角色标注(SRL)的边界守护机制，用于优化初始分块结果。
主要功能包括：
1. 检测分块边界处是否存在跨块的核心指代链
2. 识别边界处的跨块语义角色论元
3. 根据检测结果微调边界位置或增加重叠
4. 输出调整后的分块结果和变更日志

适用于需要保留上下文连贯性的RAG系统，特别是处理包含复杂指代关系和语义依赖的文本。
"""

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
import tiktoken
import random

# 尝试导入必要的库，如果失败则安装
def install_dependencies():
    try:
        import spacy
        # 尝试加载中文模型
        nlp = spacy.load("zh_core_web_sm")
    except ImportError:
        print("正在安装必要的依赖...")
        import subprocess
        subprocess.check_call(["pip", "install", "spacy"])
        # 尝试下载中文模型
        try:
            subprocess.check_call(["python", "-m", "spacy", "download", "zh_core_web_sm"])
        except Exception:
            print("中文模型下载失败，将使用规则匹配替代")

# 安装依赖
install_dependencies()

# 尝试导入spacy
HAS_SPACY = False
try:
    import spacy
    HAS_SPACY = True
    # 尝试加载中文模型，失败则使用规则匹配
    try:
        nlp = spacy.load("zh_core_web_sm")
    except:
        HAS_SPACY = False
except ImportError:
    pass

@dataclass
class Chunk:
    """表示一个文本分块的数据类"""
    chunk_id: int
    text: str
    start_index: int
    end_index: int
    tokens_count: int
    sentences: List[str] = field(default_factory=list)
    overlap_with_prev: int = 0  # 与前一分块的重叠字符数
    overlap_with_next: int = 0  # 与后一分块的重叠字符数

@dataclass
class Coreference:
    """表示指代关系的数据类"""
    mention: str  # 提及的文本
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    cluster_id: int  # 所属的指代链ID
    is_core: bool = False  # 是否为核心实体

@dataclass
class SRLArgument:
    """表示语义角色论元的数据类"""
    predicate: str  # 谓词
    predicate_start: int
    predicate_end: int
    role: str  # 角色（如ARG0, ARG1, ARGM等）
    arg_text: str  # 论元文本
    arg_start: int
    arg_end: int

@dataclass
class BoundaryAdjustment:
    """表示边界调整记录的数据类"""
    boundary_index: int  # 边界索引（在第几个分块之后）
    original_position: int  # 原始边界位置
    new_position: int  # 新边界位置
    adjustment_type: str  # 调整类型（'move'或'overlap'）
    reason: str  # 调整原因
    corefs: List[Coreference] = field(default_factory=list)  # 相关的指代链
    srl_args: List[SRLArgument] = field(default_factory=list)  # 相关的SRL论元

class CorefSRLBoundaryGuard:
    """指代链与SRL守护边界分块器"""
    def __init__(self, 
                 window_size: int = 2,  # 边界两侧检查的句子数量
                 max_overlap: int = 100,  # 最大重叠字符数
                 max_move_distance: int = 200,  # 最大移动距离（字符）
                 coref_threshold: float = 0.7,  # 指代链完整性阈值
                 srl_threshold: float = 0.5):  # SRL论元完整性阈值
        """
        初始化边界守护分块器
        
        参数:
        - window_size: 边界两侧检查的句子数量
        - max_overlap: 最大重叠字符数
        - max_move_distance: 最大移动距离（字符）
        - coref_threshold: 指代链完整性阈值，超过此值需要调整边界
        - srl_threshold: SRL论元完整性阈值，超过此值需要调整边界
        """
        self.window_size = window_size
        self.max_overlap = max_overlap
        self.max_move_distance = max_move_distance
        self.coref_threshold = coref_threshold
        self.srl_threshold = srl_threshold
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.adjustments: List[BoundaryAdjustment] = []
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """将文本分割成句子，并记录每个句子的起始和结束位置"""
        sentences_with_positions = []
        # 使用正则表达式分割句子，保留分隔符（标点符号）
        sentences = re.split(r'([.!?，。！？])', text)
        
        current_pos = 0
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加标点符号
            
            # 跳过空句子
            if sentence.strip() == "":
                current_pos += len(sentence)
                continue
            
            start_pos = current_pos
            end_pos = current_pos + len(sentence)
            sentences_with_positions.append((sentence.strip(), start_pos, end_pos))
            current_pos = end_pos
        
        return sentences_with_positions
    
    def _detect_coreferences(self, text: str) -> List[Coreference]:
        """检测文本中的指代关系
        
        注：在实际应用中，这里应该使用专门的指代消解模型
        由于环境限制，这里使用基于规则的简化模拟实现
        """
        corefs = []
        cluster_id = 0
        
        # 模拟检测的核心实体类型
        entity_patterns = [
            r'[他她它]们', r'这[个些]', r'那[个些]', r'该[公司项目研究等]',
            r'研究[结果对象等]', r'实验[数据方法等]', r'模型[训练结果等]'
        ]
        
        # 模拟的专有名词（假设这些是文档中的核心实体）
        proper_nouns = [
            '大脑中动脉', '脑梗死', '影像特点', '发病机制',
            '研究对象', '实验方法', '治疗方案', '诊断标准'
        ]
        
        # 检测专有名词（作为核心实体）
        for noun in proper_nouns:
            for match in re.finditer(re.escape(noun), text):
                corefs.append(Coreference(
                    mention=noun,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    cluster_id=cluster_id,
                    is_core=True
                ))
            cluster_id += 1
        
        # 检测代词和其他指代表达式
        for pattern in entity_patterns:
            for match in re.finditer(pattern, text):
                # 为简单起见，随机分配到某个现有簇或创建新簇
                assigned_cluster = random.choice(range(cluster_id)) if cluster_id > 0 else cluster_id
                if assigned_cluster == cluster_id:
                    cluster_id += 1
                
                corefs.append(Coreference(
                    mention=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    cluster_id=assigned_cluster,
                    is_core=False
                ))
        
        # 如果有spacy，尝试使用其NER功能增强结果
        if HAS_SPACY:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    # 为实体分配新的簇ID
                    corefs.append(Coreference(
                        mention=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        cluster_id=cluster_id,
                        is_core=True
                    ))
                    cluster_id += 1
            except Exception as e:
                print(f"使用spaCy增强指代检测时出错: {e}")
        
        return corefs
    
    def _detect_srl_arguments(self, text: str) -> List[SRLArgument]:
        """检测文本中的语义角色论元
        
        注：在实际应用中，这里应该使用专门的SRL模型
        由于环境限制，这里使用基于规则的简化模拟实现
        """
        srl_args = []
        
        # 模拟的谓词和其可能的论元模式
        predicate_patterns = [
            (r'研究[了过]', ['研究对象', '研究方法', '研究结果']),
            (r'发现[了过]', ['发现的现象', '发现的结果']),
            (r'表明[了]', ['表明的结论', '支持的观点']),
            (r'包括[了]', ['包括的内容', '涉及的方面']),
            (r'影响[了]', ['影响因素', '受影响的事物']),
            (r'导致[了]', ['原因', '结果'])
        ]
        
        # 角色映射
        role_mapping = {
            0: 'ARG0',  # 通常是施事
            1: 'ARG1',  # 通常是受事或主题
            2: 'ARG2',  # 通常是其他参与者
            3: 'ARGM-LOC',  # 地点
            4: 'ARGM-TMP'   # 时间
        }
        
        # 检测谓词和论元
        for pred_pattern, arg_patterns in predicate_patterns:
            for match in re.finditer(pred_pattern, text):
                predicate = match.group()
                pred_start = match.start()
                pred_end = match.end()
                
                # 为简单起见，在谓词前后一定范围内搜索可能的论元
                search_range = 200  # 搜索范围（字符）
                start_search = max(0, pred_start - search_range)
                end_search = min(len(text), pred_end + search_range)
                context = text[start_search:end_search]
                
                # 模拟查找论元
                for i, arg_pattern in enumerate(arg_patterns):
                    # 随机决定是否在此处找到该论元
                    if random.random() > 0.5:
                        continue
                    
                    # 在上下文中查找论元模式
                    arg_matches = list(re.finditer(arg_pattern, context))
                    if arg_matches:
                        arg_match = random.choice(arg_matches)
                        arg_text = arg_match.group()
                        arg_start = start_search + arg_match.start()
                        arg_end = start_search + arg_match.end()
                        
                        # 分配角色
                        role = role_mapping.get(i % len(role_mapping), f'ARG{i}')
                        
                        srl_args.append(SRLArgument(
                            predicate=predicate,
                            predicate_start=pred_start,
                            predicate_end=pred_end,
                            role=role,
                            arg_text=arg_text,
                            arg_start=arg_start,
                            arg_end=arg_end
                        ))
        
        return srl_args
    
    def _check_boundary_corefs(self, 
                              chunk1: Chunk, 
                              chunk2: Chunk, 
                              all_corefs: List[Coreference]) -> Tuple[bool, List[Coreference]]:
        """检查边界处是否存在跨块的核心指代链"""
        # 获取边界位置
        boundary_pos = chunk1.end_index
        
        # 收集边界附近的指代
        window_size = 100  # 边界两侧的检查窗口大小（字符）
        boundary_corefs = []
        
        for coref in all_corefs:
            # 检查指代是否跨越边界或在边界附近
            if (coref.start_pos < boundary_pos < coref.end_pos or 
                (boundary_pos - window_size < coref.start_pos < boundary_pos + window_size)):
                boundary_corefs.append(coref)
        
        # 分析这些指代是否形成了跨越边界的指代链
        cluster_ids = set([coref.cluster_id for coref in boundary_corefs])
        cross_boundary_clusters = []
        
        for cluster_id in cluster_ids:
            cluster_corefs = [coref for coref in boundary_corefs if coref.cluster_id == cluster_id]
            # 检查该簇是否有指代跨越边界
            has_cross = any(coref.start_pos < boundary_pos < coref.end_pos for coref in cluster_corefs)
            # 检查该簇是否有指代分布在边界两侧
            has_left = any(coref.end_pos <= boundary_pos for coref in cluster_corefs)
            has_right = any(coref.start_pos >= boundary_pos for coref in cluster_corefs)
            
            if (has_cross or (has_left and has_right)) and any(coref.is_core for coref in cluster_corefs):
                cross_boundary_clusters.extend(cluster_corefs)
        
        # 计算需要调整的置信度
        if cross_boundary_clusters:
            # 计算核心实体的比例
            core_ratio = sum(1 for coref in cross_boundary_clusters if coref.is_core) / len(cross_boundary_clusters)
            return core_ratio > self.coref_threshold, cross_boundary_clusters
        
        return False, []
    
    def _check_boundary_srl(self, 
                           chunk1: Chunk, 
                           chunk2: Chunk, 
                           all_srl_args: List[SRLArgument]) -> Tuple[bool, List[SRLArgument]]:
        """检查边界处是否存在跨块的SRL论元"""
        # 获取边界位置
        boundary_pos = chunk1.end_index
        
        # 收集边界附近的SRL论元
        window_size = 100  # 边界两侧的检查窗口大小（字符）
        boundary_srl_args = []
        
        for srl_arg in all_srl_args:
            # 检查论元或谓词是否跨越边界或在边界附近
            if (srl_arg.arg_start < boundary_pos < srl_arg.arg_end or 
               srl_arg.predicate_start < boundary_pos < srl_arg.predicate_end or
                (boundary_pos - window_size < srl_arg.arg_start < boundary_pos + window_size) or
                (boundary_pos - window_size < srl_arg.predicate_start < boundary_pos + window_size)):
                boundary_srl_args.append(srl_arg)
        
        # 分析这些论元是否形成了跨越边界的语义依赖
        cross_boundary_args = []
        predicate_ids = set()
        
        for srl_arg in boundary_srl_args:
            # 检查论元或谓词是否跨越边界
            if (srl_arg.arg_start < boundary_pos < srl_arg.arg_end or 
               srl_arg.predicate_start < boundary_pos < srl_arg.predicate_end):
                cross_boundary_args.append(srl_arg)
                predicate_ids.add((srl_arg.predicate, srl_arg.predicate_start))
            
            # 检查同一谓词的论元是否分布在边界两侧
            elif ((srl_arg.predicate_start < boundary_pos and srl_arg.arg_start > boundary_pos) or 
                 (srl_arg.predicate_start > boundary_pos and srl_arg.arg_start < boundary_pos)):
                cross_boundary_args.append(srl_arg)
                predicate_ids.add((srl_arg.predicate, srl_arg.predicate_start))
        
        # 计算需要调整的置信度
        if cross_boundary_args:
            # 计算跨越边界的谓词比例
            total_predicates = len(predicate_ids)
            if total_predicates > 0:
                # 这里简化处理，只要有跨越边界的重要论元就返回True
                # 实际应用中可以使用更复杂的评分机制
                return True, cross_boundary_args
        
        return False, []
    
    def _adjust_boundary(self, 
                        chunks: List[Chunk], 
                        boundary_idx: int, 
                        corefs: List[Coreference], 
                        srl_args: List[SRLArgument],
                        original_text: str) -> None:
        """调整边界位置或增加重叠"""
        if boundary_idx < 0 or boundary_idx >= len(chunks) - 1:
            return
        
        chunk1 = chunks[boundary_idx]
        chunk2 = chunks[boundary_idx + 1]
        original_boundary = chunk1.end_index
        
        # 决定调整策略：优先移动边界，其次增加重叠
        adjustment_type = "move"
        
        # 找到最合适的新边界位置（在句子边界处）
        # 获取chunk1和chunk2的句子
        chunk1_sentences = self._split_into_sentences(chunk1.text)
        chunk2_sentences = self._split_into_sentences(chunk2.text)
        
        # 如果有足够的句子，尝试移动边界到更合适的句子边界
        if len(chunk1_sentences) > 1 and len(chunk2_sentences) > 1:
            # 尝试在chunk1的后window_size个句子或chunk2的前window_size个句子中寻找新边界
            # 简化处理：移动到chunk1的倒数第window_size个句子结束处或chunk2的第window_size个句子开始处
            # 实际应用中可以使用更复杂的算法来选择最优边界
            sentences_to_move = min(self.window_size, len(chunk1_sentences) - 1, len(chunk2_sentences) - 1)
            
            if sentences_to_move > 0:
                # 计算新的边界位置
                # 这里我们选择移动边界到chunk1的倒数第sentences_to_move个句子结束处
                # 或者chunk2的第sentences_to_move个句子开始处，取较近的一个
                move_back = sentences_to_move
                move_forward = sentences_to_move
                
                # 计算向后移动的距离
                if move_back <= len(chunk1_sentences) - 1:
                    new_end_pos_back = chunk1.start_index
                    for i in range(len(chunk1_sentences) - move_back):
                        new_end_pos_back += len(chunk1_sentences[i][0])
                else:
                    new_end_pos_back = chunk1.start_index
                
                # 计算向前移动的距离
                if move_forward <= len(chunk2_sentences) - 1:
                    new_start_pos_forward = chunk2.start_index
                    for i in range(move_forward):
                        new_start_pos_forward += len(chunk2_sentences[i][0])
                else:
                    new_start_pos_forward = chunk2.end_index
                
                # 选择距离原边界较近的移动方向
                distance_back = original_boundary - new_end_pos_back
                distance_forward = new_start_pos_forward - original_boundary
                
                if distance_back <= distance_forward and distance_back <= self.max_move_distance:
                    new_boundary = new_end_pos_back
                elif distance_forward <= self.max_move_distance:
                    new_boundary = new_start_pos_forward
                else:
                    # 如果移动距离太大，改为增加重叠
                    adjustment_type = "overlap"
                    overlap_size = min(self.max_overlap, len(chunk1.text) // 2, len(chunk2.text) // 2)
                    
                    # 记录调整
                    self.adjustments.append(BoundaryAdjustment(
                        boundary_index=boundary_idx,
                        original_position=original_boundary,
                        new_position=original_boundary,
                        adjustment_type=adjustment_type,
                        reason="跨块的核心指代/论元，移动距离过大，改为增加重叠",
                        corefs=corefs,
                        srl_args=srl_args
                    ))
                    
                    # 调整chunk1和chunk2的重叠
                    chunk1.overlap_with_next = overlap_size
                    chunk2.overlap_with_prev = overlap_size
                    return
                
                # 更新分块
                # 重新计算chunk1的文本和结束位置
                chunk1_new_text = original_text[chunk1.start_index:new_boundary]
                chunk1.text = chunk1_new_text
                chunk1.end_index = new_boundary
                chunk1.tokens_count = len(self.tokenizer.encode(chunk1_new_text))
                
                # 重新计算chunk2的文本和开始位置
                chunk2_new_text = original_text[new_boundary:chunk2.end_index]
                chunk2.text = chunk2_new_text
                chunk2.start_index = new_boundary
                chunk2.tokens_count = len(self.tokenizer.encode(chunk2_new_text))
                
                # 记录调整
                self.adjustments.append(BoundaryAdjustment(
                    boundary_index=boundary_idx,
                    original_position=original_boundary,
                    new_position=new_boundary,
                    adjustment_type=adjustment_type,
                    reason="移动边界以保持指代链和语义角色的完整性",
                    corefs=corefs,
                    srl_args=srl_args
                ))
        else:
            # 如果句子数量不足，增加重叠
            adjustment_type = "overlap"
            overlap_size = min(self.max_overlap, len(chunk1.text) // 2, len(chunk2.text) // 2)
            
            # 记录调整
            self.adjustments.append(BoundaryAdjustment(
                boundary_index=boundary_idx,
                original_position=original_boundary,
                new_position=original_boundary,
                adjustment_type=adjustment_type,
                reason="句子数量不足，增加重叠以保持指代链和语义角色的完整性",
                corefs=corefs,
                srl_args=srl_args
            ))
            
            # 调整chunk1和chunk2的重叠
            chunk1.overlap_with_next = overlap_size
            chunk2.overlap_with_prev = overlap_size
    
    def _perform_initial_chunking(self, text: str) -> List[Chunk]:
        """执行初始语义切分
        
        注：这里使用简单的基于句子的分块作为初始分块
        实际应用中可以使用更复杂的分块策略，如之前实现的主题聚类分块器
        """
        chunks = []
        sentences_with_positions = self._split_into_sentences(text)
        
        # 简单的基于句子数量的分块
        max_sentences_per_chunk = 3
        current_chunk_sentences = []
        current_start = None
        chunk_id = 0
        
        for i, (sentence, start_pos, end_pos) in enumerate(sentences_with_positions):
            if not current_chunk_sentences:
                current_start = start_pos
            
            current_chunk_sentences.append(sentence)
            
            # 如果达到最大句子数或已经是最后一个句子，创建分块
            if len(current_chunk_sentences) >= max_sentences_per_chunk or i == len(sentences_with_positions) - 1:
                chunk_text = "".join(current_chunk_sentences)
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_index=current_start,
                    end_index=end_pos,
                    tokens_count=len(self.tokenizer.encode(chunk_text)),
                    sentences=current_chunk_sentences.copy()
                )
                chunks.append(chunk)
                chunk_id += 1
                current_chunk_sentences = []
        
        return chunks
    
    def process(self, text: str, initial_chunks: Optional[List[Dict]] = None) -> Tuple[List[Chunk], List[BoundaryAdjustment]]:
        """处理文本，执行边界守护
        
        参数:
        - text: 输入文本
        - initial_chunks: 可选的初始分块结果，如果为None则自动执行初始分块
        
        返回:
        - List[Chunk]: 调整后的分块列表
        - List[BoundaryAdjustment]: 边界调整记录
        """
        # 重置调整记录
        self.adjustments = []
        
        # 执行初始分块
        if initial_chunks is None:
            chunks = self._perform_initial_chunking(text)
        else:
            # 转换初始分块格式
            chunks = []
            for i, chunk_dict in enumerate(initial_chunks):
                chunk = Chunk(
                    chunk_id=i,
                    text=chunk_dict.get('text', ''),
                    start_index=chunk_dict.get('start_index', 0),
                    end_index=chunk_dict.get('end_index', 0),
                    tokens_count=chunk_dict.get('tokens_count', 0),
                    sentences=chunk_dict.get('sentences', []),
                    overlap_with_prev=chunk_dict.get('overlap_with_prev', 0),
                    overlap_with_next=chunk_dict.get('overlap_with_next', 0)
                )
                chunks.append(chunk)
        
        # 如果只有一个分块，无需调整边界
        if len(chunks) <= 1:
            return chunks, self.adjustments
        
        # 检测文本中的指代关系和SRL论元
        print("正在检测指代关系和SRL论元...")
        all_corefs = self._detect_coreferences(text)
        all_srl_args = self._detect_srl_arguments(text)
        print(f"检测到 {len(all_corefs)} 个指代关系和 {len(all_srl_args)} 个SRL论元")
        
        # 对每个边界进行检查和调整
        print("正在检查和调整分块边界...")
        # 需要从后向前处理，避免调整前面的边界影响后面边界的位置
        for i in range(len(chunks) - 2, -1, -1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # 检查边界处是否存在跨块的核心指代链
            need_adjust_coref, boundary_corefs = self._check_boundary_corefs(chunk1, chunk2, all_corefs)
            
            # 检查边界处是否存在跨块的SRL论元
            need_adjust_srl, boundary_srl_args = self._check_boundary_srl(chunk1, chunk2, all_srl_args)
            
            # 如果需要调整，执行边界调整
            if need_adjust_coref or need_adjust_srl:
                print(f"调整边界 {i}（位于字符位置 {chunk1.end_index}）")
                self._adjust_boundary(
                    chunks, 
                    i, 
                    boundary_corefs, 
                    boundary_srl_args,
                    text
                )
        
        print(f"完成边界调整，共调整了 {len(self.adjustments)} 个边界")
        
        return chunks, self.adjustments
    
    def save_chunks_to_json(self, 
                           chunks: List[Chunk], 
                           output_path: str, 
                           adjustments: Optional[List[BoundaryAdjustment]] = None) -> None:
        """将调整后的分块结果保存为JSON文件
        
        参数:
        - chunks: 调整后的分块列表
        - output_path: 输出文件路径
        - adjustments: 可选的边界调整记录
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换分块为字典格式
        chunks_dict = []
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            # 对于复杂对象类型，可能需要特殊处理
            chunks_dict.append(chunk_dict)
        
        # 准备完整的输出数据
        output_data = {
            "chunks": chunks_dict,
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk.tokens_count for chunk in chunks),
            "adjustment_count": len(adjustments) if adjustments else 0,
            "adjustments": []
        }
        
        # 添加调整记录
        if adjustments:
            for adj in adjustments:
                adj_dict = {
                    "boundary_index": adj.boundary_index,
                    "original_position": adj.original_position,
                    "new_position": adj.new_position,
                    "adjustment_type": adj.adjustment_type,
                    "reason": adj.reason,
                    "coref_count": len(adj.corefs),
                    "srl_arg_count": len(adj.srl_args)
                }
                output_data["adjustments"].append(adj_dict)
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"分块结果已保存至: {output_path}")

# 读取文本文件
def read_text_file(file_path: str) -> str:
    """从文本文件中读取内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 从JSON文件加载初始分块
def load_initial_chunks(json_path: str) -> List[Dict]:
    """从JSON文件加载初始分块结果"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 处理不同格式的JSON文件
            if isinstance(data, list):
                return data
            elif "chunks" in data:
                return data["chunks"]
            else:
                print(f"警告：JSON文件 {json_path} 格式不符合预期")
                return []
    except Exception as e:
        print(f"加载初始分块时出错: {e}")
        return []

# 主函数
def main():
    """主函数，演示指代链与SRL守护边界分块器的使用"""
    # 设置文件路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # 选择一个文本文件或PDF文件作为输入
    # 这里我们使用一个示例文本
    sample_text = """
    大脑中动脉狭窄与闭塞是导致脑梗死的重要原因。本研究回顾性分析了2019年1月至2021年12月期间在我院神经内科住院治疗的120例大脑中动脉狭窄与闭塞患者的临床资料。
    研究对象包括68例男性和52例女性，年龄范围为45-78岁，平均年龄为(62.3±8.5)岁。所有患者均经头颅CT或MRI证实为脑梗死，并经脑血管造影或CT血管造影证实存在大脑中动脉狭窄或闭塞。
    研究方法包括收集患者的一般资料、临床症状、影像学检查结果、实验室检查结果以及治疗方案和预后情况。影像学特点分析包括脑梗死的部位、范围、形态以及大脑中动脉狭窄或闭塞的程度和部位。
    发病机制研究主要探讨了大脑中动脉狭窄与闭塞导致脑梗死的病理生理过程，包括血栓形成、脑血流灌注不足、侧支循环代偿以及神经细胞损伤和坏死的机制。
    研究结果表明，大脑中动脉狭窄与闭塞患者的脑梗死部位主要位于基底节区、额叶和颞叶，其中以基底节区最为常见。大脑中动脉闭塞患者的脑梗死范围通常较狭窄患者更大，神经功能缺损更严重。
    此外，研究还发现，大脑中动脉狭窄与闭塞的程度与脑梗死的严重程度和预后密切相关。早期诊断和及时治疗对于改善患者的预后至关重要。
    结论部分指出，大脑中动脉狭窄与闭塞是导致脑梗死的重要原因，其影像学特点和发病机制具有一定的特征性。深入研究这些特征有助于提高对该病的认识和诊治水平，改善患者的预后。
    """
    
    # 设置输出文件路径
    output_json_path = os.path.join(results_dir, "coref_srl_guard_chunks.json")
    
    # 尝试加载之前的分块结果作为初始分块
    initial_chunks_path = os.path.join(results_dir, "topic_cluster_chunks.json")
    initial_chunks = []
    if os.path.exists(initial_chunks_path):
        print(f"正在加载初始分块结果: {initial_chunks_path}")
        initial_chunks = load_initial_chunks(initial_chunks_path)
    
    # 创建边界守护分块器
    boundary_guard = CorefSRLBoundaryGuard(
        window_size=2,
        max_overlap=100,
        max_move_distance=200,
        coref_threshold=0.7,
        srl_threshold=0.5
    )
    
    # 处理文本，执行边界守护
    print("正在执行指代链与SRL守护边界分块...")
    chunks, adjustments = boundary_guard.process(sample_text, initial_chunks if initial_chunks else None)
    
    # 保存结果
    boundary_guard.save_chunks_to_json(chunks, output_json_path, adjustments)
    
    # 打印统计信息
    print(f"\n分块统计信息:")
    print(f"分块总数: {len(chunks)}")
    print(f"总token数: {sum(chunk.tokens_count for chunk in chunks)}")
    print(f"平均每个分块的token数: {sum(chunk.tokens_count for chunk in chunks) / len(chunks):.2f}")
    print(f"调整的边界数量: {len(adjustments)}")
    
    # 打印前3个分块的信息作为示例
    print("\n前3个分块示例:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n分块 {i+1} (tokens: {chunk.tokens_count}):")
        # 打印前150个字符
        preview = chunk.text[:150] + ("..." if len(chunk.text) > 150 else "")
        print(f"{preview}")
        print(f"  起始位置: {chunk.start_index}, 结束位置: {chunk.end_index}")
        print(f"  与前一分块重叠: {chunk.overlap_with_prev}字符")
        print(f"  与后一分块重叠: {chunk.overlap_with_next}字符")
    
    # 打印调整记录
    if adjustments:
        print("\n边界调整记录示例:")
        for i, adj in enumerate(adjustments[:3]):  # 只显示前3个调整记录
            print(f"\n调整 {i+1}:")
            print(f"  边界索引: {adj.boundary_index}")
            print(f"  原始位置: {adj.original_position}")
            print(f"  新位置: {adj.new_position}")
            print(f"  调整类型: {adj.adjustment_type}")
            print(f"  原因: {adj.reason}")
            print(f"  涉及指代数量: {len(adj.corefs)}")
            print(f"  涉及SRL论元数量: {len(adj.srl_args)}")

if __name__ == "__main__":
    main()