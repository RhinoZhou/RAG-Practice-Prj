#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体密度自适应分块器

该模块实现了一个基于命名实体识别(NER)密度的自适应分块策略，主要功能包括：
1. 执行NER实体抽取（基于规则和可选的spaCy增强）
2. 计算每百字的实体密度
3. 根据实体密度动态设置分块大小（密度高则变小）
4. 使用滑窗切分策略进行文本分块
5. 输出包含实体密度和实际使用分块大小的元数据

适用于包含大量领域实体的专业文档分块，如医疗、法律、金融等领域的文本处理。
"""

import json
import os
import re
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Set
import time

# 尝试导入必要的库，如果失败则记录但继续执行
HAS_TIKTOKEN = False
try:
    import tiktoken
    HAS_TIKTOKEN = True
    print("成功导入tiktoken库")
except ImportError:
    print("未找到tiktoken库，将使用简单token计数")

HAS_SPACY = False
SPACY_NLP = None
try:
    import spacy
    HAS_SPACY = True
    # 尝试加载中文模型，失败则使用规则匹配
    try:
        SPACY_NLP = spacy.load("zh_core_web_sm")
        print("成功加载spaCy中文模型")
    except:
        HAS_SPACY = False
        print("无法加载spaCy中文模型，将使用规则匹配")
except ImportError:
    print("未找到spaCy库，将使用规则匹配")

# 进度条显示函数
def show_progress(current: int, total: int, description: str = "处理中") -> None:
    """显示进度条"""
    bar_length = 30
    progress = current / total if total > 0 else 0
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = progress * 100
    print(f'\r{description}: [{bar}] {percent:.1f}% ({current}/{total})', end='')
    if current >= total:
        print()

@dataclass
class Entity:
    """表示一个实体的数据类"""
    text: str  # 实体文本
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    entity_type: str  # 实体类型
    density_score: float = 0.0  # 实体密度得分

@dataclass
class NERChunk:
    """表示一个基于实体密度的分块的数据类"""
    chunk_id: int
    text: str
    start_index: int
    end_index: int
    tokens_count: int
    entities: List[Dict] = field(default_factory=list)  # 实体列表
    entity_density: float = 0.0  # 每百字实体数量
    size_used: int = 0  # 实际使用的分块大小（tokens）
    overlap_with_prev: int = 0  # 与前一分块的重叠字符数

class SimpleTokenizer:
    """简单的字符级tokenizer"""
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(text)

class NERDensityAdaptiveChunker:
    """实体密度自适应分块器"""
    def __init__(self,
                 default_max_tokens: int = 500,
                 min_max_tokens: int = 100,
                 max_max_tokens: int = 1000,
                 overlap_ratio: float = 0.2,
                 density_thresholds: List[float] = None,
                 tokenizer_name: str = "cl100k_base",
                 use_simple_tokenizer: bool = False,
                 enable_progress: bool = True):
        """
        初始化实体密度自适应分块器
        
        参数:
        - default_max_tokens: 默认最大token数量
        - min_max_tokens: 最小最大token数量（高密度区域使用）
        - max_max_tokens: 最大最大token数量（低密度区域使用）
        - overlap_ratio: 重叠比例
        - density_thresholds: 密度阈值列表，用于映射密度到分块大小
        - tokenizer_name: 使用的tokenizer名称
        - use_simple_tokenizer: 是否强制使用简单的字符tokenizer
        - enable_progress: 是否显示进度条
        """
        self.default_max_tokens = default_max_tokens
        self.min_max_tokens = min_max_tokens
        self.max_max_tokens = max_max_tokens
        self.overlap_ratio = overlap_ratio
        self.enable_progress = enable_progress
        
        # 默认密度阈值：低、中、高
        self.density_thresholds = density_thresholds or [0.5, 1.5, 3.0]
        
        # 加载tokenizer
        self.use_simple_tokenizer = use_simple_tokenizer or not HAS_TIKTOKEN
        if self.use_simple_tokenizer:
            self.tokenizer = SimpleTokenizer()
            print("使用简单字符tokenizer")
        else:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_name)
                print(f"使用tiktoken tokenizer: {tokenizer_name}")
            except:
                self.tokenizer = SimpleTokenizer()
                self.use_simple_tokenizer = True
                print("无法加载指定tokenizer，使用简单字符tokenizer")
        
        # 领域实体词典（可以根据实际需求扩展）
        self.domain_entities = {
            # 医疗领域实体示例
            "疾病": ["脑梗死", "脑出血", "心脏病", "糖尿病", "高血压", "肿瘤"],
            "症状": ["头痛", "恶心", "呕吐", "乏力", "发热", "咳嗽"],
            "药物": ["阿司匹林", "青霉素", "布洛芬", "胰岛素", "降压药"],
            "部位": ["大脑中动脉", "基底节区", "额叶", "颞叶", "心脏", "肺部"],
            
            # 科研领域实体示例
            "方法": ["实验方法", "研究方法", "问卷调查", "统计分析"],
            "结果": ["研究结果", "实验结果", "统计结果"],
            "对象": ["研究对象", "受试者", "样本"],
        }
        
        # 预编译正则表达式模式以提高性能
        self.pattern_cache = {}
        for entity_type, patterns in self.domain_entities.items():
            compiled_patterns = []
            for pattern in patterns:
                # 简单文本匹配，使用预编译正则表达式提高性能
                compiled_patterns.append(re.compile(re.escape(pattern)))
            self.pattern_cache[entity_type] = compiled_patterns
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.use_simple_tokenizer:
            return self.tokenizer.count_tokens(text)
        else:
            return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取实体
        
        使用规则匹配和可选的spaCy增强来识别文本中的实体
        """
        start_time = time.time()
        entities = []
        entity_positions = set()  # 用于去重
        
        # 基于规则的实体抽取
        for entity_type, compiled_patterns in self.pattern_cache.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    # 检查是否重复
                    pos_key = (match.start(), match.end())
                    if pos_key not in entity_positions:
                        entity_positions.add(pos_key)
                        entities.append(Entity(
                            text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            entity_type=entity_type
                        ))
        
        # 如果有spacy，使用其NER功能增强结果
        if HAS_SPACY and SPACY_NLP:
            try:
                doc = SPACY_NLP(text)
                for ent in doc.ents:
                    pos_key = (ent.start_char, ent.end_char)
                    if pos_key not in entity_positions:
                        entity_positions.add(pos_key)
                        entities.append(Entity(
                            text=ent.text,
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            entity_type=ent.label_
                        ))
            except Exception as e:
                print(f"使用spaCy增强实体抽取时出错: {e}")
        
        # 按起始位置排序
        entities.sort(key=lambda x: x.start_pos)
        
        if self.enable_progress:
            print(f"实体抽取完成，耗时: {time.time() - start_time:.3f}秒，共抽取 {len(entities)} 个实体")
        
        return entities
    
    def _calculate_entity_density(self, text: str, entities: List[Entity]) -> float:
        """计算文本的实体密度（每百字实体数量）"""
        if not text or len(text) == 0:
            return 0.0
        
        # 计算文本字数
        char_count = len(text)
        
        # 计算实体数量（去重已在_extract_entities中完成）
        entity_count = len(entities)
        
        # 计算每百字实体密度
        density = (entity_count / char_count) * 100 if char_count > 0 else 0
        
        return density
    
    def _determine_chunk_size(self, density: float) -> int:
        """根据实体密度确定分块大小
        
        实体密度越高，分块大小越小
        """
        # 如果密度低于最低阈值，使用最大分块大小
        if density <= self.density_thresholds[0]:
            return self.max_max_tokens
        # 如果密度高于最高阈值，使用最小分块大小
        elif density >= self.density_thresholds[-1]:
            return self.min_max_tokens
        else:
            # 线性映射密度到分块大小
            thresholds = self.density_thresholds
            for i in range(len(thresholds) - 1):
                if thresholds[i] < density < thresholds[i+1]:
                    # 计算当前区间的比例
                    ratio = (density - thresholds[i]) / (thresholds[i+1] - thresholds[i])
                    # 映射到分块大小（注意：密度越高，分块越小）
                    chunk_size_range = self.max_max_tokens - self.min_max_tokens
                    size_segment = chunk_size_range / (len(thresholds) - 1)
                    adjusted_size = self.max_max_tokens - (i * size_segment + ratio * size_segment)
                    return max(self.min_max_tokens, min(self.max_max_tokens, int(adjusted_size)))
            
            # 默认返回中间值
            return self.default_max_tokens
    
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
    
    def _find_optimal_chunk_boundary(self, text: str, start: int, target_size: int) -> int:
        """寻找最佳的分块边界，优先在句子边界处分割"""
        # 计算目标结束位置
        end = min(start + target_size, len(text))
        
        # 如果已经是文本末尾，直接返回
        if end >= len(text):
            return end
        
        # 尝试在句子边界处分割
        # 只检查目标位置附近的文本，提高性能
        search_range = min(200, len(text) - end)  # 限制搜索范围以提高性能
        sentences = self._split_into_sentences(text[start:end+search_range])
        
        if not sentences:
            return end
        
        # 寻找最接近target_size的句子结束位置
        best_end = end
        min_diff = abs(end - start - target_size)
        
        for sentence, sent_start, sent_end in sentences:
            actual_sent_end = start + sent_end
            if actual_sent_end > end:  # 只考虑在目标位置之前或附近的句子边界
                diff = abs(actual_sent_end - start - target_size)
                if diff < min_diff and actual_sent_end - start > target_size * 0.7:  # 确保不会太小
                    min_diff = diff
                    best_end = actual_sent_end
                    break
        
        return best_end
    
    def chunk_text(self, text: str) -> List[NERChunk]:
        """根据实体密度对文本进行自适应分块"""
        if not text:
            return []
        
        start_time = time.time()
        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        
        # 提取整个文本的实体信息
        all_entities = self._extract_entities(text)
        
        # 定义滑动窗口大小（用于计算局部密度）
        window_size = min(200, text_length // 3)  # 减小窗口大小以提高性能
        if window_size < 50:
            window_size = 50  # 确保窗口大小不会太小
        
        print(f"开始分块处理，文本长度: {text_length}字符，窗口大小: {window_size}")
        
        # 最大迭代次数，防止无限循环
        max_iterations = max(20, text_length // 50)  # 限制最大迭代次数
        iteration = 0
        
        while start < text_length and iteration < max_iterations:
            iteration += 1
            
            # 显示进度
            if self.enable_progress:
                show_progress(start, text_length, "分块进度")
            
            # 确定当前窗口的文本
            window_end = min(start + window_size, text_length)
            window_text = text[start:window_end]
            
            # 提取窗口内的实体（优化：使用生成器表达式而不是列表推导式）
            window_entities = [e for e in all_entities if start <= e.start_pos < window_end]
            
            # 计算实体密度
            entity_density = self._calculate_entity_density(window_text, window_entities)
            
            # 根据密度确定分块大小
            chunk_size = self._determine_chunk_size(entity_density)
            
            # 找到最佳分块边界
            end = self._find_optimal_chunk_boundary(text, start, chunk_size)
            
            # 提取当前分块的文本
            chunk_text = text[start:end]
            
            # 提取当前分块的实体
            chunk_entities = [e for e in all_entities if start <= e.start_pos < end]
            
            # 计算分块的token数量
            tokens_count = self._count_tokens(chunk_text)
            
            # 计算与前一分块的重叠
            overlap_with_prev = 0
            if chunks:
                # 计算重叠字符数
                prev_chunk = chunks[-1]
                overlap_with_prev = max(0, start - prev_chunk.end_index)
                if overlap_with_prev > 0:
                    overlap_with_prev = min(overlap_with_prev, int(chunk_size * self.overlap_ratio))
            
            # 创建分块对象
            chunk_entities_dict = [{
                "text": e.text,
                "start_pos": e.start_pos - start,  # 相对位置
                "end_pos": e.end_pos - start,
                "entity_type": e.entity_type
            } for e in chunk_entities]
            
            chunk = NERChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_index=start,
                end_index=end,
                tokens_count=tokens_count,
                entities=chunk_entities_dict,
                entity_density=entity_density,
                size_used=chunk_size,
                overlap_with_prev=overlap_with_prev
            )
            
            chunks.append(chunk)
            
            # 计算下一个分块的起始位置（考虑重叠）
            step_size = chunk_size - int(chunk_size * self.overlap_ratio)
            next_start = end - int(chunk_size * self.overlap_ratio)
            
            # 防止无限循环：如果下一个起始位置没有明显前进，则强制前进
            if next_start <= start:
                next_start = start + max(50, chunk_size // 2)  # 确保至少前进一定距离
            
            start = next_start
            
            # 更新分块ID
            chunk_id += 1
        
        # 如果因为达到最大迭代次数而退出，确保处理完剩余文本
        if start < text_length:
            # 处理剩余文本
            remaining_text = text[start:text_length]
            remaining_entities = [e for e in all_entities if start <= e.start_pos < text_length]
            remaining_tokens = self._count_tokens(remaining_text)
            remaining_density = self._calculate_entity_density(remaining_text, remaining_entities)
            
            remaining_chunk = NERChunk(
                chunk_id=chunk_id,
                text=remaining_text,
                start_index=start,
                end_index=text_length,
                tokens_count=remaining_tokens,
                entities=[{
                    "text": e.text,
                    "start_pos": e.start_pos - start,
                    "end_pos": e.end_pos - start,
                    "entity_type": e.entity_type
                } for e in remaining_entities],
                entity_density=remaining_density,
                size_used=self.default_max_tokens,
                overlap_with_prev=0
            )
            chunks.append(remaining_chunk)
        
        if self.enable_progress:
            print(f"分块完成，耗时: {time.time() - start_time:.3f}秒，生成 {len(chunks)} 个分块")
        
        return chunks
    
    def save_chunks_to_json(self, chunks: List[NERChunk], output_path: str) -> None:
        """将分块结果保存为JSON文件
        
        参数:
        - chunks: 分块列表
        - output_path: 输出文件路径
        """
        start_time = time.time()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换分块为字典格式
        chunks_dict = []
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunks_dict.append(chunk_dict)
        
        # 准备完整的输出数据
        output_data = {
            "chunks": chunks_dict,
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk.tokens_count for chunk in chunks),
            "avg_entity_density": sum(chunk.entity_density for chunk in chunks) / len(chunks) if chunks else 0
        }
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        if self.enable_progress:
            print(f"分块结果已保存至: {output_path}，耗时: {time.time() - start_time:.3f}秒")

# 读取文本文件
def read_text_file(file_path: str) -> str:
    """从文本文件中读取内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 读取PDF文件（简化版）
def read_pdf_text(pdf_path: str) -> str:
    """从PDF文件中读取文本（简化实现）"""
    text = ""
    try:
        # 尝试导入PyPDF2
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except ImportError:
        print("PyPDF2库未安装，无法读取PDF文件")
    except Exception as e:
        print(f"读取PDF文件时出错: {e}")
    return text

# 主函数
def main():
    """主函数，演示实体密度自适应分块器的使用"""
    # 设置文件路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 选择输入文件（可以是文本文件或PDF文件）
    # 这里使用示例文本（强制使用，避免处理大文件）
    sample_text = """
    大脑中动脉狭窄与闭塞是导致脑梗死的重要原因。本研究回顾性分析了2019年1月至2021年12月期间在我院神经内科住院治疗的120例大脑中动脉狭窄与闭塞患者的临床资料。
    研究对象包括68例男性和52例女性，年龄范围为45-78岁，平均年龄为(62.3±8.5)岁。所有患者均经头颅CT或MRI证实为脑梗死，并经脑血管造影或CT血管造影证实存在大脑中动脉狭窄或闭塞。
    研究方法包括收集患者的一般资料、临床症状、影像学检查结果、实验室检查结果以及治疗方案和预后情况。影像学特点分析包括脑梗死的部位、范围、形态以及大脑中动脉狭窄或闭塞的程度和部位。
    发病机制研究主要探讨了大脑中动脉狭窄与闭塞导致脑梗死的病理生理过程，包括血栓形成、脑血流灌注不足、侧支循环代偿以及神经细胞损伤和坏死的机制。
    研究结果表明，大脑中动脉狭窄与闭塞患者的脑梗死部位主要位于基底节区、额叶和颞叶，其中以基底节区最为常见。大脑中动脉闭塞患者的脑梗死范围通常较狭窄患者更大，神经功能缺损更严重。
    此外，研究还发现，大脑中动脉狭窄与闭塞的程度与脑梗死的严重程度和预后密切相关。早期诊断和及时治疗对于改善患者的预后至关重要。
    结论部分指出，大脑中动脉狭窄与闭塞是导致脑梗死的重要原因，其影像学特点和发病机制具有一定的特征性。深入研究这些特征有助于提高对该病的认识和诊治水平，改善患者的预后。
    """
    
    print(f"使用示例文本进行测试，文本长度: {len(sample_text)} 字符")
    
    # 设置输出文件路径
    output_json_path = os.path.join(results_dir, "ner_density_adaptive_chunks.json")
    
    # 创建实体密度自适应分块器
    chunker = NERDensityAdaptiveChunker(
        default_max_tokens=150,  # 减小默认值以适应短文本
        min_max_tokens=50,       # 减小最小值
        max_max_tokens=250,      # 减小最大值
        overlap_ratio=0.2,
        density_thresholds=[0.5, 1.5, 3.0],
        use_simple_tokenizer=True,  # 使用简单tokenizer，提高性能
        enable_progress=True  # 启用进度显示
    )
    
    # 执行分块
    print("正在执行实体密度自适应分块...")
    start_time = time.time()
    chunks = chunker.chunk_text(sample_text)
    total_time = time.time() - start_time
    
    # 保存结果
    chunker.save_chunks_to_json(chunks, output_json_path)
    
    # 打印统计信息
    print(f"\n分块统计信息:")
    print(f"分块总数: {len(chunks)}")
    print(f"总token数: {sum(chunk.tokens_count for chunk in chunks)}")
    print(f"平均每个分块的token数: {sum(chunk.tokens_count for chunk in chunks) / len(chunks):.2f}")
    print(f"平均实体密度: {sum(chunk.entity_density for chunk in chunks) / len(chunks):.2f} 实体/百字")
    print(f"总执行时间: {total_time:.3f}秒")
    
    # 打印前3个分块的信息作为示例
    print("\n前3个分块示例:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n分块 {i+1} (tokens: {chunk.tokens_count}, 密度: {chunk.entity_density:.2f} 实体/百字, 使用大小: {chunk.size_used}):")
        # 打印前150个字符
        preview = chunk.text[:150] + ("..." if len(chunk.text) > 150 else "")
        print(f"{preview}")
        print(f"  实体数量: {len(chunk.entities)}")
        print(f"  起始位置: {chunk.start_index}, 结束位置: {chunk.end_index}")
        print(f"  与前一分块重叠: {chunk.overlap_with_prev}字符")
        
        # 打印实体信息
        if chunk.entities:
            print(f"  实体列表 (前3个):")
            for entity in chunk.entities[:3]:
                print(f"    - {entity['text']} ({entity['entity_type']}) [{entity['start_pos']}-{entity['end_pos']}]")

if __name__ == "__main__":
    main()