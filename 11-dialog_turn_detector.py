# -*- coding: utf-8 -*-
"""对话轮次检测器

实现段落边界检测增强，结合句子边界、说话人切换判定段落与对话轮次完整。
支持句子分割、说话人边界检测、对话轮次识别等功能。
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

@dataclass
class DialogChunk:
    """表示一个对话轮次分块及其元数据
    
    属性:
        text: 块文本内容
        speaker: 说话人
        start_index: 在原文中的起始位置
        end_index: 在原文中的结束位置
        chunk_id: 块ID
        tokens: 分词结果（可选）
    """
    text: str
    speaker: str
    start_index: int
    end_index: int
    chunk_id: int = 0
    tokens: Optional[List[str]] = None

class DialogTurnDetector:
    """对话轮次检测器
    
    该类实现了对话轮次的自动检测和分块功能，通过结合句子边界和说话人切换来确定段落边界。
    主要用于RAG系统中处理对话类型的数据，提高对话上下文理解和检索精度。
    """
    
    def __init__(self):
        """初始化对话轮次检测器
        
        初始化用于分句和识别说话人的正则表达式模式。
        """
        # 用于分句的正则表达式 - 匹配中文和英文的句末标点符号
        self.sentence_pattern = re.compile(r'(?<=[。！？.?!])\s*')
        # 用于识别说话人的正则表达式 - 匹配"说话人:内容"格式
        self.speaker_pattern = re.compile(r'^([^:：]+)[：:]\s*')
    
    def read_dialog_from_text(self, file_path: str) -> List[Dict[str, Any]]:
        """从文本文件中读取对话内容
        
        参数:
            file_path: 文本文件路径
        
        返回:
            对话内容列表，每个元素包含说话人和对话内容
        
        异常:
            当文件读取失败时抛出异常
        """
        dialogs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 按行分割对话
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 提取说话人和对话内容
                match = self.speaker_pattern.match(line)
                if match:
                    speaker = match.group(1)
                    text = line[match.end():].strip()
                    dialogs.append({
                        "speaker": speaker,
                        "text": text
                    })
                else:
                    # 如果没有明确的说话人标记，使用默认说话人
                    dialogs.append({
                        "speaker": "未知",
                        "text": line
                    })
        except Exception as e:
            print(f"读取对话文件时出错: {e}")
            raise
        
        return dialogs
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子
        
        参数:
            text: 输入文本
        
        返回:
            句子列表
        
        功能:
            使用正则表达式根据标点符号将文本分割为多个句子，并过滤掉空句子。
        """
        sentences = self.sentence_pattern.split(text)
        # 过滤掉空句子
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def detect_dialog_turns(self, dialogs: List[Dict[str, Any]]) -> List[DialogChunk]:
        """检测对话轮次
        
        参数:
            dialogs: 对话内容列表
        
        返回:
            DialogChunk对象列表
        
        功能:
            核心算法实现，根据说话人切换和句子边界来确定对话轮次。
            当说话人发生变化时，创建新的对话轮次。
        """
        chunks = []
        chunk_id = 0
        current_text = ""
        current_speaker = None
        current_start = 0
        
        for dialog in dialogs:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # 分句
            sentences = self.split_into_sentences(text)
            
            for sentence in sentences:
                # 如果是新的说话人，结束当前轮次并开始新轮次
                if current_speaker is not None and speaker != current_speaker:
                    # 保存当前轮次
                    chunks.append(DialogChunk(
                        text=current_text, 
                        speaker=current_speaker, 
                        start_index=current_start, 
                        end_index=current_start + len(current_text),
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    
                    # 开始新轮次
                    current_text = sentence
                    current_speaker = speaker
                    current_start = chunks[-1].end_index + 1 if chunks else 0
                else:
                    # 同一说话人，继续当前轮次
                    if current_text:
                        current_text += " " + sentence
                    else:
                        current_text = sentence
                        current_speaker = speaker
                        current_start = chunks[-1].end_index + 1 if chunks else 0
        
        # 保存最后一个轮次
        if current_text:
            chunks.append(DialogChunk(
                text=current_text, 
                speaker=current_speaker, 
                start_index=current_start, 
                end_index=current_start + len(current_text),
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def convert_to_json(self, chunks: List[DialogChunk]) -> List[Dict]:
        """将DialogChunk对象列表转换为JSON可序列化的字典列表
        
        参数:
            chunks: DialogChunk对象列表
        
        返回:
            可序列化的字典列表
        
        功能:
            将数据类对象转换为字典，并过滤掉值为None的字段，便于JSON序列化。
        """
        return [{k: v for k, v in chunk.__dict__.items() if v is not None} for chunk in chunks]
    
    def save_chunks_to_json(self, chunks: List[DialogChunk], output_path: str) -> None:
        """将块保存为JSON文件
        
        参数:
            chunks: DialogChunk对象列表
            output_path: 输出文件路径
        
        功能:
            将对话轮次分块结果保存为格式化的JSON文件，方便后续处理和分析。
        """
        json_data = self.convert_to_json(chunks)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        print(f"对话轮次分块结果已保存到: {output_path}")
    
    def print_chunk_summary(self, chunks: List[DialogChunk]) -> None:
        """打印分块摘要信息
        
        参数:
            chunks: DialogChunk对象列表
        
        功能:
            打印对话轮次的统计信息，包括总轮次数和各说话人的轮次分布。
        """
        print(f"总轮次数: {len(chunks)}")
        
        # 按说话人统计轮次数
        speaker_counts = {}
        for chunk in chunks:
            speaker_counts[chunk.speaker] = speaker_counts.get(chunk.speaker, 0) + 1
        print("说话人轮次统计:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count}轮")
    
    def process_dialog_file(self, input_path: str, output_path: str) -> List[DialogChunk]:
        """处理对话文件的主方法
        
        参数:
            input_path: 输入文件路径
            output_path: 输出文件路径
        
        返回:
            DialogChunk对象列表
        
        功能:
            整合各个步骤，完成从对话文件读取到结果保存的完整处理流程。
        """
        # 读取对话内容
        print(f"正在读取对话文件: {input_path}")
        dialogs = self.read_dialog_from_text(input_path)
        
        # 检测对话轮次
        print("正在检测对话轮次...")
        chunks = self.detect_dialog_turns(dialogs)
        
        # 打印摘要
        self.print_chunk_summary(chunks)
        
        # 保存结果
        self.save_chunks_to_json(chunks, output_path)
        
        return chunks

# 安装必要的依赖
def install_dependencies():
    """安装必要的依赖库
    
    功能:
        检查并安装程序运行所需的依赖库。
        注意：本程序主要使用Python标准库，无需额外依赖。
    """
    # 本程序主要使用Python标准库，无需额外依赖
    print("所有必要的依赖已安装")

# 主函数
def main():
    """主函数，演示对话轮次检测功能
    
    功能:
        提供完整的程序运行示例，包括依赖安装、文件检查、对话处理等步骤。
    """
    print("对话轮次检测器演示")
    print("=" * 50)
    
    # 安装必要的依赖
    install_dependencies()
    
    # 对话文件路径
    input_path = "data/medical_ai_dialog.txt"
    output_path = "results/dialog_turn_chunks.json"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 对话文件不存在: {input_path}")
        print("请确保文件路径正确")
        return
    
    # 初始化对话轮次检测器
    detector = DialogTurnDetector()
    
    # 处理对话文件
    detector.process_dialog_file(input_path, output_path)
    
    print("\n对话轮次检测演示完成！")

if __name__ == "__main__":
    main()