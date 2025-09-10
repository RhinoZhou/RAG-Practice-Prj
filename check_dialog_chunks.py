# -*- coding: utf-8 -*-
"""检查对话轮次分块结果"""

import json
import os


def check_dialog_chunks(json_path: str) -> None:
    """检查对话轮次分块结果
    
    参数:
        json_path: JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: JSON文件不存在: {json_path}")
        return
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"对话轮次分块检查结果")
    print("=" * 50)
    print(f"总轮次数: {len(chunks)}")
    
    # 检查说话人分布
    speaker_counts = {}
    for chunk in chunks:
        speaker = chunk["speaker"]
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    print("说话人轮次统计:")
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count}轮")
    
    # 检查索引是否连续
    indexes_ok = True
    prev_end = -1
    for i, chunk in enumerate(chunks):
        start = chunk["start_index"]
        end = chunk["end_index"]
        
        # 检查起始索引是否正确
        if i == 0 and start != 0:
            indexes_ok = False
            print(f"警告: 第一个块的起始索引应为0，但实际为{start}")
        elif i > 0 and start != prev_end + 1:
            indexes_ok = False
            print(f"警告: 第{i}个块的起始索引应为{prev_end + 1}，但实际为{start}")
        
        # 检查文本长度是否与索引差一致
        text_len = len(chunk["text"])
        if end - start != text_len:
            indexes_ok = False
            print(f"警告: 第{i}个块的文本长度({text_len})与索引差({end - start})不一致")
        
        prev_end = end
    
    if indexes_ok:
        print("索引检查通过: 所有块的索引连续且文本长度与索引差一致")
    
    # 显示前几个块的详细信息
    print("\n前3个对话轮次详情:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"轮次{i} ({chunk['speaker']}):")
        print(f"  文本: {chunk['text']}")
        print(f"  位置: {chunk['start_index']}-{chunk['end_index']}")
        print()
    
    print("对话轮次分块检查完成！")


if __name__ == "__main__":
    json_path = "results/dialog_turn_chunks.json"
    check_dialog_chunks(json_path)