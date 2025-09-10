#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块溯源与层级索引实现

功能：
- 构建段落→句子→短语三层索引结构
- 支持反向定位与上下文回溯
- 实现多层向量索引和父子关系映射
- 支持精确的内容定位和上下文重建
- 提供完整的内容溯源路径

使用示例：
```python
from pyramid_index_with_provenance import PyramidIndex

# 创建索引实例
index = PyramidIndex()

# 构建三层索引
text = "你的示例文本内容..."
index.build_index(text)

# 搜索内容
results = index.search("搜索关键词", top_k=3)

# 查看搜索结果及溯源路径
for result in results:
    print(f"\n命中内容: {result['content']}")
    print(f"溯源路径: {result['provenance_path']}")
    print(f"上下文: {result['context']}")
```

依赖：
- Python 3.7+
- typing (标准库)
- re (标准库)
- json (标准库)
- collections (标准库)
- datetime (标准库)
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime


class TextChunk:
    """文本块基类，表示分层索引中的一个基本单元"""
    def __init__(self, chunk_id: str, content: str, chunk_type: str, 
                 start_pos: int, end_pos: int):
        self.chunk_id = chunk_id        # 块唯一ID
        self.content = content          # 块内容
        self.chunk_type = chunk_type    # 块类型: paragraph, sentence, phrase
        self.start_pos = start_pos      # 在原文中的起始位置
        self.end_pos = end_pos          # 在原文中的结束位置
        self.parent_id: Optional[str] = None  # 父块ID
        self.children_ids: List[str] = []     # 子块ID列表
        self.metadata: Dict[str, Any] = {}    # 额外元数据
        self.created_at = datetime.now().isoformat()  # 创建时间
        
    def to_dict(self) -> Dict[str, Any]:
        """将文本块转换为字典格式，用于序列化存储"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class PyramidIndex:
    """分块溯源与层级索引实现类
    
    实现段落→句子→短语三层索引结构，支持反向定位与上下文回溯。
    通过多层索引结构，实现从粗粒度到细粒度的文本检索，同时保持文本块间的父子关系，
    支持精确的内容定位和完整的溯源路径。
    """
    def __init__(self):
        # 存储所有文本块，键为chunk_id
        self.chunks: Dict[str, TextChunk] = {}
        
        # 分层存储各类型的块ID
        self.paragraph_ids: List[str] = []  # 段落ID列表
        self.sentence_ids: List[str] = []   # 句子ID列表
        self.phrase_ids: List[str] = []     # 短语ID列表
        
        # 父子关系映射，用于快速查询
        self.parent_map: Dict[str, str] = {}           # 子块ID -> 父块ID
        self.children_map: Dict[str, List[str]] = defaultdict(list)  # 父块ID -> 子块ID列表
        
        # 位置索引，用于快速定位
        self.position_index: Dict[Tuple[int, int], str] = {}  # (start_pos, end_pos) -> chunk_id
        
        # 文本检索索引（简化版，实际应用中可替换为向量索引）
        self.text_index: Dict[str, List[str]] = defaultdict(list)  # 关键词 -> 块ID列表
        
        # 文档原文
        self.original_text: str = ""
        
        # 统计信息
        self.stats = {
            "paragraph_count": 0,  # 段落数量
            "sentence_count": 0,   # 句子数量
            "phrase_count": 0,     # 短语数量
            "total_chunks": 0      # 总块数量
        }
    
    def build_index(self, text: str, 
                   paragraph_separator: str = '\n\n',
                   phrase_min_length: int = 3, 
                   phrase_max_length: int = 10) -> None:
        """
        构建三层索引结构
        
        参数:
            text: 输入文本
            paragraph_separator: 段落分隔符
            phrase_min_length: 短语最小长度（字符）
            phrase_max_length: 短语最大长度（字符）
        """
        self.original_text = text
        
        # 第一步：分段落
        self._split_into_paragraphs(text, paragraph_separator)
        
        # 第二步：分句子
        self._split_into_sentences(phrase_min_length, phrase_max_length)
        
        # 第三步：构建索引
        self._build_text_index()
        
        # 更新统计信息
        self.stats = {
            "paragraph_count": len(self.paragraph_ids),
            "sentence_count": len(self.sentence_ids),
            "phrase_count": len(self.phrase_ids),
            "total_chunks": len(self.chunks)
        }
        
        print(f"索引构建完成：段落{len(self.paragraph_ids)}个，句子{len(self.sentence_ids)}个，短语{len(self.phrase_ids)}个")
    
    def _split_into_paragraphs(self, text: str, separator: str) -> None:
        """将文本分割为段落"""
        paragraphs = text.split(separator)
        
        for i, paragraph in enumerate(paragraphs):
            # 跳过空段落
            if not paragraph.strip():
                continue
                
            # 计算在原文中的位置
            start_pos = text.find(paragraph)
            end_pos = start_pos + len(paragraph)
            
            # 创建段落块
            chunk_id = f"paragraph_{i+1:03d}"
            paragraph_chunk = TextChunk(
                chunk_id=chunk_id,
                content=paragraph.strip(),
                chunk_type="paragraph",
                start_pos=start_pos,
                end_pos=end_pos
            )
            
            # 添加到存储
            self.chunks[chunk_id] = paragraph_chunk
            self.paragraph_ids.append(chunk_id)
            self.position_index[(start_pos, end_pos)] = chunk_id
    
    def _split_into_sentences(self, min_length: int, max_length: int) -> None:
        """将段落分割为句子，再将句子分割为短语"""
        # 中文句子分割正则表达式
        sentence_pattern = r'(.*?[。！？；.!?;])'
        
        sentence_counter = 1
        phrase_counter = 1
        
        for para_id in self.paragraph_ids:
            paragraph = self.chunks[para_id].content
            paragraph_start = self.chunks[para_id].start_pos
            
            # 分割句子
            sentences = re.findall(sentence_pattern, paragraph)
            if not sentences:  # 如果没有匹配到句子模式，将整个段落作为一个句子
                sentences = [paragraph]
            
            for sentence in sentences:
                # 跳过空句子
                if not sentence.strip():
                    continue
                
                # 计算在原文中的位置
                sent_start_in_para = paragraph.find(sentence)
                if sent_start_in_para == -1:
                    sent_start_in_para = 0  # 找不到时使用0作为起始位置
                
                start_pos = paragraph_start + sent_start_in_para
                end_pos = start_pos + len(sentence)
                
                # 创建句子块
                sent_id = f"sentence_{sentence_counter:04d}"
                sentence_counter += 1
                
                sentence_chunk = TextChunk(
                    chunk_id=sent_id,
                    content=sentence.strip(),
                    chunk_type="sentence",
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                
                # 设置父子关系
                sentence_chunk.parent_id = para_id
                self.parent_map[sent_id] = para_id
                self.children_map[para_id].append(sent_id)
                self.chunks[para_id].children_ids.append(sent_id)
                
                # 添加到存储
                self.chunks[sent_id] = sentence_chunk
                self.sentence_ids.append(sent_id)
                self.position_index[(start_pos, end_pos)] = sent_id
                
                # 分割短语（这里使用简单的窗口滑动方法）
                self._split_into_phrases(sentence, sent_id, start_pos, min_length, max_length, phrase_counter)
                phrase_counter += max(1, len(sentence) // (max_length - min_length + 1))
    
    def _split_into_phrases(self, sentence: str, parent_id: str, 
                           sentence_start: int, min_len: int, 
                           max_len: int, counter: int) -> None:
        """将句子分割为短语"""
        # 使用简单的滑动窗口方法生成短语
        sentence = sentence.strip()
        phrase_ids = []
        
        for i in range(0, len(sentence) - min_len + 1, min(5, min_len)):
            # 调整窗口大小，确保不超过句子长度
            window_size = min(max_len, len(sentence) - i)
            phrase_text = sentence[i:i+window_size]
            
            # 跳过纯空白或标点的短语
            if not phrase_text.strip() or re.match(r'^[\s，,。.！!?？；;]+$', phrase_text):
                continue
            
            # 创建短语块
            phrase_id = f"phrase_{counter:05d}"
            counter += 1
            
            start_pos = sentence_start + i
            end_pos = start_pos + len(phrase_text)
            
            phrase_chunk = TextChunk(
                chunk_id=phrase_id,
                content=phrase_text.strip(),
                chunk_type="phrase",
                start_pos=start_pos,
                end_pos=end_pos
            )
            
            # 设置父子关系
            phrase_chunk.parent_id = parent_id
            self.parent_map[phrase_id] = parent_id
            self.children_map[parent_id].append(phrase_id)
            self.chunks[parent_id].children_ids.append(phrase_id)
            
            # 添加到存储
            self.chunks[phrase_id] = phrase_chunk
            self.phrase_ids.append(phrase_id)
            self.position_index[(start_pos, end_pos)] = phrase_id
            phrase_ids.append(phrase_id)
    
    def _build_text_index(self) -> None:
        """构建文本检索索引（简化版）"""
        for chunk_id, chunk in self.chunks.items():
            # 提取关键词（这里使用简单的分词方法）
            keywords = self._extract_keywords(chunk.content)
            
            # 将块ID添加到每个关键词的索引中
            for keyword in keywords:
                self.text_index[keyword].append(chunk_id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本中的关键词（简化版）"""
        # 移除非中文字符和空白字符
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', ' ', text)
        # 分割成单词
        words = cleaned_text.split()
        # 返回非空单词
        return [word for word in words if word]
    
    def search(self, query: str, top_k: int = 3, 
               include_context: bool = True) -> List[Dict[str, Any]]:
        """
        搜索相关内容，支持三层索引查询路径（短语→句子→段落）
        
        参数:
            query: 搜索查询词
            top_k: 返回结果数量
            include_context: 是否包含上下文信息
            
        返回:
            搜索结果列表，每个结果包含内容、溯源路径和上下文
        """
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 存储匹配结果
        matched_chunks = set()
        
        # 第一步：搜索短语层（最细粒度）
        for keyword in query_keywords:
            if keyword in self.text_index:
                # 只考虑短语类型的块
                phrase_matches = [chunk_id for chunk_id in self.text_index[keyword] 
                                 if self.chunks[chunk_id].chunk_type == "phrase"]
                matched_chunks.update(phrase_matches)
        
        # 如果短语层没有足够的匹配，搜索句子层
        if len(matched_chunks) < top_k:
            for keyword in query_keywords:
                if keyword in self.text_index:
                    # 考虑句子类型的块
                    sentence_matches = [chunk_id for chunk_id in self.text_index[keyword] 
                                      if self.chunks[chunk_id].chunk_type == "sentence"]
                    matched_chunks.update(sentence_matches)
        
        # 如果仍然没有足够的匹配，搜索段落层
        if len(matched_chunks) < top_k:
            for keyword in query_keywords:
                if keyword in self.text_index:
                    # 考虑段落类型的块
                    paragraph_matches = [chunk_id for chunk_id in self.text_index[keyword] 
                                      if self.chunks[chunk_id].chunk_type == "paragraph"]
                    matched_chunks.update(paragraph_matches)
        
        # 转换为列表并排序（这里使用简单的排序方式）
        results = []
        for chunk_id in matched_chunks:
            chunk = self.chunks[chunk_id]
            
            # 获取溯源路径
            provenance_path = self._get_provenance_path(chunk_id)
            
            # 获取上下文
            context = ""
            if include_context:
                context = self._get_context(chunk_id)
            
            # 计算相关性分数（简化版）
            relevance_score = self._calculate_relevance_score(chunk.content, query_keywords)
            
            results.append({
                "chunk_id": chunk_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "provenance_path": provenance_path,
                "context": context,
                "relevance_score": relevance_score,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos
            })
        
        # 按相关性分数排序并取前top_k个结果
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
    
    def _get_provenance_path(self, chunk_id: str) -> str:
        """获取块的溯源路径（XPath格式）"""
        path_parts = []
        current_id = chunk_id
        
        # 向上遍历直到根节点
        while current_id in self.parent_map:
            current_chunk = self.chunks[current_id]
            path_parts.append(f"{current_chunk.chunk_type}:{current_chunk.chunk_id}")
            current_id = self.parent_map[current_id]
        
        # 添加最顶层节点
        if current_id in self.chunks:
            current_chunk = self.chunks[current_id]
            path_parts.append(f"{current_chunk.chunk_type}:{current_chunk.chunk_id}")
        
        # 反转路径，从顶层到底层
        path_parts.reverse()
        
        # 转换为XPath格式
        return "/".join(path_parts)
    
    def _get_context(self, chunk_id: str, context_window: int = 2) -> str:
        """获取块的上下文信息
        
        根据块类型返回不同级别的上下文：
        - 短语：返回所在句子
        - 句子：返回所在段落
        - 段落：返回前后段落
        """
        chunk = self.chunks[chunk_id]
        chunk_type = chunk.chunk_type
        
        # 根据块类型获取不同的上下文
        if chunk_type == "phrase":
            # 短语的上下文是其所在的句子
            parent_id = self.parent_map.get(chunk_id)
            if parent_id and parent_id in self.chunks:
                return self.chunks[parent_id].content
            return chunk.content
        
        elif chunk_type == "sentence":
            # 句子的上下文是其所在的段落
            parent_id = self.parent_map.get(chunk_id)
            if parent_id and parent_id in self.chunks:
                return self.chunks[parent_id].content
            return chunk.content
        
        elif chunk_type == "paragraph":
            # 段落的上下文是前后各context_window个段落
            para_index = self.paragraph_ids.index(chunk_id) if chunk_id in self.paragraph_ids else -1
            if para_index == -1:
                return chunk.content
            
            # 获取前后段落
            start_index = max(0, para_index - context_window)
            end_index = min(len(self.paragraph_ids), para_index + context_window + 1)
            
            context_paragraphs = []
            for i in range(start_index, end_index):
                if i == para_index:
                    # 当前段落用特殊标记
                    context_paragraphs.append(f"[当前段落] {self.chunks[self.paragraph_ids[i]].content}")
                else:
                    context_paragraphs.append(self.chunks[self.paragraph_ids[i]].content)
            
            return "\n\n".join(context_paragraphs)
        
        return chunk.content
    
    def _calculate_relevance_score(self, content: str, keywords: List[str]) -> float:
        """计算内容与查询关键词的相关性分数（简化版）"""
        if not keywords:
            return 0.0
        
        # 计算匹配的关键词数量
        matched_count = sum(1 for keyword in keywords if keyword.lower() in content.lower())
        
        # 计算匹配密度（匹配词数/内容长度）
        density = matched_count / max(1, len(content))
        
        # 综合分数：70%匹配覆盖率 + 30%匹配密度
        return (matched_count / len(keywords)) * 0.7 + density * 0.3
    
    def get_chunk_by_position(self, position: int) -> Optional[Dict[str, Any]]:
        """根据位置获取对应的块（反向定位功能）"""
        # 查找包含该位置的块
        for (start, end), chunk_id in self.position_index.items():
            if start <= position < end:
                chunk = self.chunks[chunk_id]
                return chunk.to_dict()
        
        return None
    
    def get_chunk_hierarchy(self, chunk_id: str) -> Dict[str, Any]:
        """获取块的层级结构"""
        if chunk_id not in self.chunks:
            return {}
        
        chunk = self.chunks[chunk_id]
        result = chunk.to_dict()
        
        # 添加子块信息
        if chunk_id in self.children_map:
            result["children"] = []
            for child_id in self.children_map[chunk_id]:
                child_chunk = self.chunks[child_id]
                result["children"].append({
                    "chunk_id": child_chunk.chunk_id,
                    "chunk_type": child_chunk.chunk_type,
                    "content": child_chunk.content[:50] + ("..." if len(child_chunk.content) > 50 else "")
                })
        
        # 添加父块信息
        if chunk_id in self.parent_map:
            parent_id = self.parent_map[chunk_id]
            if parent_id in self.chunks:
                parent_chunk = self.chunks[parent_id]
                result["parent"] = {
                    "chunk_id": parent_chunk.chunk_id,
                    "chunk_type": parent_chunk.chunk_type,
                    "content": parent_chunk.content[:50] + ("..." if len(parent_chunk.content) > 50 else "")
                }
        
        return result
    
    def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        index_data = {
            "chunks": {cid: chunk.to_dict() for cid, chunk in self.chunks.items()},
            "paragraph_ids": self.paragraph_ids,
            "sentence_ids": self.sentence_ids,
            "phrase_ids": self.phrase_ids,
            "parent_map": self.parent_map,
            "children_map": dict(self.children_map),
            "original_text": self.original_text,
            "stats": self.stats,
            "created_at": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            print(f"索引已成功保存到: {file_path}")
        except Exception as e:
            print(f"保存索引时出错: {e}")
    
    def load_index(self, file_path: str) -> bool:
        """从文件加载索引"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 恢复块数据
            self.chunks = {}
            for cid, chunk_data in index_data["chunks"].items():
                chunk = TextChunk(
                    chunk_id=chunk_data["chunk_id"],
                    content=chunk_data["content"],
                    chunk_type=chunk_data["chunk_type"],
                    start_pos=chunk_data["start_pos"],
                    end_pos=chunk_data["end_pos"]
                )
                chunk.parent_id = chunk_data.get("parent_id")
                chunk.children_ids = chunk_data.get("children_ids", [])
                chunk.metadata = chunk_data.get("metadata", {})
                self.chunks[cid] = chunk
            
            # 恢复其他数据
            self.paragraph_ids = index_data["paragraph_ids"]
            self.sentence_ids = index_data["sentence_ids"]
            self.phrase_ids = index_data["phrase_ids"]
            self.parent_map = index_data["parent_map"]
            self.children_map = defaultdict(list, index_data["children_map"])
            self.original_text = index_data["original_text"]
            self.stats = index_data["stats"]
            
            # 重建位置索引
            self.position_index = {}
            for chunk in self.chunks.values():
                self.position_index[(chunk.start_pos, chunk.end_pos)] = chunk.chunk_id
            
            # 重建文本索引
            self.text_index = defaultdict(list)
            self._build_text_index()
            
            print(f"索引已成功从: {file_path} 加载")
            return True
        except Exception as e:
            print(f"加载索引时出错: {e}")
            return False


# 示例用法
if __name__ == "__main__":
    # 示例文本 - 关于知识产权的中文内容
    sample_text = """
    知识产权是指人们就其智力劳动成果所依法享有的专有权利，通常是国家赋予创造者对其智力成果在一定时期内享有的专有权或独占权。
    
    知识产权从本质上说是一种无形财产权，它的客体是智力成果或是知识产品，是一种无形财产或者一种没有形体的精神财富，是创造性的智力劳动所创造的劳动成果。
    
    知识产权具有专有性、时间性和地域性的特点。专有性是指知识产权的所有人对其智力成果具有排他性的权利；时间性是指知识产权只在规定期限内受到法律保护；地域性是指知识产权只在授权范围内有效。
    
    知识产权主要包括专利权、商标权、著作权（版权）、商业秘密、植物新品种权、集成电路布图设计专有权等。
    
    随着科技的发展，知识产权保护制度不断完善，对促进创新和经济发展起到了重要作用。
    """
    
    print("===== 分块溯源与层级索引示例 =====")
    
    # 创建索引实例
    index = PyramidIndex()
    
    # 构建索引
    print("\n1. 构建三层索引...")
    index.build_index(sample_text)
    
    # 搜索内容
    print("\n2. 搜索 '知识产权'...")
    results = index.search("知识产权", top_k=3)
    
    # 显示搜索结果
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  类型: {result['chunk_type']}")
        print(f"  内容: {result['content']}")
        print(f"  溯源路径: {result['provenance_path']}")
        print(f"  相关性: {result['relevance_score']:.2f}")
        print(f"  位置: {result['start_pos']}-{result['end_pos']}")
    
    # 测试位置定位
    print("\n3. 测试位置定位...")
    position = 100
    chunk_at_pos = index.get_chunk_by_position(position)
    if chunk_at_pos:
        print(f"  在位置 {position} 找到块: {chunk_at_pos['chunk_id']} ({chunk_at_pos['chunk_type']})")
        print(f"  内容: {chunk_at_pos['content']}")
    
    # 获取块的层级结构
    print("\n4. 获取块的层级结构...")
    if results:
        first_result_id = results[0]['chunk_id']
        hierarchy = index.get_chunk_hierarchy(first_result_id)
        print(f"  块 {first_result_id} 的层级结构:")
        print(f"  类型: {hierarchy['chunk_type']}")
        print(f"  父块: {hierarchy.get('parent', {}).get('chunk_id', '无')}")
        print(f"  子块数量: {len(hierarchy.get('children', []))}")
    
    # 保存索引
    print("\n5. 保存索引到文件...")
    index.save_index("pyramid_index_example.json")
    
    print("\n===== 示例完成 =====")