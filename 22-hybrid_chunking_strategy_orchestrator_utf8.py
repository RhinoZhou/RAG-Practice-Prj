#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
娣峰悎鍒嗗潡绛栫暐缂栨帓鍣?

鍔熻兘锛?
- 灏嗘粦鍔ㄧ獥鍙ｃ€佺粨鏋勮竟鐣屻€佷富棰樻娴嬨€佸疄浣撳瘑搴︾瓑淇″彿缂栨帓涓哄彲閰嶇疆绛栫暐
- 鎸夋枃妗ｇ被鍨嬪鐢ㄤ笉鍚岀殑鍒嗗潡妯℃澘锛堝悎鍚?璁烘枃/璐㈡姤/瀹㈡湇锛?
- 瀹炵幇澶氫俊鍙疯瀺鍚堜紭鍏堢骇涓庡啿绐佽В鍐虫満鍒?
- 杈撳嚭缁熶竴鐨勫潡缁撴瀯涓庣瓥鐣ユ墽琛屾棩蹇?

浣跨敤绀轰緥锛?
```python
from hybrid_chunking_strategy_orchestrator import HybridChunkingOrchestrator

# 鍒涘缓缂栨帓鍣ㄥ疄渚?
orchestrator = HybridChunkingOrchestrator()

# 鍒嗗潡澶勭悊鏂囨。
text = "浣犵殑鏂囨。鍐呭..."
result = orchestrator.orchestrate(text, doc_type="璁烘枃")

# 杈撳嚭缁撴灉
print("鍒嗗潡缁撴灉:", result['chunks'])
print("绛栫暐鏃ュ織:", result['strategy_log'])
```

渚濊禆锛?
- Python 3.7+
- typing (鏍囧噯搴?
- re (鏍囧噯搴?
- json (鏍囧噯搴?
- collections (鏍囧噯搴?
- datetime (鏍囧噯搴?
- concurrent.futures (鏍囧噯搴?
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class ChunkMetadata:
    """鍒嗗潡鍏冩暟鎹被"""
    def __init__(self, chunk_id: str, start_pos: int, end_pos: int):
        self.chunk_id = chunk_id            # 鍧楀敮涓€ID
        self.start_pos = start_pos          # 璧峰浣嶇疆
        self.end_pos = end_pos              # 缁撴潫浣嶇疆
        self.content: str = ""              # 鍧楀唴瀹?
        self.chunk_type: str = "general"    # 鍧楃被鍨?
        self.confidence: float = 0.0        # 鍒嗗潡缃俊搴?
        self.boundary_signals: Dict[str, float] = {}  # 杈圭晫淇″彿鏉ユ簮
        self.structure_path: List[str] = [] # 缁撴瀯璺緞
        self.entity_chain: List[str] = []   # 瀹炰綋閾?
        self.topic_tags: List[str] = []     # 涓婚鏍囩
        self.created_at = datetime.now().isoformat()  # 鍒涘缓鏃堕棿
        
    def to_dict(self) -> Dict[str, Any]:
        """杞崲涓哄瓧鍏告牸寮?""
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
    """绛栫暐妯℃澘绫伙紝瀹氫箟涓嶅悓鏂囨。绫诲瀷鐨勫垎鍧楃瓥鐣?""
    def __init__(self, template_name: str):
        self.template_name = template_name  # 妯℃澘鍚嶇О
        self.params: Dict[str, Any] = {}  # 妯℃澘鍙傛暟
        self.boundary_priority: List[str] = []  # 杈圭晫浼樺厛绾?
        self.conflict_resolution: Dict[str, str] = {}  # 鍐茬獊瑙ｅ喅瑙勫垯
        self.signal_processors: Dict[str, Dict[str, Any]] = {}  # 淇″彿澶勭悊鍣ㄩ厤缃?
        
        # 鏍规嵁妯℃澘鍚嶇О鍔犺浇棰勮閰嶇疆
        self._load_preset_config()
    
    def _load_preset_config(self) -> None:
        """鍔犺浇棰勮閰嶇疆"""
        if self.template_name == "鍚堝悓":
            self.params = {
                "base_chunk_size": 300,
                "min_chunk_size": 150,
                "max_chunk_size": 500,
                "overlap": 50,
                "sentence_separators": ["銆?, "锛?, "锛?, ".", "!", "?"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"]
            }
            self.boundary_priority = ["contract_section", "paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "contract_section_vs_paragraph": "contract_section",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "contract_section": {"enabled": True, "regex": r'绗琜涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+鏉?},  # 鍚堝悓鏉℃妯″紡
                "paragraph": {"enabled": True},  # 娈佃惤杈圭晫
                "sentence": {"enabled": True},  # 鍙ュ瓙杈圭晫
                "punctuation": {"enabled": True},  # 鏍囩偣绗﹀彿
                "token_count": {"enabled": True}  # 浠ょ墝璁℃暟
            }
        
        elif self.template_name == "璁烘枃":
            self.params = {
                "base_chunk_size": 400,
                "min_chunk_size": 200,
                "max_chunk_size": 600,
                "overlap": 60,
                "sentence_separators": ["銆?, "锛?, "锛?, ".", "!", "?"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"],
                "heading_patterns": ["#+", "绗?+绔?, "1\\.", "2\\."]
            }
            self.boundary_priority = ["heading", "paragraph", "sentence", "token_count"]
            self.conflict_resolution = {
                "heading_vs_paragraph": "heading",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "heading": {"enabled": True, "patterns": self.params["heading_patterns"]},  # 鏍囬妫€娴?
                "paragraph": {"enabled": True},  # 娈佃惤杈圭晫
                "sentence": {"enabled": True},  # 鍙ュ瓙杈圭晫
                "token_count": {"enabled": True}  # 浠ょ墝璁℃暟
            }
        
        elif self.template_name == "璐㈡姤":
            self.params = {
                "base_chunk_size": 250,
                "min_chunk_size": 100,
                "max_chunk_size": 400,
                "overlap": 40,
                "sentence_separators": ["銆?, "锛?, "锛?, ".", "!", "?", ";"],
                "paragraph_separators": ["\n\n", "\r\n\r\n"],
                "table_patterns": ["(\\d+\\.\\d+)", "(\\d+,\\d+)"]
            }
            self.boundary_priority = ["table", "paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "table_vs_paragraph": "table",
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "table": {"enabled": True, "patterns": self.params["table_patterns"]},  # 琛ㄦ牸妫€娴?
                "paragraph": {"enabled": True},  # 娈佃惤杈圭晫
                "sentence": {"enabled": True},  # 鍙ュ瓙杈圭晫
                "punctuation": {"enabled": True},  # 鏍囩偣绗﹀彿
                "token_count": {"enabled": True}  # 浠ょ墝璁℃暟
            }
        
        elif self.template_name == "瀹㈡湇":
            self.params = {
                "base_chunk_size": 200,
                "min_chunk_size": 50,
                "max_chunk_size": 300,
                "overlap": 30,
                "dialog_turn_patterns": ["鐢ㄦ埛:", "瀹㈡湇:", "椤惧:", "鍛樺伐:", "鎻愰棶:", "鍥炵瓟:"]
            }
            self.boundary_priority = ["dialog_turn", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "dialog_turn_vs_sentence": "dialog_turn"
            }
            self.signal_processors = {
                "dialog_turn": {"enabled": True, "patterns": self.params["dialog_turn_patterns"]},  # 瀵硅瘽杞妫€娴?
                "sentence": {"enabled": True},  # 鍙ュ瓙杈圭晫
                "punctuation": {"enabled": True},  # 鏍囩偣绗﹀彿
                "token_count": {"enabled": True}  # 浠ょ墝璁℃暟
            }
        
        else:  # 榛樿閫氱敤妯℃澘
            self.params = {
                "base_chunk_size": 350,
                "min_chunk_size": 150,
                "max_chunk_size": 550,
                "overlap": 50,
                "sentence_separators": ["銆?, "锛?, "锛?, ".", "!", "?"]
            }
            self.boundary_priority = ["paragraph", "sentence", "punctuation", "token_count"]
            self.conflict_resolution = {
                "paragraph_vs_sentence": "paragraph"
            }
            self.signal_processors = {
                "paragraph": {"enabled": True},  # 娈佃惤杈圭晫
                "sentence": {"enabled": True},  # 鍙ュ瓙杈圭晫
                "punctuation": {"enabled": True},  # 鏍囩偣绗﹀彿
                "token_count": {"enabled": True}  # 浠ょ墝璁℃暟
            }
    
    def override_params(self, params: Dict[str, Any]) -> None:
        """瑕嗙洊妯℃澘鍙傛暟"""
        self.params.update(params)
        # 濡傛灉鏈夋柊鐨勭壒娈婂弬鏁帮紝鏇存柊淇″彿澶勭悊鍣ㄩ厤缃?
        if "heading_patterns" in params and self.template_name == "璁烘枃":
            self.signal_processors["heading"]["patterns"] = params["heading_patterns"]
        if "dialog_turn_patterns" in params and self.template_name == "瀹㈡湇":
            self.signal_processors["dialog_turn"]["patterns"] = params["dialog_turn_patterns"]


class BoundarySignalDetector:
    """杈圭晫淇″彿妫€娴嬪櫒"""
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 绛栫暐妯℃澘
        
    def detect_contract_sections(self, text: str) -> List[Tuple[int, int, float]]:
        """妫€娴嬪悎鍚屾潯娆捐竟鐣?""
        if not self.template.signal_processors.get("contract_section", {}).get("enabled", False):
            return []
        
        pattern = self.template.signal_processors["contract_section"].get("regex", r'绗琜涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+鏉?)
        matches = re.finditer(pattern, text)
        boundaries = []
        
        for match in matches:
            start = match.start()
            # 閫氬父鏉℃寮€濮嬩簬鎹㈣鍚庯紝灏濊瘯鎵惧埌鏇村噯纭殑杈圭晫
            prev_newline = text.rfind('\n', 0, start)
            if prev_newline != -1:
                start = prev_newline + 1
            boundaries.append((start, start, 0.95))  # 楂樼疆淇″害
        
        return boundaries
    
    def detect_headings(self, text: str) -> List[Tuple[int, int, float]]:
        """妫€娴嬫爣棰樿竟鐣?""
        if not self.template.signal_processors.get("heading", {}).get("enabled", False):
            return []
        
        patterns = self.template.signal_processors["heading"].get("patterns", ["#+"])
        boundaries = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = match.start()
                # 鏍囬閫氬父鍗曠嫭鎴愯
                prev_newline = text.rfind('\n', 0, start)
                next_newline = text.find('\n', match.end())
                if prev_newline != -1:
                    start = prev_newline + 1
                boundaries.append((start, start, 0.90))  # 楂樼疆淇″害
        
        return boundaries
    
    def detect_paragraphs(self, text: str) -> List[Tuple[int, int, float]]:
        """妫€娴嬫钀借竟鐣?""
        if not self.template.signal_processors.get("paragraph", {}).get("enabled", False):
            return []
        
        separators = self.template.params.get("paragraph_separators", ["\n\n"])
        boundaries = [(0, 0, 0.0)]  # 鏂囨。寮€濮?
        
        for sep in separators:
            pos = 0
            while True:
                pos = text.find(sep, pos)
                if pos == -1:
                    break
                # 娈佃惤缁撴潫浜庡垎闅旂鍓嶏紝鏂版钀藉紑濮嬩簬鍒嗛殧绗﹀悗
                boundaries.append((pos, pos, 0.85))  # 楂樼疆淇″害
                pos += len(sep)
        
        # 娣诲姞鏂囨。缁撴潫杈圭晫
        boundaries.append((len(text), len(text), 0.0))
        
        # 鍘婚噸骞舵帓搴?
        boundaries = list(set(boundaries))
        boundaries.sort()
        
        return boundaries
    
    def detect_sentences(self, text: str) -> List[Tuple[int, int, float]]:
        """妫€娴嬪彞瀛愯竟鐣?""
        if not self.template.signal_processors.get("sentence", {}).get("enabled", False):
            return []
        
        separators = self.template.params.get("sentence_separators", ["銆?, "锛?, "锛?, ".", "!", "?"])
        # 鏋勫缓鍙ュ瓙鍒嗗壊姝ｅ垯琛ㄨ揪寮?
        pattern = '([' + ''.join(re.escape(sep) for sep in separators) + '])'
        
        # 浣跨敤姝ｅ垯琛ㄨ揪寮忓垎鍓叉枃鏈?
        matches = re.finditer(pattern, text)
        boundaries = []
        
        for match in matches:
            end = match.end()
            boundaries.append((end, end, 0.80))  # 涓瓑缃俊搴?
        
        return boundaries
    
    def detect_dialog_turns(self, text: str) -> List[Tuple[int, int, float]]:
        """妫€娴嬪璇濊疆娆¤竟鐣?""
        if not self.template.signal_processors.get("dialog_turn", {}).get("enabled", False):
            return []
        
        patterns = self.template.signal_processors["dialog_turn"].get("patterns", ["鐢ㄦ埛:", "瀹㈡湇:"])
        boundaries = []
        
        for pattern in patterns:
            matches = re.finditer(re.escape(pattern), text)
            for match in matches:
                start = match.start()
                # 瀵硅瘽杞寮€濮嬩簬妯″紡鍑虹幇澶?
                prev_newline = text.rfind('\n', 0, start)
                if prev_newline != -1:
                    start = prev_newline + 1
                boundaries.append((start, start, 0.95))  # 楂樼疆淇″害
        
        return boundaries
    
    def detect_token_based_boundaries(self, text: str) -> List[Tuple[int, int, float]]:
        """鍩轰簬浠ょ墝璁℃暟鐨勮竟鐣屾娴嬶紙婊戝姩绐楀彛锛?""
        if not self.template.signal_processors.get("token_count", {}).get("enabled", False):
            return []
        
        base_chunk_size = self.template.params.get("base_chunk_size", 350)
        min_chunk_size = self.template.params.get("min_chunk_size", 150)
        overlap = self.template.params.get("overlap", 50)
        
        boundaries = []
        pos = 0
        chunk_index = 0
        
        # 绠€鍗曞湴鎸夌収瀛楃鏁板垎鍓诧紙瀹為檯搴旂敤涓簲浣跨敤tokenizer锛?
        while pos < len(text):
            chunk_end = min(pos + base_chunk_size, len(text))
            boundaries.append((chunk_end, chunk_end, 0.70))  # 涓瓑缃俊搴?
            pos = chunk_end - overlap
            chunk_index += 1
        
        return boundaries
    
    def detect_all_signals(self, text: str) -> Dict[str, List[Tuple[int, int, float]]]:
        """骞惰妫€娴嬫墍鏈夎竟鐣屼俊鍙?""
        signals = {}
        
        # 骞惰鎵ц鍚勪俊鍙锋娴嬪櫒
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 鏍规嵁妯℃澘鍚嶇О纭畾闇€瑕佹墽琛岀殑妫€娴嬪櫒
            futures = {}
            
            if self.template.template_name == "鍚堝悓" and self.template.signal_processors.get("contract_section", {}).get("enabled", False):
                futures["contract_section"] = executor.submit(self.detect_contract_sections, text)
            
            if self.template.template_name == "璁烘枃" and self.template.signal_processors.get("heading", {}).get("enabled", False):
                futures["heading"] = executor.submit(self.detect_headings, text)
            
            if self.template.signal_processors.get("paragraph", {}).get("enabled", False):
                futures["paragraph"] = executor.submit(self.detect_paragraphs, text)
            
            if self.template.signal_processors.get("sentence", {}).get("enabled", False):
                futures["sentence"] = executor.submit(self.detect_sentences, text)
            
            if self.template.template_name == "瀹㈡湇" and self.template.signal_processors.get("dialog_turn", {}).get("enabled", False):
                futures["dialog_turn"] = executor.submit(self.detect_dialog_turns, text)
            
            if self.template.signal_processors.get("token_count", {}).get("enabled", False):
                futures["token_count"] = executor.submit(self.detect_token_based_boundaries, text)
            
            # 鏀堕泦缁撴灉
            for signal_type, future in futures.items():
                try:
                    signals[signal_type] = future.result()
                except Exception as e:
                    print(f"妫€娴媨signal_type}淇″彿鏃跺嚭閿? {e}")
                    signals[signal_type] = []
        
        return signals


class SignalFusionEngine:
    """
    淇″彿铻嶅悎寮曟搸
    璐熻矗铻嶅悎澶氱杈圭晫淇″彿锛屽簲鐢ㄤ紭鍏堢骇瑙勫垯鍜屽啿绐佽В鍐虫満鍒?
    """
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 绛栫暐妯℃澘
    
    def fuse_signals(self, signals: Dict[str, List[Tuple[int, int, float]]]) -> List[Tuple[int, int, Dict[str, float]]]:
        """铻嶅悎澶氱杈圭晫淇″彿"""
        # 1. 鏀堕泦鎵€鏈変俊鍙风殑浣嶇疆鐐?
        all_boundaries = defaultdict(list)
        
        for signal_type, signal_boundaries in signals.items():
            for start, end, confidence in signal_boundaries:
                # 浣嶇疆鐐圭粺涓€浣跨敤start锛堝洜涓哄ぇ澶氭暟杈圭晫閮芥槸鐐癸級
                pos = start
                all_boundaries[pos].append((signal_type, confidence))
        
        # 2. 瀵逛綅缃偣杩涜鎺掑簭
        sorted_positions = sorted(all_boundaries.keys())
        
        # 3. 搴旂敤浼樺厛绾ц鍒欒瀺鍚堜俊鍙?
        fused_boundaries = []
        last_pos = 0
        
        for pos in sorted_positions:
            # 璺宠繃鏂囨。寮€濮嬩綅缃?
            if pos == 0:
                continue
            
            # 璺宠繃涓庝笂涓€涓綅缃お杩戠殑鐐癸紙閬垮厤杩囦簬瀵嗛泦鐨勮竟鐣岋級
            if pos - last_pos < self.template.params.get("min_chunk_size", 150) // 2:
                continue
            
            # 鑾峰彇璇ヤ綅缃殑鎵€鏈変俊鍙?
            position_signals = all_boundaries[pos]
            
            # 鏍规嵁淇″彿浼樺厛绾у拰鍐茬獊瑙ｅ喅瑙勫垯锛岄€夋嫨鏈€鍙俊鐨勮竟鐣?
            selected_signals = self._resolve_conflicts(position_signals)
            
            # 鍙湁褰撳瓨鍦ㄦ湁鏁堜俊鍙锋椂鎵嶆坊鍔犺竟鐣?
            if selected_signals:
                # 璁＄畻缁煎悎缃俊搴?
                confidence_sum = sum(conf for _, conf in selected_signals)
                avg_confidence = confidence_sum / len(selected_signals) if selected_signals else 0
                
                # 鍒涘缓杈圭晫瀛楀吀锛岃褰曞悇淇″彿鐨勮础鐚?
                boundary_signals = {signal_type: conf for signal_type, conf in selected_signals}
                
                # 娣诲姞鏂囨。缁撴潫杈圭晫
                if pos == sorted_positions[-1] and pos < len(self.template.params.get("original_text", "")):
                    pos = len(self.template.params.get("original_text", ""))
                
                fused_boundaries.append((last_pos, pos, boundary_signals))
                last_pos = pos
        
        # 纭繚鏂囨。缁撴潫
        if last_pos < len(self.template.params.get("original_text", "")):
            fused_boundaries.append((last_pos, len(self.template.params.get("original_text", "")), {"end_of_doc": 1.0}))
        
        return fused_boundaries
    
    def _resolve_conflicts(self, signals: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """瑙ｅ喅淇″彿鍐茬獊"""
        if not signals:
            return []
        
        # 鎸夌収妯℃澘涓殑杈圭晫浼樺厛绾ф帓搴忎俊鍙?
        signals.sort(key=lambda x: self.template.boundary_priority.index(x[0]) if x[0] in self.template.boundary_priority else len(self.template.boundary_priority))
        
        # 濡傛灉鍙湁涓€绉嶄俊鍙风被鍨嬫垨娌℃湁鍐茬獊锛岀洿鎺ヨ繑鍥?
        signal_types = {signal_type for signal_type, _ in signals}
        if len(signal_types) <= 1:
            return signals
        
        # 搴旂敤鍐茬獊瑙ｅ喅瑙勫垯
        resolved_signals = []
        high_priority_signal = signals[0]  # 鏈€楂樹紭鍏堢骇淇″彿
        resolved_signals.append(high_priority_signal)
        
        # 瀵逛簬鍏朵粬淇″彿锛屽鏋滃畠浠笉涓庢渶楂樹紭鍏堢骇淇″彿鍐茬獊锛屽垯淇濈暀
        for signal_type, confidence in signals[1:]:
            conflict_key = f"{high_priority_signal[0]}_vs_{signal_type}"
            if conflict_key in self.template.conflict_resolution:
                # 鏍规嵁瑙勫垯鍐冲畾鏄惁淇濈暀
                if self.template.conflict_resolution[conflict_key] == signal_type:
                    resolved_signals.append((signal_type, confidence))
            else:
                # 榛樿淇濈暀楂樼疆淇″害淇″彿
                if confidence > 0.7:
                    resolved_signals.append((signal_type, confidence))
        
        return resolved_signals


class ChunkGenerator:
    """
    鍒嗗潡鐢熸垚鍣?
    鏍规嵁铻嶅悎鍚庣殑杈圭晫淇℃伅鐢熸垚瀹為檯鐨勫垎鍧楋紝骞舵坊鍔犲厓鏁版嵁
    """
    def __init__(self, template: StrategyTemplate):
        self.template = template  # 绛栫暐妯℃澘
    
    def generate_chunks(self, text: str, boundaries: List[Tuple[int, int, Dict[str, float]]]) -> List[ChunkMetadata]:
        """鏍规嵁杈圭晫鐢熸垚鍒嗗潡"""
        chunks = []
        chunk_index = 1
        
        for start, end, boundary_signals in boundaries:
            # 鎻愬彇鍧楀唴瀹?
            content = text[start:end].strip()
            
            # 璺宠繃绌哄潡
            if not content:
                continue
            
            # 鍒涘缓鍧楀厓鏁版嵁
            chunk_id = f"chunk_{chunk_index:04d}"
            chunk = ChunkMetadata(chunk_id, start, end)
            chunk.content = content
            chunk.boundary_signals = boundary_signals
            
            # 纭畾鍧楃被鍨?
            chunk.chunk_type = self._determine_chunk_type(boundary_signals)
            
            # 璁＄畻缃俊搴?
            chunk.confidence = self._calculate_confidence(boundary_signals)
            
            # 娣诲姞缁撴瀯璺緞锛堢畝鍖栫増锛?
            chunk.structure_path = self._generate_structure_path(start, end, text)
            
            # 娣诲姞涓婚鏍囩锛堢畝鍖栫増锛?
            chunk.topic_tags = self._extract_topic_tags(content)
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def _determine_chunk_type(self, boundary_signals: Dict[str, float]) -> str:
        """纭畾鍧楃被鍨?""
        if not boundary_signals:
            return "general"
        
        # 鏍规嵁涓昏淇″彿绫诲瀷纭畾鍧楃被鍨?
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
        """璁＄畻鍧楄竟鐣岀殑缃俊搴?""
        if not boundary_signals:
            return 0.5  # 榛樿涓瓑缃俊搴?
        
        # 璁＄畻骞冲潎缃俊搴?
        avg_confidence = sum(boundary_signals.values()) / len(boundary_signals)
        
        return avg_confidence
    
    def _generate_structure_path(self, start: int, end: int, text: str) -> List[str]:
        """鐢熸垚缁撴瀯璺緞锛堢畝鍖栫増锛?""
        # 鍦ㄥ疄闄呭簲鐢ㄤ腑锛岃繖閲屽簲璇ユ牴鎹枃妗ｇ殑瀹為檯缁撴瀯鐢熸垚璺緞
        path = []
        
        # 绠€鍗曠ず渚嬶細鏌ユ壘鍧楁墍鍦ㄧ殑娈佃惤
        prev_newline = text.rfind('\n', 0, start)
        if prev_newline != -1:
            # 鏌ユ壘娈佃惤寮€澶寸殑娼滃湪鏍囬
            para_start = prev_newline + 1
            para_text = text[para_start:para_start + 50]  # 鏌ョ湅鍓?0涓瓧绗?
            
            # 妫€鏌ユ槸鍚︽槸鏍囬鏍煎紡
            if re.match(r'^#|绗?*绔爘\\d+\\.', para_text):
                path.append(f"鏍囬:{para_text[:20]}")
        
        path.append(f"娈佃惤:{start}-{end}")
        
        return path
    
    def _extract_topic_tags(self, content: str) -> List[str]:
        """鎻愬彇涓婚鏍囩锛堢畝鍖栫増锛?""
        # 鍦ㄥ疄闄呭簲鐢ㄤ腑锛岃繖閲屽簲璇ヤ娇鐢∟LP妯″瀷鎻愬彇鍏抽敭璇?
        tags = []
        
        # 绠€鍗曠ず渚嬶細鏌ユ壘甯歌鍏抽敭璇?
        common_topics = {
            "鍚堝悓": ["鏉℃", "璐ｄ换", "涔夊姟", "鏉冨埄", "杩濈害", "璧斿伩"],
            "璁烘枃": ["鐮旂┒", "瀹為獙", "缁撹", "鏂规硶", "鍒嗘瀽", "缁撴灉"],
            "璐㈡姤": ["鏀跺叆", "鍒╂鼎", "鎴愭湰", "澧為暱", "鎶曡祫", "鏀嚭"],
            "瀹㈡湇": ["闂", "瑙ｅ喅", "寤鸿", "鏈嶅姟", "婊℃剰", "鍙嶉"]
        }
        
        template_topics = common_topics.get(self.template.template_name, [])
        for topic in template_topics:
            if topic in content:
                tags.append(topic)
        
        # 闄愬埗鏍囩鏁伴噺
        return tags[:5] if len(tags) > 5 else tags


class StrategyLogger:
    """
    绛栫暐鎵ц鏃ュ織璁板綍鍣?
    璁板綍鍒嗗潡杩囩▼涓殑鍚勭淇℃伅锛屽寘鎷枃妗ｄ俊鎭€佺瓥鐣ュ弬鏁般€佷俊鍙蜂娇鐢ㄦ儏鍐电瓑
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
        """鏇存柊鏂囨。淇℃伅"""
        self.log["doc_info"] = doc_info
    
    def update_strategy_params(self, params: Dict[str, Any]) -> None:
        """鏇存柊绛栫暐鍙傛暟"""
        self.log["strategy_params"] = params
    
    def record_signal_usage(self, signal_type: str) -> None:
        """璁板綍淇″彿浣跨敤鎯呭喌"""
        self.log["signal_statistics"][signal_type] += 1
    
    def update_coverage_summary(self, chunks: List[ChunkMetadata]) -> None:
        """鏇存柊瑕嗙洊鐜囨憳瑕?""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return
        
        # 缁熻鍧楃被鍨嬪垎甯?
        type_counter = Counter()
        total_confidence = 0.0
        signal_contributions = defaultdict(int)
        
        for chunk in chunks:
            type_counter[chunk.chunk_type] += 1
            total_confidence += chunk.confidence
            
            # 缁熻鍚勪俊鍙风殑璐＄尞
            for signal_type in chunk.boundary_signals:
                signal_contributions[signal_type] += 1
        
        # 璁＄畻骞冲潎缃俊搴?
        avg_confidence = total_confidence / total_chunks if total_chunks > 0 else 0.0
        
        # 璁＄畻鍚勪俊鍙风殑瑕嗙洊鐜?
        signal_coverage = {}
        for signal_type, count in signal_contributions.items():
            signal_coverage[signal_type] = count / total_chunks if total_chunks > 0 else 0.0
        
        # 鏇存柊瑕嗙洊鐜囨憳瑕?
        self.log["coverage_summary"] = {
            "total_chunks": total_chunks,
            "chunk_type_distribution": dict(type_counter),
            "avg_confidence": avg_confidence,
            "signal_coverage": dict(signal_coverage)
        }
    
    def add_warning(self, message: str) -> None:
        """娣诲姞璀﹀憡淇℃伅"""
        self.log["warnings"].append({
            "time": datetime.now().isoformat(),
            "message": message
        })
    
    def add_error(self, message: str) -> None:
        """娣诲姞閿欒淇℃伅"""
        self.log["errors"].append({
            "time": datetime.now().isoformat(),
            "message": message
        })
    
    def finalize(self) -> Dict[str, Any]:
        """鏈€缁堝寲鏃ュ織"""
        self.log["end_time"] = datetime.now().isoformat()
        # 璁＄畻鎸佺画鏃堕棿锛堟绉掞級
        start_dt = datetime.fromisoformat(self.log["start_time"])
        end_dt = datetime.fromisoformat(self.log["end_time"])
        self.log["duration_ms"] = (end_dt - start_dt).total_seconds() * 1000
        
        # 杞崲defaultdict涓烘櫘閫歞ict
        self.log["signal_statistics"] = dict(self.log["signal_statistics"])
        
        return self.log


class HybridChunkingOrchestrator:
    """
    娣峰悎鍒嗗潡绛栫暐缂栨帓鍣?
    鍗忚皟鏁翠釜鍒嗗潡娴佺▼锛屼粠鏂囨。鍏冧俊鎭彁鍙栥€佺瓥鐣ユā鏉块€夋嫨銆佷俊鍙锋娴嬩笌铻嶅悎锛屽埌鏈€缁堢敓鎴愬垎鍧楀苟杈撳嚭缁撴灉
    """
    def __init__(self):
        self.logger = None  # 绛栫暐鎵ц鏃ュ織璁板綍鍣?
        
    def orchestrate(self, text: str, doc_type: str = "閫氱敤", 
                   template_name: Optional[str] = None, 
                   params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        鎵ц娣峰悎鍒嗗潡绛栫暐缂栨帓
        
        鍙傛暟:
            text: 杈撳叆鏂囨湰
            doc_type: 鏂囨。绫诲瀷锛堝悎鍚?璁烘枃/璐㈡姤/瀹㈡湇/閫氱敤锛?
            template_name: 鍙€夌殑妯℃澘鍚嶇О锛堝鏋滄彁渚涳紝灏嗚鐩杁oc_type锛?
            params_override: 鍙€夌殑鍙傛暟瑕嗗啓
            
        杩斿洖:
            鍖呭惈鍒嗗潡缁撴灉鍜岀瓥鐣ユ墽琛屾棩蹇楃殑瀛楀吀
        """
        # 鍒濆鍖栨棩蹇楄褰曞櫒
        self.logger = StrategyLogger()
        
        try:
            # 寮€濮嬭鏃?
            start_time = time.time()
            
            # 1. 璇诲彇鏂囨。鍏冧俊鎭?
            doc_info = self._extract_doc_info(text, doc_type)
            self.logger.update_doc_info(doc_info)
            
            # 2. 閫夋嫨绛栫暐妯℃澘
            template = self._select_strategy_template(doc_type, template_name)
            
            # 3. 搴旂敤鍙傛暟瑕嗗啓
            if params_override:
                template.override_params(params_override)
                self.logger.update_strategy_params(template.params)
            
            # 淇濆瓨鍘熸枃鍒版ā鏉垮弬鏁帮紝渚涘悗缁娇鐢?
            template.params["original_text"] = text
            
            # 4. 骞惰杩愯缁撴瀯/璇箟/瀹炰綋妫€娴?
            detector = BoundarySignalDetector(template)
            signals = detector.detect_all_signals(text)
            
            # 璁板綍淇″彿缁熻
            for signal_type, signal_boundaries in signals.items():
                self.logger.record_signal_usage(signal_type)
            
            # 5. 浣跨敤浼樺厛绾ц鍒欒瀺鍚堣竟鐣?
            fusion_engine = SignalFusionEngine(template)
            fused_boundaries = fusion_engine.fuse_signals(signals)
            
            # 6. 鐢熸垚鍧楀苟闄勫姞鍏冩暟鎹?
            chunk_generator = ChunkGenerator(template)
            chunks = chunk_generator.generate_chunks(text, fused_boundaries)
            
            # 7. 鏇存柊瑕嗙洊鐜囨憳瑕?
            self.logger.update_coverage_summary(chunks)
            
            # 杞崲鍧椾负瀛楀吀鏍煎紡
            chunks_dict = [chunk.to_dict() for chunk in chunks]
            
            # 缁撴潫璁℃椂
            duration_ms = (time.time() - start_time) * 1000
            
            # 8. 杈撳嚭鍧椾笌绛栫暐鎵ц鏃ュ織
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
                self.logger.add_error(f"鍒嗗潡杩囩▼涓彂鐢熼敊璇? {str(e)}")
                strategy_log = self.logger.finalize()
            else:
                strategy_log = {"error": str(e)}
            
            return {
                "chunks": [],
                "strategy_log": strategy_log,
                "summary": {"error": str(e)}
            }
    
    def _extract_doc_info(self, text: str, doc_type: str) -> Dict[str, Any]:
        """鎻愬彇鏂囨。鍏冧俊鎭?""
        # 绠€鍗曠殑鏂囨。淇℃伅鎻愬彇
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
        """閫夋嫨绛栫暐妯℃澘"""
        # 濡傛灉鎻愪緵浜唗emplate_name锛屼娇鐢ㄥ畠锛涘惁鍒欎娇鐢╠oc_type
        template_to_use = template_name if template_name else doc_type
        
        # 楠岃瘉妯℃澘鍚嶇О
        valid_templates = ["鍚堝悓", "璁烘枃", "璐㈡姤", "瀹㈡湇", "閫氱敤"]
        if template_to_use not in valid_templates:
            self.logger.add_warning(f"鏈煡鐨勬ā鏉垮悕绉? {template_to_use}锛屼娇鐢ㄩ粯璁ゆā鏉?)
            template_to_use = "閫氱敤"
        
        return StrategyTemplate(template_to_use)
    
    def save_results(self, result: Dict[str, Any], chunks_file: str = "chunks_result.json", 
                    log_file: str = "strategy_log.json") -> None:
        """淇濆瓨鍒嗗潡缁撴灉鍜岀瓥鐣ユ棩蹇楀埌鏂囦欢"""
        try:
            # 淇濆瓨鍒嗗潡缁撴灉
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(result["chunks"], f, ensure_ascii=False, indent=2)
              
            # 淇濆瓨绛栫暐鏃ュ織
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(result["strategy_log"], f, ensure_ascii=False, indent=2)
              
            print(f"鍒嗗潡缁撴灉宸蹭繚瀛樺埌: {chunks_file}")
            print(f"绛栫暐鏃ュ織宸蹭繚瀛樺埌: {log_file}")
              
        except Exception as e:
            print(f"淇濆瓨缁撴灉鏃跺嚭閿? {e}")


# 绀轰緥鐢ㄦ硶
if __name__ == "__main__":
    # 绀轰緥鏂囨湰 - 涓嶅悓绫诲瀷鐨勬枃妗?
    sample_texts = {
        "鍚堝悓": """
        鍚堝悓缂栧彿锛?023-001
        
        鐢叉柟锛氱鎶€鏈夐檺鍏徃
        涔欐柟锛氬垱鏂板彂灞曚腑蹇?
        
        绗竴鏉?鍚堜綔鍐呭
        鐢蹭箼鍙屾柟灏变汉宸ユ櫤鑳芥妧鏈爺鍙戣揪鎴愬悎浣滐紝鐢叉柟鎻愪緵鎶€鏈敮鎸侊紝涔欐柟鎻愪緵鐮斿彂鍦哄湴鍜岃祫閲戙€?
        
        绗簩鏉?鏉冨埄涓庝箟鍔?
        鐢叉柟鏈夋潈鑾峰彇椤圭洰鐮斿彂鎴愭灉鐨勪娇鐢ㄦ潈锛屼箼鏂规湁鏉冭幏寰楁妧鏈浆璁╄垂銆傚弻鏂瑰簲淇濆畧椤圭洰鐩稿叧鐨勫晢涓氱瀵嗐€?
        
        绗笁鏉?杩濈害璐ｄ换
        鑻ヤ竴鏂硅繚鍙嶆湰鍚堝悓绾﹀畾锛屽簲鍚戝彟涓€鏂规敮浠樿繚绾﹂噾锛屽苟璧斿伩鍥犳閫犳垚鐨勬崯澶便€?
        """,
        
        "璁烘枃": """
        # 浜哄伐鏅鸿兘鍦ㄥ尰鐤楅鍩熺殑搴旂敤鐮旂┒
        
        ## 鎽樿
        鏈爺绌舵帰璁ㄤ簡浜哄伐鏅鸿兘鎶€鏈湪鍖荤枟璇婃柇銆佹不鐤楁柟妗堝埗瀹氬拰鎮ｈ€呯鐞嗘柟闈㈢殑搴旂敤鐜扮姸鍜屽墠鏅€?
        
        ## 寮曡█
        闅忕潃娣卞害瀛︿範鎶€鏈殑蹇€熷彂灞曪紝浜哄伐鏅鸿兘鍦ㄥ尰鐤楅鍩熺殑搴旂敤鏃ョ泭骞挎硾銆傛湰鏂囩患杩颁簡杩戝勾鏉ョ殑鐩稿叧鐮旂┒鎴愭灉銆?
        
        ## 鐮旂┒鏂规硶
        鏈爺绌堕噰鐢ㄦ枃鐚患杩板拰妗堜緥鍒嗘瀽鐩哥粨鍚堢殑鏂规硶锛屾敹闆嗕簡2018-2023骞撮棿鐨勭浉鍏崇爺绌舵枃鐚€?
        
        ## 瀹為獙缁撴灉
        瀹為獙缁撴灉琛ㄦ槑锛屽熀浜庢繁搴﹀涔犵殑鍖荤枟褰卞儚璇婃柇绯荤粺鍦ㄦ煇浜涢鍩熷凡杈惧埌鎴栬秴杩囦汉绫讳笓瀹舵按骞炽€?
        """,
        
        "璐㈡姤": """
        2023骞寸涓夊搴﹁储鍔℃姤鍛?
        
        涓€銆佺粡钀ヤ笟缁?
        鏈搴﹀疄鐜拌惀涓氭敹鍏?0.5浜垮厓锛屽悓姣斿闀?5.3%锛涘噣鍒╂鼎2.1浜垮厓锛屽悓姣斿闀?2.7%銆?
        
        浜屻€佷富瑕佽储鍔℃寚鏍?
        姣涘埄鐜囷細42.5%锛堢幆姣?1.2%锛?
        鍑€鍒╃巼锛?0.0%锛堢幆姣?2.5%锛?
        缁忚惀鐜伴噾娴侊細3.2浜垮厓
        
        涓夈€佷笟鍔″垎鏋?
        鏅鸿兘纭欢涓氬姟鏀跺叆鍗犳瘮45%锛屽悓姣斿闀?0%锛涜蒋浠舵湇鍔℃敹鍏ュ崰姣?5%锛屽悓姣斿闀?0%锛涘叾浠栦笟鍔℃敹鍏ュ崰姣?0%銆?
        """,
        
        "瀹㈡湇": """
        鐢ㄦ埛锛氫綘濂斤紝鎴戞兂鍜ㄨ涓€涓嬩綘浠殑浜у搧淇濅慨鏀跨瓥銆?
        瀹㈡湇锛氭偍濂斤紒寰堥珮鍏翠负鎮ㄦ湇鍔°€傛垜浠殑浜у搧鎻愪緵涓€骞村厤璐逛繚淇湇鍔★紝淇濅慨鏈熶粠璐拱涔嬫棩璧疯绠椼€?
        鐢ㄦ埛锛氬鏋滀骇鍝佹槸浜轰负鎹熷潖鐨勶紝杩樿兘淇濅慨鍚楋紵
        瀹㈡湇锛氬鏋滅粡妫€娴嬬‘璁ゆ槸浜轰负鎹熷潖锛屽皢涓嶅湪鍏嶈垂淇濅慨鑼冨洿鍐咃紝浣嗘垜浠彲浠ユ彁渚涙湁鍋跨淮淇湇鍔°€?
        鐢ㄦ埛锛氶偅缁翠慨璐圭敤澶ф鏄灏戝憿锛?
        瀹㈡湇锛氱淮淇垂鐢ㄦ牴鎹崯鍧忕▼搴﹀拰鎵€闇€鏇存崲鐨勯浂閮ㄤ欢涓嶅悓鑰屾湁鎵€宸紓锛屽缓璁偍灏嗕骇鍝侀€佸埌鎴戜滑鐨勬湇鍔′腑蹇冭繘琛屾娴嬭瘎浼般€?
        """
    }
    
    print("===== 娣峰悎鍒嗗潡绛栫暐缂栨帓鍣ㄧず渚?=====")
    
    # 鍒涘缓缂栨帓鍣ㄥ疄渚?
    orchestrator = HybridChunkingOrchestrator()
    
    # 閫夋嫨涓€涓ず渚嬫枃妗ｇ被鍨?
    doc_type = "鍚堝悓"  # 鍙互鏇存敼涓?"璁烘枃"銆?璐㈡姤"銆?瀹㈡湇"
    sample_text = sample_texts[doc_type]
    
    print(f"\n澶勭悊 {doc_type} 绫诲瀷鏂囨。...")
    
    # 鎵ц鍒嗗潡绛栫暐缂栨帓
    result = orchestrator.orchestrate(sample_text, doc_type=doc_type)
    
    # 鎵撳嵃鎽樿淇℃伅
    print("\n鍒嗗潡缁撴灉鎽樿:")
    print(f"  鎬诲潡鏁? {result['summary']['total_chunks']}")
    print(f"  浣跨敤妯℃澘: {result['summary']['template_used']}")
    print(f"  澶勭悊鏃堕棿: {result['summary']['duration_ms']:.2f} ms")
    
    # 鎵撳嵃璇︾粏鍒嗗潡淇℃伅
    print("\n璇︾粏鍒嗗潡淇℃伅:")
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"\n鍧?{i}:")
        print(f"  ID: {chunk['chunk_id']}")
        print(f"  绫诲瀷: {chunk['chunk_type']}")
        print(f"  浣嶇疆: {chunk['start_pos']}-{chunk['end_pos']}")
        print(f"  缃俊搴? {chunk['confidence']:.2f}")
        print(f"  杈圭晫淇″彿: {chunk['boundary_signals']}")
        print(f"  缁撴瀯璺緞: {chunk['structure_path']}")
        print(f"  涓婚鏍囩: {chunk['topic_tags']}")
        print(f"  鍐呭: {chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}")
    
    # 鎵撳嵃绛栫暐鏃ュ織鎽樿
    print("\n绛栫暐鎵ц鏃ュ織鎽樿:")
    log = result['strategy_log']
    print(f"  淇″彿缁熻: {log['signal_statistics']}")
    print(f"  鍧楃被鍨嬪垎甯? {log['coverage_summary']['chunk_type_distribution']}")
    print(f"  骞冲潎缃俊搴? {log['coverage_summary']['avg_confidence']:.2f}")
    print(f"  淇″彿瑕嗙洊鐜? {log['coverage_summary']['signal_coverage']}")
    
    # 淇濆瓨缁撴灉鍒版枃浠?
    print("\n淇濆瓨缁撴灉鍒版枃浠?..")
    orchestrator.save_results(result, f"chunks_{doc_type}.json", f"log_{doc_type}.json")
    
    # 娣诲姞璋冭瘯淇℃伅
    print("\n妫€鏌ユ枃浠舵槸鍚︿繚瀛樻垚鍔燂細")
    import os
    if os.path.exists(f"chunks_{doc_type}.json"):
        print(f"鉁?鍒嗗潡缁撴灉鏂囦欢 'chunks_{doc_type}.json' 淇濆瓨鎴愬姛")
        print(f"   鏂囦欢澶у皬: {os.path.getsize(f'chunks_{doc_type}.json')} 瀛楄妭")
    else:
        print(f"鉂?鍒嗗潡缁撴灉鏂囦欢 'chunks_{doc_type}.json' 涓嶅瓨鍦?)
    
    if os.path.exists(f"log_{doc_type}.json"):
        print(f"鉁?鏃ュ織鏂囦欢 'log_{doc_type}.json' 淇濆瓨鎴愬姛")
        print(f"   鏂囦欢澶у皬: {os.path.getsize(f'log_{doc_type}.json')} 瀛楄妭")
    else:
        print(f"鉂?鏃ュ織鏂囦欢 'log_{doc_type}.json' 涓嶅瓨鍦?)
        
    # 鎵撳嵃褰撳墠宸ヤ綔鐩綍
    print(f"\n褰撳墠宸ヤ綔鐩綍: {os.getcwd()}")
    
    print("\n===== 绀轰緥瀹屾垚 =====")
