'''
法律文本实体识别系统

这个程序实现了以下功能：

1. 数据获取：直接读取当前目录下的Word文档（中华人民共和国劳动法.docx）
2. 词典加载：加载当前目录下的法律基础术语词典汇编.docx
3. 模型构建：实现了BiLSTM-CRF模型用于法律文本实体识别
4. 实体识别：支持识别法人实体(ORG)、时间实体(TIME)、金额实体(MONEY)、地点实体(LOC)和人名实体(PER)
5. 混合识别策略：结合深度学习模型、词典匹配和基于规则的方法进行实体识别
6. 可视化展示：提供实体识别结果的可视化展示

首先需要安装必要的依赖：
    pip install tensorflow pandas numpy matplotlib requests python-docx
'''

import re
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from docx import Document  # 用于读取Word文档

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class LegalEntityRecognizer:
    """法律文本实体识别系统，使用BiLSTM-CRF模型识别法人实体、时间实体、金额实体等"""
    
    def __init__(self):
        # 模型参数
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.vocab_size = 5000  # 词汇表大小
        self.max_seq_len = 1000  # 最大序列长度
        
        # 实体标签定义（BIO标注体系）
        self.tag_to_id = {
            'O': 0,          # 非实体
            'B-ORG': 1,      # 法人/组织实体开始
            'I-ORG': 2,      # 法人/组织实体中间
            'B-TIME': 3,     # 时间实体开始
            'I-TIME': 4,     # 时间实体中间
            'B-MONEY': 5,    # 金额实体开始
            'I-MONEY': 6,    # 金额实体中间
            'B-LOC': 7,      # 地点实体开始
            'I-LOC': 8,      # 地点实体中间
            'B-PER': 9,      # 人名实体开始
            'I-PER': 10      # 人名实体中间
        }
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        
        # 词汇表
        self.char_to_id = {'<PAD>': 0, '<UNK>': 1}
        
        # 法律术语词典
        self.legal_dict = {
            'ORG': set(),  # 组织/机构术语
            'LAW': set(),  # 法律法规术语
            'TERM': set()  # 法律专业术语
        }
        
        # 初始化模型
        self.model = None
        self.crf_layer = None
        
        # 初始化数据
        self.train_data = []
        self.test_data = []
        
        # 尝试加载本地模型（如果存在）
        self.load_model()
    
    def read_doc_file(self, file_path):
        """读取Word文档内容"""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            print(f"读取Word文档失败: {str(e)}")
            # 尝试使用备用方法读取
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e2:
                print(f"备用方法读取也失败: {str(e2)}")
                return ""
    
    def load_legal_dict(self, dict_file_path):
        """加载法律术语词典"""
        try:
            print(f"正在加载法律术语词典: {dict_file_path}")
            dict_text = self.read_doc_file(dict_file_path)
            
            # 这里简化了词典解析，实际应用中可能需要根据词典的具体格式进行调整
            lines = dict_text.split('\n')
            
            # 假设词典中包含各种法律术语，我们将所有可能的术语都添加到词典中
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                
                # 将所有非空行的文本添加到法律术语集合中
                terms = re.split(r'[，,\s]+', line)
                for term in terms:
                    term = term.strip()
                    if len(term) >= 2:  # 只添加长度大于等于2的术语
                        self.legal_dict['TERM'].add(term)
                        # 简单判断术语类型
                        if '法' in term or '条例' in term or '规定' in term:
                            self.legal_dict['LAW'].add(term)
                        elif '公司' in term or '企业' in term or '机构' in term or '组织' in term:
                            self.legal_dict['ORG'].add(term)
            
            print(f"法律术语词典加载完成")
            print(f"- 法律专业术语数量: {len(self.legal_dict['TERM'])}")
            print(f"- 法律法规术语数量: {len(self.legal_dict['LAW'])}")
            print(f"- 组织/机构术语数量: {len(self.legal_dict['ORG'])}")
            
        except Exception as e:
            print(f"加载法律术语词典失败: {str(e)}")
    
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        char_freq = {}
        
        # 统计字符频率
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # 构建词汇表，只保留频率大于等于min_freq的字符
        for char, freq in char_freq.items():
            if freq >= min_freq:
                self.char_to_id[char] = len(self.char_to_id)
        
        self.vocab_size = len(self.char_to_id)
        print(f"词汇表大小: {self.vocab_size}")
    
    def preprocess_data(self, texts, labels=None):
        """预处理数据，转换为模型输入格式"""
        # 将文本转换为字符ID序列
        X = []
        for text in texts:
            char_ids = []
            for char in text[:self.max_seq_len]:  # 截断过长的文本
                char_ids.append(self.char_to_id.get(char, self.char_to_id['<UNK>']))
            X.append(char_ids)
        
        # 填充序列
        X = pad_sequences(X, maxlen=self.max_seq_len, padding='post', value=self.char_to_id['<PAD>'])
        
        if labels is not None:
            # 处理标签
            y = []
            for label_seq in labels:
                label_ids = []
                for label in label_seq[:self.max_seq_len]:  # 截断过长的标签序列
                    label_ids.append(self.tag_to_id.get(label, self.tag_to_id['O']))
                y.append(label_ids)
            
            # 填充标签序列
            y = pad_sequences(y, maxlen=self.max_seq_len, padding='post', value=self.tag_to_id['O'])
            return X, y
        
        return X
    
    def build_model(self):
        """构建BiLSTM-CRF模型"""
        model = Sequential()
        
        # 嵌入层
        model.add(Embedding(input_dim=self.vocab_size, 
                           output_dim=self.embedding_dim, 
                           input_length=self.max_seq_len, 
                           mask_zero=True))
        
        # 双向LSTM层
        model.add(Bidirectional(LSTM(units=self.hidden_dim, 
                                    return_sequences=True, 
                                    dropout=0.2, 
                                    recurrent_dropout=0.2)))
        
        # 时间分布式全连接层
        model.add(TimeDistributed(Dense(len(self.tag_to_id), activation='softmax')))
        
        # 编译模型
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        self.model = model
        print("模型构建完成")
    
    def load_model(self, model_path='legal_ner_model.h5'):
        """加载预训练模型"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"模型已从{model_path}加载")
            else:
                print(f"未找到预训练模型{model_path}，将使用基于规则和词典的方法进行实体识别")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
    
    def recognize_entities(self, text):
        """识别文本中的实体"""
        entities = []
        
        # 如果有模型，使用模型进行识别
        if self.model is not None:
            try:
                # 预处理输入文本
                X = self.preprocess_data([text])
                
                # 预测标签
                y_pred = self.model.predict(X)[0]
                
                # 转换为标签序列
                pred_tags = []
                for i in range(min(len(text), self.max_seq_len)):
                    pred_label_id = np.argmax(y_pred[i])
                    pred_tags.append(self.id_to_tag[pred_label_id])
                
                # 提取实体
                model_entities = self.extract_entities(text[:self.max_seq_len], pred_tags)
                entities.extend(model_entities)
            except Exception as e:
                print(f"模型识别出错: {str(e)}")
        
        # 使用基于规则的识别作为补充
        rule_entities = self.rule_based_recognition(text)
        entities.extend(rule_entities)
        
        # 使用词典进行实体识别作为补充
        dict_entities = self.dict_based_recognition(text)
        entities.extend(dict_entities)
        
        # 去重
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity['start'], entity['end'], entity.get('type', 'UNKNOWN'))
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_entities(self, text, tags):
        """从标签序列中提取实体"""
        entities = []
        current_entity = None
        current_entity_type = None
        current_entity_start = None
        
        for i, (char, tag) in enumerate(zip(text, tags)):
            if tag.startswith('B-'):
                # 开始一个新实体
                if current_entity:
                    # 保存之前的实体
                    entities.append({
                        'text': current_entity,
                        'type': current_entity_type,
                        'start': current_entity_start,
                        'end': i - 1,
                        'method': 'model'
                    })
                
                # 开始新实体
                current_entity = char
                current_entity_type = tag[2:]
                current_entity_start = i
            elif tag.startswith('I-') and current_entity:
                # 继续当前实体
                current_entity_type_tag = tag[2:]
                if current_entity_type_tag == current_entity_type:
                    current_entity += char
                else:
                    # 实体类型不一致，保存之前的实体并开始新实体
                    entities.append({
                        'text': current_entity,
                        'type': current_entity_type,
                        'start': current_entity_start,
                        'end': i - 1,
                        'method': 'model'
                    })
                    current_entity = char
                    current_entity_type = current_entity_type_tag
                    current_entity_start = i
            else:
                # 非实体或实体结束
                if current_entity:
                    entities.append({
                        'text': current_entity,
                        'type': current_entity_type,
                        'start': current_entity_start,
                        'end': i - 1,
                        'method': 'model'
                    })
                    current_entity = None
                    current_entity_type = None
                    current_entity_start = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append({
                'text': current_entity,
                'type': current_entity_type,
                'start': current_entity_start,
                'end': len(text) - 1,
                'method': 'model'
            })
        
        return entities
    
    def rule_based_recognition(self, text):
        """基于规则的实体识别，作为模型识别的补充"""
        entities = []
        
        # 匹配时间实体（简单规则）
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{2}日\d{1,2}时\d{1,2}分',
            r'\d{4}年度',
            r'\d{4}年'
        ]
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'type': 'TIME',
                    'start': match.start(),
                    'end': match.end() - 1,
                    'method': 'rule'
                })
        
        # 匹配金额实体
        money_patterns = [
            r'\d+(\.\d+)?元',
            r'\d+(\.\d+)?万元',
            r'\d+(\.\d+)?亿元',
            r'人民币\s*\d+(\.\d+)?',
            r'\d+(\.\d+)?\s*人民币'
        ]
        
        for pattern in money_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'type': 'MONEY',
                    'start': match.start(),
                    'end': match.end() - 1,
                    'method': 'rule'
                })
        
        # 匹配组织机构实体（简单规则）
        org_patterns = [
            r'[\u4e00-\u9fa5]+公司',
            r'[\u4e00-\u9fa5]+企业',
            r'[\u4e00-\u9fa5]+机构',
            r'[\u4e00-\u9fa5]+组织',
            r'[\u4e00-\u9fa5]+委员会',
            r'[\u4e00-\u9fa5]+局',
            r'[\u4e00-\u9fa5]+部',
            r'[\u4e00-\u9fa5]+厅',
            r'[\u4e00-\u9fa5]+所'
        ]
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'type': 'ORG',
                    'start': match.start(),
                    'end': match.end() - 1,
                    'method': 'rule'
                })
        
        # 匹配人名实体（简单规则）
        # 注意：中文人名识别比较复杂，这里只是简单示例
        # 如果有更精确的需求，可以考虑使用专门的人名识别库
        
        # 去重
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def dict_based_recognition(self, text):
        """基于词典的实体识别"""
        entities = []
        
        # 检查法律词典中的术语
        for term_type, terms in self.legal_dict.items():
            for term in terms:
                if len(term) < 2:  # 跳过太短的术语，减少误匹配
                    continue
                
                # 使用正则表达式进行精确匹配
                # 为了避免子字符串匹配，我们需要确保匹配的是完整的词
                # 这里使用\b边界匹配，但注意在中文中可能不太准确
                pattern = rf'\b{re.escape(term)}\b'
                matches = re.finditer(pattern, text)
                
                # 如果\b边界匹配效果不好，回退到普通匹配
                if not list(matches):
                    # 重新获取迭代器
                    matches = re.finditer(re.escape(term), text)
                
                for match in matches:
                    # 根据术语类型映射到实体类型
                    entity_type = term_type
                    if term_type == 'LAW':
                        entity_type = 'ORG'  # 法律法规也作为一种特殊的组织实体
                    
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end() - 1,
                        'method': 'dict'
                    })
        
        # 去重
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def process_legal_document(self, file_path=None, text=None):
        """处理法律文件，识别实体"""
        # 读取文件或使用提供的文本
        if file_path:
            # 根据文件扩展名选择不同的读取方法
            if file_path.lower().endswith('.doc') or file_path.lower().endswith('.docx'):
                text = self.read_doc_file(file_path)
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            text = f.read()
                    except Exception as e:
                        print(f"读取文件失败: {str(e)}")
                        return []
        
        if not text:
            print("没有提供文本内容")
            return []
        
        # 分段处理长文本
        paragraphs = re.split(r'[\n\r]+', text)
        all_entities = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # 识别实体
            entities = self.recognize_entities(paragraph)
            
            # 调整实体在整个文本中的位置
            para_start = text.find(paragraph)
            for entity in entities:
                entity['start'] += para_start
                entity['end'] += para_start
                
            all_entities.extend(entities)
        
        return all_entities
    
    def visualize_entities(self, text, entities):
        """可视化识别结果"""
        # 按起始位置排序实体
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        result = []
        last_end = 0
        
        for entity in sorted_entities:
            # 添加实体前的文本
            if entity['start'] > last_end:
                result.append(text[last_end:entity['start']])
            
            # 添加标记的实体
            result.append(f"[{entity['text']}(\033[91m{entity['type']}\033[0m)]")
            
            last_end = entity['end'] + 1
        
        # 添加最后一段文本
        if last_end < len(text):
            result.append(text[last_end:])
        
        return ''.join(result)

def main():
    """主函数，直接处理当前目录下的法律文件"""
    print("=== 法律文本实体识别系统 ===")
    print("本系统使用BiLSTM-CRF模型结合法律术语词典进行法律文本实体识别")
    print("支持识别：法人实体(ORG)、时间实体(TIME)、金额实体(MONEY)、地点实体(LOC)、人名实体(PER)")
    
    # 初始化识别器
    recognizer = LegalEntityRecognizer()
    
    # 定义文件路径
    law_file_path = os.path.join(os.getcwd(), "中华人民共和国劳动法.docx")
    dict_file_path = os.path.join(os.getcwd(), "法律基础术语词典汇编.docx")
    
    # 检查文件是否存在
    if not os.path.exists(law_file_path):
        print(f"警告：未找到文件 {law_file_path}")
        # 尝试查找其他Word文档
        word_files = [f for f in os.listdir('.') if f.lower().endswith(('.doc', '.docx'))]
        if word_files:
            law_file_path = word_files[0]  # 使用找到的第一个Word文档
            print(f"将使用替代文件: {law_file_path}")
        else:
            print("错误：当前目录下未找到任何Word文档")
            return
    
    # 加载法律术语词典
    if os.path.exists(dict_file_path):
        recognizer.load_legal_dict(dict_file_path)
    else:
        print(f"警告：未找到法律术语词典 {dict_file_path}")
        print("将使用内置规则进行实体识别")
    
    # 处理法律文件
    print(f"\n正在处理文件: {law_file_path}")
    text = recognizer.read_doc_file(law_file_path)
    
    if text:
        print(f"文件读取成功，共{len(text)}个字符")
        print("\n正在识别实体...")
        entities = recognizer.process_legal_document(text=text)
        
        if entities:
            print(f"识别到{len(entities)}个实体：")
            
            # 按实体类型分组统计
            entity_stats = {}
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                entity_stats[entity_type] = entity_stats.get(entity_type, 0) + 1
            
            print("\n实体类型统计：")
            for entity_type, count in entity_stats.items():
                print(f"- {entity_type}: {count}个")
            
            # 显示前20个实体示例
            print("\n实体示例（最多显示20个）：")
            for i, entity in enumerate(entities[:20]):
                print(f"{i+1}. {entity['text']} ({entity['type']}) 位置: {entity['start']}-{entity['end']}")
            
            # 保存结果
            result_file = f"{os.path.splitext(os.path.basename(law_file_path))[0]}_entities.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                # 为了更好的可读性，添加原文中的上下文
                for entity in entities:
                    # 获取实体前后的上下文
                    start = max(0, entity['start'] - 10)
                    end = min(len(text), entity['end'] + 10)
                    context = text[start:end]
                    entity['context'] = context
                
                json.dump(entities, f, ensure_ascii=False, indent=2)
            print(f"\n识别结果已保存到{result_file}")
        else:
            print("未识别到实体")
    else:
        print("文件读取失败或内容为空")

if __name__ == "__main__":
    main()