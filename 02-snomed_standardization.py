''' 
  由于教学目的，目前SNOMED_ALL.csv文件中的标准属于都是英文版本，本程序中，输入的案例，都需要使用英文病例
  如果要进行中文的转换，需要在国内找到权威的，中文翻译版的SNOMEL CT术语库文件，请到国内权威网站下载
'''

# 导入必要的库
import pandas as pd  # 用于数据处理和CSV文件读取
import re  # 用于正则表达式操作，进行文本处理
import json  # 用于JSON数据的序列化和反序列化
from collections import defaultdict  # 用于创建默认值为列表的字典
import os  # 用于文件路径和文件存在性检查

class SNOMEDStandardizer:
    """ 
    SNOMED CT术语标准化器类
    负责加载SNOMED数据，从临床文本中提取医疗术语，并将其标准化为SNOMED CT标准术语
    """
    def __init__(self, snomed_file_path):
        """
        初始化SNOMED标准化器，设置数据文件路径并加载数据
        
        参数:
            snomed_file_path (str): SNOMED数据文件的路径
        """
        self.snomed_file_path = snomed_file_path  # SNOMED数据文件路径
        self.snomed_data = None  # 存储加载的SNOMED数据
        self.term_to_concept = defaultdict(list)  # 术语到概念ID的映射字典
        self.concept_to_term = {}  # 概念ID到首选术语的映射字典
        self.load_snomed_data()  # 初始化时自动加载SNOMED数据
        
    def load_snomed_data(self):
        """
        加载SNOMED数据文件并构建术语到概念ID和概念ID到术语的映射关系
        支持不同编码格式的文件读取，并在加载失败时提供示例数据
        """
        try:
            # 读取SNOMED数据文件
            print(f"正在加载SNOMED数据文件: {self.snomed_file_path}")
            # 尝试不同的编码读取文件，以增加兼容性
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.snomed_data = pd.read_csv(self.snomed_file_path, encoding=encoding)
                    break  # 读取成功则跳出循环
                except UnicodeDecodeError:
                    continue  # 解码失败则尝试下一个编码
            
            # 检查是否成功读取数据
            if self.snomed_data is None:
                raise Exception("无法读取SNOMED数据文件，请检查文件格式和编码")
            
            print(f"成功加载SNOMED数据，共 {len(self.snomed_data)} 条记录")
            
            # 显示前几行数据和列名，以便用户了解数据结构
            print("\n数据文件的列名:")
            print(self.snomed_data.columns.tolist())
            
            print("\n数据前5行:")
            print(self.snomed_data.head())
            
            # 根据SNOMED_ALL.csv的实际格式构建映射关系
            # 从CSV内容来看，第一列是概念ID，第二列是术语名称
            if len(self.snomed_data.columns) >= 2:
                concept_col = self.snomed_data.columns[0]  # 第一列是概念ID
                term_col = self.snomed_data.columns[1]     # 第二列是术语名称
                
                print(f"使用 {concept_col} 作为概念ID列，{term_col} 作为术语列")
                
                # 遍历数据，构建术语到概念ID的映射和概念ID到术语的映射
                for _, row in self.snomed_data.iterrows():
                    term = str(row[term_col]).lower()  # 转换为小写以实现不区分大小写的匹配
                    concept_id = str(row[concept_col])
                    self.term_to_concept[term].append(concept_id)  # 一个术语可能对应多个概念ID
                    self.concept_to_term[concept_id] = term  # 直接使用第二列作为首选术语
            else:
                raise Exception("SNOMED数据文件格式不符合预期，至少需要包含两列数据")
            
            print(f"构建完成术语映射，共 {len(self.term_to_concept)} 个不同的术语")
            
        except Exception as e:
            # 捕获所有可能的异常，并提供友好的错误信息
            print(f"加载SNOMED数据时出错: {str(e)}")
            # 创建一个简单的示例映射用于测试，确保程序能够继续运行
            print("创建示例映射用于测试...")
            self.term_to_concept = {
                'diabetes': ['123456'],
                'diabetes mellitus': ['123456'],
                'heart attack': ['654321'],
                'myocardial infarction': ['654321'],
                'hypertension': ['987654'],
                'high blood pressure': ['987654'],
                'pneumonia': ['456789'],
                'stroke': ['789456'],
                'cerebrovascular accident': ['789456'],
            }
            self.concept_to_term = {
                '123456': 'Diabetes mellitus',
                '654321': 'Myocardial infarction',
                '987654': 'Hypertension',
                '456789': 'Pneumonia',
                '789456': 'Cerebrovascular accident',
            }
    
    def extract_medical_terms(self, clinical_text):
        """
        从临床文本中提取可能的医疗术语
        采用两种策略: 1) 直接匹配SNOMED术语库中的术语 2) 宽松匹配策略
        
        参数:
            clinical_text (str): 临床文本内容
        
        返回:
            list: 提取出的医疗术语列表
        """
        # 转换为小写便于匹配
        text = clinical_text.lower()
        
        # 使用正则表达式移除标点符号，但保留连字符和空格
        # 这有助于提高术语匹配的准确性
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # 提取可能的医疗术语，使用集合避免重复
        extracted_terms = set()
        
        # 方法1: 直接匹配SNOMED术语库中的术语
        for term in self.term_to_concept.keys():
            if len(term.split()) >= 1:  # 至少1个单词
                # 使用边界匹配确保我们找到的是完整的术语
                # 对于单个单词的术语，使用单词边界
                if len(term.split()) == 1:
                    pattern = r'\b{}\b'.format(re.escape(term))
                # 对于多个单词的术语，使用空格分隔
                else:
                    pattern = r'{}'.format(re.escape(term))
                
                # 检查文本中是否包含该术语
                if re.search(pattern, text):
                    extracted_terms.add(term)
        
        # 如果没有提取到术语，尝试更宽松的匹配策略
        if not extracted_terms:
            print("尝试宽松匹配策略...")
            words = text.split()
            # 对于文本中的每个单词，检查它是否是某个术语的一部分
            for i in range(len(words)):
                # 尝试1-3个单词的组合，增加匹配到多词术语的概率
                for j in range(1, min(4, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+j])
                    # 检查术语库中是否有包含这个短语的术语
                    for term in self.term_to_concept.keys():
                        if phrase in term:
                            extracted_terms.add(term)
                            break  # 找到匹配后跳出内层循环，继续下一个短语
        
        # 将集合转换为列表并返回
        return list(extracted_terms)
    
    def standardize_terms(self, extracted_terms):
        """
        将提取的术语标准化为SNOMED CT标准术语
        
        参数:
            extracted_terms (list): 从临床文本中提取的术语列表
        
        返回:
            dict: 标准化后的术语字典，键为提取的术语，值为标准化信息列表
        """
        standardized_terms = {}
        
        # 遍历提取的每个术语
        for term in extracted_terms:
            # 检查术语是否在术语映射中
            if term in self.term_to_concept:
                # 获取该术语对应的所有概念ID
                concept_ids = self.term_to_concept[term]
                # 对于每个概念ID，获取标准术语
                for concept_id in concept_ids:
                    if concept_id in self.concept_to_term:
                        # 获取标准术语
                        standard_term = self.concept_to_term[concept_id]
                        # 如果该术语还没有标准化结果，创建一个空列表
                        if term not in standardized_terms:
                            standardized_terms[term] = []
                        # 添加标准化信息
                        standardized_terms[term].append({
                            'concept_id': concept_id,  # SNOMED概念ID
                            'standard_term': standard_term  # 标准化后的术语
                        })
        
        # 返回标准化结果
        return standardized_terms
    
    def process_clinical_note(self, clinical_note):
        """
        处理临床病历，提取并标准化医疗术语
        
        参数:
            clinical_note (str): 临床病历文本
        
        返回:
            dict: 包含原始病历、提取的术语和标准化结果的字典
        """
        print(f"\n处理病历: {clinical_note}")
        
        # 提取医疗术语
        extracted_terms = self.extract_medical_terms(clinical_note)
        print(f"提取的医疗术语: {extracted_terms}")
        
        # 标准化术语
        standardized_terms = self.standardize_terms(extracted_terms)
        
        # 返回处理结果
        return {
            'original_note': clinical_note,  # 原始病历文本
            'extracted_terms': extracted_terms,  # 提取的医疗术语
            'standardized_terms': standardized_terms  # 标准化后的术语
        }

# 主函数，用于演示和测试SNOMED标准化器
def main():
    # SNOMED数据文件默认路径
    default_file_path = "SNOMED_ALL.csv"
    
    # 询问用户SNOMED数据文件的路径，允许用户自定义路径
    file_path = input(f"请输入SNOMED数据文件路径 (默认: {default_file_path}): ").strip()
    if not file_path:
        file_path = default_file_path  # 如果用户未输入，使用默认路径
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}，将使用示例数据进行演示")
    
    # 创建标准化器实例
    standardizer = SNOMEDStandardizer(file_path)
    
    # 等待用户输入要标准化的病历文本
    print("\n===== SNOMED术语标准化 =====")
    print("请输入要标准化的病历文本:")
    user_note = input().strip()
    
    # 处理用户输入的内容
    if user_note:
        # 调用处理函数处理用户输入的病历
        result = standardizer.process_clinical_note(user_note)
        print(f"\n标准化结果:")
        
        # 检查是否有标准化结果
        if result['standardized_terms']:
            # 以格式化的JSON形式打印标准化结果，便于阅读
            print(json.dumps(result['standardized_terms'], ensure_ascii=False, indent=2))
        else:
            # 如果没有找到匹配的标准术语，提供友好提示
            print("未找到匹配的标准术语")
            # 提供一些建议的术语示例，基于CSV文件中的内容
            print("\n建议尝试的术语示例:")
            sample_terms = ['Somatic hallucination', 'Allergic transfusion reaction', 'Hypertension', 'Pneumonia']
            for term in sample_terms:
                if term.lower() in standardizer.term_to_concept:
                    concept_ids = standardizer.term_to_concept[term.lower()]
                    print(f"- {term}: 概念ID {concept_ids}")
    else:
        # 如果用户未输入有效的病历文本，提供提示
        print("未输入有效的病历文本")

# 当脚本直接运行时，执行主函数
if __name__ == "__main__":
    main()