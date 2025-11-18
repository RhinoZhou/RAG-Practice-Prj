
# -*- coding: utf-8 -*-
'''
法律问题分类器，用于将用户问题按照法律领域分类到不同类型的知识库中。
主要功能包括：
1. 基于关键词匹配的问题意图识别
2. 支持多类别标签的识别和返回
3. 支持的分类类别包括：法律条文、法律书籍、法律文书模板、法律案例和司法考试问题
该模块为RAG系统的前置处理组件，用于确定后续检索的知识库范围。
'''

import sys
import os

def setup_dependencies():
    """
    检查并自动安装必要的依赖包
    目前该模块仅使用Python标准库，不需要额外依赖
    """
    # 记录安装状态
    print("Questionary模块仅使用Python标准库，不需要额外依赖包。")

# 自动安装依赖
setup_dependencies()


class QuestionClassifier:
    """
    法律问题分类器类
    
    该类用于对用户提出的法律相关问题进行分类，确定问题涉及的法律知识库类型。
    分类基于关键词匹配算法，能够识别多种法律问题类型。
    """
    
    def __init__(self) -> None:
        """
        初始化问题分类器
        
        设置意图识别的关键词词典，每个类别对应一组用于匹配的关键词
        """
        # 定义各类别及其对应的关键词列表
        self.intention_reg = {
            "legal_articles": ["法律","法条","判罚","罪名","刑期","本院认为","观点","条文","条款","案由","定罪","量刑","法院观点","法官意见","条例","规定","执法依据","法规","罪责","诉讼","立法"],
            "legal_books": ["依据","法律读物","法学著作","法律参考书","法典","法规","参考书","读本","丛书","法理","法学","法哲学","法学家","著作","文献","学说","学术"],
            "legal_templates": ["文书","起诉书","法律文书","起诉状","判决书","裁定书","答辩状","法律合同","协议","证据","证明","合同","格式","模板","样本","规范","范本"],
            "legal_cases": ["法律","判罚","事实","案例","罪名","刑期","本院认为","观点","法律案件","典型案例","案情","案由","定罪","量刑","证据","判例","裁决","仲裁","先例","判决","司法"],
            "JudicialExamination": ["选项","选择","A,B,C,D","ABCD","A,B,C和D","考试","题目","法考","法律考试","考题","选择题","判断题","多选题","单选题","填空题","辨析题","案例分析题","答案","试题","试卷","法学","考研","司法考试","律师考试"]
        }
        print("QuestionClassifier初始化完成，已加载5个分类类别，共包含{}个关键词。".format(
            sum(len(vals) for vals in self.intention_reg.values())))

    def classify(self, question):
        """
        问题分类主函数
        
        Args:
            question (str): 用户提出的问题文本
            
        Returns:
            dict: 包含分类结果的字典，格式为{'kg_names': set(分类标签集合)}
            
        Raises:
            TypeError: 当输入不是字符串类型时抛出
        """
        if not isinstance(question, str):
            raise TypeError("输入问题必须是字符串类型")
            
        # 初始化返回数据结构
        data = {}
        
        # 执行关键词匹配，获取分类标签
        kg_names = self.key_words_match_intention(question)
        data['kg_names'] = kg_names
        
        # 记录分类结果日志
        if kg_names:
            print(f"问题分类结果: {question} -> {kg_names}")
        else:
            print(f"未匹配到任何类别: {question}")
            
        return data
    
    def key_words_match_intention(self, input_text):
        """
        关键词匹配函数，用于识别问题的意图类别
        
        Args:
            input_text (str): 待分类的文本
            
        Returns:
            set: 匹配到的类别标签集合
        """
        # 使用集合存储匹配到的类别，避免重复
        kg_names = set()
        
        # 遍历所有类别及其关键词
        for category, keywords in self.intention_reg.items():
            # 检查是否有任何关键词出现在输入文本中
            if any(keyword in input_text for keyword in keywords):
                kg_names.add(category)
        
        return kg_names


# 测试代码
if __name__ == "__main__":
    """
    示例用法测试
    """
    classifier = QuestionClassifier()
    
    # 测试示例问题
    test_questions = [
        "请解释《民法典》第1012条关于姓名权的规定",
        "如何撰写一份有效的离婚协议书？",
        "张三盗窃案的量刑标准是什么？",
        "推荐几本关于刑法学的经典著作",
        "法考选择题：以下关于正当防卫的说法正确的是（ ）A.可以超过必要限度 B.必须针对正在进行的不法侵害"
    ]
    
    print("\n===== 问题分类测试 =====")
    for question in test_questions:
        result = classifier.classify(question)
        print(f"问题: {question}")
        print(f"分类结果: {result}\n")