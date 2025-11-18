# -*- coding: utf-8 -*-
'''
智能语言接口模块，提供统一的语言模型交互接口和多模型适配。
定义了语言处理的抽象基类和具体实现，支持本地部署和API调用，
为知识库增强系统提供文本生成、智能问答和对话交互功能。核心特性：
1. 标准化模型接口设计
2. 多模型架构支持
3. 批量处理优化
4. 自动依赖管理
5. 智能资源调度
'''

import os
import sys
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


# 自动依赖管理
def setup_dependencies():
    """
    自动检查并安装必要的依赖库
    """
    try:
        import pip
        
        required_libraries = [
            'transformers>=4.36.0',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'tqdm>=4.65.0',
            'accelerate>=0.25.0'
        ]
        
        print("检查并安装项目依赖...")
        for lib in required_libraries:
            try:
                # 尝试导入库
                lib_name = lib.split('>=')[0].replace('-', '_')
                if lib_name == 'transformers':
                    import transformers
                elif lib_name == 'torch':
                    import torch
                elif lib_name == 'torchvision':
                    import torchvision
                elif lib_name == 'tqdm':
                    import tqdm
                elif lib_name == 'accelerate':
                    import accelerate
            except ImportError:
                print(f"安装依赖库: {lib}...")
                pip.main(['install', lib])
        
        print("✓ 依赖库安装成功")
    except Exception as e:
        print(f"警告: 依赖安装出错: {str(e)}")
        print("请手动配置所需依赖")


# 执行依赖管理
setup_dependencies()


# 对话模板配置
TEMPLATE_CONFIG = dict(
    KNOWLEDGE_PROMPT="""基于提供的上下文回答用户问题。如无法回答，请如实说明。始终使用中文回复。
用户问题: {query}
相关知识上下文：
---
{context}
---
若上下文信息不足，请明确表示数据库中无此内容。
回答:""",
    
    CONTENT_SUMMARY="""请对以下文本内容进行简明扼要的总结，使用中文表达，保留核心信息：
{text}
总结:""",
    
    DIRECT_RESPONSE="""请回答以下问题，使用中文：
问题: {query}
回答:"""
)


class LanguageProcessor(ABC):
    """
    语言处理抽象基类，定义语言处理组件必须实现的标准接口
    """
    
    def __init__(self):
        """初始化语言处理器"""
        self.processor_name = "Base Processor"
        self.ready = False
    
    @abstractmethod
    def generate_response(self, query: str, context_history: List[Dict] = None, max_output_length: int = 512) -> str:
        """
        生成针对输入的响应
        
        Args:
            query: 用户查询内容
            context_history: 历史上下文记录
            max_output_length: 最大输出字符长度
            
        Returns:
            生成的文本响应
        """
        pass
    
    def format_query(self, query: str, context: str = "", template: str = 'KNOWLEDGE_PROMPT') -> str:
        """
        根据模板格式化输入查询
        
        Args:
            query: 用户查询
            context: 上下文信息（可选）
            template: 使用的模板配置名称
            
        Returns:
            格式化后的输入文本
        """
        try:
            if template in TEMPLATE_CONFIG:
                return TEMPLATE_CONFIG[template].format(query=query, context=context, text=context)
            else:
                # 模板不存在时使用默认配置
                return TEMPLATE_CONFIG['KNOWLEDGE_PROMPT'].format(query=query, context=context)
        except Exception as e:
            print(f"警告: 查询格式化出错: {str(e)}")
            return f"用户查询: {query}\n上下文信息: {context}\n回答:"
    
    def process_batch_queries(self, batch_queries: List[tuple], max_output_length: int = 512) -> List[str]:
        """
        批量处理多个查询请求
        
        Args:
            batch_queries: 批量查询列表，每个元素为(query, history)元组
            max_output_length: 最大输出长度
            
        Returns:
            处理结果列表
        """
        outputs = []
        for query, history in batch_queries:
            try:
                result = self.generate_response(query, history, max_output_length)
                outputs.append(result)
            except Exception as e:
                print(f"警告: 批量处理异常: {str(e)}")
                outputs.append("处理请求时发生错误")
        return outputs
    
    def check_health(self) -> bool:
        """
        验证处理器功能状态
        
        Returns:
            功能是否正常
        """
        try:
            test_input = "你好"
            test_output = self.generate_response(test_input)
            return len(test_output.strip()) > 0
        except Exception as e:
            print(f"错误: 功能检查失败: {str(e)}")
            return False


class LocalChatProcessor(LanguageProcessor):
    """
    本地部署的对话模型实现
    支持多种开源大语言模型的本地化部署
    """
    
    def __init__(self, model_path: str = '') -> None:
        """
        初始化本地对话处理器
        
        Args:
            model_path: 模型文件路径
        """
        super().__init__()
        self.processor_name = "Local Dialogue Processor"
        self.model_location = model_path
        
        # 标准模型路径列表（优先级排序）
        self.standard_paths = [
            model_path,
            "./models/Local-Instruct",
            "./model_hub/local/local-instruct",
            "./resources/models/local-instruct"
        ]
        
        # 初始化模型资源
        self.initialize_model()
    
    def initialize_model(self) -> bool:
        """
        初始化模型资源，尝试多个可能的路径
        
        Returns:
            初始化是否成功
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"初始化{self.processor_name}...")
            
            # 自动检测计算设备
            self.compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  计算设备: {self.compute_device}")
            
            # 尝试多个模型路径
            initialization_success = False
            for path in self.standard_paths:
                if not path:  # 跳过空路径
                    continue
                    
                try:
                    print(f"  尝试路径: {path}")
                    
                    # 加载分词器
                    self.text_encoder = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # 配置优化参数
                    resource_config = {
                        'torch_dtype': torch.float16 if self.compute_device.type == 'cuda' else torch.float32,
                        'trust_remote_code': True,
                        'low_cpu_mem_usage': True
                    }
                    
                    # 根据设备自动配置部署
                    if self.compute_device.type == 'cuda':
                        resource_config['device_map'] = 'auto'
                    else:
                        resource_config['device_map'] = 'cpu'
                    
                    # 加载模型资源
                    self.language_model = AutoModelForCausalLM.from_pretrained(path, **resource_config)
                    
                    # 确保模型在正确设备上
                    if self.compute_device.type != 'cuda' and not hasattr(self.language_model, 'hf_device_map'):
                        self.language_model = self.language_model.to(self.compute_device)
                    
                    # 性能优化
                    if self.compute_device.type == 'cuda':
                        self.language_model = torch.compile(self.language_model)  # 启用编译优化
                    
                    initialization_success = True
                    self.ready = True
                    print(f"✓ 模型初始化成功: {path}")
                    break
                    
                except Exception as e:
                    print(f"  初始化失败: {str(e)}")
                    continue
            
            if not initialization_success:
                raise Exception("所有模型路径初始化失败")
                
            return True
            
        except Exception as e:
            print(f"错误: 模型初始化异常: {str(e)}")
            print("请确认模型路径有效性和文件完整性")
            self.ready = False
            return False
    
    def generate_response(self, query: str, context_history: List[Dict] = None, max_output_length: int = 512) -> str:
        """
        生成针对查询的响应
        
        Args:
            query: 用户查询内容
            context_history: 历史上下文记录
            max_output_length: 最大输出长度
            
        Returns:
            生成的文本响应
        """
        if not self.ready:
            print("警告: 处理器未就绪，尝试重新初始化")
            if not self.initialize_model():
                return "模型初始化失败，请检查配置"
        
        try:
            # 初始化历史上下文
            if context_history is None:
                context_history = []
            
            # 构建对话序列
            dialogue = context_history.copy()
            dialogue.append({'role': 'user', 'content': query})
            
            # 应用对话模板
            formatted_input = self.text_encoder.apply_chat_template(
                dialogue,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 生成响应
            start_time = time.time()
            result = self._process_text([formatted_input], max_output_length)[0]
            end_time = time.time()
            
            print(f"  处理耗时: {end_time - start_time:.2f}秒")
            return result
            
        except Exception as e:
            print(f"错误: 生成响应异常: {str(e)}")
            return "处理请求时发生错误"
    
    def _process_text(self, input_texts, max_output_length: int = 512) -> List[str]:
        """
        文本处理核心方法
        
        Args:
            input_texts: 输入文本列表
            max_output_length: 最大输出长度
            
        Returns:
            处理后的文本列表
        """
        try:
            # 编码输入文本
            processed_inputs = self.text_encoder(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096  # 限制输入长度防止内存问题
            ).to(self.compute_device)
            
            # 配置生成参数
            generation_params = {
                'max_new_tokens': max_output_length,
                'temperature': 0.7,
                'top_p': 0.95,
                'do_sample': True,
                'pad_token_id': self.text_encoder.eos_token_id
            }
            
            # 执行文本生成
            with self.text_encoder.as_target_tokenizer():
                output_ids = self.language_model.generate(
                    processed_inputs.input_ids,
                    attention_mask=processed_inputs.attention_mask,
                    **generation_params
                )
            
            # 提取和格式化输出
            results = []
            for input_ids, output_token_ids in zip(processed_inputs.input_ids, output_ids):
                # 提取新增部分
                new_tokens = output_token_ids[len(input_ids):]
                # 解码并清理
                text_result = self.text_encoder.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                results.append(text_result)
            
            return results
            
        except Exception as e:
            print(f"错误: 文本处理异常: {str(e)}")
            # 返回错误提示
            return ["文本处理失败"] * len(input_texts)


class APIChatProcessor(LanguageProcessor):
    """
    基于API的对话模型接口
    支持通过网络API调用云端语言模型服务
    """
    
    def __init__(self, api_credentials: str = None, model_version: str = "standard") -> None:
        """
        初始化API对话处理器
        
        Args:
            api_credentials: API访问凭证
            model_version: 模型版本标识
        """
        super().__init__()
        self.processor_name = "Cloud API Processor"
        self.api_key = api_credentials or os.getenv("API_ACCESS_KEY", "")
        self.model_type = model_version
        self.ready = False
        
        # 初始化API客户端
        self._setup_client()
    
    def _setup_client(self) -> bool:
        """
        配置API客户端连接
        """
        try:
            # 尝试导入API客户端库
            try:
                import openai
                self.api_client = openai
                
                # 配置API凭证
                if self.api_key:
                    self.api_client.api_key = self.api_key
                    self.ready = True
                    print(f"✓ API客户端初始化成功: {self.model_type}")
                    return True
                else:
                    print("警告: 未提供API访问凭证")
                    self.ready = False
                    return False
                    
            except ImportError:
                print("安装API客户端库...")
                import pip
                pip.main(['install', 'openai'])
                import openai
                self.api_client = openai
                
                if self.api_key:
                    self.api_client.api_key = self.api_key
                    self.ready = True
                    print(f"✓ API客户端初始化成功: {self.model_type}")
                    return True
                else:
                    print("警告: 未提供API访问凭证")
                    self.ready = False
                    return False
                    
        except Exception as e:
            print(f"错误: API客户端配置失败: {str(e)}")
            self.ready = False
            return False
    
    def generate_response(self, query: str, context_history: List[Dict] = None, max_output_length: int = 512) -> str:
        """
        生成API响应
        
        Args:
            query: 用户查询内容
            context_history: 历史上下文记录
            max_output_length: 最大输出长度
            
        Returns:
            API返回的响应内容
        """
        if not self.ready:
            print("警告: API客户端未就绪，尝试重新配置")
            if not self._setup_client():
                return "API客户端配置失败，请提供有效凭证"
        
        try:
            # 构建请求消息
            request_messages = []
            
            # 包含历史上下文
            if context_history:
                request_messages.extend(context_history)
            
            # 添加当前查询
            request_messages.append({"role": "user", "content": query})
            
            # 执行API请求
            start_time = time.time()
            api_response = self.api_client.chat.completions.create(
                model=self.model_type,
                messages=request_messages,
                max_tokens=max_output_length,
                temperature=0.7,
                top_p=0.95
            )
            end_time = time.time()
            
            print(f"  API响应耗时: {end_time - start_time:.2f}秒")
            
            # 提取结果内容
            return api_response.choices[0].message.content
            
        except Exception as e:
            print(f"错误: API请求异常: {str(e)}")
            return "API服务访问失败"


# 创建语言处理器实例的工厂函数
def create_language_processor(processor_type: str = "local", **kwargs) -> LanguageProcessor:
    """
    创建语言处理组件实例的工厂函数
    
    Args:
        processor_type: 处理器类型，支持 "local", "api"
        **kwargs: 初始化配置参数
        
    Returns:
        语言处理器实例
    """
    try:
        if processor_type.lower() == "local":
            return LocalChatProcessor(**kwargs)
        elif processor_type.lower() == "api":
            return APIChatProcessor(**kwargs)
        else:
            print(f"警告: 未知处理器类型: {processor_type}，使用默认本地处理器")
            return LocalChatProcessor(**kwargs)
    except Exception as e:
        print(f"错误: 处理器实例化异常: {str(e)}")
        # 返回备用处理器
        return FallbackLanguageProcessor()


class FallbackLanguageProcessor(LanguageProcessor):
    """
    备用语言处理器，当其他处理器初始化失败时使用
    """
    
    def __init__(self):
        super().__init__()
        self.processor_name = "Fallback Processor"
        self.ready = True
    
    def generate_response(self, query: str, context_history: List[Dict] = None, max_output_length: int = 512) -> str:
        return "语言处理服务暂不可用，请检查系统配置和依赖安装状态。确保所有必要组件已正确配置。"


if __name__ == '__main__':
    # 示例使用
    try:
        # 创建语言处理器
        print("=== 语言处理模块测试 ===")
        
        # 测试本地处理器
        print("\n1. 测试本地语言处理器:")
        processor = create_language_processor("local")
        
        # 功能检查
        if processor.check_health():
            print("处理器功能检查通过")
            
            # 测试基础对话
            test_query = "你好，请简要介绍一下自己"
            print(f"\n查询: {test_query}")
            result = processor.generate_response(test_query)
            print(f"响应: {result}")
            
            # 测试知识库查询
            knowledge_context = "本地部署模型是一种可在私有环境中运行的AI系统，无需联网即可使用。"
            knowledge_query = processor.format_query("本地模型有什么优势？", knowledge_context)
            print(f"\n知识库查询:\n{knowledge_query}")
            knowledge_result = processor.generate_response(knowledge_query)
            print(f"知识响应: {knowledge_result}")
        else:
            print("处理器功能检查未通过")
            
        # 注意：API处理器需要访问凭证
        print("\n如需测试API处理器，请配置有效的访问凭证")
        
    except Exception as e:
        print(f"测试运行异常: {str(e)}")
    
    

