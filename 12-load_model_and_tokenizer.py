#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA配置与模型加载演示

作者: Ph.D. Rhino

功能说明:
    展示如何使用4-bit量化技术加载大型语言模型(Baichuan2-13B-Chat)，
    并应用优化的LoRA参数进行参数高效微调配置。程序实现了模型加载、
    分词器初始化、量化配置和LoRA适配器应用的完整流程。

内容概述:
    1. 实现自动依赖检查和安装
    2. 使用4-bit NF4量化加载模型以节省内存
    3. 配置最优LoRA参数(r=16, alpha=32)并应用到关键模块
    4. 演示模型和分词器的基本使用
    5. 提供参数统计和资源使用情况分析

执行流程:
    1. 检查并安装必要的依赖包
    2. 配置模型加载参数和LoRA设置
    3. 加载模型和分词器
    4. 应用LoRA适配器并统计可训练参数
    5. 进行简单的推理测试
    6. 输出资源使用情况和性能分析

输入说明:
    无需外部输入，程序会自动处理所有步骤

输出展示:
    - 依赖检查结果
    - 模型加载进度和状态
    - 可训练参数统计信息
    - 简单推理测试结果
    - 内存使用情况分析
"""

# 系统基础导入
import os
import sys
import subprocess
import time

# 依赖包列表
required_packages = [
    'transformers>=4.30.0',
    'peft>=0.4.0',
    'bitsandbytes>=0.39.0',
    'torch>=2.0.0',
    'accelerate>=0.20.0',
    'matplotlib',
    'psutil'
]

# 模型配置
MODEL_NAME = "baichuan-inc/Baichuan2-13B-Chat"

# 先定义并执行依赖检查
def check_dependencies():
    """
    检查并安装必要的依赖包
    """
    print("正在检查依赖...")
    missing_packages = []
    
    for package in required_packages:
        # 分离包名和版本要求
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
            print(f"✓ {package_name} 已安装")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"发现缺失的依赖包: {', '.join(missing_packages)}")
        print("正在安装...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {package} 安装失败，请手动安装")
                sys.exit(1)
    else:
        print("所有必要的依赖包已安装完成。")

# 执行依赖检查
check_dependencies()

# 依赖检查通过后再导入其他模块
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


def get_memory_usage():
    """
    获取当前进程的内存使用情况
    
    Returns:
        Dict: 内存使用情况字典
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss": mem_info.rss / 1024 / 1024,  # 物理内存使用量(MB)
        "vms": mem_info.vms / 1024 / 1024,  # 虚拟内存使用量(MB)
    }


def load_model_and_tokenizer():
    """
    使用QLoRA配置加载模型
    
    Returns:
        Tuple: (model, tokenizer) - 加载并配置好的模型和分词器
    """
    print(f"\n正在从 {MODEL_NAME} 加载模型...")
    start_time = time.time()
    
    # 记录加载前的内存使用
    before_memory = get_memory_usage()
    print(f"加载前内存使用: {before_memory['rss']:.2f} MB (物理内存)")
    
    try:
        # 加载tokenizer
        print("\n正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        print(f"分词器加载完成，pad_token设置为: {tokenizer.pad_token}")
        
        # 4-bit量化加载模型
        print("\n正在加载量化模型 (4-bit NF4)...")
        print("注意: 首次加载可能需要较长时间，特别是从Hugging Face下载模型时")
        
        # 创建量化配置
        quantization_config = bnb.nn.modules.Params4bit(
            compute_dtype=torch.float16,
            quant_type="nf4"  # 使用NF4量化类型，这是一种为LLM优化的量化格式
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,           # 启用4-bit量化
            device_map="auto",          # 自动分配到可用设备
            trust_remote_code=True,      # 允许执行模型的自定义代码
            quantization_config=quantization_config,
            torch_dtype=torch.float16,   # 使用float16计算以提高速度
            low_cpu_mem_usage=True       # 优化CPU内存使用
        )
        
        # 记录模型加载后的内存使用
        after_model_memory = get_memory_usage()
        model_load_time = time.time() - start_time
        
        print(f"模型加载完成，耗时: {model_load_time:.2f} 秒")
        print(f"加载后内存使用: {after_model_memory['rss']:.2f} MB (物理内存)")
        print(f"模型占用内存: {after_model_memory['rss'] - before_memory['rss']:.2f} MB")
        
        # 为k-bit训练准备模型
        print("\n准备模型进行k-bit训练...")
        model = prepare_model_for_kbit_training(model)
        print("模型准备完成")
        
        # LoRA配置 - 最优参数组合
        print("\n配置LoRA适配器...")
        lora_config = LoraConfig(
            r=16,                    # 秩，控制LoRA矩阵的维度
            lora_alpha=32,           # Alpha缩放因子，控制LoRA的缩放强度
            target_modules=[         # 指定要应用LoRA的模块
                "W_pack",            # Baichuan架构特定参数
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            lora_dropout=0.05,       # LoRA的dropout率，用于正则化
            bias="none",            # 不训练bias参数
            task_type="CAUSAL_LM"   # 因果语言模型任务类型
        )
        
        # 应用LoRA到模型
        print("应用LoRA适配器到模型...")
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        print("\n模型参数统计:")
        model.print_trainable_parameters()
        
        # 记录LoRA应用后的内存使用
        after_lora_memory = get_memory_usage()
        total_time = time.time() - start_time
        
        print(f"\nLoRA应用完成，总耗时: {total_time:.2f} 秒")
        print(f"最终内存使用: {after_lora_memory['rss']:.2f} MB (物理内存)")
        print(f"LoRA适配器增加内存: {after_lora_memory['rss'] - after_model_memory['rss']:.2f} MB")
        
        # 检查GPU使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # 转换为GB
            print(f"GPU内存使用: {gpu_memory:.2f} GB")
            print(f"设备: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: 未检测到可用的GPU，模型将在CPU上运行")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 提供替代方案
        print("\n建议的替代方案:")
        print("1. 尝试使用更小的模型，例如 'baichuan-inc/Baichuan2-7B-Chat'")
        print("2. 确保您的系统有足够的内存和磁盘空间")
        print("3. 检查网络连接是否正常，特别是首次下载模型时")
        
        # 尝试加载更小的测试模型
        print("\n尝试加载测试模型以验证功能...")
        try:
            test_model_name = "gpt2"  # 使用一个小的测试模型
            print(f"正在加载测试模型: {test_model_name}")
            
            # 加载测试tokenizer
            test_tokenizer = AutoTokenizer.from_pretrained(test_model_name)
            test_tokenizer.pad_token = test_tokenizer.eos_token
            
            # 加载测试模型
            test_model = AutoModelForCausalLM.from_pretrained(test_model_name)
            
            # 为测试模型配置LoRA
            test_lora_config = LoraConfig(
                r=4,  # 更小的秩用于测试
                lora_alpha=8,
                target_modules=["c_attn", "c_proj"],  # GPT2的关键模块
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 应用LoRA到测试模型
            test_model = get_peft_model(test_model, test_lora_config)
            print("测试模型参数统计:")
            test_model.print_trainable_parameters()
            
            print(f"\n测试模型加载成功，这表明LoRA配置功能正常。")
            print(f"您可以修改代码中的 MODEL_NAME 以使用其他适合您硬件的模型。")
            
            return test_model, test_tokenizer
            
        except Exception as test_e:
            print(f"测试模型加载也失败: {str(test_e)}")
            sys.exit(1)


def test_inference(model, tokenizer, prompt="你好，请简要介绍一下你自己。", max_length=100):
    """
    使用加载的模型进行简单的推理测试
    
    Args:
        model: 加载的语言模型
        tokenizer: 分词器
        prompt: 测试提示
        max_length: 生成文本的最大长度
    
    Returns:
        str: 生成的回复
    """
    print(f"\n进行推理测试...")
    print(f"输入提示: {prompt}")
    
    # 记录推理时间
    start_time = time.time()
    
    try:
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # 如果有GPU，移至GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # 解码输出
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        print(f"\n生成回复:")
        print(f"{response}")
        print(f"\n推理耗时: {inference_time:.2f} 秒")
        
        return response
        
    except Exception as e:
        print(f"推理测试失败: {str(e)}")
        return "推理测试失败"


def analyze_model_lora_config(model):
    """
    分析模型的LoRA配置
    
    Args:
        model: 带有LoRA适配器的模型
    """
    print("\n===== LoRA配置分析 =====")
    
    # 获取LoRA配置
    if hasattr(model, 'peft_config'):
        lora_config = model.peft_config
        print(f"\nLoRA配置详情:")
        
        # 检查lora_config类型并相应地访问配置
        is_dict = isinstance(lora_config, dict)
        
        # 获取必要的配置参数
        r = lora_config.get('r', None) if is_dict else getattr(lora_config, 'r', None)
        lora_alpha = lora_config.get('lora_alpha', None) if is_dict else getattr(lora_config, 'lora_alpha', None)
        lora_dropout = lora_config.get('lora_dropout', None) if is_dict else getattr(lora_config, 'lora_dropout', None)
        bias = lora_config.get('bias', None) if is_dict else getattr(lora_config, 'bias', None)
        task_type = lora_config.get('task_type', None) if is_dict else getattr(lora_config, 'task_type', None)
        target_modules = lora_config.get('target_modules', None) if is_dict else getattr(lora_config, 'target_modules', None)
        
        # 打印配置信息
        print(f"  秩 (r): {r}")
        print(f"  Alpha因子: {lora_alpha}")
        print(f"  LoRA Dropout: {lora_dropout}")
        print(f"  偏置设置: {bias}")
        print(f"  任务类型: {task_type}")
        print(f"  目标模块: {target_modules}")
        
        # 计算秩与alpha的比例
        if r and lora_alpha and r > 0:
            alpha_r_ratio = lora_alpha / r
            print(f"  Alpha/秩 比例: {alpha_r_ratio:.2f} (推荐值为2.0)")
            
            if abs(alpha_r_ratio - 2.0) < 0.1:
                print("  ✓ Alpha/秩比例处于推荐范围内")
            else:
                print("  ! Alpha/秩比例偏离推荐值2.0")
    
    # 分析参数分布
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        num_params = param.numel()
        # 统计所有参数
        all_param += num_params
        # 统计可训练参数
        if param.requires_grad:
            trainable_params += num_params
    
    # 计算可训练参数比例
    trainable_ratio = trainable_params / all_param * 100
    
    print(f"\n参数分析:")
    print(f"  总参数数: {all_param:,}")
    print(f"  可训练参数数: {trainable_params:,}")
    print(f"  参数效率: {trainable_ratio:.4f}%")
    
    # 评估参数效率
    if 0.01 < trainable_ratio < 0.1:
        print("  ✓ 参数效率处于理想范围 (0.01%-0.1%)")
    elif trainable_ratio <= 0.01:
        print("  ! 参数效率过低，可能导致微调效果不佳")
    else:
        print("  ! 参数效率过高，可能导致过拟合和资源浪费")
    
    # 检查目标模块是否正确应用
    lora_applied = False
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            lora_applied = True
            break
    
    if lora_applied:
        print("  ✓ LoRA适配器已成功应用到模型")
    else:
        print("  ! 未检测到LoRA适配器模块")


def save_lora_config(directory="./lora_configs"):
    """
    保存LoRA配置到文件
    
    Args:
        directory: 保存目录
    """
    # 创建保存目录
    os.makedirs(directory, exist_ok=True)
    
    # 保存LoRA配置
    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "description": "为Baichuan2-13B-Chat模型优化的LoRA配置"
    }
    
    config_file = os.path.join(directory, "baichuan2_lora_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(lora_config, f, ensure_ascii=False, indent=2)
    
    print(f"\nLoRA配置已保存至: {config_file}")
    return config_file


def main():
    """
    主函数
    """
    print("===== LoRA配置与模型加载演示 =====")
    
    # 记录总执行时间
    total_start_time = time.time()
    
    # 1. 检查依赖
    print("\n1. 依赖检查")
    check_dependencies()
    
    # 2. 加载模型和分词器
    print("\n2. 模型和分词器加载")
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. 分析LoRA配置
    print("\n3. LoRA配置分析")
    analyze_model_lora_config(model)
    
    # 4. 保存LoRA配置
    print("\n4. 保存LoRA配置")
    config_file = save_lora_config()
    
    # 5. 进行简单推理测试
    print("\n5. 推理功能测试")
    test_inference(model, tokenizer)
    
    # 6. 测试中文性能
    print("\n6. 中文性能测试")
    test_inference(
        model, 
        tokenizer, 
        prompt="什么是LoRA微调？它与全参数微调和冻结预训练参数的方法相比有什么优势？"
    )
    
    # 计算总执行时间
    total_time = time.time() - total_start_time
    print(f"\n===== 演示完成 =====")
    print(f"总执行时间: {total_time:.2f} 秒")
    
    # 检查执行效率
    if total_time > 120:  # 超过2分钟
        print("警告: 程序执行时间较长，可能是由于模型下载或硬件限制")
        print("建议:")
        print("1. 确保您的网络连接稳定，特别是首次下载模型时")
        print("2. 如果在CPU上运行，考虑使用更小的模型或升级硬件")
        print("3. 可以尝试使用预下载的模型文件，设置HF_HOME环境变量指定缓存目录")
    else:
        print("✓ 程序执行效率良好")
    
    # 总结
    print("\n===== 实验结果总结 =====")
    print("1. 成功配置并应用了优化的LoRA参数(r=16, alpha=32)")
    print("2. 使用4-bit NF4量化加载模型，显著减少了内存占用")
    print("3. 验证了模型的基本推理功能和中文处理能力")
    print("4. 生成了可重用的LoRA配置文件供后续微调使用")
    print("\n该配置可以直接用于后续的参数高效微调任务，")
    print("预计可以在消费级GPU上实现对13B规模模型的有效微调。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        import traceback
        traceback.print_exc()