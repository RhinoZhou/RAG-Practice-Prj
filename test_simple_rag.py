import os
import sys
import time
import traceback

# 设置日志函数
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# 测试结果统计
results = {
    "tests": [],
    "passed": 0,
    "failed": 0
}

def run_test(test_name, test_func):
    """运行测试并记录结果"""
    log(f"\n{test_name}")
    try:
        result = test_func()
        log(f"✓ {test_name} 通过")
        results["tests"].append((test_name, "passed", result))
        results["passed"] += 1
        return True, result
    except Exception as e:
        log(f"✗ {test_name} 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["tests"].append((test_name, "failed", str(e)))
        results["failed"] += 1
        return False, str(e)

# 测试1: 基本库导入
def test_basic_libraries():
    imports = []
    
    # 测试torch
    try:
        import torch
        imports.append("torch")
    except Exception as e:
        log(f"  - torch导入失败: {type(e).__name__}: {e}")
    
    # 测试faiss
    try:
        import faiss
        imports.append("faiss")
    except Exception as e:
        log(f"  - faiss导入失败: {type(e).__name__}: {e}")
    
    # 测试numpy
    try:
        import numpy as np
        imports.append("numpy")
    except Exception as e:
        log(f"  - numpy导入失败: {type(e).__name__}: {e}")
    
    return f"成功导入: {', '.join(imports)}"

run_test("测试1: 基本库导入", test_basic_libraries)

# 测试2: llama_index导入
def test_llama_index():
    try:
        from llama_index.core.node_parser import SentenceSplitter
        return "成功导入SentenceSplitter"
    except Exception as e:
        raise Exception(f"llama_index导入失败: {type(e).__name__}: {e}")

run_test("测试2: llama_index导入", test_llama_index)

# 测试3: 配置文件导入
def test_config_import():
    try:
        from config import AppConfig
        result = f"成功导入AppConfig, KB_BASE_DIR={AppConfig.knowledge_base_root}"
        
        # 创建必要的目录
        KB_BASE_DIR = AppConfig.knowledge_base_root
        os.makedirs(KB_BASE_DIR, exist_ok=True)
        DEFAULT_KB = AppConfig.default_knowledge_base
        DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
        os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
        OUTPUT_DIR = AppConfig.temp_output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        return result + ", 成功创建必要目录"
    except Exception as e:
        raise Exception(f"配置文件导入或目录创建失败: {type(e).__name__}: {e}")

run_test("测试3: 配置文件导入", test_config_import)

# 测试4: Gradio导入和最小界面

def test_gradio_import():
    try:
        import gradio as gr
        return f"成功导入gradio (版本: {gr.__version__})"
    except Exception as e:
        raise Exception(f"Gradio导入失败: {type(e).__name__}: {e}")

run_test("测试4.1: Gradio导入", test_gradio_import)

def test_gradio_theme():
    try:
        import gradio as gr
        custom_theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="gray",
            text_size="lg",
            spacing_size="md",
            radius_size="md"
        )
        return "成功创建Gradio主题"
    except Exception as e:
        raise Exception(f"创建Gradio主题失败: {type(e).__name__}: {e}")

run_test("测试4.2: 创建Gradio主题", test_gradio_theme)

def test_gradio_interface():
    try:
        import gradio as gr
        custom_theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="gray",
            text_size="lg",
            spacing_size="md",
            radius_size="md"
        )
        
        # 创建最小的Gradio界面
        with gr.Blocks(title="测试界面", theme=custom_theme) as demo:
            gr.Markdown("# 医疗知识问答系统")
            gr.Textbox(placeholder="请输入您的问题...")
            gr.Button("提交")
        
        return "成功创建Gradio界面"
    except Exception as e:
        raise Exception(f"创建Gradio界面失败: {type(e).__name__}: {e}")

run_test("测试4.3: 创建Gradio界面", test_gradio_interface)

# 测试5: 测试rag模块的基本功能
def test_rag_basic():
    try:
        # 设置环境变量以跳过自动安装依赖
        os.environ['RAG_DEPENDENCIES_INSTALLED'] = '1'
        
        # 导入rag模块
        import rag
        
        # 测试基本功能
        knowledge_bases = rag.get_knowledge_bases()
        result = f"成功导入rag模块, 知识库列表: {knowledge_bases}"
        
        # 测试回答功能（如果存在）
        if hasattr(rag, 'answer_question'):
            test_question = "什么是糖尿病？"
            answer = rag.answer_question(test_question)
            result += f", 测试回答: {answer[:50]}..."
        
        return result
    except Exception as e:
        raise Exception(f"rag模块测试失败: {type(e).__name__}: {e}")

run_test("测试5: rag模块基本功能", test_rag_basic)

# 总结测试结果
log("\n" + "="*60)
log("测试结果总结")
log("="*60)
log(f"总测试数: {len(results['tests'])}")
log(f"通过: {results['passed']}")
log(f"失败: {results['failed']}")
log("="*60)

for test_name, status, result in results["tests"]:
    status_icon = "✓" if status == "passed" else "✗"
    log(f"{status_icon} {test_name}: {result}")

log("="*60)
log("测试完成！")