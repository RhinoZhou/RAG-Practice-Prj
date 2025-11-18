import time

try:
    print("准备导入gradio...")
    start_time = time.time()
    import gradio as gr
    end_time = time.time()
    print(f"✓ 成功导入gradio (版本: {gr.__version__})，耗时: {end_time - start_time:.2f}秒")
    print("测试gradio组件...")
    # 测试基本组件创建
    with gr.Blocks() as demo:
        gr.Textbox(label="测试输入")
        gr.Button("测试按钮")
    print("✓ gradio组件创建成功")
    print("Gradio导入和使用测试通过！")
except Exception as e:
    print(f"✗ 导入gradio失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()