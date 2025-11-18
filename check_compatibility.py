import time
import sys

print(f"[{time.strftime('%H:%M:%S')}] 兼容性检查开始")
print(f"[{time.strftime('%H:%M:%S')}] Python版本: {sys.version}")

# 检查Python版本号
major, minor = sys.version_info[:2]
print(f"[{time.strftime('%H:%M:%S')}] Python主版本: {major}.{minor}")

# 检查已安装的gradio版本
print(f"[{time.strftime('%H:%M:%S')}] 检查已安装的gradio...")
import subprocess

result = subprocess.run([sys.executable, "-m", "pip", "show", "gradio"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"[{time.strftime('%H:%M:%S')}] 已安装的gradio信息:")
    print(result.stdout)
else:
    print(f"[{time.strftime('%H:%M:%S')}] 未安装gradio或检查失败")

# 检查pip版本
print(f"[{time.strftime('%H:%M:%S')}] 检查pip版本...")
result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                       capture_output=True, text=True)
print(f"[{time.strftime('%H:%M:%S')}] pip版本: {result.stdout.strip()}")

# 列出所有已安装的包
print(f"[{time.strftime('%H:%M:%S')}] 列出已安装的包...")
result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                       capture_output=True, text=True)
print(f"[{time.strftime('%H:%M:%S')}] 已安装包列表: {result.stdout[:500]}...")