import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

# 创建输出目录
OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 请求头，模拟浏览器
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Cookie": ""  # 可以添加cookie以获得更好的访问权限
}

# 东方财富网千股千评URL
eastmoney_url = "https://data.eastmoney.com/stockcomment/"

print(f"正在获取网页内容: {eastmoney_url}")

# 获取网页内容
try:
    # 使用requests获取网页内容
    response = requests.get(eastmoney_url, headers=headers, timeout=30, verify=False)
    response.encoding = response.apparent_encoding
    
    print(f"请求状态码: {response.status_code}")
    print(f"检测到的编码: {response.encoding}")
    
    # 保存原始HTML内容
    html_file = OUTPUT_DIR / "eastmoney_raw.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"原始HTML内容已保存到: {html_file}")
    
    # 解析HTML内容，查看基本结构
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 保存网页标题
    title = soup.title.string if soup.title else "无标题"
    print(f"网页标题: {title}")
    
    # 保存所有的表格信息
    tables = soup.find_all('table')
    print(f"找到 {len(tables)} 个表格")
    
    # 保存表格信息到文件
    if tables:
        table_info = []
        for i, table in enumerate(tables):
            table_data = {
                "index": i,
                "class": table.get('class', []),
                "id": table.get('id', ""),
                "rows_count": len(table.find_all('tr')),
                "first_row": table.find('tr').get_text().strip()[:200] if table.find('tr') else ""
            }
            table_info.append(table_data)
            
        table_info_file = OUTPUT_DIR / "tables_info.json"
        with open(table_info_file, 'w', encoding='utf-8') as f:
            json.dump(table_info, f, ensure_ascii=False, indent=2)
        print(f"表格信息已保存到: {table_info_file}")
    
    # 查看网页中可能包含数据的容器
    containers = soup.find_all(['div', 'section'], class_=lambda x: x and ('data' in x.lower() or 'table' in x.lower() or 'list' in x.lower()))
    print(f"找到 {len(containers)} 个可能包含数据的容器")
    
    # 保存容器信息到文件
    if containers:
        container_info = []
        for i, container in enumerate(containers):
            container_data = {
                "index": i,
                "tag": container.name,
                "class": container.get('class', []),
                "id": container.get('id', ""),
                "content_preview": container.get_text().strip()[:200] if container.get_text() else ""
            }
            container_info.append(container_data)
            
        container_info_file = OUTPUT_DIR / "containers_info.json"
        with open(container_info_file, 'w', encoding='utf-8') as f:
            json.dump(container_info, f, ensure_ascii=False, indent=2)
        print(f"容器信息已保存到: {container_info_file}")
    
    # 查看是否有JavaScript加载的数据
    scripts = soup.find_all('script')
    print(f"找到 {len(scripts)} 个script标签")
    
    # 查找可能包含JSON数据的script标签
    script_with_data = []
    for i, script in enumerate(scripts):
        script_text = script.get_text()
        if script_text and ('var ' in script_text or 'data' in script_text.lower() or 'stock' in script_text.lower()):
            script_with_data.append({
                "index": i,
                "preview": script_text[:500]  # 只保存前500个字符
            })
    
    if script_with_data:
        script_info_file = OUTPUT_DIR / "scripts_with_data.json"
        with open(script_info_file, 'w', encoding='utf-8') as f:
            json.dump(script_with_data, f, ensure_ascii=False, indent=2)
        print(f"可能包含数据的script信息已保存到: {script_info_file}")
    
    print("\n===== 分析完成 ====")
    print("请查看test_output目录下的文件，了解网页结构")
    
except Exception as e:
    print(f"获取网页内容时出错: {str(e)}")