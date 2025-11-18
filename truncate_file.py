# 截断rag.py文件到2010行，删除末尾冗余代码
with open('rag.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()[:2010]

with open('rag.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"已截断文件，保留前{len(lines)}行")