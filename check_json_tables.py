import json

# 读取JSON文件
with open('results/pdf_layout_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# 搜索table类型的块
table_chunks = [chunk for chunk in chunks if chunk.get('type') == 'table']

print(f"在JSON文件中找到{len(table_chunks)}个table类型的块")
for i, table_chunk in enumerate(table_chunks):
    print(f"  table块{i+1}:")
    print(f"    id: {table_chunk.get('chunk_id')}")
    print(f"    页码: {table_chunk.get('page')}")
    print(f"    文本长度: {len(table_chunk.get('text', ''))}")
    print(f"    文本预览: {table_chunk.get('text', '')[:100]}...")
    print(f"    bbox: {table_chunk.get('bbox')}")

# 检查块ID 365
chunk_365 = next((chunk for chunk in chunks if chunk.get('chunk_id') == 365), None)
if chunk_365:
    print(f"\n块ID 365的类型: {chunk_365.get('type')}")
    print(f"  文本长度: {len(chunk_365.get('text', ''))}")
    print(f"  页码: {chunk_365.get('page')}")
else:
    print("\n未找到块ID 365")