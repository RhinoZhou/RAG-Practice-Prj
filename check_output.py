import json
import sys

def check_jsonl_file(file_path, num_lines=3):
    """读取并检查JSONL文件的内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    print('\n--- 第{}行数据 ---'.format(i+1))
                    print('文档ID: {}'.format(data.get('doc_id', 'N/A')))
                    print('段落ID: {}'.format(data.get('segment_id', 'N/A')))
                    print('文件类型: {}'.format(data.get('file_type', 'N/A')))
                    print('原始路径: {}'.format(data.get('original_path', 'N/A')))
                    print('页码: {}'.format(data.get('page_num', 'N/A')))
                    
                    # 显示前100个字符的文本内容
                    text = data.get('text', 'N/A')
                    print('文本内容（前100字符）: {}...'.format(text[:100]))
                    
                    print('元数据: {}'.format(data.get('metadata', 'N/A')))
                except json.JSONDecodeError as e:
                    print('第{}行解析失败: {}'.format(i+1, str(e)))
                    print('原始内容: {}...'.format(line[:200]))
        print('\n成功读取了{}文件的前{}行数据'.format(file_path, num_lines))
    except Exception as e:
        print('读取文件失败: {}'.format(str(e)))

if __name__ == "__main__":
    file_path = "clean_text.jsonl"
    check_jsonl_file(file_path)