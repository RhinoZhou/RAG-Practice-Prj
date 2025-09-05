# 数据导入工具集

本目录包含一组用于数据导入和处理的工具，主要用于PDF文档解析、OCR识别、表格提取和数据同步管理。这些工具可以帮助用户高效地从各种格式的文档中提取结构化数据，并支持数据一致性校验和增量更新。

## 目录结构

```
07-data_import/
├── 01-text-cleaner_standardizer.py  # 文本清洗与标准化工具
├── 02-LangChain_text_import.py      # LangChain文本批量导入工具
├── 03-semantic_chunker.py           # 语义分块工具
├── 04-cdc_consumer.py               # Kafka CDC事件消费者工具
├── 05-card_generator.py             # 结构化数据向量化与检索工具
├── 06-watermark_manager.py          # 增量导入与幂等性管理工具
├── 07-pdf_route_selector.py         # PDF解析路由选择器
├── 08-table_extract.py              # 表格抽取工具
├── 09-ocr_pipeline.py               # OCR与版面对齐处理流水线
├── 10-web_ingest.py                 # 通用网页内容提取工具
├── 11-eastmoney_stock_comment.py    # 东方财富网千股千评数据提取工具
├── README.md                        # 项目说明文档
├── data/                            # 示例数据目录
├── ocr_output/                      # OCR处理结果输出目录
└── pdf_output/                      # PDF解析结果输出目录
```

## 工具说明

### 1. 文本清洗与标准化工具 (01-text-cleaner_standardizer.py)

该工具用于对文本进行全面的清洗和标准化处理，解决各种文本质量问题，确保数据适合RAG系统使用。

### 2. LangChain文本批量导入工具 (02-LangChain_text_import.py)

这是一个基于LangChain框架实现的文本批量导入工具，专为RAG系统设计，提供完整的文本处理和向量存储流水线，支持幂等处理、失败重试和去重功能。

**核心功能：**
- **完整流水线架构**：实现Loader → Cleaner → Splitter → Embedder → VectorStore的经典RAG流水线
- **批量处理**：支持批量导入多个文本文件，自动分批处理
- **幂等处理**：通过文件哈希值实现幂等操作，避免重复处理
- **失败重试**：内置指数退避重试机制，提高系统稳定性
- **文本清洗**：自动修复编码问题、移除多余空白字符和BOM头
- **文本分割**：将长文本智能分割成适合向量化的小块
- **向量化**：集成HuggingFace嵌入模型，支持本地缓存以提高性能
- **向量存储**：使用FAISS高效存储和检索文本向量
- **示例数据生成**：内置示例文本文件生成功能，便于测试

**主要类：**
- `TextImportPipeline`：整合文本导入、清洗、分割、向量化和存储的完整流水线

**使用示例：**
```python
# 初始化文本导入流水线
pipeline = TextImportPipeline(
    data_dir='./data',
    vector_store_path='./vector_store',
    cache_dir='./embedding_cache',
    batch_size=10,
    chunk_size=1000,
    chunk_overlap=100
)

# 创建示例文本文件（可选）
pipeline.create_sample_text_files(num_files=5)

# 批量处理文本文件
pipeline.batch_process_files()

# 处理单个文件
file_path = './data/sample.txt'
file_hash = pipeline._calculate_file_hash(file_path)
pipeline.process_single_file(file_path, file_hash)
```

### 4. Kafka CDC事件消费者工具 (04-cdc_consumer.py)

这是一个专业的Kafka CDC（变更数据捕获）事件消费者工具，专为实时数据同步场景设计，能够从Kafka消费Debezium格式的数据库变更事件，并提供完整的事件处理、幂等性保证和数据一致性管理。

**核心功能：**
- **双模式支持**：提供生产模式和模拟模式，适应不同的开发和测试需求
- **事件信息提取**：智能解析Debezium格式的变更事件，提取表名、操作类型、变更数据等关键信息
- **重复事件检查**：实现基于重放窗口的幂等性处理机制，避免重复处理相同事件
- **死信队列**：自动将处理失败的事件发送到死信队列，确保数据完整性
- **Watermark更新**：跟踪和管理事件处理进度，支持断点续传
- **检查点提交**：定期提交消费偏移量，确保消息处理的可靠性
- **旧事件清理**：自动清理过期的事件记录，优化存储空间
- **性能监控**：实时统计和输出事件处理性能指标，便于系统监控
- **SQLite集成**：内置轻量级数据库支持，用于事件缓存和元数据存储

**主要类：**
- `MockKafkaMessage`：模拟Kafka消息的数据类，用于测试环境
- `CDCEventProcessor`：核心事件处理器，实现事件消费、处理和存储的完整流程

**使用示例：**
```python
# 初始化CDC事件处理器（生产模式）
processor = CDCEventProcessor(
    bootstrap_servers='localhost:9092',
    topic='dbserver1.inventory.customers',
    db_path='./cdc_events.db',
    replay_window_hours=24
)

# 初始化CDC事件处理器（模拟模式）
processor = CDCEventProcessor(
    bootstrap_servers='localhost:9092',
    topic='dbserver1.inventory.customers',
    mock_mode=True,
    mock_table_name='customers',
    mock_record_count=1000
)

# 启动事件处理
processor.run()

# 单独处理单个事件
message = MockKafkaMessage(value=json.dumps(event_data))
processor.process_event(message)

# 检查处理状态
print(f"已处理事件数: {processor.total_events_processed}")
print(f"TPS: {processor.tps:.2f}")
print(f"成功率: {processor.success_rate:.2%}")
```

### 5. 结构化数据向量化与检索工具 (05-card_generator.py)

这是一个专业的结构化数据向量化与检索工具，专为RAG系统中的实体卡片和指标卡片生成而设计，能够处理行记录或多表join结果，并提供完整的文本嵌入和向量检索功能。

**核心功能：**
- **实体卡片生成**：支持生成产品、客户等实体类型的结构化卡片
- **指标卡片生成**：支持生成销售、库存等指标类型的结构化卡片
- **多语言支持**：内置中文和英文模板，可根据需求扩展更多语言
- **单位标准化**：自动进行货币、数量和百分比的单位转换和标准化
- **空值处理**：提供智能的字段空值回退机制，确保卡片内容完整性
- **模拟嵌入**：内置模拟嵌入模型，无需API密钥也可运行和测试
- **向量存储**：集成FAISS向量存储，支持高效的相似卡片检索
- **元数据管理**：为每个生成的卡片添加丰富的元数据，包括有效期、处理时间等
- **过滤检索**：支持基于元数据的过滤检索功能
- **缓存机制**：提供向量存储的保存和加载功能，优化性能

**主要类：**
- `CardGenerator`：核心卡片生成器，负责卡片创建、向量化和检索
- `UnitConverter`：单位转换器，处理货币、数量和百分比的单位转换
- `MockEmbeddings`：模拟嵌入模型，用于在无API密钥环境下进行演示

**使用示例：**
```python
# 初始化卡片生成器（使用模拟嵌入）
card_gen = CardGenerator(use_mock_embeddings=True)

# 生成产品实体卡片
product_data = {
    "product_id": "P12345",
    "product_name": "高性能笔记本电脑",
    "category": "电子产品",
    "price": 6999.99,
    "stock": 150,
    "description": "配备最新处理器和高性能显卡，适合游戏和专业设计工作",
    "launch_date": "2023-06-15"
}
product_card = card_gen.generate_card(product_data, "entity", "product")

# 生成销售指标卡片
sales_data = {
    "date": "2023-07-01",
    "product_id": "P12345",
    "sales_amount": 125000.75,
    "sales_volume": 18,
    "avg_order_value": 6944.48,
    "yoy_growth": 0.235,
    "mom_growth": 0.087
}
sales_card = card_gen.generate_card(sales_data, "metric", "sales")

# 生成英文卡片
en_product_card = card_gen.generate_card(product_data, "entity", "product", language="en")

# 创建向量存储并添加文档
docs = [
    card_gen.create_document(product_card),
    card_gen.create_document(sales_card),
    card_gen.create_document(en_product_card)
]
card_gen.init_vector_store(docs)

# 搜索相似卡片
results = card_gen.search_similar_cards("高性能电脑", k=2)

# 带过滤条件的搜索
filtered_results = card_gen.search_similar_cards("电子产品", 
                                                filter={"language": "zh", "subtype": "product"})

# 保存向量存储
store_path = card_gen.save_vector_store("demo_vector_store")
```

### 3. 语义分块工具 (03-semantic_chunker.py)

这是一个专为RAG系统设计的语义分块工具，支持多种分块策略，可以根据文本内容的语义特征进行智能分块，提高检索和生成的质量。

**核心功能：**
- **语义分块**：基于句子间的语义相似度进行智能分块，保持语义连贯性
- **句段分块**：基于自然句子边界进行分块，保留完整语义单元
- **窗口化分块**：基于固定窗口大小进行分块，适合结构化数据
- **多模式支持**：通过枚举类型统一管理三种分块模式
- **相似度计算**：使用HuggingFace嵌入模型计算句子间的语义相似度
- **自适应阈值**：基于统计学方法自动确定分块阈值
- **元数据生成**：为每个分块生成丰富的元数据，包括偏移量、锚点、长度等信息
- **分块可视化**：提供详细的分块过程和结果输出，便于调试和评估

**主要类：**
- `ChunkingMode`：分块模式枚举类，定义三种分块策略
- `SemanticChunker`：语义分块器，实现三种分块策略的核心逻辑

**使用示例：**
```python
# 初始化语义分块器（语义分块模式）
semantic_chunker = SemanticChunker(
    mode="semantic",
    window_size=3,
    embedding_model="all-MiniLM-L6-v2"
)

# 初始化句段分块器
sentence_chunker = SemanticChunker(
    mode="sentence",
    chunk_size=1000,
    chunk_overlap=100
)

# 初始化窗口化分块器
window_chunker = SemanticChunker(
    mode="window",
    chunk_size=1000,
    chunk_overlap=100
)

# 执行分块
text = "你的输入文本..."
chunks = semantic_chunker.chunk(text)

# 访问分块结果
for chunk in chunks:
    print(f"块ID: {chunk['chunk_id']}")
    print(f"内容: {chunk['content']}")
    print(f"偏移量: {chunk['offset']['start']}-{chunk['offset']['end']}")
    print(f"锚点: 开始='{' '.join(chunk['anchor']['start_words'])}', 结束='{' '.join(chunk['anchor']['end_words'])}'")
```

**核心功能：**
- **混合编码修复**：使用ftfy库修复UTF-8/BGK混用产生的乱码
- **BOM头处理**：自动检测并移除UTF-8 BOM头残留
- **断行错误修复**：基于自然语言规则智能修复文本断行问题
- **干扰信息过滤**：使用正则表达式过滤页眉、页脚和引用干扰
- **脚注引用修复**：识别脚注并调整其位置，确保文本结构合理
- **表格ASCII乱码处理**：修复表格ASCII乱码，保留并优化表格结构

**主要类：**
- `TextCleanerStandardizer`：负责文本清洗与标准化处理

**使用示例：**
```python
# 初始化文本清洗器
cleaner = TextCleanerStandardizer()

# 读取文本文件
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 执行清洗流程
cleaned_text = cleaner.fix_mixed_encoding(text)
cleaned_text = cleaner.remove_bom(cleaned_text)
cleaned_text = cleaner.fix_line_breaks(cleaned_text)
cleaned_text = cleaner.filter_header_footer(cleaned_text)
cleaned_text = cleaner.fix_footnote_position(cleaned_text)
cleaned_text = cleaner.fix_table_ascii(cleaned_text)

# 保存清洗后的文本
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)
```

### 2. 水印管理工具 (06-watermark_manager.py)

这是一个专业的数据同步和一致性管理工具，主要用于数据导入过程中的状态跟踪、检查点管理和数据对账。

**核心功能：**
- **Watermark管理**：追踪数据源变更状态，支持自定义水印类型和持久化存储
- **检查点机制**：提供数据处理过程中的检查点创建、回滚和恢复功能
- **快照+增量策略**：支持创建数据源快照和应用增量更新，确保数据一致性
- **回放窗口**：允许在指定时间范围内回放数据变更历史
- **数据对账**：提供计数对账和主键对账功能，验证源表和目标表的数据一致性

**主要类：**
- `WatermarkManager`：管理数据同步的水印状态
- `DataReconciliation`：执行数据对账操作
- `SnapshotIncrementalManager`：管理快照和增量数据处理

**使用示例：**
```python
# 初始化水印管理器
watermark_manager = WatermarkManager(watermark_dir='./watermarks')

# 设置和获取水印
watermark_manager.set_watermark('source_db', 'table1', {'last_id': 1000, 'last_updated': '2023-01-01'})
current_watermark = watermark_manager.get_watermark('source_db', 'table1')

# 创建检查点
checkpoint_id = watermark_manager.create_checkpoint('source_db', 'table1')

# 回滚到指定检查点
watermark_manager.rollback_to_checkpoint('source_db', 'table1', checkpoint_id)
```

### 3. PDF智能解析路由选择器 (07-pdf_route_selector.py)

该工具能够智能分析PDF文档特征，并根据文本层质量和版面复杂度自动选择最优的解析路线，确保文档内容的最佳提取效果。

**核心功能：**
- **文本层检测与质量评估**：识别PDF中的文本层并评估其完整性和质量
- **版面复杂度分析**：分析文档版面结构，评估其复杂程度
- **智能路由选择**：基于分析结果自动选择最合适的解析路线（pdfplumber/版面重建/OCR）
- **OCR集成**：内置Tesseract OCR支持，自动检测和安装所需依赖
- **多平台支持**：兼容Windows和macOS系统的OCR配置

**主要类：**
- `PDFRouteSelector`：负责PDF特征分析和解析路线选择

**路由选项：**
- `ROUTE_PDFPLUMBER`：适用于有高质量文本层的PDF
- `ROUTE_LAYOUT`：适用于版面相对简单但文本层质量不佳的PDF
- `ROUTE_OCR`：适用于扫描件或无文本层的PDF

**使用示例：**
```python
# 初始化PDF路由选择器
selector = PDFRouteSelector()

# 分析PDF并选择最优解析路线
route_decision = selector.select_route('example.pdf')

# 执行选定的解析路线
result = selector.execute_route('example.pdf', route_decision['route_type'])

# 查看解析报告
selector.print_summary()
```

### 4. PDF表格抽取工具 (08-table_extract.py)

专门用于从PDF文档中精确提取表格数据，并支持多种格式导出的工具。

**核心功能：**
- **智能表格识别**：自动识别PDF中的表格结构
- **列对齐率校验**：评估表格结构完整性，确保提取数据的准确性
- **合并单元格处理**：智能识别和处理合并单元格，还原表格原始结构
- **多格式导出**：支持将提取的表格导出为JSON、Markdown和Excel格式
- **自定义配置**：允许用户根据需求调整表格提取参数

**主要类：**
- `TableExtractor`：负责PDF表格的识别和提取

**支持的导出格式：**
- JSON：包含详细的表格结构和元数据
- Markdown：适合文档编辑和阅读
- Excel：便于数据进一步处理和分析

**使用示例：**
```python
# 初始化表格提取器
extractor = TableExtractor(output_dir='./table_output')

# 从PDF中提取表格
pdf_path = 'example_with_tables.pdf'
tables = extractor.extract_tables(pdf_path)

# 导出为不同格式
extractor.export_tables_to_json(pdf_path, tables)
extractor.export_tables_to_markdown(pdf_path, tables)
extractor.export_tables_to_excel(pdf_path, tables)
```

### 5. OCR与版面对齐处理流水线 (09-ocr_pipeline.py)

提供完整的OCR处理流程，包括图像预处理、文本识别、段落合并和版面对齐，适用于需要从图像或扫描PDF中提取结构化文本的场景。

**核心功能：**
- **图像预处理**：支持去噪、倾斜检测与校正、二值化等预处理操作
- **文本识别**：集成PaddleOCR引擎，支持中文文本识别
- **段落合并**：智能合并相邻文本块，还原文档段落结构
- **版面对齐**：识别标题、正文等版面元素，生成结构化版面信息
- **多格式输出**：保存完整结果、纯文本和版面信息

**主要类：**
- `ImagePreprocessor`：处理图像预处理
- `OCRProcessor`：执行OCR文本识别
- `LayoutAligner`：负责段落合并和版面对齐
- `OCRPipeline`：整合以上组件的完整处理流水线

**处理流程：**
1. 图像预处理（加载、去噪、检测并校正倾斜、二值化）
2. OCR文本识别（使用PaddleOCR）
3. 评估识别准确率
4. 段落合并
5. 版面对齐
6. 结果保存

**使用示例：**
```python
# 初始化OCR处理流水线
pipeline = OCRPipeline(config={'output_dir': './ocr_output'})

# 处理单个文件
result = pipeline.process_file('example.pdf')

# 保存处理结果
saved_paths = pipeline.save_results(result)
```

### 6. 通用网页内容提取工具 (10-web_ingest.py)

提供一个高效、灵活的网页内容提取流水线，能够从各种新闻网站、博客、专栏等网页中提取结构化内容，包括标题、正文、作者、发布时间、分类、标签等元数据。

**核心功能：**
- **双模式爬取**：支持同步和异步两种爬取模式
- **缓存机制**：内置缓存系统，避免重复爬取已处理的网页
- **请求节流**：支持自定义请求间隔，防止对目标网站造成过大压力
- **失败重试**：实现智能重试机制，提高爬取成功率
- **动态渲染支持**：可选的Selenium支持，用于处理动态渲染和复杂反爬网站
- **编码处理**：智能检测和处理各种编码问题
- **多格式输出**：支持将提取的内容保存为文本和JSON格式
- **完善日志**：详细的日志记录，便于调试和监控

**主要类：**
- `WebIngestPipeline`：整合网页爬取、内容提取和处理的完整流水线

**使用示例：**
```python
# 初始化网页内容提取流水线
pipeline = WebIngestPipeline(config={'use_async': True, 'enable_selenium': False})

# 同步处理单个URL
result = pipeline.process_url('https://example.com/article')

# 异步处理单个URL
result = await pipeline.process_url_async('https://example.com/article')

# 批量处理多个URL
results = await pipeline.batch_process_async(['url1', 'url2', 'url3'])

# 保存处理结果
pipeline.save_results(result, output_dir='./web_output')
```

### 7. 东方财富网千股千评数据提取工具 (11-eastmoney_stock_comment.py)

该工具专门用于从东方财富网爬取和提取股票评论数据，支持同步和异步两种爬取模式，提供缓存机制、请求节流、错误重试等功能，并可将爬取的数据保存为多种格式。

**核心功能：**
- **API数据获取**：从东方财富网API获取千股千评数据
- **双模式爬取**：支持同步和异步两种爬取方式
- **缓存机制**：实现请求缓存，避免重复请求相同数据
- **请求节流**：内置请求频率控制，防止被网站反爬
- **错误重试**：实现多层错误处理和重试机制，提高数据获取成功率
- **Selenium支持**：可选支持Selenium作为备用爬取方式
- **多格式导出**：支持将爬取的数据保存为文本、JSON和CSV格式
- **批量处理**：支持批量处理多个URL和多页数据

**主要类：**
- `EastmoneyStockCommentPipeline`：整合东方财富网股票评论数据爬取和处理的完整流水线

**使用示例：**
```python
# 初始化东方财富网股票评论提取流水线
pipeline = EastmoneyStockCommentPipeline(config={'use_async': True, 'page_count': 5})

# 爬取股票评论数据
stock_code = '600000'
result = pipeline.fetch_stock_comment(stock_code)

# 批量爬取多个股票的数据
results = pipeline.batch_fetch_stock_comments(['600000', '600036', '601318'])

# 保存爬取结果
pipeline.save_results(result, output_dir='./eastmoney_output', format='json')
```

## 依赖项

以下是工具集主要依赖的第三方库：

| 工具 | 主要依赖 | 用途 |
|------|---------|------|
| 文本清洗与标准化工具 | ftfy, re, pandas | 编码修复和文本处理 |
| LangChain文本批量导入工具 | langchain, faiss-cpu, huggingface-hub, ftfy, tqdm | 文本处理和向量存储 |
| 语义分块工具 | langchain, langchain_community, numpy, huggingface-hub | 语义分析和文本分块 |
| Kafka CDC事件消费者工具 | kafka-python, sqlite3 | Kafka消息消费和事件处理 |
| 结构化数据向量化与检索工具 | langchain, faiss-cpu | 实体/指标卡片生成与向量检索 |
| 水印管理工具 | pandas, json | 数据处理和持久化 |
| PDF路由选择器 | PyMuPDF, pdfplumber, pytesseract, OpenCV | PDF解析和OCR |
| 表格抽取工具 | pdfplumber, pandas, openpyxl | 表格识别和数据处理 |
| OCR流水线 | OpenCV, PaddleOCR | 图像处理和OCR识别 |
| 通用网页内容提取工具 | requests, aiohttp, trafilatura | 网页爬取和内容提取 |
| 东方财富网数据提取工具 | requests, aiohttp, beautifulsoup4 | 金融数据爬取和处理 |

## 安装指南

1. 确保已安装Python 3.8-3.11（注意：Python 3.13可能与PyTorch不兼容）

2. 安装必要的依赖包：
   ```bash
   pip install pandas pdfplumber PyMuPDF pytesseract opencv-python paddleocr openpyxl ftfy aiohttp beautifulsoup4 trafilatura langchain langchain_community numpy huggingface-hub tqdm kafka-python faiss-cpu
   ```

3. 对于OCR功能，还需要安装Tesseract OCR引擎：
   - **Windows**：从[GitHub](https://github.com/UB-Mannheim/tesseract/wiki)下载安装，并确保添加到系统PATH
   - **macOS**：使用Homebrew安装：`brew install tesseract tesseract-lang`

4. 对于中文识别，确保安装了中文语言包

## 使用指南

### 命令行使用

每个工具都提供了命令行接口，可以直接运行：

```bash
# 运行OCR处理流水线
python 09-ocr_pipeline.py --input_file data/example.pdf --output_dir ./ocr_output

# 运行PDF路由选择器
python 07-pdf_route_selector.py --input_file data/example.pdf

# 运行表格提取器
python 08-table_extract.py --pdf_path data/example_with_tables.pdf

# 运行网页内容提取工具
python 10-web_ingest.py

# 运行东方财富网数据提取工具
python 11-eastmoney_stock_comment.py
```

### 作为模块集成

工具也可以作为Python模块集成到其他项目中：

```python
from 06_watermark_manager import WatermarkManager
from 07_pdf_route_selector import PDFRouteSelector
from 08_table_extract import TableExtractor
from 09_ocr_pipeline import OCRPipeline

# 在您的代码中使用这些类
```

## 注意事项

1. **版本兼容性**：
   - PaddleOCR可能需要特定版本的Python（3.8-3.11）
   - 不同操作系统的OCR配置可能有所不同

2. **性能考虑**：
   - OCR处理可能消耗较多系统资源，尤其是处理大文件时
   - 对于大批量文档处理，建议分批处理以提高效率

3. **依赖管理**：
   - Tesseract OCR的正确安装和配置对OCR功能至关重要
   - 确保所有依赖项版本兼容

4. **数据存储**：
   - 处理结果将保存在指定的输出目录中，请确保有足够的磁盘空间
   - 水印和检查点数据建议定期备份

## 示例

示例数据位于`data/`目录中，您可以使用这些示例来测试各个工具的功能。

## 更新日志

- **v1.0**：初始版本，包含所有基本功能
- **v1.1**：优化了OCR处理性能，修复了部分兼容性问题
- **v1.2**：增强了表格提取的准确性，支持更多复杂表格结构