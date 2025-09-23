import numpy as np

# 加载embeddings_256.npy文件
try:
    embeddings = np.load('d:/rag-project/05-rag-practice/10-VectorDB_Index/data/embeddings_256.npy')
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Embedding dimension: {embeddings.shape[1]}')
    print(f'Number of embeddings: {embeddings.shape[0]}')
    print(f'First embedding sample: {embeddings[0][:5]}...')
except Exception as e:
    print(f'Error loading embeddings: {e}')