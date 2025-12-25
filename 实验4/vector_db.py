# vector_db.py
import lancedb
import numpy as np
import os
import streamlit as st
from config import MILVUS_LITE_DATA_PATH, EMBEDDING_DIM, TOP_K, id_to_doc_map  # 👈 补充导入

# LanceDB 使用目录（config.py 已改为 .lance，无需 replace）
DB_DIR = MILVUS_LITE_DATA_PATH  # ✅ 直接使用，因为 config.py 已是 "./vector_db.lance"

@st.cache_resource
def get_milvus_client():
    try:
        abs_db_dir = os.path.abspath(MILVUS_LITE_DATA_PATH)
        os.makedirs(abs_db_dir, exist_ok=True)
        client = lancedb.connect(abs_db_dir)
        st.write(f"✅ LanceDB client initialized at: {abs_db_dir}")
        return client
    except Exception as e:
        st.error(f"❌ LanceDB 初始化失败: {type(e).__name__}: {e}")
        return None

@st.cache_resource
def setup_milvus_collection(_client):
    table_name = "medical_rag"
    try:
        _client.open_table(table_name)
        st.write("Found existing LanceDB table.")
        return True
    except (ValueError, FileNotFoundError):  # ✅ 捕获两种可能的异常
        import pyarrow as pa
        from config import EMBEDDING_DIM
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("content_preview", pa.string())
        ])
        _client.create_table(table_name, schema=schema)
        st.write("Created LanceDB table.")
        return True

def index_data_if_needed(client, data, embedding_model):
    global id_to_doc_map
    table = client.open_table("medical_rag")
    
    # 检查是否已索引（简化判断）
    count = table.count_rows()
    if count > 0:
        st.write(f"Data already indexed ({count} rows).")
        return True

    # 准备数据
    docs_for_embedding = []
    temp_id_map = {}
    data_to_insert = []

    for i, doc in enumerate(data):
        title = doc.get('title', '') or ""
        abstract = doc.get('abstract', '') or ""
        content = f"Title: {title}\nAbstract: {abstract}".strip()
        if not content:
            continue
        docs_for_embedding.append(content)
        # 👇 用 i 作为 ID（整数）
        temp_id_map[i] = {'title': title, 'abstract': abstract, 'content': content}
        data_to_insert.append({
            "id": i,
            "vector": [],  # 先占位
            "content_preview": content[:500]
        })

    if not docs_for_embedding:
        st.error("No valid documents to index.")
        return False

    # 生成嵌入
    st.write(f"Embedding {len(docs_for_embedding)} documents...")
    embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
    for i, emb in enumerate(embeddings):
        data_to_insert[i]["vector"] = emb.tolist()

    # 插入数据
    st.write("Inserting into LanceDB...")
    table.add(data_to_insert)
    id_to_doc_map.update(temp_id_map)
    st.success(f"Indexed {len(data_to_insert)} documents.")
    return True

def search_similar_documents(client, query, embedding_model):
    table = client.open_table("medical_rag")
    query_vec = embedding_model.encode([query])[0].tolist()
    # LanceDB 默认返回相似度（越高越相似），我们转为距离（越低越相似）
    results = table.search(query_vec).limit(TOP_K).to_list()  # 👈 用 to_list() 更稳定
    ids = [r["id"] for r in results]
    distances = [1 - r["_distance"] for r in results]  # 相似度 → 距离
    return ids, distances