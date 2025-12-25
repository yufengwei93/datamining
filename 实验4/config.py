# Vector Database Configuration (using LanceDB)
MILVUS_LITE_DATA_PATH = "./vector_db.lance"  # Path to store vector database (LanceDB format)
COLLECTION_NAME = "medical_rag_lite"  # Table name in LanceDB

# Data Configuration
DATA_FILE = "./data/processed_data.json"

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = "./models/qwen2-0.5b"  # 👈 确保这里是本地路径！
EMBEDDING_DIM = 384

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3

# Generation Parameters
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
id_to_doc_map = {}