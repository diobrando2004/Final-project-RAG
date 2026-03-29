from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MARKDOWN_DIR             = str(ROOT_DIR / "markdown_docs")
PARENT_STORE_PATH        = str(ROOT_DIR / "parent_store")
PARENT_STORE_PATH_SQLITE = str(ROOT_DIR / "parent_store_sqlite")
QDRANT_DB_PATH           = str(ROOT_DIR / "qdrant_db")
MODELS_DIR               = str(ROOT_DIR / "models")
DOCUMENTS_DIR            = str(ROOT_DIR / "documents")

CHILD_CHUNK_SIZE    = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE     = 2000
MAX_PARENT_SIZE     = 4000
HEADERS_TO_SPLIT_ON = [
    ("#",   "H1"),
    ("##",  "H2"),
    ("###", "H3"),
]
CHILD_COLLECTION   = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

LLM_TEMPERATURE = 0.1
LLM_N_CTX       = 8192
LLM_N_THREADS   = 2
LLM_N_BATCH     = 128

CSV_DIR             = str(ROOT_DIR / "csv_data")
CSV_DB_PATH         = str(ROOT_DIR / "database" / "csv_store.db")
CSV_METADATA_DIR    = str(ROOT_DIR / "csv_metadata")
CSV_TABLE_INDEX_DIR = str(ROOT_DIR / "csv_table_index")

CSV_MULTI_TABLE_GAP = 0.2
CSV_MIN_SCORE       = 0.15
CSV_AGGREGATE_WORDS = {
    "how many", "count", "total", "sum", "average",
    "avg", "max", "min", "most", "least", "percentage", "ratio"
}
SCAN_WORDS = {"show", "display", "list", "first", "top", "sample", "preview"}