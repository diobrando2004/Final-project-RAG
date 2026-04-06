import os
import duckdb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
def build_smart_index(db_path, index_root="smart_index_dir"):
    conn = duckdb.connect(db_path)
    tables = conn.execute("SHOW TABLES").df()['name'].tolist()
    
    for table in tables:
        persist_dir = os.path.join(index_root, table)
        if os.path.exists(persist_dir): continue

        print(f"--- Fast Indexing Table: {table} ---")
        
        # 1. Find string columns
        cols = conn.execute(f"PRAGMA table_info('{table}')").df()
        string_cols = cols[cols['type'] == 'VARCHAR']['name'].tolist()
        
        unique_hints = []
        for col in string_cols:
            # Get unique values, but cap it so we don't index unique IDs/Names
            distinct_vals = conn.execute(f"SELECT DISTINCT \"{col}\" FROM \"{table}\" LIMIT 1000").df()[col].dropna().tolist()
            
            for val in distinct_vals:
                # We store a "Hint" that links the value back to the column
                unique_hints.append(TextNode(
                    text=f"Value '{val}' belongs to column '{col}' in table '{table}'",
                    metadata={"column": col, "value": val, "table": table}
                ))

        if unique_hints:
            print(f"Indexing {len(unique_hints)} unique value hints...")
            index = VectorStoreIndex(unique_hints)
            index.storage_context.persist(persist_dir=persist_dir)

# Run the fast version
build_smart_index("my_persistent_db.db")