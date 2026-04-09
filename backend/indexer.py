import os
import json
import numpy as np

SKIP_TABLES = {"system_metadata"}

class SemanticIndexer:
    def __init__(self, embedder, table_index_dir="table_index_dir"):
        self.embedder = embedder
        self.table_index_dir = table_index_dir

    def build_custom_value_index(self, db_conn):
        os.makedirs(self.table_index_dir, exist_ok=True)
        tables = [
            t[0] for t in db_conn.execute("SHOW TABLES").fetchall()
            if t[0] not in SKIP_TABLES
        ]

        for table in tables:
            index_path = os.path.join(self.table_index_dir, f"{table}_embs.npy")
            meta_path  = os.path.join(self.table_index_dir, f"{table}_meta.json")

            if os.path.exists(index_path):
                continue

            print(f"Indexing unique values for table: {table}")
            cols = db_conn.execute(f"PRAGMA table_info('{table}')").df()
            str_cols = cols[
                cols['type'].str.contains('VARCHAR|TEXT', case=False, na=False)
            ]['name'].tolist()

            table_data = []
            for col in str_cols:
                unique_vals = db_conn.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" '
                    f'WHERE "{col}" IS NOT NULL LIMIT 150'
                ).df()[col].tolist()
                for val in unique_vals:
                    table_data.append({"column": col, "value": str(val)})

            if table_data:
                texts = [f"{d['column']}: {d['value']}" for d in table_data]
                embeddings = np.array(list(self.embedder.embed(texts)))
                np.save(index_path, embeddings)
                with open(meta_path, "w") as f:
                    json.dump(table_data, f)

    def get_custom_hints(self, user_query, table_name):
        emb_path  = os.path.join(self.table_index_dir, f"{table_name}_embs.npy")
        meta_path = os.path.join(self.table_index_dir, f"{table_name}_meta.json")

        if not os.path.exists(emb_path):
            return ""

        embeddings = np.load(emb_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        query_emb = np.array(list(self.embedder.embed([user_query])))
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        scores = (query_norm @ emb_norm.T)[0]
        top_indices = np.argsort(-scores)[:3]

        hints = "\nDATABASE HINTS (Matching values found):\n"
        found = False
        for idx in top_indices:
            if scores[idx] > 0.4:
                item = metadata[idx]
                hints += f"- Column '{item['column']}' has value '{item['value']}'\n"
                found = True
        return hints if found else ""