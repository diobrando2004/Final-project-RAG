import duckdb
import os
 
 
class DataManager:
    def __init__(self, db_path, metadata_dir):
        print(f"Connecting to persistent database: {db_path}")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
 
    def get_table_context(self, table_name):
        schema_df = self.conn.execute(f"PRAGMA table_info('{table_name}')").df()
        samples_df = self.conn.execute(f'SELECT * FROM "{table_name}" LIMIT 2').df()
        samples_str = samples_df.to_string(index=False)
 
        cols_desc = []
        for _, row in schema_df.iterrows():
            cols_desc.append(f"- {row['name']} ({row['type']})")
        return "\n".join(cols_desc), samples_str
 
    def close(self):
        self.conn.close()