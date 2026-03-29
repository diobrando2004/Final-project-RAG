import duckdb
import os
import threading


class DataManager:
    def __init__(self, db_path, metadata_dir):
        print(f"Connecting to persistent database: {db_path}")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = duckdb.connect(db_path)

    def execute(self, query, params=None):

        with self._lock:
            if params is not None:
                return self.conn.execute(query, params)
            return self.conn.execute(query)

    def safe_read(self, query, params=None):

        with self._lock:
            if params is not None:
                return self.conn.execute(query, params).fetchall()
            return self.conn.execute(query).fetchall()

    def safe_df(self, query, params=None):
        with self._lock:
            if params is not None:
                return self.conn.execute(query, params).df()
            return self.conn.execute(query).df()

    def get_table_context(self, table_name):
        with self._lock:
            schema_df = self.conn.execute(f"PRAGMA table_info('{table_name}')").df()
            samples_df = self.conn.execute(f'SELECT * FROM "{table_name}" LIMIT 2').df()
        samples_str = samples_df.to_string(index=False)
        cols_desc = []
        for _, row in schema_df.iterrows():
            cols_desc.append(f"- {row['name']} ({row['type']})")
        return "\n".join(cols_desc), samples_str

    def close(self):
        with self._lock:
            self.conn.close()