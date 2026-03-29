import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)
class ParentStore:
    def __init__(self, db_dir: str | Path = config.PARENT_STORE_PATH_SQLITE, db_name: str = "parent_store", cache_size: int = 100):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.db_dir / f"{db_name}.db"
        self.cache_size = cache_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")  # Faster writes
            conn.execute("PRAGMA synchronous = NORMAL;") # Balance safety/speed
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                source_name TEXT PRIMARY KEY,
                summary TEXT,
                metadata TEXT
            )
        """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_id ON parents(parent_id)")
    def save_document_summary(self, source_name: str, summary: str, metadata: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO documents (source_name, summary, metadata) VALUES (?, ?, ?)",
                    (source_name, summary, json.dumps(metadata))
                )
        except Exception as e:
            logger.error(f"Failed to save document summary for {source_name}: {e}")
    def _update_cache(self, parent_id: str, data: Dict[str, Any]):
        if parent_id in self._cache:
            del self._cache[parent_id]
        self._cache[parent_id] = data
        
        if len(self._cache) > self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def save(self, parent_id:str, content: str, metadata: dict)->None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO parents (parent_id, content, metadata) VALUES (?, ?, ?)",
                    (parent_id, content, json.dumps(metadata))
                )
            self._update_cache(parent_id, {"content": content, "metadata": metadata})
        except Exception as e:
            logger.error(f"Failed to save {parent_id}: {e}")
    def save_multiple(self, parents: list)->None:
        if not parents:
            return
        try:
            db_rows = [
                (parent_id, doc.page_content, json.dumps(doc.metadata))
                for parent_id, doc in parents
            ]
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO parents (parent_id, content, metadata) VALUES (?, ?, ?)",
                    db_rows
                )
            for parent_id, doc in parents:
                self._update_cache(parent_id, {
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                })
        except Exception as e:
            logger.error(f"Failed to batch save documents: {e}")
            raise
    
    def load_content(self, parent_id:str)-> dict:
        if parent_id in self._cache:
            cached = self._cache[parent_id]
            return {
                "content": cached["content"],
                "parent_id": parent_id,
                "metadata": cached["metadata"]
            }
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content, metadata FROM parents WHERE parent_id = ?", (parent_id,)
            ).fetchone()
        if row:
            content, meta_str = row
            metadata = json.loads(meta_str)
            self._update_cache(parent_id, {"content": content, "metadata": metadata})
            return {
                "content": content,
                "parent_id": parent_id,
                "metadata": metadata
            }
        return None
    
    def load_content_many(self, parent_ids:list[str]) ->list[Dict]:
        unique_ids = list(set(parent_ids))
        to_fetch = [pid for pid in unique_ids if pid not in self._cache]
        if to_fetch:
            placeholders = ', '.join(['?'] * len(to_fetch))
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT parent_id, content, metadata FROM parents WHERE parent_id IN ({placeholders})",
                    to_fetch
                )
                for pid, cont, meta in cursor:
                    self._update_cache(pid, {"content": cont, "metadata": json.loads(meta)})
        results = [self.load_content(pid) for pid in unique_ids]
        return [r for r in results if r is not None]

    def delete(self, parent_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM parents WHERE parent_id = ?", (parent_id,))
        self._cache.pop(parent_id, None)