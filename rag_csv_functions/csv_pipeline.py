import os
import re
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
 
import duckdb
from sentence_transformers import util
 
from backend import config
from database import DataManager
from indexer import SemanticIndexer

logger = logging.getLogger(__name__)
 
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def _safe_table_name(stem: str) -> str:
    name = stem.lower().replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    if not name or not name[0].isalpha():
        name = "t_" + name
    return name

class CSVPipeline:
    def __init__(self, llm, embedder,
                 db_path: str = config.CSV_DB_PATH,
                 metadata_dir: str = config.CSV_METADATA_DIR):
        self.llm = llm
        self.embedder = embedder
        self.db = DataManager(db_path, metadata_dir)
        self.indexer = SemanticIndexer(embedder)
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        self._ensure_metadata_table()
    
    def _ensure_metadata_table(self):
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                table_name  VARCHAR PRIMARY KEY,
                table_description TEXT,
                embedding   FLOAT[]
            )
        """)
 
    def _known_tables(self) -> list[str]:
        return [
            r[0] for r in
            self.db.conn.execute("SELECT table_name FROM system_metadata").fetchall()
        ]
    def _load_table_info(self, table_name: str) -> dict:
        path = os.path.join(self.metadata_dir, f"{table_name}_info.json")
        with open(path, "r") as f:
            return json.load(f)
        
    def _get_source_fn(self, csv_path: str) -> str:
        ext = Path(csv_path).suffix.lower()
        if ext == ".csv":
            return f"read_csv_auto('{csv_path}')"
        elif ext in (".xlsx", ".xls"):
            try:
                self.db.conn.execute("LOAD spatial;")
            except Exception:
                self.db.conn.execute("INSTALL spatial; LOAD spatial;")
            return f"st_read('{csv_path}')"
        raise ValueError(f"Unsupported format: {ext}")
    
    def ingest(self, csv_path: str, force_refresh: bool = False) -> dict:
        """
        Load a CSV/Excel file into DuckDB and generate + store a
        natural-language description for vector routing.
        Returns the table_info dict (from cache if already indexed).
        """
        table_name = _safe_table_name(Path(csv_path).stem)
        cache_path = os.path.join(self.metadata_dir, f"{table_name}_info.json")
 
        self._ensure_metadata_table()
 
        already_exists = self.db.conn.execute(
            "SELECT 1 FROM system_metadata WHERE table_name = ?", [table_name]
        ).fetchone()
 
        if already_exists and not force_refresh:
            logger.info(f"Table '{table_name}' already indexed, loading cache.")
            return self._load_table_info(table_name)
 
        logger.info(f"Ingesting '{table_name}' from {csv_path} ...")
        source_fn = self._get_source_fn(csv_path)
 
        # Normalise column names then create the table
        cols = self.db.conn.execute(f"SELECT * FROM {source_fn} LIMIT 0").df().columns
        clean_cols = ", ".join([
            f'"{c}" AS {_safe_table_name(c)}' for c in cols
        ])
        if force_refresh:
            self.db.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.db.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS "
            f"SELECT {clean_cols} FROM {source_fn}"
        )
 
        # Generate a plain-English description with the shared LLM
        snippet = self.db.conn.execute(
            f"SELECT * FROM {table_name} LIMIT 5"
        ).df().to_string(index=False)
        description = self._generate_description(snippet, csv_path)
 
        # Embed the description and store everything
        vector = self.embedder.encode(description, convert_to_numpy=True).astype("float32").tolist()
        self.db.conn.execute(
            "INSERT OR REPLACE INTO system_metadata VALUES (?, ?, ?)",
            [table_name, description, vector]
        )
 
        columns = {
            row["name"]: row["type"]
            for _, row in self.db.conn.execute(
                f"PRAGMA table_info('{table_name}')"
            ).df().iterrows()
        }
        table_info = {
            "table_name": table_name,
            "table_summary": description,
            "original_path": csv_path,
            "columns": columns,
        }
        with open(cache_path, "w") as f:
            json.dump(table_info, f)
 
        logger.info(f"Ingested '{table_name}' — {len(columns)} columns.")
        return table_info
    def ingest_folder(self, folder: str = config.CSV_DIR):
        """Ingest all CSV/Excel files in a folder, skipping already-known tables."""
        known = set(self._known_tables())
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"CSV folder not found: {folder}")
            return
        for f in folder_path.rglob("*"):
            if f.suffix.lower() in _SUPPORTED_EXTENSIONS:
                target = _safe_table_name(f.stem)
                if target not in known:
                    try:
                        self.ingest(str(f))
                    except Exception as e:
                        logger.error(f"Failed to ingest {f}: {e}")
    
    def _generate_description(self, snippet: str, csv_path: str) -> str:
        prompt = (
            "### Task\n"
            "Write ONE sentence describing what data this table contains.\n"
            "Start with 'This table contains'.\n"
            "Do not mention column names or technical details.\n\n"
            f"### Sample data\n{snippet}\n\n"
            "### Description\nThis table contains"
        )
        try:
            out = self.llm(prompt, max_tokens=80, temperature=0.1,
                           stop=["\n", "<|im_end|>"])
            text = out["choices"][0]["text"].strip()
            return "This table contains " + text
        except Exception as e:
            logger.warning(f"Description generation failed: {e}")
            return f"Data from {Path(csv_path).name}"
    def get_best_score(self, user_query: str) -> float:
        results = self._score_tables(user_query, top_k=1)
        return results[0][1] if results else 0.0
    
    def _score_tables(self, user_query: str, top_k: int = 2) -> list[tuple]:
        """Return [(table_name, score), ...] sorted by score descending."""
        query_vector = self.embedder.encode(
            user_query, convert_to_numpy=True
        ).astype("float32").tolist()
 
        rows = self.db.conn.execute("""
            SELECT table_name,
                   list_cosine_similarity(embedding, ?::FLOAT[]) AS score
            FROM system_metadata
            ORDER BY score DESC
            LIMIT ?
        """, [query_vector, top_k]).fetchall()
 
        return [(r[0], r[1]) for r in rows if r[1] >= config.CSV_MIN_SCORE]
    
    def retrieve_relevant_tables(self, user_query: str) -> list[dict]:
        scores = self._score_tables(user_query, top_k=2)
        if not scores:
            return []
 
        if len(scores) == 2:
            gap = scores[0][1] - scores[1][1]
            if gap < config.CSV_MULTI_TABLE_GAP:
                logger.info(
                    f"Multi-table routing: '{scores[0][0]}' ({scores[0][1]:.2f}), "
                    f"'{scores[1][0]}' ({scores[1][1]:.2f}), gap={gap:.2f}"
                )
                return [self._load_table_info(s[0]) for s in scores]
 
        logger.info(f"Single-table routing: '{scores[0][0]}' ({scores[0][1]:.2f})")
        return [self._load_table_info(scores[0][0])]
    @staticmethod
    def get_intent(user_query: str) -> str:
        q = user_query.lower()
        if any(w in q for w in config.CSV_AGGREGATE_WORDS):
            return "AGGREGATE"
        return "LOOKUP"
    
    def _build_schema_block(self, table_infos: list[dict]) -> str:

        blocks = []
        for info in table_infos:
            col_schema, sample_rows = self.db.get_table_context(info["table_name"])
            blocks.append(
                f"Table: {info['table_name']}\n"
                f"Columns:\n{col_schema}\n"
                f"Sample rows:\n{sample_rows}"
            )
        return "\n\n".join(blocks)
    
    def _build_hints_block(self, user_query: str, table_infos: list[dict]) -> str:
        hints = ""
        for info in table_infos:
            h = self.indexer.get_custom_hints(user_query, info["table_name"])
            if h:
                hints += h
        return hints
    
    def generate_sql(self, user_query: str, table_infos: list[dict]) -> str:

        intent = self.get_intent(user_query)
        schema_block = self._build_schema_block(table_infos)
        hints_block = self._build_hints_block(user_query, table_infos)
 
        # Anchor prefix steers generation and is prepended to output
        if intent == "LOOKUP":
            primary = table_infos[0]["table_name"]
            prefill = f'SELECT * FROM "{primary}" WHERE'
            intent_rule = (
                "RULE: Use SELECT * with LIMIT 3 for lookups.\n"
                f'Example: SELECT * FROM "{primary}" WHERE "name" ILIKE \'%value%\' LIMIT 3'
            )
        else:
            prefill = "SELECT"
            intent_rule = (
                "RULE: Use COUNT(*), SUM(), or AVG() for aggregates.\n"
                f'Example: SELECT COUNT(*) FROM "{table_infos[0]["table_name"]}" WHERE "col" = \'value\''
            )
 
        join_note = ""
        if len(table_infos) == 2:
            t1, t2 = table_infos[0]["table_name"], table_infos[1]["table_name"]
            join_note = (
                f"\nJOIN NOTE: Query may involve both '{t1}' and '{t2}'. "
                "Use JOIN only if the question clearly requires data from both tables."
            )
 
        prompt = (
            "### Task\n"
            "Generate a DuckDB SQL query to answer the question.\n\n"
            "### Rules\n"
            "1. Use DOUBLE QUOTES for table and column identifiers.\n"
            "2. Use single quotes for string values.\n"
            "3. Use ILIKE '%value%' for text searches.\n"
            "4. Output ONLY the SQL. No explanations.\n"
            "5. Filter ONLY by criteria mentioned in the question.\n"
            f"6. {intent_rule}\n"
            f"{join_note}\n\n"
            "### Database schema\n"
            f"{schema_block}\n\n"
            + (f"### Hints\n{hints_block}\n\n" if hints_block else "")
            + f"### Question\n{user_query}\n\n"
            "### SQL\n"
            f"{prefill}"
        )
 
        try:
            out = self.llm(
                prompt,
                max_tokens=150,
                temperature=0,
                stop=[";", "\n\n", "<|im_end|>"]
            )
            raw = out["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return ""
 
        # Reassemble with prefill, then clean up
        sql = f"{prefill} {raw}"
        sql = self._clean_sql(sql, table_infos)
        logger.info(f"Generated SQL: {sql}")
        return sql
    
    def _clean_sql(self, sql: str, table_infos: list[dict]) -> str:

        for kw in ("SELECT", "FROM", "WHERE"):
            sql = re.sub(rf"\b({kw})\s+\1\b", r"\1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"(?i)WHERE\s+SELECT", "WHERE", sql)
 
        # Collect all valid column names across all tables
        all_col_names: set[str] = set()
        for info in table_infos:
            all_col_names.update(info["columns"].keys())
 
        # If something is double-quoted but not a known column, make it a value
        for match in re.findall(r'"([^"]*)"', sql):
            if match not in all_col_names and match not in {i["table_name"] for i in table_infos}:
                sql = sql.replace(f'"{match}"', f"'{match}'")
 
        return sql.strip()
    
    
    def execute_sql(self, sql: str):
        """Run SQL against DuckDB. Returns (DataFrame, sql) or (error_str, sql)."""
        try:
            return self.db.conn.execute(sql).df(), sql
        except Exception as e:
            logger.error(f"SQL execution error: {e}\nSQL: {sql}")
            return f"Error executing SQL: {e}", sql
        

    def query(self, user_query: str) -> str:

        table_infos = self.retrieve_relevant_tables(user_query)
        if not table_infos:
            return "No relevant table found for that question."
 
        sql = self.generate_sql(user_query, table_infos)
        if not sql:
            return "Could not generate a SQL query for that question."
 
        result, sql_used = self.execute_sql(sql)
 
        if isinstance(result, str):
            return result  # error message
 
        if result.empty:
            return "The query ran successfully but returned no results."
 
        # Single-value result (COUNT, SUM, AVG) — format cleanly
        if result.shape == (1, 1):
            val = result.iloc[0, 0]
            return f"{val}"

        return result.to_string(index=False)
    
     
    def setup(self):
        """Call once at startup: ensure value index is built for all tables."""
        self.ingest_folder()
        self.indexer.build_custom_value_index(self.db.conn)
 
    def close(self):
        self.db.close()