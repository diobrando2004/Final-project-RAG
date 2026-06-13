import json
import logging
from pathlib import Path
from typing import Optional
import re
import sqlalchemy as sa
from sqlalchemy import text, inspect
import config
from get_models import AIProvider
logger = logging.getLogger(__name__)

SUPPORTED_DIALECTS = {"postgresql", "mysql"}
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    pass
class LiveSQLManager:
    
    def __init__(self, embedder, llm, summary_collection):
        self.embedder = embedder
        self.llm = llm
        self.ai = AIProvider(llm, embedder)
        self.summary_collection = summary_collection
        self._store_path = Path(config.SQL_CONNECTIONS_PATH)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._connections: dict[str, dict] = self._load_store()
        self._engines: dict[str, sa.Engine] = {}

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_store(self) -> dict:
        if self._store_path.exists():
            with open(self._store_path) as f:
                return json.load(f)
        return {}

    def _save_store(self):
        with open(self._store_path, "w") as f:
            json.dump(self._connections, f, indent=2)

    # ── Engine management ────────────────────────────────────────────────────

    def _get_engine(self, db_name: str) -> Optional[sa.Engine]:
        if db_name not in self._engines:
            if db_name not in self._connections:
                return None
            conn_str = self._connections[db_name]["connection_string"]
            try:
                engine = sa.create_engine(conn_str, pool_pre_ping=True)
                self._engines[db_name] = engine
            except Exception as e:
                logger.error(f"Failed to create engine for '{db_name}': {e}")
                return None
        return self._engines[db_name]

    def test_connection(self, connection_string: str) -> tuple[bool, str]:
        try:
            engine = sa.create_engine(connection_string, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful."
        except Exception as e:
            return False, str(e)

    # ── Discovery ────────────────────────────────────────────────────────────

    def _discover_tables(self, engine: sa.Engine) -> dict[str, dict]:
        """Return {table_name: {columns: {col: type}, sample: str}} for all tables."""
        inspector = inspect(engine)
        tables = {}
        for table_name in inspector.get_table_names():
            cols = {
                c["name"]: str(c["type"])
                for c in inspector.get_columns(table_name)
            }
            # Get a small sample
            try:
                with engine.connect() as conn:
                    rows = conn.execute(
                        text(f'SELECT * FROM "{table_name}" LIMIT 3')
                    ).fetchall()
                    keys = list(cols.keys())
                    sample = "\n".join(
                        ", ".join(f"{k}={v}" for k, v in zip(keys, row))
                        for row in rows
                    )
            except Exception:
                sample = ""
            tables[table_name] = {"columns": cols, "sample": sample}
        return tables

    # ── Summary generation ───────────────────────────────────────────────────

    def _generate_db_summary(self, db_name: str, tables: dict) -> str:
        table_list = ", ".join(tables.keys())
        self.llm.reset()
        prompt = (
            "### Task\n"
            "Write ONE sentence describing what this database contains.\n"
            "Start with 'This database contains'.\n"
            "Do not mention table names or column names. Do not explain yourself.\n\n"
            f"### Tables\n{table_list}\n\n"
            "### Description\nThis database contains"
        )
        output = self.llm(prompt, max_tokens=80, temperature=0.1, stop=["\n", "<|im_end|>"])
        return "This database contains " + output["choices"][0]["text"].strip()

    def _generate_table_summary(self, table_name: str, columns: dict, sample: str) -> str:
        self.llm.reset()
        snippet = "\n".join(f"{k}: {v}" for k, v in columns.items())
        prompt = (
            "### Task\n"
            "Write ONE sentence describing what data this table contains.\n"
            "Start with 'This table contains'.\n"
            "Do not mention column names. Do not explain yourself.\n\n"
            f"### Columns\n{snippet}\n\n"
            f"### Sample\n{sample}\n\n"
            "### Description\nThis table contains"
        )
        output = self.llm(prompt, max_tokens=80, temperature=0.1, stop=["\n", "<|im_end|>"])
        return "This table contains " + output["choices"][0]["text"].strip()

    # ── Qdrant summary ───────────────────────────────────────────────────────

    def _save_to_qdrant(self, source_name: str, summary: str):
        self.summary_collection.add_texts(
            texts=[summary],
            metadatas=[{"source": source_name, "file_type": "sql"}]
        )

    def _delete_from_qdrant(self, source_name: str):
        from qdrant_client.http import models as qmodels
        client = self.summary_collection.client
        client.delete(
            collection_name=self.summary_collection.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchValue(value=source_name)
                    )]
                )
            )
        )

    # ── Connect / disconnect ─────────────────────────────────────────────────

    def connect(self, db_name: str, connection_string: str) -> str:
        """Register a new live SQL database connection."""
        dialect = connection_string.split("://")[0].split("+")[0].lower()
        if dialect not in SUPPORTED_DIALECTS:
            return f"Unsupported dialect '{dialect}'. Supported: {', '.join(SUPPORTED_DIALECTS)}."

        ok, msg = self.test_connection(connection_string)
        if not ok:
            return f"Connection failed: {msg}"

        if db_name in self._connections:
            return f"'{db_name}' already connected. Use reload to refresh."

        engine = sa.create_engine(connection_string, pool_pre_ping=True)
        self._engines[db_name] = engine

        tables = self._discover_tables(engine)
        if not tables:
            return f"Connected but no tables found in '{db_name}'."

        db_summary = self._generate_db_summary(db_name, tables)
        table_meta = {}
        for table_name, info in tables.items():
            summary = self._generate_table_summary(
                table_name, info["columns"], info["sample"]
            )
            table_meta[table_name] = {
                "columns": info["columns"],
                "summary": summary
            }

        self._connections[db_name] = {
            "connection_string": connection_string,
            "dialect": dialect,
            "summary": db_summary,
            "tables": table_meta
        }
        self._save_store()
        self._save_to_qdrant(db_name, db_summary)

        print(f"Connected live SQL DB: '{db_name}' ({len(tables)} tables)")
        return f"'{db_name}' connected successfully with {len(tables)} table(s)."

    def disconnect(self, db_name: str) -> str:
        if db_name not in self._connections:
            return f"'{db_name}' not found."
        self._connections.pop(db_name)
        self._engines.pop(db_name, None)
        self._save_store()
        try:
            self._delete_from_qdrant(db_name)
        except Exception as e:
            logger.warning(f"Qdrant delete failed for '{db_name}': {e}")
        return f"'{db_name}' disconnected."

    def reload(self, db_name: str) -> str:
        if db_name not in self._connections:
            return f"'{db_name}' not found."
        conn_str = self._connections[db_name]["connection_string"]
        self.disconnect(db_name)
        return self.connect(db_name, conn_str)

    def check_on_startup(self):
        """On startup, verify all stored connections are still reachable."""
        for db_name, meta in list(self._connections.items()):
            ok, msg = self.test_connection(meta["connection_string"])
            if ok:
                print(f"Live SQL '{db_name}': connection OK.")
            else:
                print(f"Live SQL '{db_name}': connection FAILED ({msg}) — keeping metadata.")

    def list_databases(self) -> list[dict]:
        return [
            {
                "name": db_name,
                "dialect": meta["dialect"],
                "summary": meta["summary"],
                "tables": list(meta["tables"].keys()),
                "file_type": "sql"
            }
            for db_name, meta in self._connections.items()
        ]

    # ── Query ────────────────────────────────────────────────────────────────

    def get_table_info(self, db_name: str, table_name: str) -> Optional[dict]:
        if db_name not in self._connections:
            return None
        tables = self._connections[db_name]["tables"]
        if table_name not in tables:
            return None
        return {
            "table_name": table_name,
            "columns": tables[table_name]["columns"],
            "summary": tables[table_name]["summary"],
            "db_name": db_name
        }

    def get_best_table(self, db_name: str, query: str, table_filter: list[str] = None) -> Optional[dict]:
        """Pick the most relevant table for a query using embedding similarity."""
        if db_name not in self._connections:
            return None
        tables = self._connections[db_name]["tables"]
        if not tables:
            return None
        if table_filter:
            tables = {k: v for k, v in tables.items() if k in table_filter}
            if not tables:
                return None
        if len(tables) == 1:
            table_name = list(tables.keys())[0]
            return self.get_table_info(db_name, table_name)
        query_emb = list(self.embedder.embed([query]))[0]
        best_name, best_score = None, -1.0
        import numpy as np
        q = np.array(query_emb)
        for table_name, meta in tables.items():
            t = np.array(list(self.embedder.embed([meta["summary"]]))[0])
            score = float(np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t) + 1e-10))
            if score > best_score:
                best_score, best_name = score, table_name

        if best_score < config.SUMMARY_MIN_SCORE:
            print(f"Best table '{best_name}' score {best_score:.2f} below threshold {config.SUMMARY_MIN_SCORE}")
            return None
        return self.get_table_info(db_name, best_name)

    def generate_and_execute_sql(self, query: str, table_info: dict, db_name: str):
        dialect = self._connections[db_name]["dialect"]
        table_name = table_info["table_name"]
        columns = table_info["columns"]
        col_schema = "\n".join(f"- {c} ({t})" for c, t in columns.items())

        # Get sample rows for context
        engine = self._get_engine(db_name)
        if not engine:
            return None, "Error: no engine"
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(f'SELECT * FROM "{table_name}" LIMIT 3')
                ).fetchall()
            sample = "\n".join(str(dict(zip(columns.keys(), r))) for r in rows)
        except Exception:
            sample = "No sample available"

        # Aggregate or lookup intent
        q_lower = query.lower()
        is_agg = any(w in q_lower for w in config.CSV_AGGREGATE_WORDS)
        if is_agg:
            intent_rule = (
                "RULE: Use COUNT(*), SUM(), or AVG() for aggregates.\n"
                "If selecting a non-aggregated column alongside an aggregate, "
                "or ordering by an aggregate, you MUST add GROUP BY on the non-aggregated column(s).\n"
                f'Example: SELECT COUNT(*) FROM "{table_name}" WHERE "col" = \'value\'\n'
                f'Example: SELECT "col", SUM("value") FROM "{table_name}" GROUP BY "col" ORDER BY SUM("value") DESC LIMIT 1'
            )
            prefill = "SELECT"
        else:
            intent_rule = (
                "RULE: Use SELECT * with LIMIT 3 for lookups.\n"
                f'Example: SELECT * FROM "{table_name}" WHERE "name" ILIKE \'%value%\' LIMIT 3'
            )
            prefill = f'SELECT * FROM "{table_name}" WHERE'

        prompt = (
            "### Task\n"
            "Generate a SQL query to answer the question.\n\n"
            "### Rules\n"
            "1. Use DOUBLE QUOTES for identifiers.\n"
            "2. Use single quotes for string values.\n"
            "3. Use ILIKE for text searches if PostgreSQL, LIKE otherwise.\n"
            "4. Output ONLY the SQL. No explanations.\n"
            "5. Filter ONLY by criteria mentioned in the question.\n"
            f'6. The ONLY table you may query is "{table_name}".\n\n'
            f"7. {intent_rule}\n\n"
            "### Schema\n"
            f"{col_schema}\n\n"
            "### Sample\n"
            f"{sample}\n\n"
            f"### Question\n{query}\n\n"
            "### SQL\n"
            f"{prefill}"
        )

        self.llm.reset()
        output = self.ai.generate_sql(prompt)
        sql = f"{prefill} {output}"
        sql = self.deduplicate_sql(sql)

        col_names = set(columns.keys())
        for match in re.findall(r'"([^"]*)"', sql):
            if match != table_name and match not in col_names:
                sql = sql.replace(f'"{match}"', f"'{match}'")

        if dialect == "mysql":
            sql = re.sub(r'"([^"]*)"', r'`\1`', sql)
        # Guard: ensure correct table name in SQL
        if f'"{table_name}"' not in sql and table_name not in sql:
            return None, f"Error: LLM generated SQL for wrong table."

        print(f"Live SQL [{db_name}.{table_name}]: {sql}")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                keys = list(result.keys())
                import pandas as pd
                df = pd.DataFrame(rows, columns=keys)
                return df, sql
        except Exception as e:
            return None, f"Error executing SQL: {e}"

    def synthesize(self, query: str, df, error: str = None) -> str:
        if error:
            data_context = f"A database error occurred: {error}"
        elif df is None or df.empty:
            data_context = "The query returned no results."
        elif df.shape == (1, 1):
            data_context = f"Result: {df.iloc[0, 0]}"
        else:
            rows = []
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                fields = ", ".join(f"{col}={val}" for col, val in row.items())
                rows.append(f"Row {i}: {fields}")
            data_context = f"{len(df)} row(s) found:\n" + "\n".join(rows)

        self.llm.reset()
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data assistant. Summarize the data in plain English.\n"
                        "If there is an error, explain in plain terms what likely went wrong.\n"
                        "If no results, say so clearly. Be concise. No markdown. Two sentences max."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nSummarize this data:\n{data_context}"
                }
            ],
            max_tokens=120,
            temperature=0.1,
            stop=["<|im_end|>"]
        )
        return response["choices"][0]["message"]["content"].strip()
    
    
    @staticmethod
    def deduplicate_sql(sql):
        sql = re.split(r'```', sql)[0].strip()
        sql = re.sub(r'(?i)WHERE\s+SELECT', 'WHERE', sql)
        for word in ["SELECT", "FROM", "WHERE"]:
            sql = re.sub(rf'\b({word})\s+\1\b', r'\1', sql, flags=re.IGNORECASE)
        lines = sql.split('\n')
        if len(lines) > 1 and lines[0].strip() == lines[1].strip():
            sql = lines[0]
        return sql
