import os
import json
import re
from pathlib import Path

import config
from database import DataManager
from get_models import AIProvider
from indexer import SemanticIndexer


DATA_FILES = [
    str(file) for file in Path(config.CSV_DIR).rglob("*")
    if file.suffix.lower() in {'.csv', '.xlsx', '.xls'}
]


class RAGPipeline:
    def __init__(self, llm, embedder, db=None):
        # Use shared connection if provided, otherwise open own connection
        self.db = db if db is not None else DataManager(config.CSV_DB_PATH, config.CSV_METADATA_DIR)
        self.ai = AIProvider(llm, embedder)
        self.indexer = SemanticIndexer(embedder, config.CSV_TABLE_INDEX_DIR)
        self._spatial_loaded = False


    def _setup(self):
        """Create system_metadata table if it does not exist yet."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                table_name        VARCHAR PRIMARY KEY,
                table_description TEXT,
                embedding         FLOAT[]
            )
        """)

    def _load_spatial(self):
        """Load the spatial extension once for xlsx support."""
        if not self._spatial_loaded:
            try:
                self.db.execute("LOAD spatial;")
            except Exception:
                self.db.execute("INSTALL spatial; LOAD spatial;")
            self._spatial_loaded = True


    def ingest_and_describe_csv(self, csv_path, force_refresh=False):
        table_name = Path(csv_path).stem.lower().replace("-", "_").replace(" ", "_")
        cache_path = os.path.join(config.CSV_METADATA_DIR, f"{table_name}_info.json")
        ext = Path(csv_path).suffix.lower()

        print(f"Adding table '{table_name}' if not exists in DB...")

        if ext == '.csv':
            source_fn = f"read_csv_auto('{csv_path}')"
        elif ext in ('.xlsx', '.xls'):
            self._load_spatial()
            source_fn = f"st_read('{csv_path}')"
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        check = self.db.execute(
            "SELECT table_description FROM system_metadata WHERE table_name = ?",
            [table_name]
        ).fetchone()

        if not check or force_refresh:
            print(f"Metadata not found for '{table_name}'. Processing...")

            cols = self.db.execute(
                f"SELECT * FROM {source_fn} LIMIT 0"
            ).df().columns
            clean_cols = ", ".join([
                f'"{c}" AS {c.lower().replace(" ", "_").replace("-", "_")}'
                for c in cols
            ])

            if force_refresh:
                self.db.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            self.db.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" '
                f'AS SELECT {clean_cols} FROM {source_fn}'
            )

            snippet_str = self.db.execute(
                f'SELECT * FROM "{table_name}" LIMIT 5'
            ).df().to_string(index=False)

            description = self.ai.generate_description(table_name, snippet_str)

            print(f"Embedding description for '{table_name}'...")
            vector = self.ai.embedder.encode(description).tolist()

            self.db.execute(
                "INSERT OR REPLACE INTO system_metadata VALUES (?, ?, ?)",
                [table_name, description, vector]
            )

            columns = {
                row['name']: row['type']
                for _, row in self.db.execute(
                    f"PRAGMA table_info('{table_name}')"
                ).df().iterrows()
            }
            table_info = {
                'table_name':    table_name,
                'table_summary': description,
                'original_path': csv_path,
                'columns':       columns,
            }
            with open(cache_path, 'w') as f:
                json.dump(table_info, f)

            return table_info

        else:
            print(f"Metadata for '{table_name}' already indexed in DB.")
            with open(cache_path, 'r') as f:
                return json.load(f)


    def retrieve_relevant_table_by_name(self, table_name: str):
        print(f"Loading pinned table: {table_name}")
        if not self._table_exists_in_duckdb(table_name):
            print(f"Table '{table_name}' does not exist in DuckDB — skipping.")
            return None
        cache_path = os.path.join(config.CSV_METADATA_DIR, f"{table_name}_info.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                info = json.load(f)
            if info.get('table_name') == table_name:
                return info
            print(f"Cache mismatch for '{table_name}' — rebuilding.")

        # Build from DuckDB if cache missing
        description = self.db.execute(
            "SELECT table_description FROM system_metadata WHERE table_name = ?",
            [table_name]
        ).fetchone()
        if not description:
            print(f"Table '{table_name}' not found in system_metadata.")
            return None
        columns = {
            row['name']: row['type']
            for _, row in self.db.execute(
                f"PRAGMA table_info('{table_name}')"
            ).df().iterrows()
        }
        table_info = {
            'table_name':    table_name,
            'table_summary': description[0] if description else "",
            'original_path': "",
            'columns':       columns,
        }
        with open(cache_path, 'w') as f:
            json.dump(table_info, f)
        return table_info

    def retrieve_relevant_table(self, user_query):
        print("Routing query via vector search...")

        query_vector = self.ai.embedder.encode(user_query).tolist()

        result = self.db.execute("""
            SELECT table_name,
                   list_cosine_similarity(embedding, ?::FLOAT[]) AS score
            FROM system_metadata
            ORDER BY score DESC
            LIMIT 1
        """, [query_vector]).fetchone()

        if not result or result[1] < config.CSV_MIN_SCORE:
            print("No relevant table found.")
            return None

        best_name = result[0]
        print(f"-> Selected '{best_name}' (score: {result[1]:.2f})")

        cache_path = os.path.join(config.CSV_METADATA_DIR, f"{best_name}_info.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        description = self.db.execute(
            "SELECT table_description FROM system_metadata WHERE table_name = ?",
            [best_name]
        ).fetchone()
        columns = {
            row['name']: row['type']
            for _, row in self.db.execute(
                f"PRAGMA table_info('{best_name}')"
            ).df().iterrows()
        }
        table_info = {
            'table_name':    best_name,
            'table_summary': description[0] if description else "",
            'original_path': "",
            'columns':       columns,
        }
        # Write cache so next call is fast
        with open(cache_path, 'w') as f:
            json.dump(table_info, f)
        return table_info

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    @staticmethod
    def get_query_intent(user_query):
        q = user_query.lower()
        if any(w in q for w in config.CSV_AGGREGATE_WORDS):
            return "AGGREGATE"
        return "LOOKUP"

    # ------------------------------------------------------------------
    # SQL generation and execution
    # ------------------------------------------------------------------

    def deduplicate_sql(self, sql):
        sql = re.sub(r'(?i)WHERE\s+SELECT', 'WHERE', sql)
        for word in ["SELECT", "FROM", "WHERE"]:
            sql = re.sub(rf'\b({word})\s+\1\b', r'\1', sql, flags=re.IGNORECASE)
        return sql

    def generate_and_execute_sql(self, user_query, target_table_info):
        table_name = target_table_info['table_name']

        if not self._table_exists_in_duckdb(table_name):
            err = f"Error executing SQL: Table '{table_name}' does not exist in DuckDB."
            print(err)
            return err, ""
        
        col_schema, sample_rows = self.db.get_table_context(table_name)
        hints = self.indexer.get_custom_hints(user_query, table_name)
        intent = self.get_query_intent(user_query)

        if intent == "LOOKUP":
            intent_rule = (
                "RULE: Use SELECT * with LIMIT 3 for lookups.\n"
                f'Example: SELECT * FROM "{table_name}" WHERE "name" ILIKE \'%value%\' LIMIT 3'
            )
            prefill = f'SELECT * FROM "{table_name}" WHERE'
        else:
            intent_rule = (
                "RULE: Use COUNT(*), SUM(), or AVG() for aggregates.\n"
                f'Example: SELECT COUNT(*) FROM "{table_name}" WHERE "col" = \'value\''
            )
            prefill = "SELECT"

        prompt = (
            "### Task\n"
            "Generate a DuckDB SQL query to answer the question.\n\n"
            "### Rules\n"
            "1. Use DOUBLE QUOTES for table and column identifiers.\n"
            "2. Use single quotes for string values.\n"
            "3. Use ILIKE '%value%' for text searches.\n"
            "4. Output ONLY the SQL. No explanations.\n"
            "5. Filter ONLY by criteria mentioned in the question.\n"
            f"6. {intent_rule}\n\n"
            "### Database schema\n"
            f"{col_schema}\n\n"
            "### Sample data\n"
            f"{sample_rows}\n\n"
            + (f"### Hints\n{hints}\n\n" if hints else "")
            + f"### Question\n{user_query}\n\n"
            "### SQL\n"
            f"{prefill}"
        )

        generated_text = self.ai.generate_sql(prompt)
        clean_sql = f"{prefill} {generated_text}"
        clean_sql = self.deduplicate_sql(clean_sql)

        col_names = set(target_table_info['columns'].keys())
        for match in re.findall(r'"([^"]*)"', clean_sql):
            if match != table_name and match not in col_names:
                clean_sql = clean_sql.replace(f'"{match}"', f"'{match}'")

        print(f"Executing SQL: {clean_sql}")
        try:
            result_df = self.db.execute(clean_sql).df()
            return result_df, clean_sql
        except Exception as e:
            return f"Error executing SQL: {e}", clean_sql
        


    def try_get_csv_context(self, user_query: str, table_name: str) -> str | None:
        try:
            table_info = self.retrieve_relevant_table_by_name(table_name)
            if not table_info:
                return None
 
            result_df, sql_used = self.generate_and_execute_sql(user_query, table_info)
            print(f"CSV context SQL: {sql_used}")
 
            if isinstance(result_df, str):
                print(f"CSV SQL failed silently: {result_df}")
                return None
 
            if result_df is None or result_df.empty:
                return None
 
            if result_df.shape == (1, 1):
                return f"[From table '{table_name}']: {result_df.iloc[0, 0]}"

            rows = []
            for i, (_, row) in enumerate(result_df.iterrows(), start=1):
                fields = ", ".join(f"{col}={val}" for col, val in row.items())
                rows.append(f"Row {i}: {fields}")
 
            return f"[From table '{table_name}']:\n" + "\n".join(rows)
 
        except Exception as e:
            print(f"CSV context failed silently: {e}")
            return None

    def _table_exists_in_duckdb(self, table_name: str) -> bool:
        """Check that the physical table exists in DuckDB (not just in metadata)."""
        try:
            result = self.db.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, user_query, result, error=None):
        print(result)
        if error:
            data_context = f"A database error occurred: {error}"
        elif result is None or result.empty:
            data_context = "The query returned no results."
        elif result.shape == (1, 1):
            data_context = f"Result: {result.iloc[0, 0]}"
            print(data_context)
        else:
            rows = []
            for i, (_, row) in enumerate(result.iterrows(), start=1):
                fields = ", ".join(f"{col}={val}" for col, val in row.items())
                rows.append(f"Row {i}: {fields}")
            data_context = f"{len(result)} row(s) found:\n" + "\n".join(rows)
        self.ai.llm.reset()
        response = self.ai.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data assistant. Your job is to summarize "
                        "the data provided in plain English.\n"
                        "If there is an error, explain in plain terms what likely "
                        "went wrong without using technical SQL jargon.\n"
                        "If there are no results, say so clearly.\n"
                        "Be concise. No markdown. Two sentences max."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {user_query}\n\n"
                        f"Summarize this data:\n{data_context}"
                    )
                }
            ],
            max_tokens=120,
            temperature=0.1,
            stop=["<|im_end|>"]
        )
        return response["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Standalone run loop
    # ------------------------------------------------------------------

    def run(self):
        print("\n--- Setup ---")
        self._setup()

        known_tables = [
            t[0] for t in
            self.db.execute("SELECT table_name FROM system_metadata").fetchall()
        ]

        for csv_file in DATA_FILES:
            target_name = Path(csv_file).stem.lower().replace("-", "_").replace(" ", "_")
            if target_name not in known_tables:
                print(f"New file found: {csv_file}. Ingesting...")
                self.ingest_and_describe_csv(csv_file)

        print(f"Pipeline ready. {len(known_tables)} table(s) indexed.")
        self.indexer.build_custom_value_index(self.db.conn)

        while True:
            print("\n--- New Query ---")
            user_input = input("Enter your question (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break

            table_info = self.retrieve_relevant_table(user_input)
            if not table_info:
                print("Could not find a relevant table for that question.")
                continue

            result_data, sql_used = self.generate_and_execute_sql(user_input, table_info)

            if isinstance(result_data, str) and result_data.startswith("Error"):
                answer = self.synthesize(user_input, result=None, error=result_data)
            else:
                answer = self.synthesize(user_input, result=result_data)

            print(f"\nSQL: {sql_used}")
            print(f"\nAnswer: {answer}")

        self.db.close()
        print("Database connection closed.")