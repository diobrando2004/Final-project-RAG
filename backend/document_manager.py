from pathlib import Path
import shutil
import config
from pdfs_to_md import pdfs_to_markdowns, pdf_to_markdown, docx_to_markdown
import re
import logging
import sqlite3
from database import DataManager
from indexer import SemanticIndexer
logger = logging.getLogger(__name__)

 
PDF_EXTENSIONS = {".pdf", ".md", ".docx"}
CSV_EXTENSIONS = {".csv", ".xlsx", ".xls"}
class DocumentManager:
    def __init__(self,rag_system):
        self.rag_system = rag_system
        self.md_dir = Path(config.MARKDOWN_DIR)
        self.md_dir.mkdir(parents=True, exist_ok=True)

        
        self.csv_db = DataManager(config.CSV_DB_PATH, config.CSV_METADATA_DIR)
        self.csv_indexer = SemanticIndexer(
            self.rag_system.embedder, config.CSV_TABLE_INDEX_DIR
        )
        self._spatial_loaded = False
        self._setup_csv_db()
    def _setup_csv_db(self):
        self.csv_db.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                table_name        VARCHAR PRIMARY KEY,
                table_description TEXT,
                embedding         FLOAT[]
            )
        """)
    def _load_spatial(self):
        if not self._spatial_loaded:
            try:
                self.csv_db.execute("LOAD spatial;")
            except Exception:
                self.csv_db.execute("INSTALL spatial; LOAD spatial;")
            self._spatial_loaded = True
    def clean_text_for_summary(self, text: str) -> str:
        text = re.sub(r'\.\s*\.\s*\.', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text[500:4500].strip() if len(text) > 1000 else text.strip()
    
    def _generate_pdf_summary(self, text: str) -> str:
        self.rag_system.llm.reset()
        cleaned_sample = self.clean_text_for_summary(text)
        prompt = (
            "### TASK: You are a document classifier. "
            "Identify the main subject of the text below.\n"
            "### RULES:\n"
            "1. Ignore all navigation menus, page numbers, and table of contents.\n"
            "2. Do not repeat the text.\n"
            "3. Answer in one direct sentence starting with 'This document covers...'\n\n"
            f"### INPUT TEXT:\n{cleaned_sample}\n### END OF INPUT\n\n"
            "### SUMMARY:\nThis document covers"
        )
        output = self.rag_system.llm(prompt, max_tokens=80, temperature=0.2)
        summary = output['choices'][0]['text'].strip()
        return "This document covers " + summary

    def _generate_csv_summary(self, table_name: str, snippet_str: str) -> str:
        self.rag_system.llm.reset()
        prompt = (
            "### Task\n"
            "Write ONE sentence describing what data this table contains.\n"
            "Start with 'This table contains'.\n"
            "Do not mention column names. Do not explain yourself.\n\n"
            f"### Sample data\n{snippet_str}\n\n"
            "### Description\nThis table contains"
        )
        output = self.rag_system.llm(
            prompt, max_tokens=80, temperature=0.1, stop=["\n", "<|im_end|>"]
        )
        summary = output['choices'][0]['text'].strip()
        return "This table contains " + summary

    def _save_to_qdrant_summary(self, source_name: str, summary: str, file_type: str):
        summary_store = self.rag_system.vector_db.get_collection(
            self.rag_system.summary_collection_name
        )
        summary_store.add_texts(
            texts=[summary],
            metadatas=[{"source": source_name, "file_type": file_type}]
        )
 
    def _delete_from_qdrant_summary(self, source_name: str):
        from qdrant_client.http import models as qmodels
        client = self.rag_system.vector_db._Collection__client
        client.delete(
            collection_name=self.rag_system.summary_collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=source_name)
                        )
                    ]
                )
            )
        )

    def add_documents(self, document_paths):
        if not document_paths:
            return 0, 0
        document_paths = (
            [document_paths] if isinstance(document_paths, str) else document_paths
        )
 
        added, skipped = 0, 0
        for doc_path in document_paths:
            ext = Path(doc_path).suffix.lower()
            try:
                if ext in PDF_EXTENSIONS:
                    result = self._ingest_pdf(doc_path)
                elif ext in CSV_EXTENSIONS:
                    result = self._ingest_csv(doc_path)
                else:
                    logger.warning(f"Unsupported file type: {doc_path}")
                    skipped += 1
                    continue
 
                if result == "skipped":
                    skipped += 1
                else:
                    added += 1
            except Exception as e:
                logger.error(f"Error processing '{doc_path}': {e}")
                skipped += 1
 
        return added, skipped
    
    def _ingest_pdf(self, doc_path: str) -> str:
        doc_name = Path(doc_path).stem
        md_path = self.md_dir / f"{doc_name}.md"
 
        if md_path.exists():
            print(f"Skipping '{doc_name}' — already indexed.")
            return "skipped"
 
        if Path(doc_path).suffix.lower() == ".md":
            shutil.copy(doc_path, md_path)
        elif Path(doc_path).suffix.lower() == ".docx":
            docx_to_markdown(str(doc_path), self.md_dir)
        else:
            pdfs_to_markdowns(str(doc_path), overwrite=False)
 
        with open(md_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
 
        doc_summary = self._generate_pdf_summary(full_content)
 
        self.rag_system.parent_store.save_document_summary(
            source_name=doc_name,
            summary=doc_summary,
            metadata={"source": doc_name}
        )
 
        self._save_to_qdrant_summary(doc_name, doc_summary, file_type="pdf")
 
        parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
        child_collection = self.rag_system.vector_db.get_collection(
            self.rag_system.collection_name
        )
        child_collection.add_documents(child_chunks)
        self.rag_system.parent_store.save_multiple(parent_chunks)
 
        print(f"Ingested PDF: '{doc_name}'")
        return "added"
    
    def _ingest_csv(self, csv_path: str) -> str:
        table_name = Path(csv_path).stem.lower().replace("-", "_").replace(" ", "_")
        import re as _re
        table_name = _re.sub(r"[^a-z0-9_]", "", table_name)
 
        check = self.csv_db.execute(
            "SELECT 1 FROM system_metadata WHERE table_name = ?", [table_name]
        ).fetchone()
        if check:
            print(f"Skipping '{table_name}' — already indexed.")
            return "skipped"
 
        ext = Path(csv_path).suffix.lower()
        if ext == ".csv":
            source_fn = f"read_csv_auto('{csv_path}')"
        else:
            self._load_spatial()
            source_fn = f"st_read('{csv_path}')"
 
        cols = self.csv_db.execute(
            f"SELECT * FROM {source_fn} LIMIT 0"
        ).df().columns
        clean_cols = ", ".join([
            f'"{c}" AS {c.lower().replace(" ", "_").replace("-", "_")}'
            for c in cols
        ])
        self.csv_db.execute(
            f'CREATE TABLE IF NOT EXISTS "{table_name}" '
            f'AS SELECT {clean_cols} FROM {source_fn}'
        )
 
        snippet_str = self.csv_db.execute(
            f'SELECT * FROM "{table_name}" LIMIT 5'
        ).df().to_string(index=False)
        summary = self._generate_csv_summary(table_name, snippet_str)
 
        vector = self.rag_system.embedder.encode(summary).tolist()
        self.csv_db.execute(
            "INSERT OR REPLACE INTO system_metadata VALUES (?, ?, ?)",
            [table_name, summary, vector]
        )

        self._save_to_qdrant_summary(table_name, summary, file_type="csv")

        self.csv_indexer.build_custom_value_index(self.csv_db.conn)
 
        print(f"Ingested CSV: '{table_name}'")
        return "added"

    def delete_document(self, doc_name: str) -> str:
        doc_name = Path(doc_name).stem
 
        is_csv = self.csv_db.execute(
            "SELECT 1 FROM system_metadata WHERE table_name = ?", [doc_name]
        ).fetchone()
 
        if is_csv:
            return self._delete_csv(doc_name)
        else:
            return self._delete_pdf(doc_name)
        

    def _delete_pdf(self, doc_name: str) -> str:
        from qdrant_client.http import models as qmodels
        client = self.rag_system.vector_db._Collection__client
        errors = []
 
        try:
            client.delete(
                collection_name=self.rag_system.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=doc_name)
                        )]
                    )
                )
            )
        except Exception as e:
            errors.append(f"Qdrant child delete failed: {e}")
            logger.error(errors[-1])
 
        try:
            self._delete_from_qdrant_summary(doc_name)
        except Exception as e:
            errors.append(f"Qdrant summary delete failed: {e}")
            logger.error(errors[-1])
 
        try:
            db_path = self.rag_system.parent_store.db_path
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "DELETE FROM parents WHERE parent_id LIKE ?",
                    (f"{doc_name}_parent_%",)
                )
            for k in [k for k in self.rag_system.parent_store._cache
                      if k.startswith(f"{doc_name}_parent_")]:
                del self.rag_system.parent_store._cache[k]
        except Exception as e:
            errors.append(f"SQLite parents delete failed: {e}")
            logger.error(errors[-1])
 
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE source_name = ?", (doc_name,)
                )
        except Exception as e:
            errors.append(f"SQLite documents delete failed: {e}")
            logger.error(errors[-1])
 
        try:
            md_path = self.md_dir / f"{doc_name}.md"
            if md_path.exists():
                md_path.unlink()
        except Exception as e:
            errors.append(f"Markdown file delete failed: {e}")
            logger.error(errors[-1])
 
        if errors:
            return f" Deleted with some errors:\n" + "\n".join(errors)
        return f" '{doc_name}' fully removed."
 
    def _delete_csv(self, doc_name: str) -> str:
        errors = []
 
        # 1. Drop table from DuckDB
        try:
            self.csv_db.execute(f'DROP TABLE IF EXISTS "{doc_name}"')
        except Exception as e:
            errors.append(f"DuckDB table drop failed: {e}")
            logger.error(errors[-1])
 
        # 2. Delete from DuckDB system_metadata
        try:
            self.csv_db.execute(
                "DELETE FROM system_metadata WHERE table_name = ?", [doc_name]
            )
        except Exception as e:
            errors.append(f"DuckDB metadata delete failed: {e}")
            logger.error(errors[-1])
 
        # 3. Delete from Qdrant summary collection
        try:
            self._delete_from_qdrant_summary(doc_name)
        except Exception as e:
            errors.append(f"Qdrant summary delete failed: {e}")
            logger.error(errors[-1])
 
        # 4. Delete value index files
        import os
        for ext in ("_embs.npy", "_meta.json"):
            fpath = Path(config.CSV_TABLE_INDEX_DIR) / f"{doc_name}{ext}"
            try:
                if fpath.exists():
                    fpath.unlink()
            except Exception as e:
                errors.append(f"Index file delete failed: {e}")
                logger.error(errors[-1])
 
        # 5. Delete metadata cache JSON
        cache_path = Path(config.CSV_METADATA_DIR) / f"{doc_name}_info.json"
        try:
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            errors.append(f"Metadata cache delete failed: {e}")
            logger.error(errors[-1])
 
        if errors:
            return f"Deleted with some errors:\n" + "\n".join(errors)
        return f"'{doc_name}' fully removed."


    
    def list_documents(self) -> list[dict]:
        docs = []
 
        try:
            db_path = self.rag_system.parent_store.db_path
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT source_name, summary FROM documents ORDER BY source_name"
                ).fetchall()
            for row in rows:
                docs.append({
                    "name": row[0],
                    "summary": row[1] or "No summary",
                    "file_type": "pdf"
                })
        except Exception as e:
            logger.error(f"Failed to list PDF documents: {e}")
 
        try:
            rows = self.csv_db.safe_read(
                "SELECT table_name, table_description FROM system_metadata "
                "ORDER BY table_name"
            )
            for row in rows:
                docs.append({
                    "name": row[0],
                    "summary": row[1] or "No summary",
                    "file_type": "csv"
                })
        except Exception as e:
            logger.error(f"Failed to list CSV tables: {e}")
 
        return docs