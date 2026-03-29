from pathlib import Path
import shutil
import config
from pdfs_to_md import pdfs_to_markdowns, pdf_to_markdown
import re
import logging
import sqlite3
from database import DataManager
from indexer import SemanticIndexer
logger = logging.getLogger(__name__)

 
PDF_EXTENSIONS = {".pdf", ".md"}
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
        self.csv_db.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                table_name        VARCHAR PRIMARY KEY,
                table_description TEXT,
                embedding         FLOAT[]
            )
        """)
    def clean_text_for_summary(self, text: str) -> str:
        text = re.sub(r'\.\s*\.\s*\.', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text[500:4500].strip() if len(text) > 1000 else text.strip()
    def _generate_document_summary(self, text: str) -> str:
        self.rag_system.llm.reset()
        cleaned_sample = self.clean_text_for_summary(text)
        print(cleaned_sample)
    # We use a 'Factual Identification' prompt instead of 'Summarization'
        prompt = prompt = f"""### TASK: You are a document classifier. Identify the main subject of the text below.
### RULES: 
1. Ignore all navigation menus, page numbers, and table of contents.
2. Do not repeat the text.
3. nswer in one direct sentence starting with "This document covers..."

### INPUT TEXT:
{cleaned_sample}
### END OF INPUT

### SUMMARY:
This document covers"""


        output = self.rag_system.llm(
            prompt,
            max_tokens=80, 
            temperature=0.2 
        )
        print(output['choices'][0]['text'].strip())
        return output['choices'][0]['text'].strip()

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

    def add_documents(self, document_paths):
        if not document_paths: return 0, 0
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        
        added, skipped = 0, 0
        for doc_path in document_paths:
            doc_name = Path(doc_path).stem
            md_path = self.md_dir / f"{doc_name}.md"
            
            if md_path.exists():
                print("skipped here")
                skipped += 1; continue
                
            try:
                if Path(doc_path).suffix.lower() == ".md":
                    shutil.copy(doc_path, md_path)
                else:
                    pdfs_to_markdowns(str(doc_path), overwrite=False)

                with open(md_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()

                doc_summary = self._generate_document_summary(full_content)
                
                self.rag_system.parent_store.save_document_summary(
                    source_name=doc_name, 
                    summary=doc_summary, 
                    metadata={"source": doc_name}
                )
                
                # Save to Qdrant Summary Collection
                # We use the source name as page_content so similarity search 
                # maps the query to the summary we embed here.
                summary_store = self.rag_system.vector_db.get_collection("document_summaries")
                summary_data = {
                    "page_content": doc_summary,
                    "metadata": {"source": doc_name}
                }
                summary_store.add_texts(
                    texts=[summary_data["page_content"]],
                    metadatas=[summary_data["metadata"]]
                )

                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                
                child_collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                child_collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_multiple(parent_chunks)
                
                added += 1
            except Exception as e:
                print(f"Error processing document '{doc_name}': {e}")
                skipped += 1

        return added, skipped

    def delete_document(self, doc_name: str) -> str:
        """
        Fully removes a document from:
        - Qdrant child chunk collection
        - Qdrant summary collection
        - SQLite parents table
        - SQLite documents table
        - Markdown file on disk
        Returns a status message.
        """
        doc_name = Path(doc_name).stem  # normalize
        errors = []
 
        # 1. Delete child chunks from Qdrant
        try:
            from qdrant_client.http import models as qmodels
            client = self.rag_system.vector_db._Collection__client
            client.delete(
                collection_name=self.rag_system.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="metadata.source",
                                match=qmodels.MatchValue(value=doc_name)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            errors.append(f"Qdrant child delete failed: {e}")
            logger.error(errors[-1])
 
        # 2. Delete from Qdrant summary collection
        try:
            client.delete(
                collection_name=self.rag_system.summary_collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="metadata.source",
                                match=qmodels.MatchValue(value=doc_name)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            errors.append(f"Qdrant summary delete failed: {e}")
            logger.error(errors[-1])
 
        # 3. Delete parent chunks from SQLite
        try:
            db_path = self.rag_system.parent_store.db_path
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "DELETE FROM parents WHERE parent_id LIKE ?",
                    (f"{doc_name}_parent_%",)
                )
            keys_to_delete = [
                k for k in self.rag_system.parent_store._cache
                if k.startswith(f"{doc_name}_parent_")
            ]
            for k in keys_to_delete:
                del self.rag_system.parent_store._cache[k]
        except Exception as e:
            errors.append(f"SQLite parents delete failed: {e}")
            logger.error(errors[-1])
 
        # 4. Delete document summary from SQLite
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "DELETE FROM documents WHERE source_name = ?",
                    (doc_name,)
                )
        except Exception as e:
            errors.append(f"SQLite documents delete failed: {e}")
            logger.error(errors[-1])
 
        # 5. Delete markdown file
        md_path = self.md_dir / f"{doc_name}.md"
        try:
            if md_path.exists():
                md_path.unlink()
        except Exception as e:
            errors.append(f"Markdown file delete failed: {e}")
            logger.error(errors[-1])
 
        if errors:
            return f"⚠️ Deleted with some errors:\n" + "\n".join(errors)
        return f"✅ '{doc_name}' fully removed."
    
    def list_documents(self) -> list[dict]:
        """Returns ingested documents with name and summary from SQLite."""
        try:
            db_path = self.rag_system.parent_store.db_path
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT source_name, summary FROM documents ORDER BY source_name"
                ).fetchall()
            return [{"name": row[0], "summary": row[1] or "No summary"} for row in rows]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []