import os
import glob
import shutil
import config

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_system import RAGsystem
from document_manager import DocumentManager
from retrieval import Retrieval, filter_by_score
from rag_pipe_line import RAGPipeline


class RAGExecutor:
    def __init__(self):
        self.rag = RAGsystem()
        self.rag.initialize()

        self.collection = self.rag.vector_db.get_collection(self.rag.collection_name)
        self.summary_collection = self.rag.vector_db.get_collection(self.rag.summary_collection_name)
        self.doc_manager = DocumentManager(self.rag)
        self.retriever = Retrieval(self.collection, self.summary_collection)
        self.llm = self.rag.llm
        self.csv_pipeline = RAGPipeline(
            llm=self.rag.llm,
            embedder=self.rag.embedder,
            db=self.doc_manager.csv_db
        )

    def ask(self, query: str, source_filters: list[str] = None) -> tuple:
        self.llm.reset()
        print(f"\n--- Query: {query} ---")
 
        if not source_filters:
            source_filters = ["Auto/All"]
        is_auto = source_filters == ["Auto/All"] or "Auto/All" in source_filters

        if is_auto:
            result = self.retriever.hierarchical_search(query, chunk_limit=6)
            matched = [result] if isinstance(result, dict) else result
            if not matched:
                return "I couldn't find any relevant information.", None
            if len(matched) == 1:
                entry = matched[0]
                if entry["file_type"] == "csv":
                    return self._ask_csv(query, entry["source"])
                else:
                    return self._ask_pdf(query, results=entry["results"])
            return self._ask_combined(query, matched)
 
        pdf_sources = []
        csv_sources = []
        for src in source_filters:
            is_csv = self.doc_manager.csv_db.safe_read(
                "SELECT 1 FROM system_metadata WHERE table_name = ?", [src]
            )
            if is_csv:
                csv_sources.append(src)
            else:
                pdf_sources.append(src)

        if len(source_filters) == 1:
            if csv_sources:
                return self._ask_csv(query, csv_sources[0])
            else:
                return self._ask_pdf(query, source_filter=pdf_sources[0])

        pdf_candidates = []
        for src in pdf_sources:
            results, best_score = self.retriever.search_child_with_score(query, limit=6, source_filter=src)
            if results:
                pdf_candidates.append({"source": src, "results": results, "score": best_score})
                print(f"Pinned PDF '{src}' best chunk score: {best_score:.2f}")
 
        # Only include PDFs whose best chunk score is within the gap of the top scorer
        matched = []
        if pdf_candidates:
            kept = filter_by_score(
                [(c["source"], c["score"]) for c in pdf_candidates],
                min_score=0.0, gap=config.CHUNK_SCORE_GAP, label="pinned PDF"
            )
            matched = [
                {"file_type": "pdf", "source": c["source"], "results": c["results"]}
                for c in pdf_candidates if c["source"] in kept
            ]
 
        for src in csv_sources:
            score_row = self.doc_manager.csv_db.safe_read(
                "SELECT list_cosine_similarity(embedding, ?::FLOAT[]) FROM system_metadata WHERE table_name = ?",
                [self.rag.embedder.encode(query).tolist(), src]
            )
            score = score_row[0][0] if score_row and score_row[0][0] is not None else 0.0
            print(f"Pinned CSV '{src}' summary score: {score:.3f}")
            if score >= config.SUMMARY_MIN_SCORE:
                matched.append({"file_type": "csv", "source": src, "results": []})
            else:
                print(f"Dropping pinned CSV '{src}' — score {score:.3f} below threshold {config.SUMMARY_MIN_SCORE}")
 
        if not matched:
            return "I couldn't find any relevant information in the selected documents.", None
        if len(matched) == 1:
            entry = matched[0]
            if entry["file_type"] == "csv":
                return self._ask_csv(query, entry["source"])
            else:
                return self._ask_pdf(query, results=entry["results"])
        return self._ask_combined(query, matched)
 
    def _ask_pdf(self, query: str, results=None, source_filter: str = None) -> tuple:
        if source_filter:
            results = self.retriever.search_child(
                query, limit=6, source_filter=source_filter
            )
        if not results:
            return "I couldn't find any relevant information in the documents.", None
 
        parent_ids = list({
            doc.metadata.get("parent_id")
            for doc in results
            if doc.metadata.get("parent_id")
        })
        context_text = self.retriever.retrieve_parent_many(parent_ids)
        print(f"PDF context: {len(context_text)} chars")
 
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question using ONLY the context.\n"
                        "Output the answer as plain text. Do not use markdown code blocks.\n"
                        "Do not explain your reasoning or mention the context.\n"
                        "If the answer is not in the context, respond with exactly: I don't know.\n"
                        "Do not repeat yourself."
                    )
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"
                }
            ],
            max_tokens=500,
            stop=["<|endoftext|>"],
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.2
        )
        return response["choices"][0]["message"]["content"], None
 
    def _ask_csv(self, query: str, table_source: str) -> tuple:
        if table_source:
            table_info = self.csv_pipeline.retrieve_relevant_table_by_name(table_source)
        else:
            table_info = self.csv_pipeline.retrieve_relevant_table(query)
        if not table_info:
            return "I couldn't find a relevant table for that question.", None
 
        result_data, sql_used = self.csv_pipeline.generate_and_execute_sql(
            query, table_info
        )
        print(f"CSV SQL: {sql_used}")
 
        if isinstance(result_data, str) and result_data.startswith("Error"):
            return self.csv_pipeline.synthesize(query, result=None, error=result_data), None
 
        summary = self.csv_pipeline.synthesize(query, result=result_data)
 
        if result_data is not None and not result_data.empty and result_data.shape != (1, 1):
            table = result_data.to_dict(orient="records")
        else:
            table = None
 
        return summary, table

    def _ask_combined(self, query: str, matched: list[dict]) -> tuple:
        context_parts = []
        csv_only = all(e["file_type"] == "csv" for e in matched)
        if csv_only:
            answers, all_tables = [], []
            for entry in matched:
                answer, table = self._ask_csv(query, entry["source"])
                if answer and "couldn't find" not in answer and "Error" not in answer:
                    answers.append(f"[{entry['source']}]: {answer}")
                if table:
                    all_tables.append(table)
            if not answers:
                return "I couldn't find any relevant information.", None
            return "\n\n".join(answers), all_tables[0] if len(all_tables) == 1 else None
 
        for entry in matched:
            if entry["file_type"] == "pdf" and entry["results"]:
                parent_ids = list({
                    doc.metadata.get("parent_id")
                    for doc in entry["results"]
                    if doc.metadata.get("parent_id")
                })
                pdf_context = self.retriever.retrieve_parent_many(parent_ids)
                if pdf_context:
                    source_label = entry["source"] or "document"
                    context_parts.append(
                        f"[From document '{source_label}']:\n{pdf_context}"
                    )
 
            elif entry["file_type"] == "csv":
                csv_context = self.csv_pipeline.try_get_csv_context(
                    query, entry["source"]
                )
                if csv_context:
                    context_parts.append(csv_context)
 
        if not context_parts:
            return "I couldn't find any relevant information.", None
 
        combined_context = "\n\n---\n\n".join(context_parts)
        print(f"Combined context: {len(combined_context)} chars from {len(context_parts)} source(s) {combined_context}")
 
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question using ONLY the context.\n"
                        "The context may contain both document text and structured data rows.\n"
                        "Output the answer as plain text. Do not use markdown code blocks.\n"
                        "Do not explain your reasoning or mention the context.\n"
                        "If the answer is not in the context, respond with exactly: I don't know.\n"
                        "Do not repeat yourself."
                    )
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{combined_context}\n\nQUESTION:\n{query}"
                }
            ],
            max_tokens=500,
            stop=["<|endoftext|>"],
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.2
        )
        return response["choices"][0]["message"]["content"], None

    def startup_ingest(self):
        all_files = []
        # PDF / MD files
        docs_dir = Path(config.DOCUMENTS_DIR)
        if docs_dir.exists():
            all_files += [
                str(f) for f in docs_dir.iterdir()
                if f.suffix.lower() in (".pdf", ".md")
            ]
        # CSV / Excel files
        csv_dir = Path(config.CSV_DIR)
        if csv_dir.exists():
            all_files += [
                str(f) for f in csv_dir.rglob("*")
                if f.suffix.lower() in (".csv", ".xlsx", ".xls")
            ]
 
        if not all_files:
            print("No documents found on startup.")
            return
 
        added, skipped = self.doc_manager.add_documents(all_files)
        print(f"Startup ingest complete — added: {added}, skipped: {skipped}")


# The executor is created once and reused across all requests
executor: RAGExecutor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor

    # Create all required directories
    for d in [
        config.MARKDOWN_DIR,
        config.DOCUMENTS_DIR,
        config.PARENT_STORE_PATH_SQLITE,
        config.QDRANT_DB_PATH,
        config.CSV_DIR,
        config.CSV_METADATA_DIR,
        config.CSV_TABLE_INDEX_DIR,
        str(Path(config.CSV_DB_PATH).parent),
    ]:
        os.makedirs(d, exist_ok=True)

    print("Initializing RAG system...")
    executor = RAGExecutor()
    executor.startup_ingest()
    print("RAG system ready.")

    yield  # server runs here

    print("Shutting down.")
    executor.doc_manager.csv_db.close()


app = FastAPI(title="RAG API", lifespan=lifespan)

# Allow the React dev server (port 5173) and production (port 80) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:80"],
    allow_methods=["*"],
    allow_headers=["*"],
)





class ChatRequest(BaseModel):
    query: str
    sources: list[str] = ["Auto/All"]

class ChatResponse(BaseModel):
    answer: str
    table: list | None = None

class DocumentInfo(BaseModel):
    name: str
    summary: str
    file_type: str = "pdf"

class DeleteResponse(BaseModel):
    status: str



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    answer, table = executor.ask(req.query, req.sources)
    return ChatResponse(answer=answer, table=table)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    docs = executor.doc_manager.list_documents()
    return [
        DocumentInfo(name=d["name"], summary=d["summary"], file_type=d.get("file_type", "pdf"))
        for d in docs
    ]


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    pdf_extensions = {".pdf", ".md"}
    csv_extensions = {".csv", ".xlsx", ".xls"}
    allowed = pdf_extensions | csv_extensions
    saved_paths = []
 
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {file.filename}. "
                    "Allowed: PDF, Markdown, CSV, Excel."
                )
            )
        dest_dir = config.CSV_DIR if suffix in csv_extensions else config.DOCUMENTS_DIR
        dest = Path(dest_dir) / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(dest))
 
    added, skipped = executor.doc_manager.add_documents(saved_paths)
    return {
        "added": added,
        "skipped": skipped,
        "message": f"{added} document(s) added, {skipped} skipped (already existed)."
    }


@app.delete("/documents/{doc_name}", response_model=DeleteResponse)
def delete_document(doc_name: str):
    status = executor.doc_manager.delete_document(doc_name)
    return DeleteResponse(status=status)


@app.get("/sources")
def get_sources():
    pdf_sources = list(executor.rag.vector_db.get_unique_sources(executor.rag.collection_name))
    csv_sources = [
        r[0] for r in executor.doc_manager.csv_db.safe_read(
            "SELECT table_name FROM system_metadata ORDER BY table_name"
        )
    ]
    all_sources = sorted(set(pdf_sources + csv_sources))
    return {"sources": ["Auto/All"] + all_sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)