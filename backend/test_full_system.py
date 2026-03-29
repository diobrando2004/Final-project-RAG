import os
import config
from rag_system import RAGsystem
from document_manager import DocumentManager
from retrieval import Retrieval
from qdrant_client.http import models as qmodels
import gradio as gr

class RAGExecutor:
    def __init__(self):
        self.rag = RAGsystem()
        self.rag.initialize()
        
        self.collection = self.rag.vector_db.get_collection(self.rag.collection_name)
        self.summary_collection = self.rag.vector_db.get_collection(self.rag.summary_collection_name)
        self.doc_manager = DocumentManager(self.rag)
        self.retriever = Retrieval(self.collection, self.summary_collection)
        self.llm = self.rag.llm

    def ingest_docs(self, folder_path):
        print(f"--- Ingesting documents from: {folder_path} ---")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith(('.pdf', '.md'))]
        added, skipped = self.doc_manager.add_documents(files)
        print(f"Indexing complete. Added: {added}, Skipped/Existing: {skipped}")

    def ask(self, query: str, source_filter_str: str = None):
        self.llm.reset()
        print(f"\n--- Query: {query} ---")
        
        if not source_filter_str or source_filter_str == "Auto/All":
            results = self.retriever.hierarchical_search(query, chunk_limit=6)
        else:
            results = self.retriever.search_child(query, limit=6, source_filter=source_filter_str)

        if not results:
            return "I couldn't find any relevant information in the documents."

        parent_ids = list(set([doc.metadata.get('parent_id') for doc in results if doc.metadata.get('parent_id')]))
        context_text = self.retriever.retrieve_parent_many(parent_ids)
        print(f"Context retrieved: {len(context_text)} chars")

        prompt = f"""
CONTEXT:
{context_text}

QUESTION:
{query}"""

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": '''You are a helpful assistant. Answer the question using ONLY the context.
Output the answer as plain text. Do not use markdown code blocks.
Do not explain your reasoning or mention the context.
if there is no relevance detail say "I don't know"
Do not repeat yourself.'''},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            stop=["<|endoftext|>", "I don't know", "The answer should be"],
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.2
        )
        return response["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Document management actions (new)
    # ------------------------------------------------------------------

    def ui_list_documents(self) -> str:
        docs = self.doc_manager.list_documents()
        if not docs:
            return "No documents ingested yet."
        return "\n".join([f"• {d['name']}" for d in docs])

    def ui_upload_and_ingest(self, files) -> str:
        if not files:
            return "No files selected."
        paths = [f.name if hasattr(f, "name") else str(f) for f in files]
        added, skipped = self.doc_manager.add_documents(paths)
        parts = []
        if added:
            parts.append(f"{added} document(s) added.")
        if skipped:
            parts.append(f"{skipped} already existed, skipped.")
        return " ".join(parts) if parts else "Nothing processed."

    def ui_delete_document(self, doc_name: str) -> str:
        """Deletes a document by name, returns status."""
        if not doc_name or not doc_name.strip():
            return "⚠️ Please enter a document name."
        return self.doc_manager.delete_document(doc_name.strip())


if __name__ == "__main__":
    os.makedirs(config.MARKDOWN_DIR, exist_ok=True)
    system = RAGExecutor()

    system.ingest_docs("./documents")

    tags = list(system.rag.vector_db.get_unique_sources(system.rag.collection_name))
    dropdown_choices = ["Auto/All"] + tags

    # ── Tab 1: Chat
    chat_interface = gr.Interface(
        fn=system.ask,
        inputs=[
            gr.Textbox(label="Question"),
            gr.Dropdown(choices=dropdown_choices, value="Auto/All", label="Source Tag")
        ],
        outputs="text",
        title="Ask a Question"
    )

    # ── Tab 2: Document Management
    with gr.Blocks() as docs_interface:
        gr.Markdown("## 📁 Document Management")

        # Section: current documents
        gr.Markdown("### Ingested Documents")
        doc_list_output = gr.Textbox(
            label="Documents in database",
            value=system.ui_list_documents(),
            interactive=False,
            lines=8
        )
        refresh_btn = gr.Button("🔄 Refresh list")
        refresh_btn.click(fn=system.ui_list_documents, outputs=doc_list_output)

        gr.Markdown("---")

        # Section: upload new documents
        gr.Markdown("### Upload New Documents")
        upload_box = gr.File(
            label="Drag & drop PDFs or Markdown files here",
            file_types=[".pdf", ".md"],
            file_count="multiple"
        )
        ingest_btn = gr.Button("⬆️ Ingest uploaded files", variant="primary")
        ingest_status = gr.Textbox(label="Upload status", interactive=False)
        ingest_btn.click(
            fn=system.ui_upload_and_ingest,
            inputs=[upload_box],
            outputs=[ingest_status]
        )

        gr.Markdown("---")

        # Section: delete a document
        gr.Markdown("### Delete a Document")
        gr.Markdown("<small>Copy the document name exactly from the list above.</small>")
        delete_input = gr.Textbox(label="Document name to delete", placeholder="e.g. my_report")
        delete_btn = gr.Button("🗑️ Delete document", variant="stop")
        delete_status = gr.Textbox(label="Delete status", interactive=False)
        delete_btn.click(
            fn=system.ui_delete_document,
            inputs=[delete_input],
            outputs=[delete_status]
        )

    # ── Combine into tabs ─────────────────────────────────────────────
    app = gr.TabbedInterface(
        interface_list=[chat_interface, docs_interface],
        tab_names=["💬 Chat", "📁 Documents"]
    )
    app.launch(server_name="0.0.0.0", server_port=7860)