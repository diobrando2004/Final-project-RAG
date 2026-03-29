from store_parents import ParentStore
from qdrant_client.http import models as qmodels
import config   
def filter_by_score(
    candidates: list[tuple[str, float]],
    min_score: float,
    gap: float,
    label: str = "source"
) -> list[str]:

    qualified = [(src, s) for src, s in candidates if s >= min_score]
    if not qualified:
        return []
    best = max(s for _, s in qualified)
    kept, dropped = [], []
    for src, s in qualified:
        if best - s <= gap:
            kept.append(src)
        else:
            dropped.append((src, s))
    for src, s in dropped:
        print(f"Dropping {label} '{src}' — score {s:.3f} too far behind top {best:.3f}")
    return kept
class Retrieval:
    def __init__(self, collection, summary_collection):
        self.collection = collection
        self.summary_collection = summary_collection
        self.parent_store = ParentStore()
    
    def hierarchical_search(self, query: str, chunk_limit: int = 5) -> list:
        try:
            # --- Step 1: Score all CSV summaries and filter ---
            summary_results = self.summary_collection.similarity_search_with_score(query, k=10)
 
            all_csv = [
                (doc.metadata.get("source"), score)
                for doc, score in summary_results
                if doc.metadata.get("file_type") == "csv"
            ]
            for src, score in all_csv:
                print(f"CSV score: '{src}' = {score:.3f} (min={config.SUMMARY_MIN_SCORE})")
 
            csv_sources = filter_by_score(
                all_csv, config.SUMMARY_MIN_SCORE, config.SUMMARY_SCORE_GAP, label="CSV"
            )
 
            # If one CSV clearly dominates, keep only that one
            if len(csv_sources) > 1:
                top_score = max(s for src, s in all_csv if src in csv_sources)
                if top_score >= 0.5:
                    csv_sources = [src for src, s in all_csv if s == top_score]
                    print(f"Dominant CSV (score={top_score:.3f}), keeping only: {csv_sources}")
 
            # --- Step 2: Search child chunks across all PDF sources ---
            all_chunks = self.collection.similarity_search_with_score(
                query, k=chunk_limit * 3, score_threshold=0.3
            )
 
            sources: dict[str, dict] = {}
            for doc, score in all_chunks:
                src = doc.metadata.get("source")
                if src not in sources:
                    sources[src] = {"best_score": score, "docs": []}
                sources[src]["docs"].append(doc)
                if score > sources[src]["best_score"]:
                    sources[src]["best_score"] = score
 
            # --- Step 3: Filter PDF sources by chunk score gap ---
            matched = []
            if sources:
                pdf_candidates = [(src, d["best_score"]) for src, d in sources.items()]
                kept_pdfs = filter_by_score(
                    pdf_candidates, min_score=0.0, gap=config.CHUNK_SCORE_GAP, label="PDF"
                )
                per_source_limit = max(2, chunk_limit // max(len(kept_pdfs), 1))
                for src in kept_pdfs:
                    data = sources[src]
                    capped_docs = data["docs"][:per_source_limit]
                    print(
                        f"Matched PDF: '{src}' (chunk score={data['best_score']:.2f}, "
                        f"chunks={len(capped_docs)}/{len(data['docs'])} cap={per_source_limit})"
                    )
                    matched.append({"file_type": "pdf", "source": src, "results": capped_docs})
 
            # --- Step 4: Append filtered CSV sources ---
            seen = {e["source"] for e in matched}
            for src in csv_sources:
                if src not in seen:
                    print(f"Matched CSV: '{src}' (via summary routing)")
                    matched.append({"file_type": "csv", "source": src, "results": []})
 
            if not matched:
                print("No sources passed score threshold, falling back to raw child search.")
                return [{"file_type": "pdf", "source": None,
                         "results": self.search_child(query, limit=chunk_limit)}]
 
            return matched
 
        except Exception as e:
            print(f"HIERARCHICAL_RETRIEVAL_ERROR: {str(e)}")
            return [{"file_type": "pdf", "source": None, "results": []}]
 
        except Exception as e:
            print(f"HIERARCHICAL_RETRIEVAL_ERROR: {str(e)}")
            return [{"file_type": "pdf", "source": None, "results": []}]
    def search_child(self, query: str, limit: int, source_filter: str = None) -> list:
        try:
            search_filter = None
            if source_filter:
                search_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=source_filter)
                        )
                    ]
                )
 
            results = self.collection.similarity_search(
                query, k=limit, score_threshold=0.3, filter=search_filter
            )
            return results if results else []
 
        except Exception as e:
            print(f"RETRIEVAL_ERROR: {str(e)}")
            return []
    
    def search_child_with_score(self, query: str, limit: int, source_filter: str = None) -> tuple[list, float]:
        try:
            search_filter = None
            if source_filter:
                search_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=source_filter)
                        )
                    ]
                )
            results = self.collection.similarity_search_with_score(
                query, k=limit, score_threshold=0.3, filter=search_filter
            )
            if not results:
                return [], 0.0
            docs = [doc for doc, _ in results]
            best_score = results[0][1]
            return docs, best_score
        except Exception as e:
            print(f"RETRIEVAL_ERROR: {str(e)}")
            return [], 0.0
        
    def retrieve_parent(self, parent_id: str) -> str:
        try:
            parent = self.parent_store.load_content(parent_id=parent_id)
            if not parent:
                return "no parent document"
            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    def retrieve_parent_many(self, parent_ids: list[str]) -> str:
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store.load_content_many(ids)
            if not raw_parents:
                return "No parent documents in database."
 
            return "\n\n".join([
                f"Source: {p.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {p.get('content', '').strip()}"
                for p in raw_parents
            ])
 
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"