from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from getEmbeddingFunction import get_embedding_function
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pypdf import PdfReader
import re

DATA_PATH = "F:\project3\documents"
CHROMA_PATH = "chroma_index"

def main():
    documents = load_documents()
    for doc in documents:
        doc.page_content = normalize_text(doc.page_content)
    chunks = split_documents2(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
#CHROMA
def add_to_chroma(chunks: list[Document]):
    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    chunks_with_ids = get_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"Added {len(chunks)} chunks to Chroma at '{CHROMA_PATH}'")
    else:
        print("No new documents to add")


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["/n","/n/n",".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
DEBUG = True

        
def split_documents2(documents):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=[".", "!", "?", " ", ""],
    )

    all_chunks = []

    pdf_groups = {}
    for doc in documents:
        src = doc.metadata.get("source")
        if src not in pdf_groups:
            pdf_groups[src] = []
        pdf_groups[src].append(doc)


    for src, pages in pdf_groups.items():

        if DEBUG:
            print("\n===============================")
            print("Processing PDF:", src)
            print("===============================\n")

        pages = sorted(pages, key=lambda d: d.metadata.get("page", 0))
        full_text = "\n".join(p.page_content for p in pages)

        reader = None
        bookmarks = []
        try:
            reader = PdfReader(src)
        except:
            if DEBUG:
                print(" Could not load PDF for bookmarks")

        if reader:
            try:
                outlines = reader.outline
                bookmarks = extract_bookmarks_recursive(outlines, reader)
            except:
                pass

        if DEBUG and bookmarks:
            print(f" Found {len(bookmarks)} real bookmarks:")
            for b in bookmarks:
                print(f"  • L{b['level']}  p{b['page']}  {b['title']}")

        sections = []
        if bookmarks:
            sections = split_by_bookmarks(pages, bookmarks)
        else:
            sections = split_by_heuristics(pages)

        if not sections:
            sections = [{"title": None, "content": full_text, "metadata": pages[0].metadata}]

        for sec in sections:
            sec_doc = Document(page_content=sec["content"], metadata=sec["metadata"])
            chunks = recursive_splitter.split_documents([sec_doc])

            for c in chunks:
                md = dict(c.metadata)
                if sec["title"]:
                    md["section_title"] = sec["title"]
                if "section_level" in sec:
                    md["section_level"] = sec["section_level"]
                c.metadata = md
                all_chunks.append(c)

    return all_chunks

def extract_bookmarks_recursive(outlines, reader, level=1, results=None):
    if results is None:
        results = []

    for item in outlines:
        if isinstance(item, list):
            extract_bookmarks_recursive(item, reader, level + 1, results)

        else:
            try:
                page_num = reader.get_destination_page_number(item)
                print(item.title.strip())
                results.append({
                    "title": item.title.strip(),
                    "page": page_num,
                    "level": level
                })
            except:
                continue

    return sorted(results, key=lambda x: x["page"])

import re

def split_by_bookmarks(pages, bookmarks):
    sections = []

    joined_text = "\n".join(p.page_content for p in pages)

    page_offsets = []
    pos = 0
    for p in pages:
        page_offsets.append(pos)
        pos += len(p.page_content) + 1

    for i, bm in enumerate(bookmarks):
        title = bm["title"]
        start_page = bm["page"]

        page_text = pages[start_page].page_content

        title_pattern = title_to_regex(title)

        match = re.search(title_pattern, page_text, re.IGNORECASE)

        if match:
            print("start true")
            start_offset = page_offsets[start_page] + match.start()
        else:
            print("start false")
            start_offset = page_offsets[start_page]

        if i + 1 < len(bookmarks):
            next_bm = bookmarks[i + 1]
            next_page = next_bm["page"]
            next_page_text = pages[next_page].page_content

            next_title_pattern = title_to_regex(next_bm["title"])
            match_next = re.search(next_title_pattern, next_page_text, re.IGNORECASE)

            if match_next:
                print("end true")
                end_offset = page_offsets[next_page] + match_next.start()
            else:
                print("end false")
                end_offset = page_offsets[next_page]
        else:
            end_offset = len(joined_text)

        md = dict(pages[start_page].metadata)
        md["start_page"] = start_page
        md["end_page"] = (
            bookmarks[i + 1]["page"]
            if i + 1 < len(bookmarks)
            else len(pages) - 1
        )

        sec_text = joined_text[start_offset:end_offset].strip()

        sections.append({
            "title": title,
            "content": sec_text,
            "metadata": md,
            "section_level": bm["level"]
        })
    #for section in sections:
        #print(section)
        #print("---------------------------")
    return sections

def title_to_regex(title: str) -> str:
    words = title.strip().split()
    return r"\s+".join(map(re.escape, words))


def split_by_heuristics(pages):
    text = "\n".join(p.page_content for p in pages)
    lines = text.splitlines()

    numbered = re.compile(r'^\s*\d+(\.\d+)*\s+.+$')
    caps = re.compile(r'^[A-Z0-9\s,\-\(\)\'"]{4,}$')
    titlecase = re.compile(r'^[A-Z][A-Za-z0-9\'\-\s]{3,80}$')

    headings = []

    for i, line in enumerate(lines):
        s = line.strip()

        if not s:
            continue

        # Must have blank line before OR after (prevents inline headings like "1200")
        before_blank = (i > 0 and not lines[i-1].strip())
        after_blank = (i+1 < len(lines) and not lines[i+1].strip())

        # Must satisfy at least 1 formatting rule
        looks_like_heading = (
            numbered.match(s) or
            caps.match(s) or
            titlecase.match(s)
        )

        # Require blank before/after to be safe
        if looks_like_heading and (before_blank or after_blank):
            # Extra filter: skip pure numbers (fixes "1200")
            if s.isdigit():
                continue
            headings.append(i)

    if not headings:
        md = dict(pages[0].metadata)
        md["start_page"] = 0
        md["end_page"] = len(pages) - 1
        return [{
            "title": None,
            "content": text,
            "metadata": md,
            "section_level": 1
        }]

    headings.append(len(lines))
    headings = sorted(set(headings))


    page_line_ranges = []
    line_pos = 0
    for p in pages:
        num_lines = len(p.page_content.splitlines())
        start = line_pos
        end = line_pos + num_lines
        page_line_ranges.append((start, end))  # (line_start, line_end)
        line_pos += num_lines


    sections = []
    for idx in range(len(headings)-1):
        start = headings[idx]
        end = headings[idx + 1]

        title = lines[start].strip()
        content = "\n".join(lines[start:end]).strip()


        contributing_pages = []
        for page_num, (ps, pe) in enumerate(page_line_ranges):
            if not (pe <= start or ps >= end):
                contributing_pages.append(page_num)

        if contributing_pages:
            start_page = min(contributing_pages)
            end_page = max(contributing_pages)
        else:
            start_page = end_page = 0

        md = dict(pages[0].metadata)
        md["start_page"] = start_page
        md["end_page"] = end_page



        sections.append({
            "title": title,
            "content": content,
            "metadata": md,
            "section_level": 1  # unknown hierarchy
        })

    return sections

def normalize_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = text.strip()
    return text

def get_chunk_ids1(chunks):
    last_page_id = None
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if(chunk.metadata.get("start_page")):
            page = chunk.metadata.get("start_page", 0)
        else: page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{source}:{page}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks
def get_chunk_ids(chunks):
    # Track per-file global chunk index
    file_chunk_counter = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if source not in file_chunk_counter:
            file_chunk_counter[source] = 0

        chunk_index = file_chunk_counter[source]

        chunk_id = f"{source}:{chunk_index}"
        chunk.metadata["id"] = chunk_id

        file_chunk_counter[source] += 1

    return chunks
if __name__ == "__main__":
    main()