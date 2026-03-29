import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
class Chunker:
    def __init__(self):
            self.__parent_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
                strip_headers=False
            )
            self.__child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHILD_CHUNK_SIZE, 
                chunk_overlap=config.CHILD_CHUNK_OVERLAP
            )
            self.__min_parent_size = config.MIN_PARENT_SIZE
            self.__max_parent_size = config.MAX_PARENT_SIZE
    
    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parents = []
        all_children = []
        for doc_path in sorted(glob.glob(os.path.join(path_dir,"*.md"))):
              path = Path(doc_path)
              parent_chunks, child_chunks = self.create_chunks_single(path)
              all_parents.extend(parent_chunks)
              all_children.extend(child_chunks)
        return all_parents, all_children

    def create_chunks_single(self, md_path):
          doc_path = Path(md_path)
          with open(doc_path,"r", encoding="utf-8") as f:
                parent_chunks = self.__parent_splitter.split_text(f.read())
          merged_parent =  self.__merge_small_parents(parent_chunks)
          split_parents = self.__split_large_parents(merged_parent)
          cleaned_parents = self.__clean_small_chunks(split_parents)
          all_parents = []
          all_children = []
          self.__create_child_chunks(all_parents, all_children, cleaned_parents, doc_path)
          return all_parents, all_children


    def __merge_small_parents(self, chunks):
          if not chunks:
                return []
          merged = []
          current = None
          for chunk in chunks:
                  if current is None:
                      current = chunk
                  else:
                        current.page_content += "\n\n" + chunk.page_content
                        for k, v in chunk.metadata.items():
                              if k in current.metadata:
                                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                              else:
                                    current.metadata[k] = v
                  if len(current.page_content) >= self.__min_parent_size:
                        merged.append(current)
                        current=None
          if current:
                if merged:
                      merged[-1].page_content += "\n\n" + current.page_content
                      for k, v in current.metadata.items():
                         if k in merged[-1].metadata:
                              merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                         else:
                              merged[-1].metadata[k] = v
                else:
                      merged.append(current)
                        
          return merged
    
    def __split_large_parents(self, chunks):
          split_chunks = []
          for chunk in chunks:
                if len(chunk.page_content) <= self.__max_parent_size:
                      split_chunks.append(chunk)
                else:
                      splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.__max_parent_size,
                        chunk_overlap=config.CHILD_CHUNK_OVERLAP
                        )
                      sub_chunks = splitter.split_documents([chunk])
                      split_chunks.extend(sub_chunks)
          return split_chunks
    
    def __clean_small_chunks(self, chunks):
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned
    
    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for i, parent_chunk in enumerate(parent_chunks):
             parent_id = f"{doc_path.stem}_parent_{i}"
             parent_chunk.metadata.update({"source": str(doc_path.stem), "parent_id": parent_id})

             all_parent_pairs.append((parent_id, parent_chunk))
             all_child_chunks.extend(self.__child_splitter.split_documents([parent_chunk]))
      
"""
def main():
    chunker = Chuncker()
    
    parents, children = chunker.create_chunks()

    print(f"Total parent chunks: {len(parents)}")
    print(f"Total child chunks: {len(children)}")

    output_path = "chunk_output.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== PARENT CHUNKS =====\n\n")

        for parent_id, parent_doc in parents:
            f.write(f"Parent ID: {parent_id}\n")
            f.write(f"Metadata: {parent_doc.metadata}\n")
            f.write("Content:\n")
            f.write(parent_doc.page_content)
            f.write("\n" + "=" * 80 + "\n\n")

        f.write("\n\n===== CHILD CHUNKS =====\n\n")

        for i, child_doc in enumerate(children):
            f.write(f"Child Index: {i}\n")
            f.write(f"Metadata: {child_doc.metadata}\n")
            f.write("Content:\n")
            f.write(child_doc.page_content)
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\nChunks written to {output_path}")


if __name__ == "__main__":
    main()
    """