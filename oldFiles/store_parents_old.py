import re
import json
import shutil
import config
from pathlib import Path
from typing import List, Dict

class ParentStore:
    __store_path: Path
    def __init__(self, store_path= config.PARENT_STORE_PATH):
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)


    def save(self, parent_id:str, content: str, metadata: dict)->None:
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=True, indent=2),
            encoding="utf-8"
        )

    def save_multiple(self, parents: list)->None:
        for parent_id, doc in parents:
            self.save(parent_id=parent_id, content=doc.page_content, metadata=doc.metadata)

    def load(self, parent_id:str)-> dict:
        file_path = self.__store_path / (parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json")
        return json.loads(file_path.read_text(encoding="utf-8"))
    
    def load_content(self, parent_id:str)-> dict:
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"]
        }
    
    def load_content_many(self, parent_ids:list[str]) ->list[Dict]:
        unique_ids = set(parent_ids)
        results = [self.load_content(pid) for pid in unique_ids]
        return [r for r in results if r is not None]