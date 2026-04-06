import os
import json
import duckdb
from pathlib import Path
from llama_cpp import Llama
from pydantic import BaseModel, Field

# --- SETUP ---
CSV_FILES = [r"csv to test/customers-100000.csv", r"csv to test/titanic_train.csv"] # List of your CSVs
tableinfo_dir = "metadata_cache"
MODEL_PATH = r"Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"

os.makedirs(tableinfo_dir, exist_ok=True)
conn = duckdb.connect(':memory:')
llm = Llama(
    model_path=str(MODEL_PATH), # Ensure it's a string
    n_ctx=4096,                 # 24k might be too high for a weak machine; 4k is plenty for SQL
    n_gpu_layers=0,
    temperature=0,
    verbose=False
)

class TableInfo(BaseModel):
    table_name: str
    table_summary: str

def get_table_metadata(idx, csv_path, existing_names):
    # 1. Check Cache
    match = list(Path(tableinfo_dir).glob(f"{idx}_*.json"))
    if match:
        with open(match[0], 'r') as f:
            return TableInfo(**json.load(f))

    # 2. If not in cache, ask LLM
    df_snippet = conn.execute(f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 5").df()
    df_str = df_snippet.to_string()

    while True:
        prompt = f"""<|im_start|>system
        Return ONLY JSON with "table_name" and "table_summary". 
        Avoid these names: {list(existing_names)}
        <|im_end|>
        <|im_start|>user
        Data: {df_str}
        <|im_end|>
        <|im_start|>assistant
        {{"""
        
        output = llm(prompt, stop=["}"], max_tokens=150)
        res_text = "{" + output['choices'][0]['text'] + "}"
        
        try:
            data = json.loads(res_text)
            name = data['table_name'].replace(" ", "_") # Ensure no spaces
            
            if name not in existing_names:
                info = TableInfo(table_name=name, table_summary=data['table_summary'])
                # 3. Save to Cache
                out_path = os.path.join(tableinfo_dir, f"{idx}_{name}.json")
                with open(out_path, 'w') as f:
                    json.dump(info.dict(), f)
                return info
        except:
            print("LLM made a mistake, retrying...")

# --- EXECUTION ---
table_names = set()
all_tables = []

for i, path in enumerate(CSV_FILES):
    info = get_table_metadata(i, path, table_names)
    table_names.add(info.table_name)
    all_tables.append(info)
    print(f"Loaded: {info.table_name}")