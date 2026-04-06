import duckdb
from pathlib import Path
from llama_cpp import Llama

# 1. Configuration & Path Setup
# Using raw strings or forward slashes prevents Windows path errors
EXCEL_PATH = r"excel file test\file_example_XLSX_5000.xlsx"
MODEL_PATH = r"qwen_coder/qwen2.5-coder-1.5b_q4_k_m.gguf"

# 2. Initialize LLM (Small model for weak environment)
llm = Llama(
    model_path=str(MODEL_PATH), # Ensure it's a string
    n_ctx=4096,                 # 24k might be too high for a weak machine; 4k is plenty for SQL
    n_gpu_layers=0,
    temperature=0,
    verbose=False
)

# 3. Initialize DuckDB with Excel Extension
conn = duckdb.connect(database=':memory:')
conn.execute("INSTALL excel;")
conn.execute("LOAD excel;")

def run_automated_query(user_question):
    # Detect Schema to help the LLM
    schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_xlsx('{EXCEL_PATH}')").df()
    cols = ", ".join(schema_df['column_name'].tolist())

    # Step A: Generate the Query
    prompt = f"""<|im_start|>system
You are a SQL generator. Write ONLY a DuckDB SQL query.
The table you must use in the FROM clause is exactly: read_xlsx('{EXCEL_PATH}')
Rules:
1. Use ONLY the column names provided below. 
2. If a column name has spaces, wrap it in double quotes (e.g., "First Name").
3. Use the exact table: read_xlsx('{EXCEL_PATH}')
Columns: {cols}
<|im_end|>
<|im_start|>user
{user_question}
<|im_end|>
<|im_start|>assistant
```sql\n"""

    output = llm(prompt, stop=["```"], max_tokens=150)
    generated_sql = output['choices'][0]['text'].strip()
    
    # Clean SQL (removes markdown backticks if the model added them)
    clean_sql = generated_sql.replace('```sql', '').replace('```', '').strip()
    
    print(f"\n[DEBUG] LLM Generated SQL:\n{clean_sql}\n")
    print("-" * 30)

    # Step B: Run the Query and Print Result
    try:
        # We use .df() to get a nice Pandas-style table output
        result_df = conn.execute(clean_sql).df()
        print("[SYSTEM] Query Results:")
        print(result_df)
    except Exception as e:
        print(f"[ERROR] Could not run query: {e}")

# --- TEST ---
if __name__ == "__main__":
    query_task = "how old is Dulce?"
    run_automated_query(query_task)