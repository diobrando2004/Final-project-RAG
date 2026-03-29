import json


class AIProvider:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder

    def generate_description(self, table_name, snippet_str):
        prompt = (
            "### Task\n"
            "Write ONE sentence describing what data this table contains.\n"
            "Start with 'This table contains'.\n"
            "Do not mention column names. Do not explain yourself.\n\n"
            f"### Sample data\n{snippet_str}\n\n"
            "### Description\nThis table contains"
        )
        output = self.llm(
            prompt,
            max_tokens=80,
            temperature=0.1,
            stop=["\n", "<|im_end|>"]
        )
        text = output['choices'][0]['text'].strip()
        return "This table contains " + text

    def generate_sql(self, sql_prompt):
        output = self.llm(
            sql_prompt,
            max_tokens=150,
            temperature=0,
            stop=[";", "\n\n", "<|im_end|>"]
        )
        return output['choices'][0]['text'].strip()