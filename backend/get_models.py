import json
import re

class AIProvider:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
    def clean_text_for_summary(self, text: str) -> str:
        text = re.sub(r'\.\s*\.\s*\.', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text[500:4500].strip() if len(text) > 1000 else text.strip()
    def generate_description_pdf(self, text):
        self.llm.reset()
        cleaned_sample = self.clean_text_for_summary(text)
        prompt = (
            "### TASK: You are a document classifier. "
            "Identify the main subject of the text below.\n"
            "### RULES:\n"
            "1. Ignore all navigation menus, page numbers, and table of contents.\n"
            "2. Do not repeat the text.\n"
            "3. Answer in one direct sentence starting with 'This document covers...'\n\n"
            f"### INPUT TEXT:\n{cleaned_sample}\n### END OF INPUT\n\n"
            "### SUMMARY:\nThis document covers"
        )
        output = self.llm(prompt, max_tokens=80, temperature=0.2)
        summary = output['choices'][0]['text'].strip()
        return "This document covers " + summary

    def generate_description(self, table_name, snippet_str):
        self.llm.reset()
        prompt = (
            "### Task\n"
            f"The table name is '{table_name}'.\n"
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
            stop=[";", "\n\n", "<|im_end|>", "<eos>"]
        )
        return output['choices'][0]['text'].strip()