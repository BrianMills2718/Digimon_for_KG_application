import json
import os

input_question = "Data/HotpotQA/Question.json"
input_corpus = "Data/HotpotQA/Corpus.json"

output_dir = "Data/HotpotQAsmallest"
os.makedirs(output_dir, exist_ok=True)

# === LOAD FIRST 10 QUESTIONS ===
with open(input_question, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for _, line in zip(range(10), f)]

# === LOAD ALL CORPUS ENTRIES (they're deduplicated already) ===
with open(input_corpus, "r", encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f]

# === WRITE ===
with open(os.path.join(output_dir, "Question.json"), "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")

with open(os.path.join(output_dir, "Corpus.json"), "w", encoding="utf-8") as f:
    for c in corpus:
        f.write(json.dumps(c) + "\n")

print("✅ Created HotpotQAsmallest with 10 queries and full corpus.")
