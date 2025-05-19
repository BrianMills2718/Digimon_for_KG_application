import json
import os

# === CONFIG ===
input_file = "hotpot_dev_distractor_v1.json"  # or hotpot_train_v1.1.json if available
output_dir = "Data/HotpotQA"
os.makedirs(output_dir, exist_ok=True)

question_out = os.path.join(output_dir, "Question.json")
corpus_out = os.path.join(output_dir, "Corpus.json")

# === LOAD AND PARSE ===
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []
corpus = []
seen_titles = set()

for i, entry in enumerate(data):
    questions.append({
        "id": f"q{i}",
        "question": entry["question"],
        "answer": entry["answer"],
        "type": entry.get("type", ""),
        "level": entry.get("level", "")
    })

    for title, context in entry["context"]:
        if title not in seen_titles:
            corpus.append({
                "title": title,
                "context": " ".join(context),  # <- FIX: join context list to single string
                "doc_id": len(seen_titles)
            })
            seen_titles.add(title)

# === WRITE OUT ===
with open(question_out, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")

with open(corpus_out, "w", encoding="utf-8") as f:
    for c in corpus:
        f.write(json.dumps(c) + "\n")

print("✅ Preprocessed full HotpotQA -> Question.json + Corpus.json")
