from collections import Counter

from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion", "split")
names = dataset["train"].features["label"].names

rows = dataset["train"].filter(lambda row: "not happy" in row["text"])
print(len(rows), "training sentences contain 'not happy'")
print(Counter(names[label] for label in rows["label"]))

for row in rows.select(range(min(5, len(rows)))):
    print(names[row["label"]], "|", row["text"])