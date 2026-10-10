"""Step 1: look at the data before training anything.

Run:  python explore.py
"""

from collections import Counter

from datasets import disable_progress_bars, load_dataset

disable_progress_bars()  # keeps the printed output clean

# Downloads the dataset the first time (a few MB), then reads it from a local cache.
dataset = load_dataset("dair-ai/emotion", "split")

# The dataset stores labels as numbers (0, 1, 2, ...). This list translates
# a number back into its name, e.g. label_names[1] -> "joy".
label_names = dataset["train"].features["label"].names

print("=== Splits ===")
for split_name, split in dataset.items():
    print(f"{split_name:<12}{len(split):>7} examples")

print("\n=== Labels ===")
for number, name in enumerate(label_names):
    print(f"{number} = {name}")

print("\n=== How often does each label appear in the training set? ===")
counts = Counter(dataset["train"]["label"])
total = len(dataset["train"])
for number, name in enumerate(label_names):
    share = counts[number] / total
    bar = "#" * round(share * 50)
    print(f"{name:<10}{counts[number]:>6}  {share:>6.1%}  {bar}")

print("\n=== Three examples per label ===")
for number, name in enumerate(label_names):
    print(f"\n[{name}]")
    examples = dataset["train"].filter(lambda row: row["label"] == number).select(range(3))
    for row in examples:
        print("  -", row["text"])

lengths = [len(text.split()) for text in dataset["train"]["text"]]
print("\n=== Text length (in words) ===")
print(f"shortest {min(lengths)}, average {sum(lengths) / len(lengths):.1f}, longest {max(lengths)}")
