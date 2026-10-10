"""Step 2: a simple baseline - TF-IDF + logistic regression.

No neural network here. This gives the number the fine-tuned model has to beat.

Run:  python baseline.py
"""

import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

dataset = load_dataset("dair-ai/emotion", "split")
label_names = dataset["train"].features["label"].names

train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
val_texts, val_labels = dataset["validation"]["text"], dataset["validation"]["label"]
test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]

# --- 1. Turn text into numbers -------------------------------------------
# TF-IDF gives every word a column. A text gets a high number in a column if
# the word appears often in THIS text but rarely in the others, so "the" gets
# a low weight and "furious" a high one.
# ngram_range=(1, 2) also counts pairs of words, so "not happy" is one feature.
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

# fit_transform: learn the vocabulary from the TRAINING texts, then convert them.
X_train = vectorizer.fit_transform(train_texts)
# transform only: validation and test use the vocabulary learned above.
# Fitting on them would let information from the test leak into training.
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

print(f"Vocabulary size: {X_train.shape[1]} features")

# --- 2. Train the classifier ---------------------------------------------
# Logistic regression learns one weight per (word, emotion) pair.
# C controls how freely it may fit the training data (higher = more freely).
model = LogisticRegression(C=10, max_iter=1000)
model.fit(X_train, train_labels)


# --- 3. Evaluate ----------------------------------------------------------
def report(name, X, labels):
    predictions = model.predict(X)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    print(f"{name:<12}accuracy {accuracy:.3f}   macro-F1 {macro_f1:.3f}")
    return predictions


print("\n=== Results ===")
report("train", X_train, train_labels)
report("validation", X_val, val_labels)
test_predictions = report("test", X_test, test_labels)

print("\n=== Per emotion, on the test set ===")
print(classification_report(test_labels, test_predictions, target_names=label_names, digits=3))

# --- 4. Confusion matrix: which emotions get mixed up? --------------------
ConfusionMatrixDisplay.from_predictions(
    test_labels, test_predictions, display_labels=label_names, cmap="Blues"
)
plt.title("Baseline (TF-IDF + logistic regression)")
plt.tight_layout()
plt.savefig("baseline_confusion_matrix.png", dpi=150)
print("Saved baseline_confusion_matrix.png")

# --- 5. Look at some mistakes ---------------------------------------------
print("\n=== Five wrong predictions ===")
shown = 0
for text, true, predicted in zip(test_texts, test_labels, test_predictions):
    if true != predicted:
        print(f"  true: {label_names[true]:<9} predicted: {label_names[predicted]:<9} | {text}")
        shown += 1
        if shown == 5:
            break
