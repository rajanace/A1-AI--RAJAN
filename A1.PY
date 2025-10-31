import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")
df.rename(columns={"condition": "target"}, inplace=True)
df.dropna(inplace=True)

# Shuffle and split manually
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(0.8 * len(df))
train_data = df[:split_index]
test_data = df[split_index:]

# -------------------------------
# Algorithm 1: Simple Decision Tree (Manual)
# -------------------------------
def simple_tree(row):
    if row["cp"] >= 2 and row["chol"] > 240:
        return 1
    elif row["thal"] == 2 and row["slope"] == 2:
        return 1
    elif row["oldpeak"] > 2.5 and row["exang"] == 1:
        return 1
    else:
        return 0

test_data["pred_tree"] = test_data.apply(simple_tree, axis=1)

# -------------------------------
# Algorithm 2: Manual Rule-Based
# -------------------------------
def manual_rules(row):
    if row["age"] > 55 and row["chol"] > 250 and row["cp"] >= 1:
        return 1
    elif row["thal"] == 2 and row["ca"] >= 2:
        return 1
    elif row["oldpeak"] > 3 and row["slope"] == 2:
        return 1
    else:
        return 0

test_data["pred_manual"] = test_data.apply(manual_rules, axis=1)

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(predictions, actual, title):
    tp = sum((predictions == 1) & (actual == 1))
    tn = sum((predictions == 0) & (actual == 0))
    fp = sum((predictions == 1) & (actual == 0))
    fn = sum((predictions == 0) & (actual == 1))
    accuracy = (tp + tn) / len(actual)

    print(f"{title} Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")

    # Plot confusion matrix
    matrix = np.array([[tp, fp], [fn, tn]])
    plt.figure(figsize=(4,3))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    plt.show()

# Evaluate both models
evaluate(test_data["pred_tree"], test_data["target"], "Simple Decision Tree")
evaluate(test_data["pred_manual"], test_data["target"], "Manual Rule-Based")
