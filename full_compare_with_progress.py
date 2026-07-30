import os
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from config import AVAILABLE_MODELS, MODEL_PATHS, OUTPUT_DIR
from dataset import val_loader, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plots_dir = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

results = {}

for key, checkpoint_path in MODEL_PATHS.items():
    model_name = AVAILABLE_MODELS[key]
    print(f"\n{'='*60}\nEvaluating {key} ({model_name})\n{'='*60}")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=key):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = 0.0

    results[key] = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1, "roc_auc": roc_auc,
    }

    print(f"Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}  ROC AUC={roc_auc:.4f}")

    # confusion matrix for this model
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="g")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {key}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{key}.png"), dpi=200)
    plt.close()

    del model
    torch.cuda.empty_cache()

# ---- Final table ----
print("\n\n" + "=" * 90)
print("FINAL COMPARISON — ALL 5 MODELS")
print("=" * 90)
print(f"{'Model':10s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'ROC AUC':>10s}")
print("-" * 90)
for key, m in results.items():
    print(f"{key:10s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}")

# ---- Save CSV ----
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"])
    for key, m in results.items():
        writer.writerow([key, m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["roc_auc"]])
print(f"\nSaved table to {csv_path}")

# ---- Accuracy bar chart ----
plt.figure(figsize=(9, 5))
models = list(results.keys())
accs = [results[k]["accuracy"] * 100 for k in models]
bars = plt.bar(models, accs, color="#4C72B0")
plt.ylabel("Validation Accuracy (%)")
plt.title("Model Comparison — Crop Disease Classification")
plt.ylim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.show()

print(f"\nSaved chart to {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")
print(f"Saved confusion matrices to {plots_dir}/")
