"""
Runs every model listed in config.MODEL_PATHS through the FULL validation
set (not a single image) and produces a side-by-side metrics table —
accuracy, precision, recall, F1, ROC AUC for each of the 5 architectures.
This is the actual results table for your proposal, not a spot check.

Usage (from repo root, after cloning):
    python compare_all_metrics.py

Saves:
    outputs/model_comparison.csv   — the table, for your report
    outputs/model_comparison.png   — bar chart of accuracy per model
"""
import os
import csv

import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt

from config import AVAILABLE_MODELS, MODEL_PATHS, OUTPUT_DIR
from dataset import val_loader, CLASS_NAMES
from evaluate import evaluate_model


def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    criterion = nn.CrossEntropyLoss()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for key, checkpoint_path in MODEL_PATHS.items():
        model_name = AVAILABLE_MODELS[key]
        print("\n" + "=" * 60)
        print(f"Evaluating {key.upper()} ({model_name})")
        print("=" * 60)

        model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

        metrics, _, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(CLASS_NAMES),
            save_dir=None,  # skip per-model confusion matrix here, keep this script fast
        )

        all_results[key] = metrics

        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1_score']:.4f}")
        print(f"ROC AUC   : {metrics['roc_auc']:.4f}")

        # free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()

    # ---- Print final table ----
    print("\n\n" + "=" * 90)
    print("FINAL COMPARISON — ALL 5 MODELS")
    print("=" * 90)
    header = f"{'Model':10s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1 Score':>10s} {'ROC AUC':>10s}"
    print(header)
    print("-" * 90)
    for key, m in all_results.items():
        print(f"{key:10s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}")

    # ---- Save CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"])
        for key, m in all_results.items():
            writer.writerow([key, m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["roc_auc"]])
    print(f"\nSaved table to {csv_path}")

    # ---- Save bar chart ----
    fig, ax = plt.subplots(figsize=(9, 5))
    models = list(all_results.keys())
    accs = [all_results[k]["accuracy"] for k in models]
    bars = ax.bar(models, accs, color="#4C72B0")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Model Comparison — Crop Disease Classification")
    ax.set_ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}",
                 ha="center", fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved chart to {png_path}")

    return all_results


if __name__ == "__main__":
    run_all()
