
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, dataloader, criterion, device, num_classes, save_dir=None):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            np.array(all_labels),
            np.array(all_probs),
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        roc_auc = 0.0

    metrics = {

        "loss": epoch_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)

        plot_confusion_matrix(
            all_labels,
            all_preds,
            save_path=os.path.join(
                save_dir,
                "confusion_matrix.png"
            )
        )

    return metrics, all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, save_path=None):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(20,18))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        fmt="g"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    if save_path:

        plt.savefig(save_path, dpi=300)

    plt.close()
