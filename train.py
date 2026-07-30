
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from config import (
    TRAIN_DIR,
    VAL_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    IMAGE_SIZE,
    MODEL_NAME,
)

from dataset import get_dataloaders
from model import create_crop_model
from evaluate import evaluate_model


def train(dry_run=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    print("Setting up data loaders...")

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        num_classes,
    ) = get_dataloaders()

    print(f"Initializing {MODEL_NAME}...")

    model = create_crop_model(
        MODEL_NAME,
        NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    epochs = 1 if dry_run else EPOCHS

    best_val_acc = 0.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1}/{epochs}")
        print("=" * 60)

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc="Training",
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            if dry_run and batch_idx >= 1:

                print("\nDry Run Finished.")

                break

        if dry_run:

            train_loss = running_loss / (2 * BATCH_SIZE)

        else:

            train_loss = running_loss / len(train_loader.dataset)

        print(f"\nTrain Loss: {train_loss:.4f}")

        print("\nRunning Validation...")

        metrics, _, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=NUM_CLASSES,
            save_dir=OUTPUT_DIR if epoch == epochs - 1 else None,
        )

        print(f"Validation Loss      : {metrics['loss']:.4f}")
        print(f"Validation Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Validation Precision : {metrics['precision']:.4f}")
        print(f"Validation Recall    : {metrics['recall']:.4f}")
        print(f"Validation F1 Score  : {metrics['f1_score']:.4f}")
        print(f"Validation ROC AUC   : {metrics['roc_auc']:.4f}")
                # ==========================================================
        # Save Best Model
        # ==========================================================

        if metrics["accuracy"] > best_val_acc:

            best_val_acc = metrics["accuracy"]

            model_path = os.path.join(
                OUTPUT_DIR,
                f"best_{MODEL_NAME}.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print("\nBest model saved!")
            print(model_path)

    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)

    print(f"Best Validation Accuracy : {best_val_acc:.4f}")

    return model


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Runs only two batches for testing."
    )

    args = parser.parse_args()

    train(dry_run=args.dry_run)


if __name__ == "__main__":

    main()
