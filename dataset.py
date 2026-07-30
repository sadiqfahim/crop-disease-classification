
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
)

# ==========================================================
# Image Transforms
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# Datasets
# ==========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

# ==========================================================
# Class Information
# ==========================================================

CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


def get_dataloaders():
    return (
        train_loader,
        val_loader,
        test_loader,
        CLASS_NAMES,
        NUM_CLASSES
    )


if __name__ == "__main__":

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")
    print(f"Number of Classes : {NUM_CLASSES}")

    print("\nFirst 10 Classes:\n")

    for c in CLASS_NAMES[:10]:
        print(c)
