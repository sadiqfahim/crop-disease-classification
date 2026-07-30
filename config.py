
import os

# ==========================================================
# Base Directory
# ==========================================================
BASE_DIR = "/kaggle/working/crop-disease-classification"

# ==========================================================
# Dataset Paths
# ==========================================================
DATASET_ROOT = "/kaggle/input/datasets/alimransonet/plant-disease-dataset/Dataset_Final_V2_Split"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# ==========================================================
# Output Directory
# ==========================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Training Parameters
# ==========================================================
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# Number of Classes
NUM_CLASSES = 51

# ==========================================================
# Available Models
# ==========================================================
AVAILABLE_MODELS = {
    "vit": "vit_base_patch16_224",
    "deit": "deit_base_patch16_224",
    "pit": "pit_b_224",
    "swin": "swin_base_patch4_window7_224",
    "caformer": "caformer_b36.sail_in22k_ft_in1k",
}

# ==========================================================
# Select Model
# ==========================================================
CURRENT_MODEL = "vit"

MODEL_NAME = AVAILABLE_MODELS[CURRENT_MODEL]

# ==========================================================
# Device
# ==========================================================
DEVICE = "cuda"

# ==========================================================
# Saved ViT Model
# ==========================================================
VIT_MODEL_PATH = "/kaggle/input/models/mdfarhansadiqfahim/vit-base-patch16-224-crop-disease-classifier/pytorch/default/1/best_vit_base_patch16_224.pth"
