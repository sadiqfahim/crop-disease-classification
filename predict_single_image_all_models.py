import os
import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

# ==========================================================
# IMAGE TO TEST
# (You can point this at a specific image file, OR at a class
#  folder — if it's a folder, the first image inside is used
#  automatically instead of crashing.)
# ==========================================================

IMAGE_PATH = "/kaggle/input/datasets/alimransonet/plant-disease-dataset/Dataset_Final_V2_Split/val/Tomato___Spider_mites_Two-spotted_spider_mite/002835d1-c18e-4471-aa6e-8d8c29585e9b___Com.G_SpM_FL 8584_aug2.jpg"

TRAIN_DIR = "/kaggle/input/datasets/alimransonet/plant-disease-dataset/Dataset_Final_V2_Split/train"

# ==========================================================
# TRAINED MODELS
# ==========================================================

MODELS = {

    "ViT": {
        "name": "vit_base_patch16_224",
        "path": "/kaggle/input/models/mdfarhansadiqfahim/vit-base-patch16-224-crop-disease-classifier/pytorch/default/1/best_vit_base_patch16_224.pth",
    },

    "DeiT": {
        "name": "deit_base_patch16_224",
        "path": "/kaggle/input/models/mdfarhansadiqfahim/deit-base-patch16-224-crop-disease-classifier/pytorch/default/1/best_deit_base_patch16_224.pth",
    },

    "PiT": {
        "name": "pit_b_224",
        "path": "/kaggle/input/models/mdfarhansadiqfahim/pit-b-224-crop-disease-classifier/pytorch/default/1/best_pit_b_224.pth",
    },

    "Swin": {
        "name": "swin_base_patch4_window7_224",
        "path": "/kaggle/input/models/mdfarhansadiqfahim/swin-base-patch4-224-crop-disease-classifier/pytorch/default/1/best_swin_base_patch4_window7_224.pth",
    },

    "CAFormer": {
        "name": "caformer_s18",
        "path": "/kaggle/input/models/mdfarhansadiqfahim/caformer-s18-crop-disease-classifier/pytorch/default/1/best_caformer_s18.pth",
    },
}

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = datasets.ImageFolder(TRAIN_DIR).classes

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# LOAD IMAGE (auto-fixes if IMAGE_PATH is a folder, not a file)
# ==========================================================

if os.path.isdir(IMAGE_PATH):
    print(f"IMAGE_PATH is a folder — picking the first image inside it.")
    files = [f for f in sorted(os.listdir(IMAGE_PATH))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise FileNotFoundError(f"No image files found inside: {IMAGE_PATH}")
    IMAGE_PATH = os.path.join(IMAGE_PATH, files[0])
    print(f"Using: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

true_label = os.path.basename(os.path.dirname(IMAGE_PATH))

results = []

# ==========================================================
# RUN PREDICTION
# ==========================================================

for model_name, cfg in MODELS.items():

    print(f"Loading {model_name}...")

    model = timm.create_model(
        cfg["name"],
        pretrained=False,
        num_classes=len(class_names)
    )

    model.load_state_dict(torch.load(cfg["path"], map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

        top5_prob, top5_idx = torch.topk(probs, 5)

    prediction = class_names[pred.item()]

    results.append({

        "Model": model_name,
        "Prediction": prediction,
        "Confidence": confidence.item() * 100,
        "Correct": prediction == true_label,
        "Top5": [
            (class_names[idx.item()], prob.item() * 100)
            for prob, idx in zip(top5_prob[0], top5_idx[0])
        ]

    })

    # free GPU memory before loading the next model
    del model
    torch.cuda.empty_cache()

# ==========================================================
# SHOW IMAGE
# ==========================================================

plt.figure(figsize=(7, 7))
plt.imshow(image)
plt.axis("off")
plt.title(f"True Label\n{true_label}", fontsize=15)
plt.show()

# ==========================================================
# SORT RESULTS
# ==========================================================

results = sorted(results, key=lambda x: x["Confidence"], reverse=True)

print("=" * 120)
print("FINAL MODEL COMPARISON")
print("=" * 120)
print(f"True Label : {true_label}")
print("=" * 120)

print(f"{'Rank':<6}{'Model':<12}{'Prediction':<45}{'Confidence':<15}{'Correct'}")
print("-" * 120)

for i, r in enumerate(results, 1):

    mark = "CORRECT" if r["Correct"] else "WRONG"

    print(f"{i:<6}{r['Model']:<12}{r['Prediction']:<45}{r['Confidence']:.2f}%{'':<7}{mark}")

# ==========================================================
# TOP-5 RESULTS
# ==========================================================

for r in results:

    print("\n")
    print("=" * 120)
    print(r["Model"])
    print("=" * 120)

    print("Prediction :", r["Prediction"])
    print("Confidence :", f"{r['Confidence']:.2f}%")
    print("Correct    :", "YES" if r["Correct"] else "NO")

    print("\nTop-5 Predictions")

    for cls, prob in r["Top5"]:

        print(f"{cls:<55}{prob:.2f}%")

# ==========================================================
# BAR CHART
# ==========================================================

names = [r["Model"] for r in results]
conf = [r["Confidence"] for r in results]

plt.figure(figsize=(9, 5))
plt.bar(names, conf)

for i, v in enumerate(conf):
    plt.text(i, v + 0.2, f"{v:.2f}%", ha='center')

plt.ylim(0, 100)
plt.ylabel("Confidence (%)")
plt.xlabel("Models")
plt.title("Confidence Comparison of Trained Models")
plt.grid(axis='y')

plt.show()

print("\nComparison Completed Successfully.")
