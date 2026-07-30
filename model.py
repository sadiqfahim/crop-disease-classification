
import timm
import torch.nn as nn


def create_crop_model(model_name, num_classes, pretrained=True):
    """
    Creates a vision transformer model using timm.
    Supports:
    - ViT
    - DeiT
    - Swin Transformer
    - PiT
    - CAFormer
    """

    print(f"Creating model: {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


if __name__ == "__main__":

    MODELS = [
        "vit_base_patch16_224",
        "deit_base_patch16_224",
        "pit_b_224",
        "swin_base_patch4_window7_224",
        "caformer_b36.sail_in22k_ft_in1k"
    ]

    for m in MODELS:
        try:
            model = create_crop_model(
                m,
                num_classes=51,
                pretrained=False
            )

            total_params = sum(p.numel() for p in model.parameters())

            print("=" * 60)
            print(m)
            print(f"Parameters: {total_params:,}")
            print("=" * 60)

        except Exception as e:
            print(f"{m} -> ERROR")
            print(e)
