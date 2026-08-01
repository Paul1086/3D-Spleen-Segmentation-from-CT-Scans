"""
Run a one-batch sanity check for the 3D segmentation pipeline.
The script loads one spleen CT patch, performs a forward and backward
pass, and verifies the model output shape and foreground content.
Author: Sudipta Paul; Email: pauls5@rpi.edu
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spleen3d.data import find_cases
from spleen3d.model import build_model
from spleen3d.transforms import train_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    set_determinism(seed=int(config["training"]["seed"]))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    cases = find_cases(config["data"]["root_dir"])

    dataset = Dataset(
        data=[cases[0]],
        transform=train_transforms(config, num_samples=1,
        ),
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
    )
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].long().to(device)
    model = build_model(config).to(device)
    loss_function = DiceCELoss(
        include_background=False, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0,
    )

    use_amp = (
        bool(config["training"]["amp"])
        and device.type == "cuda")

    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp,
    )

    model.train()
    model.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp,
    ):
        outputs = model(images)
        loss = loss_function(outputs, labels)

    scaler.scale(loss).backward()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)

    print("Device:", device)
    print("Case:", cases[0]["case_id"])
    print("Input shape: ", tuple(images.shape))
    print("Label shape: ", tuple(labels.shape))
    print("Output shape:", tuple(outputs.shape))
    print("Spleen voxels:", int(labels.sum().item()))
    print("Trainable parameters:", f"{parameter_count:,}")
    print("Loss:", f"{loss.item():.6f}")

    if device.type == "cuda":
        memory = torch.cuda.max_memory_allocated() / 1024**3
        print("Peak GPU memory:", f"{memory:.2f} GB")
    if images.ndim != 5:
        raise RuntimeError("The model input is not five-dimensional.")
    if outputs.shape[1] != 2:
        raise RuntimeError("The model must return two output classes.")
    if labels.sum() == 0:
        raise RuntimeError("The sampled patch contains no spleen voxels.")
    print("\nPipeline check passed.")

if __name__ == "__main__":
    main()