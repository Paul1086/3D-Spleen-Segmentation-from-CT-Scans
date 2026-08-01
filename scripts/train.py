"""
Train and evaluate the 3D spleen segmentation model.
The script supports mixed-precision training, checkpoint recovery,
sliding-window validation, and final evaluation on the held-out test set.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from tqdm.auto import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spleen3d.loaders import build_loaders
from spleen3d.metrics import dice_iou, logits_to_mask
from spleen3d.model import build_model


# Run inference on overlapping 3D patches and reconstruct the full volume
def mixed_precision_context(enabled):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16,
        )
    return nullcontext()

@torch.no_grad()
def evaluate(model, loader, device, roi_size, sw_batch_size, overlap, use_amp,
):
    model.eval()
    dice_scores = []
    iou_scores = []
    for batch in tqdm(
        loader, desc="Evaluation", leave=False,
    ):
        images = batch["image"].to(
            device, non_blocking=True,
        )
        labels = batch["label"].to(
            device, non_blocking=True,
        ) > 0.5
        with mixed_precision_context(use_amp):
            logits = sliding_window_inference(
                inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size,
                predictor=model, overlap=overlap,
            )
        predictions = logits_to_mask(logits.float())
        dice, iou = dice_iou(predictions, labels,
        )
        dice_scores.extend(dice.cpu().tolist())
        iou_scores.extend(iou.cpu().tolist())
    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)
    return mean_dice, mean_iou

def save_history(history, path):
    columns = [
        "epoch", "train_loss", "val_dice", "val_iou", "learning_rate",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(history)

def save_checkpoint(
    path, epoch, model, optimizer,
    scheduler, scaler, best_val_dice, best_epoch, history, config,
):
    torch.save(
        {
            "epoch": epoch, 
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "best_val_dice": best_val_dice,
            "best_epoch": best_epoch, "history": history, "config": config,
        },
        path,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
    )

    parser.add_argument(
        "--resume", default=None, help="Checkpoint used to resume training.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    training_config = config["training"]
    inference_config = config["inference"]
    seed = int(training_config["seed"])
    set_determinism(seed=seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (
        bool(training_config["amp"])
        and device.type == "cuda")

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    train_loader, val_loader, test_loader, splits = (build_loaders(config))

    with open(output_dir / "splits.json", "w") as file:
        json.dump(splits, file, indent=2)

    model = build_model(config).to(device)
    loss_function = DiceCELoss(
        include_background=False, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp,
    ) 

    start_epoch = 1
    best_val_dice = -1.0
    best_epoch = 0
    history = []

    # Restore full training state when resuming an interrupted run
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_dice = float(checkpoint["best_val_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        history = checkpoint.get("history", [])
        print(f"Resuming from epoch {start_epoch}.")

    epochs = int(training_config["epochs"])
    val_every = int(training_config["val_every"])
    checkpoint_every = int(training_config["checkpoint_every"])

    roi_size = tuple(config["data"]["roi_size"])
    sw_batch_size = int(inference_config["sw_batch_size"])
    overlap = float(inference_config["overlap"])

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Device:", device)
    print(
        "Training/validation/test volumes:",
        f"{len(splits['training'])}/"
        f"{len(splits['validation'])}/"
        f"{len(splits['test'])}",
    )
    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )
    print("Epochs:", epochs)
    print("Output directory:", output_dir)
    print()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        number_of_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False,
        )

        for batch in progress:
            images = batch["image"].to(device, non_blocking=True,
            )
            labels = batch["label"].long().to(device, non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with mixed_precision_context(use_amp):
                logits = model(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            number_of_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
        mean_train_loss = (running_loss / number_of_steps)
        val_dice = None
        val_iou = None

        # Evaluate on the validation set at the configured interval
        if epoch % val_every == 0:
            val_dice, val_iou = evaluate(model=model, loader=val_loader, device=device,
                roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, use_amp=use_amp,
            )
            scheduler.step(val_dice)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                improved = True
            else:
                improved = False
        else:
            improved = False
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch, "train_loss": mean_train_loss,
                "val_dice": (
                    "" if val_dice is None
                    else val_dice
                ),
                "val_iou": (
                    "" if val_iou is None
                    else val_iou
                ),
                "learning_rate": current_lr,
            })
        save_history(history, output_dir / "history.csv",
        )
        if improved:
            save_checkpoint(
                path=output_dir / "best_model.pt", epoch=epoch,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_val_dice=best_val_dice, best_epoch=best_epoch, history=history, config=config,
            )

        # Save a resumable checkpoint every few epochs
        if (
            epoch % checkpoint_every == 0
            or epoch == epochs
        ):
            save_checkpoint(
                path=output_dir / "last_model.pt", epoch=epoch,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_val_dice=best_val_dice, best_epoch=best_epoch, history=history, config=config,
            )
        if val_dice is None:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss {mean_train_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss {mean_train_loss:.4f} | "
                f"val Dice {val_dice:.4f} | "
                f"val IoU {val_iou:.4f} | "
                f"lr {current_lr:.2e}"
            )
    best_checkpoint_path = output_dir / "best_model.pt"
    if not best_checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was created.")

    # Evaluate the selected best model once on the held-out test set
    best_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_dice, test_iou = evaluate(
        model=model, loader=test_loader, device=device, roi_size=roi_size,
        sw_batch_size=sw_batch_size, overlap=overlap, use_amp=use_amp,
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_dice": best_val_dice, "test_dice": test_dice, "test_iou": test_iou,
    }
    with open(
        output_dir / "final_metrics.json",
        "w",
    ) as file:
        json.dump(final_metrics, file, indent=2)
    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print(
        "Best validation Dice:", f"{best_val_dice:.4f}",
    )
    print(
        "Held-out test Dice:", f"{test_dice:.4f}",
    )
    print(
        "Held-out test IoU:", f"{test_iou:.4f}",
    )
    print(
        "Best checkpoint:", best_checkpoint_path,
    )

if __name__ == "__main__":
    main()