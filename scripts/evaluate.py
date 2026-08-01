"""
Evaluate the trained 3D U-Net on the held-out test set and save
case-level metrics, segmentation examples, and training plots.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

import argparse
import csv
import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D
from monai.inferers import sliding_window_inference
from tqdm.auto import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spleen3d.loaders import build_loaders
from spleen3d.metrics import logits_to_mask, segmentation_metrics
from spleen3d.model import build_model

# Use mixed precision only when running on CUDA
def amp_context(enabled):
    if enabled:
        return torch.autocast(
            device_type="cuda", dtype=torch.float16,
        )
    return nullcontext()

def save_case_figure(
    case_id, image, label, prediction, dice, iou, output_path,
):
    image = image[0, 0].cpu().numpy()
    label = label[0, 0].cpu().numpy()
    prediction = prediction[0, 0].cpu().numpy()

    # Show the slice containing the largest ground-truth spleen area
    slice_index = int(label.sum(axis=(0, 1)).argmax())
    ct_slice = image[:, :, slice_index]
    label_slice = label[:, :, slice_index]
    prediction_slice = prediction[:, :, slice_index]
    figure, axes = plt.subplots(1, 4, figsize=(17, 5),
    )
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"CT slice {slice_index}")
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.4,
    )
    axes[1].set_title("Ground truth")
    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(
        prediction_slice, cmap="Blues", alpha=0.4,
    )
    axes[2].set_title("Prediction")
    axes[3].imshow(ct_slice, cmap="gray")

    if label_slice.max() > 0:
        axes[3].contour(
            label_slice, levels=[0.5], colors="lime", linewidths=2,
        )

    if prediction_slice.max() > 0:
        axes[3].contour(
            prediction_slice, levels=[0.5], colors="magenta", linewidths=2, linestyles="--",
        )

    legend = [
        Line2D(
            [0], [0], color="lime", linewidth=2, label="Ground truth",
        ),
        Line2D(
            [0], [0], color="magenta", linewidth=2, linestyle="--", label="Prediction",
        ),
    ]

    axes[3].legend(
        handles=legend, loc="lower right", fontsize=8,
    )

    axes[3].set_title(
        f"3D Dice: {dice:.3f}\n"
        f"3D IoU: {iou:.3f}"
    )

    for axis in axes:
        axis.axis("off")
    figure.suptitle(case_id, fontsize=14)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_training_plots(history_path, results_dir):
    history = pd.read_csv(history_path)

    plt.figure(figsize=(7, 5))
    plt.plot(
        history["epoch"],
        history["train_loss"],
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        results_dir / "training_loss.png",
        dpi=200,
    )
    plt.close()

    validation = history.dropna(
        subset=["val_dice"]
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        validation["epoch"],
        validation["val_dice"],
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("Validation performance")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        results_dir / "validation_dice.png",
        dpi=200,
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (
        bool(config["training"]["amp"])
        and device.type == "cuda")
    training_dir = Path(config["output"]["directory"])
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else training_dir / "best_model.pt")
    results_dir = PROJECT_ROOT / "results"
    case_dir = results_dir / "test_cases"

    results_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader, splits = build_loaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    roi_size = tuple(config["data"]["roi_size"])
    sw_batch_size = int(
        config["inference"]["sw_batch_size"]
    )
    overlap = float(
        config["inference"]["overlap"])
    rows = []
    with torch.no_grad():
        for batch in tqdm(
            test_loader, desc="Evaluating test set",
        ):
            case_id = batch["case_id"][0]
            images = batch["image"].to(
                device, non_blocking=True,
            )
            labels = batch["label"].to(
                device, non_blocking=True,
            ) > 0.5
            # Run full-volume inference with overlapping 3D windows
            with amp_context(use_amp):
                logits = sliding_window_inference(
                    inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size,
                    predictor=model, overlap=overlap,
                )
            predictions = logits_to_mask(
                logits.float())
            dice, iou, precision, recall = (
                segmentation_metrics( predictions, labels,
                ))

            dice_value = float(dice.item())
            iou_value = float(iou.item())
            precision_value = float(
                precision.item())
            recall_value = float(recall.item())

            rows.append(
                {
                    "case_id": case_id, "dice": dice_value,
                    "iou": iou_value, "precision": precision_value,
                    "recall": recall_value, "predicted_voxels": int(
                        predictions.sum().item()
                    ),
                    "true_voxels": int(
                        labels.sum().item()
                    ),
                }
            )
            save_case_figure(
                case_id=case_id, image=images, label=labels,
                prediction=predictions, dice=dice_value, iou=iou_value,
                output_path=(
                    case_dir
                    / f"{case_id}_comparison.png"
                ),
            )
    metrics_path = results_dir / "test_metrics.csv"

    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    rows = sorted(
        rows, key=lambda item: item["dice"],
    )
    # Keep the strongest and weakest cases for qualitative review
    best_case = rows[-1]
    worst_case = rows[0]

    shutil.copyfile(
        case_dir
        / f"{best_case['case_id']}_comparison.png",
        results_dir / "best_segmentation.png",
    )

    shutil.copyfile(
        case_dir
        / f"{worst_case['case_id']}_comparison.png",
        results_dir / "worst_segmentation.png",
    )

    mean_dice = sum(row["dice"] for row in rows) / len(rows)
    mean_iou = sum(row["iou"] for row in rows) / len(rows)
    mean_precision = sum(row["precision"] for row in rows) / len(rows)
    mean_recall = sum(row["recall"] for row in rows) / len(rows)

    summary = {
        "checkpoint_epoch": checkpoint["epoch"], "number_of_test_cases": len(rows),
        "mean_dice": mean_dice, "mean_iou": mean_iou, "mean_precision": mean_precision, "mean_recall": mean_recall,
        "best_case": best_case, "worst_case": worst_case, "test_cases": splits["test"],
    }

    with open(
        results_dir / "summary.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=2)

    plt.figure(figsize=(8, 5))
    plt.bar(
        [row["case_id"] for row in rows], [row["dice"] for row in rows],
    )
    plt.axhline(
        mean_dice, linestyle="--", label=f"Mean Dice = {mean_dice:.3f}",
    )
    plt.xlabel("Test case")
    plt.ylabel("Dice")
    plt.title("Held-out test performance")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        results_dir / "test_dice_by_case.png", dpi=200,
    )
    plt.close()

    create_training_plots(
        training_dir / "history.csv", results_dir,
    )

    print("\nTest results")
    print("-" * 66)

    for row in rows:
        print(
            f"{row['case_id']:12s} | "
            f"Dice {row['dice']:.4f} | "
            f"IoU {row['iou']:.4f} | "
            f"Precision {row['precision']:.4f} | "
            f"Recall {row['recall']:.4f}"
        )

    print("-" * 66)
    print(f"Mean Dice:      {mean_dice:.4f}")
    print(f"Mean IoU:       {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print()
    print("Best case:", best_case["case_id"], f"(Dice {best_case['dice']:.4f})",
    )
    print("Worst case:", worst_case["case_id"], f"(Dice {worst_case['dice']:.4f})",
    )
    print("Results saved to:", results_dir)

if __name__ == "__main__":
    main()