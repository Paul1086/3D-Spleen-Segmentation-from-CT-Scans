"""
The script performs full-volume inference, computes 3D metrics, and saves
representative 2D slices for use in the project repository.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
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

def amp_context(enabled):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16,
        )
    return nullcontext()

def save_figure(
    sample_number, case_id, image, label, prediction, dice, iou, output_path,
):
    image = image[0, 0].cpu().numpy()
    label = label[0, 0].cpu().numpy()
    prediction = prediction[0, 0].cpu().numpy()
    slice_index = int(label.sum(axis=(0, 1)).argmax())
    ct_slice = image[:, :, slice_index]
    label_slice = label[:, :, slice_index]
    prediction_slice = prediction[:, :, slice_index]
    figure, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"CT slice {slice_index}")
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground truth")
    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(prediction_slice, cmap="Blues", alpha=0.4)
    axes[2].set_title("Prediction")
    axes[3].imshow(ct_slice, cmap="gray")

    if label_slice.max() > 0:
        axes[3].contour(label_slice, levels=[0.5], colors="lime", linewidths=2,
        )
    if prediction_slice.max() > 0:
        axes[3].contour(prediction_slice, levels=[0.5], colors="magenta", linewidths=2, linestyles="--",
        )
    legend = [
        Line2D([0], [0], color="lime", linewidth=2, label="Ground truth",
        ),
        Line2D([0], [0], color="magenta", linewidth=2, linestyle="--", label="Prediction",
        ),
    ]
    axes[3].legend( handles=legend, loc="lower right", fontsize=8,
    )

    axes[3].set_title(
        f"Full-volume metrics\n"
        f"3D Dice = {dice:.3f} | 3D IoU = {iou:.3f}")

    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Sample Result {sample_number}", fontsize=16, fontweight="bold",
    )
    figure.text(0.5, 0.91, f"Test case: {case_id}", ha="center", fontsize=11,
    )
    figure.text(0.5, 0.02,
        "Representative 2D axial slice shown; Dice and IoU were "
        "computed over the entire 3D CT volume.",
        ha="center", fontsize=10,
    )
    figure.tight_layout(rect=[0, 0.06, 1, 0.88])
    figure.savefig(output_path, dpi=200, bbox_inches="tight",
    )
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = (
        bool(config["training"]["amp"])
        and device.type == "cuda"
    )
    output_dir = PROJECT_ROOT / "results/github_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader, _ = build_loaders(config)
    model = build_model(config).to(device)
    # Load the trained checkpoint and disable training-specific behavior
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for sample_number, batch in enumerate(
            tqdm(test_loader, desc="Creating sample figures"),
            start=1,
        ):
            case_id = batch["case_id"][0]
            image = batch["image"].to(device)
            label = batch["label"].to(device) > 0.5
            with amp_context(use_amp):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=tuple(config["data"]["roi_size"]),
                    sw_batch_size=int(
                        config["inference"]["sw_batch_size"]
                    ),
                    predictor=model,
                    overlap=float(
                        config["inference"]["overlap"]
                    ),
                )
            prediction = logits_to_mask(logits.float())

            # Metrics are calculated over the full 3D volume, not the displayed slice
            dice, iou, precision, recall = segmentation_metrics(prediction, label,
            )
            dice = float(dice.item())
            iou = float(iou.item())
            precision = float(precision.item())
            recall = float(recall.item())
            output_path = (
                output_dir
                / f"sample_result_{sample_number}_{case_id}.png"
            )
            save_figure(
                sample_number=sample_number, case_id=case_id, image=image, label=label, prediction=prediction,
                dice=dice, iou=iou, output_path=output_path,
            )

            # Save one summary row for each held-out test volume
            rows.append(
                {
                    "sample": sample_number, "case_id": case_id,
                    "3d_dice": dice, "3d_iou": iou, "precision": precision,"recall": recall, "figure": output_path.name,
                }
            )

    with open(
        output_dir / "sample_results.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAll held-out test cases")
    print("-" * 70)

    for row in rows:
        print(
            f"Sample Result {row['sample']} | "
            f"{row['case_id']:12s} | "
            f"3D Dice {row['3d_dice']:.4f} | "
            f"3D IoU {row['3d_iou']:.4f}"
        )
    print("-" * 70)
    print("Figures saved to:", output_dir)

if __name__ == "__main__":
    main()