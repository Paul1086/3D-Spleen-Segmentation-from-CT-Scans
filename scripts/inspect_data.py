"""
Inspect paired CT images and spleen masks, verify their geometry,
and save a representative slice with the annotation overlay.
Author: Sudipta Paul; email: pauls5@rpi.edu
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spleen3d.data import find_cases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    cases = find_cases(config["data"]["root_dir"])
    print(f"Found {len(cases)} paired image/label volumes.\n")
    for case in cases[:3]:
        image_nii = nib.load(case["image"])
        label_nii = nib.load(case["label"])
        image = image_nii.get_fdata()
        label = label_nii.get_fdata()
        print(f"Case: {case['case_id']}")
        print("  Image shape:", image.shape)
        print("  Label shape:", label.shape)
        print("  Voxel spacing:", image_nii.header.get_zooms()[:3])
        print("  Label values:", np.unique(label))
        print("  Spleen voxels:", int((label > 0).sum()))
        print()


        if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-4,
        ):
            raise ValueError(
                f"Affine mismatch in {case['case_id']}")

    sample = cases[0]
    image = nib.load(sample["image"]).get_fdata()
    label = nib.load(sample["label"]).get_fdata()

    slice_index = int(label.sum(axis=(0, 1)).argmax())
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(
        image[:, :, slice_index], cmap="gray", vmin=-100, vmax=300,
        )
    axes[0].set_title(
        f"{sample['case_id']} — CT slice {slice_index}")
    axes[1].imshow(
        image[:, :, slice_index], cmap="gray", vmin=-100, vmax=300,
        )
    axes[1].imshow(
        label[:, :, slice_index], cmap="Reds", alpha=0.4,
        )
    axes[1].set_title("Ground-truth spleen mask")

    for axis in axes:
        axis.axis("off")
        
    figure.tight_layout()
    output_path = output_dir / "data_check.png"
    figure.savefig(output_path, dpi=180)
    plt.show()
    print("Saved figure:", output_path)


if __name__ == "__main__":
    main()
