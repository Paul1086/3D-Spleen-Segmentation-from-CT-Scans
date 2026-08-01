"""
Load and match spleen CT volumes with their segmentation masks.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

from pathlib import Path

def case_name(path):
    # Remove the NIfTI extension to obtain a consistent case identifier
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem

def nifti_files(folder):
    # Ignore macOS metadata files beginning with "._"
    paths = list(folder.glob("*.nii.gz"))
    paths.extend(folder.glob("*.nii"))
    return sorted(
        path
        for path in paths
        if not path.name.startswith("._"))

def find_cases(data_root):
    root = Path(data_root)
    # Index images and labels by case identifier
    images = {
        case_name(path): path
        for path in nifti_files(root / "imagesTr")}
    labels = {
        case_name(path): path
        for path in nifti_files(root / "labelsTr")}

    # Verify that every image has a corresponding annotation
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))

    if missing_labels:
        raise ValueError(f"Images without labels: {missing_labels}")

    if missing_images:
        raise ValueError(f"Labels without images: {missing_images}")

    return [
        {
            "case_id": name, "image": str(images[name]), "label": str(labels[name]),
        }
        for name in sorted(images)
    ]