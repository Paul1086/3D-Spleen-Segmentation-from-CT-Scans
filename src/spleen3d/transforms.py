"""Build preprocessing, augmentation, and evaluation transforms for 3D spleen CT segmentation.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""


from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd,
    EnsureTyped, LoadImaged, Orientationd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, ScaleIntensityRanged,
    Spacingd, SpatialPadd,
)

def preprocessing(config):
    keys = ["image", "label"]
    spacing = tuple(config["data"]["spacing"])

    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(
            keys=keys, axcodes="RAS", labels=None,
        ),
        Spacingd(
            keys=keys, pixdim=spacing, mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),

        CropForegroundd(
            keys=keys, source_key="image",
        ),
    ]

def train_transforms(config, num_samples=None):
    keys = ["image", "label"]
    roi_size = tuple(config["data"]["roi_size"])
    if num_samples is None:
        num_samples = int(config["data"]["num_samples"])
    transforms = preprocessing(config)
    transforms.extend(
        [
            SpatialPadd(
                keys=keys, spatial_size=roi_size,
            ),
            RandCropByPosNegLabeld(
                keys=keys, label_key="label", spatial_size=roi_size, pos=1, neg=1, num_samples=num_samples,
                image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=keys, spatial_axis=0, prob=0.10,
            ),
            RandFlipd(
                keys=keys, spatial_axis=1, prob=0.10,
            ),
            RandFlipd(
                keys=keys, spatial_axis=2, prob=0.10,
            ),
            RandRotate90d(
                keys=keys, spatial_axes=(0, 1), max_k=3, prob=0.10,
            ),
            RandShiftIntensityd(
                keys="image", offsets=0.10, prob=0.50,
            ),
            EnsureTyped(
                keys=keys, track_meta=False,
            ),
        ]
    )
    return Compose(transforms)


def eval_transforms(config):
    transforms = preprocessing(config)
    transforms.append(
        EnsureTyped(
            keys=["image", "label"],
            track_meta=False,
        )
    )
    return Compose(transforms)