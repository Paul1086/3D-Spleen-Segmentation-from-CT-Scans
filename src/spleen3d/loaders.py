"""
This code creates reproducible training, validation, and test data loaders.
Deterministic preprocessing is cached with MONAI PersistentDataset
to reduce repeated loading and resampling time.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

import random
from pathlib import Path
import torch
from monai.data import DataLoader, PersistentDataset
from .data import find_cases
from .transforms import eval_transforms, train_transforms

def split_cases(cases, seed, val_count, test_count):
    cases = list(cases)
    if val_count + test_count >= len(cases):
        raise ValueError("Validation and test sets leave no training cases.")
    random.Random(seed).shuffle(cases)
    test_cases = cases[:test_count]
    val_cases = cases[test_count:test_count + val_count]
    train_cases = cases[test_count + val_count:]
    return train_cases, val_cases, test_cases

def build_loaders(config):
    data_config = config["data"]
    training_config = config["training"]
    cases = find_cases(data_config["root_dir"])
    train_cases, val_cases, test_cases = split_cases(
        cases=cases, seed=int(training_config["seed"]), val_count=int(data_config["val_count"]),
        test_count=int(data_config["test_count"]),
    )
    cache_dir = Path(data_config["cache_dir"])
    num_workers = int(data_config["num_workers"])
    # Cache each split separately to avoid repeating preprocessing
    train_dataset = PersistentDataset(
        data=train_cases, transform=train_transforms(config), cache_dir=cache_dir / "train",
    )
    val_dataset = PersistentDataset(
        data=val_cases, transform=eval_transforms(config), cache_dir=cache_dir / "validation",
    )

    test_dataset = PersistentDataset(
        data=test_cases, transform=eval_transforms(config), cache_dir=cache_dir / "test",
    )
    # Enable faster host-to-GPU transfers when CUDA is available
    loader_options = {
        "num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_options["persistent_workers"] = True


    # Shuffle only the training set
    train_loader = DataLoader(
        train_dataset, batch_size=int(training_config["batch_size"]), shuffle=True,
        **loader_options,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        **loader_options,
    )

    splits = {
        "training": [case["case_id"] for case in train_cases],
        "validation": [case["case_id"] for case in val_cases],
        "test": [case["case_id"] for case in test_cases],
    }

    return train_loader, val_loader, test_loader, splits