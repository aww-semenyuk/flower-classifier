"""Module for initial data preprocessing"""

import pathlib
import tempfile
import zipfile

from datasets import ClassLabel, Dataset, DatasetDict, Image
from omegaconf import DictConfig

from flower_classifier.config import DATA_DIR, get_hydra_cfg


def prepare_splits(cfg: DictConfig) -> None:
    """Unzip raw data, load as dataset and split into train/val/test, save to disk"""

    zip_path = DATA_DIR / cfg.data.source_zip_filename
    dst_path = DATA_DIR / cfg.data.dataset_output_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zipref:
            zipref.extractall(tmpdir)

        # iterate over subdirectories representing each class
        # to create "image path"-"label" dataset
        subdirs = [x for x in (pathlib.Path(tmpdir) / zip_path.stem).iterdir() if x.is_dir()]
        labels = []
        paths = []

        for classdir in subdirs:
            class_name = classdir.name
            imgs = [entry for entry in classdir.iterdir() if entry.is_file()]
            paths.extend(imgs)
            labels.extend([class_name] * len(imgs))

        ds = Dataset.from_dict({"image": list(map(str, paths)), "label": labels})

        # cast paths to images and string labels to integers
        ds = ds.cast_column("image", Image())
        ds = ds.cast_column("label", ClassLabel(names=list(set(labels))))

        # split dataset into train / test-val
        ds_train_testval = ds.train_test_split(
            train_size=cfg.data.split.train_split_size,
            stratify_by_column="label",
            seed=cfg.data.split.seed,
        )

        # split test and val equally
        ds_test_val = ds_train_testval["test"].train_test_split(
            test_size=0.5, stratify_by_column="label", seed=cfg.data.split.seed
        )

        ds = DatasetDict(
            {
                "train": ds_train_testval["train"],
                "val": ds_test_val["train"],
                "test": ds_test_val["test"],
            }
        )

        ds.save_to_disk(DATA_DIR / dst_path)


def main(overrides: list[str] | None = None) -> None:
    cfg = get_hydra_cfg(overrides)
    prepare_splits(cfg)
