import pathlib
import tempfile
import zipfile

from datasets import ClassLabel, Dataset, DatasetDict, Image
from omegaconf import DictConfig

from flower_classifier.config import DATA_DIR, get_hydra_cfg


def prepare_splits(cfg: DictConfig) -> None:
    """Unzip raw data, load as dataset and split into train/val/test, save to disk"""

    data_cfg = cfg.data
    split_cfg = data_cfg.split

    zip_path = DATA_DIR / data_cfg.source_zip_filename
    dst_path = DATA_DIR / data_cfg.dataset_output_dir

    seed = split_cfg.seed
    train_size = split_cfg.train_split_size

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zipref:
            zipref.extractall(tmpdir)

        subdirs = [x for x in (pathlib.Path(tmpdir) / zip_path.stem).iterdir() if x.is_dir()]
        labels = []
        paths = []

        for classdir in subdirs:
            class_name = classdir.name
            imgs = [entry for entry in classdir.iterdir() if entry.is_file()]
            paths.extend(imgs)
            labels.extend([class_name] * len(imgs))

        ds = Dataset.from_dict({"image": list(map(str, paths)), "label": labels})
        ds = (ds
            .cast_column("image", Image())
            .cast_column("label", ClassLabel(names=list(set(labels))))
        )  # fmt: skip

        ds_train_testvalid = ds.train_test_split(
            train_size=train_size, stratify_by_column="label", seed=seed
        )

        ds_test_valid = ds_train_testvalid["test"].train_test_split(
            test_size=0.5, stratify_by_column="label", seed=seed
        )

        ds = DatasetDict(
            {
                "train": ds_train_testvalid["train"],
                "test": ds_test_valid["test"],
                "valid": ds_test_valid["train"],
            }
        )

        ds.save_to_disk(DATA_DIR / dst_path)


def main(overrides: list[str] | None = None) -> None:
    cfg = get_hydra_cfg(overrides)
    prepare_splits(cfg)
