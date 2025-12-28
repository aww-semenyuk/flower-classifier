"""Module for global constansts and config helpers"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data"


def get_hydra_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        return compose("config", overrides=overrides or [])
