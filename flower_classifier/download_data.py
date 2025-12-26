from dvc.repo import Repo

from flower_classifier.config import PROJECT_DIR


def download_data() -> None:
    """Pull data from DVC remote"""
    repo = Repo(str(PROJECT_DIR))
    repo.pull()


def main():
    download_data()
