"""CLI interface"""

import fire


def download_data():
    from flower_classifier.download_data import main  # noqa: PLC0415

    return main()


def preprocess_data():
    from flower_classifier.preprocess_data import main  # noqa: PLC0415

    return main()


def train():
    from flower_classifier.train import main  # noqa: PLC0415

    return main()


def main():
    fire.Fire(
        {
            "download_data": download_data,
            "preprocess_data": preprocess_data,
            "train": train,
        }
    )


if __name__ == "__main__":
    main()
