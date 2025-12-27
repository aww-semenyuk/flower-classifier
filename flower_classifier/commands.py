import fire

from flower_classifier.download_data import main as download_data
from flower_classifier.preprocess_data import main as preprocess_data


def main():
    fire.Fire({"download_data": download_data, "preprocess_data": preprocess_data})


if __name__ == "__main__":
    main()
