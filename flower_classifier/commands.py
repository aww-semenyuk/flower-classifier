import fire

from flower_classifier.download_data import main as download_data


def main():
    fire.Fire({"download_data": download_data})


if __name__ == "__main__":
    main()
