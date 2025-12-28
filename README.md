# flower-classifier

```
uv sync
uv run pre-commit install
uv run pre-commit run -a
uv run python -m flower_classifier.commands download_data
uv run python -m flower_classifier.commands preprocess_data
uv run python -m flower_classifier.commands train
uv run python -m flower_classifier.commands train --overrides='["model=vit"]'
```
