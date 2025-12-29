# flower-classifier

## Project overview

### Purpose

Проект представляет собой готовый пайплайн для решения задачи классификации
изображений при помощи нейросетевой модели с применением современных подходов и
инструментов для обучения, управления окружением, кодом и данными

### Dataset

[соревнование](https://aiplanet.com/challenges/61/data-sprint-25-flower-recognition-61/overview/about)
от aiplanet (ранее DPhi) (также доступно на
[Kaggle](https://www.kaggle.com/datasets/imsparsh/flowers-dataset))

**Характеристики:**

- Обучающая выборка: 2746 изображений формата jpeg в 5 подпапках
  (соответствующих классам): daisy (501), rose (497), tulip (607), dandelion
  (646), sunflower (495)
- Разрешение и соотношение сторон варьируются

### Split and Evaluation

Исходная выборка разбивается на train/validation/test с фиксированным seed в
соотношении 75/12.5/12.5 со стратификацией, управлять долей выделяемых под train
данных можно через конфиг `data.yaml`.

Оценка моделей производится на основе метрик **accuracy** (основная) и
**macro-averaged f1-score** (дополнительно).

### Models

По умолчанию доступно 2 модели:

- Бейзлайн: самописная сверточная сеть (CNN), определена в модуле `models.py`.
  Конфиг: `model/cnn.yaml`
- Основная: Vision Transformer via hugging face's transformers API (по умолчанию
  `google/vit-base-patch16-224`, можно изменить в конфиге). Доступно как
  дообучение всей архитектуры, так и вариант с заморозкой всех весов кроме
  classifier head. Конфиг: `model/vit.yaml`

Для добавления новой модели необходимо завести под нее конфиг в `model/`, а
также определить инициализацию transform'а в `utils.py` и самой модели в
`lightning_modules.py`

### Logging

При обучении моделей метрики, гиперпараметры и чекпоинты логируются в `mlflow`,
предполагается наличие развернутого сервиса по адресу http://127.0.0.1:8080
(можно изменить в конфиге `logging.yaml`)

## Working with project

Управление окружением и зависимостями в проекте ведется при помощи `uv`. Перед
начало работы убедитесь, что `uv`
[установлен](https://docs.astral.sh/uv/getting-started/installation/).
Склонируйте репозиторий и перейдите в корень проекта, все команды ниже
выполняются из корня проета.

### Setup

Установка зависимостей:

```sh
uv sync
```

Установка pre-commit хуков и запуск проверки на всех файлах:

```sh
uv run pre-commit install && uv run pre-commit run -a
```

### Train

Для того чтобы скачать и предобработать данные для дальнейшего обучения
необходимо выполнить следующие команды:

```sh
uv run python -m flower_classifier.commands download_data
```

```sh
uv run python -m flower_classifier.commands preprocess_data
```

Обучение моделей реализовано посредством вызова команды `train` с передачей
названия `model` конфига (по умолчанию `model`=`cnn` aka baseline):

- тренировка baseline модели:

```sh
uv run python -m flower_classifier.commands train
```

- тренировка модели Vision Transformer:

```sh
uv run python -m flower_classifier.commands train --overrides='["model=vit"]'
```

**Примечания**:

- чувствительные к модели параметры (такие как кол-во эпох, батч-сайз, learning
  rate) можно изменить в конфиге модели или переопределить во время вызова CLI.
  Например:

```sh
uv run python -m flower_classifier.commands train --overrides='["model=cnn", "model.epochs=10", "model.learning_rate=1e-3"]'
```

- не зависящие от модели параметры обучения (например, num_workers для
  dataloader'ов) управляются через конфиг `train.yaml`

- параметры обучения, метрики и чекпоинты сохраняются в `mlflow`
