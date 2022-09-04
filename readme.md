# Задача 1

Решение задачи 1 в папке `task1\task1.py` (менять input_array).

# Задача 2

Тренировка: 
- `task2/junior_classification_clean_noisy.ipynb` - задача уровня Junior
- `task2/middle_sound_denoising.ipynb` - задача уровня Middle

Этапы тренировки + точность на val-выборке подробно расписаны внутри ноутбук-файлов.

Натренированные модели для инференса:
- Скачать: https://disk.yandex.ru/d/67LiZOIQGub3CA
- Распаковка внутрь `task2\pretrained_models`

Инференс:
- `python task2/inference_classify.py` - классификация зашумленной/чистой аудиозаписи (добавить путь к входному .npy файлу в main). Результат - в консоль.
- `python task2/inference_denoise.py` - фильтрация от шума (добавить путь к входному npy файлу в main). Результат - аудиофайлы в `task2/out` (по-умолчанию) в папке с названием обработанного .npy файла. (Выход: _noised.flac_ - восстановленный аудиофайл из исходной mel-спектрограммы, _unnoised.flac_ - выход нейросети, _unnoised_median.flac_ - выход нейросети с дополнительной фильтрацией от эффекта "свиста",  _clean.flac_ - оригинальный незашумленный файл, если он найдется (найдется, если делать инференс любого файла из train директория)).

