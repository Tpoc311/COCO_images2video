# COCO Video Demo

Небольшой скрипт для сборки `.mp4`-видео из набора изображений и COCO-аннотаций.  
Для каждого кадра скрипт рисует bounding boxes и подписи классов, после чего сохраняет результат в видеофайл.

### Что делает

- читает COCO-аннотацию из `lbl/COCO_annotation.json`
- загружает изображения из указанной папки
- отрисовывает рамки объектов и названия классов
- собирает последовательность кадров в видео `.mp4`

### Требования

- Python 3.8+
- OpenCV
- NumPy
- `pycocotools`

Установка зависимостей:

```
pip install opencv-python numpy pycocotools
```

### Использование:

```
python make_video_coco.py \
  --folder_path ./dataset \
  --fps 25 \
  --classes_list person car dog \
  --colors_list red blue green
```
