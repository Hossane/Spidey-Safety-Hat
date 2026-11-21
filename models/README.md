# Models

Model files are **not** committed (see `.gitignore`). Each teammate downloads them here.

| File | What it is | Where to get it |
|------|-----------|-----------------|
| `yamnet.tflite` | YAMNet sound classifier (521 classes) | Kaggle Models → `google/yamnet` → TFLite variant |
| `yamnet_class_map.csv` | Index → sound name | Same YAMNet page / TensorFlow models repo (`research/audioset/yamnet`) |
| `efficientdet_lite0.tflite` | Object detector (80 COCO classes) | Kaggle Models → `tensorflow/efficientdet` → TFLite `lite0-detection` |
| `coco_labels.txt` | Index → object name | Bundled in the EfficientDet-Lite0 model metadata, or the COCO 2017 label list |

Expected layout:

```
models/
├── yamnet.tflite
├── yamnet_class_map.csv
├── efficientdet_lite0.tflite
└── coco_labels.txt
```
