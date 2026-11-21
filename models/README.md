# Models

All model files live in this `models/` folder. Run the commands below **from inside `models/`**.

| File | What it is | Size |
|------|-----------|------|
| `yamnet.tflite` | YAMNet sound classifier (521 sounds), classification version | ~4.1 MB |
| `yamnet_class_map.csv` | Index → sound name for YAMNet | ~14 KB |
| `efficientdet_lite0.tflite` | Object detector (80 COCO objects) | *added later* |
| `coco_labels.txt` | Index → object name for EfficientDet | *added later* |

> If these files are already committed to the repo, `git pull` gets them and you can skip the downloads.

---

## YAMNet (sound)

Use the **classification** version: fixed input of 15,600 samples (~1 s at 16 kHz), output of 521 scores.

### Windows (PowerShell)

> In PowerShell, `curl` is a shortcut for `Invoke-WebRequest` and doesn't accept `-L`. Use `curl.exe`.

```powershell
cd models
curl.exe -L -o yamnet.tflite "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
curl.exe -L -o yamnet_class_map.csv "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
```

Or with PowerShell's own command:

```powershell
Invoke-WebRequest "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite" -OutFile yamnet.tflite
Invoke-WebRequest "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv" -OutFile yamnet_class_map.csv
```

### Mac / Linux

```bash
cd models
curl -L -o yamnet.tflite "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
curl -L -o yamnet_class_map.csv "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
```

### If the model link fails (browser download)

1. Go to **kaggle.com/models/google/yamnet**
2. Open the **TFLite** tab and pick the **classification-tflite** variant
3. Download it, unzip if needed, rename the file to `yamnet.tflite`, and put it in `models/`
4. For the CSV, open the raw GitHub link above in a browser and save it as `yamnet_class_map.csv`

### Check the downloads

| Check | Windows (PowerShell) | Mac / Linux | Expected |
|-------|---------------------|-------------|----------|
| Sizes | `ls` (see the Length column) | `ls -lh` | `yamnet.tflite` ≈ 4,100,000 bytes; CSV ≈ 14,000 bytes |
| CSV header | `Get-Content yamnet_class_map.csv -TotalCount 3` | `head -3 yamnet_class_map.csv` | Starts with `index,mid,display_name` |
| CSV rows | `(Get-Content yamnet_class_map.csv).Count` | `wc -l yamnet_class_map.csv` | 522 (1 header + 521 sounds) |

If `yamnet.tflite` is only a few KB, you downloaded an HTML error page. Use the browser method instead.

### Check the model loads

From the **project root** (not `models/`), with the venv active:

```bash
python check_model.py
```

Expected last two lines:

```
[15600] <class 'numpy.float32'>
[  1 521]
```

---

## EfficientDet-Lite0 (video)

*Instructions to be added once the video side is set up.*

---

## Copying the models to the Pi

From the project root on your laptop:

```bash
scp models/yamnet.tflite models/yamnet_class_map.csv qnxuser@<pi-ip>:~/models/
```
