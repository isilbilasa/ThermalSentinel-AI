import json
import os
from PIL import Image

TXT_DIR = "./cleaned_dataset/review_needed/labels"
IMAGE_DIR = "./cleaned_dataset/review_needed/images"
LOCAL_STORAGE_PREFIX = "/data/local-files/?d=cleaned_dataset/review_needed/images/"

CLASS_NAME = "human"
tasks = []

for txt_file in os.listdir(TXT_DIR):
    if not txt_file.endswith(".txt"):
        continue

    base_name = os.path.splitext(txt_file)[0]
    img_name = f"{base_name}.jpg"  # Uzantınız png ise .png yapın
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    # Gerçek resim boyutlarını alalım
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    result = []
    txt_path = os.path.join(TXT_DIR, txt_file)

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        x_center, y_center, width, height = map(float, parts[1:5])

        # YOLO -> Label Studio Yüzde (%)
        x = (x_center - width / 2) * 100
        y = (y_center - height / 2) * 100
        w = width * 100
        h = height * 100

        result.append({
            "id": f"box_{base_name}_{idx}",
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": img_w,
            "original_height": img_h,
            "image_rotation": 0,
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": 0,
                "rectanglelabels": [CLASS_NAME]
            }
        })

    # Doğrudan 'annotations' içerisine yerleştiriyoruz ki ekran açılır açılmaz onaylı box olarak gelsin
    tasks.append({
        "data": {
            "image": f"{LOCAL_STORAGE_PREFIX}{img_name}"
        },
        "annotations": [
            {
                "result": result
            }
        ]
    })

# Dosyaları 1000'erli olarak dışa aktar
CHUNK_SIZE = 1000
for i in range(0, len(tasks), CHUNK_SIZE):
    chunk = tasks[i:i + CHUNK_SIZE]
    filename = f"fixed_labels_part_{i//CHUNK_SIZE + 1}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunk, f, indent=2)
    print(f"'{filename}' oluşturuldu.")