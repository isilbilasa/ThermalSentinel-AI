import shutil
from pathlib import Path

# --- YOLLARI AYARLAYIN ---
EXPORTED_FOLDER = Path("/Users/isilbilasa/Downloads/review_needed_images")  # Zip'ten çıkan klasör
VERIFIED_DIR = Path("/Users/isilbilasa/Desktop/thermal_project /cleaned_dataset/verified")

# Görselleri kopyala
exp_images = EXPORTED_FOLDER / "images"
target_images = VERIFIED_DIR / "images"
target_images.mkdir(parents=True, exist_ok=True)

for img in exp_images.glob("*"):
    if img.is_file():
        shutil.copy2(img, target_images / img.name)

# Etiketleri (labels) kopyala
exp_labels = EXPORTED_FOLDER / "labels"
target_labels = VERIFIED_DIR / "labels"
target_labels.mkdir(parents=True, exist_ok=True)

for lbl in exp_labels.glob("*.txt"):
    if lbl.is_file():
        shutil.copy2(lbl, target_labels / lbl.name)

print("İşlem tamamlandı! Düzeltilen 3.070 görsel ve etiket verified klasörüne eklendi.")