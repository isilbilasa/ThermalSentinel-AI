import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

#yollar
MODEL_V1_PATH = BASE_DIR / "models" / "baseline_best.pt"
MODEL_V2_PATH = BASE_DIR / "models" / "full_dataset_best.pt"

RAW_TRAIN_IMAGES = BASE_DIR / "dataset_thermal_camera" / "train" / "images"
RAW_TRAIN_LABELS = BASE_DIR / "dataset_thermal_camera" / "train" / "labels"

OUTPUT_DIR = BASE_DIR / "cleaned_dataset"
VERIFIED_DIR = OUTPUT_DIR / "verified"
REVİEW_DIR = OUTPUT_DIR / "review_needed"

#klasörleri oluştur
(VERIFIED_DIR / "images").mkdir(parents=True, exist_ok=True)
(VERIFIED_DIR / "labels").mkdir(parents=True, exist_ok=True)
(REVİEW_DIR / "images").mkdir(parents=True, exist_ok=True)
(REVİEW_DIR / "labels").mkdir(parents=True, exist_ok=True)

#modelleri yükle 
model_v1 = YOLO(MODEL_V1_PATH)
model_v2 = YOLO(MODEL_V2_PATH)

images = [p for p in RAW_TRAIN_IMAGES.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
print(f"Toplam {len(images)} adet görsel çapraz kontrolden geçiriliyor...")

stats = {"verified": 0, "review": 0}

for img_path in images:
    lbl_path = RAW_TRAIN_LABELS / (img_path.stem + ".txt")

    #iki modelden de tahmin al 
    res_v1 = model_v1.predict(source=img_path, conf=0.50, verbose=False)[0]
    res_v2 = model_v2.predict(source= img_path, conf=0.50, verbose= False)[0]

    count_v1 = len(res_v1.boxes)
    count_v2 = len(res_v2.boxes)

    #kriter: iki model de aynı sayıda insan tespit ettiyse ve güven skorları yüksekse
    if count_v1 == count_v2 and count_v1 > 0:
        target_dir = VERIFIED_DIR
        stats["verified"] += 1
    else:
        #tespitte çelişki var(biri insan buldu diğeri bulamadı veya sayı uymuyor)
        target_dir = REVİEW_DIR
        stats["review"] += 1

    # Görsel ve mevcut etiketi kopyala
    shutil.copy(img_path, target_dir / "images" / img_path.name)
    if lbl_path.exists():
        shutil.copy(lbl_path, target_dir / "labels" / lbl_path.name)

print("\n=== Çapraz Kontrol Tamamlandı ===")
print(f"Iki Modelin de Onayladığı Temiz Veriler: {stats['verified']} adet")
print(f"Çelişkili / Şüpheli Veriler (Manuel İnceleme): {stats['review']} adet")