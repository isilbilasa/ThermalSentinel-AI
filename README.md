# ThermalSentinel-AI

**Termal kamera görüntülerinden gerçek zamanlı insan tespiti, sıcaklık anomali analizi ve Edge-to-Cloud izleme platformu — YOLOv8, ONNX/TensorRT ve Data-Centric AI ile geliştirilmiş bir mühendislik prototipi.**

ThermalSentinel-AI, termal kamera görüntülerinden insan tespiti ve sıcaklık anomalisi analizi yapan, uçtan uca bir Edge-to-Cloud izleme platformudur. Projenin çekirdeğinde Data-Centric AI yaklaşımıyla kurulmuş bir veri kalite pipeline'ı yer alır: gürültülü, ~12.000 görsellik bir termal veri setinden çapraz model doğrulama ve Label Studio destekli yarı-otomatik etiketleme ile "Gold Standard" bir eğitim seti üretilmiş, sonuçta YOLOv8 tabanlı model %95.11 mAP50 ve %94.60 precision değerlerine ulaşmıştır. Edge cihazlarda gerçek zamanlı çalışabilmesi için model PyTorch'tan ONNX ve TensorRT'ye (FP16/INT8) dönüştürülmüş, böylece çıkarım süresi milisaniye seviyesine indirilmiştir. Sadece kutu tespiti yapan YOLO çıktısı, özel geliştirilmiş bir hibrit termal analiz katmanı (CLAHE ön işleme, geometri ve sıcaklık tavanı filtreleri, 90. persentil piksel sıcaklığı) ile gerçek anomali kararına dönüştürülür. Sistem, ham video yerine yalnızca hafif JSON metadata'yı MQTT üzerinden buluta ileterek bant genişliğinden büyük ölçüde tasarruf sağlar; FastAPI tabanlı bir köprü sunucusu bu akışı WebSocket ile canlı bir web paneline yansıtır. Proje şu anda Python tabanlı, Colab/Tesla T4 üzerinde geliştirilmiş bir prototip niteliğindedir; Docker'laştırma, gerçek edge donanımı (Jetson) testi ve C++ native portu gelecek çalışmalar olarak planlanmaktadır.

---

## İçindekiler

1. [Mimari Genel Bakış](#mimari-genel-bakış)
2. [Veri Pipeline'ı: Data-Centric AI Yaklaşımı](#veri-pipelinei-data-centric-ai-yaklaşımı)
3. [Model Eğitimi (Gold Standard)](#model-eğitimi-gold-standard)
4. [Model Optimizasyonu: ONNX & TensorRT](#model-optimizasyonu-onnx--tensorrt)
5. [Hibrit Termal Analiz Motoru](#hibrit-termal-analiz-motoru)
6. [Edge Video İşleme ve Nesne Takibi](#edge-video-i̇şleme-ve-nesne-takibi)
7. [Edge-to-Cloud İletişim Mimarisi](#edge-to-cloud-i̇letişim-mimarisi)
8. [Proje Yapısı](#proje-yapısı)
9. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
10. [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
11. [Gelecek Çalışmalar](#gelecek-çalışmalar)
12. [Lisans / İletişim](#lisans--i̇letişim)

---

## Mimari Genel Bakış

Sistem, ağır çıkarım işini edge tarafında tutup buluta yalnızca hafif metadata gönderen bir **Python + TensorRT hibrit mimarisi** üzerine kuruludur. Görüntü hiçbir zaman ağ üzerinden taşınmaz; yalnızca tespit ve anomali sonuçları JSON olarak iletilir.

```
┌──────────────────────────────── EDGE ────────────────────────────────┐
│                                                                        │
│  Termal Kamera / Video Kaynağı                                        │
│           │                                                           │
│           ▼                                                           │
│  YOLOv8 (Gold Standard, TensorRT INT8/FP16 engine)                    │
│           │  bounding box'lar                                         │
│           ▼                                                           │
│  HybridThermalSentinel  (CLAHE + geometri filtresi                    │
│                           + piksel sıcaklık tavanı                    │
│                           + %90 persentil sıcaklık)                   │
│           │  anomali kararı                                           │
│           ▼                                                           │
│  ByteTrack (nesne takibi, ID ataması)                                 │
│           │                                                           │
│           ▼                                                           │
│  EdgeVideoMetadataPipeline → JSON metadata (~1-2 KB/kare)              │
│           │                                                           │
│           ▼                                                           │
│  MQTT Publish → broker.emqx.io  (topic: thermal/sentinel/alerts)      │
│                                                                        │
└────────────────────────────────┬──────────────────────────────────────┘
                                  │  MQTT (JSON, hafif)
                                  ▼
┌──────────────────────────────── CLOUD ────────────────────────────────┐
│                                                                        │
│  FastAPI (main.py) — MQTT subscriber                                  │
│           │                                                           │
│           ▼                                                           │
│  WebSocket endpoint (/ws/live-telemetry)                              │
│           │                                                           │
│           ▼                                                           │
│  index.html — Canlı izleme paneli (anomali kartları kırmızı vurgulu)  │
│                                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

Bu tasarımın temel motivasyonu, ham video akışının (~2.5 MB/s) bulut tarafına taşınmasının hem maliyetli hem de gecikmeli olmasıdır. Bunun yerine, ağır çıkarım ve anomali kararı tamamen edge katmanında (TensorRT ile hızlandırılmış) alınır; buluta yalnızca sonuç akar. Bu yaklaşım şu an Python tabanlı bir prototip olarak çalışıyor; C++ native port (DeepStream/ROS2) gelecek çalışmalar arasındadır.

---

## Veri Pipeline'ı: Data-Centric AI Yaklaşımı

Projenin en kritik mühendislik katkısı, model mimarisinden çok **veri kalitesine** odaklanmasıdır. Ham Kaggle termal veri seti (`dataset_thermal_camera/`, train 12.023 / val 58 / test 3.522 görsel) gürültülü etiketler içeriyordu; bu nedenle tek bir modele güvenmek yerine **çapraz model doğrulama** stratejisi uygulandı:

1. **İki bağımsız model** (`baseline_best.pt` ve `full_dataset_best.pt`) aynı görsel seti üzerinde çalıştırılır.
2. `cross_check_pipeline.py`, iki modelin tespit sayılarını karşılaştırır:
   - Uyumlu sonuçlar → `cleaned_dataset/verified/` (12.023 görsel, doğrudan eğitime hazır)
   - Çelişkili/şüpheli sonuçlar → `cleaned_dataset/review_needed/` (3.070 görsel, manuel inceleme gerektirir)
3. Başlangıçta review_needed için beklenen oran %3-5 iken, gerçekte veri setindeki gürültü nedeniyle bu oran **~%25**'e çıkmıştır.
4. `txt_to_labelstudio.py`, review_needed setindeki YOLO `.txt` etiketlerini Label Studio'nun rectangle formatına çevirerek ön-etiketlenmiş `fixed_labels_part_1-4.json` dosyalarını üretir. Bu, etiketleyicinin sıfırdan çizim yapmak yerine yalnızca hataları düzeltmesini sağlar.
5. Manuel düzeltme sonrası `compose_verified_and_review.py`, düzeltilmiş etiketleri tekrar `cleaned_dataset/verified/` içine birleştirerek nihai "Gold Standard" veri setini oluşturur.

Bu döngü sayesinde, tüm veri setinin manuel olarak yeniden etiketlenmesi yerine yalnızca şüpheli alt küme incelenmiş, manuel etiketleme iş yükünde **%90+** oranında bir azalma sağlanmıştır.

---

## Model Eğitimi (Gold Standard)

Temizlenmiş veri seti üzerinde `thermal_gold.yaml` konfigürasyonu ile YOLOv8n modeli eğitilmiştir (`thermal_gold_model_train.ipynb`):

```yaml
# thermal_gold.yaml
path: /path/to/thermal_project

train: cleaned_dataset/verified/images
val: dataset_thermal_camera/val/images
test: dataset_thermal_camera/test/images

names:
  0: human
```

Eğitim sonucunda elde edilen `yolov8_gold_best.pt` modeli, 3.522 test görseli / 8.543 instance üzerinde aşağıdaki sonuçları vermiştir (kaynak: Colab çalışma zamanı çıktı logları):

| Metrik | Değer | Not |
|---|---|---|
| Test mAP50 | 0.9511 | YOLOv8n, 3.522 test görseli / 8.543 instance |
| Test mAP50-95 | 0.6201 | |
| Precision | 0.9460 | |
| Recall | 0.8873 | |

---

## Model Optimizasyonu: ONNX & TensorRT

Edge dağıtımı için model önce ONNX'e, ardından TensorRT motorlarına (FP16 ve INT8) dönüştürülmüştür.

- `onnx_tensor_optimizasyonu.ipynb`: `.pt → .onnx → TensorRT FP16 .engine` dönüşüm zinciri ve benchmark.
- `thermal_calibration.ipynb`: `baseline_images/` (584 görsel) kullanılarak TensorRT INT8 kalibrasyonu ve FP16 vs INT8 karşılaştırması.

Benchmark sonuçları (Tesla T4, Colab ortamı):

| Yapılandırma | Süre | Hız | Not |
|---|---|---|---|
| ONNX Runtime (CPU fallback) | 110.21 ms | 9.1 FPS | CUDA provider bulunamadığı için CPU'da çalıştı |
| TensorRT FP16 (dummy input) | 2.60 ms | 384.2 FPS | ~42x hızlanma (ONNX CPU'ya kıyasla) |
| TensorRT FP16 (gerçekçi, batch=1, NMS dahil) | 4.16 ms | 240.3 FPS | |
| TensorRT INT8 (gerçekçi, batch=1) | 4.04 ms | 247.4 FPS | Batch=1'de sınırlı kazanç (~1.03x); asıl avantaj VRAM/güç tüketiminde |

**Not:** ONNX CPU sonucu, çalışma ortamında CUDA execution provider bulunamaması nedeniyle CPU fallback'e düşmüştür; bu yüzden ONNX-GPU ile doğrudan bir karşılaştırma değildir. INT8'in FP16'ya göre batch=1'deki sınırlı hız kazancı, TensorRT'nin doğruluğu korumak amacıyla bazı katmanları FP16'da bırakan hibrit derleme davranışından kaynaklanır; INT8'in asıl beklenen kazanımı (VRAM ve güç tüketimi tasarrufu) bu projede ölçülmemiş, kavramsal düzeyde kalmıştır.

Üretilen `.onnx` ve `.engine` dosyaları `models/` klasöründe yer almaktadır.

---

## Hibrit Termal Analiz Motoru

YOLOv8'in çıktısı yalnızca "burada bir insan var" bilgisini verir; sıcaklık anomalisi kararını vermez. Bu nedenle `HybridThermalSentinel` sınıfı (`thermal_mask_and_thresholding.ipynb`, INT8 engine versiyonu `thermal_mask_thresholding_engineopt.ipynb`) geliştirilmiştir:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization) ön işleme**: Termal görüntülerdeki kontrast dengesizliklerini gidererek hem tespit hem de sıcaklık okuma kalitesini artırır.
2. **Aspect-ratio (geometri) filtresi**: YOLO'nun bazen gövde dışı nesneleri (örn. sıcak yüzeyler, yansımalar) insan olarak işaretlemesine karşı, tespit kutusunun en-boy oranı insan siluetine uymayan sonuçları eler. Bu, yanlış pozitiflerin anomali zincirine sızmasını engeller.
3. **Piksel sıcaklık tavanı filtresi**: Kutu içindeki piksellerin bir üst sıcaklık sınırını aşan aşırı uç (outlier) değerleri (örn. sensör gürültüsü, arka plandaki sıcak bir nesnenin kutuya taşması) hesaplamadan dışlar.
4. **90. persentil piksel sıcaklığı**: Kutu içindeki ortalama sıcaklık yerine bilinçli olarak **90. persentil** kullanılır. Düz ortalama, kutunun soğuk arka plan piksellerini de içerdiği için gerçek vücut sıcaklığını sulandırır ve anomalileri maskeler; 90. persentil ise kutunun en sıcak bölgesine (genellikle yüz/vücut merkezi) odaklanarak daha temsili ve gürbüz bir sıcaklık okuması sağlar, tek bir aşırı uç pikselin (maksimum değer) kararı domine etmesini de önler.

Bu dört adımın birleşimi, ham YOLO kutusunu güvenilir bir anomali kararına dönüştürür. `test_hybrid_solution.ipynb`, bu mantığı tekil görseller üzerinde hızlıca doğrulamak için kullanılan bir test defteridir.

---

## Edge Video İşleme ve Nesne Takibi

- `edge_video_pipeline.ipynb`: `baseline_images/` içindeki statik görsellerden sentetik bir video üretir ve gerçek zamanlı video-inference FPS ölçümü yapar (INT8 engine ile ~15.9 FPS elde edilmiştir, hedef kamera FPS aralığı olan 9-15 FPS ile karşılaştırılabilir düzeydedir).
- `tracking_anomali_thermal_camera.ipynb`: ByteTrack ile nesne takibi uygulayarak her tespit edilen kişiye kalıcı bir ID atar; anomali tespit edilen kareler CSV log ve snapshot görüntü olarak kaydedilir.

| Metrik | Değer | Not |
|---|---|---|
| Edge video gerçek zamanlı işleme hızı | ~15.9 FPS | INT8 engine, `edge_video_pipeline.ipynb` |
| Kamera hedef FPS aralığı | 9-15 FPS | Karşılaştırma referansı |

---

## Edge-to-Cloud İletişim Mimarisi

`mqtt_websocket_communication.ipynb` içindeki `EdgeVideoMetadataPipeline`, her karenin tespit ve anomali sonucunu JSON'a dönüştürüp MQTT üzerinden yayınlar:

```json
{
  "device_id": "thermal_edge_node_01",
  "timestamp": "2026-08-17 10:32:15",
  "frame_id": 142,
  "total_humans": 3,
  "anomaly_count": 1,
  "detections": [
    {
      "detection_id": "f142_det_1",
      "temperature_c": 38.4,
      "is_anomaly": true,
      "bbox": [120, 84, 210, 260]
    }
  ]
}
```

- **Broker**: `broker.emqx.io` (public MQTT broker)
- **Topic**: `thermal/sentinel/alerts`
- **Payload boyutu**: ~1-2 KB/kare (ham video ~2.5 MB/s ile karşılaştırıldığında teorik olarak büyük oranda tasarruf sağlar; bu, ölçülmüş bir ağ testi değil, veri boyutu üzerinden yapılmış bir hesaplamadır)

`main.py` (FastAPI), bu MQTT topic'ini dinleyen bir backend sunucusudur; gelen her mesajı `/ws/live-telemetry` WebSocket endpoint'i üzerinden bağlı istemcilere anlık olarak iletir. `index.html`, bu WebSocket'e bağlanan saf HTML/JS bir panel olup anomali tespit edilen kartları kırmızı renkte vurgular.

---

## Proje Yapısı

| Dosya / Klasör | Açıklama |
|---|---|
| `main.py` | FastAPI backend; MQTT'den (`broker.emqx.io`, topic `thermal/sentinel/alerts`) gelen JSON telemetriyi dinleyip WebSocket (`/ws/live-telemetry`) ile web paneline canlı yayınlar. |
| `cross_check_pipeline.py` | İki YOLOv8 modelini (`baseline_best.pt`, `full_dataset_best.pt`) aynı veri seti üzerinde çalıştırıp tespit sayılarını karşılaştırır; uyumlu görselleri `verified`'a, çelişkilileri `review_needed`'a ayırır. |
| `compose_verified_and_review.py` | Label Studio'da düzeltilmiş etiketleri tekrar `cleaned_dataset/verified`'a birleştirir. |
| `txt_to_labelstudio.py` | YOLO `.txt` etiketlerini Label Studio rectangle formatına çevirip `fixed_labels_part_1-4.json` olarak dışa aktarır. |
| `thermal_gold.yaml` | Gold model eğitimi için Ultralytics veri config dosyası (train=verified, val/test=dataset_thermal_camera, tek sınıf: `human`). |
| `index.html` | Saf HTML/JS canlı telemetri izleme paneli. |
| `baseline_images_model.ipynb` | Baseline (v1) YOLOv8n modelinin 500 görsellik alt kümeyle eğitimi. |
| `thermal_gold_model_train.ipynb` | Gold Standard (v3) modelin `thermal_gold.yaml` ile eğitimi ve test seti değerlendirmesi. |
| `onnx_tensor_optimizasyonu.ipynb` | `.pt → ONNX → TensorRT FP16 .engine` dönüşümü ve benchmark. |
| `thermal_calibration.ipynb` | TensorRT INT8 kalibrasyonu (`baseline_images/` ile) ve FP16 vs INT8 benchmark. |
| `thermal_mask_and_thresholding.ipynb` | `HybridThermalSentinel` sınıfının ana tanımı ve `.pt` model ile toplu test. |
| `thermal_mask_thresholding_engineopt.ipynb` | Aynı hibrit mantık, INT8 `.engine` ile optimize edilmiş versiyon. |
| `test_hybrid_solution.ipynb` | `HybridThermalSentinel` için tekil görsel hızlı test defteri. |
| `edge_video_pipeline.ipynb` | Statik görsellerden sentetik video üretimi + gerçek zamanlı video-inference FPS testi. |
| `tracking_anomali_thermal_camera.ipynb` | ByteTrack tabanlı nesne takibi ve anomali CSV/snapshot loglaması. |
| `mqtt_websocket_communication.ipynb` | Frame bazlı JSON metadata üretip MQTT ile yayınlayan edge pipeline. |
| `baseline_images/` | 584 ham termal görsel; INT8 kalibrasyon ve video/pipeline testleri için referans set. |
| `cleaned_dataset/verified/` | Çapraz doğrulamadan geçmiş temiz veri (12.023 görsel). |
| `cleaned_dataset/review_needed/` | Şüpheli/düzeltilmiş veri (3.070 görsel). |
| `dataset_thermal_camera/` | Orijinal ham veri seti (train 12.023 / val 58 / test 3.522). |
| `models/` | `baseline_best.pt`, `full_dataset_best.pt`, `yolov8_gold_best.pt`, `yolov8_gold_best.onnx`, `yolov8_gold_best.fp16.onnx`, `yolov8_gold_best_int8.engine`. |

---

## Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

> Not: `requirements.txt` içindeki `onnxruntime-gpu` ve `tensorrt` paketleri CUDA destekli bir GPU gerektirir (proje Tesla T4 üzerinde geliştirilmiştir). Sadece FastAPI backend + MQTT/WebSocket katmanını (`main.py`) çalıştırmak için bu iki paket şart değildir. Notebook'lar Google Colab (GPU runtime) üzerinde çalıştırılacak şekilde tasarlanmıştır.

### Genel Çalıştırma Adımları

1. **Veri hazırlığı**: `cross_check_pipeline.py` ile ham veri setini çapraz doğrulamadan geçirin, ardından `txt_to_labelstudio.py` → Label Studio'da manuel düzeltme → `compose_verified_and_review.py` döngüsünü tamamlayın.
2. **Model eğitimi**: `thermal_gold_model_train.ipynb` notebook'unu GPU'lu bir ortamda (Colab önerilir) çalıştırarak `yolov8_gold_best.pt` dosyasını üretin.
3. **Optimizasyon**: `onnx_tensor_optimizasyonu.ipynb` ve `thermal_calibration.ipynb` notebook'larını sırasıyla çalıştırarak ONNX, TensorRT FP16 ve INT8 motorlarını üretin.
4. **Hibrit analiz ve test**: `thermal_mask_and_thresholding.ipynb` veya `thermal_mask_thresholding_engineopt.ipynb` ile `HybridThermalSentinel` katmanını test edin.
5. **Edge video ve takip**: `edge_video_pipeline.ipynb` ve `tracking_anomali_thermal_camera.ipynb` ile gerçek zamanlı video işleme ve takip senaryolarını çalıştırın.
6. **Edge-to-Cloud yayın**: `mqtt_websocket_communication.ipynb`'ı çalıştırarak edge tarafında MQTT publish akışını başlatın.
7. **Backend'i başlatın**:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

8. **Web panelini açın**: `index.html` dosyasını doğrudan tarayıcıda açarak `/ws/live-telemetry` WebSocket bağlantısı üzerinden canlı telemetriyi izleyin.

---

## Bilinen Sınırlamalar

- ByteTrack takibi, sentetik/slayt-birleştirme videosunda (gerçek akıcı kamera görüntüsü olmadığı için) kararsız davranıyor; gerçek video akışında doğrulanmadı.
- INT8 kuantizasyonun FP16'ya göre hız kazancı batch=1 senaryosunda sınırlı kaldı (TensorRT'nin bazı katmanları doğruluğu korumak için FP16'da bıraktığı hibrit derleme nedeniyle); VRAM/güç tasarrufu iddiaları ölçülmedi, kavramsal düzeyde kalıyor.
- Bant genişliği tasarrufu iddiası teorik bir hesaplamadır; gerçek bir ağ ölçümüne dayanmaz.

---

## Gelecek Çalışmalar

- Docker / docker-compose ile paketleme
- MLflow (veya benzeri) ile deney takibi
- REST API endpoint'leri (`POST /predict/image`, `POST /predict/batch`)
- Telegram/Slack üzerinden anlık anomali bildirimi
- ROI / yasaklı bölge poligon filtresi
- Web dashboard'a zaman serisi grafik / ısı haritası ekleme
- NVIDIA Jetson gibi gerçek edge donanımında dockerize edilmiş dağıtım + systemd servisi
- Python + TensorRT hibrit mimariden tam C++ native porta (DeepStream/ROS2) geçiş

---

## Lisans / İletişim

Bu proje MIT License (veya tercih ettiğiniz lisans) altında paylaşılabilir. Sorularınız ve katkı önerileriniz için lütfen bir issue açın.
