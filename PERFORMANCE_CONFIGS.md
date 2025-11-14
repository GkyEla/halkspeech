# HalkSpeech Performance Configurations

Bu dosya H100 GPU için optimize edilmiş farklı performans senaryolarını içerir.

## 📋 İçindekiler
1. [Maksimum Hız (Turbo Model)](#1-maksimum-hız-turbo-model)
2. [Maksimum Kalite (Large-v3 Full)](#2-maksimum-kalite-large-v3-full)
3. [Dengeli (Distilled Model)](#3-dengeli-distilled-model)
4. [Model İndirme](#4-model-i̇ndirme)
5. [Test & Benchmark](#5-test--benchmark)

---

## 1. Maksimum Hız (Turbo Model) 🚀

**Kullanım Senaryosu:**
- Düşük latency kritik (300-500ms)
- 1000'lerce concurrent request
- 20-30 saniyelik kısa ses kayıtları
- Hız > Kalite

**Model:** `deepdml/faster-whisper-large-v3-turbo-ct2`

### Docker Compose

```yaml
version: "3.8"
services:
  halkspeech:
    image: gokay/halkspeach:latest
    container_name: halkspeach
    ports:
      - "8000:8000"
    volumes:
      - /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub
    environment:
      # Model Settings
      - WHISPER__PRELOAD_MODEL=true

      # Batch Processing (Maksimum Hız)
      - WHISPER__USE_BATCHED_MODE=true
      - WHISPER__BATCH_SIZE=256                      # H100 için optimal
      - WHISPER__BATCH_WINDOW_MS=25                  # Düşük latency (25ms)

      # GPU Settings
      - WHISPER__INFERENCE_DEVICE=cuda
      - WHISPER__DEVICE_INDEX=0
      - WHISPER__COMPUTE_TYPE=float16                # H100 için en hızlı
      - WHISPER__CPU_THREADS=1
      - WHISPER__NUM_WORKERS=1

      # Queue & Concurrency (Yüksek Throughput)
      - WHISPER__MAX_QUEUE_SIZE=4096                 # Binlerce istek için
      - WHISPER__MAX_CONCURRENT_REQUESTS=1000        # Aynı anda 1000 istek

      # Model TTL
      - WHISPER__MODEL_TTL=-1                        # Never unload (low latency)

      # Logging
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Docker Run Komutu

```bash
docker run -d --gpus all --name halkspeach \
  -e WHISPER__PRELOAD_MODEL=true \
  -e WHISPER__USE_BATCHED_MODE=true \
  -e WHISPER__BATCH_SIZE=256 \
  -e WHISPER__BATCH_WINDOW_MS=25 \
  -e WHISPER__INFERENCE_DEVICE=cuda \
  -e WHISPER__COMPUTE_TYPE=float16 \
  -e WHISPER__MAX_QUEUE_SIZE=4096 \
  -e WHISPER__MAX_CONCURRENT_REQUESTS=1000 \
  -e WHISPER__MODEL_TTL=-1 \
  -p 8000:8000 \
  -v /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub \
  gokay/halkspeach:latest
```

### Beklenen Performans

| Metrik | Değer |
|--------|-------|
| **VRAM Kullanımı** | ~2.5-3GB |
| **Latency (20-30s audio)** | 300-500ms |
| **Throughput** | ~1500-2000 req/min |
| **Word Error Rate (WER)** | ~5-7% (Türkçe) |
| **GPU Utilization** | %90-95 |

### Test

```bash
# Model indir
curl "localhost:8000/v1/models/deepdml/faster-whisper-large-v3-turbo-ct2" -X POST

# Test et
time curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test-audio.mp3" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

---

## 2. Maksimum Kalite (Large-v3 Full) 🎯

**Kullanım Senaryosu:**
- Kalite en önemli öncelik
- Latency 600-800ms kabul edilebilir
- Transkripsiyon doğruluğu kritik
- Kalite > Hız

**Model:** `Systran/faster-whisper-large-v3`

### Docker Compose

```yaml
version: "3.8"
services:
  halkspeech:
    image: gokay/halkspeach:latest
    container_name: halkspeach
    ports:
      - "8000:8000"
    volumes:
      - /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub
    environment:
      # Model Settings
      - WHISPER__PRELOAD_MODEL=true

      # Batch Processing (Maksimum Kalite)
      - WHISPER__USE_BATCHED_MODE=true
      - WHISPER__BATCH_SIZE=192                      # Biraz düşür (model daha büyük)
      - WHISPER__BATCH_WINDOW_MS=40                  # Daha fazla batch topla

      # GPU Settings
      - WHISPER__INFERENCE_DEVICE=cuda
      - WHISPER__DEVICE_INDEX=0
      - WHISPER__COMPUTE_TYPE=float16                # En iyi kalite
      - WHISPER__CPU_THREADS=1
      - WHISPER__NUM_WORKERS=1

      # Queue & Concurrency (Orta Throughput)
      - WHISPER__MAX_QUEUE_SIZE=2048
      - WHISPER__MAX_CONCURRENT_REQUESTS=600         # Biraz düşür

      # Model TTL
      - WHISPER__MODEL_TTL=-1                        # Never unload

      # Logging
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Docker Run Komutu

```bash
docker run -d --gpus all --name halkspeach \
  -e WHISPER__PRELOAD_MODEL=true \
  -e WHISPER__USE_BATCHED_MODE=true \
  -e WHISPER__BATCH_SIZE=192 \
  -e WHISPER__BATCH_WINDOW_MS=40 \
  -e WHISPER__INFERENCE_DEVICE=cuda \
  -e WHISPER__COMPUTE_TYPE=float16 \
  -e WHISPER__MAX_QUEUE_SIZE=2048 \
  -e WHISPER__MAX_CONCURRENT_REQUESTS=600 \
  -e WHISPER__MODEL_TTL=-1 \
  -p 8000:8000 \
  -v /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub \
  gokay/halkspeach:latest
```

### Beklenen Performans

| Metrik | Değer |
|--------|-------|
| **VRAM Kullanımı** | ~10-12GB |
| **Latency (20-30s audio)** | 600-800ms |
| **Throughput** | ~1000-1200 req/min |
| **Word Error Rate (WER)** | ~4-5% (Türkçe) ✅ (%15-20 daha iyi) |
| **GPU Utilization** | %85-90 |

### Test

```bash
# Model indir (büyük model, biraz zaman alabilir)
curl "localhost:8000/v1/models/Systran/faster-whisper-large-v3" -X POST

# Test et
time curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test-audio.mp3" \
  -F "model=Systran/faster-whisper-large-v3" \
  -F "response_format=json"
```

---

## 3. Dengeli (Distilled Model) ⚖️

**Kullanım Senaryosu:**
- Hız ve kalite dengesi
- Orta latency (400-600ms)
- İyi WER ama hızlı
- Hız ≈ Kalite

**Model:** `Systran/faster-distil-whisper-large-v3`

### Docker Compose

```yaml
version: "3.8"
services:
  halkspeech:
    image: gokay/halkspeach:latest
    container_name: halkspeach
    ports:
      - "8000:8000"
    volumes:
      - /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub
    environment:
      # Model Settings
      - WHISPER__PRELOAD_MODEL=true

      # Batch Processing (Dengeli)
      - WHISPER__USE_BATCHED_MODE=true
      - WHISPER__BATCH_SIZE=224                      # Ortası
      - WHISPER__BATCH_WINDOW_MS=30                  # Dengeli

      # GPU Settings
      - WHISPER__INFERENCE_DEVICE=cuda
      - WHISPER__DEVICE_INDEX=0
      - WHISPER__COMPUTE_TYPE=float16
      - WHISPER__CPU_THREADS=1
      - WHISPER__NUM_WORKERS=1

      # Queue & Concurrency (Dengeli)
      - WHISPER__MAX_QUEUE_SIZE=3072
      - WHISPER__MAX_CONCURRENT_REQUESTS=800

      # Model TTL
      - WHISPER__MODEL_TTL=-1

      # Logging
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Docker Run Komutu

```bash
docker run -d --gpus all --name halkspeach \
  -e WHISPER__PRELOAD_MODEL=true \
  -e WHISPER__USE_BATCHED_MODE=true \
  -e WHISPER__BATCH_SIZE=224 \
  -e WHISPER__BATCH_WINDOW_MS=30 \
  -e WHISPER__INFERENCE_DEVICE=cuda \
  -e WHISPER__COMPUTE_TYPE=float16 \
  -e WHISPER__MAX_QUEUE_SIZE=3072 \
  -e WHISPER__MAX_CONCURRENT_REQUESTS=800 \
  -e WHISPER__MODEL_TTL=-1 \
  -p 8000:8000 \
  -v /mnt/drive1/models:/home/ubuntu/.cache/huggingface/hub \
  gokay/halkspeach:latest
```

### Beklenen Performans

| Metrik | Değer |
|--------|-------|
| **VRAM Kullanımı** | ~5-6GB |
| **Latency (20-30s audio)** | 400-600ms |
| **Throughput** | ~1300-1500 req/min |
| **Word Error Rate (WER)** | ~5-6% (Türkçe) |
| **GPU Utilization** | %90-93 |

---

## 4. Model İndirme

### Tüm Modelleri İndir

```bash
# Turbo (Hız)
curl "localhost:8000/v1/models/deepdml/faster-whisper-large-v3-turbo-ct2" -X POST

# Large-v3 Full (Kalite)
curl "localhost:8000/v1/models/Systran/faster-whisper-large-v3" -X POST

# Distilled (Dengeli)
curl "localhost:8000/v1/models/Systran/faster-distil-whisper-large-v3" -X POST
```

### İndirme Süresi (H100 + hızlı internet)

| Model | Boyut | İndirme Süresi |
|-------|-------|----------------|
| Turbo | ~1.5GB | ~1-2 dakika |
| Distilled | ~2.5GB | ~2-3 dakika |
| Large-v3 Full | ~3GB | ~3-4 dakika |

---

## 5. Test & Benchmark

### Latency Test (Tek İstek)

```bash
# Her model için latency ölç
for model in \
  "deepdml/faster-whisper-large-v3-turbo-ct2" \
  "Systran/faster-distil-whisper-large-v3" \
  "Systran/faster-whisper-large-v3"
do
  echo "=== Testing $model ==="
  time curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
    -F "file=@test-audio.mp3" \
    -F "model=$model" \
    -F "response_format=json" | jq -r '.text' | head -c 100
  echo -e "\n"
done
```

### Throughput Test (Apache Bench)

```bash
# 1000 istek, 100 concurrent
ab -n 1000 -c 100 \
  -p test-audio.mp3 \
  -T "audio/mpeg" \
  "http://localhost:8000/v1/audio/transcriptions?model=deepdml/faster-whisper-large-v3-turbo-ct2"
```

### GPU Monitoring

```bash
# GPU kullanımını izle
watch -n 1 nvidia-smi

# Detaylı metrics
nvidia-smi dmon -s pucvmet -d 1
```

### Docker Logs

```bash
# Real-time logs
docker logs -f halkspeach

# Son 100 satır
docker logs --tail 100 halkspeach

# Sadece hatalar
docker logs halkspeach 2>&1 | grep ERROR
```

---

## 📊 Model Karşılaştırma Tablosu

| Özellik | Turbo (Hız) | Distilled (Dengeli) | Large-v3 (Kalite) |
|---------|-------------|---------------------|-------------------|
| **Model** | deepdml/turbo | Systran/distilled | Systran/large-v3 |
| **VRAM** | 2.5GB | 5-6GB | 10-12GB |
| **Latency** | 300-500ms ✅ | 400-600ms | 600-800ms |
| **Throughput** | 2000/min ✅ | 1500/min | 1200/min |
| **WER (Türkçe)** | ~5-7% | ~5-6% | ~4-5% ✅ |
| **Batch Size** | 256 | 224 | 192 |
| **Batch Window** | 25ms | 30ms | 40ms |
| **Max Requests** | 1000 | 800 | 600 |
| **Use Case** | Hız kritik | Dengeli | Kalite kritik |

---

## ⚙️ Diğer Optimizasyon Seçenekleri

### Ultra Düşük Latency (Acil Durumlar)

```bash
# Batch window'u minimuma indir
WHISPER__BATCH_WINDOW_MS=10

# Batch size'ı düşür
WHISPER__BATCH_SIZE=128
```

### Ultra Yüksek Throughput (Latency önemsiz)

```bash
# Batch window'u artır
WHISPER__BATCH_WINDOW_MS=100

# Batch size'ı artır
WHISPER__BATCH_SIZE=384

# Queue'yu artır
WHISPER__MAX_QUEUE_SIZE=8192
```

### VRAM Tasarrufu (Quantization)

```bash
# int8 quantization kullan (~50% VRAM tasarrufu)
WHISPER__COMPUTE_TYPE=int8_float16

# Batch size'ı düşür
WHISPER__BATCH_SIZE=128
```

---

## 🎯 Hızlı Başlangıç

### Senaryo 1: "Hız lazım, kalite yeter"
```bash
# Turbo model kullan (yukarıdaki 1. config)
docker run ... -e WHISPER__BATCH_SIZE=256 ...
```

### Senaryo 2: "Kalite çok önemli"
```bash
# Large-v3 Full kullan (yukarıdaki 2. config)
docker run ... -e WHISPER__BATCH_SIZE=192 ...
```

### Senaryo 3: "Ortasını bul"
```bash
# Distilled model kullan (yukarıdaki 3. config)
docker run ... -e WHISPER__BATCH_SIZE=224 ...
```

---

## 📝 Notlar

### ⚠️ Önemli Uyarılar

1. **SAMPLES_PER_SECOND asla değiştirme!**
   - Whisper sadece 16kHz için eğitilmiş
   - Bu değer değişirse transkripsiyonlar bozulur

2. **Batch size çok yüksek → OOM (Out of Memory)**
   - H100 80GB için max ~384-512 (model boyutuna göre)
   - İlk başta önerilen değerlerle başla

3. **Model preload öneriliir**
   - `WHISPER__PRELOAD_MODEL=true` kullan
   - İlk istek daha hızlı gelir

4. **VAD filter otomatik aktif**
   - BatchedInferencePipeline için otomatik `vad_filter=True`
   - Büyük dosyalar için gerekli

### 🔧 Troubleshooting

**Problem:** GPU kullanımı düşük (%30-40)
```bash
# Batch size ve concurrent requests artır
WHISPER__BATCH_SIZE=320
WHISPER__MAX_CONCURRENT_REQUESTS=1200
```

**Problem:** Latency çok yüksek (>1s)
```bash
# Batch window düşür, batch size düşür
WHISPER__BATCH_WINDOW_MS=15
WHISPER__BATCH_SIZE=128
```

**Problem:** OOM (Out of Memory) hatası
```bash
# Batch size düşür veya quantization kullan
WHISPER__BATCH_SIZE=128
WHISPER__COMPUTE_TYPE=int8_float16
```

---

## 📚 Ek Kaynaklar

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Model Card](https://huggingface.co/openai/whisper-large-v3)
- [HalkSpeech Docs](https://github.com/GkyEla/halkspeech)

---

**Son Güncelleme:** 2025-11-14
**GPU:** NVIDIA H100 80GB
**Docker Image:** gokay/halkspeach:latest
