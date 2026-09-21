# YOLO Model Inspector

Yapay zeka modelleriyle çalışan profesyonel bir YOLO etiketleme (annotation) aracı. Üç paralel ekranda modelleri karşılaştırabilir, etiketleri yönetebilir ve eğitim veri setlerini hazırlayabilirsiniz.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Kullanım Kılavuzu](#kullanım-kılavuzu)
- [Kısayollar](#kısayollar)
- [Klasör Yapısı](#klasör-yapısı)
- [Sorun Çözme](#sorun-çözme)

---

## ✨ Özellikler

### Temel Özellikler
- **3'e Kadar Paralel Ekran**: 1, 2 veya 3 model/veri setini yan yana karşılaştırma
- **Maksimum %5000 Zoom**: 50x'e kadar büyütme + kaydırma (Pan) desteği
- **Zoom & Konum Kilidi**: Ekranlar arasında geçişte aynı görünümü korur
- **Otomatik Model Çıkarımı**: Seçili YOLO modelleriyle otomatik tahmin yapma
- **Manuel Etiketleme**: İnteraktif kutular çizerek manuel annotasyon
- **Undo/Redo Sistemi**: Kutu değişikliklerini ve silme işlemlerini geri alın

### Gelişmiş Özellikler
- **Dinamik Sınıf Filtreleme**: Sınıf türlerine göre kutuları göster/gizle
- **Hata Sınıflandırması**: 4 kategoriye göre otomatik klasörlendirme
- **Batch İşleme**: Birden fazla resme aynı anda etiket uygulama
- **YAML Desteği**: `data.yaml` dosyasından sınıf bilgisi otomatik yükleme
- **Fluent Onay Sistemi**: Değişiklikleri kaydet/iptal hızlı aksiyonları

---

## 📦 Gereksinimler

### Sistem Gereksinimleri
- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows, macOS, Linux
- **RAM**: Minimum 8GB (12GB+ önerilir)
- **GPU** (opsiyonel): CUDA desteği hızlı çıkarım için

### Python Kütüphaneleri
```bash
PyQt5>=5.15.0           # GUI Framework
ultralytics>=8.0.0      # YOLO modelleri
PyYAML>=6.0             # YAML işleme
opencv-python>=4.5.0    # İmaj işleme (opsiyonel)
```

---

## 🚀 Kurulum

### 1. Python Ortamı Hazırlama

**Sanal ortam oluşturma (önerilir):**
```bash
python -m venv yolo_env
```

**Ortamı aktifleştirme:**
- **Windows:**
  ```bash
  yolo_env\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source yolo_env/bin/activate
  ```

### 2. Kütüphaneleri Yükleme

```bash
pip install --upgrade pip
pip install PyQt5 ultralytics PyYAML opencv-python
```

**Yalnızca GPU kullanıyorsanız (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

### 3. Uygulamayı Çalıştırma

```bash
python yolo_inspector.py
```

Veya Windows'ta doğrudan çift tıklayın (`.bat` dosyası oluşturmak isteyebilirsiniz):
```batch
@echo off
python yolo_inspector.py
pause
```

---

## 🎯 Hızlı Başlangıç

### Adım 1: Klasör Seçme
1. **Panel 1, 2 veya 3 için "📁 Klasör" butonuna tıklayın**
2. Görüntülerin bulunduğu klasörü seçin
3. Uygulama otomatik olarak resim ve etiketleri yükler

### Adım 2: Model Yükleme (İsteğe bağlı)
1. **"📦 Model" butonuna tıklayın**
2. Önceden eğitilmiş bir YOLO modeli seçin (`.pt` dosyası)
3. Model seçilirse otomatik tahmin çalışır

### Adım 3: Etiketleri Görüntüleme
1. **"📌 Box Karşılaştırma"** = Mevcut `.txt` etiketlerini göster
2. **"🧠 Model Karşılaştırma"** = Modelin tahminlerini göster
3. Sınıf filtreleriyle (K, Y, V tuşları) kutulara filtre uygulayın

### Adım 4: Etiket Düzenleme veya Onaylama
- **Kutuları sürükleyin/yeniden boyutlandırın**
- **Manuel kutu oluşturmak için görüntüye tıklayın** (gelecek sürümde)
- **Enter** ile kaydedin veya **Esc** ile iptal edin

---

## 📚 Kullanım Kılavuzu

### Ana Arayüz

```
┌─────────────────────────────────────────────────────────┐
│  🔍 Zoom: [========] %100  🔒 Sabit  🖥️ 3 Ekran      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Panel 1    │  │   Panel 2    │  │   Panel 3    │  │
│  │  (Klasör 1)  │  │  (Klasör 2)  │  │  (Klasör 3)  │  │
│  │              │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  ◀ Önceki  [===============] %50  Sonraki ▶  1/250    │
├─────────────────────────────────────────────────────────┤
│  ❌ Hata 1  ⚠️ Hata 2  ✅ Doğru  ❔ Arada Kalanlar    │
└─────────────────────────────────────────────────────────┘
```

### Zoom Kontrolü

**Slider ile Zoom:**
- Zoom kaydırıcısını sürükleyerek %10 ile %5000 arasında ayarlayın
- Değer canlı olarak panellerde senkronize edilir

**Zoom Kilidi (Lock):**
- Paneller arasında geçerken görünümü sabitle
- 🔓 Serbest = Her resimde %100'e sıfırla
- 🔒 Sabit = Son zoom seviyesini koru

**Mouse Tekerleği:**
- Kaydır: Tekerleği yukarı/aşağı çevirin (yakında)
- Pan (Kaydırma): Sağ tıkla + sürükle (ScrollHandDrag modu)

### Kutu Yönetimi

#### Mevcut Kutuları Düzenleme
1. **Kutuyu seçmek:** Kutuya tıklayın (mavi çerçeve görünür)
2. **Kutunun tamamını taşımak:** Kutu içinde sürükleyin
3. **Boyutunu değiştirmek:**
   - Köşeleri sürükleyin (çapraz yeniden boyutlandırma)
   - Kenarları sürükleyin (tek yön yeniden boyutlandırma)
4. **Kutuyu silmek:** Delete veya Backspace tuşu

#### Kutuları Kaydetme
- **Enter tuşu** = Düzenleri kaydet ve `.txt` dosyasını güncelle
- **Esc tuşu** = Son kaydedilmiş duruma dön

#### Undo/Redo
- **Ctrl+Z** = Son kutu işlemini geri al
- **Ctrl+Shift+Z** = İşlemi yeniden yap

### Sınıf Filtreleme

Konvansiyonel Sınıflar (örnek):
- **K tuşu** = "kapakli" kutularını göster/gizle
- **Y tuşu** = "sivi-yok" kutularını göster/gizle
- **V tuşu** = "sivi-var" kutularını göster/gizle

Not: Sınıf isimleri `data.yaml` dosyasından okunur.

### Hata Sınıflandırması

Resimleri otomatik olarak hata kategorilerine ayırın:

```
Ana Klasör
├── images/
│   └── IMG_001.jpg
└── labels/
    └── IMG_001.txt

Aşağıdaki yapı oluşturulur:
├── Ana Klasör_siniflandirma/
│   ├── hata_tipi_1/
│   │   └── camera_1/
│   │       └── IMG_001.jpg
│   ├── hata_tipi_2/
│   ├── dogru_etiket/
│   └── arada_kalanlar/
```

**Butonlar:**
- **❌ Hata Tipi 1 (1)** = Yanlış etiket
- **⚠️ Hata Tipi 2 (2)** = Belirsiz
- **✅ Doğru Etiket (3)** = Doğru annotasyon
- **❔ Arada Kalanlar (4)** = Sınırda durumlar

**Işlem Sırası:**
1. Resim yeni klasöre kopyalanır
2. `.txt` dosyası da kopyalanır
3. **Ctrl+Z** ile işlemi geri alabilirsiniz

---

## ⌨️ Kısayollar

### Navigasyon
| Tuş | İşlem |
|-----|-------|
| ← / → | Önceki/Sonraki resim |
| Home | İlk resime git |
| End | Son resime git |
| Space | Geçerli resmi sil |

### Etiketleme
| Tuş | İşlem |
|-----|-------|
| Delete/Backspace | Seçili kutuyu sil |
| Enter | Etiketleri kaydet |
| Esc | Değişiklikleri iptal et |
| M | Etiketleri aç/kapat |

### Undo/Redo
| Tuş | İşlem |
|-----|-------|
| Ctrl+Z | Geri al |
| Ctrl+Shift+Z | Yeniden yap |

### Sınıf Filtreleri
| Tuş | İşlem |
|-----|-------|
| K | Kapakli sınıfını göster/gizle |
| Y | Sivi-yok sınıfını göster/gizle |
| V | Sivi-var sınıfını göster/gizle |

### Hata İşaretleme
| Tuş | İşlem |
|-----|-------|
| 1 | Hata Tipi 1 olarak işaretle |
| 2 | Hata Tipi 2 olarak işaretle |
| 3 | Doğru Etiket olarak işaretle |
| 4 | Arada Kalanlar olarak işaretle |

### Diğer
| Tuş | İşlem |
|-----|-------|
| Ctrl+0 | Zoom sıfırla (%100) |
| Ctrl+S | Tüm değişiklikleri kaydet |

---

## 📁 Klasör Yapısı

### Beklenen Veri Seti Formatı

```
dataset/
├── data.yaml                    # Sınıf tanımları (opsiyonel)
├── images/
│   ├── camera_1/
│   │   ├── 2024_01_15_14_30_45_123_frame_001.jpg
│   │   ├── 2024_01_15_14_30_46_456_frame_002.jpg
│   │   └── ...
│   └── camera_2/
│       └── ...
└── labels/
    ├── camera_1/
    │   ├── 2024_01_15_14_30_45_123_frame_001.txt
    │   └── ...
    └── camera_2/
        └── ...
```

### data.yaml Formatı

```yaml
nc: 3  # Sınıf sayısı
names:
  0: kapakli
  1: sivi-yok
  2: sivi-var
```

### Etiket Dosyası Formatı (.txt)

YOLO standardı (normalized koordinatlar):
```
<class_id> <x_center> <y_center> <width> <height> [confidence]

Örnek:
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
2 0.7 0.8 0.25 0.3
```

---

## 🔧 Sorun Çözme

### "ultralytics yüklü değil" Uyarısı
**Çözüm:** Model çıkarımı devre dışı kalır ama manuel annotasyon çalışır.
```bash
pip install ultralytics
```

### "yaml kütüphanesi bulunamadı"
**Çözüm:** Sınıf isimleri yüklenmez, id numaraları gösterilir.
```bash
pip install PyYAML
```

### Resimler Yüklenmedi
- Klasörün doğru seçildiğini kontrol edin
- Desteklenen formatlar: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Dosya adı kurala uygun mu kontrol edin: `YYYY_MM_DD_HH_MM_SS_MS_frameID.jpg`

### Model Yükleme Hatası
```
"Model yüklenemedi: ..."
```
- Dosya `.pt` uzantılı mı kontrol edin
- Model dosyası bozuk olabilir, yeniden indirin
- CUDA versiyonunuz PyTorch sürümüyle uyumlu mu kontrol edin

### Kutuları Kaydederken Hata
- `.txt` dosyasının yazma izni var mı
- Klasör salt okunur değil mi kontrol edin
- Disk alanı yeterli mi kontrol edin

### Çok Yavaş Çıkarım
- GPU kullanımını kontrol edin: `nvidia-smi`
- Daha küçük bir model deneyin (YOLOv5n, YOLOv6n vs.)
- Confidence threshold'u artırın (daha az kutu = daha hızlı)

### Zoom Kilidi Düzgün Çalışmamıyor
- 🔒 Sabit butonunun turuncu olduğunu kontrol edin
- Paneller arasında geçerken toast mesajını gözlemleyin

---

## 💡 İpuçları ve Tricks

### Verimli Çalışma
1. **Modelleri karşılaştırma:** 2-3 model yükleyerek performa göre seçin
2. **Güven seviyesi ayarı:** Model karşılaştırma modunda confidence'ı artırın (daha az yanlış pozitif)
3. **Toplu İşlem:** Hata butonlarıyla hızlıca resimleri sınıflandırın
4. **Kontrol Paneli:** Undo/Redo ile hızlı düzeltmeler yapın

### Data Augmentation
- Eğitim sırasında veri seti kalitesi önemlidir
- Doğru etiketlenen veriler modeli iyileştirir
- Hata sınıflandırması ile problem alanları belirleyin

### Model Eğitimi
Annotasyondan sonra:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

## 📊 Teknik Özellikler

| Özellik | Değer |
|---------|-------|
| Maksimum Ekran Sayısı | 3 |
| Maksimum Zoom | %5000 (50x) |
| Desteklenen Formatlar | JPG, PNG, BMP, WebP |
| YOLO Versiyonu | v5, v6, v8+ |
| Python Sürümü | 3.8+ |
| GUI Framework | PyQt5 |

---

## 📝 Notlar

- Tüm değişiklikler otomatik `.txt` dosyasına yazılır
- Silinen resimler diskten silinir (geri alınamaz!)
- Hata sınıflandırması orijinal dosyaları taşımaz, kopyalar
- Multi-panel view tam senkronizasyon sağlar (zoom, kaydırma, konum)

---

## 🤝 Katkı ve Geliştirme

**Gelecek özellikler:**
- [ ] Şekil çizme aracı (kutunun yanında çokgen)
- [ ] Otomatik etiket tamamlama
- [ ] Batch dışa aktarma (coco JSON)
- [ ] İleri düzey filtreleme ve arama
- [ ] Multi-GPU desteği

---

## 📧 İletişim

Sorular veya hata raporları için:
- Kod üzerinden issue açın
- Veya doğrudan mühendislik ekibine ulaşın

---

**Sürüm:** 1.0.0 | **Güncelleme:** 2024
