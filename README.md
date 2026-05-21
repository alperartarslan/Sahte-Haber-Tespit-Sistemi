# Yapay Zeka Destekli Sahte Haber Tespit Sistemi 📰

Bu proje, bir haber metninin gerçek ("Real News") veya sahte ("Fake News") olup olmadığını tespit etmek için Doğal Dil İşleme (NLP) ve Makine Öğrenimi algoritmalarını kullanan modüler, nesne yönelimli (OOP) bir yapay zeka sistemidir. Arayüz için premium tasarımlı Web UI altyapısı olarak **Streamlit** kullanılmıştır.

---

## 📋 Proje Özellikleri ve Karşılanan Gereksinimler

Proje dokümanında belirtilen tüm **Fonksiyonel ve Teknik Gereksinimler** eksiksiz şekilde karşılanmıştır:

### 1. Fonksiyonel Gereksinimler
- **Metin ve Dosya Girişi:** Kullanıcı arayüz üzerinden metin kutusuna doğrudan haber metni yapıştırabilir (**Metin Analizi**) veya bir `.txt` uzantılı haber dosyasını yükleyebilir (**Dosyadan Analiz**).
- **Otomatik Ön İşleme:** Metin temizleme, küçük harfe çevirme, noktalama temizliği, stopword kaldırma ve sözcük köklendirme (Lemmatization) adımları otomatik uygulanır.
- **Gerçek Zamanlı Analiz:** Kullanıcı butona bastığında model anlık olarak çalışır ve tahmini gösterir.
- **Tahmin Çıktısı:** Sonuçlar net bir şekilde **"Real News" (Gerçek Haber)** veya **"Fake News" (Sahte Haber)** olarak güven oranı yüzdesi ile birlikte görsel olarak sunulur.

### 2. Teknik Detaylar
- **Sayısal Vektörleştirme:** Metinler tokenize edildikten sonra **TF-IDF (Term Frequency-Inverse Document Frequency)** yöntemiyle sayısal vektörlere dönüştürülmüştür.
- **Makine Öğrenimi Modeli:** Algoritma olarak yüksek doğruluk ve hızlı çalışmasıyla bilinen **Logistic Regression (Lojistik Regresyon)** tercih edilmiştir.
- **Gelişmiş Ön İşleme (Lemmatizer):** Kelimeleri anlamsızca budayan klasik Stemmer yerine, kelimeleri sözlük anlamlarına göre köklendiren **WordNet Lemmatizer** kullanılmıştır. Ayrıca modelin ajans ismine göre ezberleme yapmasını engellemek için **Kaynak/Ajans Temizleme Filtresi** eklenmiştir.

---

## 📊 Model Performans Değerlendirmesi

Model, Kaggle'daki **Fake and Real News Dataset**'in %75'i ile eğitilmiş ve kalan %25'lik test setiyle değerlendirilmiştir. Performans sonuçları aşağıdaki gibidir:

- **Genel Doğruluk (Accuracy):** **%98.24**

### Detaylı Metrik Raporu (Classification Report):

| Sınıf | Tahmin Gücü (Precision) | Duyarlılık (Recall) | F1-Score | Destek Verisi (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Sahte Haber - Fake)** | 0.99 | 0.98 | 0.98 | 5878 |
| **1 (Gerçek Haber - Real)** | 0.98 | 0.98 | 0.98 | 5347 |
| **Ağırlıklı Ortalama** | **0.98** | **0.98** | **0.98** | **11225** |

---

## 📁 Proje Klasör Yapısı

```text
Yapay Zeka ve Bulut Bilişimi Proje/
│
├── data/                      # Kaggle'dan indirilen veri setleri (Fake.csv, True.csv)
├── models/                    # Eğitilmiş model ve vektörizer (.pkl dosyaları)
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/                       # Modüler python kodları
│   ├── preprocessing.py       # Metin ön işleme ve temizleme sınıfı (TextPreprocessor)
│   └── model.py               # Model eğitimi ve tahmin sınıfları (FakeNewsModel)
│
├── app.py                     # Streamlit premium web arayüzü kodu
├── requirements.txt           # Gerekli python paketleri
├── README.md                  # Proje dokümantasyonu (Bu dosya)
└── Proposal.md                # Proje öneri formu
```

---

## 💻 Kurulum ve Çalıştırma

Projeyi çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1- Bağımlılıkların Kurulması
Terminalde proje ana dizinine gelerek kütüphaneleri yükleyin:
```powershell
pip install -r requirements.txt
```

### 2- Modelin Eğitilmesi
Kaggle'dan indirilen `True.csv` ve `Fake.csv` dosyalarının `data/` klasörünün altında olduğundan emin olun. Ardından eğitimi başlatın:
```powershell
python -m src.model
```
*Bu işlem verileri ön işlemeden geçirecek, modeli eğitecek ve model dosyalarını `models/` klasörüne kaydedecektir.*

### 3- Web Uygulamasının Başlatılması
Model eğitildikten sonra Streamlit arayüzünü çalıştırmak için şu komutu çalıştırın:
```powershell
python -m streamlit run app.py
```
*Tarayıcınızda otomatik olarak açılacak olan premium arayüz üzerinden analizlerinizi gerçekleştirebilirsiniz.*
