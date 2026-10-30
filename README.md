# Derin Öğrenme ile Apple (AAPL) Hisse Senedi Fiyat Tahmini 📈

Bu proje, Apple Inc. (AAPL) hisse senedinin geçmiş piyasa hareketlerini kullanarak gelecekteki fiyat yönelimini tahmin eden bir **Uzun Kısa Süreli Bellek (LSTM - Long Short-Term Memory)** zaman serisi modelidir. 

Çalışma, gelecekte kurulması planlanan daha kapsamlı ve otonom bir algoritmik ticaret (algorithmic trading) sisteminin **"Fiyat Tahmin Motoru"** altyapısını oluşturmak üzere tasarlanmıştır.

---

## 🛠 Kullanılan Teknolojiler
* **Python 3.x**
* **TensorFlow / Keras** (LSTM Mimarisi & Derin Öğrenme)
* **yfinance** (Yahoo Finance Finansal Zaman Serisi Veri Çekimi)
* **Pandas & NumPy** (Veri Analizi & Tensör Dönüşümleri)
* **Scikit-learn** (MinMaxScaler ile Normalizasyon)
* **Matplotlib & Seaborn** (Teknik Analiz ve Hata Görselleştirme)

---

## 📊 Veri Seti ve Ön İşleme
Veri seti `yfinance` API'si üzerinden **1 Ocak 2021 – 1 Ekim 2026** tarihleri arasını kapsayacak şekilde dinamik olarak çekilmiştir. Toplamda 1442 işlem günü ve 5 temel finansal metrik (Open, High, Low, Close, Volume) içermektedir. Hedef değişken (target) **Kapanış (Close)** fiyatı olarak belirlenmiştir.

* **Ölçeklendirme:** Gradyan kaybolmasının (vanishing gradient) önüne geçmek ve aktivasyon fonksiyonlarının verimini artırmak için `MinMaxScaler` ile veriler 0-1 aralığına normalize edilmiştir.
* **Zaman Penceresi (Lookback Window):** Modelin ardışık fiyat örüntülerini yakalayabilmesi amacıyla otoregresif bir pencereleme uygulanmış; *son 60 günün verisi (X) kullanılarak 61. günün kapanış fiyatı (y)* tahmin edilecek şekilde yapılandırılmıştır.
* **Kronolojik Ayrım:** Finansal zaman serilerinin doğasına sadık kalınarak veriler rastgele karıştırılmamış; ilk %80 Eğitim (Train), son %20 Test seti olarak kronolojik sırayla bölünmüştür.

---

## 🔎 Keşifçi Veri Analizi (EDA)

Fiyat serisi üzerindeki makro trendler ve teknik göstergeler incelenmiştir:

![Fiyat ve Hareketli Ortalamalar](images/02_moving_averages.png)
> **Not:** 50 ve 200 günlük hareketli ortalamalar (MA50 & MA200) incelendiğinde, kısa vadeli ortalamanın uzun vadeli ortalamayı kestiği periyotların trend dönüşümlerini net şekilde yansıttığı tespit edilmiştir.

![Korelasyon Matrisi](images/03_correlation.png)
> **Not:** Fiyat değişkenleri (Open, High, Low, Close) arasında mükemmele yakın (+1.0) pozitif doğrusal ilişki varken; işlem hacmi (Volume) ile fiyatlar arasında zayıf negatif (-0.25) bir korelasyon saptanmıştır.

---

## 🧠 Model Mimarisi
* **1. Katman:** LSTM (50 Nöron, `return_sequences=True`)
* **2. Katman:** LSTM (50 Nöron, `return_sequences=False`)
* **Tam Bağlantılı Katmanlar (Dense):** 25 Nöron -> 1 Nöron (Regresyon Çıktısı)
* **Optimizasyon & Kayıp Fonksiyonu:** Adam Optimizer, Mean Squared Error (MSE)
* **Eğitim Parametreleri:** 32 Batch Size, 20 Epoch, %10 Validation Split

---

## 📉 Eğitim ve Doğrulama (Validation) Performansı
Modelin aşırı öğrenme (overfitting) yaşayıp yaşamadığını denetlemek adına eğitim ve doğrulama kayıpları döngü bazında takip edilmiştir:

![Kayıp Eğrisi](images/05_loss_curve.png)
> **Çıkarım:** Hem eğitim kaybının hem de doğrulama kaybının istikrarlı bir şekilde sıfıra yaklaşması ve paralel seyretmesi, modelin varyansının dengeli olduğunu ve yeni verilere genelleme yapabilme kabiliyetini doğrulamaktadır.

---

## 🚀 Sonuçlar ve Hata Analizi
Eğitilen model, daha önce hiç görmediği %20'lik test verisi üzerinde sınanmış ve tahminler ters ölçeklendirme (inverse transform) ile tekrar USD değerine çevrilmiştir.

* **Test Performansı (RMSE):** **6.78 USD**

![Tahmin Performansı](images/04_predictions.png)

* **Güçlü Yönler:** Model, test dönemindeki genel yükseliş trendini ve orta vadeli dalgalanmaları yüksek bir doğrulukla yakalamıştır.
* **Hata Analizi (Gecikme/Lag Problemi):** Ani fiyat kırılımlarında model tahminlerinin gerçek piyasa hareketini 1 gün geriden takip ettiği (lag) gözlemlenmiştir. Yalnızca geçmiş fiyat hareketlerine dayanan otoregresif modeller, anlık piyasa şoklarını ve haber akışlarını önceden sezemedikleri için bir önceki günün kapanışına yakın güvenli tahminler üretme eğilimindedir.

---

## 🔮 Gelecek Vizyonu (Algoritmik Trading Botu)
1. **Duygu Analizi Modülü (NLP):** Finansal haber başlıklarını ve piyasa duyarlılığını işleyen FinBERT tabanlı bir Doğal Dil İşleme katmanının entegre edilmesi.
2. **Multimodal Otonom Sistem:** Teknik göstergeler ile haber akışını birleştirerek "gecikme" sorununu aşan, aracı kurum API'leri ile canlı emir iletebilen tam otonom bir algoritmik ticaret sistemine dönüştürülmesi.

---
**Geliştirici:** Rıdvan Efe Birben  
**Bağlantılar:** [Google Colab Defteri](https://colab.research.google.com/drive/1-nHTCQaqRPfP1og0wAziQgKuDJwpG7Id)
