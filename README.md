# California Konut Fiyat Tahmini (XGBoost ve Ensemble Learning)   
Bu proje, California konut veri seti üzerinde keşifsel veri analizi (EDA), veri temizleme, aykırı değer yönetimi ve çeşitli makine öğrenmesi modelleri ile fiyat tahmini yapmayı amaçlamaktadır. Projede temel odak noktası, veriyi en iyi şekilde hazırlayarak XGBoost ve diğer topluluk öğrenmesi (ensemble learning) modelleriyle yüksek doğruluklu tahminler üretmektir.

## 🚀 Proje Adımları
1. Veri Hazırlama ve Keşifsel Veri Analizi (EDA)
Eksik Veri Analizi: Veri setindeki total_bedrooms sütunundaki eksik değerler tespit edilmiştir.

Korelasyon Analizi: Değişkenler arasındaki ilişkiler heatmap kullanılarak görselleştirilmiştir.

Veri Dönüştürme: ocean_proximity kategorik değişkeni, modele uygun hale getirmek için One-Hot Encoding yöntemiyle sayısal verilere dönüştürülmüştür.

Eksik Değerlerin Tamamlanması: Eksik total_bedrooms değerleri sütunun medyan değeri ile doldurulmuştur.

2. Aykırı Değer (Outlier) Yönetimi
Modelin doğruluğunu artırmak için Interquartile Range (IQR) yöntemi kullanılarak aykırı değer analizi yapılmıştır:

find_outliers: Sütun bazlı aykırı değer sayılarını ve yüzdelerini hesaplayan fonksiyon geliştirilmiştir.

remove_all_clean: Belirlenen eşik değerine göre tüm sayısal sütunlardaki aykırı değerleri temizleyen bir yapı kurulmuştur.

3. Model Eğitimi ve Karşılaştırma
Proje kapsamında birçok farklı regresyon algoritması eğitilmiş ve performansları (MAE, MSE, R2 Score) karşılaştırılmıştır:

Lineer Modeller: Linear Regression, Lasso, Ridge.

Komşuluk ve Karar Ağaçları: KNeighbors Regressor, Decision Tree.

Topluluk Modelleri: Random Forest, AdaBoost, Gradient Boosting ve XGBoost.

4. Hiperparametre Optimizasyonu
En iyi performansı gösteren XGBoost modeli için RandomizedSearchCV ve GridSearchCV yöntemleri kullanılarak ince ayar yapılmıştır.

Optimize Edilen Parametreler: n_estimators, max_depth, learning_rate ve colsample_bytree.

En İyi Sonuçlar: GridSearchCV sonucunda learning_rate: 0.1 ve n_estimators: 400 gibi en iyi parametre setine ulaşılmıştır.

## 🛠 Kullanılan Teknolojiler
Programlama Dili: Python.

Veri Analizi: Pandas, NumPy.

Görselleştirme: Matplotlib, Seaborn.

Makine Öğrenmesi: Scikit-learn, XGBoost.

## 📊 Öne Çıkan Kod Blokları
Aykırı Değer Temizleme Fonksiyonu
```Python
def remove_all_clean(data, thres_hold=1.5):
    numeric_colums = data.select_dtypes(include=["float64","int64"]).columns
    data_clean = data.copy()
    for col in numeric_colums:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - thres_hold * IQR
        upper_bound = Q3 + thres_hold * IQR
        data_clean = data_clean[(data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)]
    return data_clean.copy()
```

## Model Performans Değerlendirmesi
```Python
def model_score (true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    r2 = r2_score(true, predicted)
    return mae, mse, r2
```
✅ Sonuç
Yapılan denemeler ve optimizasyonlar sonucunda, XGBoost modelinin California konut fiyatlarını tahmin etmede en yüksek başarıyı (R2 Score) sağladığı gözlemlenmiştir.
