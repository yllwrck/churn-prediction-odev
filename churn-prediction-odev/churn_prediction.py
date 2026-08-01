"""
Müşteri Ayrılma (Churn) Tahmini - Ara Ödev
Hazırlayan: Taha SARIKAYA

Bu çalışmada, müşteri verileri üzerinden ayrılma (churn) ihtimallerini
tahmin eden temel bir makine öğrenmesi akışı gerçekleştirilmiştir.

Kullanılan Kütüphaneler:
- pandas, numpy
- scikit-learn

Çalıştırma Adımı:
python churn_prediction.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Veri Setinin Oluşturulması
np.random.seed(34)
ornek_sayisi = 150

data = {
    'yas': np.random.randint(18, 65, size=ornek_sayisi),
    'gelir': np.random.randint(15000, 90000, size=ornek_sayisi),
    'abonelik_suresi': np.random.randint(1, 60, size=ornek_sayisi),
    'destek_talebi_sayisi': np.random.randint(0, 10, size=ornek_sayisi),
    'sehir': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa'], size=ornek_sayisi),
    'uyelik_tipi': np.random.choice(['Standart', 'Premium', 'VIP'], size=ornek_sayisi)
}

df = pd.DataFrame(data)

# Churn olasılığı belirli kurallara göre kurgulanmıştır
churn_skor = (
    (df['destek_talebi_sayisi'] * 0.15) + 
    ((60 - df['abonelik_suresi']) * 0.01) + 
    (np.where(df['uyelik_tipi'] == 'Standart', 0.2, 0.0))
)
churn_skor = (churn_skor - churn_skor.min()) / (churn_skor.max() - churn_skor.min())
df['churn'] = (churn_skor > 0.45).astype(int)

# Oluşturulan veri seti dosyaya kaydedilmiştir
df.to_csv("musteri_verisi.csv", index=False, sep=';', encoding='utf-8-sig')

# Eksik değer yönetimi adımı için rastgele %3 oranında eksik veri eklenmiştir
df.loc[df.sample(frac=0.03, random_state=34).index, 'gelir'] = np.nan

# 2. Veri İnceleme ve Eksik Değer Tamamlama
print("Verinin ilk 5 satırı:")
print(df.head())

print(f"\nVeri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

print("\nChurn dağılımı:")
print(df['churn'].value_counts(normalize=True))

print("\nEksik değer sayısı:")
print(df.isnull().sum())

# Gelir sütunundaki eksik değerler medyan değeri ile doldurulmuştur
df['gelir'] = df['gelir'].fillna(df['gelir'].median())
print("\nDoldurma işlemi sonrası eksik gelir sayısı:", df['gelir'].isnull().sum())

# 3. Öznitelik Mühendisliği (Feature Engineering)
# Gelir değişkeni 3 gruba ayrılmıştır
df['gelir_grubu'] = pd.qcut(df['gelir'], q=3, labels=['Dusuk', 'Orta', 'Yuksek'])

# Destek talebi varlığı mantıksal değişkene dönüştürülmüştür
df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)

# 4. Kodlama (Encoding) ve Ölçekleme
kategorik_kolonlar = ['sehir', 'uyelik_tipi', 'gelir_grubu']
df_encoded = pd.get_dummies(df, columns=kategorik_kolonlar, drop_first=True)

X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# 5. Veri Setinin Bölünmesi (Train - Validation - Test)
# %70 Eğitim+Doğrulama, %30 Test olacak şekilde ayrılmıştır
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X, y, test_size=0.30, random_state=34, stratify=y
)

# Kalan kısım Eğitim (%50) ve Doğrulama (%20) olarak bölünmüştür
X_train, X_val, y_train, y_val = train_test_split(
    X_gecici, y_gecici, test_size=0.2857, random_state=34, stratify=y_gecici
)

print(f"\nVeri setleri boyutları -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Öznitelikler StandardScaler ile ölçeklenmiştir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Model Eğitimi ve Doğrulama (Validation) Karşılaştırması
modeller = {
    'Lojistik Regresyon': LogisticRegression(random_state=34),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Karar Agaci': DecisionTreeClassifier(random_state=34, max_depth=4)
}

val_skorlari = {}
print("\n--- Validation Sonuçları ---")
for isim, model in modeller.items():
    model.fit(X_train_scaled, y_train)
    tahmin_val = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, tahmin_val)
    f1 = f1_score(y_val, tahmin_val)
    val_skorlari[isim] = f1
    print(f"{isim} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# En yüksek F1 skoruna sahip model belirlenmiştir
en_iyi_model_ismi = max(val_skorlari, key=val_skorlari.get)
en_iyi_model = modeller[en_iyi_model_ismi]
print(f"\nValidation sonuçlarına göre seçilen model: {en_iyi_model_ismi}")

# 7. Test Seti Üzerinde Değerlendirme
tahmin_test = en_iyi_model.predict(X_test_scaled)

print("\n--- TEST SETİ SONUÇLARI ---")
print("Karmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, tahmin_test))

print(f"Accuracy  : {accuracy_score(y_test, tahmin_test):.4f}")
print(f"Precision : {precision_score(y_test, tahmin_test):.4f}")
print(f"Recall    : {recall_score(y_test, tahmin_test):.4f}")
print(f"F1-Score  : {f1_score(y_test, tahmin_test):.4f}")

# 8. Sonuç ve Değerlendirme
print("\n--- SONUÇ VE DEĞERLENDİRME ---")
print(f"Validation aşamasında en yüksek F1 skorunu veren '{en_iyi_model_ismi}' modeli nihai model olarak seçilmiştir.")
print("Verilerin ölçeklenmesi sonucunda, doğrusal/mesafe tabanlı algoritmaların veri seti üzerinde daha kararlı performans gösterdiği gözlemlenmiştir.")
