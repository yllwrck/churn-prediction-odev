"""
=============================================================================
PROJE: Müşteri Ayrılma Tahmini (Churn Prediction) - Ara Ödev
HAZIRLAYAN: Taha SARIKAYA

AMAC:
Bu betik, müşteri verileri üzerinden müşterilerin ayrılıp ayrılmayacağını (churn)
tahmin eden uçtan uca temel bir makine öğrenmesi akışını gerçekleştirmektedir.

KULLANILAN KÜTÜPHANELER:
- pandas, numpy: Veri işleme ve analizi
- scikit-learn: Ön işleme, model eğitimi, veri bölme ve değerlendirme metrikleri

ÇALIŞTIRMA ADIMLARI:
1. Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt
2. Dosyayı çalıştırın: python churn_prediction.py
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# ADIM 1: Veri Setinin Hazırlanması / Sentetik Veri Oluşturma
# =============================================================================
np.random.seed(42)
n_samples = 150

data = {
    'yas': np.random.randint(18, 65, size=n_samples),
    'gelir': np.random.randint(15000, 90000, size=n_samples),
    'abonelik_suresi': np.random.randint(1, 60, size=n_samples),  # Ay cinsinden
    'destek_talebi_sayisi': np.random.randint(0, 10, size=n_samples),
    'sehir': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa'], size=n_samples),
    'uyelik_tipi': np.random.choice(['Standart', 'Premium', 'VIP'], size=n_samples)
}

df = pd.DataFrame(data)

# Mantıksal bir churn (hedef değişken) kurgulama:
# Destek talebi yüksek, abonelik süresi az veya geliri düşük olanların ayrılma olasılığı daha yüksek olsun
churn_prob = (
    (df['destek_talebi_sayisi'] * 0.15) + 
    ((60 - df['abonelik_suresi']) * 0.01) + 
    (np.where(df['uyelik_tipi'] == 'Standart', 0.2, 0.0))
)
churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
df['churn'] = (churn_prob > 0.45).astype(int)
df.to_csv("musteri_verisi.csv", index=False, sep=';', encoding='utf-8-sig')

# Rastgele %3 oranında eksik değer (NaN) enjekte edelim (ödev adımı için)
df.loc[df.sample(frac=0.03, random_state=42).index, 'gelir'] = np.nan

# =============================================================================
# ADIM 2: İlk İnceleme ve Eksik Değer Kontrolü
# =============================================================================
print("--- 1. Verinin İlk 5 Satırı ---")
print(df.head())

print(f"\n--- Veri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun ---")

print("\n--- Hedef Değişken (Churn) Dağılımı ---")
print(df['churn'].value_counts(normalize=True))

print("\n--- Eksik Değer Kontrolü ---")
print(df.isnull().sum())

# Eksik değerleri medyan ile doldurma
df['gelir'] = df['gelir'].fillna(df['gelir'].median())
print("\nEksik değerler dolduruldu. Kalan eksik sayısı:", df['gelir'].isnull().sum())

# =============================================================================
# ADIM 3: Öznitelik Mühendisliği (Feature Engineering)
# =============================================================================
# 1. Gelir Grubu kurgulanıyor
df['gelir_grubu'] = pd.qcut(df['gelir'], q=3, labels=['Dusuk', 'Orta', 'Yuksek'])

# 2. Destek talebi var mı kontrolü (Mantıksal öznitelik)
df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)

print("\nYeni Öznitelikler Eklendi. Güncel Sütunlar:")
print(df.columns.tolist())

# =============================================================================
# ADIM 4: Kategorik Dönüşüm (One-Hot Encoding) & Ölçekleme
# =============================================================================
kategorik_sutunlar = ['sehir', 'uyelik_tipi', 'gelir_grubu']
df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True)

X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# =============================================================================
# ADIM 5: Train, Validation ve Test Kümelerine Ayırma
# =============================================================================
# %70 Train+Validation, %30 Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Kalan %70'lik kısmı Train (%50) ve Validation (%20) olarak bölme
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2857, random_state=42, stratify=y_temp
)

print(f"\nVeri Kümeleri Boyutları:")
print(f"Train seti: {X_train.shape[0]} örnek")
print(f"Validation seti: {X_val.shape[0]} örnek")
print(f"Test seti: {X_test.shape[0]} örnek")

# Sayısal Değişken Ölçekleme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# ADIM 6: Model Eğitimi ve Validation Karşılaştırması
# =============================================================================
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree (Bonus)': DecisionTreeClassifier(random_state=42, max_depth=4)
}

val_scores = {}
print("\n--- Validation Performansı ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    val_scores[name] = f1
    print(f"{name} -> Validation Accuracy: {acc:.4f} | Validation F1-Score: {f1:.4f}")

# En iyi modeli seçme
best_model_name = max(val_scores, key=val_scores.get)
best_model = models[best_model_name]
print(f"\n Validation sonuçlarına göre seçilen en iyi model: **{best_model_name}**")

# =============================================================================
# ADIM 7: Seçilen Modelin Test Seti Üzerinde Değerlendirilmesi
# =============================================================================
test_pred = best_model.predict(X_test_scaled)

print("\n==================================================")
print(f"TEST SETİ DEĞERLENDİRME SONUÇLARI ({best_model_name})")
print("==================================================")
print("Confusion Matrix (Karmaşıklık Matrisi):")
print(confusion_matrix(y_test, test_pred))

print(f"\nAccuracy  : {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision : {precision_score(y_test, test_pred):.4f}")
print(f"Recall    : {recall_score(y_test, test_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, test_pred):.4f}")

# =============================================================================
# ADIM 8: Sonuç ve Yorum Çıktısı
# =============================================================================
print("\n--- SONUÇ VE YORUM ---")
print(
    f"Validation kümesinde yapılan değerlendirme sonucunda en yüksek F1 skorunu ({val_scores[best_model_name]:.4f}) "
    f"veren '{best_model_name}' modeli final modeli olarak seçilmiştir.\n"
    "Nedeni: Bu problemde sınıflar arası karar sınırlarının çizilmesinde ve lineer bağıntıların "
    "yakalanmasında seçilen algoritma daha dengeli bir öğrenme sağlamıştır. Özellikle "
    "sayısal özniteliklerin ölçeklenmiş olması mesafe/lineer bazlı modellerin "
    "performansını olumlu etkilemiştir."
)
