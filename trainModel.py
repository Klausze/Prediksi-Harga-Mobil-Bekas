import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. LOAD DATA
# Pastikan nama filenya sesuai dengan output dari langkah sebelumnya
path_data = r'c:\Users\RADIT\kulyeah\Porto\HargaKendaraanBekas\data_mobil_gabungan.csv'
df = pd.read_csv(path_data)

print(f"Memuat {len(df)} data mobil...")

# 2. FEATURE ENGINEERING (TRIK RAHASIA)
# Model mobil seringkali terlalu spesifik (misal: "AVANZA 1.3 G M/T"). 
# Kita ambil kata pertamanya saja ("AVANZA") agar model lebih pintar mengenali pola umum.
df['short_model'] = df['model'].astype(str).apply(lambda x: x.split()[0])

print("Contoh data yang akan dilatih:")
print(df[['brand', 'short_model', 'year', 'price']].head())

# 3. PERSIAPAN DATA (X dan y)
# Kita gunakan 'short_model' sebagai ganti 'model' yang asli
X = df[['brand', 'short_model', 'year', 'mileage', 'transmission']]
y = df['price']

# 4. MEMBUAT PIPELINE
# Pipeline ini akan otomatis mengubah teks -> angka saat ada data baru masuk nanti
categorical_features = ['brand', 'short_model', 'transmission']
numerical_features = ['year', 'mileage']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Menggunakan Random Forest (Algoritma "Bapak Segala Tahu" untuk data campuran)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. TRAINING MODEL
print("\nSedang melatih model... (Tunggu sebentar)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 6. EVALUASI SEDERHANA
score = model_pipeline.score(X_test, y_test)
print(f"Training Selesai! Akurasi (R^2 Score): {score:.2f}")

# 7. SIMPAN MODEL
output_path = r'c:\Users\RADIT\kulyeah\Porto\HargaKendaraanBekas\car_price_model.pkl'
joblib.dump(model_pipeline, output_path)
print(f"Model berhasil disimpan di: {output_path}")