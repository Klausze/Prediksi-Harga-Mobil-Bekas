import pandas as pd

# Pastikan path file sudah benar sesuai lokasi di komputer kamu
path_indo = r'c:\Users\RADIT\kulyeah\Porto\HargaKendaraanBekas\used_car.csv'
path_new = r'c:\Users\RADIT\kulyeah\Porto\HargaKendaraanBekas\used_car_data_new.csv'

# 1. LOAD DATA
df_indo = pd.read_csv(path_indo)
df_new = pd.read_csv(path_new)

# --- TAHAP 1: BERSIHKAN DATASET PERTAMA ---
df_indo_clean = df_indo[['brand', 'car name', 'year', 'mileage (km)', 'transmission', 'price (Rp)']].copy()
df_indo_clean.columns = ['brand', 'model', 'year', 'mileage', 'transmission', 'price']

# Fix mileage (dikali 1000)
df_indo_clean['mileage'] = df_indo_clean['mileage'] * 1000 

# --- TAHAP 2: BERSIHKAN DATASET KEDUA ---
df_new_clean = df_new[['id_merk', 'type', 'year', 'mileage', 'id_transmission', 'price_cash']].copy()
df_new_clean.columns = ['brand', 'model', 'year', 'mileage', 'transmission', 'price']

# Mapping Transmisi (1=Manual, 2=Automatic)
df_new_clean['transmission'] = df_new_clean['transmission'].map({1: 'Manual', 2: 'Automatic'})

# --- TAHAP 3: GABUNGKAN (MERGE) ---
df_gabungan = pd.concat([df_indo_clean, df_new_clean], ignore_index=True)

# PERBAIKAN DI SINI: Gunakan .str.upper()
df_gabungan['brand'] = df_gabungan['brand'].str.strip().str.title()
df_gabungan['model'] = df_gabungan['model'].str.strip().str.upper() 
df_gabungan['transmission'] = df_gabungan['transmission'].str.strip().str.title()

# Simpan jadi file baru
path_output = r'c:\Users\RADIT\kulyeah\Porto\HargaKendaraanBekas\data_mobil_gabungan.csv'
df_gabungan.to_csv(path_output, index=False)

print(f"Sukses! Total data gabungan: {len(df_gabungan)} baris.")
print(f"File tersimpan di: {path_output}")