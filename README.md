# Prediksi-Harga-Mobil-Bekas

AutoTrade AI adalah aplikasi *Full-Stack Machine Learning* yang dirancang untuk memprediksi harga mobil bekas di pasar Indonesia sekaligus mensimulasikan perhitungan biaya **Tukar Tambah (Trade-In)** secara *real-time*.

Proyek ini dibuat untuk menjembatani kompetensi di bidang **Data Science / Machine Learning** dan **Web Development (Backend & Frontend)**.

## 🌟 Fitur Utama
- **Data Integration:** Menggabungkan dan menstandardisasi dua dataset mobil bekas Indonesia yang berbeda (termasuk penanganan mata uang, format teks, dan pembersihan data).
- **Smart Price Prediction:** Menggunakan algoritma *Random Forest Regressor* dengan pendekatan *Machine Learning Pipeline* untuk memprediksi harga mobil berdasarkan brand, model, tahun, transmisi, dan jarak tempuh.
- **Dynamic Trade-In Logic:** Simulasi logika bisnis *dealer* mobil asli (pemotongan margin keuntungan dealer 15% dari harga pasar mobil lama).
- **Interactive Web UI:** Tampilan web *clean* dan responsif menggunakan CSS minimalis untuk mempermudah pengguna awam melakukan input dan melihat estimasi biaya.

## 🛠️ Tech Stack
- **Data Science / AI:** Python, Pandas, Scikit-Learn, Joblib
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend Website:** HTML5, JavaScript (Fetch API), Pico CSS

## 📁 Struktur Proyek
```text
HargaKendaraanBekas/
│
├── api/
│   ├── main.py               # Server Backend API (FastAPI)
│   └── car_price_model.pkl   # Model Machine Learning
│
├── dataset/
│   ├── used_car.csv
│   └── used_car_data_new.csv
│
├── notebooks/
│   ├── 1_data_prep.py        # Script integrasi & cleaning data
│   └── 2_train_model.py      # Script melatih Machine Learning
│
├── index.html                # Tampilan Website Frontend
├── merged_car_data.csv       # Hasil gabungan data bersih             
├── requirements.txt          # Daftar dependencies Python
└── vercel.json               