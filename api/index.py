from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- 1. IMPORT INI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import sklearn
import os

# --- 1. INISIALISASI APP & MODEL ---
app = FastAPI(title="Trade-In Calculator API", description="API untuk prediksi harga mobil & tukar tambah")

# mengizinkan Frontend (HTML) untuk mengakses Backend ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
# --------------------------------

# Load model yang sudah kamu latih
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, 'car_price_model.pkl')

try:
    model = joblib.load(model_path)
    print("✅ Model berhasil dimuat di Vercel!")
except Exception as e:
    print(f"❌ Error saat loading pickle: {e}")

class CarItem(BaseModel):
    brand: str          
    model: str          
    year: int           
    mileage: float      
    transmission: str   

class TradeInRequest(BaseModel):
    old_car: CarItem    
    new_car: CarItem    

def predict_price(item: CarItem):
    short_model_input = item.model.split()[0].upper() 
    
    data = pd.DataFrame([{
        'brand': item.brand,
        'short_model': short_model_input, 
        'year': item.year,
        'mileage': item.mileage,
        'transmission': item.transmission
    }])
    
    prediction = model.predict(data)
    return prediction[0]

@app.get("/")
def home():
    return {"message": "Selamat datang di API Trade-In Mobil! Akses /docs untuk dokumentasi."}

@app.post("/predict/single")
def predict_single_car(item: CarItem):
    estimated_price = predict_price(item)
    return {
        "car": f"{item.brand} {item.model} ({item.year})",
        "estimated_price": int(estimated_price),
        "formatted_price": f"Rp {int(estimated_price):,}"
    }

@app.post("/predict/trade-in")
def calculate_trade_in(request: TradeInRequest):
    price_old_market = predict_price(request.old_car)
    dealer_buy_price = price_old_market * 0.85 
    price_new_market = predict_price(request.new_car)
    cost_to_pay = price_new_market - dealer_buy_price
    
    return {
        "old_car_valuation": {
            "market_price": int(price_old_market),
            "dealer_offer": int(dealer_buy_price), 
            "note": "Harga tawar dealer ~15% di bawah harga pasar."
        },
        "new_car_price": int(price_new_market),
        "trade_in_summary": {
            "amount_to_pay": int(cost_to_pay), 
            "formatted_pay": f"Rp {int(cost_to_pay):,}",
            "verdict": "Uang Muka Cukup!" if dealer_buy_price > (price_new_market * 0.3) else "Perlu Tambah DP"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)