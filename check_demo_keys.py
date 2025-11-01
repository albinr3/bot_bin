import os, time, hmac, hashlib, requests
from urllib.parse import urlencode
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

API_KEY = os.getenv("BINANCE_API_KEY","")
API_SECRET = os.getenv("BINANCE_API_SECRET","")
BASE = os.getenv("FAPI_BASE_URL", "https://demo-fapi.binance.com")

def ts(): return int(time.time()*1000)

def sign(params: dict) -> dict:
    qs = urlencode(params, doseq=True)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return {**params, "signature": sig}

headers = {
    "X-MBX-APIKEY": API_KEY,
    "Content-Type": "application/x-www-form-urlencoded"
}

print("BASE:", BASE)
print("API_KEY len:", len(API_KEY), "SECRET len:", len(API_SECRET))

# 1) público
r = requests.get(f"{BASE}/fapi/v1/time", timeout=10); r.raise_for_status()
print("Time OK:", r.json())

# 2) firmado
params = sign({"timestamp": ts(), "recvWindow": 5000})
r = requests.get(f"{BASE}/fapi/v2/account", headers=headers, params=params, timeout=10)
print("Account code:", r.status_code, "body:", r.text[:200])
r.raise_for_status()
print("Account OK (firmado)")
