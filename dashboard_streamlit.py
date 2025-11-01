"""
Dashboard visual (solo lectura) para **USDS‑M Futures** en Binance (REST puro)
===========================================================================

- Sin dependencias de UMFutures ni SDKs.
- Llama a los endpoints oficiales `/fapi/*` con firma HMAC (REST).
- Muestra: Mark Price, Balance, UPnL, ratio de riesgo, posiciones LONG/SHORT,
  exposición neta/bruta, órdenes abiertas y PnL realizado del día (incluye
  funding y comisiones).

⚠️ Seguridad y entornos
- Este dashboard es **solo lectura**; no envía órdenes.
- Funciona en **Demo/Testnet** y **Real**. Para **Demo** usa:
  FAPI_BASE_URL = https://demo-fapi.binance.com
  BINANCE_SYMBOL = BTCUSDT (recomendado en demo)
- Agrega un "airbag" que bloquea si no estás en demo (puedes desactivarlo).

Requisitos
---------
- Python 3.10+
- pip install: streamlit python-dotenv pandas requests

Uso
---
1) Crea un archivo `.env` junto a este script:
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   FAPI_BASE_URL=https://demo-fapi.binance.com
   BINANCE_SYMBOL=BTCUSDT
   DASHBOARD_TZ=America/Santo_Domingo
   AIRBAG_REQUIRE_DEMO=1           # 1 = bloquear si no es demo-fapi
2) Ejecuta:  streamlit run dashboard_streamlit.py
"""
from __future__ import annotations
import os
import time
import hmac
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ============================== Carga de entorno ===============================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BASE_URL = os.environ.get("FAPI_BASE_URL", "https://demo-fapi.binance.com").rstrip("/")
SYMBOL = os.environ.get("BINANCE_SYMBOL", "BTCUSDT").upper()
TZ_NAME = os.environ.get("DASHBOARD_TZ", "America/Santo_Domingo")
AIRBAG_DEMO = os.environ.get("AIRBAG_REQUIRE_DEMO", "1") == "1"

# Bloqueo opcional si no estás en demo
if AIRBAG_DEMO and "demo-fapi" not in BASE_URL:
    raise SystemExit("Bloqueado: cambia FAPI_BASE_URL a https://demo-fapi.binance.com para modo demo.")

if not API_KEY or not API_SECRET:
    raise SystemExit("Faltan BINANCE_API_KEY / BINANCE_API_SECRET en .env")

# ================================ HTTP/Signing =================================
session = requests.Session()
session.headers.update({
    "X-MBX-APIKEY": API_KEY,
    "Content-Type": "application/x-www-form-urlencoded",
})
SECRET_BYTES = API_SECRET.encode()

def _ts() -> int:
    return int(time.time() * 1000)

def _qs_signed(params: dict) -> str:
    qs = urlencode(params, doseq=True)
    sig = hmac.new(SECRET_BYTES, qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig

# ================================ Helpers tiempo ===============================

def today_local_utc_range(tz_name: str) -> tuple[int, int]:
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    start_ms = int(start_local.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end_local.astimezone(timezone.utc).timestamp() * 1000)
    return start_ms, end_ms

# ================================= Endpoints ===================================

def get_mark_price(symbol: str) -> float:
    r = session.get(f"{BASE_URL}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["markPrice"])


def get_account() -> dict:
    qs = _qs_signed({"timestamp": _ts(), "recvWindow": 5000})
    r = session.get(f"{BASE_URL}/fapi/v2/account?{qs}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_positions(symbol: str) -> list[dict]:
    qs = _qs_signed({"timestamp": _ts(), "recvWindow": 5000, "symbol": symbol})
    r = session.get(f"{BASE_URL}/fapi/v2/positionRisk?{qs}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_open_orders(symbol: str) -> list[dict]:
    qs = _qs_signed({"timestamp": _ts(), "recvWindow": 5000, "symbol": symbol})
    r = session.get(f"{BASE_URL}/fapi/v1/openOrders?{qs}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_income_today(symbol: str, tz_name: str) -> tuple[pd.DataFrame, float, float, float]:
    start_ms, end_ms = today_local_utc_range(tz_name)
    rows = []
    totals = {"REALIZED_PNL": 0.0, "FUNDING_FEE": 0.0, "COMMISSION": 0.0}
    for t in ("REALIZED_PNL", "FUNDING_FEE", "COMMISSION"):
        qs = _qs_signed({
            "timestamp": _ts(), "recvWindow": 5000,
            "symbol": symbol, "incomeType": t,
            "startTime": start_ms, "endTime": end_ms, "limit": 1000,
        })
        r = session.get(f"{BASE_URL}/fapi/v1/income?{qs}", timeout=10)
        r.raise_for_status()
        for it in r.json():
            amt = float(it.get("income", 0.0))
            totals[t] += amt
            rows.append({
                "time": datetime.fromtimestamp(it["time"]/1000, tz=timezone.utc).astimezone(ZoneInfo(tz_name)),
                "type": it.get("incomeType"),
                "amount": amt,
                "asset": it.get("asset"),
                "info": it.get("info", "")
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("time", ascending=False, inplace=True)
    return df, totals["REALIZED_PNL"], totals["FUNDING_FEE"], totals["COMMISSION"]

# ================================== UI =========================================

st.set_page_config(page_title=f"Futures Dashboard • {SYMBOL}", layout="wide")
st.title(f"📊 Binance Futures Dashboard · {SYMBOL}")

# Inicializa session_state para manejo de errores
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

refresh_ms = st.sidebar.slider("Refrescar cada (ms)", 1000, 10000, 3000, 500)
st.sidebar.caption("Sugerencia: 2000–4000 ms suele ir bien.")

# Variables globales con defaults
mark = 0.0
bal = 0.0
upnl = 0.0
risk_ratio = 0.0

# Top metrics
c1, c2, c3, c4 = st.columns(4)
try:
    mark = get_mark_price(SYMBOL)
    c1.metric("Mark Price", f"${mark:,.2f}")
    st.session_state.error_count = 0  # Reset si OK
except Exception as e:
    c1.error(f"Mark: {e}")
    st.session_state.error_count += 1

try:
    acc = get_account()
    bal = float(acc.get("totalWalletBalance", 0.0))
    upnl = float(acc.get("totalUnrealizedProfit", 0.0))
    maint = float(acc.get("totalMaintMargin", 0.0))
    risk_ratio = (maint / bal) if bal else 0.0
    c2.metric("Balance (Wallet)", f"${bal:,.2f}")
    c3.metric("UPnL", f"${upnl:,.2f}")
    c4.metric("Risk Ratio", f"{risk_ratio:.2%}")
    st.session_state.error_count = 0
except Exception as e:
    st.error(f"Cuenta: {e}")
    st.session_state.error_count += 1

st.divider()

# Posiciones
try:
    pos = get_positions(SYMBOL)
    rows = []
    long_qty = 0.0
    short_qty = 0.0
    for p in pos:
        side = p.get("positionSide")
        qty = abs(float(p.get("positionAmt", 0.0)))
        if qty == 0:
            continue
        entry = float(p.get("entryPrice", 0.0))
        up = float(p.get("unRealizedProfit", 0.0))
        notional = qty * mark if mark > 0 else 0.0
        if side == "LONG":
            long_qty += qty
        elif side == "SHORT":
            short_qty += qty
        rows.append({
            "Side": side,
            "Qty": qty,
            "Entry": entry,
            "Notional (USDT)": notional,
            "UPnL (USDT)": up,
            "ROE %": (up / max(notional, 1e-9)) * 100.0
        })
    st.subheader("Posiciones abiertas")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("(Sin posiciones)")

    net_exposure = (long_qty - short_qty) * mark if mark > 0 else 0.0
    gross_exposure = (long_qty + short_qty) * mark if mark > 0 else 0.0
    hedged = abs(net_exposure) <= 0.05 * max(gross_exposure, 1.0)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Exposición neta (USDT)", f"${net_exposure:,.2f}")
    cc2.metric("Exposición bruta (USDT)", f"${gross_exposure:,.2f}")
    cc3.metric("¿Hedged?", "Sí" if hedged else "No")
    st.session_state.error_count = 0
except Exception as e:
    st.error(f"Posiciones: {e}")
    st.session_state.error_count += 1

# Órdenes abiertas
try:
    oo = get_open_orders(SYMBOL)
    st.subheader(f"Órdenes abiertas · {len(oo)}")
    if not oo:
        st.caption("(Ninguna)")
    else:
        tbl = []
        for o in oo:
            tbl.append({
                "Side": o.get("side"),
                "PosSide": o.get("positionSide"),
                "Type": o.get("type"),
                "Price": float(o.get("price", 0.0)),
                "Qty": float(o.get("origQty", 0.0)),
                "ReduceOnly": o.get("reduceOnly", False),
                "Status": o.get("status"),
                "ClientId": o.get("clientOrderId")
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.session_state.error_count = 0
except Exception as e:
    st.error(f"Órdenes: {e}")
    st.session_state.error_count += 1

st.divider()

# Ingresos del día
try:
    df_inc, pnl_today, funding_today, fees_today = get_income_today(SYMBOL, TZ_NAME)
    d1, d2, d3 = st.columns(3)
    d1.metric("Realized PnL (hoy)", f"${pnl_today:,.2f}")
    d2.metric("Funding (hoy)", f"${funding_today:,.2f}")
    d3.metric("Comisiones (hoy)", f"${fees_today:,.2f}")
    with st.expander("Detalle ingresos (hoy)"):
        if df_inc.empty:
            st.caption("(Sin movimientos hoy)")
        else:
            st.dataframe(df_inc, use_container_width=True, hide_index=True)
    st.session_state.error_count = 0
except Exception as e:
    st.error(f"Ingresos del día: {e}")
    st.session_state.error_count += 1

# Manejo de errores consecutivos
if st.session_state.error_count >= 5:
    st.error("⚠️ Demasiados errores consecutivos. Auto-refresh pausado.")
    st.stop()

# Auto-refresh
time.sleep(refresh_ms / 1000.0)
st.rerun()