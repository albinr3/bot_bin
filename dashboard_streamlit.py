"""
Dashboard visual (solo lectura) para **USDS-M Futures** en Binance (REST puro)
===========================================================================

- Sin dependencias de UMFutures ni SDKs.
- Llama a los endpoints oficiales `/fapi/*` con firma HMAC (REST).
- Muestra: Mark Price, Balance, UPnL, ratio de riesgo, posiciones LONG/SHORT,
  exposición neta/bruta, órdenes abiertas y PnL realizado del día (incluye
  funding y comisiones).
- Panel "Inversión y margen": margen usado, desglose, notional de órdenes abiertas
  y % de balance comprometido + notional teórico de grid.
- PnL del día con dos fuentes: `/income` (agregado local) y respaldo con
  `/userTrades` (suma de realizedPnl y comisiones de hoy).
- CORREGIDO: Prioriza trades si income no está sincronizado (común en testnet).

IMPORTANTE: realizedPnl YA incluye comisiones descontadas (neto). Las comisiones
se muestran como información separada, NO para restarlas del PnL.

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
   # Opcional (para cálculo teórico del grid):
   ORDER_NOTIONAL_USD=500
   GRID_LEVELS_PER_SIDE=5

2) Ejecuta:
   streamlit run dashboard_streamlit.py
"""
from __future__ import annotations

import os
import time
import hmac
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
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

# Para cálculo teórico del grid (opcional)
try:
    ORDER_NOTIONAL_USD = float(os.environ.get("ORDER_NOTIONAL_USD") or 0)
    GRID_LEVELS_PER_SIDE = int(os.environ.get("GRID_LEVELS_PER_SIDE") or 0)
except (ValueError, TypeError):
    ORDER_NOTIONAL_USD = 0
    GRID_LEVELS_PER_SIDE = 0

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

# ---------- Incomes del día (paginación correcta con IDs) ---------------------
def get_income_today(symbol: str, tz_name: str) -> tuple[pd.DataFrame, float, float, float]:
    """
    IMPORTANTE: realizedPnl en income YA incluye comisiones descontadas.
    Las comisiones se reportan por separado solo como información.
    """
    start_ms, end_ms = today_local_utc_range(tz_name)
    rows = []
    last_id = None
    
    while True:
        params = {
            "timestamp": _ts(),
            "recvWindow": 5000,
            "symbol": symbol,
            "limit": 1000
        }
        
        if last_id is None:
            params["startTime"] = start_ms
            params["endTime"] = end_ms
        else:
            # Usa fromId para paginación sin perder registros
            params["fromId"] = last_id + 1
        
        qs = _qs_signed(params)
        r = session.get(f"{BASE_URL}/fapi/v1/income?{qs}", timeout=10)
        r.raise_for_status()
        batch = r.json()
        
        if not batch:
            break
            
        for it in batch:
            t_ms = int(it["time"])
            # Filtra solo registros del día de hoy
            if last_id is None and not (start_ms <= t_ms < end_ms):
                continue
                
            rows.append({
                "time": datetime.fromtimestamp(t_ms/1000, tz=timezone.utc).astimezone(ZoneInfo(tz_name)),
                "type": it.get("incomeType"),
                "amount": float(it.get("income", 0.0)),
                "asset": it.get("asset"),
                "symbol": it.get("symbol", ""),
                "info": it.get("info", ""),
                "tranId": it.get("tranId")
            })
            # Guarda el último ID para paginación
            last_id = int(it.get("tranId", last_id or 0))
        
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0.0, 0.0, 0.0

    df.sort_values("time", ascending=False, inplace=True)
    pnl = float(df.loc[df["type"] == "REALIZED_PNL", "amount"].sum())
    funding = float(df.loc[df["type"] == "FUNDING_FEE", "amount"].sum())
    fees = float(df.loc[df["type"] == "COMMISSION", "amount"].sum())
    return df, pnl, funding, fees

# ---------- Respaldo por trades (paginación correcta) --------------------------
def get_trades_pnl_today(symbol: str, tz_name: str) -> tuple[pd.DataFrame, float, float]:
    """
    IMPORTANTE: realizedPnl en trades YA incluye comisiones descontadas.
    """
    start_ms, end_ms = today_local_utc_range(tz_name)
    rows = []
    total_realized = 0.0
    total_commission = 0.0
    last_id = None

    while True:
        params = {
            "timestamp": _ts(),
            "recvWindow": 5000,
            "symbol": symbol,
            "limit": 1000
        }
        
        if last_id is None:
            params["startTime"] = start_ms
            params["endTime"] = end_ms
        else:
            params["fromId"] = last_id + 1
        
        qs = _qs_signed(params)
        r = session.get(f"{BASE_URL}/fapi/v1/userTrades?{qs}", timeout=10)
        r.raise_for_status()
        batch = r.json()
        
        if not batch:
            break
            
        for tr in batch:
            t_ms = int(tr["time"])
            if last_id is None and not (start_ms <= t_ms < end_ms):
                continue
                
            realized = float(tr.get("realizedPnl", 0.0))
            commission = float(tr.get("commission", 0.0))
            total_realized += realized
            total_commission += commission
            
            rows.append({
                "time": datetime.fromtimestamp(t_ms/1000, tz=timezone.utc).astimezone(ZoneInfo(tz_name)),
                "id": tr.get("id"),
                "orderId": tr.get("orderId"),
                "side": tr.get("side"),
                "qty": float(tr.get("qty", 0.0)),
                "price": float(tr.get("price", 0.0)),
                "realizedPnl": realized,
                "commission": commission,
                "buyer": tr.get("buyer"),
                "maker": tr.get("maker")
            })
            last_id = int(tr.get("id", last_id or 0))
        
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("time", ascending=False, inplace=True)
    return df, total_realized, total_commission

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
acc = {}  # Default vacío

# Top metrics
c1, c2, c3, c4 = st.columns(4)
try:
    mark = get_mark_price(SYMBOL)
    c1.metric("Mark Price", f"${mark:,.2f}")
    st.session_state.error_count = 0
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

# ==================== Panel Inversión y margen =================================
try:
    # Desglose de márgenes desde la cuenta
    tot_init = float(acc.get("totalInitialMargin", 0.0))
    pos_init = float(acc.get("totalPositionInitialMargin", 0.0))
    ord_init = float(acc.get("totalOpenOrderInitialMargin", 0.0))
    avail = float(acc.get("availableBalance", 0.0))
    pct_committed = (tot_init / bal) if bal else 0.0

    # Notional activo en órdenes abiertas (LIMIT no-reduceOnly)
    oo = get_open_orders(SYMBOL)
    open_notional = 0.0
    grid_open_orders = []
    for o in oo:
        try:
            typ = (o.get("type") or "").upper()
            ro = str(o.get("reduceOnly", "false")).lower() == "true"
            price = float(o.get("price", 0.0))
            qty = float(o.get("origQty", 0.0))
            if typ == "LIMIT" and not ro and price > 0 and qty > 0:
                n = price * qty
                open_notional += n
                grid_open_orders.append((o.get("side"), o.get("positionSide"), price, qty, n))
        except Exception:
            pass

    # Notional teórico del grid (si viene en .env)
    theoretical_grid_notional = 0.0
    if ORDER_NOTIONAL_USD > 0 and GRID_LEVELS_PER_SIDE > 0:
        theoretical_grid_notional = ORDER_NOTIONAL_USD * GRID_LEVELS_PER_SIDE * 2

    # Bloque de métricas
    st.subheader("Inversión y margen")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Margen usado (totalInitial)", f"${tot_init:,.2f}")
    m2.metric("Margen posiciones", f"${pos_init:,.2f}")
    m3.metric("Margen órdenes", f"${ord_init:,.2f}")
    m4.metric("% Balance comprometido", f"{pct_committed:.2%}")

    mm1, mm2, mm3 = st.columns(3)
    mm1.metric("Notional abierto en libro (LIMIT)", f"${open_notional:,.2f}")
    if theoretical_grid_notional > 0:
        mm2.metric("Grid teórico (2×N×orden)", f"${theoretical_grid_notional:,.2f}")
    mm3.metric("Disponible (availableBalance)", f"${avail:,.2f}")

    with st.expander("Detalle órdenes GRID (LIMIT no-reduceOnly)"):
        if grid_open_orders:
            df_go = pd.DataFrame(grid_open_orders, columns=["Side", "PosSide", "Price", "Qty", "Notional"])
            st.dataframe(df_go, use_container_width=True, hide_index=True)
        else:
            st.caption("(Sin órdenes LIMIT válidas no-reduceOnly)")

    st.session_state.error_count = 0
except Exception as e:
    st.error(f"Inversión y margen: {e}")
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

# Órdenes abiertas (lista completa)
try:
    oo = get_open_orders(SYMBOL)  # Siempre fetch fresco
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

# ==================== INGRESOS DEL DÍA (CORREGIDO CON FALLBACK) ================
try:
    df_inc, pnl_today, funding_today, fees_today = get_income_today(SYMBOL, TZ_NAME)
    df_tr, pnl_trades, fees_trades = get_trades_pnl_today(SYMBOL, TZ_NAME)

    st.subheader("💰 Ganancias del día")
    
    # EXPLICACIÓN IMPORTANTE
    st.info("""
    **ℹ️ Importante:** El PnL realizado YA incluye las comisiones descontadas (es el valor neto).
    Las comisiones se muestran por separado solo como información de cuánto se pagó en fees.
    """)
    
    # ✅ NUEVA LÓGICA: Usa trades si income está vacío (común en testnet)
    if pnl_today == 0.0 and pnl_trades != 0.0:
        st.warning("⚠️ `/income` aún no sincronizado (normal en testnet). Mostrando datos de `/userTrades` (confiables).")
        pnl_display = pnl_trades
        fees_display = fees_trades
        source = "trades (backup)"
    else:
        pnl_display = pnl_today
        fees_display = fees_today
        source = "income"
    
    # Métricas principales
    d1, d2, d3 = st.columns(3)
    d1.metric("💵 PnL neto realizado", f"${pnl_display:,.2f}", 
              help=f"Fuente activa: {source}")
    d2.metric("📊 Funding pagado/recibido", f"${funding_today:,.2f}")
    d3.metric("📋 Comisiones (info)", f"${fees_display:,.2f}", 
              help="Ya están descontadas del PnL. Mostradas solo como información.")

    # TOTAL NETO DEL DÍA (PnL + Funding, SIN restar comisiones otra vez)
    total_neto_dia = pnl_display + funding_today
    st.metric("🎯 **TOTAL NETO DEL DÍA**", f"${total_neto_dia:,.2f}", 
              help="PnL realizado (ya neto) + Funding. Las comisiones YA están descontadas del PnL.")

    st.divider()

    # Mostrar ambas fuentes para comparación y debugging
    st.caption("**Comparación de fuentes (para verificación):**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 PnL (/income)", f"${pnl_today:,.2f}")
        st.metric("Comisiones (/income)", f"${fees_today:,.2f}")
        st.metric("Funding (/income)", f"${funding_today:,.2f}")
    with col2:
        st.metric("📈 PnL (/trades)", f"${pnl_trades:,.2f}")
        st.metric("Comisiones (/trades)", f"${fees_trades:,.2f}")
        if pnl_today == 0.0 and pnl_trades != 0.0:
            st.success("✅ Usando esta fuente")

    # Expandibles con detalles
    with st.expander("📜 Detalle completo de incomes (hoy)"):
        if df_inc.empty:
            st.caption("⚠️ Sin registros de income aún (puede haber delay de sincronización en testnet)")
            st.caption("Los trades aparecen inmediatamente, pero los 'income' pueden tardar minutos u horas.")
        else:
            st.dataframe(df_inc, use_container_width=True, hide_index=True)
            st.caption(f"Total registros: {len(df_inc)}")

    with st.expander("📈 Detalle de trades con realizedPnl (hoy)"):
        if df_tr.empty:
            st.caption("(Sin trades hoy)")
        else:
            st.dataframe(df_tr, use_container_width=True, hide_index=True)
            st.caption(f"Total trades: {len(df_tr)}")

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