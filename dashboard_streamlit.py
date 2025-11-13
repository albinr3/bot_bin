"""
Dashboard Simplificado para Grid Bot
=====================================
Versión simple y clara - Solo lo esencial
"""
import os
import time
import hmac
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ============================== CONFIG ========================================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BASE_URL = os.environ.get("FAPI_BASE_URL", "https://demo-fapi.binance.com").rstrip("/")
SYMBOL = os.environ.get("BINANCE_SYMBOL", "BTCUSDC").upper()
TZ_NAME = os.environ.get("DASHBOARD_TZ", "America/Santo_Domingo")

if not API_KEY or not API_SECRET:
    st.error("❌ Faltan credenciales en .env")
    st.stop()

# ============================== HTTP ==========================================
session = requests.Session()
session.headers.update({"X-MBX-APIKEY": API_KEY})
SECRET = API_SECRET.encode()

def _ts() -> int:
    return int(time.time() * 1000)

def _sign(params: dict) -> str:
    qs = urlencode(params)
    sig = hmac.new(SECRET, qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig

def api_call(endpoint: str, params: dict = None):
    """Llamada API simplificada con manejo de errores"""
    try:
        params = params or {}
        params.update({"timestamp": _ts(), "recvWindow": 10000})
        url = f"{BASE_URL}{endpoint}?{_sign(params)}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error API {endpoint}: {e}")
        return None

# ============================== FUNCIONES =====================================

def get_mark_price():
    """Obtiene precio mark del símbolo"""
    try:
        r = session.get(f"{BASE_URL}/fapi/v1/premiumIndex", 
                       params={"symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        return float(r.json()["markPrice"])
    except:
        return 0.0

def get_account_info():
    """Obtiene información de cuenta"""
    data = api_call("/fapi/v2/account")
    if not data:
        return 0.0, 0.0, 0.0, 0.0
    
    balance = float(data.get("totalWalletBalance", 0))
    upnl = float(data.get("totalUnrealizedProfit", 0))
    margin = float(data.get("totalInitialMargin", 0))
    maint = float(data.get("totalMaintMargin", 0))
    
    return balance, upnl, margin, maint

def get_positions():
    """Obtiene posiciones abiertas"""
    data = api_call("/fapi/v2/positionRisk", {"symbol": SYMBOL})
    if not data:
        return 0.0, 0.0
    
    long_amt = 0.0
    short_amt = 0.0
    
    for p in data:
        pos_amt = float(p.get("positionAmt", 0))
        if p.get("positionSide") == "LONG":
            long_amt = abs(pos_amt)
        elif p.get("positionSide") == "SHORT":
            short_amt = abs(pos_amt)
    
    return long_amt, short_amt

def get_open_orders():
    """Obtiene órdenes abiertas"""
    data = api_call("/fapi/v1/openOrders", {"symbol": SYMBOL})
    return data if data else []

def get_today_pnl():
    """Calcula PnL de hoy de forma simple"""
    # Inicio del día en UTC
    tz = ZoneInfo(TZ_NAME)
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    
    # Obtiene trades de hoy
    params = {
        "symbol": SYMBOL,
        "startTime": start_ms,
        "limit": 1000
    }
    
    trades = api_call("/fapi/v1/userTrades", params)
    if not trades:
        return 0.0, 0.0, []
    
    # Suma PnL realizado y comisiones
    pnl = sum(float(t.get("realizedPnl", 0)) for t in trades)
    commission = sum(float(t.get("commission", 0)) for t in trades)
    
    return pnl, commission, trades

def get_funding_today():
    """Obtiene funding fees de hoy"""
    tz = ZoneInfo(TZ_NAME)
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    
    params = {
        "symbol": SYMBOL,
        "incomeType": "FUNDING_FEE",
        "startTime": start_ms,
        "limit": 1000
    }
    
    income = api_call("/fapi/v1/income", params)
    if not income:
        return 0.0
    
    return sum(float(i.get("income", 0)) for i in income)

# ============================== UI ============================================

st.set_page_config(
    page_title=f"Bot Dashboard · {SYMBOL}",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(f"🤖 Grid Bot Dashboard · {SYMBOL}")
st.caption(f"{'🟢 DEMO' if 'demo' in BASE_URL else '🔴 LIVE'} · Actualizado: {datetime.now(ZoneInfo(TZ_NAME)).strftime('%H:%M:%S')}")

# ==================== SIDEBAR =================================================
with st.sidebar:
    st.header("⚙️ Config")
    refresh = st.slider("Actualizar cada (seg)", 3, 60, 30)
    st.divider()
    if st.button("🔄 Actualizar Ahora"):
        st.rerun()

# ==================== MÉTRICAS PRINCIPALES ====================================
st.header("📊 Estado General")

col1, col2, col3, col4 = st.columns(4)

# Precio
mark_price = get_mark_price()
col1.metric("💰 Precio Actual", f"${mark_price:,.2f}")

# Balance y UPnL
balance, upnl, margin, maint = get_account_info()
col2.metric("💵 Balance Total", f"${balance:,.2f}")
col3.metric("📊 PnL No Realizado", f"${upnl:,.2f}", 
           delta=f"{(upnl/balance*100):.2f}%" if balance > 0 else None)

# Margin Ratio
margin_ratio = (maint / balance * 100) if balance > 0 else 0
if margin_ratio < 25:
    status = "✅ Normal"
    color = "normal"
elif margin_ratio < 35:
    status = "⚠️ Cuidado"
    color = "inverse"
else:
    status = "🔴 Alto"
    color = "inverse"

col4.metric("⚠️ Uso de Margen", f"{margin_ratio:.1f}%", 
           delta=status, delta_color=color)

st.divider()

# ==================== POSICIONES ==============================================
st.header("📍 Posiciones Activas")

long_qty, short_qty = get_positions()

col1, col2, col3 = st.columns(3)

long_value = long_qty * mark_price
short_value = short_qty * mark_price
net = long_value - short_value

col1.metric("🟢 LONG", f"{long_qty:.4f}", 
           help=f"Valor: ${long_value:,.2f}")
col2.metric("🔴 SHORT", f"{short_qty:.4f}", 
           help=f"Valor: ${short_value:,.2f}")

# Balance de posiciones
if abs(net) < 100:
    net_status = "✅ Equilibrado"
else:
    net_status = "⚠️ Desbalanceado"

col3.metric("⚖️ Neto", f"${net:,.2f}", delta=net_status)

st.divider()

# ==================== ÓRDENES ABIERTAS ========================================
st.header("📝 Órdenes Abiertas")

orders = get_open_orders()

if not orders:
    st.info("Sin órdenes abiertas")
else:
    # Separar por tipo
    buy_orders = [o for o in orders if o.get("side") == "BUY"]
    sell_orders = [o for o in orders if o.get("side") == "SELL"]
    
    col1, col2 = st.columns(2)
    
    col1.metric("🟢 Órdenes BUY", len(buy_orders))
    col2.metric("🔴 Órdenes SELL", len(sell_orders))
    
    # Mostrar tabla de órdenes con valores reales invertidos
    if st.checkbox("Ver detalle de órdenes"):
        order_data = []
        total_invertido = 0.0
        total_valor_ordenes = 0.0
        
        # Obtener leverage de la configuración (por defecto 5x)
        leverage = int(os.getenv("LEVERAGE", "5"))
        
        for o in orders:
            precio = float(o.get('price', 0))
            cantidad = float(o.get('origQty', 0))
            valor_orden = precio * cantidad  # Valor total con apalancamiento
            margen_requerido = valor_orden / leverage  # Tu inversión real
            
            total_invertido += margen_requerido
            total_valor_ordenes += valor_orden
            
            order_data.append({
                "Lado": o.get("side"),
                "Tipo": o.get("type"),
                "Precio": f"${precio:,.2f}",
                "Mi Inversión": f"${margen_requerido:,.2f}",
                "Valor Orden": f"${valor_orden:,.2f}",
                "Estado": o.get("status")
            })
        
        # Mostrar totales
        col1, col2, col3 = st.columns(3)
        col1.metric("💵 Mi Capital Invertido", f"${total_invertido:,.2f}",
                   help=f"Capital real invertido (sin apalancamiento {leverage}x)")
        col2.metric("📊 Valor Total Órdenes", f"${total_valor_ordenes:,.2f}",
                   help=f"Valor total con apalancamiento {leverage}x")
        col3.metric("🔢 Apalancamiento", f"{leverage}x")
        
        df = pd.DataFrame(order_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ==================== PNL DEL DÍA =============================================
st.header("💰 Ganancias de Hoy")

pnl_trades, commission, trades = get_today_pnl()
funding = get_funding_today()

# Total del día
total_dia = pnl_trades + funding

col1, col2, col3, col4 = st.columns(4)

col1.metric("✅ PnL Realizado", f"${pnl_trades:,.2f}")
col2.metric("📊 Funding", f"${funding:,.2f}")
col3.metric("💳 Comisiones", f"${abs(commission):,.2f}", 
           help="Ya incluidas en PnL")

# Delta vs balance
delta_pct = (total_dia / balance * 100) if balance > 0 else 0
col4.metric("🎯 TOTAL HOY", f"${total_dia:,.2f}", 
           delta=f"{delta_pct:.2f}%")

# Contexto visual
if total_dia > 50:
    st.success(f"✅ Buen día! +${total_dia:.2f}")
elif total_dia < -100:
    st.error(f"⚠️ Pérdidas: ${total_dia:.2f}")
else:
    st.info(f"➡️ Neutral: ${total_dia:.2f}")

# Detalle de trades con selector de rango
if st.checkbox("Ver historial de trades"):
    st.subheader("📜 Historial de Trades")
    
    # Selector de rango de fechas
    col1, col2 = st.columns(2)
    
    with col1:
        tz = ZoneInfo(TZ_NAME)
        today = datetime.now(tz).date()
        fecha_inicio = st.date_input(
            "📅 Fecha inicio",
            value=today,
            max_value=today
        )
        
        col_h1, col_m1 = st.columns(2)
        with col_h1:
            hora_inicio_h = st.selectbox("Hora", range(0, 24), index=0, key="hora_inicio")
        with col_m1:
            hora_inicio_m = st.selectbox("Minuto", range(0, 60), index=0, key="min_inicio")
    
    with col2:
        fecha_fin = st.date_input(
            "📅 Fecha fin",
            value=today,
            max_value=today,
            min_value=fecha_inicio
        )
        
        col_h2, col_m2 = st.columns(2)
        with col_h2:
            hora_fin_h = st.selectbox("Hora", range(0, 24), index=23, key="hora_fin")
        with col_m2:
            hora_fin_m = st.selectbox("Minuto", range(0, 60), index=59, key="min_fin")
    
    # Combinar fecha y hora con zona horaria
    dt_inicio = datetime.combine(fecha_inicio, datetime.min.time()).replace(hour=hora_inicio_h, minute=hora_inicio_m)
    dt_fin = datetime.combine(fecha_fin, datetime.min.time()).replace(hour=hora_fin_h, minute=hora_fin_m)
    
    # Crear datetime con zona horaria
    dt_inicio_tz = dt_inicio.replace(tzinfo=tz)
    dt_fin_tz = dt_fin.replace(tzinfo=tz)
    
    # Convertir a UTC y milisegundos
    start_ms = int(dt_inicio_tz.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(dt_fin_tz.astimezone(timezone.utc).timestamp() * 1000)
    
    # Obtener trades del rango
    if st.button("🔍 Buscar Trades"):
        with st.spinner("Buscando trades..."):
            params = {
                "symbol": SYMBOL,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000
            }
            
            trades_rango = api_call("/fapi/v1/userTrades", params)
            
            if not trades_rango:
                st.warning("No se encontraron trades en este rango")
            else:
                st.success(f"✅ {len(trades_rango)} trades encontrados")
                
                # Calcular totales
                total_pnl_bruto = 0.0
                total_fees = 0.0
                total_pnl_neto = 0.0
                
                trade_data = []
                for t in trades_rango:
                    pnl_bruto = float(t.get('realizedPnl', 0))
                    fee = float(t.get('commission', 0))
                    pnl_neto = pnl_bruto - fee
                    
                    total_pnl_bruto += pnl_bruto
                    total_fees += fee
                    total_pnl_neto += pnl_neto
                    
                    trade_data.append({
                        "Fecha/Hora": datetime.fromtimestamp(
                            int(t.get("time", 0))/1000, 
                            tz=ZoneInfo(TZ_NAME)
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "Lado": t.get("side"),
                        "Posición": t.get("positionSide", "N/A"),
                        "Precio": f"${float(t.get('price', 0)):,.2f}",
                        "Cantidad": f"{float(t.get('qty', 0)):.4f}",
                        "PnL Bruto": f"${pnl_bruto:.2f}",
                        "Fee": f"${fee:.2f}",
                        "PnL Neto": f"${pnl_neto:.2f}"
                    })
                
                # Mostrar resumen con totales
                st.subheader("📊 Resumen del Periodo")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("📝 Total Trades", len(trades_rango))
                col2.metric("💰 PnL Bruto", f"${total_pnl_bruto:.2f}")
                col3.metric("💳 Fees Totales", f"${total_fees:.2f}")
                col4.metric("✅ PnL Neto", f"${total_pnl_neto:.2f}",
                           delta=f"{(total_pnl_neto/balance*100):.2f}%" if balance > 0 else None)
                
                # Mostrar tabla de trades
                st.subheader("📋 Detalle de Trades")
                df = pd.DataFrame(trade_data)
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
                
                # Opción de descargar CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name=f"trades_{fecha_inicio}_{fecha_fin}.csv",
                    mime="text/csv"
                )

st.divider()

# ==================== ALERTAS =================================================
st.header("⚠️ Alertas")

alerts = []

# Margin ratio alto
if margin_ratio > 35:
    alerts.append(("🔴 CRÍTICO", f"Margen muy alto: {margin_ratio:.1f}% (máx recomendado: 38%)"))
elif margin_ratio > 25:
    alerts.append(("🟡 ATENCIÓN", f"Margen elevado: {margin_ratio:.1f}%"))

# Balance bajo
if balance < 10000:
    alerts.append(("🟡 ATENCIÓN", f"Balance bajo: ${balance:,.2f}"))

# Pérdidas del día
if total_dia < -200:
    alerts.append(("🔴 CRÍTICO", f"Pérdidas significativas hoy: ${total_dia:.2f}"))
elif total_dia < -100:
    alerts.append(("🟡 ATENCIÓN", f"Pérdidas moderadas hoy: ${total_dia:.2f}"))

# Funding alto
if abs(funding) > 20:
    alerts.append(("🟡 ATENCIÓN", f"Funding alto: ${funding:.2f}"))

# Desbalance de posiciones
if abs(net) > 500:
    alerts.append(("🟡 ATENCIÓN", f"Posiciones desbalanceadas: ${net:,.2f}"))

# Mostrar alertas
if alerts:
    for level, msg in alerts:
        if "CRÍTICO" in level:
            st.error(f"{level}: {msg}")
        else:
            st.warning(f"{level}: {msg}")
else:
    st.success("✅ Todo normal - Sin alertas")

# ==================== FOOTER ==================================================
st.divider()
st.caption(f"Dashboard Simplificado v1.0 · Auto-refresh en {refresh}s")

# Auto-refresh
time.sleep(refresh)
st.rerun()