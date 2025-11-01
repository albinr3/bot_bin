r"""
Bot Grid + Hedge para USDS-M Futures (Binance) usando SDK si existe o REST oficial.

Cambios clave:
- Auto-TP basado en FILLS reales (/fapi/v1/userTrades), NO en entryPrice promedio.
- Bootstrap de last_trade_id al iniciar para NO procesar trades históricos.
- TP LIMIT maker (GTX) SIN reduceOnly (testnet suele rechazarlo); fallback a TAKE_PROFIT_MARKET por cantidad.
- Hedge por defecto en modo REDUCE (cierra parte de la posición existente). OFFSET opcional.
- Precio maker configurable con MAKER_EXTRA_TICKS (por defecto 2 ticks).
- Re-grid por time-stop: **rearma solo las órdenes vencidas** al mismo precio con la cantidad restante.
- Validación de MIN_NOTIONAL si existe.
- Backoff exponencial en el loop ante errores persistentes.
- Simulador opcional de fills en DRY_RUN (DRY_RUN_SIM=1) con de-duplicación de niveles.
- Tracking de orderIds del GRID para filtrar fills; ignoramos hedges/TPs.
- Kill-switch con doble cancelación defensiva de órdenes residuales.
- **NUEVO**: _last_trade_id se actualiza SOLO tras procesar TPs (evita perder fills si cae el bot).
- **NUEVO**: poda de _grid_order_ids contra openOrders (sin romper parciales).

Cómo usar (Windows)
-------------------
1) python -m venv .venv && .\.venv\Scripts\Activate.ps1
2) python -m pip install -U python-dotenv requests
3) Crear .env al lado del script:

BINANCE_API_KEY=TU_KEY
BINANCE_API_SECRET=TU_SECRET
BINANCE_SYMBOL=BTCUSDT
FAPI_BASE_URL=https://demo-fapi.binance.com
GRID_LEVELS_PER_SIDE=5
GRID_STEP_PCT=0.003
ORDER_NOTIONAL_USD=500
LEVERAGE=3
MARGIN_MODE=ISOLATED
HEDGE_ON_DRIFT_PCT=0.012
HEDGE_ON_NET_FRACTION=0.30
TIMESTOP_HOURS=3
MAX_MARGIN_RATIO=0.35
DRY_RUN=1
HEDGE_MODE=REDUCE              # REDUCE (recomendado) u OFFSET
MAKER_EXTRA_TICKS=2            # ticks extra para asegurar maker
DRY_RUN_SIM=0                  # 1 para simular fills en dry-run

Seguridad:
- Si DRY_RUN=0, exige que FAPI_BASE_URL contenga 'demo-fapi' para evitar enviar a real por error.
"""
from __future__ import annotations

import os
import time
import hmac
import hashlib
import logging
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# ============================== Carga de entorno ===============================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

BASE = os.getenv("FAPI_BASE_URL", "")
if os.getenv("DRY_RUN", "1") == "0":
    assert "demo-fapi" in BASE, "Bloqueado: DRY_RUN=0 pero no estás en demo-fapi (testnet)."

# ============================= Utilidades numéricas ============================

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    q = Decimal(str(step))
    v = Decimal(str(value))
    return float((v // q) * q)

def round_to_precision(value: float, precision: int) -> float:
    q = Decimal(10) ** -precision
    return float((Decimal(str(value))).quantize(q, rounding=ROUND_DOWN))

# ================================== Config ====================================

@dataclass
class Config:
    api_key: str
    api_secret: str
    base_url: str
    symbol: str = "BTCUSDT"
    levels_per_side: int = 5
    step_pct: float = 0.003
    order_notional: float = 500.0
    leverage: int = 3
    margin_mode: str = "ISOLATED"
    hedge_on_drift_pct: float = 0.012
    hedge_on_net_fraction: float = 0.30
    time_stop_hours: float = 3.0
    max_margin_ratio: float = 0.35
    dry_run: bool = True
    # Nuevos
    hedge_mode: str = "REDUCE"          # REDUCE (cierra parte) u OFFSET (abre opuesta)
    maker_extra_ticks: int = 2           # ticks extra para asegurar maker
    dry_run_sim: bool = False            # simulador de fills en dry-run

    @staticmethod
    def from_env() -> "Config":
        return Config(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            base_url=os.getenv("FAPI_BASE_URL", "https://demo-fapi.binance.com").rstrip("/"),
            symbol=os.getenv("BINANCE_SYMBOL", "BTCUSDT").upper(),
            levels_per_side=int(os.getenv("GRID_LEVELS_PER_SIDE", "5")),
            step_pct=float(os.getenv("GRID_STEP_PCT", "0.003")),
            order_notional=float(os.getenv("ORDER_NOTIONAL_USD", "500")),
            leverage=int(os.getenv("LEVERAGE", "3")),
            margin_mode=os.getenv("MARGIN_MODE", "ISOLATED").upper(),
            hedge_on_drift_pct=float(os.getenv("HEDGE_ON_DRIFT_PCT", "0.012")),
            hedge_on_net_fraction=float(os.getenv("HEDGE_ON_NET_FRACTION", "0.30")),
            time_stop_hours=float(os.getenv("TIMESTOP_HOURS", "3")),
            max_margin_ratio=float(os.getenv("MAX_MARGIN_RATIO", "0.35")),
            dry_run=os.getenv("DRY_RUN", "1") == "1",
            hedge_mode=os.getenv("HEDGE_MODE", "REDUCE").upper(),
            maker_extra_ticks=int(os.getenv("MAKER_EXTRA_TICKS", "2")),
            dry_run_sim=os.getenv("DRY_RUN_SIM", "0") == "1",
        )

# =========================== Cliente (SDK/REST) ================================

class FuturesClient:
    def mark_price(self, symbol: str) -> float: ...
    def exchange_info(self) -> Dict: ...
    def get_position_risk(self, symbol: str) -> List[Dict]: ...
    def account(self) -> Dict: ...
    def change_position_mode(self, dual: bool) -> Dict: ...
    def change_leverage(self, symbol: str, leverage: int) -> Dict: ...
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict: ...
    def new_order(self, **params) -> Optional[Dict]: ...
    def cancel_all_open_orders(self, symbol: str) -> Dict: ...
    def cancel_order(self, symbol: str, order_id: int) -> Dict: ...
    def get_open_orders(self, symbol: str) -> List[Dict]: ...
    def user_trades(self, symbol: str, limit: int = 50, from_id: Optional[int] = None) -> List[Dict]: ...

class RestClient(FuturesClient):
    def __init__(self, cfg: Config):
        self.key = cfg.api_key
        self.secret = cfg.api_secret.encode()
        self.base = cfg.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.key,
            "Content-Type": "application/x-www-form-urlencoded",
        })

    def _ts(self) -> int:
        return int(time.time() * 1000)

    def _qs_signed(self, params: Dict) -> str:
        qs = urlencode(params, doseq=True)
        sig = hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    def _get(self, path: str, params: Dict | None = None, signed: bool = False):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": 5000, **(params or {})})
            full = url + "?" + qs
            r = self.session.get(full, timeout=10)
        else:
            r = self.session.get(url, params=params or {}, timeout=10)
        if r.status_code >= 400:
            logging.error("GET %s %s", path, r.text)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: Dict | None = None, signed: bool = True):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": 5000, **(data or {})})
            r = self.session.post(url, data=qs, timeout=10)  # body form-urlencoded exacto
        else:
            r = self.session.post(url, data=(data or {}), timeout=10)
        if r.status_code >= 400:
            logging.error("POST %s %s", path, r.text)
        r.raise_for_status()
        return r.json() if r.text else {}

    def _delete(self, path: str, params: Dict | None = None, signed: bool = True):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": 5000, **(params or {})})
            full = url + "?" + qs
            r = self.session.delete(full, timeout=10)
        else:
            r = self.session.delete(url, params=params or {}, timeout=10)
        if r.status_code >= 400:
            logging.error("DELETE %s %s", path, r.text)
        r.raise_for_status()
        return r.json() if r.text else {}

    # ----- Métodos públicos -----
    def mark_price(self, symbol: str) -> float:
        j = self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(j["markPrice"])

    def exchange_info(self) -> Dict:
        return self._get("/fapi/v1/exchangeInfo")

    def get_position_risk(self, symbol: str) -> List[Dict]:
        return self._get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

    def account(self) -> Dict:
        return self._get("/fapi/v2/account", signed=True)

    def change_position_mode(self, dual: bool) -> Dict:
        return self._post("/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if dual else "false"})

    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        return self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        try:
            return self._post("/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
        except requests.HTTPError as e:
            logging.warning("change_margin_type warning: %s", e)
            return {}

    def new_order(self, **params) -> Optional[Dict]:
        try:
            return self._post("/fapi/v1/order", params)
        except requests.HTTPError as e:
            logging.error("new_order error: %s", e)
            return None

    def cancel_all_open_orders(self, symbol: str) -> Dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        return self._delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def get_open_orders(self, symbol: str) -> List[Dict]:
        return self._get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def user_trades(self, symbol: str, limit: int = 50, from_id: Optional[int] = None) -> List[Dict]:
        params = {"symbol": symbol, "limit": limit}
        if from_id is not None:
            params["fromId"] = from_id
        return self._get("/fapi/v1/userTrades", params, signed=True)

class SdkClient(RestClient):
    """Intenta usar el SDK; si no está, hereda REST."""
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._sdk = None
        try:
            import binance_sdk_derivatives_trading_usds_futures as usds
            if hasattr(usds, "Client"):
                self._sdk = usds.Client(api_key=cfg.api_key, api_secret=cfg.api_secret, base_url=cfg.base_url)
            elif hasattr(usds, "Api"):
                self._sdk = usds.Api(api_key=cfg.api_key, api_secret=cfg.api_secret, base_url=cfg.base_url)
            else:
                logging.warning("SDK USDS-M instalado pero sin clase conocida; usando REST")
        except Exception as e:
            logging.info("SDK USDS-M no disponible (%s); usando REST", e)

# ============================== Estrategia Grid + Hedge =========================

@dataclass
class GridState:
    center: float
    last_recenter_ts: float

class GridHedgeBot:
    def __init__(self, client: FuturesClient, cfg: Config):
        self.c = client
        self.cfg = cfg
        self.state = GridState(center=self.c.mark_price(cfg.symbol), last_recenter_ts=time.time())
        # Defaults seguros; se sobreescriben con exchangeInfo()
        self.price_precision = 2
        self.qty_precision = 6
        self.tick_size = 0.1
        self.step_size = 0.000001
        self.min_notional = None
        # Track de fills y grid
        self._last_trade_id: Optional[int] = None
        self._grid_order_ids: set[int] = set()   # ids de órdenes del grid activas
        self._simulated_fills = set()            # de-duplicación del simulador
        # Caches (si en algún momento miras deltas por posición)
        self._prev_long_qty = 0.0
        self._prev_short_qty = 0.0
        self._load_filters()

    def _load_filters(self):
        info = self.c.exchange_info()
        sym = None
        for s in info.get("symbols", []):
            if s.get("symbol") == self.cfg.symbol:
                sym = s
                break
        if not sym:
            raise RuntimeError(f"Símbolo no disponible: {self.cfg.symbol}")
        fs = {f["filterType"]: f for f in sym.get("filters", [])}
        self.price_precision = sym.get("pricePrecision", 2)
        self.qty_precision = sym.get("quantityPrecision", 6)
        self.tick_size = float(fs["PRICE_FILTER"]["tickSize"]) if "PRICE_FILTER" in fs else 0.1
        self.step_size = float(fs["LOT_SIZE"]["stepSize"]) if "LOT_SIZE" in fs else 0.000001
        if "MIN_NOTIONAL" in fs:
            try:
                self.min_notional = float(fs["MIN_NOTIONAL"]["notional"])
            except Exception:
                self.min_notional = None
        logging.info(
            "Symbol filters: tick_size=%s step_size=%s price_precision=%s qty_precision=%s min_notional=%s",
            self.tick_size, self.step_size, self.price_precision, self.qty_precision, self.min_notional
        )

    # -------------------- Helpers de cantidad / precio ------------------

    @staticmethod
    def _decimals_from_step(step: float) -> int:
        d = Decimal(str(step)).normalize()
        return -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0

    def _snap_to_tick(self, price: float) -> float:
        ts = self.tick_size
        ticks = math.floor(price / ts + 1e-12)  # evita glitches de float
        p = ticks * ts
        decs = self._decimals_from_step(ts)
        return float(Decimal(str(p)).quantize(Decimal('1.' + '0'*decs)) if decs > 0 else Decimal(int(p)))

    def _maker_price(self, is_buy: bool, raw_price: float) -> float:
        """
        Devuelve un precio válido maker:
        - múltiplo exacto de tickSize
        - post-only (GTX) alejándolo K ticks del cruce (K = maker_extra_ticks)
        """
        ts = self.tick_size
        k = max(1, int(self.cfg.maker_extra_ticks))
        if is_buy:
            p = math.floor(raw_price / ts) * ts - k * ts
        else:
            p = math.ceil(raw_price / ts) * ts + k * ts
        p = max(ts, p)
        return self._snap_to_tick(p)

    def _qty_from_notional(self, notional_usd: float, price: float) -> float:
        # Ajuste a MIN_NOTIONAL si existe
        if self.min_notional is not None and notional_usd < self.min_notional:
            notional_usd = self.min_notional
        qty = notional_usd / price
        qty = max(self.step_size, round_step(qty, self.step_size))
        return round_to_precision(qty, self.qty_precision)

    # ------------------------- Colocación de grid -----------------------

    def _place_limit(self, side: str, pos_side: str, price: float, qty: float,
                     reduce_only: bool = False, tag: str = ""):
        price = self._maker_price(side.upper() == "BUY", price)
        # Validación de min_notional
        if self.min_notional is not None and price * qty < self.min_notional - 1e-8:
            logging.warning("skip order: notional %.2f < MIN_NOTIONAL %.2f", price * qty, self.min_notional)
            return None

        logging.info("ORDER %s %s qty=%.8f @ %.2f reduceOnly=%s %s",
                     side, pos_side, qty, price, reduce_only, tag)
        if self.cfg.dry_run:
            return None

        params = {
            "symbol": self.cfg.symbol,
            "side": side.upper(),          # BUY / SELL
            "type": "LIMIT",
            "timeInForce": "GTX",          # Post-Only (maker)
            "quantity": str(qty),
            "price": str(price),
            "positionSide": pos_side.upper()
        }
        # LIMIT soporta reduceOnly; para TPs evitamos reduceOnly por errores de testnet
        if reduce_only:
            params["reduceOnly"] = "true"

        return self.c.new_order(**params)

    def _reset_grid_tracking(self):
        self._grid_order_ids.clear()

    def place_take_profit_for_fill(self, side: str, pos_side: str, fill_price: float, qty: float):
        """
        1) TP LIMIT maker (GTX) usando positionSide correcto y SIN reduceOnly (testnet a veces rechaza reduceOnly).
        2) Si falla, fallback a TAKE_PROFIT_MARKET con 'quantity' (sin reduceOnly/closePosition).
           En dual-hedge + positionSide, eso reduce la posición de ese lado.
        """
        if side.upper() == "BUY":
            tp = fill_price * (1 + self.cfg.step_pct)
            close_side = "SELL"   # cerrar LONG
        else:
            tp = fill_price * (1 - self.cfg.step_pct)
            close_side = "BUY"    # cerrar SHORT

        logging.info("TP PLAN: close_side=%s pos_side=%s fill_px=%.2f qty=%.6f", close_side, pos_side, fill_price, qty)

        # intento 1: LIMIT maker (GTX) sin reduceOnly (positionSide garantiza reducción)
        r = self._place_limit(close_side, pos_side, tp, qty, reduce_only=False, tag="TP")
        if r is not None:
            return r

        # fallback: TAKE_PROFIT_MARKET por cantidad (sin reduceOnly/closePosition)
        stop = self._snap_to_tick(tp)
        params = {
            "symbol": self.cfg.symbol,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": str(stop),
            "positionSide": pos_side.upper(),
            "workingType": "MARK_PRICE",
            "quantity": str(round_step(qty, self.step_size))
        }
        logging.warning("TP LIMIT rechazado; usando TAKE_PROFIT_MARKET qty=%s stop=%s", params["quantity"], stop)
        if self.cfg.dry_run:
            return None
        try:
            return self.c.new_order(**params)
        except Exception as e:
            logging.error("fallback TP error: %s", e)
            return None

    def build_grid_orders(self, side_mode: str = "BOTH") -> List[Tuple[str, str, float, float, bool]]:
        mp = self.c.mark_price(self.cfg.symbol)
        self.state.center = mp
        out: List[Tuple[str, str, float, float, bool]] = []
        for i in range(1, self.cfg.levels_per_side + 1):
            off = mp * self.cfg.step_pct * i
            if side_mode in ("BOTH", "SHORT_ONLY"):
                p = mp + off
                q = self._qty_from_notional(self.cfg.order_notional, p)
                out.append(("SELL", "SHORT", p, q, False))
            if side_mode in ("BOTH", "LONG_ONLY"):
                p = mp - off
                q = self._qty_from_notional(self.cfg.order_notional, p)
                out.append(("BUY", "LONG", p, q, False))
        return out

    def place_grid(self, side_mode: str = "BOTH"):
        # Reinicia tracking porque estás armando un grid nuevo
        self._reset_grid_tracking()
        for side, ps, price, qty, ro in self.build_grid_orders(side_mode):
            res = self._place_limit(side, ps, price, qty, reduce_only=ro)
            # Guarda orderIds del GRID (si no estás en dry_run y la orden fue aceptada)
            if res and isinstance(res, dict) and "orderId" in res:
                try:
                    self._grid_order_ids.add(int(res["orderId"]))
                except Exception:
                    pass

    # -------------------------- Gestión de riesgo -----------------------

    def net_exposure_notional(self) -> float:
        pr = self.c.get_position_risk(self.cfg.symbol)
        mp = self.c.mark_price(self.cfg.symbol)
        long_n = 0.0
        short_n = 0.0
        for p in pr:
            side = p.get("positionSide")
            qty = float(p.get("positionAmt", 0.0))
            if qty == 0:
                continue
            if side == "LONG" and qty > 0:
                long_n += qty * mp
            elif side == "SHORT" and qty < 0:
                short_n += abs(qty) * mp
        return long_n - short_n

    def maybe_hedge_and_recenter(self):
        mp = self.c.mark_price(self.cfg.symbol)
        drift = abs(mp - self.state.center) / max(self.state.center, 1e-9)
        net_n = self.net_exposure_notional()
        gross_grid_n = self.cfg.order_notional * self.cfg.levels_per_side * 2
        need = (drift >= self.cfg.hedge_on_drift_pct) or (abs(net_n) >= self.cfg.hedge_on_net_fraction * gross_grid_n)
        if not need:
            return
        logging.info("Recentrar/hedge: drift=%.3f net=%.2f / gross=%.2f", drift, net_n, gross_grid_n)

        # 1) Cancelar órdenes viejas (mantenemos posiciones)
        self.c.cancel_all_open_orders(self.cfg.symbol)
        self._reset_grid_tracking()

        # 2) Hedge según modo (por defecto REDUCE)
        if abs(net_n) > 0:
            qty = self._qty_from_notional(abs(net_n), mp)
            if self.cfg.hedge_mode == "OFFSET":
                if net_n > 0:
                    self._place_limit("SELL", "SHORT", mp, qty, reduce_only=False, tag="hedge-offset")
                else:
                    self._place_limit("BUY", "LONG", mp, qty, reduce_only=False, tag="hedge-offset")
            else:
                if net_n > 0:
                    self._place_limit("SELL", "LONG", mp, qty, reduce_only=True, tag="hedge-reduce")
                else:
                    self._place_limit("BUY", "SHORT", mp, qty, reduce_only=True, tag="hedge-reduce")

        # 3) Recentrar y reponer grid
        self.state.center = mp
        self.state.last_recenter_ts = time.time()
        self.place_grid(side_mode="BOTH")

    def kill_switch_if_needed(self):
        if self.cfg.dry_run:
            return
        acc = self.c.account()
        bal = float(acc.get("totalWalletBalance", 0.0))
        maint = float(acc.get("totalMaintMargin", 0.0))
        ratio = (maint / bal) if bal else 0.0
        if ratio >= self.cfg.max_margin_ratio:
            logging.error("KILL-SWITCH: ratio %.2f ≥ %.2f", ratio, self.cfg.max_margin_ratio)
            # Cancelación “amplia”
            self.c.cancel_all_open_orders(self.cfg.symbol)
            # Barrido defensivo por si quedara alguna stop/oco colgada
            try:
                oo_left = self.c.get_open_orders(self.cfg.symbol)
                for o in oo_left:
                    try:
                        self.c.cancel_order(self.cfg.symbol, int(o["orderId"]))
                    except Exception as e:
                        logging.warning("cancel residual %s err: %s", o.get("orderId"), e)
            except Exception as e:
                logging.warning("post-cancel openOrders err: %s", e)

            # Reduce neto
            net_n = self.net_exposure_notional()
            if abs(net_n) > 0:
                mp = self.c.mark_price(self.cfg.symbol)
                qty = self._qty_from_notional(abs(net_n), mp)
                if net_n > 0:
                    self._place_limit("SELL", "LONG", mp, qty, reduce_only=True, tag="hedge-kill")   # reduce LONG
                else:
                    self._place_limit("BUY", "SHORT", mp, qty, reduce_only=True, tag="hedge-kill")  # reduce SHORT
            raise SystemExit("Detenido por seguridad")

    # ----------------------- Auto-TP via /userTrades ---------------------

    def bootstrap_last_trade_id(self):
        """Ancla _last_trade_id al último trade existente para NO procesar históricos."""
        try:
            trades = self.c.user_trades(self.cfg.symbol, limit=1)
            if trades:
                self._last_trade_id = int(trades[-1]["id"])
                logging.info("Bootstrap last_trade_id=%s (saltando históricos)", self._last_trade_id)
            else:
                self._last_trade_id = -1
        except Exception as e:
            logging.warning("bootstrap_last_trade_id err: %s", e)
            self._last_trade_id = -1

    def fetch_new_fills(self) -> List[Dict]:
        """
        Lee trades recientes y devuelve solo los nuevos (posteriores a _last_trade_id),
        filtrando para procesar SOLO fills de órdenes del GRID (por orderId).
        NO actualiza _last_trade_id aquí (se hace en el loop principal tras procesar).
        """
        try:
            if self._last_trade_id is None:
                self.bootstrap_last_trade_id()
                return []
            trades = self.c.user_trades(self.cfg.symbol, limit=50, from_id=(self._last_trade_id + 1))
            trades = sorted(trades, key=lambda x: int(x["id"]))
            out: List[Dict] = []
            for tr in trades:
                tid = int(tr["id"])
                if tid <= self._last_trade_id:
                    continue
                try:
                    oid = int(tr.get("orderId", -1))
                    if oid in self._grid_order_ids:
                        out.append(tr)
                except Exception:
                    pass
            return out
        except Exception as e:
            logging.warning("fetch_new_fills err: %s", e)
            return []

    # ------------------------- Simulador (dry-run) -----------------------

    def simulate_fills_against_grid(self):
        """
        Simula fills si el mark cruza niveles del grid (solo dry-run y si DRY_RUN_SIM=1).
        Usa un set para evitar TPs duplicados en el mismo nivel.
        """
        if not self.cfg.dry_run or not self.cfg.dry_run_sim:
            return
        mp = self.c.mark_price(self.cfg.symbol)
        step = self.cfg.step_pct
        levels = []
        for i in range(1, self.cfg.levels_per_side + 1):
            levels.append(("SELL", "SHORT", self.state.center * (1 + step * i)))
            levels.append(("BUY", "LONG",  self.state.center * (1 - step * i)))
        for side, ps, p in levels:
            key = (side, round(p, 2))
            if key in self._simulated_fills:
                continue
            if (side == "BUY" and mp <= p) or (side == "SELL" and mp >= p):
                self._simulated_fills.add(key)
                q = self._qty_from_notional(self.cfg.order_notional, p)
                logging.info("SIM FILL %s %s qty=%.6f @ %.2f", side, ps, q, p)
                self.place_take_profit_for_fill(side, ps, p, q)

    # -------------------------------- Loop --------------------------------

    def preflight(self):
        if not self.cfg.api_key or not self.cfg.api_secret:
            raise RuntimeError("Faltan BINANCE_API_KEY/SECRET")
        try:
            self.c.change_position_mode(True)  # Hedge mode
        except Exception as e:
            logging.warning("position mode: %s", e)
        try:
            self.c.change_leverage(self.cfg.symbol, self.cfg.leverage)
        except Exception as e:
            logging.warning("leverage: %s", e)
        try:
            self.c.change_margin_type(self.cfg.symbol, self.cfg.margin_mode)
        except Exception as e:
            logging.warning("margin type: %s", e)
        logging.info("Centro inicial: %.2f", self.state.center)
        # Evita TPs por histórico:
        self.bootstrap_last_trade_id()

    def run(self):
        self.preflight()
        self.place_grid(side_mode="BOTH")
        logging.info("Grid inicial colocado")

        err_count = 0
        while True:
            try:
                # Seguridad
                self.kill_switch_if_needed()

                # Gestión de grid/hedge
                self.maybe_hedge_and_recenter()

                # Auto-TP por FILLS reales (solo grid)
              
                fills = self.fetch_new_fills()
                max_ok_id = None
                for tr in fills:
                    try:
                        side = tr.get("side", "").upper()      # BUY/SELL de la ORDEN
                        qty  = float(tr.get("qty", "0"))
                        px   = float(tr.get("price", "0"))
                        if not qty or not px:
                            continue
                        # Solo grid → inferencia es correcta
                        pos_side = "LONG" if side == "BUY" else "SHORT"
                        logging.info("FILL DETECTED: id=%s orderId=%s side=%s qty=%s px=%s",
                                    tr.get("id"), tr.get("orderId"), side, qty, px)
                        
                        # ✅ NUEVA LÍNEA: guarda el resultado del TP
                        tp_result = self.place_take_profit_for_fill(side, pos_side, px, qty)
                        
                        # ✅ NUEVA LÓGICA: solo marca como procesado si el TP se colocó
                        if tp_result is not None or self.cfg.dry_run:
                            # TP exitoso O estamos en dry-run (donde tp_result siempre es None)
                            max_ok_id = int(tr["id"])
                        else:
                            # TP falló (rechazado por Binance)
                            logging.warning(
                                "TP falló para fill id=%s (orderId=%s). Reintentará en próximo ciclo.",
                                tr.get("id"), tr.get("orderId")
                            )
                            # NO actualizar max_ok_id → este fill se re-procesará
                            
                    except Exception as e:
                        logging.exception("error procesando fill id=%s: %s", tr.get("id"), e)

                if max_ok_id is not None:
                    self._last_trade_id = max_ok_id

                # Re-arma SOLO órdenes vencidas por time-stop (cancelación selectiva + reponer)
                oo = self.c.get_open_orders(self.cfg.symbol)
                now = time.time()

                stale_orders = []
                open_ids = set()
                for o in oo:
                    try:
                        open_ids.add(int(o["orderId"]))
                    except Exception:
                        pass
                    t = (o.get("time") or o.get("updateTime") or 0) / 1000.0
                    if t and (now - t) / 3600.0 >= self.cfg.time_stop_hours:
                        stale_orders.append(o)

                for o in stale_orders:
                    oid = int(o["orderId"])
                    side = o.get("side", "BUY").upper()
                    pos_side = o.get("positionSide", "BOTH").upper()
                    price = float(o.get("price", "0"))
                    # cantidad restante si hubo parcial
                    try:
                        orig_qty = float(o.get("origQty", "0"))
                        exec_qty = float(o.get("executedQty", o.get("cumQty", "0")))
                        qleft = max(orig_qty - exec_qty, 0.0) if orig_qty else 0.0
                    except Exception:
                        qleft = float(o.get("origQty", "0")) or 0.0

                    try:
                        self.c.cancel_order(self.cfg.symbol, oid)
                        self._grid_order_ids.discard(oid)
                    except Exception as e:
                        logging.warning("cancel stale %s err: %s", oid, e)
                        continue

                    if qleft > 0:
                        res = self._place_limit(side, pos_side, price, qleft, reduce_only=False, tag="regen-time")
                        if res and "orderId" in res:
                            try:
                                self._grid_order_ids.add(int(res["orderId"]))
                            except Exception:
                                pass

                # Poda defensiva del set contra órdenes realmente abiertas
                try:
                    oo2 = self.c.get_open_orders(self.cfg.symbol)
                    still_open = {int(x["orderId"]) for x in oo2}
                    self._grid_order_ids.intersection_update(still_open)
                except Exception as e:
                    logging.warning("prune grid ids err: %s", e)

                # Simulador (solo si DRY_RUN_SIM=1)
                self.simulate_fills_against_grid()

                # éxito → resetea backoff
                err_count = 0
                time.sleep(2)

            except Exception as e:
                err_count += 1
                delay = min(60, 2 ** min(err_count, 5))  # 2,4,8,16,32,32,... tope 60
                logging.exception("Loop error (%s). Backoff %ss", e, delay)
                time.sleep(delay)

# ================================== Main =======================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    cfg = Config.from_env()
    client = SdkClient(cfg)  # usa SDK si existe; si no, REST
    bot = GridHedgeBot(client, cfg)
    bot.run()
