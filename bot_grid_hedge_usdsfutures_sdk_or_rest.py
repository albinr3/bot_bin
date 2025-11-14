from __future__ import annotations
import os
import time
import math
import hmac
import json
import hashlib
import logging
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from logging.handlers import TimedRotatingFileHandler

# =========================
#  CARGA .env
# =========================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

BASE = os.getenv("FAPI_BASE_URL", "https://demo-fapi.binance.com").rstrip("/")
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
AIRBAG_REQUIRE_DEMO = os.getenv("AIRBAG_REQUIRE_DEMO", "0") == "1"

if not DRY_RUN and AIRBAG_REQUIRE_DEMO:
    assert "demo-fapi" in BASE, "Bloqueado: DRY_RUN=0 pero AIRBAG exige demo-fapi."

# =========================
#  HELPERS NUMÉRICOS
# =========================

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    q = Decimal(str(step))
    v = Decimal(str(value))
    return float((v // q) * q)

def round_to_precision(value: float, precision: int) -> float:
    q = Decimal(10) ** -precision
    return float((Decimal(str(value))).quantize(q, rounding=ROUND_DOWN))

def decimals_from_step(step: float) -> int:
    d = Decimal(str(step)).normalize()
    return -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0

# =========================
#  CONFIG
# =========================

@dataclass
class Config:
    api_key: str
    api_secret: str
    base_url: str
    symbol: str = "BTCUSDC"
    levels_per_side: int = 10
    step_pct: float = 0.0020
    order_notional: float = 300.0
    leverage: int = 5
    margin_mode: str = "ISOLATED"
    min_available_balance_usd: float = 500.0

    hedge_on_drift_pct: float = 0.012
    hedge_on_net_fraction: float = 0.25
    hedge_fraction_min: float = 0.50
    hedge_fraction_max: float = 0.70

    recenter_drift_partial: float = 0.012
    recenter_drift_full: float = 0.025

    timestop_hours_base: float = 6.0
    timestop_hours_fast: float = 2.0
    drift_fast_pct: float = 0.008

    funding_rate_warn_8h: float = 0.0004
    funding_fee_warn_usd: float = 15.0

    panic_range_5m_pct: float = 0.032
    unpause_range_5m_pct: float = 0.008
    unpause_stable_blocks: int = 4

    max_margin_ratio: float = 0.35
    max_same_side_levels: int = 10
    daily_stop_loss_usd: float = 260.0
    circuit_breaker_loss_l1: float = 150.0
    circuit_breaker_loss_l2: float = 250.0
    max_hedges_per_4h: int = 3

    dry_run: bool = True
    maker_ticks_away: int = 2
    time_stop_cancel_only: bool = False

    recv_window_ms: int = 10000
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 25.0
    http_retry_total: int = 5
    http_retry_connect: int = 3
    http_retry_read: int = 3
    http_backoff_factor: float = 0.5
    http_status_forcelist: tuple = (429, 500, 502, 503, 504)

    # === Logging / Events flags ===
    event_include_pnl: bool = False
    event_include_oid: bool = True
    snapshot_freq_min: int = 60

    @staticmethod
    def from_env() -> "Config":
        # Clamp de la frecuencia de snapshots a [1, +∞)
        snapshot_freq_min = int(os.getenv("SNAPSHOT_FREQ_MIN", "60"))
        snapshot_freq_min = max(1, snapshot_freq_min)

        return Config(
            event_include_pnl=os.getenv("EVENT_INCLUDE_PNL", "0") == "1",
            event_include_oid=os.getenv("EVENT_INCLUDE_OID", "1") == "1",
            snapshot_freq_min=snapshot_freq_min,

            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            base_url=os.getenv("FAPI_BASE_URL", BASE),
            symbol=os.getenv("BINANCE_SYMBOL", "BTCUSDC").upper(),
            levels_per_side=int(os.getenv("GRID_LEVELS_PER_SIDE", "10")),
            step_pct=float(os.getenv("GRID_STEP_PCT", "0.0020")),
            order_notional=float(os.getenv("ORDER_NOTIONAL_USD", "300")),
            leverage=int(os.getenv("LEVERAGE", "5")),
            margin_mode=os.getenv("MARGIN_MODE", "ISOLATED").upper(),
            min_available_balance_usd=float(os.getenv("MIN_AVAILABLE_BALANCE_USD","500")),

            hedge_on_drift_pct=float(os.getenv("HEDGE_ON_DRIFT_PCT", "0.012")),
            hedge_on_net_fraction=float(os.getenv("HEDGE_ON_NET_FRACTION", "0.25")),
            hedge_fraction_min=float(os.getenv("HEDGE_FRACTION_MIN", "0.50")),
            hedge_fraction_max=float(os.getenv("HEDGE_FRACTION_MAX", "0.70")),

            recenter_drift_partial=float(os.getenv("RECENTER_DRIFT_PARTIAL", "0.012")),
            recenter_drift_full=float(os.getenv("RECENTER_DRIFT_FULL", "0.025")),

            timestop_hours_base=float(os.getenv("TIMESTOP_HOURS_BASE", "6")),
            timestop_hours_fast=float(os.getenv("TIMESTOP_HOURS_FAST", "2")),
            drift_fast_pct=float(os.getenv("DRIFT_FAST_PCT", "0.008")),

            funding_rate_warn_8h=float(os.getenv("FUNDING_RATE_WARN_8H", "0.0004")),
            funding_fee_warn_usd=float(os.getenv("FUNDING_FEE_WARN_USD", "15")),

            panic_range_5m_pct=float(os.getenv("PANIC_RANGE_5M_PCT", "0.032")),
            unpause_range_5m_pct=float(os.getenv("UNPAUSE_RANGE_5M_PCT", "0.008")),
            unpause_stable_blocks=int(os.getenv("UNPAUSE_STABLE_BLOCKS", "4")),

            max_margin_ratio=float(os.getenv("MAX_MARGIN_RATIO", "0.35")),
            max_same_side_levels=int(os.getenv("MAX_SAME_SIDE_LEVELS", "10")),
            daily_stop_loss_usd=float(os.getenv("DAILY_STOP_LOSS_USD", "260")),
            circuit_breaker_loss_l1=float(os.getenv("CIRCUIT_BREAKER_L1", "150")),
            circuit_breaker_loss_l2=float(os.getenv("CIRCUIT_BREAKER_L2", "250")),
            max_hedges_per_4h=int(os.getenv("MAX_HEDGES_PER_4H", "3")),

            dry_run=os.getenv("DRY_RUN", "1") == "1",
            maker_ticks_away=int(os.getenv("MAKER_TICKS_AWAY", "2")),
            time_stop_cancel_only=os.getenv("TIME_STOP_CANCEL_ONLY", "0") == "1",

            recv_window_ms=int(os.getenv("RECV_WINDOW_MS", "10000")),
            http_connect_timeout=float(os.getenv("HTTP_CONNECT_TIMEOUT", "5")),
            http_read_timeout=float(os.getenv("HTTP_READ_TIMEOUT", "25")),
            http_retry_total=int(os.getenv("HTTP_RETRY_TOTAL", "5")),
            http_retry_connect=int(os.getenv("HTTP_RETRY_CONNECT", "3")),
            http_retry_read=int(os.getenv("HTTP_RETRY_READ", "3")),
            http_backoff_factor=float(os.getenv("HTTP_BACKOFF_FACTOR", "0.5")),
        )


# =========================
#  CLIENTE REST
# =========================

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
    def get_open_orders(self, symbol: str) -> List[Dict]: ...
    def get_order(self, symbol: str, origClientOrderId: str | None = None, orderId: int | None = None) -> Dict: ...
    def cancel_order(self, symbol: str, order_id: int) -> Dict: ...
    def user_trades(self, symbol: str, from_id: int | None = None, limit: int = 1000,
                    start_time: int | None = None, end_time: int | None = None) -> List[Dict]: ...
    def premium_index(self, symbol: str) -> Dict: ...
    def income_history(self, symbol: str, start_time: int | None = None,
                      end_time: int | None = None, limit: int = 1000) -> List[Dict]: ...
    def recreate_session(self): ...

class RestClient(FuturesClient):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.key = cfg.api_key
        self.secret = cfg.api_secret.encode()
        self.base = cfg.base_url.rstrip("/")
        self._build_session()

    def _build_session(self):
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.key,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        retry = Retry(
            total=self.cfg.http_retry_total,
            connect=self.cfg.http_retry_connect,
            read=self.cfg.http_retry_read,
            backoff_factor=self.cfg.http_backoff_factor,
            status_forcelist=self.cfg.http_status_forcelist,
            allowed_methods={"GET", "POST", "DELETE"},
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.timeout = (self.cfg.http_connect_timeout, self.cfg.http_read_timeout)

    def recreate_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self._build_session()

    def _ts(self) -> int:
        return int(time.time() * 1000)

    def _qs_signed(self, params: Dict) -> str:
        qs = urlencode(params, doseq=True)
        sig = hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    def _get(self, path: str, params: Dict | None = None, signed: bool = False):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": self.cfg.recv_window_ms, **(params or {})})
            full = url + "?" + qs
            r = self.session.get(full, timeout=self.timeout)
        else:
            r = self.session.get(url, params=params or {}, timeout=self.timeout)
        # Logging de errores GET
        if r.status_code >= 400:
            try:
                j = r.json() if r.text else {}
            except Exception:
                j = {}
            logging.error("GET %s %s", path, j.get("msg", r.text))
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: Dict | None = None, signed: bool = True):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": self.cfg.recv_window_ms, **(data or {})})
            r = self.session.post(url, data=qs, timeout=self.timeout)
        else:
            r = self.session.post(url, data=(data or {}), timeout=self.timeout)
        # Manejo de no-ops 4059/4046
        try:
            j = r.json() if r.text else {}
        except Exception:
            j = {}
        if r.status_code >= 400:
            code = j.get("code")
            if code in (-4059, -4046):
                logging.info("No-op POST %s: %s (code %s)", path, j.get("msg", ""), code)
                return {"ok": True, "_noop": True, "code": code, "msg": j.get("msg", "")}
            logging.error("POST %s %s", path, r.text)
            r.raise_for_status()
            return j
        return j if j is not None else {}

    def _delete(self, path: str, params: Dict | None = None, signed: bool = True):
        url = f"{self.base}{path}"
        if signed:
            qs = self._qs_signed({"timestamp": self._ts(), "recvWindow": self.cfg.recv_window_ms, **(params or {})})
            full = url + "?" + qs
            r = self.session.delete(full, timeout=self.timeout)
        else:
            r = self.session.delete(url, params=params or {}, timeout=self.timeout)
        if r.status_code >= 400:
            logging.error("DELETE %s %s", path, r.text)
        r.raise_for_status()
        return r.json() if r.text else {}

    def mark_price(self, symbol: str) -> float:
        j = self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(j["markPrice"])

    def premium_index(self, symbol: str) -> Dict:
        return self._get("/fapi/v1/premiumIndex", {"symbol": symbol})

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
            logging.warning("change_margin_type: %s", e)
            return {}

    def new_order(self, **params) -> Optional[Dict]:
        try:
            return self._post("/fapi/v1/order", params)
        except requests.HTTPError as e:
            logging.error("new_order error: %s", e)
            return None

    def get_open_orders(self, symbol: str) -> List[Dict]:
        return self._get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> Dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        return self._delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def get_order(self, symbol: str, origClientOrderId: str | None = None, orderId: int | None = None) -> Dict:
        p = {"symbol": symbol}
        if origClientOrderId:
            p["origClientOrderId"] = origClientOrderId
        if orderId:
            p["orderId"] = orderId
        return self._get("/fapi/v1/order", p, signed=True)

    def user_trades(self, symbol: str, from_id: int | None = None, limit: int = 1000,
                    start_time: int | None = None, end_time: int | None = None) -> List[Dict]:
        p = {"symbol": symbol, "limit": limit}
        if from_id is not None:
            p["fromId"] = from_id
        if start_time:
            p["startTime"] = start_time
        if end_time:
            p["endTime"] = end_time
        return self._get("/fapi/v1/userTrades", p, signed=True)

    def income_history(self, symbol: str, start_time: int | None = None,
                      end_time: int | None = None, limit: int = 1000) -> List[Dict]:
        p = {"symbol": symbol, "limit": limit}
        if start_time:
            p["startTime"] = start_time
        if end_time:
            p["endTime"] = end_time
        return self._get("/fapi/v1/income", p, signed=True)

# =========================
#  ESTADO
# =========================

@dataclass
class GridState:
    center: float
    last_recenter_ts: float

@dataclass
class HedgePosition:
    side: str  # "LONG" o "SHORT" (posición hedge)
    qty: float
    entry_price: float
    entry_time: float
    notional: float

# =========================
#  BOT
# =========================
LOCAL_TZ = timezone(timedelta(hours=-4))  # República Dominicana (UTC-4)

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def utcnow():
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")

class GridHedgeBot:
    def __init__(self, client: FuturesClient, cfg: Config):
        self._last_drift_print = None

        self.c = client
        self.cfg = cfg
        self.state = GridState(center=self.c.mark_price(cfg.symbol), last_recenter_ts=time.time())
        self._last_drift_log_ts = 0.0
        self.price_precision = 2
        self.qty_precision = 6
        self.tick_size = 0.1
        self.step_size = 0.001
        self.min_notional = 5.0
        self.min_qty = 0.0
        self._load_filters()

        self._grid_order_ids: set[int] = set()
        self._grid_order_meta: dict[int, Tuple[str,str,float,float]] = {}
        self._last_trade_id: Optional[int] = None

        self._price_history_5m: List[Tuple[float, float]] = []  # para pánico
        self._price_hist_long: List[Tuple[float, float]] = []   # para trending 4h
        self._paused = False
        self._stable_blocks = 0

        self._active_hedges: List[HedgePosition] = []
        # Historial: (timestamp, "open"/"close", side[LONG/SHORT])
        self._hedge_history: List[Tuple[float, str, str]] = []

        self._pnl_cache: Optional[float] = None
        self._pnl_cache_time: float = 0
        self._pnl_cache_ttl: float = 60.0

        self._daily_start_balance: Optional[float] = None
        self._daily_start_time: float = time.time()
        self._daily_realized_pnl: float = 0.0
        self._daily_funding_paid: float = 0.0

        self._circuit_level: int = 0
        self._last_trending_log_ts: float = 0.0

    # === Logger de eventos estructurados (JSONL) ===

    def _event(self, name: str, **data):
        # Obtener mark price con tolerancia
        try:
            mp = self.c.mark_price(self.cfg.symbol)
        except Exception:
            mp = None

        # No mutar 'data' original
        ev = dict(data or {})

        # Filtros por flags
        if not self.cfg.event_include_oid:
            ev.pop("oid", None)
        if not self.cfg.event_include_pnl:
            ev.pop("pnl", None)

        payload = {
            "ts": utcnow(),
            "run_id": RUN_ID,
            "symbol": getattr(self.cfg, "symbol", None),
            "event": name,
            "mp": round(mp, 2) if isinstance(mp, (int, float)) else mp,
            "center": (
                round(getattr(getattr(self, "state", None), "center", None) or 0, 2)
                if hasattr(self, "state") else None
            ),
            "circuit": getattr(self, "_circuit_level", None),
            "paused": getattr(self, "_paused", None),
        }
        payload.update(ev)
        logging.getLogger("events").info(json.dumps(payload, separators=(",", ":")))


    # === Filtros del exchange ===
    def _load_filters(self):
        info = self.c.exchange_info()
        syms = {s["symbol"]: s for s in info.get("symbols", [])}
        sym = syms.get(self.cfg.symbol, {})
        fs = {f["filterType"]: f for f in sym.get("filters", [])}

        self.price_precision = sym.get("pricePrecision", 2)
        self.qty_precision = sym.get("quantityPrecision", 6)
        self.tick_size = float(fs["PRICE_FILTER"]["tickSize"]) if "PRICE_FILTER" in fs else 0.1
        self.step_size = float(fs["LOT_SIZE"]["stepSize"]) if "LOT_SIZE" in fs else 0.001
        self.min_notional = float(fs.get("MIN_NOTIONAL", {}).get("notional", 5.0))

        if "LOT_SIZE" in fs:
            self.min_qty = float(fs["LOT_SIZE"].get("minQty", 0.0))

        logging.info("Filtros: tick=%.4f step=%.6f minQty=%.6f min_not=%.2f",
                     self.tick_size, self.step_size, self.min_qty, self.min_notional)

    # === Utilidades ===
    def _qty_from_notional(self, notional_usd: float, price: float) -> float:
        if price <= 0:
            return 0.0
        if notional_usd < self.min_notional:
            raise ValueError(f"ORDER_NOTIONAL_USD menor a minNotional ({self.min_notional}).")
        qty = notional_usd / price
        qty = max(self.step_size, round_step(qty, self.step_size))

        if self.min_qty > 0 and qty < self.min_qty:
            qty = self.min_qty

        return round_to_precision(qty, self.qty_precision)

    def _snap_to_tick(self, price: float) -> float:
        ts = self.tick_size
        ticks = math.floor(price / ts + 1e-12)
        p = ticks * ts
        decs = decimals_from_step(ts)
        return float(Decimal(str(p)).quantize(Decimal('1.' + '0'*decs)) if decs > 0 else Decimal(int(p)))

    def _maker_price(self, is_buy: bool, raw_price: float) -> float:
        ts = self.tick_size
        n = max(1, int(self.cfg.maker_ticks_away))
        if is_buy:
            p = math.floor(raw_price / ts) * ts - n * ts
        else:
            p = math.ceil(raw_price / ts) * ts + n * ts
        p = max(ts, p)
        return self._snap_to_tick(p)

    def _cid(self, tag: str, side: str, pos_side: str, price: float, qty: float) -> str:
        raw = f"{tag}|{self.cfg.symbol}|{side}|{pos_side}|{price:.2f}|{qty:.6f}|{time.time()}"
        return "CID_" + hashlib.sha1(raw.encode()).hexdigest()[:20]

    # === Órdenes ===
    def _place_limit(self, side: str, pos_side: str, price: float, qty: float, tag: str = "") -> Optional[Dict]:
        # Para TP usamos el precio tal cual (solo ajuste al tick),
        # para el resto seguimos usando _maker_price (post-only n ticks).
        if tag.upper() == "TP":
            price_used = self._snap_to_tick(price)
        else:
            price_used = self._maker_price(side.upper() == "BUY", price)

        cid = self._cid(tag or "GRID", side, pos_side, price_used, qty)

        logging.info("LIMIT %s %s qty=%.6f @ %.2f %s", side, pos_side, qty, price_used, tag or "")

        if self.cfg.dry_run:
            return None

        params = {
            "symbol": self.cfg.symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTX",  # seguimos siendo post-only
            "quantity": str(qty),
            "price": str(price_used),
            "positionSide": pos_side.upper(),
            "newClientOrderId": cid,
        }

        try:
            return self.c.new_order(**params)
        except (requests.ReadTimeout, requests.ConnectionError):
            try:
                ex = self.c.get_order(self.cfg.symbol, origClientOrderId=cid)
                st = (ex or {}).get("status", "")
                if st in ("NEW", "PARTIALLY_FILLED", "FILLED"):
                    logging.warning("Orden existe: %s", cid)
                    return ex
            except Exception:
                pass
            raise
        except Exception as e:
            logging.error("place_limit: %s", e)
            return None
            

    def _place_market_hedge(self, side: str, pos_side: str, qty: float, tag: str = "HEDGE") -> Optional[Dict]:
        cid = self._cid(tag, side, pos_side, 0, qty)

        logging.info("🛡️ MARKET %s %s qty=%.6f %s", side, pos_side, qty, tag)

        if self.cfg.dry_run:
            # Simular hedges en DRY_RUN
            if tag.upper() in ("HEDGE-OPEN", "PANIC"):
                self._hedge_history.append((time.time(), "open", pos_side.upper()))
                self._active_hedges.append(HedgePosition(
                    side=pos_side.upper(),
                    qty=qty,
                    entry_price=self.c.mark_price(self.cfg.symbol),
                    entry_time=time.time(),
                    notional=qty * self.c.mark_price(self.cfg.symbol)
                ))
                logging.info("📝 DRY: Hedge simulado")
            elif tag.upper() == "CLOSE-HEDGE":
                self._hedge_history.append((time.time(), "close", pos_side.upper()))
                if self._active_hedges:
                    self._active_hedges.pop(0)
                logging.info("📝 DRY: Cierre simulado")
            return None

        params = {
            "symbol": self.cfg.symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(qty),
            "positionSide": pos_side.upper(),
            "newClientOrderId": cid,
        }

        try:
            res = self.c.new_order(**params)
            # En live, registra también en historial con lado de la posición hedge
            if tag.upper() in ("HEDGE-OPEN", "PANIC"):
                self._hedge_history.append((time.time(), "open", pos_side.upper()))
            elif tag.upper() == "CLOSE-HEDGE":
                self._hedge_history.append((time.time(), "close", pos_side.upper()))
            return res
        except Exception as e:
            logging.error("market hedge: %s", e)
            return None

    def place_take_profit_for_fill(self, entry_side: str, pos_side: str, fill_price: float, qty: float):
        step = self.cfg.step_pct
        mp = self.c.mark_price(self.cfg.symbol)
        ticks = max(1, int(self.cfg.maker_ticks_away))
        offset = ticks * self.tick_size

        if entry_side.upper() == "BUY":
            close_side = "SELL"
            raw_tp = fill_price * (1 + step)
            tp_px = max(raw_tp, mp + offset)
        else:
            close_side = "BUY"
            raw_tp = fill_price * (1 - step)
            tp_px = min(raw_tp, mp - offset)

        # --- LOG PREVIO ---
        notional = fill_price * qty
        pnl_gross = notional * step
        fees_est = notional * 0.0004
        pnl_net = pnl_gross - fees_est
        logging.info("PnL esp: gross=$%.2f fees=$%.2f net=$%.2f", pnl_gross, fees_est, pnl_net)

        # === PRIMER INTENTO ===
        res = self._place_limit(close_side, pos_side, tp_px, qty, tag="TP")
        used = self._snap_to_tick(tp_px)

        if res or self.cfg.dry_run:
            oid = int(res["orderId"]) if isinstance(res, dict) and res.get("orderId") else None
            self._event("TP_PLACED", side=close_side, posSide=pos_side, price=round(used, 2), qty=qty, oid=oid)
            return

        # === FALLÓ → RETRY ===
        logging.warning("TP rechazado (GTX). Reintentando con pequeño delay...")
        time.sleep(0.10)

        # recalcular con nuevo mark
        mp2 = self.c.mark_price(self.cfg.symbol)
        if entry_side.upper() == "BUY":
            raw_tp2 = fill_price * (1 + step)
            tp_px2 = max(raw_tp2, mp2 + offset)
        else:
            raw_tp2 = fill_price * (1 - step)
            tp_px2 = min(raw_tp2, mp2 - offset)

        res2 = self._place_limit(close_side, pos_side, tp_px2, qty, tag="TP")
        used2 = self._snap_to_tick(tp_px2)

        if res2 or self.cfg.dry_run:
            oid2 = int(res2["orderId"]) if isinstance(res2, dict) and res2.get("orderId") else None
            self._event("TP_PLACED_RETRY", side=close_side, posSide=pos_side, price=round(used2, 2), qty=qty, oid=oid2)
            return

        # === TOTAL FAIL ===
        logging.error("TP totalmente rechazado luego del retry")


    def build_grid_orders(self) -> List[Tuple[str, str, float, float]]:
        mp = self.c.mark_price(self.cfg.symbol)
        self.state.center = mp
        out = []

        levels = self.cfg.levels_per_side
        if self._circuit_level == 1:
            levels = max(8, levels // 2)
            logging.info("Circuit L1: %d niveles", levels)

        for i in range(1, levels + 1):
            off = mp * self.cfg.step_pct * i
            p_sell = mp + off
            q_sell = self._qty_from_notional(self.cfg.order_notional, p_sell)
            out.append(("SELL", "SHORT", p_sell, q_sell))

            p_buy = mp - off
            q_buy = self._qty_from_notional(self.cfg.order_notional, p_buy)
            out.append(("BUY", "LONG", p_buy, q_buy))
        return out

    def _count_same_side_orders_optimized(self, open_orders: List[Dict], side: str, pos_side: str) -> int:
        return sum(1 for o in open_orders
                  if o.get("side") == side.upper()
                  and o.get("positionSide") == pos_side.upper()
                  and o.get("status") == "NEW")

    def place_grid(self):
        if self._paused or self._circuit_level >= 2:
            logging.warning("Grid pausado (panic=%s circuit=%d)", self._paused, self._circuit_level)
            return

        try:
            open_orders = self.c.get_open_orders(self.cfg.symbol)
        except Exception as e:
            logging.error("Error get_open_orders: %s", e)
            open_orders = []

        orders = self.build_grid_orders()
        for side, ps, price, qty in orders:
            current = self._count_same_side_orders_optimized(open_orders, side, ps)
            if current >= self.cfg.max_same_side_levels:
                logging.warning("⚠️ MAX_SAME_SIDE_LEVELS: %s %s (%d/%d)",
                               side, ps, current, self.cfg.max_same_side_levels)
                continue

            res = self._place_limit(side, ps, price, qty, tag="GRID")
            used = self._maker_price(side.upper()=="BUY", price)

            oid = None
            if res and isinstance(res, dict) and res.get("orderId"):
                try:
                    oid = int(res["orderId"])
                    self._grid_order_ids.add(oid)
                    self._grid_order_meta[oid] = (
                        side,
                        ps,
                        float(res.get("price", used)),
                        float(res.get("origQty", qty)),
                    )
                except Exception:
                    pass

            self._event(
                "GRID_ORDER",
                side=side,
                posSide=ps,
                price=round(used, 2),
                qty=qty,
                created=bool(res or self.cfg.dry_run),
                oid=oid,                     # ← agregado
            )


    # === Métricas diarias ===
    def _calculate_realized_pnl_today(self, force_refresh: bool = False) -> float:
        now = time.time()

        if not force_refresh and self._pnl_cache is not None:
            age = now - self._pnl_cache_time
            if age < self._pnl_cache_ttl:
                return self._pnl_cache

        try:
            now_ms = int(now * 1000)
            start_ms = int(self._daily_start_time * 1000)

            trades = self.c.user_trades(self.cfg.symbol, start_time=start_ms, end_time=now_ms, limit=1000)
            total_realized = sum(float(t.get("realizedPnl", 0)) for t in trades)

            income = self.c.income_history(self.cfg.symbol, start_time=start_ms, end_time=now_ms, limit=1000)
            funding = sum(float(i.get("income", 0)) for i in income if i.get("incomeType") == "FUNDING_FEE")

            total = total_realized + funding

            self._daily_realized_pnl = total_realized
            self._daily_funding_paid = funding

            self._pnl_cache = total
            self._pnl_cache_time = now

            return total
        except Exception as e:
            logging.error("Error PnL: %s", e)
            return self._pnl_cache if self._pnl_cache is not None else 0.0


    def _snapshot_balance(self):
        try:
            acc = self.c.account()
            bal = float(acc.get("totalWalletBalance", 0.0))
            avail = float(acc.get("availableBalance", 0.0))
            init = float(acc.get("totalInitialMargin", 0.0))
            maint = float(acc.get("totalMaintMargin", 0.0))
            ratio = (maint / bal) if bal > 0 else 0.0
            self._event(
                "BALANCE_SNAPSHOT",
                balance=round(bal, 2),
                available=round(avail, 2),
                init_margin=round(init, 2),
                maint_margin=round(maint, 2),
                margin_ratio=round(ratio, 4),
            )
        except Exception as e:
            logging.debug("snapshot_balance: %s", e)

    def _snapshot_daily_pnl(self):
        pnl = self._calculate_realized_pnl_today(force_refresh=True)
        self._event(
            "DAILY_PNL_SNAPSHOT",
            realized=round(self._daily_realized_pnl, 2),
            funding=round(self._daily_funding_paid, 2),
            total=round(pnl, 2),
        )

    def _init_daily_tracking(self):
        if self._daily_start_balance is None:
            try:
                acc = self.c.account()
                self._daily_start_balance = float(acc.get("totalWalletBalance", 0))
                self._daily_start_time = time.time()
                logging.info("💰 Balance inicial: $%.2f", self._daily_start_balance)
            except Exception as e:
                logging.error("Error init: %s", e)

    def _check_daily_reset(self):
        if time.time() - self._daily_start_time > 86400:

            # 👉 foto final del día antes de resetear contadores
            try:
                self._snapshot_daily_pnl()
                self._snapshot_balance()
            except Exception:
                pass
            logging.info("🔄 Reset diario")
            try:
                acc = self.c.account()
                self._daily_start_balance = float(acc.get("totalWalletBalance", 0))
                self._daily_start_time = time.time()
                self._daily_realized_pnl = 0.0
                self._daily_funding_paid = 0.0
                self._circuit_level = 0
                self._pnl_cache = None
            except Exception as e:
                logging.error("Error reset: %s", e)

    def _detect_strong_trend(self) -> bool:
        """Detector SMART-LITE: combina varias señales pero solo con datos locales."""
        cutoff = time.time() - 4*3600
        # Señal 1: muchos hedges recientes (proxy de estrés)
        hedges_recent = sum(1 for h in self._hedge_history if h and h[0] >= cutoff and (len(h) >= 2 and h[1] == "open"))
        many_hedges = hedges_recent >= self.cfg.max_hedges_per_4h

        # Señal 2: movimiento direccional 4h
        move_4h = abs(self._price_movement_pct(4.0))
        strong_move = move_4h >= float(os.getenv("TREND_DRIFT_4H_PCT", "0.025"))

        # Señal 3: consistencia de lado en hedges
        consist = self._hedge_direction_consistency()
        same_side = consist >= float(os.getenv("TREND_HEDGE_CONSIST_PCT", "0.70"))

        # Señal 4: volatilidad 5m
        vol_5m = self._volatility_range_pct(5)
        high_vol = vol_5m >= float(os.getenv("TREND_VOL_5M_PCT", "0.02"))

        # Trending: requiere many_hedges + al menos N señales adicionales (default 1)
        min_extra = int(os.getenv("TREND_MIN_EXTRA_CONDS", "1"))
        met = sum([strong_move, same_side, high_vol])
        trending = many_hedges and (met >= min_extra)

        if trending:
            now = time.time()
            if now - getattr(self, "_last_trending_log_ts", 0) >= 60:
                logging.warning("⚠️ Trending SMART: hedges=%d move4h=%.2f%% consist=%.0f%% vol5m=%.2f%%",
                                hedges_recent, move_4h*100, consist*100, vol_5m*100)
                self._last_trending_log_ts = now
        return trending

    def _price_movement_pct(self, hours: float = 4.0) -> float:
        if not self._price_hist_long:
            return 0.0
        now = time.time()
        cutoff = now - hours*3600
        # tomar el precio más cercano previo al cutoff, o el primero disponible
        candidates = [(t,p) for t,p in self._price_hist_long if t <= cutoff]
        if candidates:
            t0, p0 = candidates[0]
        else:
            t0, p0 = self._price_hist_long[0]
        p1 = self._price_hist_long[-1][1]
        if p0 <= 0:
            return 0.0
        return (p1 - p0) / p0

    def _volatility_range_pct(self, minutes: int = 5) -> float:
        if not self._price_hist_long:
            return 0.0
        now = time.time()
        cutoff = now - minutes*60
        window = [p for (t,p) in self._price_hist_long if t >= cutoff]
        if len(window) < 2:
            return 0.0
        hi, lo = max(window), min(window)
        return (hi - lo) / max(lo, 1e-9)

    def _hedge_direction_consistency(self, window_sec: int = 4*3600) -> float:
        cutoff = time.time() - window_sec
        recent = [h for h in self._hedge_history if h and len(h) >= 3 and h[0] >= cutoff and h[1] == "open"]
        if len(recent) < 2:
            return 0.0
        longs = sum(1 for *_, side in recent if str(side).upper() == "LONG")
        shorts = len(recent) - longs
        return max(longs, shorts) / len(recent)

    def _check_daily_stop_loss(self):
        pnl = self._calculate_realized_pnl_today()
        if pnl <= -self.cfg.daily_stop_loss_usd:
            logging.error("🚨 STOP LOSS: $%.2f", pnl)
            self._event("DAILY_STOP", realized_pnl=round(pnl,2))
            try:
                self.c.cancel_all_open_orders(self.cfg.symbol)
            except Exception as e:
                logging.error("Error: %s", e)
            self._paused = True
            self._circuit_level = 2
            logging.error("Bot pausado hasta reset diario")

    def _check_circuit_breaker(self):
        pnl = self._calculate_realized_pnl_today()
        if pnl <= -self.cfg.circuit_breaker_loss_l2:
            if self._circuit_level < 2:
                logging.error("🔴 CIRCUIT L2: $%.2f", pnl)
                self._event("CIRCUIT_L2", pnl=round(pnl,2))
                try:
                    self.c.cancel_all_open_orders(self.cfg.symbol)
                    self._grid_order_ids.clear()
                    self._grid_order_meta.clear()
                except Exception as e:
                    logging.error("Error: %s", e)
                self._circuit_level = 2
        elif pnl <= -self.cfg.circuit_breaker_loss_l1:
            if self._circuit_level < 1:
                logging.warning("⚠️ CIRCUIT L1: $%.2f", pnl)
                self._event("CIRCUIT_L1", pnl=round(pnl,2))
                self._circuit_level = 1
                try:
                    self.c.cancel_all_open_orders(self.cfg.symbol)
                    self._grid_order_ids.clear()
                    self._grid_order_meta.clear()
                    self.place_grid()
                except Exception as e:
                    logging.error("Error: %s", e)
        elif pnl > -100 and self._circuit_level > 0:
            logging.info("✅ Circuit reset")
            self._event("CIRCUIT_RESET")
            self._circuit_level = 0
            try:
                self.c.cancel_all_open_orders(self.cfg.symbol)
                self._grid_order_ids.clear()
                self._grid_order_meta.clear()
                self.place_grid()
            except Exception as e:
                logging.error("Error: %s", e)

    def net_exposure_notional(self) -> float:
        pr = self.c.get_position_risk(self.cfg.symbol)
        mp = self.c.mark_price(self.cfg.symbol)
        long_n = sum(abs(float(p.get("positionAmt", 0))) * mp for p in pr if p.get("positionSide") == "LONG")
        short_n = sum(abs(float(p.get("positionAmt", 0))) * mp for p in pr if p.get("positionSide") == "SHORT")
        return long_n - short_n

    def _calculate_hedge_fraction(self, drift: float) -> float:
        h = self.cfg.hedge_on_drift_pct
        h_ratio = drift / h if h > 0 else 0
        frac = self.cfg.hedge_fraction_min + 0.4 * h_ratio
        return min(frac, self.cfg.hedge_fraction_max)

    def _open_hedge_position(self, net_n: float, drift: float):
        mp = self.c.mark_price(self.cfg.symbol)
        hedge_frac = self._calculate_hedge_fraction(drift)
        hedge_notional = abs(net_n) * hedge_frac

        # 🔹 Evitar el ValueError por debajo del mínimo de Binance
        if hedge_notional < self.min_notional:
            logging.info(
                "Hedge demasiado pequeño, lo salto: notional=%.2f < minNotional=%.2f",
                hedge_notional,
                self.min_notional,
            )
            return

        qty = self._qty_from_notional(hedge_notional, mp)

        if net_n > 0:
            logging.info(
                "🛡️ HEDGE: net LONG → abre SHORT qty=%.6f (%.1f%%)",
                qty,
                hedge_frac * 100,
            )
            res = self._place_market_hedge("SELL", "SHORT", qty, tag="HEDGE-OPEN")
            self._event(
                "HEDGE_OPEN",
                net_n=round(net_n, 2),
                drift=round(drift, 4),
                hedge_frac=round(hedge_frac, 3),
                qty=qty,
                side="SELL",
                posSide="SHORT",
            )
            if res or self.cfg.dry_run:
                if not self.cfg.dry_run:
                    self._active_hedges.append(
                        HedgePosition(
                            side="SHORT",
                            qty=qty,
                            entry_price=mp,
                            entry_time=time.time(),
                            notional=hedge_notional,
                        )
                    )
        else:
            logging.info(
                "🛡️ HEDGE: net SHORT → abre LONG qty=%.6f (%.1f%%)",
                qty,
                hedge_frac * 100,
            )
            res = self._place_market_hedge("BUY", "LONG", qty, tag="HEDGE-OPEN")
            self._event(
                "HEDGE_OPEN",
                net_n=round(net_n, 2),
                drift=round(drift, 4),
                hedge_frac=round(hedge_frac, 3),
                qty=qty,
                side="BUY",
                posSide="LONG",
            )
            if res or self.cfg.dry_run:
                if not self.cfg.dry_run:
                    self._active_hedges.append(
                        HedgePosition(
                            side="LONG",
                            qty=qty,
                            entry_price=mp,
                            entry_time=time.time(),
                            notional=hedge_notional,
                        )
                    )

            

    def _close_hedge_positions(self):
        if not self._active_hedges:
            return
        mp = self.c.mark_price(self.cfg.symbol)
        for hedge in self._active_hedges[:]:
            if hedge.side == "SHORT":
                pnl = hedge.notional * (hedge.entry_price - mp) / hedge.entry_price
                close_side = "BUY"
            else:
                pnl = hedge.notional * (mp - hedge.entry_price) / hedge.entry_price
                close_side = "SELL"
            logging.info("✅ Cerrando %s: PnL=$%.2f", hedge.side, pnl)
            self._event("HEDGE_CLOSE", side=hedge.side, qty=hedge.qty, entry=round(hedge.entry_price,2), exit=round(mp,2), pnl=round(pnl,2))
            res = self._place_market_hedge(close_side, hedge.side, hedge.qty, tag="CLOSE-HEDGE")
            if res or self.cfg.dry_run:
                self._active_hedges.remove(hedge)

    def maybe_hedge_and_recenter(self):
        mp = self.c.mark_price(self.cfg.symbol)
        drift = abs(mp - self.state.center) / self.state.center
        net_n = self.net_exposure_notional()
        gross_grid_n = self.cfg.order_notional * self.cfg.levels_per_side * 2

        if self._active_hedges and drift < 0.005:
            logging.info("📍 Precio regresó, cerrando hedges")
            self._close_hedge_positions()

        need_hedge = (drift >= self.cfg.hedge_on_drift_pct) or (abs(net_n) >= self.cfg.hedge_on_net_fraction * gross_grid_n)
        need_recenter_full = drift >= self.cfg.recenter_drift_full
        need_recenter_partial = drift >= self.cfg.recenter_drift_partial

        if not (need_hedge or need_recenter_full or need_recenter_partial):
            return
        
        now = time.time()

        # Calculamos el drift en %
        drift_pct = drift * 100  # drift llega en decimal

        # Inicializar variable del último valor impreso
        last_drift_val = getattr(self, "_last_drift_pct", None)

        # Condición de impresión:
        # 1) han pasado 30s
        # 2) o el drift cambió más de 0.05%
        should_print = False

        if now - getattr(self, "_last_drift_log_ts", 0) >= 30:
            should_print = True
        elif last_drift_val is None or abs(drift_pct - last_drift_val) >= 0.05:
            should_print = True

        if should_print:
            logging.info("⚡ drift=%.3f%% net=$%.2f", drift_pct, net_n)
            self._last_drift_log_ts = now
            self._last_drift_pct = drift_pct



        if need_recenter_full:
            hedges_before = len(self._active_hedges)
            logging.warning("🔄 RECENTRADO COMPLETO (hedges: %d)", hedges_before)
            self._event("RECENTER_FULL", old_center=round(self.state.center,2), new_center=round(mp,2), drift=round(drift,4))
            try:
                self.c.cancel_all_open_orders(self.cfg.symbol)
            except Exception as e:
                logging.warning("cancel: %s", e)
            self._grid_order_ids.clear()
            self._grid_order_meta.clear()
            if abs(net_n) > 0 and not self._detect_strong_trend():
                self._open_hedge_position(net_n, drift)
            self.state.center = mp
            self.state.last_recenter_ts = time.time()
            self.place_grid()
            hedges_after = len(self._active_hedges)
            logging.info("Hedges post-recenter: %d (antes: %d)", hedges_after, hedges_before)
            return

        if need_recenter_partial:
            hedges_before = len(self._active_hedges)
            logging.warning("🔄 RECENTRADO PARCIAL (hedges: %d)", hedges_before)
            old_center = self.state.center
            self.state.center = (old_center + mp) / 2
            self._event("RECENTER_PARTIAL", old_center=round(old_center,2), new_center=round(self.state.center,2), drift=round(drift,4))
            logging.info("Centro: %.2f → %.2f", old_center, self.state.center)
            try:
                self.c.cancel_all_open_orders(self.cfg.symbol)
            except Exception as e:
                logging.warning("cancel: %s", e)
            self._grid_order_ids.clear()
            self._grid_order_meta.clear()
            if abs(net_n) > 0 and not self._detect_strong_trend():
                self._open_hedge_position(net_n, drift)
            self.state.last_recenter_ts = time.time()
            self.place_grid()
            hedges_after = len(self._active_hedges)
            logging.info("Hedges post-recenter: %d (antes: %d)", hedges_after, hedges_before)
            return

        if need_hedge and abs(net_n) > 0:
            if self._detect_strong_trend():
                # Anti-spam: el warning se emite dentro de _detect_strong_trend con rate-limit
                return
            logging.info("🛡️ Hedge")
            self._open_hedge_position(net_n, drift)

    def _check_funding_rate(self):
        try:
            info = self.c.premium_index(self.cfg.symbol)
            rate = abs(float(info.get("lastFundingRate", 0)))
            if rate >= self.cfg.funding_rate_warn_8h:
                pr = self.c.get_position_risk(self.cfg.symbol)
                mp = self.c.mark_price(self.cfg.symbol)
                total = sum(abs(float(p.get("positionAmt", 0))) * mp for p in pr)
                cost = total * rate
                logging.warning("⚠️ Funding: %.4f%%", rate*100)
                logging.warning("Costo: $%.2f/periodo", cost)
                self._event("FUNDING_WARN", rate=rate, est_cost=round(cost,2))
                if abs(self._daily_funding_paid) >= self.cfg.funding_fee_warn_usd:
                    logging.error("🚨 Funding hoy: $%.2f", abs(self._daily_funding_paid))
        except Exception as e:
            logging.debug("Error funding: %s", e)

    def _check_panic_mode(self):
        mp = self.c.mark_price(self.cfg.symbol)
        now = time.time()
        # Ventana corta (5m) para pánico
        self._price_history_5m.append((now, mp))
        self._price_history_5m = [(t, p) for t, p in self._price_history_5m if t >= now - 300]
        # Ventana larga (4h) para trending
        self._price_hist_long.append((now, mp))
        cut_long = now - 4*3600
        self._price_hist_long = [(t, p) for t, p in self._price_hist_long if t >= cut_long]

        if len(self._price_history_5m) < 2:
            return

        oldest = self._price_history_5m[0][1]
        change = abs(mp - oldest) / oldest

        if change >= self.cfg.panic_range_5m_pct and not self._paused:
            logging.error("🚨 PÁNICO: %.2f%%/5min", change*100)
            self._paused = True
            self._stable_blocks = 0
            self._event("PANIC_ON", change_5m=round(change,4))
            try:
                self.c.cancel_all_open_orders(self.cfg.symbol)
                self._grid_order_ids.clear()
                self._grid_order_meta.clear()
            except Exception as e:
                logging.error("Error: %s", e)
            net_n = self.net_exposure_notional()
            if abs(net_n) > 0 and not self._detect_strong_trend():
                hedge_n = abs(net_n) * self.cfg.hedge_fraction_max
                qty = self._qty_from_notional(hedge_n, mp)
                if net_n > 0:
                    logging.warning("PÁNICO: net LONG → hedge SHORT")
                    self._event("PANIC_HEDGE", side="SELL", posSide="SHORT", qty=qty)
                    self._place_market_hedge("SELL", "SHORT", qty, tag="PANIC")
                else:
                    logging.warning("PÁNICO: net SHORT → hedge LONG")
                    self._event("PANIC_HEDGE", side="BUY", posSide="LONG", qty=qty)
                    self._place_market_hedge("BUY", "LONG", qty, tag="PANIC")
        elif self._paused and change < self.cfg.unpause_range_5m_pct:
            self._stable_blocks += 1
            if self._stable_blocks >= self.cfg.unpause_stable_blocks:
                logging.info("✅ Reanudando")
                self._event("PANIC_OFF")
                self._paused = False
                self._stable_blocks = 0
                self.place_grid()
        elif self._paused and change >= self.cfg.unpause_range_5m_pct:
            self._stable_blocks = 0

    def _bootstrap_last_trade_id(self):
        try:
            trades = self.c.user_trades(self.cfg.symbol, limit=1)
            if trades:
                self._last_trade_id = int(trades[-1]["id"])
                logging.info("Bootstrap: %s", self._last_trade_id)
        except Exception as e:
            logging.warning("bootstrap: %s", e)

    def fetch_new_fills(self) -> List[Dict]:
        from_id = self._last_trade_id + 1 if self._last_trade_id is not None else None
        try:
            fills = self.c.user_trades(self.cfg.symbol, from_id=from_id, limit=1000)
        except Exception as e:
            logging.warning("user_trades: %s", e)
            return []

        out = []
        for tr in fills:
            try:
                oid = int(tr.get("orderId", -1))
                if oid in self._grid_order_ids:
                    out.append(tr)
                    self._grid_order_ids.discard(oid)
            except Exception:
                pass
        return out

    def _get_timestop_hours(self) -> float:
        mp = self.c.mark_price(self.cfg.symbol)
        drift = abs(mp - self.state.center) / self.state.center
        return self.cfg.timestop_hours_fast if drift >= self.cfg.drift_fast_pct else self.cfg.timestop_hours_base

    def _stale_orders(self, hours: float) -> List[int]:
        if hours <= 0:
            return []
        try:
            oo = self.c.get_open_orders(self.cfg.symbol)
        except Exception as e:
            logging.warning("Error al obtener órdenes abiertas: %s", e)
            return []
        now = time.time()
        stale = []
        for o in oo:
            try:
                t_raw = o.get("time") or o.get("updateTime")
                if not t_raw:
                    continue
                t = float(t_raw) / 1000.0
                if t <= 0 or t > now:
                    continue
                age_hours = (now - t) / 3600.0
                if age_hours >= hours:
                    order_id = int(o["orderId"])
                    stale.append(order_id)
            except Exception:
                continue
        return stale

    def _rearm_order_from_meta(self, oid: int):
        meta = self._grid_order_meta.get(oid)
        if not meta:
            return
        side, ps, price, qty = meta
        used = self._maker_price(side.upper()=="BUY", price)
        res = self._place_limit(side, ps, price, qty, tag="REGEN")

        new_id = None
        if res and isinstance(res, dict) and res.get("orderId"):
            try:
                new_id = int(res["orderId"])
                self._grid_order_ids.add(new_id)
                self._grid_order_meta[new_id] = (
                    side, ps,
                    float(res.get("price", used)),
                    float(res.get("origQty", qty))
                )
                self._grid_order_meta.pop(oid, None)
            except Exception:
                pass

        self._event(
            "GRID_ORDER",
            side=side,
            posSide=ps,
            price=round(used, 2),
            qty=qty,
            tag="REGEN",
            rearmOf=oid,
            created=bool(res or self.cfg.dry_run),
            oid=new_id,                 # ← agregado (id de la nueva orden)
        )


    def print_status(self):
        try:
            acc = self.c.account()
            bal = float(acc.get("totalWalletBalance", 0))
            margin = float(acc.get("totalInitialMargin", 0))
            upnl = float(acc.get("totalUnrealizedProfit", 0))
            print(f"""
╔══════════════════════════════════════════╗
║   Grid Bot SMART - {self.cfg.symbol}           ║
╠══════════════════════════════════════════╣
║ Balance:      ${bal:>12,.2f}          ║
║ Margin:       ${margin:>12,.2f} ({(margin/bal*100 if bal else 0):>5.1f}%) ║
║ UPnL:         ${upnl:>12,.2f}          ║
║ PnL hoy:      ${self._daily_realized_pnl:>12,.2f}          ║
║ Funding:      ${self._daily_funding_paid:>12,.2f}          ║
╠══════════════════════════════════════════╣
║ Grid:         {len(self._grid_order_ids):>4}                     ║
║ Hedges:       {len(self._active_hedges):>4}                     ║
║ Circuit:      L{self._circuit_level}                       ║
║ Pánico:       {'SÍ' if self._paused else 'NO':>4}                     ║
╚══════════════════════════════════════════╝
            """)
        except Exception:
            pass

    def preflight(self):
        if not self.cfg.api_key or not self.cfg.api_secret:
            raise RuntimeError("Faltan credenciales")
        try:
            self.c.change_position_mode(True)
            logging.info("✓ Hedge mode")
        except Exception as e:
            logging.info("position mode: %s", e)
        try:
            self.c.change_leverage(self.cfg.symbol, self.cfg.leverage)
            logging.info("✓ Leverage: %dx", self.cfg.leverage)
        except Exception as e:
            logging.info("leverage: %s", e)
        try:
            self.c.change_margin_type(self.cfg.symbol, self.cfg.margin_mode)
            logging.info("✓ Margin: %s", self.cfg.margin_mode)
        except Exception as e:
            logging.info("margin: %s", e)
        logging.info("Centro: %.2f", self.state.center)
        self._init_daily_tracking()

    def run(self):
        self.preflight()
        self.place_grid()
        logging.info("✓ Grid inicial")
        self._bootstrap_last_trade_id()

        backoff = 2.0
        err_count = 0
        loop_count = 0

        while True:
            try:
                loop_count += 1

                self._check_daily_reset()
                self.kill_switch_if_needed()
                self._check_daily_stop_loss()
                self._check_circuit_breaker()
                self._check_panic_mode()

                if loop_count % 20 == 0:
                    self._check_funding_rate()

                if not self._paused and self._circuit_level < 2:
                    self.maybe_hedge_and_recenter()

                fills = self.fetch_new_fills()
                if fills:
                    for tr in fills:
                        side = tr.get("side", "").upper()
                        ps = tr.get("positionSide", "").upper() or ("LONG" if side == "BUY" else "SHORT")
                        px = float(tr.get("price", 0.0))
                        qty = float(tr.get("qty", 0.0))

                        # ← EXTRAER orderId del trade para trazabilidad
                        oid = None
                        try:
                            if "orderId" in tr and tr["orderId"] is not None:
                                oid = int(tr["orderId"])
                        except Exception:
                            pass

                        logging.info("✓ FILL: %s %s %.6f@%.2f (oid=%s)", side, ps, qty, px, oid)

                        # ← Incluir oid en el evento
                        self._event("FILL", side=side, posSide=ps, qty=qty, fill_price=round(px, 2), oid=oid)

                        # TP igual que antes
                        self.place_take_profit_for_fill(side, ps, px, qty)

                    # avanzar _last_trade_id como ya haces
                    try:
                        self._last_trade_id = int(fills[-1]["id"])
                    except Exception as e:
                        logging.error("Error last_trade_id: %s", e)


                if not self._paused and self._circuit_level < 2:
                    ts_h = self._get_timestop_hours()
                    stale = self._stale_orders(ts_h)
                    if stale:
                        logging.info("Time-stop: %d (>%.1fh)", len(stale), ts_h)
                        for oid in stale:
                            try:
                                self.c.cancel_order(self.cfg.symbol, oid)
                                self._grid_order_ids.discard(oid)
                                if self.cfg.time_stop_cancel_only:
                                    self._grid_order_meta.pop(oid, None)
                                else:
                                    self._rearm_order_from_meta(oid)
                            except Exception as e:
                                logging.warning("cancel/rearm: %s", e)
                                self._grid_order_meta.pop(oid, None)

                backoff = 2.0
                err_count = 0
                # Snapshots cada hora (sincronizado por tiempo, no por número de loops)
                now = time.time()
                if not hasattr(self, "_last_snapshot_ts"):
                    self._last_snapshot_ts = 0.0
                if now - self._last_snapshot_ts >= self.cfg.snapshot_freq_min * 60:
                    self._snapshot_balance()
                    self._snapshot_daily_pnl()
                    self._last_snapshot_ts = now

                time.sleep(2)

            except KeyboardInterrupt:
                logging.info("Detenido")
                break
            except SystemExit:
                raise
            except Exception as e:
                err_count += 1
                logging.error("Loop error (%s). Backoff %.0fs", e, backoff)
                time.sleep(backoff + random.uniform(0, 0.5*backoff))
                backoff = min(backoff * 2, 60.0)
                if err_count >= 8:
                    logging.warning("Recreando sesión...")
                    try:
                        self.c.recreate_session()
                    except Exception:
                        pass
                    err_count = 0

    def kill_switch_if_needed(self):
        if self.cfg.dry_run:
            return
        try:
            acc = self.c.account()
            bal = float(acc.get("totalWalletBalance", 0.0))
            maint = float(acc.get("totalMaintMargin", 0.0))
            avail = float(acc.get("availableBalance", 0.0))
            ratio = (maint / bal) if bal > 0 else 0.0
            if ratio >= self.cfg.max_margin_ratio or avail < self.cfg.min_available_balance_usd:
                logging.error("🚨 KILL-SWITCH: ratio=%.2f%% avail=$%.2f", ratio*100, avail)
                self._event("KILL_SWITCH", margin_ratio=round(ratio,4), available=round(avail,2))
                try:
                    self.c.cancel_all_open_orders(self.cfg.symbol)
                except Exception as e:
                    logging.error("Error: %s", e)
                try:
                    pr = self.c.get_position_risk(self.cfg.symbol)
                    for p in pr:
                        ps = p.get("positionSide", "")
                        qty = abs(float(p.get("positionAmt", 0)))
                        if qty > 0:
                            side = "SELL" if ps == "LONG" else "BUY"
                            logging.error("Cerrando %s: %.6f", ps, qty)
                            self.c.new_order(symbol=self.cfg.symbol, side=side, type="MARKET", quantity=str(qty), positionSide=ps)
                except Exception as e:
                    logging.error("Error: %s", e)
                raise SystemExit("Kill-switch")
        except SystemExit:
            raise
        except Exception as e:
            logging.error("Error kill_switch: %s", e)

# =========================
#  MAIN + LOGGING A ARCHIVOS
# =========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Logging a archivos (rotación diaria)
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 1) Log humano rotado diario
    human_handler = TimedRotatingFileHandler(
    os.path.join(log_dir, "runtime.log"),
    when="midnight",
    backupCount=14,
    encoding="utf-8",
    utc=False  # rotar a medianoche LOCAL
)

    # Usar hora local (por defecto logging usa time.localtime)
    human_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    human_handler.setFormatter(human_fmt)
    logging.getLogger().addHandler(human_handler)


    # 2) Log estructurado JSONL rotado diario
    events_logger = logging.getLogger("events")
    events_logger.setLevel(logging.INFO)
    events_handler = TimedRotatingFileHandler(os.path.join(log_dir, "events.jsonl"), when="midnight", backupCount=14, encoding="utf-8", utc=False)
    events_handler.setFormatter(logging.Formatter("%(message)s"))
    events_logger.addHandler(events_handler)
    events_logger.propagate = False  # evita duplicar eventos en runtime.log

    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logging.info("Logging iniciado. RUN_ID=%s  log_dir=%s", RUN_ID, log_dir)

    cfg = Config.from_env()

    print("="*70)
    print("Grid Hedge Bot - SMART TRENDING + LOGGING")
    print("="*70)
    print(f"Symbol: {cfg.symbol}")
    print(f"Levels: {cfg.levels_per_side} | Step: {cfg.step_pct*100:.2f}%")
    print(f"Notional: ${cfg.order_notional:,.0f} | Leverage: {cfg.leverage}x")
    print(f"Modo: {'🟢 DRY RUN' if cfg.dry_run else '🔴 LIVE'}")
    print(f"URL: {cfg.base_url}")
    print("="*70)

    if cfg.dry_run:
        print("⚠️  DRY RUN ACTIVADO - Sin órdenes reales")
    else:
        print("🔴 MODO LIVE - ÓRDENES REALES • Iniciando en 5s...")
        time.sleep(5)

    client = RestClient(cfg)
    bot = GridHedgeBot(client, cfg)

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n✓ Bot detenido por usuario")
    except Exception as e:
        logging.exception("Error fatal: %s", e)
        raise
