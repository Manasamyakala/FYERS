# simulator.py
# Implements the market data feed engines.
# Includes the abstract DataFeed base class, SimulatorFeed (mock data), and FyersFeed (live API).

import time
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
import pandas as pd

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

import config

class TickData:
    """Represents a single real-time stock tick, including depth and transaction details."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ltp = 0.0
        self.bid_price = 0.0
        self.bid_qty = 0
        self.ask_price = 0.0
        self.ask_qty = 0
        self.total_bid_qty = 0
        self.total_ask_qty = 0
        self.last_traded_qty = 0
        self.total_volume_today = 0
        self.timestamp = 0.0
        # Detailed 5-level depth
        self.bids: List[Tuple[float, int]] = []
        self.asks: List[Tuple[float, int]] = []

    def update_from_tick(self, ltp: float, last_qty: int, volume: int, timestamp: float):
        self.ltp = ltp
        self.last_traded_qty = last_qty
        self.total_volume_today = volume
        self.timestamp = timestamp

    def update_from_depth(self, bids: List[Tuple[float, int]], asks: List[Tuple[float, int]], tot_bid: int, tot_ask: int):
        self.bids = bids[:5]
        self.asks = asks[:5]
        self.total_bid_qty = tot_bid
        self.total_ask_qty = tot_ask
        if bids:
            self.bid_price = bids[0][0]
            self.bid_qty = bids[0][1]
        if asks:
            self.ask_price = asks[0][0]
            self.ask_qty = asks[0][1]

class Candle:
    """Represents a 1-minute OHLCV bar."""
    def __init__(self, timestamp: float, open_pr: float, high_pr: float, low_pr: float, close_pr: float, volume: float):
        self.timestamp = timestamp  # Epoch seconds
        self.open = open_pr
        self.high = high_pr
        self.low = low_pr
        self.close = close_pr
        self.volume = volume

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }

class DataFeed:
    """Abstract base class for market data feeds (Simulator or Live Broker)."""
    def __init__(self):
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.candle_callbacks: List[Callable[[str, List[Candle]], None]] = []
        
        # State storage
        self.ticks: Dict[str, TickData] = {}
        self.candles: Dict[str, List[Candle]] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Executed volume tracking: symbol -> list of (timestamp, qty)
        self.trade_history: Dict[str, List[Tuple[float, int]]] = {}

    def register_tick_callback(self, cb: Callable[[TickData], None]):
        self.tick_callbacks.append(cb)

    def register_candle_callback(self, cb: Callable[[str, List[Candle]], None]):
        self.candle_callbacks.append(cb)

    def _notify_tick(self, tick: TickData):
        for cb in self.tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

    def _notify_candle_close(self, symbol: str, history: List[Candle]):
        for cb in self.candle_callbacks:
            try:
                cb(symbol, history)
            except Exception as e:
                print(f"Error in candle callback: {e}")

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_historical_candles(self, symbol: str) -> List[Candle]:
        with self.lock:
            return list(self.candles.get(symbol, []))

    def record_trade(self, symbol: str, timestamp: float, qty: int):
        """Records a trade execution to calculate rolling execution volume."""
        with self.lock:
            if symbol not in self.trade_history:
                self.trade_history[symbol] = []
            self.trade_history[symbol].append((timestamp, qty))
            
            # Prune trades older than 1 hour (3600 seconds)
            cutoff = timestamp - 3600
            self.trade_history[symbol] = [t for t in self.trade_history[symbol] if t[0] >= cutoff]

    def get_executed_quantity(self, symbol: str, duration_minutes: int) -> int:
        """Returns total executed quantity in the last N minutes."""
        with self.lock:
            history = self.trade_history.get(symbol, [])
            if not history:
                return 0
            
            now = time.time()
            cutoff = now - (duration_minutes * 60)
            
            total_qty = 0
            for ts, qty in history:
                if ts >= cutoff:
                    total_qty += qty
            return total_qty

    def get_average_ltp(self, symbol: str, duration_minutes: int) -> float:
        """Returns the average Close price (LTP) over the last N 1-minute candles."""
        with self.lock:
            candles = self.candles.get(symbol, [])
            if not candles:
                # Fallback to current tick LTP
                tick = self.ticks.get(symbol)
                return tick.ltp if tick else 0.0
            
            # Get the close prices of the last N candles
            closes = [c.close for c in candles[-duration_minutes:]]
            if not closes:
                return 0.0
            return float(np.mean(closes))


class SimulatorFeed(DataFeed):
    """Generates high-fidelity simulated real-time data and depth for testing."""
    def __init__(self):
        super().__init__()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.symbols = config.DEFAULT_SYMBOLS
        
        # Base reference prices for simulation
        self.ref_prices = config.REF_PRICES

    def start(self):
        self.running = True
        
        # 1. Pre-generate historical candles for all symbols
        self._pregenerate_history()
        
        # 2. Start simulation thread for live ticks
        self.thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _pregenerate_history(self):
        """Generates 240 bars (4 hours) of historical 1-minute OHLCV candles."""
        now = time.time()
        start_time = now - (240 * 60)
        
        for symbol in self.symbols:
            base_price = self.ref_prices.get(symbol, 100.0)
            candles_list = []
            
            # Determine volume level based on symbol
            is_liquid = symbol in ["NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:IDEA-EQ", "NSE:YESBANK-EQ", "NSE:SOUTHBANK-EQ"]
            base_volume = 120000 if is_liquid else 8000
            
            current_price = base_price
            for i in range(240):
                candle_time = start_time + (i * 60)
                
                # Random walk candle generation
                change = current_price * random.normalvariate(0.0001, 0.001)  # small positive drift
                c_open = current_price
                c_close = current_price + change
                c_high = max(c_open, c_close) + abs(random.normalvariate(0, current_price * 0.0005))
                c_low = min(c_open, c_close) - abs(random.normalvariate(0, current_price * 0.0005))
                c_volume = int(max(100, base_volume * random.uniform(0.5, 1.8)))
                
                candle = Candle(candle_time, c_open, c_high, c_low, c_close, c_volume)
                candles_list.append(candle)
                
                # Pre-populate trade history for rolling quantity aggregations
                # Put a historical block trade representing this minute
                self.record_trade(symbol, candle_time, c_volume)
                
                current_price = c_close

            self.candles[symbol] = candles_list
            
            # Setup initial tick data
            tick = TickData(symbol)
            # Generate realistic depth
            tot_bid, tot_ask, bids, asks = self._generate_depth_profile(symbol, current_price, is_liquid)
            tick.update_from_tick(current_price, int(c_volume / 60), c_volume * 10, now)
            tick.update_from_depth(bids, asks, tot_bid, tot_ask)
            self.ticks[symbol] = tick

    def _generate_depth_profile(self, symbol: str, price: float, is_liquid: bool) -> Tuple[int, int, List[Tuple[float, int]], List[Tuple[float, int]]]:
        """Generates realistic bid/ask order book depth."""
        # Liquidity quantities: liquid stocks cross the 10,00,000 threshold
        # We fluctuate it so it goes slightly below and above 10L, showing the filter in action
        if is_liquid:
            # Let YESBANK fluctuate near boundary
            if "YESBANK" in symbol:
                tot_bid = int(random.uniform(900000, 1400000))
                tot_ask = int(random.uniform(920000, 1380000))
            else:
                tot_bid = int(random.uniform(1050000, 2200000))
                tot_ask = int(random.uniform(1020000, 2150000))
        else:
            tot_bid = int(random.uniform(10000, 250000))
            tot_ask = int(random.uniform(12000, 270000))
            
        spread = 0.05 if price < 100 else (0.10 if price < 500 else 0.50)
        
        # 5-level depth
        bids = []
        asks = []
        
        bid_rem = tot_bid
        ask_rem = tot_ask
        
        for k in range(5):
            bp = round(price - spread - (k * spread), 2)
            ap = round(price + spread + (k * spread), 2)
            
            # Distribute volume: 40% at level 1, decreasing outwards
            if k == 4:
                bq = bid_rem
                aq = ask_rem
            else:
                bq = int(tot_bid * (0.35 - (k * 0.06)) * random.uniform(0.8, 1.2))
                aq = int(tot_ask * (0.35 - (k * 0.06)) * random.uniform(0.8, 1.2))
                bid_rem -= bq
                ask_rem -= aq
                
            bids.append((bp, max(1, bq)))
            asks.append((ap, max(1, aq)))
            
        return tot_bid, tot_ask, bids, asks

    def _run_simulation(self):
        """Runs the loop generating live tick and depth data."""
        while self.running:
            time.sleep(random.uniform(0.1, 0.3))  # 100ms to 300ms intervals
            
            # Pick a random symbol to update
            symbol = random.choice(self.symbols)
            tick = self.ticks.get(symbol)
            if not tick:
                continue
                
            is_liquid = symbol in ["NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:IDEA-EQ", "NSE:YESBANK-EQ", "NSE:SOUTHBANK-EQ"]
            
            # Generate new price tick
            prev_price = tick.ltp
            change = prev_price * random.normalvariate(0, 0.0008)
            new_price = round(prev_price + change, 2)
            
            # Prevent prices going negative
            if new_price <= 1.0:
                new_price = 1.0
                
            trade_qty = int(max(1, abs(random.normalvariate(5000 if is_liquid else 100, 2000 if is_liquid else 50))))
            now = time.time()
            
            # Update TickData metrics
            tick.update_from_tick(
                ltp=new_price,
                last_qty=trade_qty,
                volume=tick.total_volume_today + trade_qty,
                timestamp=now
            )
            
            # Generate depth
            tot_bid, tot_ask, bids, asks = self._generate_depth_profile(symbol, new_price, is_liquid)
            tick.update_from_depth(bids, asks, tot_bid, tot_ask)
            
            # Record execution in history
            self.record_trade(symbol, now, trade_qty)
            
            # Update 1-minute candle aggregation
            self._update_candle_aggregation(symbol, new_price, trade_qty, now)
            
            # Notify subscribers
            self._notify_tick(tick)

    def _update_candle_aggregation(self, symbol: str, price: float, qty: int, now: float):
        """Aggregates live ticks into 1-minute candles."""
        with self.lock:
            history = self.candles.get(symbol)
            if not history:
                return
                
            current_minute_start = int(now / 60) * 60
            last_candle = history[-1]
            
            if last_candle.timestamp == current_minute_start:
                # Update current open candle
                last_candle.high = max(last_candle.high, price)
                last_candle.low = min(last_candle.low, price)
                last_candle.close = price
                last_candle.volume += qty
            else:
                # Current candle has closed, start a new one!
                new_candle = Candle(
                    timestamp=current_minute_start,
                    open_pr=last_candle.close,  # Open equals last close
                    high_pr=max(last_candle.close, price),
                    low_pr=min(last_candle.close, price),
                    close_pr=price,
                    volume=qty
                )
                history.append(new_candle)
                
                # Keep history size managed (max 300 candles)
                if len(history) > 300:
                    history.pop(0)
                    
                # Fire candle callback as a new bar just closed
                threading.Thread(target=self._notify_candle_close, args=(symbol, list(history)), daemon=True).start()


class FyersFeed(DataFeed):
    """Connects to the Fyers API v3 for live quotes, historical data, and WebSocket streaming."""
    def __init__(self, client_id: str, access_token: str):
        super().__init__()
        self.client_id = client_id
        self.access_token = access_token
        self.full_token = f"{client_id}:{access_token}"
        
        self.fyers: Optional[fyersModel.FyersModel] = None
        self.websocket: Optional[data_ws.FyersDataSocket] = None
        
        self.all_symbols = config.DEFAULT_SYMBOLS
        self.screened_symbols: List[str] = []      # LTP ₹30 - ₹500
        self.active_filtered_symbols: List[str] = [] # Screened + Liquidity passed (websocket filters)
        
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_running = False
        
        self.history_loaded_symbols = set()
        
        self.is_demo = self.access_token.startswith("demo") or self.client_id.upper().startswith("DEMO")
        self.ref_prices = config.REF_PRICES

    def start(self):
        if self.is_demo:
            self._start_demo_feed()
            return
            
        # 1. Initialize Rest client
        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            is_async=False,
            log_path=""
        )
        
        # 2. Stage 1 Screening: Screen symbols by price (₹30 - ₹500)
        self._run_price_screening()
        
        if not self.screened_symbols:
            print("No symbols passed the price filter (₹30 - ₹500) in initial quotes screening.")
            return

        # 3. Pre-create TickData structures for screened symbols
        for symbol in self.screened_symbols:
            self.ticks[symbol] = TickData(symbol)

        # 4. Connect to WebSockets
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._connect_websocket, daemon=True)
        self.ws_thread.start()

    def _start_demo_feed(self):
        print("Starting Fyers Live API Demo Feed...")
        self.screened_symbols = []
        
        # Screen by price in REF_PRICES
        for sym in self.all_symbols:
            price = self.ref_prices.get(sym, 100.0)
            if config.PRICE_MIN <= price <= config.PRICE_MAX:
                self.screened_symbols.append(sym)
                
        # Pre-create tick data structures
        for sym in self.screened_symbols:
            self.ticks[sym] = TickData(sym)
            
        # Pre-generate history
        now = time.time()
        start_time = now - (240 * 60)
        for symbol in self.screened_symbols:
            base_price = self.ref_prices.get(symbol, 100.0)
            candles_list = []
            
            is_liquid = symbol in ["NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:IDEA-EQ", "NSE:YESBANK-EQ", "NSE:SOUTHBANK-EQ"]
            base_volume = 120000 if is_liquid else 8000
            
            current_price = base_price
            for i in range(240):
                candle_time = start_time + (i * 60)
                change = current_price * random.normalvariate(0.0001, 0.001)
                c_open = current_price
                c_close = current_price + change
                c_high = max(c_open, c_close) + abs(random.normalvariate(0, current_price * 0.0005))
                c_low = min(c_open, c_close) - abs(random.normalvariate(0, current_price * 0.0005))
                c_volume = int(max(100, base_volume * random.uniform(0.5, 1.8)))
                
                candle = Candle(candle_time, c_open, c_high, c_low, c_close, c_volume)
                candles_list.append(candle)
                self.record_trade(symbol, candle_time, c_volume)
                current_price = c_close
                
            self.candles[symbol] = candles_list
            
            # Initialize tick data
            tick = self.ticks[symbol]
            tot_bid, tot_ask, bids, asks = self._generate_depth_profile(symbol, current_price, is_liquid)
            tick.update_from_tick(current_price, int(c_volume / 60), c_volume * 10, now)
            tick.update_from_depth(bids, asks, tot_bid, tot_ask)
            
        # Add liquid ones to active screened symbols automatically for demo
        demo_liquid_symbols = [s for s in self.screened_symbols if s in [
            "NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", 
            "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:YESBANK-EQ"
        ]]
        self.active_filtered_symbols = demo_liquid_symbols
        self.history_loaded_symbols.update(self.screened_symbols)
        
        # Start simulation thread
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._run_demo_simulation, daemon=True)
        self.ws_thread.start()

    def _generate_depth_profile(self, symbol: str, price: float, is_liquid: bool) -> Tuple[int, int, List[Tuple[float, int]], List[Tuple[float, int]]]:
        if is_liquid:
            if "YESBANK" in symbol:
                tot_bid = int(random.uniform(900000, 1400000))
                tot_ask = int(random.uniform(920000, 1380000))
            else:
                tot_bid = int(random.uniform(1050000, 2200000))
                tot_ask = int(random.uniform(1020000, 2150000))
        else:
            tot_bid = int(random.uniform(10000, 250000))
            tot_ask = int(random.uniform(12000, 270000))
            
        spread = 0.05 if price < 100 else (0.10 if price < 500 else 0.50)
        bids = []
        asks = []
        bid_rem = tot_bid
        ask_rem = tot_ask
        
        for k in range(5):
            bp = round(price - spread - (k * spread), 2)
            ap = round(price + spread + (k * spread), 2)
            if k == 4:
                bq = bid_rem
                aq = ask_rem
            else:
                bq = int(tot_bid * (0.35 - (k * 0.06)) * random.uniform(0.8, 1.2))
                aq = int(tot_ask * (0.35 - (k * 0.06)) * random.uniform(0.8, 1.2))
                bid_rem -= bq
                ask_rem -= aq
            bids.append((bp, max(1, bq)))
            asks.append((ap, max(1, aq)))
        return tot_bid, tot_ask, bids, asks

    def _run_demo_simulation(self):
        while self.ws_running:
            time.sleep(random.uniform(0.1, 0.3))
            symbol = random.choice(self.screened_symbols)
            tick = self.ticks.get(symbol)
            if not tick:
                continue
                
            is_liquid = symbol in ["NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:IDEA-EQ", "NSE:YESBANK-EQ", "NSE:SOUTHBANK-EQ"]
            
            prev_price = tick.ltp
            change = prev_price * random.normalvariate(0, 0.0008)
            new_price = round(prev_price + change, 2)
            if new_price <= 1.0:
                new_price = 1.0
                
            trade_qty = int(max(1, abs(random.normalvariate(5000 if is_liquid else 100, 2000 if is_liquid else 50))))
            now = time.time()
            
            tick.update_from_tick(new_price, trade_qty, tick.total_volume_today + trade_qty, now)
            tot_bid, tot_ask, bids, asks = self._generate_depth_profile(symbol, new_price, is_liquid)
            tick.update_from_depth(bids, asks, tot_bid, tot_ask)
            
            self.record_trade(symbol, now, trade_qty)
            
            # Dynamic liquidity check
            passed_liquidity = (tot_bid >= config.LIQUIDITY_MIN_QTY) and (tot_ask >= config.LIQUIDITY_MIN_QTY)
            
            with self.lock:
                if passed_liquidity and symbol not in self.active_filtered_symbols:
                    self.active_filtered_symbols.append(symbol)
                elif not passed_liquidity and symbol in self.active_filtered_symbols:
                    if len(self.active_filtered_symbols) > 4 or symbol not in ["NSE:SUZLON-EQ", "NSE:ZOMATO-EQ"]:
                        self.active_filtered_symbols.remove(symbol)
                        
            self._update_live_candle(symbol, new_price, trade_qty, now)
            self._notify_tick(tick)

    def stop(self):
        self.ws_running = False
        if self.websocket:
            try:
                self.websocket.close()
            except Exception as e:
                print(f"Error closing Fyers WebSocket: {e}")

    def _run_price_screening(self):
        """Screens all symbols from config by LTP using Quotes REST API."""
        print(f"Running price screening on {len(self.all_symbols)} symbols...")
        self.screened_symbols = []
        
        # Fyers quotes API allows max 50 symbols in one call, chunk them
        chunk_size = 40
        for i in range(0, len(self.all_symbols), chunk_size):
            chunk = self.all_symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            
            try:
                response = self.fyers.quotes({"symbols": symbols_str})
                if response and response.get("s") == "ok":
                    quotes_data = response.get("d", [])
                    for q in quotes_data:
                        symbol = q.get("n")  # n is symbol name in quotes response
                        ltp = q.get("v", {}).get("lp", 0.0)  # lp is last price
                        
                        if config.PRICE_MIN <= ltp <= config.PRICE_MAX:
                            self.screened_symbols.append(symbol)
                            print(f"Passed Price Screen: {symbol} (LTP: ₹{ltp})")
            except Exception as e:
                print(f"Error fetching quotes for screening: {e}")

    def _load_historical_candles(self, symbol: str):
        """Loads last 240 1-minute historical candles from Fyers History API."""
        if symbol in self.history_loaded_symbols:
            return
            
        print(f"Fetching historical candles for: {symbol}")
        
        # Define range (last 2 days to cover weekends/holidays)
        today = datetime.now()
        from_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        data = {
            "symbol": symbol,
            "resolution": "1",
            "date_format": "0",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }
        
        try:
            response = self.fyers.history(data=data)
            if response and response.get("s") == "ok":
                candles_raw = response.get("candles", [])
                
                # Convert raw candles to Candle objects
                # Raw format: [timestamp_epoch, open, high, low, close, volume]
                candles_list = []
                # Keep last 240 candles
                for c in candles_raw[-240:]:
                    candle = Candle(
                        timestamp=float(c[0]),
                        open_pr=float(c[1]),
                        high_pr=float(c[2]),
                        low_pr=float(c[3]),
                        close_pr=float(c[4]),
                        volume=float(c[5])
                    )
                    candles_list.append(candle)
                    
                    # Pre-populate trade history for rolling volume aggregations
                    self.record_trade(symbol, float(c[0]), int(c[5]))
                
                with self.lock:
                    self.candles[symbol] = candles_list
                self.history_loaded_symbols.add(symbol)
                print(f"Successfully loaded {len(candles_list)} historical candles for {symbol}")
            else:
                print(f"Failed to fetch history for {symbol}: {response}")
        except Exception as e:
            print(f"Exception fetching history for {symbol}: {e}")

    def _connect_websocket(self):
        """Initializes and connects Fyers Data WebSocket."""
        # WebSocket Callbacks
        def on_message(message):
            self._handle_ws_message(message)

        def on_error(message):
            print(f"Fyers WS Error: {message}")

        def on_close(message):
            print(f"Fyers WS Closed: {message}")

        def on_open():
            # Subscribe to symbol ticks (price/volume) and depth (bid/ask book)
            # Fyers requires separate subscriptions for SymbolUpdate and DepthUpdate
            print(f"Fyers WS Connected. Subscribing to {len(self.screened_symbols)} screened symbols...")
            
            # Subscribe to prices (SymbolUpdate)
            self.websocket.subscribe(symbols=self.screened_symbols, data_type="SymbolUpdate")
            
            # Subscribe to market depth (DepthUpdate)
            self.websocket.subscribe(symbols=self.screened_symbols, data_type="DepthUpdate")

        # Initialize the Fyers data socket
        self.websocket = data_ws.FyersDataSocket(
            access_token=self.full_token,
            log_path="",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # This blocks, so we run it inside the daemon thread
        try:
            self.websocket.connect()
        except Exception as e:
            print(f"WebSocket connection exception: {e}")

    def _handle_ws_message(self, message: Dict[str, Any]):
        """Parses websocket packets and updates ticker state."""
        # Messages from Fyers can be lists or dicts
        msg_type = message.get("type")
        symbol = message.get("symbol")
        
        if not symbol or symbol not in self.ticks:
            return
            
        tick = self.ticks[symbol]
        now = time.time()
        
        if msg_type == "sf":  # Symbol Update (Quotes)
            # Extract fields
            ltp = message.get("ltp")
            last_qty = message.get("last_traded_qty", 0)
            vol = message.get("vol_traded_today", 0)
            
            if ltp is not None:
                tick.update_from_tick(ltp, last_qty, vol, now)
                self.record_trade(symbol, now, last_qty)
                
                # Check price range filter dynamically (in case price shifts)
                if not (config.PRICE_MIN <= ltp <= config.PRICE_MAX):
                    # Remove from screened if price moves outside limits
                    if symbol in self.active_filtered_symbols:
                        self.active_filtered_symbols.remove(symbol)
                
                # Dynamically update open 1-minute candle
                self._update_live_candle(symbol, ltp, last_qty, now)
                self._notify_tick(tick)
                
        elif msg_type == "dp":  # Depth Update
            # Extract depth data
            # Bids/Asks in Fyers are lists of dicts containing 'price' and 'volume' or 'qty'
            # Check structure in Fyers API v3 docs
            bids_raw = message.get("bids", [])
            asks_raw = message.get("asks", [])
            tot_buy = message.get("tot_buy_qty", 0)
            tot_sell = message.get("tot_sell_qty", 0)
            
            bids = [(b.get("price", 0.0), b.get("qty", 0)) for b in bids_raw]
            asks = [(a.get("price", 0.0), a.get("qty", 0)) for a in asks_raw]
            
            tick.update_from_depth(bids, asks, tot_buy, tot_sell)
            
            # Apply dynamic Liquidity Screening
            passed_liquidity = (tot_buy >= config.LIQUIDITY_MIN_QTY) and (tot_sell >= config.LIQUIDITY_MIN_QTY)
            
            if passed_liquidity:
                if symbol not in self.active_filtered_symbols:
                    # Once a symbol passes the liquidity screen, load history and add it
                    # Fetching history is run in background to avoid blocking socket thread
                    threading.Thread(target=self._initialize_liquid_symbol, args=(symbol,), daemon=True).start()
            else:
                if symbol in self.active_filtered_symbols:
                    self.active_filtered_symbols.remove(symbol)
                    
            self._notify_tick(tick)

    def _initialize_liquid_symbol(self, symbol: str):
        """Initializes a symbol when it first passes liquidity requirements."""
        self._load_historical_candles(symbol)
        with self.lock:
            if symbol not in self.active_filtered_symbols:
                self.active_filtered_symbols.append(symbol)

    def _update_live_candle(self, symbol: str, price: float, qty: int, now: float):
        """Updates and rotates candles based on live websocket ticks."""
        with self.lock:
            history = self.candles.get(symbol)
            if not history:
                # History not loaded yet
                return
                
            current_minute_start = int(now / 60) * 60
            last_candle = history[-1]
            
            if last_candle.timestamp == current_minute_start:
                last_candle.high = max(last_candle.high, price)
                last_candle.low = min(last_candle.low, price)
                last_candle.close = price
                last_candle.volume += qty
            else:
                new_candle = Candle(
                    timestamp=current_minute_start,
                    open_pr=last_candle.close,
                    high_pr=max(last_candle.close, price),
                    low_pr=min(last_candle.close, price),
                    close_pr=price,
                    volume=qty
                )
                history.append(new_candle)
                if len(history) > 300:
                    history.pop(0)
                    
                # Fire candle callback in background
                threading.Thread(target=self._notify_candle_close, args=(symbol, list(history)), daemon=True).start()
