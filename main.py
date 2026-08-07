import os
import json
import time
import webbrowser
import threading
from typing import Dict, List, Any, Optional
import numpy as np
import customtkinter as ctk

import config
import indicators
from fyers_apiv3 import fyersModel
from simulator import TickData, Candle, SimulatorFeed, FyersFeed
from ml_model import CrossoverPredictor
from ui_components import RealTimeTable, CanvasChart, MarketDepthPanel, MLDiagnosticsPanel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class StockAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Main Window
        self.title("AI/ML-Based Stock Market Screening & Analysis System (NSE)")
        self.geometry("1400x850")
        self.configure(fg_color=config.BG_MAIN)
        
        # Core Models & State
        self.predictor = CrossoverPredictor()
        self.feed: Optional[Any] = None
        self.feed_type = ctk.StringVar(value="SIMULATOR")  # "SIMULATOR" or "FYERS"
        self.selected_symbol = ctk.StringVar(value="")
        
        # Keep track of active symbol configurations
        self.active_symbols: List[str] = []
        self.latest_ticks: Dict[str, TickData] = {}
        self.symbol_smma20: Dict[str, np.ndarray] = {}
        self.symbol_smma120: Dict[str, np.ndarray] = {}
        self.latest_predictions: Dict[str, Dict[str, Any]] = {}
        
        self.lock = threading.Lock()
        
       
        self._build_ui()
        
      
        self._start_feed()
        
    
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
       
        self.after(200, self._refresh_loop)

    def _build_ui(self):
        """Builds the main dashboard panels."""
        self.grid_columnconfigure(0, weight=3)  
        self.grid_columnconfigure(1, weight=1)  
        self.grid_rowconfigure(0, weight=0)    
        self.grid_rowconfigure(1, weight=1)     
        
       
        self.ctrl_bar = ctk.CTkFrame(self, fg_color=config.BG_CARD, height=60, corner_radius=0)
        self.ctrl_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
   
        self.lbl_brand = ctk.CTkLabel(
            self.ctrl_bar, text="TRADING SYSTEMS", 
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=15, weight="bold"),
            text_color=config.COLOR_ACCENT
        )
        self.lbl_brand.pack(side="left", padx=20)
        
   
        self.lbl_mode = ctk.CTkLabel(self.ctrl_bar, text="Mode:", font=(config.FONT_FAMILY, 11, "bold"))
        self.lbl_mode.pack(side="left", padx=(10, 5))
        
        self.opt_mode = ctk.CTkOptionMenu(
            self.ctrl_bar, values=["SIMULATOR", "FYERS LIVE API"],
            command=self._on_mode_changed, variable=self.feed_type,
            fg_color=config.BG_CONTROL, button_color="#334155"
        )
        self.opt_mode.pack(side="left", padx=5)
        
        # Status Light indicator
        self.status_frame = ctk.CTkFrame(self.ctrl_bar, fg_color="transparent")
        self.status_frame.pack(side="left", padx=20)
        
        self.status_light = ctk.CTkLabel(
            self.status_frame, text="●", text_color=config.COLOR_BULLISH, font=("Segoe UI", 16)
        )
        self.status_light.pack(side="left")
        
        self.status_text = ctk.CTkLabel(
            self.status_frame, text="SIMULATOR LIVE (MARKET CLOSED OK)", font=(config.FONT_FAMILY, 10, "bold")
        )
        self.status_text.pack(side="left", padx=5)

        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(0, weight=1)  
        self.left_panel.grid_rowconfigure(1, weight=1)  
        
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)  
        self.right_panel.grid_rowconfigure(1, weight=1)  
        
        self.table = RealTimeTable(self.left_panel, on_row_select_callback=self.selected_symbol)
        self.table.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        self.chart = CanvasChart(self.left_panel)
        self.chart.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        
        self.depth_panel = MarketDepthPanel(self.right_panel)
        self.depth_panel.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        # 6. ML Diagnostics Panel (Right Bottom)
        self.ml_panel = MLDiagnosticsPanel(self.right_panel, predictor=self.predictor)
        self.ml_panel.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

    def _on_mode_changed(self, mode: str):
        """Triggers UI setup popups or stops current feed based on mode selection."""
        if mode == "SIMULATOR":
            self.status_light.configure(text_color=config.COLOR_BULLISH)
            self.status_text.configure(text="SIMULATOR LIVE (MARKET CLOSED OK)")
            self._start_feed()
        else:
            # Fyers Live API chosen
            self._show_fyers_login_dialog()

    def _show_fyers_login_dialog(self):
        """Displays Fyers API credentials configuration overlay window."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Fyers API Credentials Integration")
        dialog.geometry("500x380")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Center dialog
        dialog.update_idletasks()
        dx = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        dy = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{dx}+{dy}")
        
        lbl_info = ctk.CTkLabel(
            dialog, text="Connect Fyers API V3", 
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=15, weight="bold"),
            text_color=config.COLOR_ACCENT
        )
        lbl_info.pack(pady=15)
        
        # Form fields
        form = ctk.CTkFrame(dialog, fg_color="transparent")
        form.pack(fill="x", padx=40)
        form.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(form, text="Client ID (App ID):").grid(row=0, column=0, sticky="w", pady=5)
        ent_client_id = ctk.CTkEntry(form, placeholder_text="XXXXX-100")
        ent_client_id.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        ctk.CTkLabel(form, text="Secret Key:").grid(row=1, column=0, sticky="w", pady=5)
        ent_secret_key = ctk.CTkEntry(form, placeholder_text="Masked Key", show="*")
        ent_secret_key.grid(row=1, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        ctk.CTkLabel(form, text="Redirect URI:").grid(row=2, column=0, sticky="w", pady=5)
        ent_redirect_uri = ctk.CTkEntry(form, placeholder_text="https://trade.fyers.in/...")
        ent_redirect_uri.insert(0, "https://trade.fyers.in/api-login/redirect-uri/")
        ent_redirect_uri.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        # Load cached settings
        cached_file = "fyers_session.json"
        if os.path.exists(cached_file):
            try:
                with open(cached_file, "r") as f:
                    cache = json.load(f)
                    ent_client_id.insert(0, cache.get("client_id", ""))
                    ent_secret_key.insert(0, cache.get("secret_key", ""))
            except:
                pass
                
        # Link Generation Button
        def generate_link():
            cid = ent_client_id.get().strip()
            ruri = ent_redirect_uri.get().strip()
            if not cid:
                lbl_status.configure(text="App ID (Client ID) is required!", text_color=config.COLOR_BEARISH)
                return
            if cid.upper().startswith("DEMO"):
                login_url = f"{ruri}?auth_code=demoPayload1234567890.demoSignature987654321&state=antigravity"
                lbl_status.configure(text="Simulating login... Check browser!", text_color=config.COLOR_WARNING)
            else:
                login_url = f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={cid}&redirect_uri={ruri}&response_type=code&state=antigravity"
            webbrowser.open(login_url)
            
        btn_link = ctk.CTkButton(dialog, text="Generate Login URL", command=generate_link, fg_color="#475569")
        btn_link.pack(pady=10)
        
        # Auth code entry
        auth_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        auth_frame.pack(fill="x", padx=40, pady=5)
        auth_frame.columnconfigure(1, weight=1)
        
        ctk.CTkLabel(auth_frame, text="Pasted Auth Code:").grid(row=0, column=0, sticky="w")
        ent_auth_code = ctk.CTkEntry(auth_frame, placeholder_text="eyJhbGciOiJ...")
        ent_auth_code.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        
        # Authenticate Action
        def authenticate():
            cid = ent_client_id.get().strip()
            secret = ent_secret_key.get().strip()
            code = ent_auth_code.get().strip()
            ruri = ent_redirect_uri.get().strip()
            
            if not (cid and secret and code):
                lbl_status.configure(text="Fill Client ID, Secret, and Auth Code!", text_color=config.COLOR_BEARISH)
                return
                
            lbl_status.configure(text="Exchanging code for token...", text_color=config.COLOR_WARNING)
            dialog.update()
            
            # Start Fyers Session model
            try:
                # If demo credentials, bypass real API exchange
                if cid.upper().startswith("DEMO") or code.lower().startswith("demo"):
                    response = {
                        "s": "ok",
                        "access_token": f"demo_token_{int(time.time())}"
                    }
                else:
                    session = fyersModel.SessionModel(
                        client_id=cid,
                        secret_key=secret,
                        redirect_uri=ruri,
                        response_type="code",
                        grant_type="authorization_code"
                    )
                    session.set_token(code)
                    response = session.generate_token()
                
                if response and response.get("s") == "ok":
                    token = response.get("access_token")
                    
                    # Cache credentials (excluding token directly if requested, or saving for session)
                    with open(cached_file, "w") as f:
                        json.dump({"client_id": cid, "secret_key": secret}, f)
                        
                    lbl_status.configure(text="Success! Starting data feed...", text_color=config.COLOR_BULLISH)
                    dialog.update()
                    time.sleep(1)
                    
                    # Start feed in Fyers mode
                    self._start_fyers_feed(cid, token)
                    dialog.destroy()
                else:
                    lbl_status.configure(text=f"Auth Failed: {response.get('message', 'Unknown error')}", text_color=config.COLOR_BEARISH)
            except Exception as e:
                lbl_status.configure(text=f"Exception: {str(e)}", text_color=config.COLOR_BEARISH)
                
        btn_auth = ctk.CTkButton(dialog, text="Authenticate & Connect", command=authenticate, fg_color=config.COLOR_ACCENT)
        btn_auth.pack(pady=15)
        
        lbl_status = ctk.CTkLabel(dialog, text="", font=(config.FONT_FAMILY, 10, "bold"))
        lbl_status.pack()
        
        # Callback on window close to revert choice if not authenticated
        def on_close_dialog():
            if not self.feed or isinstance(self.feed, SimulatorFeed):
                self.feed_type.set("SIMULATOR")
                self.opt_mode.configure(variable=self.feed_type)
            dialog.destroy()
            
        dialog.protocol("WM_DELETE_WINDOW", on_close_dialog)

    def _start_fyers_feed(self, client_id: str, token: str):
        """Stops simulator and hooks up the Fyers Feed listener."""
        self.status_light.configure(text_color=config.COLOR_WARNING)
        self.status_text.configure(text="CONNECTING TO FYERS LIVE...")
        
        if self.feed:
            self.feed.stop()
            
        # Initialize Fyers feed
        self.feed = FyersFeed(client_id, token)
        self.feed.register_tick_callback(self._on_tick)
        self.feed.register_candle_callback(self._on_candle_close)
        
        # Run startup in background to avoid freezing the GUI
        def launch():
            try:
                self.feed.start()
                self.status_light.configure(text_color=config.COLOR_BULLISH)
                self.status_text.configure(text="FYERS LIVE API ACTIVE")
            except Exception as e:
                print(f"Failed to start Fyers feed: {e}")
                self.status_light.configure(text_color=config.COLOR_BEARISH)
                self.status_text.configure(text="FYERS CONNECT ERROR - FALLBACK TO SIM")
                self.feed_type.set("SIMULATOR")
                self._start_feed()
                
        threading.Thread(target=launch, daemon=True).start()

    def _start_feed(self):
        """Instantiates and starts the Simulator data feed."""
        if self.feed:
            self.feed.stop()
            
        self.feed = SimulatorFeed()
        self.feed.register_tick_callback(self._on_tick)
        self.feed.register_candle_callback(self._on_candle_close)
        self.feed.start()
        
        # Initialise rows in the table
        # Since simulator filters for liquid stocks:
        liquid_sim_symbols = [s for s in config.DEFAULT_SYMBOLS if s in [
            "NSE:SUZLON-EQ", "NSE:GMRINFRA-EQ", "NSE:PNB-EQ", "NSE:TATASTEEL-EQ", 
            "NSE:ZOMATO-EQ", "NSE:NHPC-EQ", "NSE:IRFC-EQ", "NSE:IREDA-EQ", "NSE:YESBANK-EQ"
        ]]
        
        self.active_symbols = liquid_sim_symbols
        self.table.rebuild_rows(liquid_sim_symbols)
        
        # Pre-populate dashboard with historical trade data
        self._prepopulate_historical_crossovers()

    def _prepopulate_historical_crossovers(self):
        """Simulates historical crossovers over the pregenerated candles to populate the dashboard immediately at startup."""
        import random
        # Temporarily disable real-time CSV file logging during backtest pre-population
        original_log_to_csv = self.predictor.log_trade_to_csv
        original_save_csv = self.predictor.save_backtest_report_csv
        
        self.predictor.log_trade_to_csv = lambda trade, portfolio: None
        self.predictor.save_backtest_report_csv = lambda: None
        
        for symbol in self.active_symbols:
            history = self.feed.get_historical_candles(symbol)
            if len(history) < 125:
                continue
                
            closes = [c.close for c in history]
            smma20_arr = indicators.calculate_smma(closes, config.SMMA_FAST_PERIOD)
            smma120_arr = indicators.calculate_smma(closes, config.SMMA_SLOW_PERIOD)
            
            for idx in range(125, len(history)):
                p_20 = smma20_arr[idx - 1]
                p_120 = smma120_arr[idx - 1]
                c_20 = smma20_arr[idx]
                c_120 = smma120_arr[idx]
                
                if np.isnan(p_20) or np.isnan(p_120) or np.isnan(c_20) or np.isnan(c_120):
                    continue
                    
                signal = "NONE"
                if p_20 <= p_120 and c_20 > c_120:
                    signal = "BUY"
                elif p_20 >= p_120 and c_20 < c_120:
                    signal = "SELL"
                    
                if signal != "NONE":
                    candle = history[idx]
                    # Simulate realistic feature metrics for backtested trades
                    ratio_val = random.uniform(0.7, 2.2)
                    side_share_val = random.uniform(0.55, 0.85) if signal == "BUY" else random.uniform(0.15, 0.45)
                    
                    features = {
                        "A2": 1500.0 * ratio_val,
                        "A5": 1500.0,
                        "ratio": ratio_val,
                        "side_share": side_share_val,
                        "max_ltq": 5000.0 if ratio_val > 1.5 else 1500.0,
                        "slope_gap": abs(c_20 - c_120) / candle.close * 100.0,
                        "time_of_day": (candle.timestamp - history[0].timestamp) / 60.0
                    }
                    
                    decision, prob, reason = self.predictor.handle_crossover(
                        symbol=symbol,
                        signal_type=signal,
                        price=candle.close,
                        timestamp=candle.timestamp,
                        features=features
                    )
                    
                    # Log the historical crossover details into the logs tab
                    self.ml_panel.log_crossover(symbol, signal, candle.close, features, decision, prob, reason)
                    
        # Restore CSV logging functionality
        self.predictor.log_trade_to_csv = original_log_to_csv
        self.predictor.save_backtest_report_csv = original_save_csv
        
        # Initial refresh of diagnostics
        self.ml_panel.refresh_diagnostics()

    def _on_tick(self, tick: TickData):
        """Processes real-time ticks to perform indicator updates and UI updates."""
        # Update latest ticks cache
        with self.lock:
            self.latest_ticks[tick.symbol] = tick
            
        # Get candles history to calculate current indicators incrementally
        candles = self.feed.get_historical_candles(tick.symbol)
        if len(candles) < 20:
            return
            
        # Retrieve previous closed candle's SMMA values
        closed_smma20_arr = self.symbol_smma20.get(tick.symbol, np.array([]))
        closed_smma120_arr = self.symbol_smma120.get(tick.symbol, np.array([]))
        
        # If arrays are not populated yet, compute them from history
        if len(closed_smma20_arr) < len(candles) or len(closed_smma120_arr) < len(candles):
            closes = [c.close for c in candles]
            closed_smma20_arr = indicators.calculate_smma(closes, config.SMMA_FAST_PERIOD)
            closed_smma120_arr = indicators.calculate_smma(closes, config.SMMA_SLOW_PERIOD)
            self.symbol_smma20[tick.symbol] = closed_smma20_arr
            self.symbol_smma120[tick.symbol] = closed_smma120_arr
            
        # Calculate live incremental SMMA value using current tick price
        # This reflects tick-by-tick changes in the indicators
        live_smma20 = indicators.calculate_smma_step(
            prev_smma=closed_smma20_arr[-2],  # Prev closed bar's SMMA
            current_price=tick.ltp,
            period=config.SMMA_FAST_PERIOD,
            history_len=len(candles)
        )
        
        live_smma120 = indicators.calculate_smma_step(
            prev_smma=closed_smma120_arr[-2],
            current_price=tick.ltp,
            period=config.SMMA_SLOW_PERIOD,
            history_len=len(candles)
        )
        
        # Get rolling quantities
        vol_5m = self.feed.get_executed_quantity(tick.symbol, 5)
        vol_20m = self.feed.get_executed_quantity(tick.symbol, 20)
        vol_60m = self.feed.get_executed_quantity(tick.symbol, 60)
        
        avg_20m = self.feed.get_average_ltp(tick.symbol, 20)
        avg_60m = self.feed.get_average_ltp(tick.symbol, 60)
        
        # Get LTQ Metrics
        a2, a5, ratio, side_share, max_ltq = self.feed.get_ltq_metrics(tick.symbol)
        
        # Get crossover signal and AI recommendation if cached
        pred_cache = self.latest_predictions.get(tick.symbol, {})
        
        # Format update packet
        update_data = {
            "ltp": tick.ltp,
            "bid_price": tick.bid_price,
            "bid_qty": tick.bid_qty,
            "ask_price": tick.ask_price,
            "ask_qty": tick.ask_qty,
            "smma20": live_smma20,
            "smma120": live_smma120,
            "vol_5m": vol_5m,
            "vol_20m": vol_20m,
            "vol_60m": vol_60m,
            "avg_price_20m": avg_20m,
            "avg_price_60m": avg_60m,
            "ltq": tick.last_traded_qty,
            "a2": a2,
            "a5": a5,
            "ratio": ratio,
            "side_share": side_share,
            "signal": pred_cache.get("signal", "NONE"),
            "ml_rec": pred_cache.get("rec", "NONE"),
            "ml_conf": pred_cache.get("conf", 0.0)
        }
        
        # Update Table widget
        self.table.update_row_data(tick.symbol, update_data)

    def _on_candle_close(self, symbol: str, history: List[Candle]):
        """Runs indicator recalculation and crossover check on minute candle boundaries."""
        if len(history) < 125:
            return
            
        closes = [c.close for c in history]
        
        # Full recalculation of historical SMMA arrays
        smma20_arr = indicators.calculate_smma(closes, config.SMMA_FAST_PERIOD)
        smma120_arr = indicators.calculate_smma(closes, config.SMMA_SLOW_PERIOD)
        
        with self.lock:
            self.symbol_smma20[symbol] = smma20_arr
            self.symbol_smma120[symbol] = smma120_arr
            
        # Crossover Detection
        # Check between last closed candle (index -1) and previous (index -2)
        idx_prev = -2
        idx_curr = -1
        
        p_20 = smma20_arr[idx_prev]
        p_120 = smma120_arr[idx_prev]
        c_20 = smma20_arr[idx_curr]
        c_120 = smma120_arr[idx_curr]
        
        if np.isnan(p_20) or np.isnan(p_120) or np.isnan(c_20) or np.isnan(c_120):
            return
            
        signal = "NONE"
        if p_20 <= p_120 and c_20 > c_120:
            signal = "BUY"
        elif p_20 >= p_120 and c_20 < c_120:
            signal = "SELL"
            
        if signal != "NONE":
            # Crossover Detected! Invoke AI/ML analyzer
            tick = self.latest_ticks.get(symbol)
            if tick:
                features = self.predictor.extract_features(symbol, signal, history, tick, self.feed)
                if features:
                    decision, prob, reason = self.predictor.handle_crossover(symbol, signal, tick.ltp, time.time(), features)
                    
                    with self.lock:
                        self.latest_predictions[symbol] = {
                            "signal": signal,
                            "rec": decision,
                            "conf": prob,
                            "reason": reason
                        }
                    print(f"Crossover Detected: {symbol} | {signal} | Rec: {decision} | Conf: {prob:.1f}% | {reason}")
                    self.ml_panel.log_crossover(symbol, signal, tick.ltp, features, decision, prob, reason)

    def _refresh_loop(self):
        """Core UI refresh loop updating charts, depth, and diagnostics frames."""
        try:
            # Check if active symbols changed in Fyers Live feed
            if isinstance(self.feed, FyersFeed):
                with self.lock:
                    new_active = list(self.feed.active_filtered_symbols)
                if set(new_active) != set(self.active_symbols):
                    self.active_symbols = new_active
                    self.table.rebuild_rows(new_active)
                    
            # 1. Update ML Diagnostics Panel (win rate, features, active trades)
            self.predictor.update_active_trades(self.latest_ticks, time.time())
            self.ml_panel.refresh_diagnostics()
            
            # 2. Update Detail panel for selected symbol
            sel_symbol = self.selected_symbol.get()
            if sel_symbol and sel_symbol in self.latest_ticks:
                tick = self.latest_ticks[sel_symbol]
                
                # Update Market Depth
                self.depth_panel.update_depth(tick)
                
                # Update Candlestick/SMMA chart
                candles = self.feed.get_historical_candles(sel_symbol)
                s20 = self.symbol_smma20.get(sel_symbol, np.array([]))
                s120 = self.symbol_smma120.get(sel_symbol, np.array([]))
                
                if len(candles) > 0 and len(s20) == len(candles) and len(s120) == len(candles):
                    self.chart.update_data(candles, s20, s120)
                    
        except Exception as e:
            print(f"Error in refresh loop: {e}")
            
        # Reschedule loop
        self.after(150, self._refresh_loop)

    def on_closing(self):
        """Cleans up threads and files when exiting."""
        if self.feed:
            self.feed.stop()
        self.destroy()

if __name__ == "__main__":
    app = StockAnalyzerApp()
    app.mainloop()
