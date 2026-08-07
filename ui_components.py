# ui_components.py
# Implements the custom UI widgets for the Stock market terminal.
# Includes RealTimeTable, CanvasChart, MarketDepthPanel, and MLDiagnosticsPanel.

import tkinter as tk
import customtkinter as ctk
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import config
from simulator import TickData, Candle
from ml_model import TradeRecord

class RealTimeTable(ctk.CTkFrame):
    """High-performance real-time stock grid showing price, depth, indicators, and ML signals."""
    def __init__(self, master, on_row_select_callback: ctk.StringVar, **kwargs):
        super().__init__(master, **kwargs)
        self.on_row_select = on_row_select_callback
        self.configure(fg_color=config.BG_CARD)
        
        self.headers = [
            "Symbol", "LTP", "Bid (Qty)", "Ask (Qty)", 
            "SMMA(20)", "SMMA(120)", 
            "LTQ", "A2(2M)", "A5(5M)", "A2/A5", "SideShare",
            "Vol(5M)", "Vol(20M)", "Vol(60M)", 
            "Avg(20M)", "Avg(60M)", "Signal", "ML Rec"
        ]
        
        self.row_symbols: List[str] = []
        self.cell_widgets: Dict[str, Dict[str, ctk.CTkLabel]] = {}
        self.row_frames: Dict[str, ctk.CTkFrame] = {}
        self.prev_ltps: Dict[str, float] = {}
        self.selected_symbol: Optional[str] = None
        
        # Setup Grid structure
        self._setup_table_headers()
        
        # Scrollable container for rows
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.grid(row=1, column=0, columnspan=len(self.headers), sticky="nsew", padx=2, pady=2)
        
        # Column weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def _setup_table_headers(self):
        """Creates the header row widgets."""
        header_frame = ctk.CTkFrame(self, fg_color=config.BG_CONTROL, height=35, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=1)
        
        for col_idx, text in enumerate(self.headers):
            # Give symbol column more weight
            weight = 2 if col_idx == 0 else (1 if col_idx in [12, 13] else 1)
            header_frame.grid_columnconfigure(col_idx, weight=weight)
            
            lbl = ctk.CTkLabel(
                header_frame, text=text, 
                font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"),
                text_color=config.TEXT_SECONDARY, anchor="center"
            )
            lbl.grid(row=0, column=col_idx, padx=5, pady=5, sticky="ew")

    def rebuild_rows(self, symbols: List[str]):
        """Re-instantiates the grid rows when the screened/filtered list changes."""
        # Clear existing row widgets
        for frame in self.row_frames.values():
            frame.destroy()
            
        self.row_symbols = symbols
        self.cell_widgets.clear()
        self.row_frames.clear()
        
        for idx, symbol in enumerate(symbols):
            # Alternating row background colors
            bg_color = "#1a2236" if idx % 2 == 0 else "#151b2d"
            
            row_frame = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, height=30, corner_radius=4)
            row_frame.grid(row=idx, column=0, sticky="ew", pady=1, padx=2)
            self.row_frames[symbol] = row_frame
            
            # Setup row columns
            self.cell_widgets[symbol] = {}
            for col_idx in range(len(self.headers)):
                # Adjust column weight
                weight = 2 if col_idx == 0 else 1
                row_frame.grid_columnconfigure(col_idx, weight=weight)
                
                # Alignments: Left for symbol, center for others
                anchor = "w" if col_idx == 0 else "center"
                
                lbl = ctk.CTkLabel(
                    row_frame, text="-", 
                    font=ctk.CTkFont(family=config.FONT_FAMILY, size=11),
                    text_color=config.TEXT_PRIMARY, anchor=anchor
                )
                lbl.grid(row=0, column=col_idx, padx=5, pady=4, sticky="ew")
                self.cell_widgets[symbol][self.headers[col_idx]] = lbl
                
            # Row selection click binding
            row_frame.bind("<Button-1>", lambda event, sym=symbol: self._select_row(sym))
            for widget in row_frame.winfo_children():
                widget.bind("<Button-1>", lambda event, sym=symbol: self._select_row(sym))

        # Re-apply selection styling if previously selected symbol is still present
        if self.selected_symbol in symbols:
            self._select_row(self.selected_symbol)
        elif symbols:
            self._select_row(symbols[0])

    def _select_row(self, symbol: str):
        """Highlights the selected row and triggers selection updates."""
        self.selected_symbol = symbol
        self.on_row_select.set(symbol)
        
        for sym, frame in self.row_frames.items():
            if sym == symbol:
                frame.configure(fg_color="#2e3b5e", border_width=1, border_color=config.COLOR_ACCENT)
            else:
                # Reset alternating bg colors
                idx = self.row_symbols.index(sym)
                bg_color = "#1a2236" if idx % 2 == 0 else "#151b2d"
                frame.configure(fg_color=bg_color, border_width=0)

    def update_row_data(self, symbol: str, data: Dict[str, Any]):
        """Dynamically updates the cell texts and handles flashing on price change."""
        if symbol not in self.cell_widgets:
            return
            
        cells = self.cell_widgets[symbol]
        
        # LTP Flashing logic
        new_ltp = data.get("ltp", 0.0)
        prev_ltp = self.prev_ltps.get(symbol, 0.0)
        ltp_label = cells["LTP"]
        
        ltp_label.configure(text=f"₹{new_ltp:.2f}")
        if prev_ltp != 0.0 and new_ltp != prev_ltp:
            if new_ltp > prev_ltp:
                ltp_label.configure(text_color=config.COLOR_BULLISH, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
                self.after(250, lambda lbl=ltp_label: lbl.configure(text_color=config.TEXT_PRIMARY, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11)))
            else:
                ltp_label.configure(text_color=config.COLOR_BEARISH, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
                self.after(250, lambda lbl=ltp_label: lbl.configure(text_color=config.TEXT_PRIMARY, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11)))
                
        self.prev_ltps[symbol] = new_ltp
        
        # Update details
        cells["Symbol"].configure(text=symbol.replace("NSE:", "").replace("-EQ", ""), font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
        
        # Depth
        cells["Bid (Qty)"].configure(text=f"₹{data.get('bid_price', 0.0):.2f} ({self._format_qty(data.get('bid_qty', 0))})")
        cells["Ask (Qty)"].configure(text=f"₹{data.get('ask_price', 0.0):.2f} ({self._format_qty(data.get('ask_qty', 0))})")
        
        # Indicators
        s20 = data.get("smma20")
        s120 = data.get("smma120")
        cells["SMMA(20)"].configure(text=f"₹{s20:.2f}" if s20 and not np.isnan(s20) else "-")
        cells["SMMA(120)"].configure(text=f"₹{s120:.2f}" if s120 and not np.isnan(s120) else "-")
        
        # LTQ Burst
        ltq = data.get("ltq", 0)
        a2 = data.get("a2", 0.0)
        a5 = data.get("a5", 0.0)
        ratio = data.get("ratio", 1.0)
        side_share = data.get("side_share", 0.5)
        
        cells["LTQ"].configure(text=self._format_qty(ltq))
        cells["A2(2M)"].configure(text=f"{a2:.1f}" if a2 > 0 else "-")
        cells["A5(5M)"].configure(text=f"{a5:.1f}" if a5 > 0 else "-")
        cells["A2/A5"].configure(text=f"{ratio:.2f}x" if a5 > 0 else "-")
        cells["SideShare"].configure(text=f"{side_share*100:.1f}%" if a2 > 0 else "-")
        
        # Aggregations
        cells["Vol(5M)"].configure(text=self._format_qty(data.get("vol_5m", 0)))
        cells["Vol(20M)"].configure(text=self._format_qty(data.get("vol_20m", 0)))
        cells["Vol(60M)"].configure(text=self._format_qty(data.get("vol_60m", 0)))
        
        ap20 = data.get("avg_price_20m", 0.0)
        ap60 = data.get("avg_price_60m", 0.0)
        cells["Avg(20M)"].configure(text=f"₹{ap20:.2f}" if ap20 > 0 else "-")
        cells["Avg(60M)"].configure(text=f"₹{ap60:.2f}" if ap60 > 0 else "-")
        
        # Signals & ML
        sig = data.get("signal", "NONE")
        if sig == "BUY":
            cells["Signal"].configure(text="BUY", text_color=config.COLOR_BULLISH, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
        elif sig == "SELL":
            cells["Signal"].configure(text="SELL", text_color=config.COLOR_BEARISH, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
        else:
            cells["Signal"].configure(text="-", text_color=config.TEXT_PRIMARY, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11))
            
        rec = data.get("ml_rec", "NONE")
        conf = data.get("ml_conf", 0.0)
        if rec == "ACCEPT":
            cells["ML Rec"].configure(text=f"ACCEPT ({conf:.1f}%)", text_color=config.COLOR_BULLISH, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
        elif rec == "AVOID":
            cells["ML Rec"].configure(text=f"AVOID ({conf:.1f}%)", text_color=config.COLOR_WARNING, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11, weight="bold"))
        else:
            cells["ML Rec"].configure(text="-", text_color=config.TEXT_PRIMARY, font=ctk.CTkFont(family=config.FONT_FAMILY, size=11))

    def _format_qty(self, qty: int) -> str:
        """Formats volume into compact Indian representation (Lakhs/Cr)."""
        if qty >= 10000000:
            return f"{qty / 10000000:.2f} Cr"
        elif qty >= 100000:
            return f"{qty / 100000:.2f} L"
        elif qty >= 1000:
            return f"{qty / 1000:.1f} K"
        return str(qty)


class CanvasChart(ctk.CTkFrame):
    """Zero-dependency high-performance Candlestick, Volume, and SMMA chart."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=config.BG_CARD)
        
        self.canvas = tk.Canvas(self, bg=config.BG_CARD, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Internal state
        self.candles: List[Candle] = []
        self.smma20: np.ndarray = np.array([])
        self.smma120: np.ndarray = np.array([])
        
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Motion>", self._on_mouse_move)
        
        self.crosshair_x = -1
        self.crosshair_y = -1

    def update_data(self, candles: List[Candle], smma20: np.ndarray, smma120: np.ndarray):
        """Sets data buffers and updates the drawing."""
        self.candles = candles
        self.smma20 = smma20
        self.smma120 = smma120
        self.redraw()

    def redraw(self):
        """Clears and draws the entire canvas."""
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if len(self.candles) < 2 or w < 50 or h < 50:
            self.canvas.create_text(w/2, h/2, text="No Data Available", fill=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 14))
            return
            
        # Chart layouts:
        # Main candle panel: Top 75%
        # Volume panel: Bottom 20%
        # Separator margin: 5%
        main_h = int(h * 0.72)
        vol_h = int(h * 0.20)
        vol_offset = int(h * 0.77)
        
        # Render window: show the last 60 candles for clarity
        max_bars = 60
        candles = self.candles[-max_bars:]
        s20 = self.smma20[-len(candles):]
        s120 = self.smma120[-len(candles):]
        
        # Scaling limits
        prices = [c.high for c in candles] + [c.low for c in candles]
        # Include non-NaN SMMA prices in scaling
        valid_smmas = [v for v in s20 if not np.isnan(v)] + [v for v in s120 if not np.isnan(v)]
        if valid_smmas:
            prices.extend(valid_smmas)
            
        min_p = min(prices) * 0.999
        max_p = max(prices) * 1.001
        p_range = max_p - min_p if max_p - min_p > 0 else 1.0
        
        volumes = [c.volume for c in candles]
        max_v = max(volumes) if max(volumes) > 0 else 1.0
        
        # Margins
        left_margin = 10
        right_margin = 60
        top_margin = 15
        bottom_margin = 15
        
        chart_w = w - left_margin - right_margin
        bar_w = chart_w / len(candles)
        
        # 1. Draw Grid Lines & Price Axis labels
        self._draw_grid_and_axis(w, h, main_h, left_margin, right_margin, top_margin, min_p, max_p, p_range)
        
        # 2. Draw Candlesticks & Volumes & SMMAs
        points_s20: List[Tuple[float, float]] = []
        points_s120: List[Tuple[float, float]] = []
        
        for idx, candle in enumerate(candles):
            # Calculate coordinates
            cx = left_margin + (idx * bar_w) + (bar_w / 2)
            
            # Candle wick
            cy_high = top_margin + (1 - (candle.high - min_p) / p_range) * (main_h - top_margin)
            cy_low = top_margin + (1 - (candle.low - min_p) / p_range) * (main_h - top_margin)
            
            # Candle body
            cy_open = top_margin + (1 - (candle.open - min_p) / p_range) * (main_h - top_margin)
            cy_close = top_margin + (1 - (candle.close - min_p) / p_range) * (main_h - top_margin)
            
            # Volume bar
            vy = h - bottom_margin
            vh = (candle.volume / max_v) * (vol_h)
            vy_top = vy - vh
            
            # Style colors
            is_bullish = candle.close >= candle.open
            color = config.COLOR_BULLISH if is_bullish else config.COLOR_BEARISH
            
            # Draw volume
            self.canvas.create_rectangle(
                left_margin + (idx * bar_w) + 1, vy_top,
                left_margin + ((idx+1) * bar_w) - 1, vy,
                fill=color, outline="", stipple="gray25"  # stipple transparent
            )
            
            # Draw wick
            self.canvas.create_line(cx, cy_high, cx, cy_low, fill=color, width=1.5)
            
            # Draw body (ensure at least 1px height)
            body_top = min(cy_open, cy_close)
            body_bot = max(cy_open, cy_close)
            if body_bot - body_top < 1:
                body_bot = body_top + 1
                
            self.canvas.create_rectangle(
                left_margin + (idx * bar_w) + 1, body_top,
                left_margin + ((idx+1) * bar_w) - 1, body_bot,
                fill=color, outline=color
            )
            
            # Accumulate SMMA coordinates
            if not np.isnan(s20[idx]):
                sy20 = top_margin + (1 - (s20[idx] - min_p) / p_range) * (main_h - top_margin)
                points_s20.append((cx, sy20))
            if not np.isnan(s120[idx]):
                sy120 = top_margin + (1 - (s120[idx] - min_p) / p_range) * (main_h - top_margin)
                points_s120.append((cx, sy120))
                
            # Draw Crossover Markers
            # Detect crossover at this index relative to full history
            hist_idx = len(self.candles) - len(candles) + idx
            if hist_idx > 0:
                p_s20 = self.smma20[hist_idx - 1]
                p_s120 = self.smma120[hist_idx - 1]
                c_s20 = self.smma20[hist_idx]
                c_s120 = self.smma120[hist_idx]
                
                if not np.isnan(p_s20) and not np.isnan(p_s120) and not np.isnan(c_s20) and not np.isnan(c_s120):
                    if p_s20 <= p_s120 and c_s20 > c_s120:  # Buy Crossover
                        # Draw green up triangle above candle high
                        self.canvas.create_polygon(
                            cx, cy_high - 15, cx - 6, cy_high - 5, cx + 6, cy_high - 5,
                            fill=config.COLOR_BULLISH, outline=""
                        )
                    elif p_s20 >= p_s120 and c_s20 < c_s120:  # Sell Crossover
                        # Draw red down triangle below candle low
                        self.canvas.create_polygon(
                            cx, cy_low + 5, cx - 6, cy_low + 15, cx + 6, cy_low + 15,
                            fill=config.COLOR_BEARISH, outline=""
                        )

        # 3. Draw SMMA Overlay Lines
        if len(points_s20) > 1:
            for k in range(len(points_s20) - 1):
                self.canvas.create_line(
                    points_s20[k][0], points_s20[k][1],
                    points_s20[k+1][0], points_s20[k+1][1],
                    fill=config.COLOR_ACCENT, width=1.5
                )
        if len(points_s120) > 1:
            for k in range(len(points_s120) - 1):
                self.canvas.create_line(
                    points_s120[k][0], points_s120[k][1],
                    points_s120[k+1][0], points_s120[k+1][1],
                    fill=config.COLOR_WARNING, width=1.5
                )

        # 4. Draw Crosshair cursor overlays
        if self.crosshair_x > left_margin and self.crosshair_x < w - right_margin:
            # Vertical line
            self.canvas.create_line(self.crosshair_x, top_margin, self.crosshair_x, h - bottom_margin, fill="#475569", dash=(3, 3))
            # Horizontal line
            if self.crosshair_y > top_margin and self.crosshair_y < main_h:
                self.canvas.create_line(left_margin, self.crosshair_y, w - right_margin, self.crosshair_y, fill="#475569", dash=(3, 3))
                
                # Draw price tooltip on Y-axis
                calc_p = min_p + (1 - (self.crosshair_y - top_margin) / (main_h - top_margin)) * p_range
                self.canvas.create_rectangle(
                    w - right_margin, self.crosshair_y - 10,
                    w, self.crosshair_y + 10,
                    fill=config.BG_CONTROL, outline=config.COLOR_ACCENT
                )
                self.canvas.create_text(
                    w - right_margin/2, self.crosshair_y,
                    text=f"{calc_p:.2f}", fill=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 9)
                )

    def _draw_grid_and_axis(self, w: int, h: int, main_h: int, left_margin: int, right_margin: int, top_margin: int, min_p: float, max_p: float, p_range: float):
        """Draws horizontal price grid lines and Y-axis price labels."""
        grid_lines = 4
        axis_x = w - right_margin
        
        for k in range(grid_lines + 1):
            y = top_margin + (k / grid_lines) * (main_h - top_margin)
            # Grid line
            self.canvas.create_line(left_margin, y, axis_x, y, fill="#1e293b", width=1)
            
            # Price Label
            price = max_p - (k / grid_lines) * p_range
            self.canvas.create_text(
                axis_x + 5, y, text=f"{price:.2f}",
                fill=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 9), anchor="w"
            )
            
        # Draw volume panel separator
        self.canvas.create_line(left_margin, main_h + 5, axis_x, main_h + 5, fill="#334155", width=1.5)

    def _on_mouse_move(self, event):
        """Captures mouse movements to draw crosshairs."""
        self.crosshair_x = event.x
        self.crosshair_y = event.y
        self.redraw()


class MarketDepthPanel(ctk.CTkFrame):
    """Displays the real-time Level 2 bid/ask order book depth with ratio bars."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=config.BG_CARD)
        
        # Panel Title
        self.lbl_title = ctk.CTkLabel(
            self, text="Real-Time Market Depth",
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=13, weight="bold"),
            text_color=config.COLOR_ACCENT
        )
        self.lbl_title.pack(anchor="w", padx=10, pady=5)
        
        # Ratio Canvas Bar (Buy vs Sell percent)
        self.ratio_canvas = tk.Canvas(self, height=12, bg=config.BG_CARD, highlightthickness=0)
        self.ratio_canvas.pack(fill="x", padx=10, pady=4)
        
        # Labels for Buy/Sell Qty totals
        self.total_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.total_frame.pack(fill="x", padx=10, pady=2)
        
        self.lbl_total_buy = ctk.CTkLabel(
            self.total_frame, text="Buy: -", text_color=config.COLOR_BULLISH,
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=10, weight="bold")
        )
        self.lbl_total_buy.pack(side="left")
        
        self.lbl_total_sell = ctk.CTkLabel(
            self.total_frame, text="Sell: -", text_color=config.COLOR_BEARISH,
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=10, weight="bold")
        )
        self.lbl_total_sell.pack(side="right")
        
        # Bid/Ask columns container
        self.depth_grid = ctk.CTkFrame(self, fg_color="transparent")
        self.depth_grid.pack(fill="both", expand=True, padx=10, pady=5)
        self.depth_grid.grid_columnconfigure(0, weight=1)
        self.depth_grid.grid_columnconfigure(1, weight=1)
        
        # Create Bids / Asks cells
        self.bid_rows: List[Tuple[ctk.CTkLabel, ctk.CTkLabel, ctk.CTkProgressBar]] = []
        self.ask_rows: List[Tuple[ctk.CTkLabel, ctk.CTkLabel, ctk.CTkProgressBar]] = []
        
        self._setup_depth_rows()

    def _setup_depth_rows(self):
        """Creates the progress bar and label structures for the order book."""
        # Headers
        lbl_b_hdr = ctk.CTkLabel(self.depth_grid, text="Bids (Buy)", text_color=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 10, "bold"))
        lbl_b_hdr.grid(row=0, column=0, sticky="w", pady=2)
        lbl_a_hdr = ctk.CTkLabel(self.depth_grid, text="Asks (Sell)", text_color=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 10, "bold"))
        lbl_a_hdr.grid(row=0, column=1, sticky="e", pady=2)
        
        for i in range(5):
            # Bid row setup
            bid_cell = ctk.CTkFrame(self.depth_grid, fg_color="transparent")
            bid_cell.grid(row=i+1, column=0, sticky="nsew", padx=(0, 5), pady=1)
            bid_cell.grid_columnconfigure(0, weight=1)
            
            bp_bar = ctk.CTkProgressBar(bid_cell, progress_color="#10b981", fg_color="#10b981", height=18)
            bp_bar.grid(row=0, column=0, columnspan=2, sticky="nsew")
            bp_bar.set(0.0)
            
            bl_price = ctk.CTkLabel(bid_cell, text="-", text_color=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 11), fg_color="transparent", bg_color="#10b981")
            bl_price.grid(row=0, column=0, sticky="w", padx=5)
            
            bl_qty = ctk.CTkLabel(bid_cell, text="-", text_color=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 11), fg_color="transparent", bg_color="#10b981")
            bl_qty.grid(row=0, column=1, sticky="e", padx=5)
            
            self.bid_rows.append((bl_price, bl_qty, bp_bar))
            
            # Ask row setup
            ask_cell = ctk.CTkFrame(self.depth_grid, fg_color="transparent")
            ask_cell.grid(row=i+1, column=1, sticky="nsew", padx=(5, 0), pady=1)
            ask_cell.grid_columnconfigure(0, weight=1)
            
            ap_bar = ctk.CTkProgressBar(ask_cell, progress_color="#f43f5e", fg_color="#f43f5e", height=18)
            ap_bar.grid(row=0, column=0, columnspan=2, sticky="nsew")
            ap_bar.set(0.0)
            
            al_price = ctk.CTkLabel(ask_cell, text="-", text_color=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 11), fg_color="transparent", bg_color="#f43f5e")
            al_price.grid(row=0, column=0, sticky="w", padx=5)
            
            al_qty = ctk.CTkLabel(ask_cell, text="-", text_color=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 11), fg_color="transparent", bg_color="#f43f5e")
            al_qty.grid(row=0, column=1, sticky="e", padx=5)
            
            self.ask_rows.append((al_price, al_qty, ap_bar))

    def update_depth(self, tick: TickData):
        """Draws the live order book depth, total quantities, and ratios."""
        tot_buy = tick.total_bid_qty
        tot_sell = tick.total_ask_qty
        
        # Format total labels
        self.lbl_total_buy.configure(text=f"Total Buy Qty: {self._format_qty(tot_buy)}")
        self.lbl_total_sell.configure(text=f"Total Sell Qty: {self._format_qty(tot_sell)}")
        
        # Redraw ratio canvas bar
        self.ratio_canvas.delete("all")
        rc_w = self.ratio_canvas.winfo_width()
        if rc_w < 10:
            rc_w = 200
            
        tot = tot_buy + tot_sell
        if tot > 0:
            buy_pct = tot_buy / tot
            split_x = int(rc_w * buy_pct)
            
            # Green (Buy side)
            self.ratio_canvas.create_rectangle(0, 0, split_x, 12, fill=config.COLOR_BULLISH, outline="")
            # Red (Sell side)
            self.ratio_canvas.create_rectangle(split_x, 0, rc_w, 12, fill=config.COLOR_BEARISH, outline="")
            # Label
            self.ratio_canvas.create_text(rc_w/2, 6, text=f"Buyers {buy_pct*100:.0f}% | Sellers {(1-buy_pct)*100:.0f}%", fill=config.TEXT_PRIMARY, font=(config.FONT_FAMILY, 9, "bold"))
        else:
            self.ratio_canvas.create_rectangle(0, 0, rc_w, 12, fill="#334155", outline="")
            
        # Extract top 5 bid/ask quantities for bar percentage scaling
        all_qtys = [b[1] for b in tick.bids] + [a[1] for a in tick.asks]
        max_qty = max(all_qtys) if all_qtys else 1
        
        # Populate Bids (Buy side)
        for i in range(5):
            lbl_price, lbl_qty, progress = self.bid_rows[i]
            if i < len(tick.bids):
                price, qty = tick.bids[i]
                lbl_price.configure(text=f"₹{price:.2f}")
                lbl_qty.configure(text=str(qty))
                # Custom color bar matching density
                pct = qty / max_qty
                progress.set(pct)
                progress.configure(progress_color=self._interpolate_color(config.COLOR_BULLISH, "#1a2236", pct))
            else:
                lbl_price.configure(text="-")
                lbl_qty.configure(text="-")
                progress.set(0.0)
                
        # Populate Asks (Sell side)
        for i in range(5):
            lbl_price, lbl_qty, progress = self.ask_rows[i]
            if i < len(tick.asks):
                price, qty = tick.asks[i]
                lbl_price.configure(text=f"₹{price:.2f}")
                lbl_qty.configure(text=str(qty))
                pct = qty / max_qty
                progress.set(pct)
                progress.configure(progress_color=self._interpolate_color(config.COLOR_BEARISH, "#1a2236", pct))
            else:
                lbl_price.configure(text="-")
                lbl_qty.configure(text="-")
                progress.set(0.0)

    def _interpolate_color(self, hex1: str, hex2: str, factor: float) -> str:
        """Interpolates between active color and base color to give opacity-like depth bars."""
        # Convert hex to RGB
        h1 = [int(hex1[i:i+2], 16) for i in (1, 3, 5)]
        h2 = [int(hex2[i:i+2], 16) for i in (1, 3, 5)]
        # Interpolate
        res = [int(h1[j] * factor + h2[j] * (1 - factor)) for j in range(3)]
        return f"#{res[0]:02x}{res[1]:02x}{res[2]:02x}"

    def _format_qty(self, qty: int) -> str:
        if qty >= 10000000:
            return f"{qty / 10000000:.2f} Cr"
        elif qty >= 100000:
            return f"{qty / 100000:.2f} Lakhs"
        return f"{qty:,}"


class MLDiagnosticsPanel(ctk.CTkFrame):
    """Displays the AI/ML diagnostics details, completed trade history, and real-time crossover signal logs."""
    def __init__(self, master, predictor: CrossoverPredictor, **kwargs):
        super().__init__(master, **kwargs)
        self.predictor = predictor
        self.configure(fg_color=config.BG_CARD)
        
        # Create Tab View inside Diagnostics Panel
        self.tabview = ctk.CTkTabview(self, fg_color="transparent")
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tabview.add("Diagnostics")
        self.tabview.add("Trade History")
        self.tabview.add("Signal Logs")
        
        # Setup Tabs
        self._setup_diagnostics_tab()
        self._setup_history_tab()
        self._setup_logs_tab()

    def _setup_diagnostics_tab(self):
        """Builds the success rate, feature weights, and side-by-side backtest table."""
        tab = self.tabview.tab("Diagnostics")
        
        # Success Rate Row
        acc_frame = ctk.CTkFrame(tab, fg_color="transparent")
        acc_frame.pack(fill="x", padx=5, pady=2)
        
        lbl_acc_text = ctk.CTkLabel(acc_frame, text="Model Active Success Rate:", font=(config.FONT_FAMILY, 11))
        lbl_acc_text.pack(side="left")
        
        self.lbl_acc_val = ctk.CTkLabel(acc_frame, text="-%", text_color=config.COLOR_BULLISH, font=(config.FONT_FAMILY, 11, "bold"))
        self.lbl_acc_val.pack(side="right")
        
        # Backtest Grid Header
        lbl_bt_hdr = ctk.CTkLabel(tab, text="Portfolio Backtest Comparison:", text_color=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 10, "bold"))
        lbl_bt_hdr.pack(anchor="w", padx=5, pady=(5, 1))
        
        self._setup_backtest_table(tab)
        
        # Feature Importance section
        lbl_feat_hdr = ctk.CTkLabel(tab, text="RF Feature Importances:", text_color=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 10, "bold"))
        lbl_feat_hdr.pack(anchor="w", padx=5, pady=(5, 1))
        
        self.feat_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.feat_frame.pack(fill="x", padx=5, pady=1)
        
        self.feat_bars: Dict[str, Tuple[ctk.CTkLabel, ctk.CTkProgressBar]] = {}
        self._setup_feature_bars()
        
        # Active positions list at bottom of diagnostics tab
        lbl_trades_hdr = ctk.CTkLabel(tab, text="Active Positions Monitor:", text_color=config.TEXT_SECONDARY, font=(config.FONT_FAMILY, 10, "bold"))
        lbl_trades_hdr.pack(anchor="w", padx=5, pady=(5, 1))
        
        self.trades_scroll = ctk.CTkScrollableFrame(tab, height=65, fg_color="#101524")
        self.trades_scroll.pack(fill="x", padx=5, pady=1)
        
        self.trade_widgets: Dict[str, ctk.CTkFrame] = {}

    def _setup_backtest_table(self, tab):
        """Creates the grid for the Unfiltered vs ML Filtered comparison."""
        table_frame = ctk.CTkFrame(tab, fg_color="#101524")
        table_frame.pack(fill="x", padx=5, pady=2)
        
        # Headers
        headers = ["Metric", "Unfiltered", "ML Filtered"]
        for c_idx, h in enumerate(headers):
            table_frame.grid_columnconfigure(c_idx, weight=1)
            lbl = ctk.CTkLabel(table_frame, text=h, font=(config.FONT_FAMILY, 9, "bold"), text_color=config.TEXT_SECONDARY)
            lbl.grid(row=0, column=c_idx, padx=5, pady=1, sticky="ew")
            
        self.backtest_labels = {}
        metrics = [
            ("trades", "Total Trades"),
            ("win_rate", "Win Rate %"),
            ("net_pnl", "Net P&L (pts)"),
            ("max_dd", "Max Drawdown")
        ]
        
        for r_idx, (key, name) in enumerate(metrics, start=1):
            lbl_name = ctk.CTkLabel(table_frame, text=name, font=(config.FONT_FAMILY, 9), anchor="w")
            lbl_name.grid(row=r_idx, column=0, padx=5, pady=1, sticky="w")
            
            lbl_unf = ctk.CTkLabel(table_frame, text="-", font=(config.FONT_FAMILY, 9, "bold"), text_color=config.TEXT_PRIMARY)
            lbl_unf.grid(row=r_idx, column=1, padx=5, pady=1)
            
            lbl_filt = ctk.CTkLabel(table_frame, text="-", font=(config.FONT_FAMILY, 9, "bold"), text_color=config.COLOR_BULLISH)
            lbl_filt.grid(row=r_idx, column=2, padx=5, pady=1)
            
            self.backtest_labels[key] = (lbl_unf, lbl_filt)

    def _setup_feature_bars(self):
        """Creates horizontal progress bars for each model feature weight."""
        features = {
            "A2": "A2 (2M)",
            "A5": "A5 (5M)",
            "ratio": "A2/A5 Ratio",
            "side_share": "Side Share",
            "max_ltq": "Max LTQ",
            "slope_gap": "Slope Gap",
            "time_of_day": "Time of Day"
        }
        
        for col, display_name in features.items():
            row = ctk.CTkFrame(self.feat_frame, fg_color="transparent")
            row.pack(fill="x", pady=0)
            
            lbl_name = ctk.CTkLabel(row, text=display_name, width=80, font=(config.FONT_FAMILY, 9), anchor="w")
            lbl_name.pack(side="left")
            
            bar = ctk.CTkProgressBar(row, progress_color=config.COLOR_ACCENT, fg_color="#1e293b", height=6)
            bar.pack(side="left", fill="x", expand=True, padx=5)
            bar.set(0.0)
            
            lbl_val = ctk.CTkLabel(row, text="-%", width=35, font=(config.FONT_FAMILY, 9), anchor="e")
            lbl_val.pack(side="right")
            
            self.feat_bars[col] = (lbl_val, bar)

    def _setup_history_tab(self):
        """Builds Completed Trades history list table (Buy/Sell logic, entries, exits, P&L)."""
        tab = self.tabview.tab("Trade History")
        
        # Table column headers
        hdrs_frame = ctk.CTkFrame(tab, fg_color="#101524", height=24)
        hdrs_frame.pack(fill="x", padx=2, pady=2)
        
        cols = [("Sym", 1), ("Type", 1), ("Entry", 1), ("Exit", 1), ("P&L", 1), ("System", 2)]
        for col_idx, (text, weight) in enumerate(cols):
            hdrs_frame.grid_columnconfigure(col_idx, weight=weight)
            lbl = ctk.CTkLabel(hdrs_frame, text=text, font=(config.FONT_FAMILY, 9, "bold"), text_color=config.TEXT_SECONDARY)
            lbl.grid(row=0, column=col_idx, padx=2, pady=2, sticky="ew")
            
        # Scrollable container for closed trade rows
        self.history_scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.history_scroll.pack(fill="both", expand=True, padx=2, pady=1)
        self.history_rows: List[ctk.CTkFrame] = []

    def _setup_logs_tab(self):
        """Builds scrollable textbox log for crossover signal details."""
        tab = self.tabview.tab("Signal Logs")
        
        self.log_text = ctk.CTkTextbox(tab, font=(config.FONT_FAMILY, 9), fg_color="#101524", border_width=1, border_color="#1e293b")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text.insert("end", "=== Waiting for Crossover Signals ===\n")
        self.log_text.configure(state="disabled")

    def refresh_diagnostics(self):
        """Updates accuracy rates, feature importances, backtest table, and active trade list."""
        # 1. Update Backtest Table
        report = self.predictor.get_backtest_report()
        unf = report["unfiltered"]
        filt = report["filtered"]
        
        self.backtest_labels["trades"][0].configure(text=str(unf["trades"]))
        self.backtest_labels["trades"][1].configure(text=str(filt["trades"]))
        
        self.backtest_labels["win_rate"][0].configure(text=f"{unf['win_rate']:.1f}%")
        self.backtest_labels["win_rate"][1].configure(text=f"{filt['win_rate']:.1f}%")
        
        unf_pnl = unf["net_pnl"]
        filt_pnl = filt["net_pnl"]
        
        self.backtest_labels["net_pnl"][0].configure(
            text=f"₹{unf_pnl:.2f}" if abs(unf_pnl) > 0 else "-",
            text_color=config.COLOR_BULLISH if unf_pnl >= 0 else config.COLOR_BEARISH
        )
        self.backtest_labels["net_pnl"][1].configure(
            text=f"₹{filt_pnl:.2f}" if abs(filt_pnl) > 0 else "-",
            text_color=config.COLOR_BULLISH if filt_pnl >= 0 else config.COLOR_BEARISH
        )
        
        self.backtest_labels["max_dd"][0].configure(text=f"₹{unf['max_dd']:.2f}" if unf["max_dd"] > 0 else "-", text_color=config.COLOR_BEARISH)
        self.backtest_labels["max_dd"][1].configure(text=f"₹{filt['max_dd']:.2f}" if filt["max_dd"] > 0 else "-", text_color=config.COLOR_BEARISH)
        
        # 2. Update Feature Importances
        importances = self.predictor.get_feature_importances()
        for col, val in importances.items():
            if col in self.feat_bars:
                lbl, bar = self.feat_bars[col]
                lbl.configure(text=f"{val:.1f}%")
                bar.set(val / 100.0)
                
        # 3. Update Success Rate
        self.lbl_acc_val.configure(text=f"{filt['win_rate']:.1f}%" if filt['trades'] > 0 else "-%")
        
        # 4. Update Active Trades
        active_ids = set()
        active_list = []
        
        with self.predictor.lock:
            for symbol, t in self.predictor.active_unfiltered.items():
                active_list.append((t, "UNFILTERED"))
            for symbol, t in self.predictor.active_filtered.items():
                active_list.append((t, "ML FILTERED"))
                
        for trade, port_name in active_list:
            active_ids.add(f"{trade.symbol}_{trade.signal_type}_{trade.entry_time}_{port_name}")
            
        for key in list(self.trade_widgets.keys()):
            if key not in active_ids:
                self.trade_widgets[key].destroy()
                del self.trade_widgets[key]
                
        for trade, port_name in active_list:
            key = f"{trade.symbol}_{trade.signal_type}_{trade.entry_time}_{port_name}"
            floating_pnl = (trade.current_price - trade.entry_price) if trade.signal_type == "BUY" else (trade.entry_price - trade.current_price)
            floating_pnl_pct = (floating_pnl / trade.entry_price) * 100 if trade.entry_price > 0 else 0.0
            
            pnl_color = config.COLOR_BULLISH if floating_pnl >= 0 else config.COLOR_BEARISH
            pnl_sign = "+" if floating_pnl >= 0 else ""
            
            symbol_short = trade.symbol.replace("NSE:", "").replace("-EQ", "")
            trade_text = f"[{port_name[:4]}] {trade.signal_type} {symbol_short} @ ₹{trade.entry_price:.2f}"
            pnl_text = f"{pnl_sign}₹{floating_pnl:.2f} ({pnl_sign}{floating_pnl_pct:.2f}%)"
            
            if key not in self.trade_widgets:
                row_frame = ctk.CTkFrame(self.trades_scroll, fg_color="#1a2236", height=20)
                row_frame.pack(fill="x", pady=1)
                
                lbl_info = ctk.CTkLabel(row_frame, text=trade_text, font=(config.FONT_FAMILY, 9), anchor="w")
                lbl_info.pack(side="left", padx=5)
                
                lbl_pnl = ctk.CTkLabel(row_frame, text=pnl_text, text_color=pnl_color, font=(config.FONT_FAMILY, 9, "bold"), anchor="e")
                lbl_pnl.pack(side="right", padx=5)
                
                self.trade_widgets[key] = row_frame
            else:
                frame = self.trade_widgets[key]
                children = frame.winfo_children()
                if len(children) >= 2:
                    children[0].configure(text=trade_text)
                    children[1].configure(text=pnl_text, text_color=pnl_color)
                    
        # Update Completed Trades tab
        self.update_completed_trades()

    def log_crossover(self, symbol: str, signal_type: str, price: float, features: Dict[str, float], decision: str, prob: float, reason: str):
        """Appends a detailed log line to the scrolling text log tab showing FYI and ML tags."""
        self.log_text.configure(state="normal")
        
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        symbol_short = symbol.replace("NSE:", "").replace("-EQ", "")
        
        # FYI check: Average LTQ over last 2 minutes > Average LTQ over last 5 minutes
        fyi_burst_active = "ACTIVE" if features['A2'] > features['A5'] else "INACTIVE"
        
        log_line = f"[{time_str}] {symbol_short}: {signal_type} Crossover @ ₹{price:.2f}\n"
        log_line += f"  - FYI Burst Test: A2 ({self._format_qty(int(features['A2']))}) > A5 ({self._format_qty(int(features['A5']))}) -> {fyi_burst_active} ({features['ratio']:.2f}x Ratio)\n"
        log_line += f"  - Side classified LTQ Share: {features['side_share']*100:.1f}%\n"
        log_line += f"  - ML Decision: {decision} (Probability: {prob:.1f}%)\n"
        log_line += f"  - Reason: {reason}\n\n"
        
        self.log_text.insert("end", log_line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def update_completed_trades(self):
        """Rebuilds Completed Trades rows in the Trade History tab from actual trade logs."""
        # Clear existing rows
        for row in self.history_rows:
            row.destroy()
        self.history_rows.clear()
        
        # Merge Unfiltered and Filtered trades
        all_completed = []
        with self.predictor.lock:
            for t in self.predictor.unfiltered_trades:
                all_completed.append((t, "UNFILTERED"))
            for t in self.predictor.filtered_trades:
                all_completed.append((t, "ML FILTERED"))
                
        # Sort by exit time descending (newest first)
        all_completed.sort(key=lambda x: x[0].exit_time, reverse=True)
        
        # Display rows
        for idx, (trade, portfolio) in enumerate(all_completed):
            bg_color = "#151b2d" if idx % 2 == 0 else "#101422"
            row_frame = ctk.CTkFrame(self.history_scroll, fg_color=bg_color, height=22, corner_radius=2)
            row_frame.pack(fill="x", pady=1, padx=2)
            self.history_rows.append(row_frame)
            
            # Setup columns
            row_frame.grid_columnconfigure(0, weight=1) # Sym
            row_frame.grid_columnconfigure(1, weight=1) # Type
            row_frame.grid_columnconfigure(2, weight=1) # Entry
            row_frame.grid_columnconfigure(3, weight=1) # Exit
            row_frame.grid_columnconfigure(4, weight=1) # P&L
            row_frame.grid_columnconfigure(5, weight=2) # System
            
            pnl_color = config.COLOR_BULLISH if trade.pnl >= 0 else config.COLOR_BEARISH
            pnl_sign = "+" if trade.pnl >= 0 else ""
            
            sym = trade.symbol.replace("NSE:", "").replace("-EQ", "")
            
            ctk.CTkLabel(row_frame, text=sym, font=(config.FONT_FAMILY, 9, "bold")).grid(row=0, column=0, sticky="ew")
            
            type_color = config.COLOR_BULLISH if trade.signal_type == "BUY" else config.COLOR_BEARISH
            ctk.CTkLabel(row_frame, text=trade.signal_type, text_color=type_color, font=(config.FONT_FAMILY, 9, "bold")).grid(row=0, column=1, sticky="ew")
            
            ctk.CTkLabel(row_frame, text=f"₹{trade.entry_price:.2f}", font=(config.FONT_FAMILY, 9)).grid(row=0, column=2, sticky="ew")
            ctk.CTkLabel(row_frame, text=f"₹{trade.exit_price:.2f}", font=(config.FONT_FAMILY, 9)).grid(row=0, column=3, sticky="ew")
            
            ctk.CTkLabel(row_frame, text=f"{pnl_sign}₹{trade.pnl:.2f}", text_color=pnl_color, font=(config.FONT_FAMILY, 9, "bold")).grid(row=0, column=4, sticky="ew")
            
            sys_color = config.COLOR_ACCENT if portfolio == "ML FILTERED" else config.TEXT_SECONDARY
            ctk.CTkLabel(row_frame, text=portfolio, text_color=sys_color, font=(config.FONT_FAMILY, 9, "bold")).grid(row=0, column=5, sticky="ew")

    def _format_qty(self, qty: int) -> str:
        if qty >= 10000000:
            return f"{qty / 10000000:.2f} Cr"
        elif qty >= 100000:
            return f"{qty / 100000:.2f} L"
        elif qty >= 1000:
            return f"{qty / 1000:.1f} K"
        return str(qty)
