# config.py
# Configuration settings for the Stock Screening and Analysis System

# Screening & Liquidity Parameters
PRICE_MIN = 30.0
PRICE_MAX = 500.0
LIQUIDITY_MIN_QTY = 1000000.0  # 10,00,000 (10 Lakhs)

# Technical Indicator Parameters
SMMA_FAST_PERIOD = 20
SMMA_SLOW_PERIOD = 120
TIMEFRAME = "1"  # 1-minute timeframe

# AI/ML Trade Acceptance Threshold
ML_CONFIDENCE_THRESHOLD = 55.0  # Accept trade if P(win) >= 55%

# Curated pool of NSE stocks for scanning
# Includes stocks within the price range, outside the price range,
# highly liquid stocks, and lower liquidity stocks.
DEFAULT_SYMBOLS = [
    # ---- Screen Targets (Price ₹30 - ₹500 & High Liquidity) ----
    "NSE:SUZLON-EQ",     # Price ~ 50-70, Vol/Depth very high
    "NSE:YESBANK-EQ",    # Price ~ 20-30, Vol/Depth very high (right on border)
    "NSE:GMRINFRA-EQ",   # Price ~ 80-95, Vol/Depth very high
    "NSE:PNB-EQ",        # Price ~ 110-120, Vol/Depth high
    "NSE:TATASTEEL-EQ",  # Price ~ 140-160, Vol/Depth high
    "NSE:ZOMATO-EQ",     # Price ~ 200-250, Vol/Depth high
    "NSE:NHPC-EQ",       # Price ~ 80-100, Vol/Depth high
    "NSE:IRFC-EQ",       # Price ~ 150-180, Vol/Depth high
    "NSE:SJVN-EQ",       # Price ~ 120-140, Vol/Depth high
    "NSE:HUDCO-EQ",      # Price ~ 200-280, Vol/Depth high
    "NSE:IOC-EQ",        # Price ~ 160-180, Vol/Depth high
    "NSE:IREDA-EQ",      # Price ~ 200-260, Vol/Depth high
    "NSE:SAIL-EQ",       # Price ~ 120-140, Vol/Depth high
    "NSE:IDFCFIRSTB-EQ", # Price ~ 70-85, Vol/Depth high
    "NSE:GIPCL-EQ",      # Price ~ 200-230, Vol/Depth medium
    "NSE:SOUTHBANK-EQ",  # Price ~ 25-35, Vol/Depth high
    
    # ---- Outside Price Filter (Too low or too high) ----
    "NSE:IDEA-EQ",       # Price ~ 10-15 (Too cheap)
    "NSE:SBIN-EQ",       # Price ~ 700-850 (Too expensive)
    "NSE:INFY-EQ",       # Price ~ 1700-1900 (Too expensive)
    "NSE:TCS-EQ",        # Price ~ 3800-4200 (Too expensive)
    "NSE:RELIANCE-EQ",   # Price ~ 2400-2600 (Too expensive)
    "NSE:HDFCBANK-EQ",   # Price ~ 1500-1700 (Too expensive)
    "NSE:ITC-EQ",        # Price ~ 450-510 (Close to upper limit, often outside)
    "NSE:JPASSOCIAT-EQ", # Price ~ 8-12 (Too cheap)
    
    # ---- Inside Price, but fails Liquidity (< 10L bid/ask qty) ----
    "NSE:MRF-EQ",        # Price ~ 1,30,000 (Too expensive)
    "NSE:TITAN-EQ",      # Price ~ 3200-3400 (Too expensive)
    "NSE:TATAELXSI-EQ",  # Price ~ 7000-8000 (Too expensive)
    "NSE:PAGEIND-EQ",    # Price ~ 40,000 (Too expensive)
    "NSE:LTIM-EQ",       # Price ~ 5000-5500 (Too expensive)
]

# UI Styling Constants (Modern Slate Dark Theme)
BG_MAIN = "#0b0f19"          # Deep dark slate
BG_CARD = "#151b2d"          # Lighter card background
BG_CONTROL = "#1e293b"       # Control panel background
TEXT_PRIMARY = "#f8fafc"     # Near white
TEXT_SECONDARY = "#94a3b8"   # Slate gray
TEXT_MUTED = "#64748b"       # Muted gray
COLOR_ACCENT = "#38bdf8"     # Sky blue
COLOR_BORDER = "#1e293b"     # Subtle border

COLOR_BULLISH = "#10b981"    # Neon Emerald green
COLOR_BEARISH = "#f43f5e"    # Neon Rose red
COLOR_WARNING = "#f59e0b"    # Amber yellow

FONT_FAMILY = "Segoe UI"     # Standard Windows font

# Base reference prices for simulation & demo mode
REF_PRICES = {
    "NSE:SUZLON-EQ": 64.50,
    "NSE:YESBANK-EQ": 26.20,
    "NSE:GMRINFRA-EQ": 87.80,
    "NSE:PNB-EQ": 115.40,
    "NSE:TATASTEEL-EQ": 152.30,
    "NSE:ZOMATO-EQ": 224.50,
    "NSE:NHPC-EQ": 92.15,
    "NSE:IRFC-EQ": 172.90,
    "NSE:SJVN-EQ": 134.20,
    "NSE:HUDCO-EQ": 272.50,
    "NSE:IOC-EQ": 168.40,
    "NSE:IREDA-EQ": 242.10,
    "NSE:SAIL-EQ": 131.60,
    "NSE:IDFCFIRSTB-EQ": 75.80,
    "NSE:GIPCL-EQ": 218.00,
    "NSE:SOUTHBANK-EQ": 29.40,
    "NSE:IDEA-EQ": 12.30,
    "NSE:SBIN-EQ": 820.50,
    "NSE:INFY-EQ": 1845.00,
    "NSE:TCS-EQ": 3950.00,
    "NSE:RELIANCE-EQ": 2510.00,
    "NSE:HDFCBANK-EQ": 1625.00,
    "NSE:ITC-EQ": 492.00,
    "NSE:JPASSOCIAT-EQ": 9.80,
    "NSE:MRF-EQ": 132000.00,
    "NSE:TITAN-EQ": 3340.00,
    "NSE:TATAELXSI-EQ": 7450.00,
    "NSE:PAGEIND-EQ": 41200.00,
    "NSE:LTIM-EQ": 5280.00,
}

