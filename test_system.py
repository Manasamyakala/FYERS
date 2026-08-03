# test_system.py
# Unit tests to verify indicators, calculations, crossovers, and ML logic.

import unittest
import numpy as np
import pandas as pd
import time

import config
import indicators
from simulator import Candle, TickData
from ml_model import CrossoverPredictor, TradeRecord

class TestStockSystem(unittest.TestCase):
    def test_smma_calculation(self):
        """Verifies that the SMMA calculation matches MT4/MT5 Smoothed Moving Average definition."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        period = 5
        
        # Calculate SMMA
        smma_arr = indicators.calculate_smma(prices, period)
        
        # First period entries should be NaN
        for k in range(period - 1):
            self.assertTrue(np.isnan(smma_arr[k]))
            
        # First calculated entry is simple moving average (SMA) of first 'period' elements
        # (10 + 11 + 12 + 13 + 14) / 5 = 12.0
        self.assertEqual(smma_arr[period - 1], 12.0)
        
        # Second entry should be: (prev_smma * (period - 1) + current_price) / period
        # (12.0 * 4 + 15.0) / 5 = (48 + 15) / 5 = 12.6
        self.assertAlmostEqual(smma_arr[period], 12.6)
        
        # Third entry should be: (12.6 * 4 + 16.0) / 5 = (50.4 + 16) / 5 = 13.28
        self.assertAlmostEqual(smma_arr[period + 1], 13.28)

    def test_smma_incremental_step(self):
        """Verifies that real-time incremental step updates match batch calculation output."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        period = 5
        
        # Batch SMMA
        smma_batch = indicators.calculate_smma(prices, period)
        
        # Incremental SMMA
        prev_smma = smma_batch[period - 1] # SMMA for first 5 prices is 12.0
        next_price = prices[5]             # 15.0
        
        inc_smma = indicators.calculate_smma_step(prev_smma, next_price, period, len(prices))
        self.assertAlmostEqual(inc_smma, smma_batch[5])

    def test_trade_tracking_profitability(self):
        """Verifies that the TradeRecord resolves WIN/LOSS states correctly based on targets."""
        features = {"trend_slope": 0.05, "smma_diff": 0.1, "volatility": 0.15, "volume_ratio": 1.2, "bid_ask_imbalance": 0.2, "spread_pct": 0.05}
        
        # Test BUY Win
        trade_buy = TradeRecord("NSE:SUZLON-EQ", "BUY", 100.0, time.time(), features)
        self.assertEqual(trade_buy.status, "PENDING")
        
        # Price goes up by 0.5% (Take Profit target is 0.5%)
        trade_buy.update(100.5, time.time())
        self.assertEqual(trade_buy.status, "WIN")
        
        # Test BUY Loss (Stop Loss target is 0.3%)
        trade_buy_loss = TradeRecord("NSE:SUZLON-EQ", "BUY", 100.0, time.time(), features)
        trade_buy_loss.update(99.69, time.time())
        self.assertEqual(trade_buy_loss.status, "LOSS")
        
        # Test SELL Win
        trade_sell = TradeRecord("NSE:SUZLON-EQ", "SELL", 100.0, time.time(), features)
        trade_sell.update(99.5, time.time())
        self.assertEqual(trade_sell.status, "WIN")

    def test_ml_predictor_synthetic(self):
        """Verifies that CrossoverPredictor pre-trains and runs predictions with correct formats."""
        predictor = CrossoverPredictor()
        
        # Verify columns are set
        self.assertEqual(len(predictor.features_columns), 6)
        
        # Test sample prediction
        sample_features = {
            "trend_slope": 0.02,
            "smma_diff": 0.05,
            "volatility": 0.12,
            "volume_ratio": 1.5,
            "bid_ask_imbalance": 0.3,
            "spread_pct": 0.03
        }
        
        decision, prob, reason = predictor.predict(sample_features, "BUY")
        self.assertIn(decision, ["ACCEPT", "AVOID"])
        self.assertTrue(0.0 <= prob <= 100.0)
        self.assertIsInstance(reason, str)

if __name__ == "__main__":
    unittest.main()
