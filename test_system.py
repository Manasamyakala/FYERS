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
        """Verifies that the TradeRecord handles continuous crossover exits and P&L calculations correctly."""
        features = {"A2": 150000, "A5": 100000, "ratio": 1.5, "side_share": 0.65, "max_ltq": 200000, "slope_gap": 0.25, "time_of_day": 30.0}
        
        # Test BUY (Long) trade P&L: Exit Sell LTP - Entry Buy LTP
        trade_buy = TradeRecord("NSE:SUZLON-EQ", "BUY", 100.0, time.time(), features)
        self.assertEqual(trade_buy.status, "PENDING")
        
        # Crossover exits trade at higher price (Sell LTP = 104.50)
        trade_buy.exit_trade(104.5, time.time())
        self.assertEqual(trade_buy.status, "WIN")
        self.assertEqual(trade_buy.pnl, 4.5)
        
        # Test SELL (Short) trade P&L: Entry Sell LTP - Exit Buy LTP
        trade_sell = TradeRecord("NSE:SUZLON-EQ", "SELL", 100.0, time.time(), features)
        # Crossover exits trade at lower price (Buy LTP = 96.00)
        trade_sell.exit_trade(96.0, time.time())
        self.assertEqual(trade_sell.status, "WIN")
        self.assertEqual(trade_sell.pnl, 4.0)

    def test_ml_predictor_synthetic(self):
        """Verifies that CrossoverPredictor pre-trains and runs predictions with the updated 7 features."""
        predictor = CrossoverPredictor()
        
        # Verify columns are set to the new 7 features
        self.assertEqual(len(predictor.features_columns), 7)
        self.assertIn("A2", predictor.features_columns)
        self.assertIn("side_share", predictor.features_columns)
        
        # Test sample prediction
        sample_features = {
            "A2": 160000.0,
            "A5": 100000.0,
            "ratio": 1.6,
            "side_share": 0.65,
            "max_ltq": 250000.0,
            "slope_gap": 0.25,
            "time_of_day": 45.0
        }
        
        decision, prob, reason = predictor.predict(sample_features, "BUY")
        self.assertIn(decision, ["ACCEPT", "AVOID"])
        self.assertTrue(0.0 <= prob <= 100.0)
        self.assertIsInstance(reason, str)

    def test_trade_side_tagging(self):
        """Verifies that ticks are tagged correctly as buyer-initiated or seller-initiated."""
        from simulator import SimulatorFeed
        feed = SimulatorFeed()
        symbol = "NSE:YESBANK-EQ"
        
        # Setup history
        now = time.time()
        
        # 1. Test Trade price >= Ask -> Buyer Initiated (side = 1)
        feed.record_tick(symbol, now - 10, 100.0, 1000, 99.9, 100.0)
        self.assertEqual(feed.tick_history[symbol][-1][5], 1)
        
        # 2. Test Trade price <= Bid -> Seller Initiated (side = -1)
        feed.record_tick(symbol, now - 8, 99.8, 1000, 99.8, 99.9)
        self.assertEqual(feed.tick_history[symbol][-1][5], -1)
        
        # 3. Test Tick Rule fallback: Price went up -> Buyer Initiated (1)
        feed.record_tick(symbol, now - 6, 99.9, 1000, 0.0, 0.0) # no quotes
        self.assertEqual(feed.tick_history[symbol][-1][5], 1)
        
        # 4. Test Tick Rule fallback: Price went down -> Seller Initiated (-1)
        feed.record_tick(symbol, now - 4, 99.7, 1000, 0.0, 0.0) # no quotes
        self.assertEqual(feed.tick_history[symbol][-1][5], -1)

if __name__ == "__main__":
    unittest.main()
