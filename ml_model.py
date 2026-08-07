# ml_model.py
# Implements the AI/ML Crossover Analysis engine.
# Uses a Scikit-Learn Random Forest Classifier to predict trade outcomes,
# supports online learning, and provides logical trade explanations.

import time
import threading
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config

class TradeRecord:
    """Represents a mock or live trade triggered by a crossover, closed ONLY by the opposite crossover."""
    def __init__(self, symbol: str, signal_type: str, entry_price: float, timestamp: float, features: Dict[str, float]):
        self.symbol = symbol
        self.signal_type = signal_type  # "BUY" (Long) or "SELL" (Short)
        self.entry_price = entry_price
        self.entry_time = timestamp
        self.features = features
        self.current_price = entry_price
        
        self.status = "PENDING"  # "PENDING", "WIN", "LOSS"
        self.exit_price = 0.0
        self.exit_time = 0.0
        self.pnl = 0.0
        self.pnl_pct = 0.0

    def exit_trade(self, exit_price: float, exit_time: float):
        """Closes the trade at the opposite crossover and calculates final P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        if self.signal_type == "BUY":
            # Entry = Buy LTP, Exit = Sell LTP
            self.pnl = exit_price - self.entry_price
        else:  # SELL
            # Entry = Sell LTP, Exit = Buy LTP
            self.pnl = self.entry_price - exit_price
            
        self.pnl_pct = (self.pnl / self.entry_price) * 100 if self.entry_price > 0 else 0.0
        self.status = "WIN" if self.pnl > 0.0 else "LOSS"

    def update(self, current_price: float, now: float) -> bool:
        """Updates the trade with the latest price."""
        self.current_price = current_price
        return self.status != "PENDING"


class CrossoverPredictor:
    """Extracts features, runs Random Forest predictions, and updates model on trade outcomes."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.features_columns = [
            "A2",
            "A5",
            "ratio",
            "side_share",
            "max_ltq",
            "slope_gap",
            "time_of_day"
        ]
        
        # Training dataset accumulator
        self.X_train: List[List[float]] = []
        self.y_train: List[int] = []  # 1 for Profit (Win), 0 for Loss
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Double Portfolio Databases: Unfiltered vs ML Filtered
        self.unfiltered_trades: List[TradeRecord] = []
        self.filtered_trades: List[TradeRecord] = []
        
        self.active_unfiltered: Dict[str, TradeRecord] = {}
        self.active_filtered: Dict[str, TradeRecord] = {}
        
        # Generate synthetic data and fit initial model so it functions immediately
        self._initialize_synthetic_model()

    def _initialize_synthetic_model(self):
        """Generates synthetic dataset and pre-trains Random Forest to model real trading dynamics."""
        print("Pre-training AI/ML model with synthetic market patterns...")
        np.random.seed(42)
        n_samples = 1500
        
        # Synthesize typical features
        a5 = np.random.uniform(10000, 500000, n_samples)
        ratio = np.random.uniform(0.4, 3.0, n_samples)
        a2 = a5 * ratio
        side_share = np.random.uniform(0.1, 0.9, n_samples)
        max_ltq = a5 * np.random.uniform(1.5, 4.0, n_samples)
        slope_gap = np.random.uniform(0.01, 1.5, n_samples)
        time_of_day = np.random.uniform(0, 375, n_samples)
        
        y = np.zeros(n_samples, dtype=int)
        
        # Pre-train buy/sell rules
        for i in range(n_samples):
            is_buy = i < (n_samples / 2)
            score = 0.0
            
            # Burst rule: A2 > A5 (ratio > 1.2 or so)
            if ratio[i] > 1.2:
                score += 2.0
            elif ratio[i] < 0.8:
                score -= 1.5
                
            # Crossover angle strength
            if slope_gap[i] > 0.15:
                score += 1.5
                
            if is_buy:
                # Buy side share: buy-side share should be high (> 0.55)
                if side_share[i] > 0.55:
                    score += 2.5
                elif side_share[i] < 0.45:
                    score -= 2.0
            else:
                # Sell side share: buy-side share should be low (< 0.45, meaning sell-side share > 0.55)
                if side_share[i] < 0.45:
                    score += 2.5
                elif side_share[i] > 0.55:
                    score -= 2.0
                    
            prob = 1.0 / (1.0 + np.exp(-(score - 2.0)))
            y[i] = 1 if np.random.rand() < prob else 0
            
        # Format into Pandas DataFrame
        df_x = pd.DataFrame({
            "A2": a2,
            "A5": a5,
            "ratio": ratio,
            "side_share": side_share,
            "max_ltq": max_ltq,
            "slope_gap": slope_gap,
            "time_of_day": time_of_day
        })
        
        self.model.fit(df_x, y)
        self.X_train = df_x.values.tolist()
        self.y_train = y.tolist()

    def extract_features(self, symbol: str, signal_type: str, candles: List[Any], tick: Any, feed: Any) -> Optional[Dict[str, float]]:
        """Extracts numerical features from the current market state for the ML model."""
        if len(candles) < 20:
            return None
            
        closes = [c.close for c in candles]
        
        # 1. Slope Gap: Distance and angle of crossover
        # Calculate fast and slow averages
        import indicators
        smma20_history = indicators.calculate_smma(closes, config.SMMA_FAST_PERIOD)
        smma120_history = indicators.calculate_smma(closes, config.SMMA_SLOW_PERIOD)
        
        if len(smma20_history) < 2 or len(smma120_history) < 2 or np.isnan(smma20_history[-2]) or np.isnan(smma120_history[-2]):
            slope_gap = 0.05
        else:
            diff_now = smma20_history[-1] - smma120_history[-1]
            diff_prev = smma20_history[-2] - smma120_history[-2]
            # Speed/slope gap of crossing
            slope_gap = abs(diff_now - diff_prev) / tick.ltp * 100 if tick.ltp > 0 else 0.05
            
        # 2. LTQ metrics
        a2, a5, ratio, side_share, max_ltq = feed.get_ltq_metrics(symbol)
        
        # 3. Time of day (minutes since open, assume 9:15 AM = 555 mins)
        from datetime import datetime
        now_dt = datetime.now()
        current_mins = now_dt.hour * 60 + now_dt.minute
        time_of_day = max(0, current_mins - 555)
        
        return {
            "A2": float(a2),
            "A5": float(a5),
            "ratio": float(ratio),
            "side_share": float(side_share),
            "max_ltq": float(max_ltq),
            "slope_gap": float(slope_gap),
            "time_of_day": float(time_of_day)
        }

    def predict(self, features: Dict[str, float], signal_type: str) -> Tuple[str, float, str]:
        """
        Predicts if a crossover signal should be accepted or avoided.
        Returns: Tuple (Decision: "ACCEPT"/"AVOID", Success Probability: 0.0-100.0, Reason string)
        """
        x_df = pd.DataFrame([features], columns=self.features_columns)
        
        with self.lock:
            prob_profitable = self.model.predict_proba(x_df)[0][1]
            
        prob_pct = float(prob_profitable * 100)
        
        # Accept if confidence is >= ML_CONFIDENCE_THRESHOLD
        decision = "ACCEPT" if prob_pct >= config.ML_CONFIDENCE_THRESHOLD else "AVOID"
        reason = self._generate_explanation(features, signal_type, decision, prob_pct)
        
        return decision, prob_pct, reason

    def _generate_explanation(self, features: Dict[str, float], signal_type: str, decision: str, prob: float) -> str:
        """Translates features into human-readable explanation."""
        reasons = []
        ratio = features["ratio"]
        side_share = features["side_share"]
        slope_gap = features["slope_gap"]
        
        if decision == "AVOID":
            if ratio < 1.0:
                reasons.append(f"No volume burst (Ratio: {ratio:.2f}x < 1.0)")
            if signal_type == "BUY" and side_share < 0.50:
                reasons.append(f"Burst not buyer-side (Buy share: {side_share*100:.1f}% < 50%)")
            elif signal_type == "SELL" and side_share > 0.50:
                reasons.append(f"Burst not seller-side (Sell share: {(1-side_share)*100:.1f}% < 50%)")
            if slope_gap < 0.10:
                reasons.append(f"Weak/shallow crossover (Slope Gap: {slope_gap:.3f})")
                
            if not reasons:
                reasons.append(f"Trade probability ({prob:.1f}%) is below acceptance threshold")
            return "Avoided: " + " | ".join(reasons)
        else:
            reasons.append("Supported by: ")
            if ratio > 1.2:
                reasons.append(f"strong volume burst ({ratio:.2f}x)")
            if signal_type == "BUY" and side_share > 0.55:
                reasons.append(f"buyer dominance ({side_share*100:.1f}%)")
            elif signal_type == "SELL" and side_share < 0.45:
                reasons.append(f"seller dominance ({(1-side_share)*100:.1f}%)")
            if slope_gap > 0.15:
                reasons.append("sharp crossover angle")
            return f"Accepted: Probability ({prob:.1f}%) favorable. " + ", ".join(reasons) + "."

    def handle_crossover(self, symbol: str, signal_type: str, price: float, timestamp: float, features: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Handles entry and exit for both Unfiltered and ML Filtered portfolios.
        Returns: decision, prob_pct, reason
        """
        # Run ML Prediction
        decision, prob_pct, reason = self.predict(features, signal_type)
        symbol_short = symbol.replace("NSE:", "").replace("-EQ", "")
        
        print(f"\n[CROSSOVER DETECTED] {symbol_short} | Crossover: {signal_type} @ Price: ₹{price:.2f}")
        print(f"  [FEATURES] LTQ Burst Ratio: {features['ratio']:.2f}x | Side Share: {features['side_share']*100:.1f}% | Slope Gap: {features['slope_gap']:.4f}")
        print(f"  [ML GATEKEEPER] P(win): {prob_pct:.1f}% -> Decision: {decision}")
        print(f"  [REASON] {reason}")

        with self.lock:
            # 1. Unfiltered Portfolio Processing
            if symbol in self.active_unfiltered:
                prev_trade = self.active_unfiltered[symbol]
                prev_trade.exit_trade(price, timestamp)
                self.unfiltered_trades.append(prev_trade)
                self.active_unfiltered.pop(symbol)
                print(f"  [PORTFOLIO: UNFILTERED] Exited previous {prev_trade.signal_type} trade. Entry: ₹{prev_trade.entry_price:.2f} -> Exit: ₹{price:.2f} | P&L: ₹{prev_trade.pnl:.2f} ({prev_trade.pnl_pct:+.2f}%)")
                self.log_trade_to_csv(prev_trade, "UNFILTERED")
                
            # Open new Unfiltered trade
            unf_trade = TradeRecord(symbol, signal_type, price, timestamp, features)
            self.active_unfiltered[symbol] = unf_trade
            print(f"  [PORTFOLIO: UNFILTERED] Entered new {signal_type} position.")
            
            # 2. ML Filtered Portfolio Processing
            # Crossover exits the previous trade regardless of whether the new signal is accepted or skipped.
            # If the new signal is avoided, the portfolio stays flat.
            if symbol in self.active_filtered:
                prev_trade = self.active_filtered[symbol]
                prev_trade.exit_trade(price, timestamp)
                self.filtered_trades.append(prev_trade)
                self.active_filtered.pop(symbol)
                print(f"  [PORTFOLIO: ML FILTERED] Exited previous {prev_trade.signal_type} trade. Entry: ₹{prev_trade.entry_price:.2f} -> Exit: ₹{price:.2f} | P&L: ₹{prev_trade.pnl:.2f} ({prev_trade.pnl_pct:+.2f}%)")
                self.log_trade_to_csv(prev_trade, "ML FILTERED")
                
            # If accepted, open new trade in Filtered portfolio
            if decision == "ACCEPT":
                filt_trade = TradeRecord(symbol, signal_type, price, timestamp, features)
                self.active_filtered[symbol] = filt_trade
                print(f"  [PORTFOLIO: ML FILTERED] Accepted signal. Entered new {signal_type} position.")
            else:
                print(f"  [PORTFOLIO: ML FILTERED] Avoided signal. Staying FLAT.")
                
        # Trigger online retraining when an unfiltered trade completes
        if symbol in self.active_unfiltered:
            # Retrain on the last completed unfiltered trade (if any)
            completed_list = [t for t in self.unfiltered_trades if t.symbol == symbol]
            if completed_list:
                self._retrain_model_online([completed_list[-1]])
                
        # Print current backtest report summary
        report = self.get_backtest_report()
        unf_rep = report["unfiltered"]
        filt_rep = report["filtered"]
        print(f"  [SESSION REPORT SUMMARY]")
        print(f"    - Unfiltered Portfolio: {unf_rep['trades']} trades | Net P&L: ₹{unf_rep['net_pnl']:.2f} | WinRate: {unf_rep['win_rate']:.1f}% | MaxDD: ₹{unf_rep['max_dd']:.2f}")
        print(f"    - ML Filtered Portfolio: {filt_rep['trades']} trades | Net P&L: ₹{filt_rep['net_pnl']:.2f} | WinRate: {filt_rep['win_rate']:.1f}% | MaxDD: ₹{filt_rep['max_dd']:.2f}\n")
                
        # Save backtest report to CSV (Deliverable #6)
        self.save_backtest_report_csv()

        return decision, prob_pct, reason

    def log_trade_to_csv(self, trade: TradeRecord, portfolio: str):
        """Logs a completed trade record to a CSV file (Deliverable #3)."""
        try:
            log_file = "trade_history_log.csv"
            import os
            write_header = not os.path.exists(log_file)
            with open(log_file, "a") as f:
                if write_header:
                    f.write("symbol,portfolio,direction,entry_time,entry_price,exit_time,exit_price,pnl,pnl_pct,status\n")
                
                from datetime import datetime
                entry_dt = datetime.fromtimestamp(trade.entry_time).strftime("%Y-%m-%d %H:%M:%S")
                exit_dt = datetime.fromtimestamp(trade.exit_time).strftime("%Y-%m-%d %H:%M:%S")
                
                f.write(f"{trade.symbol},{portfolio},{trade.signal_type},{entry_dt},{trade.entry_price:.2f},{exit_dt},{trade.exit_price:.2f},{trade.pnl:.4f},{trade.pnl_pct:.2f}%,{trade.status}\n")
        except Exception as e:
            print(f"Error writing trade log: {e}")

    def save_backtest_report_csv(self):
        """Saves the cumulative backtest comparison report to a CSV file (Deliverable #6)."""
        try:
            report = self.get_backtest_report()
            unf = report["unfiltered"]
            filt = report["filtered"]
            
            with open("backtest_report_log.csv", "w") as f:
                f.write("Metric,Unfiltered Portfolio,ML Filtered Portfolio\n")
                f.write(f"Total Trades,{unf['trades']},{filt['trades']}\n")
                f.write(f"Win Rate %,{unf['win_rate']:.2f}%,{filt['win_rate']:.2f}%\n")
                f.write(f"Net P&L (Points/Rs),{unf['net_pnl']:.4f},{filt['net_pnl']:.4f}\n")
                f.write(f"Max Drawdown (Points/Rs),{unf['max_dd']:.4f},{filt['max_dd']:.4f}\n")
        except Exception as e:
            print(f"Error writing backtest report: {e}")

    def update_active_trades(self, ticks: Dict[str, Any], now: float):
        """Updates prices of active trades in both portfolios."""
        with self.lock:
            for symbol, trade in list(self.active_unfiltered.items()):
                tick = ticks.get(symbol)
                if tick and tick.ltp > 0:
                    trade.update(tick.ltp, now)
            for symbol, trade in list(self.active_filtered.items()):
                tick = ticks.get(symbol)
                if tick and tick.ltp > 0:
                    trade.update(tick.ltp, now)

    def _retrain_model_online(self, resolved_trades: List[TradeRecord]):
        """Performs incremental training of the model."""
        with self.lock:
            for trade in resolved_trades:
                features_list = [trade.features[col] for col in self.features_columns]
                outcome = 1 if trade.status == "WIN" else 0
                self.X_train.append(features_list)
                self.y_train.append(outcome)
            
            # Re-fit
            try:
                self.model.fit(np.array(self.X_train), np.array(self.y_train))
            except Exception as e:
                print(f"Error during online retraining: {e}")

    def get_backtest_report(self) -> Dict[str, Dict[str, Any]]:
        """Returns side-by-side performance metrics for Unfiltered vs ML Filtered portfolios."""
        with self.lock:
            unf_comp = list(self.unfiltered_trades)
            filt_comp = list(self.filtered_trades)
            
        def analyze(trades: List[TradeRecord]) -> Dict[str, Any]:
            total = len(trades)
            if total == 0:
                return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "max_dd": 0.0}
            
            wins = sum(1 for t in trades if t.status == "WIN")
            win_rate = (wins / total) * 100
            net_pnl = sum(t.pnl for t in trades)
            
            # Drawdown
            equity = 0.0
            peak = 0.0
            max_dd = 0.0
            # Sort by exit time to build equity curve
            sorted_trades = sorted(trades, key=lambda t: t.exit_time)
            for t in sorted_trades:
                equity += t.pnl
                peak = max(peak, equity)
                dd = peak - equity
                max_dd = max(max_dd, dd)
                
            return {
                "trades": total,
                "win_rate": win_rate,
                "net_pnl": net_pnl,
                "max_dd": max_dd
            }
            
        return {
            "unfiltered": analyze(unf_comp),
            "filtered": analyze(filt_comp)
        }

    def get_feature_importances(self) -> Dict[str, float]:
        with self.lock:
            importances = self.model.feature_importances_
        result = {}
        for col, imp in zip(self.features_columns, importances):
            result[col] = float(imp * 100)
        return result
