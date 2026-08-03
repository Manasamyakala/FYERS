# indicators.py
# Technical indicators implementation for the Stock Market Screening and Analysis System
# Implements SMMA (Smoothed Moving Average) and other utility calculations.

import numpy as np
import pandas as pd
from typing import Union, List, Optional

def calculate_smma(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Calculates the Smoothed Moving Average (SMMA) for a given series of prices and period.
    SMMA Formula:
    - First value: Simple Moving Average (SMA) of the first N periods.
    - Subsequent values: SMMA(i) = (SMMA(i-1) * (N - 1) + Price(i)) / N
    
    Parameters:
    - prices: Sequence of prices (typically close prices).
    - period: Indicator period (e.g., 20 or 120).
    
    Returns:
    - numpy array of SMMA values of the same length as input, filled with NaN for the first (period-1) entries.
    """
    # Convert input to numpy array
    if isinstance(prices, pd.Series):
        arr = prices.values
    elif isinstance(prices, list):
        arr = np.array(prices, dtype=float)
    else:
        arr = np.array(prices, dtype=float)
        
    n = len(arr)
    smma = np.empty(n)
    smma[:] = np.nan
    
    if n < period:
        return smma
        
    # First value is the simple moving average (SMA)
    sma = np.mean(arr[:period])
    smma[period - 1] = sma
    
    # Calculate subsequent values
    for i in range(period, n):
        smma[i] = (smma[i - 1] * (period - 1) + arr[i]) / period
        
    return smma

def calculate_smma_step(prev_smma: Optional[float], current_price: float, period: int, history_len: int) -> float:
    """
    Incrementally updates the SMMA with a new close price.
    Useful for real-time calculation as new candles form.
    
    Parameters:
    - prev_smma: The SMMA value of the previous candle.
    - current_price: The close price of the current candle.
    - period: Indicator period (e.g., 20 or 120).
    - history_len: Total number of candles in history (including current one).
    
    Returns:
    - The new SMMA value.
    """
    if prev_smma is None or np.isnan(prev_smma):
        # We cannot calculate SMMA if history is too short
        return np.nan
        
    # If we are exactly at the boundary of completing the first SMA period
    # This is handled when history_len == period in the caller.
    
    return (prev_smma * (period - 1) + current_price) / period
