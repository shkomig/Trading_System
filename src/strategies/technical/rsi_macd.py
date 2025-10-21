"""
אסטרטגיית RSI + MACD + Bollinger Bands

אסטרטגיה משולבת המשתמשת במספר אינדיקטורים טכניים:
- RSI (Relative Strength Index) - זיהוי oversold/overbought
- MACD (Moving Average Convergence Divergence) - זיהוי momentum
- Bollinger Bands - זיהוי volatility ורמות תמיכה/התנגדות
"""

import pandas as pd
import numpy as np
from ..base_strategy import BaseStrategy
from typing import Tuple


class RSI_MACD_Strategy(BaseStrategy):
    """
    אסטרטגיה משולבת של RSI, MACD ו-Bollinger Bands
    
    תנאי קנייה:
    - RSI נמוך (oversold)
    - MACD חיובי (bullish)
    - מחיר קרוב ל-Bollinger Band התחתון
    
    תנאי מכירה:
    - RSI גבוה (overbought)
    - MACD שלילי (bearish)
    - מחיר קרוב ל-Bollinger Band העליון
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: float = 2.0):
        """
        אתחול
        
        Args:
            rsi_period: תקופת RSI
            rsi_oversold: סף oversold
            rsi_overbought: סף overbought
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
        """
        params = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bb_period': bb_period,
            'bb_std': bb_std
        }
        super().__init__('RSI_MACD_BB', params)
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """
        חישוב RSI (Relative Strength Index)
        
        Args:
            data: Series של מחירים
            period: תקופה
            
        Returns:
            Series של RSI
        """
        delta = data.diff()
        
        # חישוב gains ו-losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # חישוב ממוצעים נעים
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # חישוב RS ו-RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        חישוב MACD
        
        Args:
            data: Series של מחירים
            
        Returns:
            Tuple של (macd, signal, histogram)
        """
        # חישוב EMAs
        ema_fast = data.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data.ewm(span=self.macd_slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        # Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        חישוב Bollinger Bands
        
        Args:
            data: Series של מחירים
            
        Returns:
            Tuple של (upper, middle, lower)
        """
        # Middle band (SMA)
        middle = data.rolling(window=self.bb_period).mean()
        
        # Standard deviation
        std = data.rolling(window=self.bb_period).std()
        
        # Upper and Lower bands
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return upper, middle, lower
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב כל האינדיקטורים
        
        Args:
            data: נתוני שוק
            
        Returns:
            DataFrame עם כל האינדיקטורים
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # חישוב %B (מיקום המחיר בין הבנדים)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # חישוב רוחב הבנד (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        self.logger.info(f"Calculated RSI, MACD, and Bollinger Bands")
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מסחר מבוססי כללים משולבים
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם אותות
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: RSI נמוך + MACD חיובי + מחיר קרוב ל-BB התחתון
        buy_condition = (
            (df['rsi'] < self.rsi_oversold) &
            (df['macd'] > df['macd_signal']) &
            (df['macd_hist'] > 0) &
            (df['bb_percent'] < 0.2)  # קרוב ל-BB התחתון
        )
        
        # תנאי מכירה: RSI גבוה + MACD שלילי + מחיר קרוב ל-BB העליון
        sell_condition = (
            (df['rsi'] > self.rsi_overbought) &
            (df['macd'] < df['macd_signal']) &
            (df['macd_hist'] < 0) &
            (df['bb_percent'] > 0.8)  # קרוב ל-BB העליון
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        num_buy = (signals == 1).sum()
        num_sell = (signals == -1).sum()
        
        self.logger.info(f"Generated {num_buy} buy signals and {num_sell} sell signals")
        
        return signals
    
    def get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        זיהוי מצב שוק (trending vs ranging)
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series: 'trending' או 'ranging'
        """
        df = self.calculate_indicators(data)
        
        # שוק trending כאשר BB width גבוה
        # שוק ranging כאשר BB width נמוך
        median_width = df['bb_width'].rolling(window=50).median()
        
        regime = pd.Series('ranging', index=df.index)
        regime[df['bb_width'] > median_width] = 'trending'
        
        return regime


class RSI_Divergence_Strategy(BaseStrategy):
    """
    אסטרטגיה מבוססת RSI Divergence
    
    מחפשת divergences בין המחיר ל-RSI כאינדיקציה לשינוי טרנד.
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 lookback_period: int = 14):
        """
        אתחול
        
        Args:
            rsi_period: תקופת RSI
            lookback_period: תקופה לחיפוש divergence
        """
        params = {
            'rsi_period': rsi_period,
            'lookback_period': lookback_period
        }
        super().__init__('RSI_Divergence', params)
        
        self.rsi_period = rsi_period
        self.lookback_period = lookback_period
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """חישוב RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב RSI"""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        return df
    
    def detect_divergence(self, price: pd.Series, rsi: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        זיהוי divergences
        
        Returns:
            Tuple של (bullish_divergence, bearish_divergence)
        """
        bullish_div = pd.Series(False, index=price.index)
        bearish_div = pd.Series(False, index=price.index)
        
        for i in range(window, len(price)):
            # חיפוש בחלון
            price_window = price.iloc[i-window:i+1]
            rsi_window = rsi.iloc[i-window:i+1]
            
            # Bullish Divergence: מחיר עושה low חדש, אבל RSI לא
            if (price.iloc[i] == price_window.min() and 
                rsi.iloc[i] > rsi_window.min()):
                bullish_div.iloc[i] = True
            
            # Bearish Divergence: מחיר עושה high חדש, אבל RSI לא
            if (price.iloc[i] == price_window.max() and 
                rsi.iloc[i] < rsi_window.max()):
                bearish_div.iloc[i] = True
        
        return bullish_div, bearish_div
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מבוססי divergence
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם אותות
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # זיהוי divergences
        bullish_div, bearish_div = self.detect_divergence(
            df['close'], 
            df['rsi'], 
            self.lookback_period
        )
        
        # אות קנייה על bullish divergence
        signals[bullish_div] = 1
        
        # אות מכירה על bearish divergence
        signals[bearish_div] = -1
        
        return signals

