"""
אסטרטגיית Moving Average Crossover

אסטרטגיה קלאסית המבוססת על חציית ממוצעים נעים.
קונה כאשר ממוצע קצר חוצה מעל ממוצע ארוך, ומוכר כאשר חוצה מתחת.
"""

import pandas as pd
import numpy as np
from ..base_strategy import BaseStrategy
from typing import Optional


class MovingAverageCrossover(BaseStrategy):
    """
    אסטרטגיית חציית ממוצעים נעים
    
    Attributes:
        short_window: חלון ממוצע קצר
        long_window: חלון ממוצע ארוך
        use_ema: האם להשתמש ב-EMA במקום SMA
    
    Example:
        >>> strategy = MovingAverageCrossover(short_window=50, long_window=200)
        >>> signals = strategy.generate_signals(data)
        >>> print(strategy.get_performance_summary())
    """
    
    def __init__(self,
                 short_window: int = 50,
                 long_window: int = 200,
                 use_ema: bool = False):
        """
        אתחול אסטרטגיה
        
        Args:
            short_window: חלון ממוצע קצר
            long_window: חלון ממוצע ארוך
            use_ema: האם להשתמש ב-EMA במקום SMA
        """
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        
        params = {
            'short_window': short_window,
            'long_window': long_window,
            'use_ema': use_ema
        }
        super().__init__('MA_Crossover', params)
        
        self.short_window = short_window
        self.long_window = long_window
        self.use_ema = use_ema
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב ממוצעים נעים
        
        Args:
            data: נתוני שוק
            
        Returns:
            DataFrame עם ממוצעים נעים
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        if self.use_ema:
            # Exponential Moving Average
            df['short_ma'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
            df['long_ma'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
            self.logger.info(f"Calculated EMA({self.short_window}, {self.long_window})")
        else:
            # Simple Moving Average
            df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
            df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
            self.logger.info(f"Calculated SMA({self.short_window}, {self.long_window})")
        
        # חישוב המרחק בין הממוצעים (%)
        df['ma_distance'] = ((df['short_ma'] - df['long_ma']) / df['long_ma']) * 100
        
        # חישוב slope של הממוצע הארוך (טרנד)
        df['long_ma_slope'] = df['long_ma'].pct_change(periods=5) * 100
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מסחר
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם אותות: 1 (קנה), -1 (מכור), 0 (החזק)
        """
        df = self.calculate_indicators(data)
        
        # אתחול עמודת אותות
        signals = pd.Series(0, index=df.index)
        
        # חישוב פוזיציה מבוססת ממוצעים
        # 1 כאשר short MA > long MA, -1 אחרת
        df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
        
        # זיהוי חציות - שינוי בפוזיציה
        df['signal'] = df['position'].diff()
        
        # אות קנייה: short MA חוצה מעל long MA (signal = 2)
        buy_signals = df['signal'] == 2
        signals[buy_signals] = 1
        
        # אות מכירה: short MA חוצה מתחת ל-long MA (signal = -2)
        sell_signals = df['signal'] == -2
        signals[sell_signals] = -1
        
        # סינון אותות חלשים - רק אם המרחק בין הממוצעים מספיק גדול
        min_distance = 0.5  # 0.5% מרחק מינימלי
        weak_signals = abs(df['ma_distance']) < min_distance
        signals[weak_signals] = 0
        
        num_buy = (signals == 1).sum()
        num_sell = (signals == -1).sum()
        
        self.logger.info(f"Generated {num_buy} buy signals and {num_sell} sell signals")
        
        return signals
    
    def get_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        קבלת חוזק האות
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם חוזק אות (0-1)
        """
        df = self.calculate_indicators(data)
        
        # חוזק האות מבוסס על המרחק בין הממוצעים
        strength = abs(df['ma_distance']) / 10  # נרמול ל-0-1 בקירוב
        strength = strength.clip(0, 1)
        
        return strength


class TripleMovingAverage(BaseStrategy):
    """
    אסטרטגיה עם 3 ממוצעים נעים
    
    משתמשת בממוצע קצר, בינוני וארוך לזיהוי טרנדים חזקים יותר.
    """
    
    def __init__(self,
                 fast_window: int = 20,
                 medium_window: int = 50,
                 slow_window: int = 200,
                 use_ema: bool = False):
        """
        אתחול
        
        Args:
            fast_window: ממוצע מהיר
            medium_window: ממוצע בינוני
            slow_window: ממוצע איטי
            use_ema: האם להשתמש ב-EMA
        """
        params = {
            'fast_window': fast_window,
            'medium_window': medium_window,
            'slow_window': slow_window,
            'use_ema': use_ema
        }
        super().__init__('Triple_MA', params)
        
        self.fast_window = fast_window
        self.medium_window = medium_window
        self.slow_window = slow_window
        self.use_ema = use_ema
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב 3 ממוצעים נעים"""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        if self.use_ema:
            df['fast_ma'] = df['close'].ewm(span=self.fast_window, adjust=False).mean()
            df['medium_ma'] = df['close'].ewm(span=self.medium_window, adjust=False).mean()
            df['slow_ma'] = df['close'].ewm(span=self.slow_window, adjust=False).mean()
        else:
            df['fast_ma'] = df['close'].rolling(window=self.fast_window).mean()
            df['medium_ma'] = df['close'].rolling(window=self.medium_window).mean()
            df['slow_ma'] = df['close'].rolling(window=self.slow_window).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מבוססת 3 ממוצעים
        
        אות קנייה: fast > medium > slow (טרנד עולה חזק)
        אות מכירה: fast < medium < slow (טרנד יורד חזק)
        """
        df = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: סדר עולה של הממוצעים
        buy_condition = (
            (df['fast_ma'] > df['medium_ma']) &
            (df['medium_ma'] > df['slow_ma']) &
            (df['fast_ma'].shift(1) <= df['medium_ma'].shift(1))  # חציה חדשה
        )
        
        # תנאי מכירה: סדר יורד של הממוצעים
        sell_condition = (
            (df['fast_ma'] < df['medium_ma']) &
            (df['medium_ma'] < df['slow_ma']) &
            (df['fast_ma'].shift(1) >= df['medium_ma'].shift(1))  # חציה חדשה
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals

