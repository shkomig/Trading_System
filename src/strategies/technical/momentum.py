"""
אסטרטגיות מבוססות Momentum

אסטרטגיות המשתמשות במומנטום (תנופה) של המחיר לזיהוי טרנדים.
"""

import pandas as pd
import numpy as np
from ..base_strategy import BaseStrategy
from typing import Optional


class MomentumStrategy(BaseStrategy):
    """
    אסטרטגיית Momentum בסיסית
    
    קונה כאשר המומנטום חיובי ומעל סף מסוים,
    מוכרת כאשר המומנטום שלילי או מתחת לסף.
    """
    
    def __init__(self,
                 lookback_period: int = 14,
                 threshold: float = 0.02,
                 use_volume: bool = False):
        """
        אתחול
        
        Args:
            lookback_period: תקופה לחישוב מומנטום
            threshold: סף מינימלי למומנטום (2% = 0.02)
            use_volume: האם להשתמש בנפח כפילטר
        """
        params = {
            'lookback_period': lookback_period,
            'threshold': threshold,
            'use_volume': use_volume
        }
        super().__init__('Momentum', params)
        
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.use_volume = use_volume
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב אינדיקטורי מומנטום
        
        Args:
            data: נתוני שוק
            
        Returns:
            DataFrame עם אינדיקטורים
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        # מומנטום פשוט - שינוי במחיר
        df['momentum'] = df['close'].pct_change(periods=self.lookback_period)
        
        # Rate of Change (ROC)
        df['roc'] = ((df['close'] - df['close'].shift(self.lookback_period)) / 
                     df['close'].shift(self.lookback_period)) * 100
        
        # מומנטום מנורמל (z-score)
        df['momentum_zscore'] = (
            (df['momentum'] - df['momentum'].rolling(window=50).mean()) /
            df['momentum'].rolling(window=50).std()
        )
        
        # Acceleration (שינוי במומנטום)
        df['acceleration'] = df['momentum'].diff()
        
        if self.use_volume:
            # Volume momentum
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            df['obv_ma'] = df['obv'].rolling(window=20).mean()
        
        self.logger.info(f"Calculated momentum indicators (period={self.lookback_period})")
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מסחר
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם אותות
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: מומנטום חיובי ומעל הסף
        buy_condition = (df['momentum'] > self.threshold)
        
        # תנאי מכירה: מומנטום שלילי או מתחת לסף
        sell_condition = (df['momentum'] < -self.threshold)
        
        if self.use_volume:
            # הוספת פילטר נפח - רק אם הנפח גבוה מהממוצע
            buy_condition = buy_condition & (df['volume_ratio'] > 1.0)
            sell_condition = sell_condition & (df['volume_ratio'] > 1.0)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        num_buy = (signals == 1).sum()
        num_sell = (signals == -1).sum()
        
        self.logger.info(f"Generated {num_buy} buy signals and {num_sell} sell signals")
        
        return signals


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum Strategy
    
    משלבת Absolute Momentum (טרנד) ו-Relative Momentum (השוואה למדד)
    """
    
    def __init__(self,
                 lookback_period: int = 126,  # ~6 חודשים
                 benchmark_data: Optional[pd.DataFrame] = None):
        """
        אתחול
        
        Args:
            lookback_period: תקופה לחישוב מומנטום
            benchmark_data: נתוני benchmark להשוואה
        """
        params = {
            'lookback_period': lookback_period
        }
        super().__init__('Dual_Momentum', params)
        
        self.lookback_period = lookback_period
        self.benchmark_data = benchmark_data
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב מומנטום אבסולוטי ויחסי"""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        # Absolute Momentum - תשואה לאורך תקופה
        df['absolute_momentum'] = df['close'].pct_change(periods=self.lookback_period)
        
        # Relative Momentum (אם יש benchmark)
        if self.benchmark_data is not None:
            # ודא שהאינדקסים תואמים
            benchmark_returns = self.benchmark_data['close'].pct_change(periods=self.lookback_period)
            df['relative_momentum'] = df['absolute_momentum'] - benchmark_returns
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות dual momentum
        
        קנה רק אם:
        1. Absolute momentum חיובי (טרנד עולה)
        2. Relative momentum חיובי (מנצח את ה-benchmark)
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: שני המומנטומים חיוביים
        if 'relative_momentum' in df.columns:
            buy_condition = (
                (df['absolute_momentum'] > 0) &
                (df['relative_momentum'] > 0)
            )
        else:
            # אם אין benchmark, רק absolute momentum
            buy_condition = df['absolute_momentum'] > 0
        
        # תנאי מכירה: absolute momentum שלילי
        sell_condition = df['absolute_momentum'] < 0
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals


class TrendFollowingStrategy(BaseStrategy):
    """
    אסטרטגיית Trend Following
    
    משתמשת ב-ADX (Average Directional Index) ו-Momentum לזיהוי טרנדים חזקים.
    """
    
    def __init__(self,
                 adx_period: int = 14,
                 adx_threshold: int = 25,
                 momentum_period: int = 14):
        """
        אתחול
        
        Args:
            adx_period: תקופת ADX
            adx_threshold: סף ADX לזיהוי טרנד חזק
            momentum_period: תקופת מומנטום
        """
        params = {
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'momentum_period': momentum_period
        }
        super().__init__('Trend_Following', params)
        
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.momentum_period = momentum_period
    
    def calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """
        חישוב ADX (Average Directional Index)
        
        Args:
            data: נתוני OHLC
            
        Returns:
            Series של ADX
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # חישוב True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()
        
        # חישוב Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # חישוב DI
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        # חישוב DX ו-ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב ADX ומומנטום"""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        # ADX
        df['adx'] = self.calculate_adx(df)
        
        # Momentum
        df['momentum'] = df['close'].pct_change(periods=self.momentum_period)
        
        # EMA for trend direction
        df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות trend following
        
        קונה כאשר:
        - ADX מעל הסף (טרנד חזק)
        - מומנטום חיובי
        - EMA מהיר מעל EMA איטי
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: טרנד חזק + כיוון עולה
        buy_condition = (
            (df['adx'] > self.adx_threshold) &
            (df['momentum'] > 0) &
            (df['ema_fast'] > df['ema_slow'])
        )
        
        # תנאי מכירה: טרנד חזק + כיוון יורד
        sell_condition = (
            (df['adx'] > self.adx_threshold) &
            (df['momentum'] < 0) &
            (df['ema_fast'] < df['ema_slow'])
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    אסטרטגיית Mean Reversion
    
    ההפך ממומנטום - מחפשת הזדמנויות כאשר המחיר רחוק מהממוצע.
    """
    
    def __init__(self,
                 ma_period: int = 20,
                 std_threshold: float = 2.0):
        """
        אתחול
        
        Args:
            ma_period: תקופת ממוצע נע
            std_threshold: מספר סטיות תקן
        """
        params = {
            'ma_period': ma_period,
            'std_threshold': std_threshold
        }
        super().__init__('Mean_Reversion', params)
        
        self.ma_period = ma_period
        self.std_threshold = std_threshold
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב ממוצע וסטיית תקן"""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided")
        
        df = data.copy()
        
        # ממוצע נע
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()
        
        # סטיית תקן
        df['std'] = df['close'].rolling(window=self.ma_period).std()
        
        # Z-Score (כמה סטיות תקן מהממוצע)
        df['zscore'] = (df['close'] - df['ma']) / df['std']
        
        # Upper and Lower bands
        df['upper_band'] = df['ma'] + (df['std'] * self.std_threshold)
        df['lower_band'] = df['ma'] - (df['std'] * self.std_threshold)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות mean reversion
        
        קונה כאשר המחיר נמוך מדי (מתחת ל-lower band)
        מוכר כאשר המחיר גבוה מדי (מעל ל-upper band)
        """
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: מחיר מתחת ל-lower band
        buy_condition = df['close'] < df['lower_band']
        
        # תנאי מכירה: מחיר מעל ל-upper band
        sell_condition = df['close'] > df['upper_band']
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals

