"""
עיבוד ועיבוד נתוני שוק

כולל פונקציות לטעינה, ניקוי ועיבוד של נתוני שוק.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging


logger = logging.getLogger(__name__)


class DataProcessor:
    """
    מעבד נתוני שוק
    
    כולל פונקציות לניקוי, נרמול, וחישוב אינדיקטורים בסיסיים.
    """
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        ניקוי נתונים
        
        Args:
            data: DataFrame עם נתוני שוק
            
        Returns:
            DataFrame נקי
        """
        df = data.copy()
        
        # הסרת שורות עם ערכים חסרים
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with missing values")
        
        # הסרת duplic ates
        df = df[~df.index.duplicated(keep='first')]
        
        # מיון לפי תאריך
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def validate_ohlc(data: pd.DataFrame) -> pd.DataFrame:
        """
        תיקוף נתוני OHLC
        
        Args:
            data: DataFrame עם נתוני OHLC
            
        Returns:
            DataFrame מתוקן
        """
        df = data.copy()
        
        # וידוא ש-high >= low
        invalid_rows = df['high'] < df['low']
        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows where high < low, swapping values")
            df.loc[invalid_rows, ['high', 'low']] = df.loc[invalid_rows, ['low', 'high']].values
        
        # וידוא ש-high >= close >= low
        df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])
        df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
        
        # הסרת ערכים שליליים או אפס
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid = df[col] <= 0
            if invalid.any():
                logger.warning(f"Found {invalid.sum()} non-positive values in {col}, removing rows")
                df = df[~invalid]
        
        return df
    
    @staticmethod
    def resample_data(
        data: pd.DataFrame,
        timeframe: str,
        aggregation: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        דגימה מחדש של נתונים לתקופת זמן אחרת
        
        Args:
            data: DataFrame עם נתוני שוק
            timeframe: תקופת זמן חדשה ('1H', '4H', '1D', וכו')
            aggregation: מילון מותאם אישית לאגרגציה
            
        Returns:
            DataFrame עם דגימה חדשה
        """
        if aggregation is None:
            aggregation = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        df = data.resample(timeframe).agg(aggregation)
        df = df.dropna()
        
        return df
    
    @staticmethod
    def calculate_returns(
        data: pd.DataFrame,
        column: str = 'close',
        periods: int = 1,
        method: str = 'simple'
    ) -> pd.Series:
        """
        חישוב תשואות
        
        Args:
            data: DataFrame עם נתוני מחירים
            column: עמודה לחישוב
            periods: מספר תקופות
            method: 'simple' או 'log'
            
        Returns:
            Series עם תשואות
        """
        if method == 'simple':
            returns = data[column].pct_change(periods=periods)
        elif method == 'log':
            returns = np.log(data[column] / data[column].shift(periods))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns
    
    @staticmethod
    def calculate_volatility(
        data: pd.DataFrame,
        column: str = 'close',
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        חישוב תנודתיות (volatility)
        
        Args:
            data: DataFrame עם נתוני מחירים
            column: עמודה לחישוב
            window: גודל חלון
            annualize: האם להמיר לשנתי
            
        Returns:
            Series עם volatility
        """
        returns = data[column].pct_change()
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            # הנחה של 252 ימי מסחר בשנה
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def normalize_data(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        נרמול נתונים
        
        Args:
            data: DataFrame לנרמול
            columns: עמודות לנרמול (None = כל העמודות)
            method: 'minmax' או 'zscore'
            
        Returns:
            DataFrame מנורמל
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'minmax':
            for col in columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0
        
        elif method == 'zscore':
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return df
    
    @staticmethod
    def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        הוספת תכונות זמן
        
        Args:
            data: DataFrame עם אינדקס תאריכים
            
        Returns:
            DataFrame עם תכונות זמן נוספות
        """
        df = data.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping time features")
            return df
        
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['hour'] = df.index.hour
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    @staticmethod
    def handle_outliers(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'clip',
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        טיפול ב-outliers
        
        Args:
            data: DataFrame
            columns: עמודות לטיפול (None = כל העמודות המספריות)
            method: 'clip', 'remove', או 'winsorize'
            n_std: מספר סטיות תקן
            
        Returns:
            DataFrame מעובד
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            if method == 'clip':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == 'winsorize':
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df
    
    @staticmethod
    def create_lagged_features(
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        יצירת תכונות lag
        
        Args:
            data: DataFrame
            columns: עמודות ליצירת lag
            lags: רשימת lags (למשל [1, 2, 3])
            
        Returns:
            DataFrame עם תכונות lag
        """
        df = data.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(
        data: pd.DataFrame,
        column: str,
        windows: List[int],
        functions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        יצירת תכונות rolling
        
        Args:
            data: DataFrame
            column: עמודה לחישוב
            windows: רשימת גדלי חלונות
            functions: פונקציות לחישוב (['mean', 'std', 'min', 'max'])
            
        Returns:
            DataFrame עם תכונות rolling
        """
        if functions is None:
            functions = ['mean', 'std']
        
        df = data.copy()
        
        for window in windows:
            for func in functions:
                col_name = f'{column}_rolling_{func}_{window}'
                if func == 'mean':
                    df[col_name] = df[column].rolling(window=window).mean()
                elif func == 'std':
                    df[col_name] = df[column].rolling(window=window).std()
                elif func == 'min':
                    df[col_name] = df[column].rolling(window=window).min()
                elif func == 'max':
                    df[col_name] = df[column].rolling(window=window).max()
                elif func == 'sum':
                    df[col_name] = df[column].rolling(window=window).sum()
        
        return df

