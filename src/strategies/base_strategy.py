"""
מחלקת בסיס לכל אסטרטגיות המסחר

מחלקה אבסטרקטית שמגדירה את הממשק הבסיסי לכל אסטרטגיית מסחר.
כל אסטרטגיה חדשה צריכה לרשת ממחלקה זו וליישם את המתודות האבסטרקטיות.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging


class BaseStrategy(ABC):
    """
    מחלקת בסיס לכל אסטרטגיות המסחר
    
    Attributes:
        name: שם האסטרטגיה
        params: פרמטרים של האסטרטגיה
        position: פוזיציה נוכחית (-1: short, 0: neutral, 1: long)
        trades: רשימת עסקאות שבוצעו
        performance_metrics: מדדי ביצועים
        logger: Logger למעקב
    
    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def __init__(self, param1=10):
        ...         super().__init__('MyStrategy', {'param1': param1})
        ...     
        ...     def calculate_indicators(self, data):
        ...         # חישוב אינדיקטורים
        ...         return data
        ...     
        ...     def generate_signals(self, data):
        ...         # יצירת אותות
        ...         return pd.Series(0, index=data.index)
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        אתחול אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            params: פרמטרים (dict)
        """
        self.name = name
        self.params = params or {}
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.trades: List[Dict] = []
        self.performance_metrics: Dict = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        self.logger.info(f"Initialized strategy: {name} with params: {params}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מסחר
        
        Args:
            data: נתוני שוק (DataFrame עם עמודות: open, high, low, close, volume)
            
        Returns:
            Series עם אותות: 1 (קנה), -1 (מכור), 0 (החזק)
            
        Raises:
            NotImplementedError: אם לא מומש
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב אינדיקטורים טכניים
        
        Args:
            data: נתוני שוק
            
        Returns:
            DataFrame עם אינדיקטורים נוספים
            
        Raises:
            NotImplementedError: אם לא מומש
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        בדיקת תקינות נתוני שוק
        
        Args:
            data: נתוני שוק
            
        Returns:
            True אם הנתונים תקינים
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # בדיקת עמודות נדרשות
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Need: {required_columns}")
            return False
        
        # בדיקת ערכים חסרים
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains null values")
            return False
        
        # בדיקת ערכים שליליים
        if (data[required_columns] < 0).any().any():
            self.logger.error("Data contains negative values")
            return False
        
        # בדיקת high >= low
        if not (data['high'] >= data['low']).all():
            self.logger.error("High prices should be >= low prices")
            return False
        
        return True
    
    def update_performance(self, trade_result: Dict):
        """
        עדכון מדדי ביצועים
        
        Args:
            trade_result: תוצאת עסקה
                - symbol: str
                - entry_price: float
                - exit_price: float
                - quantity: int
                - pnl: float
                - entry_time: datetime
                - exit_time: datetime
        """
        self.trades.append(trade_result)
        self._calculate_metrics()
        
        pnl = trade_result.get('pnl', 0)
        self.logger.info(
            f"Trade completed: {trade_result.get('symbol')} | "
            f"P&L: ${pnl:.2f} | "
            f"Total trades: {len(self.trades)}"
        )
    
    def _calculate_metrics(self):
        """חישוב מדדי ביצועים מעודכנים"""
        if not self.trades:
            return
        
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in self.trades if t.get('pnl', 0) < 0)
        
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        
        # חישוב רווח/הפסד ממוצע
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': self._calculate_profit_factor(),
            'max_consecutive_wins': self._calculate_max_streak(True),
            'max_consecutive_losses': self._calculate_max_streak(False),
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0,
        }
    
    def _calculate_profit_factor(self) -> float:
        """
        חישוב Profit Factor
        
        Returns:
            Profit Factor (gross profit / gross loss)
        """
        gross_profit = sum(t['pnl'] for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_max_streak(self, winning: bool) -> int:
        """
        חישוב רצף מקסימלי של ניצחונות/הפסדים
        
        Args:
            winning: True לניצחונות, False להפסדים
            
        Returns:
            מספר העסקאות ברצף המקסימלי
        """
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            pnl = trade.get('pnl', 0)
            is_win = pnl > 0
            
            if is_win == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_performance_summary(self) -> str:
        """
        קבלת סיכום ביצועים כטקסט
        
        Returns:
            מחרוזת עם סיכום ביצועים
        """
        if not self.performance_metrics:
            return "No trades yet"
        
        metrics = self.performance_metrics
        
        summary = f"""
Strategy: {self.name}
{'=' * 50}
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']*100:.2f}%
Total P&L: ${metrics['total_pnl']:.2f}
Average P&L per Trade: ${metrics['avg_pnl_per_trade']:.2f}

Winning Trades: {metrics['winning_trades']}
Average Win: ${metrics['avg_win']:.2f}
Largest Win: ${metrics['largest_win']:.2f}
Max Consecutive Wins: {metrics['max_consecutive_wins']}

Losing Trades: {metrics['losing_trades']}
Average Loss: ${metrics['avg_loss']:.2f}
Largest Loss: ${metrics['largest_loss']:.2f}
Max Consecutive Losses: {metrics['max_consecutive_losses']}

Profit Factor: {metrics['profit_factor']:.2f}
{'=' * 50}
"""
        return summary
    
    def reset(self):
        """איפוס האסטרטגיה למצב התחלתי"""
        self.position = 0
        self.trades = []
        self.performance_metrics = {}
        self.logger.info(f"Strategy {self.name} reset")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
    
    def __str__(self) -> str:
        return f"{self.name} Strategy"

