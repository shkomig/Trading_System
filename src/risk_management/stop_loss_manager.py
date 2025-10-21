"""
Stop Loss Manager - ניהול stop loss ו-take profit

מודול לחישוב וניהול של stop loss ו-take profit levels.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    """סוגי Stop Loss"""
    FIXED_PERCENT = "fixed_percent"
    FIXED_AMOUNT = "fixed_amount"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    CHANDELIER = "chandelier"


class StopLossManager:
    """
    מנהל Stop Loss ו-Take Profit
    
    תומך במספר שיטות לקביעת רמות stop loss ו-take profit.
    
    Example:
        >>> manager = StopLossManager()
        >>> sl, tp = manager.calculate_stops(
        ...     entry_price=150.0,
        ...     side='long',
        ...     stop_type=StopLossType.FIXED_PERCENT,
        ...     stop_percent=0.05
        ... )
    """
    
    def __init__(self, default_stop_percent: float = 0.05, default_take_profit_ratio: float = 2.0):
        """
        אתחול
        
        Args:
            default_stop_percent: אחוז stop loss ברירת מחדל (5% = 0.05)
            default_take_profit_ratio: יחס risk/reward (2.0 = take profit פי 2 מה-stop loss)
        """
        self.default_stop_percent = default_stop_percent
        self.default_take_profit_ratio = default_take_profit_ratio
        
        logger.info(f"StopLossManager initialized (stop={default_stop_percent}, tp_ratio={default_take_profit_ratio})")
    
    def calculate_stops(self,
                        entry_price: float,
                        side: str = 'long',
                        stop_type: StopLossType = StopLossType.FIXED_PERCENT,
                        stop_percent: Optional[float] = None,
                        stop_amount: Optional[float] = None,
                        atr: Optional[float] = None,
                        atr_multiplier: float = 2.0,
                        volatility: Optional[float] = None,
                        take_profit_ratio: Optional[float] = None) -> Tuple[float, float]:
        """
        חישוב stop loss ו-take profit
        
        Args:
            entry_price: מחיר כניסה
            side: 'long' או 'short'
            stop_type: סוג stop loss
            stop_percent: אחוז stop loss
            stop_amount: סכום stop loss בדולרים
            atr: Average True Range
            atr_multiplier: מכפיל ATR
            volatility: volatility
            take_profit_ratio: יחס risk/reward
            
        Returns:
            Tuple של (stop_loss_price, take_profit_price)
        """
        if side not in ['long', 'short']:
            raise ValueError("side must be 'long' or 'short'")
        
        if stop_percent is None:
            stop_percent = self.default_stop_percent
        
        if take_profit_ratio is None:
            take_profit_ratio = self.default_take_profit_ratio
        
        # חישוב stop loss לפי השיטה
        if stop_type == StopLossType.FIXED_PERCENT:
            stop_loss = self._fixed_percent_stop(entry_price, side, stop_percent)
        
        elif stop_type == StopLossType.FIXED_AMOUNT:
            if stop_amount is None:
                raise ValueError("stop_amount required for FIXED_AMOUNT type")
            stop_loss = self._fixed_amount_stop(entry_price, side, stop_amount)
        
        elif stop_type == StopLossType.ATR_BASED:
            if atr is None:
                raise ValueError("atr required for ATR_BASED type")
            stop_loss = self._atr_based_stop(entry_price, side, atr, atr_multiplier)
        
        elif stop_type == StopLossType.VOLATILITY_BASED:
            if volatility is None:
                raise ValueError("volatility required for VOLATILITY_BASED type")
            stop_loss = self._volatility_based_stop(entry_price, side, volatility)
        
        else:
            logger.warning(f"Unknown stop type {stop_type}, using fixed percent")
            stop_loss = self._fixed_percent_stop(entry_price, side, stop_percent)
        
        # חישוב take profit
        take_profit = self._calculate_take_profit(entry_price, stop_loss, side, take_profit_ratio)
        
        logger.debug(
            f"Calculated stops: entry=${entry_price:.2f}, "
            f"stop=${stop_loss:.2f}, tp=${take_profit:.2f}, side={side}"
        )
        
        return stop_loss, take_profit
    
    def _fixed_percent_stop(self, entry_price: float, side: str, percent: float) -> float:
        """Stop loss באחוזים קבועים"""
        if side == 'long':
            return entry_price * (1 - percent)
        else:  # short
            return entry_price * (1 + percent)
    
    def _fixed_amount_stop(self, entry_price: float, side: str, amount: float) -> float:
        """Stop loss בסכום קבוע"""
        if side == 'long':
            return entry_price - amount
        else:  # short
            return entry_price + amount
    
    def _atr_based_stop(self, entry_price: float, side: str, atr: float, multiplier: float) -> float:
        """
        Stop loss מבוסס ATR
        
        ATR (Average True Range) מודד את ה-volatility.
        Stop loss = entry ± (ATR * multiplier)
        """
        stop_distance = atr * multiplier
        
        if side == 'long':
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def _volatility_based_stop(self, entry_price: float, side: str, volatility: float) -> float:
        """
        Stop loss מבוסס volatility
        
        volatility = סטיית תקן של תשואות
        """
        # המרת volatility יומי לסף stop loss
        # נשתמש ב-2 סטיות תקן כברירת מחדל
        stop_distance = entry_price * volatility * 2
        
        if side == 'long':
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def _calculate_take_profit(self,
                                entry_price: float,
                                stop_loss: float,
                                side: str,
                                ratio: float) -> float:
        """
        חישוב take profit מבוסס risk/reward ratio
        
        Args:
            entry_price: מחיר כניסה
            stop_loss: מחיר stop loss
            side: 'long' או 'short'
            ratio: יחס risk/reward
            
        Returns:
            מחיר take profit
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * ratio
        
        if side == 'long':
            take_profit = entry_price + reward
        else:  # short
            take_profit = entry_price - reward
        
        return take_profit
    
    def calculate_trailing_stop(self,
                                 current_price: float,
                                 highest_price: float,
                                 side: str,
                                 trailing_percent: float = 0.05) -> float:
        """
        חישוב trailing stop loss
        
        Args:
            current_price: מחיר נוכחי
            highest_price: מחיר הגבוה ביותר מאז הכניסה (long) או הנמוך ביותר (short)
            side: 'long' או 'short'
            trailing_percent: אחוז trailing
            
        Returns:
            מחיר trailing stop
        """
        if side == 'long':
            # trailing stop נמצא מתחת למחיר הגבוה ביותר
            trailing_stop = highest_price * (1 - trailing_percent)
        else:  # short
            # trailing stop נמצא מעל למחיר הנמוך ביותר
            trailing_stop = highest_price * (1 + trailing_percent)
        
        return trailing_stop
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        חישוב ATR (Average True Range)
        
        Args:
            data: DataFrame עם high, low, close
            period: תקופה
            
        Returns:
            Series של ATR
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # חישוב True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = moving average של TR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def should_trigger_stop(self,
                            current_price: float,
                            stop_loss: float,
                            side: str) -> bool:
        """
        בדיקה האם צריך להפעיל stop loss
        
        Args:
            current_price: מחיר נוכחי
            stop_loss: מחיר stop loss
            side: 'long' או 'short'
            
        Returns:
            True אם צריך לסגור את הפוזיציה
        """
        if side == 'long':
            return current_price <= stop_loss
        else:  # short
            return current_price >= stop_loss
    
    def should_trigger_take_profit(self,
                                    current_price: float,
                                    take_profit: float,
                                    side: str) -> bool:
        """
        בדיקה האם צריך להפעיל take profit
        
        Args:
            current_price: מחיר נוכחי
            take_profit: מחיר take profit
            side: 'long' או 'short'
            
        Returns:
            True אם צריך לסגור את הפוזיציה
        """
        if side == 'long':
            return current_price >= take_profit
        else:  # short
            return current_price <= take_profit
    
    def calculate_risk_reward_ratio(self,
                                     entry_price: float,
                                     stop_loss: float,
                                     take_profit: float) -> float:
        """
        חישוב יחס risk/reward
        
        Args:
            entry_price: מחיר כניסה
            stop_loss: מחיר stop loss
            take_profit: מחיר take profit
            
        Returns:
            יחס risk/reward
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return float('inf')
        
        return reward / risk
    
    def adjust_stop_loss(self,
                         current_stop: float,
                         new_high_or_low: float,
                         side: str,
                         method: str = 'trailing',
                         trailing_percent: float = 0.05) -> float:
        """
        עדכון stop loss (למשל, trailing stop)
        
        Args:
            current_stop: stop loss נוכחי
            new_high_or_low: מחיר high חדש (long) או low חדש (short)
            side: 'long' או 'short'
            method: 'trailing' או 'none'
            trailing_percent: אחוז trailing
            
        Returns:
            stop loss מעודכן
        """
        if method != 'trailing':
            return current_stop
        
        if side == 'long':
            # trailing stop רק עולה, אף פעם לא יורד
            new_stop = new_high_or_low * (1 - trailing_percent)
            return max(current_stop, new_stop)
        else:  # short
            # trailing stop רק יורד, אף פעם לא עולה
            new_stop = new_high_or_low * (1 + trailing_percent)
            return min(current_stop, new_stop)
    
    def get_stop_summary(self, entry_price: float, stop_loss: float, take_profit: float, shares: int) -> Dict:
        """
        קבלת סיכום של הסיכון והרווח הפוטנציאלי
        
        Args:
            entry_price: מחיר כניסה
            stop_loss: מחיר stop loss
            take_profit: מחיר take profit
            shares: מספר מניות
            
        Returns:
            מילון עם סיכום
        """
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(take_profit - entry_price)
        
        total_risk = risk_per_share * shares
        total_reward = reward_per_share * shares
        
        risk_reward_ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'shares': shares,
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'total_risk': total_risk,
            'total_reward': total_reward,
            'risk_reward_ratio': risk_reward_ratio,
            'risk_percent': (risk_per_share / entry_price) * 100,
            'reward_percent': (reward_per_share / entry_price) * 100
        }

