"""
Kelly Criterion לחישוב גודל פוזיציה אופטימלי

ה-Kelly Criterion הוא נוסחה מתמטית לקביעת הגודל האופטימלי של פוזיציה
בהתבסס על היסטוריית ביצועים.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    יישום Kelly Criterion לניהול גודל פוזיציות
    
    הנוסחה: f* = (p * b - q) / b
    כאשר:
    - p = הסתברות לניצחון
    - q = הסתברות להפסד (1-p)
    - b = יחס win/loss
    - f* = אחוז ההון להשקעה
    
    Example:
        >>> kelly = KellyCriterion(kelly_fraction=0.5)
        >>> trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 150}]
        >>> fraction = kelly.calculate_from_trades(trades)
        >>> print(f"Recommended position size: {fraction*100:.2f}%")
    """
    
    def __init__(self,
                 kelly_fraction: float = 0.5,
                 max_position_size: float = 0.25):
        """
        אתחול
        
        Args:
            kelly_fraction: אחוז Kelly (0.5 = Half Kelly)
            max_position_size: גודל מקסימלי של פוזיציה (25% = 0.25)
        """
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        
        if not 0 < max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        
        logger.info(f"Kelly Criterion initialized (fraction={kelly_fraction}, max_size={max_position_size})")
    
    def calculate_from_trades(self, trades: List[Dict]) -> float:
        """
        חישוב Kelly מתוצאות עסקאות
        
        Args:
            trades: רשימת עסקאות עם 'pnl'
            
        Returns:
            Kelly fraction מומלץ (0-max_position_size)
        """
        if not trades:
            logger.warning("No trades provided, returning 0")
            return 0.0
        
        # הפרדת ניצחונות והפסדים
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if not winning_trades or not losing_trades:
            logger.warning("Need both winning and losing trades for Kelly calculation")
            return 0.0
        
        # חישוב win rate
        win_rate = len(winning_trades) / len(trades)
        loss_rate = 1 - win_rate
        
        # חישוב avg win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
        
        if avg_loss == 0:
            logger.warning("Average loss is 0, returning 0")
            return 0.0
        
        # יחס win/loss
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f* = (p*b - q) / b
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # חישוב fractional Kelly
        kelly_fraction = kelly * self.kelly_fraction
        
        # הגבלה למקסימום
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        logger.info(
            f"Kelly calculated: {kelly:.4f} | "
            f"Fractional: {kelly_fraction:.4f} | "
            f"Win rate: {win_rate:.2%}"
        )
        
        return kelly_fraction
    
    def calculate_from_sharpe(self,
                               mean_return: float,
                               std_return: float,
                               risk_free_rate: float = 0.02) -> float:
        """
        חישוב Kelly מ-Sharpe Ratio
        
        נוסחה: f* = (μ - r) / σ²
        כאשר:
        - μ = תשואה ממוצעת
        - r = ריבית חסרת סיכון
        - σ² = variance
        
        Args:
            mean_return: תשואה ממוצעת שנתית
            std_return: סטיית תקן שנתית
            risk_free_rate: ריבית חסרת סיכון
            
        Returns:
            Kelly fraction מומלץ
        """
        if std_return == 0:
            logger.warning("Standard deviation is 0, returning 0")
            return 0.0
        
        # חישוב excess return
        excess_return = mean_return - risk_free_rate
        
        # Kelly formula
        variance = std_return ** 2
        kelly = excess_return / variance
        
        # חישוב fractional Kelly
        kelly_fraction = kelly * self.kelly_fraction
        
        # הגבלה למקסימום
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        logger.info(f"Kelly from Sharpe calculated: {kelly_fraction:.4f}")
        
        return kelly_fraction
    
    def calculate_from_win_loss(self,
                                 win_rate: float,
                                 avg_win: float,
                                 avg_loss: float) -> float:
        """
        חישוב Kelly מפרמטרים ישירים
        
        Args:
            win_rate: אחוז ניצחונות (0.6 = 60%)
            avg_win: רווח ממוצע
            avg_loss: הפסד ממוצע (ערך חיובי)
            
        Returns:
            Kelly fraction מומלץ
        """
        if avg_loss == 0:
            logger.warning("Average loss is 0, returning 0")
            return 0.0
        
        if not 0 <= win_rate <= 1:
            raise ValueError("win_rate must be between 0 and 1")
        
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # חישוב fractional Kelly
        kelly_fraction = kelly * self.kelly_fraction
        
        # הגבלה למקסימום
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        logger.info(f"Kelly calculated: {kelly_fraction:.4f}")
        
        return kelly_fraction
    
    def calculate_position_size(self,
                                account_value: float,
                                kelly_fraction: float) -> float:
        """
        חישוב גודל פוזיציה בדולרים
        
        Args:
            account_value: ערך חשבון
            kelly_fraction: Kelly fraction
            
        Returns:
            גודל פוזיציה בדולרים
        """
        position_size = account_value * kelly_fraction
        
        logger.debug(f"Position size: ${position_size:.2f} ({kelly_fraction:.2%} of ${account_value:.2f})")
        
        return position_size
    
    def adjust_for_correlation(self,
                                kelly_fraction: float,
                                num_positions: int,
                                avg_correlation: float = 0.5) -> float:
        """
        התאמת Kelly עבור פוזיציות מרובות מתואמות
        
        כאשר יש מספר פוזיציות מתואמות, צריך להקטין את הגודל של כל אחת.
        
        Args:
            kelly_fraction: Kelly fraction בסיסי
            num_positions: מספר פוזיציות
            avg_correlation: מתאם ממוצע בין פוזיציות (0-1)
            
        Returns:
            Kelly fraction מותאם
        """
        if num_positions <= 1:
            return kelly_fraction
        
        # נוסחה להתאמה עבור מתאם
        # f_adjusted = f / (1 + (n-1) * ρ)
        # כאשר n = מספר פוזיציות, ρ = מתאם
        
        adjustment_factor = 1 + (num_positions - 1) * avg_correlation
        adjusted_fraction = kelly_fraction / adjustment_factor
        
        logger.info(
            f"Adjusted Kelly for {num_positions} positions (corr={avg_correlation:.2f}): "
            f"{adjusted_fraction:.4f}"
        )
        
        return max(0, min(adjusted_fraction, self.max_position_size))
    
    def calculate_optimal_f(self, trades: List[Dict]) -> float:
        """
        חישוב Optimal F (Ralph Vince)
        
        גרסה מתקדמת יותר של Kelly שמתחשבת בהתפלגות המלאה של התוצאות.
        
        Args:
            trades: רשימת עסקאות
            
        Returns:
            Optimal F
        """
        if not trades:
            return 0.0
        
        # מציאת ההפסד הגדול ביותר
        losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
        if not losses:
            return 0.0
        
        max_loss = abs(min(losses))
        
        if max_loss == 0:
            return 0.0
        
        # חיפוש F אופטימלי באמצעות maximization של TWR (Terminal Wealth Relative)
        best_f = 0
        best_twr = 0
        
        # חיפוש grid
        for f in np.linspace(0.01, 1.0, 100):
            twr = 1.0
            for trade in trades:
                pnl = trade.get('pnl', 0)
                # הנחה שמשקיעים f * capital על כל עסקה
                twr *= (1 + f * pnl / max_loss)
                
                if twr <= 0:  # פשיטת רגל
                    twr = 0
                    break
            
            if twr > best_twr:
                best_twr = twr
                best_f = f
        
        # החלה של fractional Kelly
        optimal_f = best_f * self.kelly_fraction
        optimal_f = max(0, min(optimal_f, self.max_position_size))
        
        logger.info(f"Optimal F calculated: {optimal_f:.4f}")
        
        return optimal_f

