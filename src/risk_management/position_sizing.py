"""
Position Sizing - קביעת גודל פוזיציות

מודול המשלב מספר שיטות לקביעת גודל פוזיציות בצורה אופטימלית.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .kelly_criterion import KellyCriterion
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    מחלקה לניהול גודל פוזיציות
    
    תומכת במספר שיטות:
    - Kelly Criterion
    - Fixed Fractional
    - Fixed Risk
    - Volatility Based
    - Risk Parity
    
    Example:
        >>> sizer = PositionSizer(method='kelly', risk_per_trade=0.02)
        >>> shares = sizer.calculate(
        ...     account_value=100000,
        ...     entry_price=150.0,
        ...     stop_loss=145.0,
        ...     trades_history=historical_trades
        ... )
    """
    
    def __init__(self,
                 method: str = 'kelly',
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5,
                 kelly_fraction: float = 0.5):
        """
        אתחול
        
        Args:
            method: שיטה ('kelly', 'fixed_fractional', 'fixed_risk', 'volatility_based')
            risk_per_trade: סיכון לעסקה (2% = 0.02)
            max_positions: מספר מקסימלי של פוזיציות
            kelly_fraction: אחוז Kelly (רק לשיטת Kelly)
        """
        valid_methods = ['kelly', 'fixed_fractional', 'fixed_risk', 'volatility_based', 'risk_parity']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        self.method = method
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.kelly = KellyCriterion(kelly_fraction=kelly_fraction)
        
        logger.info(f"PositionSizer initialized (method={method}, risk={risk_per_trade})")
    
    def calculate(self,
                  account_value: float,
                  entry_price: float,
                  stop_loss: Optional[float] = None,
                  trades_history: Optional[List[Dict]] = None,
                  volatility: Optional[float] = None,
                  current_positions: int = 0) -> int:
        """
        חישוב מספר מניות
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            stop_loss: stop loss (נדרש לשיטת fixed_risk)
            trades_history: היסטוריית עסקאות (נדרש לשיטת kelly)
            volatility: volatility (נדרש לשיטת volatility_based)
            current_positions: מספר פוזיציות קיימות
            
        Returns:
            מספר מניות
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        
        if account_value <= 0:
            raise ValueError("account_value must be positive")
        
        # בחירת שיטה
        if self.method == 'kelly':
            if trades_history is None:
                logger.warning("No trades history provided for Kelly method, using fixed_fractional")
                return self._fixed_fractional(account_value, entry_price)
            return self._kelly_method(account_value, entry_price, trades_history, current_positions)
        
        elif self.method == 'fixed_fractional':
            return self._fixed_fractional(account_value, entry_price)
        
        elif self.method == 'fixed_risk':
            if stop_loss is None:
                raise ValueError("stop_loss required for fixed_risk method")
            return self._fixed_risk(account_value, entry_price, stop_loss)
        
        elif self.method == 'volatility_based':
            if volatility is None:
                logger.warning("No volatility provided, using fixed_fractional")
                return self._fixed_fractional(account_value, entry_price)
            return self._volatility_based(account_value, entry_price, volatility)
        
        elif self.method == 'risk_parity':
            if volatility is None:
                logger.warning("No volatility provided for risk_parity, using fixed_fractional")
                return self._fixed_fractional(account_value, entry_price)
            return self._risk_parity(account_value, entry_price, volatility, current_positions)
    
    def _kelly_method(self,
                      account_value: float,
                      entry_price: float,
                      trades_history: List[Dict],
                      current_positions: int = 0) -> int:
        """
        שיטת Kelly Criterion
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            trades_history: היסטוריית עסקאות
            current_positions: פוזיציות קיימות
            
        Returns:
            מספר מניות
        """
        # חישוב Kelly fraction
        kelly_fraction = self.kelly.calculate_from_trades(trades_history)
        
        # התאמה למספר פוזיציות קיימות
        if current_positions > 0:
            # הנחת מתאם של 0.5 בין פוזיציות
            kelly_fraction = self.kelly.adjust_for_correlation(
                kelly_fraction,
                current_positions + 1,
                avg_correlation=0.5
            )
        
        # חישוב גודל פוזיציה
        position_value = self.kelly.calculate_position_size(account_value, kelly_fraction)
        shares = int(position_value / entry_price)
        
        logger.debug(f"Kelly method: {shares} shares (fraction={kelly_fraction:.4f})")
        
        return shares
    
    def _fixed_fractional(self, account_value: float, entry_price: float) -> int:
        """
        שיטת fractional קבוע
        
        משקיעים אחוז קבוע מההון בכל עסקה.
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            
        Returns:
            מספר מניות
        """
        position_value = account_value * self.risk_per_trade
        shares = int(position_value / entry_price)
        
        logger.debug(f"Fixed fractional: {shares} shares ({self.risk_per_trade:.2%} of account)")
        
        return shares
    
    def _fixed_risk(self,
                    account_value: float,
                    entry_price: float,
                    stop_loss: float) -> int:
        """
        שיטה מבוססת סיכון קבוע
        
        מחשבת כמה מניות אפשר לקנות כך שההפסד ב-stop loss יהיה risk_per_trade מההון.
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            stop_loss: מחיר stop loss
            
        Returns:
            מספר מניות
        """
        # סכום הסיכון המקסימלי
        risk_amount = account_value * self.risk_per_trade
        
        # סיכון למניה
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            logger.warning("Risk per share is 0, returning 0")
            return 0
        
        # מספר מניות
        shares = int(risk_amount / risk_per_share)
        
        logger.debug(
            f"Fixed risk: {shares} shares "
            f"(risk=${risk_amount:.2f}, risk/share=${risk_per_share:.2f})"
        )
        
        return shares
    
    def _volatility_based(self,
                          account_value: float,
                          entry_price: float,
                          volatility: float) -> int:
        """
        שיטה מבוססת volatility
        
        ככל ש-volatility גבוה יותר, גודל פוזיציה קטן יותר.
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            volatility: volatility (סטיית תקן שנתית)
            
        Returns:
            מספר מניות
        """
        # התאמת הסיכון ל-volatility
        # volatility גבוה = סיכון מותאם נמוך
        target_volatility = 0.15  # 15% volatility יעד
        volatility_adjustment = target_volatility / volatility if volatility > 0 else 1.0
        
        # הגבלה לטווח סביר
        volatility_adjustment = min(max(volatility_adjustment, 0.1), 3.0)
        
        adjusted_risk = self.risk_per_trade * volatility_adjustment
        position_value = account_value * adjusted_risk
        shares = int(position_value / entry_price)
        
        logger.debug(
            f"Volatility-based: {shares} shares "
            f"(vol={volatility:.2%}, adjustment={volatility_adjustment:.2f})"
        )
        
        return shares
    
    def _risk_parity(self,
                     account_value: float,
                     entry_price: float,
                     volatility: float,
                     current_positions: int) -> int:
        """
        שיטת Risk Parity
        
        מחלקת את הסיכון באופן שווה בין כל הפוזיציות.
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            volatility: volatility
            current_positions: פוזיציות קיימות
            
        Returns:
            מספר מניות
        """
        # חישוב התקציב לפוזיציה זו
        total_budget = account_value * self.risk_per_trade
        budget_per_position = total_budget / max(self.max_positions, 1)
        
        # התאמה ל-volatility
        if volatility > 0:
            target_vol = 0.15  # 15% target
            shares = int((budget_per_position / entry_price) * (target_vol / volatility))
        else:
            shares = int(budget_per_position / entry_price)
        
        logger.debug(
            f"Risk parity: {shares} shares "
            f"(budget/pos=${budget_per_position:.2f}, vol={volatility:.2%})"
        )
        
        return shares
    
    def calculate_position_value(self,
                                  shares: int,
                                  price: float) -> float:
        """
        חישוב ערך פוזיציה בדולרים
        
        Args:
            shares: מספר מניות
            price: מחיר
            
        Returns:
            ערך בדולרים
        """
        return shares * price
    
    def get_max_shares(self,
                       account_value: float,
                       price: float,
                       max_position_pct: float = 0.25) -> int:
        """
        חישוב מספר מניות מקסימלי מותר
        
        Args:
            account_value: ערך חשבון
            price: מחיר מניה
            max_position_pct: אחוז מקסימלי מההון (25% = 0.25)
            
        Returns:
            מספר מניות מקסימלי
        """
        max_value = account_value * max_position_pct
        max_shares = int(max_value / price)
        
        return max_shares
    
    def validate_position_size(self,
                                shares: int,
                                price: float,
                                account_value: float,
                                max_position_pct: float = 0.25) -> Tuple[bool, str]:
        """
        בדיקת תקינות גודל פוזיציה
        
        Args:
            shares: מספר מניות
            price: מחיר
            account_value: ערך חשבון
            max_position_pct: אחוז מקסימלי
            
        Returns:
            Tuple של (is_valid, message)
        """
        if shares <= 0:
            return False, "Number of shares must be positive"
        
        position_value = shares * price
        position_pct = position_value / account_value
        
        if position_pct > max_position_pct:
            return False, f"Position size {position_pct:.2%} exceeds maximum {max_position_pct:.2%}"
        
        if position_value > account_value:
            return False, "Position value exceeds account value"
        
        return True, "Position size is valid"

