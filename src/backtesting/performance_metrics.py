"""
חישוב מטריקות ביצועים למערכת מסחר

פונקציות ומחלקות לחישוב מטריקות ביצועים שונות.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    מחשבון מטריקות ביצועים
    
    כולל חישובים של Sharpe, Sortino, Drawdown ועוד.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        חישוב Sharpe Ratio
        
        Args:
            returns: Series של תשואות
            risk_free_rate: שיעור ריבית חסרת סיכון (שנתי)
            periods_per_year: מספר תקופות בשנה (252 לימים, 52 לשבועות)
            
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        חישוב Sortino Ratio
        
        דומה ל-Sharpe אבל משתמש רק ב-downside volatility.
        
        Args:
            returns: Series של תשואות
            risk_free_rate: שיעור ריבית חסרת סיכון (שנתי)
            periods_per_year: מספר תקופות בשנה
            
        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, int, int]:
        """
        חישוב Max Drawdown
        
        Args:
            returns: Series של תשואות
            
        Returns:
            Tuple של (max_drawdown, start_idx, end_idx)
        """
        if len(returns) == 0:
            return 0.0, 0, 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        end_idx = drawdown.idxmin()
        start_idx = cumulative[:end_idx].idxmax()
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        חישוב Calmar Ratio
        
        תשואה שנתית / Max Drawdown
        
        Args:
            returns: Series של תשואות
            periods_per_year: מספר תקופות בשנה
            
        Returns:
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = PerformanceCalculator.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        calmar = abs(annual_return / max_dd)
        
        return calmar
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        חישוב Value at Risk (VaR)
        
        Args:
            returns: Series של תשואות
            confidence_level: רמת ביטחון (0.95 = 95%)
            
        Returns:
            VaR
        """
        if len(returns) == 0:
            return 0.0
        
        var = returns.quantile(1 - confidence_level)
        
        return var
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        חישוב Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Series של תשואות
            confidence_level: רמת ביטחון
            
        Returns:
            CVaR
        """
        if len(returns) == 0:
            return 0.0
        
        var = PerformanceCalculator.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """
        חישוב Win Rate
        
        Args:
            trades: רשימת עסקאות
            
        Returns:
            Win Rate (%)
        """
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
        win_rate = (winning_trades / len(trades)) * 100
        
        return win_rate
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """
        חישוב Profit Factor
        
        Args:
            trades: רשימת עסקאות
            
        Returns:
            Profit Factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss
        
        return profit_factor
    
    @staticmethod
    def calculate_expectancy(trades: List[Dict]) -> float:
        """
        חישוב Expectancy
        
        תוחלת רווח לעסקה.
        
        Args:
            trades: רשימת עסקאות
            
        Returns:
            Expectancy
        """
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) < 0]
        
        win_prob = len(winning_trades) / len(trades)
        loss_prob = len(losing_trades) / len(trades)
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)
        
        return expectancy
    
    @staticmethod
    def calculate_r_squared(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> float:
        """
        חישוב R-Squared
        
        Args:
            returns: תשואות האסטרטגיה
            benchmark_returns: תשואות benchmark (None = linear trend)
            
        Returns:
            R-Squared (0-1)
        """
        if len(returns) < 2:
            return 0.0
        
        if benchmark_returns is None:
            # השוואה לטרנד ליניארי
            x = np.arange(len(returns))
            benchmark_returns = pd.Series(np.polyval(np.polyfit(x, returns.values, 1), x))
        
        # חישוב R²
        ss_res = ((returns - benchmark_returns) ** 2).sum()
        ss_tot = ((returns - returns.mean()) ** 2).sum()
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        return max(0.0, min(1.0, r_squared))
    
    @staticmethod
    def calculate_alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Tuple[float, float]:
        """
        חישוב Alpha ו-Beta
        
        Args:
            returns: תשואות האסטרטגיה
            benchmark_returns: תשואות benchmark
            risk_free_rate: שיעור ריבית חסרת סיכון
            
        Returns:
            Tuple של (alpha, beta)
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0
        
        # ודא שהאורך זהה
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # חישוב Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        
        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance
        
        # חישוב Alpha
        excess_return = returns.mean() - (risk_free_rate / 252)
        benchmark_excess = benchmark_returns.mean() - (risk_free_rate / 252)
        alpha = excess_return - (beta * benchmark_excess)
        
        return alpha * 252, beta  # Alpha שנתי
    
    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        """
        חישוב Ulcer Index
        
        מדד למדידת עומק ומשך של drawdowns.
        
        Args:
            returns: Series של תשואות
            
        Returns:
            Ulcer Index
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        
        ulcer = np.sqrt((drawdown ** 2).mean())
        
        return ulcer
    
    @staticmethod
    def calculate_recovery_factor(
        total_return: float,
        max_drawdown: float
    ) -> float:
        """
        חישוב Recovery Factor
        
        Net Profit / Max Drawdown
        
        Args:
            total_return: תשואה כוללת
            max_drawdown: Max Drawdown
            
        Returns:
            Recovery Factor
        """
        if max_drawdown == 0:
            return 0.0
        
        recovery_factor = abs(total_return / max_drawdown)
        
        return recovery_factor
    
    @staticmethod
    def calculate_information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        חישוב Information Ratio
        
        Args:
            returns: תשואות האסטרטגיה
            benchmark_returns: תשואות benchmark
            
        Returns:
            Information Ratio
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        # ודא שהאורך זהה
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252)
        
        return information_ratio
    
    @staticmethod
    def generate_full_report(
        returns: pd.Series,
        trades: List[Dict],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        יצירת דוח ביצועים מלא
        
        Args:
            returns: תשואות
            trades: עסקאות
            initial_capital: הון התחלתי
            benchmark_returns: תשואות benchmark (אופציונלי)
            
        Returns:
            מילון עם כל המטריקות
        """
        calc = PerformanceCalculator()
        
        # תשואות ורווחיות
        total_return = ((1 + returns).prod() - 1) * 100
        final_capital = initial_capital * (1 + returns).prod()
        
        # סיכון
        sharpe = calc.calculate_sharpe_ratio(returns)
        sortino = calc.calculate_sortino_ratio(returns)
        max_dd, _, _ = calc.calculate_max_drawdown(returns)
        max_dd_pct = max_dd * 100
        calmar = calc.calculate_calmar_ratio(returns)
        
        # מטריקות עסקאות
        win_rate = calc.calculate_win_rate(trades)
        profit_factor = calc.calculate_profit_factor(trades)
        expectancy = calc.calculate_expectancy(trades)
        
        # סיכון מתקדם
        var_95 = calc.calculate_var(returns, 0.95) * 100
        cvar_95 = calc.calculate_cvar(returns, 0.95) * 100
        ulcer = calc.calculate_ulcer_index(returns)
        recovery = calc.calculate_recovery_factor(total_return, max_dd_pct)
        
        report = {
            # תשואות
            'total_return': total_return,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'net_profit': final_capital - initial_capital,
            
            # סיכון
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd_pct,
            'calmar_ratio': calmar,
            'volatility': returns.std() * np.sqrt(252) * 100,
            
            # עסקאות
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            
            # סיכון מתקדם
            'var_95': var_95,
            'cvar_95': cvar_95,
            'ulcer_index': ulcer,
            'recovery_factor': recovery,
        }
        
        # מטריקות benchmark
        if benchmark_returns is not None:
            alpha, beta = calc.calculate_alpha_beta(returns, benchmark_returns)
            info_ratio = calc.calculate_information_ratio(returns, benchmark_returns)
            
            report['alpha'] = alpha
            report['beta'] = beta
            report['information_ratio'] = info_ratio
        
        return report

