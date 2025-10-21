"""
מנוע Backtesting מתקדם

מאפשר להריץ backtests על נתונים היסטוריים עם תמיכה מלאה בעמלות, slippage ומטריקות ביצועים.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    מנוע Backtesting מתקדם
    
    תכונות:
    - תמיכה בעמלות ו-slippage
    - מטריקות ביצועים מקיפות
    - ויזואליזציה של תוצאות
    - Walk-forward optimization support
    
    Example:
        >>> engine = BacktestEngine(initial_capital=100000, commission=0.001)
        >>> results = engine.run(data, signals)
        >>> print(f"Total Return: {results['total_return']:.2f}%")
        >>> engine.plot_results()
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        אתחול
        
        Args:
            initial_capital: הון התחלתי
            commission: עמלות (0.1% = 0.001)
            slippage: slippage (0.05% = 0.0005)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.results: Optional[Dict] = None
        self.trades: List[Dict] = []
        self.portfolio_history: Optional[pd.DataFrame] = None
        
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f}")
    
    def run(self,
            data: pd.DataFrame,
            signals: pd.Series,
            position_size: float = 1.0,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None) -> Dict:
        """
        הרצת Backtest
        
        Args:
            data: נתוני שוק (צריך עמודות: open, high, low, close, volume)
            signals: אותות מסחר (1: קנה, -1: מכור, 0: החזק)
            position_size: גודל פוזיציה (0-1, אחוז מההון)
            stop_loss: אחוז stop loss (0.05 = 5%)
            take_profit: אחוז take profit (0.10 = 10%)
            
        Returns:
            מילון עם תוצאות הBacktest
        """
        logger.info("Starting backtest...")
        
        df = data.copy()
        df['signal'] = signals
        
        # אתחול משתנים
        capital = self.initial_capital
        position = 0  # מספר מניות בפוזיציה
        entry_price = 0
        entry_date = None
        stop_loss_price = None
        take_profit_price = None
        
        portfolio_values = []
        cash_history = []
        position_history = []
        
        self.trades = []
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # בדיקת stop loss ו-take profit
            if position > 0:
                hit_stop_loss = stop_loss_price and current_price <= stop_loss_price
                hit_take_profit = take_profit_price and current_price >= take_profit_price
                
                if hit_stop_loss or hit_take_profit:
                    signal = -1  # מכירה מאולצת
                    reason = "stop_loss" if hit_stop_loss else "take_profit"
                    logger.debug(f"{reason.upper()} hit at {current_date}: ${current_price:.2f}")
            
            # חישוב ערך תיק נוכחי
            current_portfolio_value = capital + (position * current_price)
            portfolio_values.append(current_portfolio_value)
            cash_history.append(capital)
            position_history.append(position * current_price if position > 0 else 0)
            
            # ביצוע פעולות מסחר
            if signal == 1 and position == 0:  # אות קנייה
                # חישוב מספר מניות
                available_capital = capital * position_size
                shares = int(available_capital / current_price)
                
                if shares > 0:
                    cost = shares * current_price
                    cost_with_fees = cost * (1 + self.commission + self.slippage)
                    
                    if cost_with_fees <= capital:
                        position = shares
                        entry_price = current_price * (1 + self.slippage)  # מחיר כניסה עם slippage
                        entry_date = current_date
                        capital -= cost_with_fees
                        
                        # קביעת stop loss ו-take profit
                        if stop_loss:
                            stop_loss_price = entry_price * (1 - stop_loss)
                        if take_profit:
                            take_profit_price = entry_price * (1 + take_profit)
                        
                        logger.debug(f"BUY: {shares} shares @ ${entry_price:.2f} on {current_date}")
            
            elif signal == -1 and position > 0:  # אות מכירה
                exit_price = current_price * (1 - self.slippage)  # מחיר יציאה עם slippage
                revenue = position * exit_price
                revenue_after_fees = revenue * (1 - self.commission)
                
                profit = revenue_after_fees - (position * entry_price * (1 + self.commission))
                capital += revenue_after_fees
                
                # שמירת עסקה
                trade = {
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position,
                    'profit': profit,
                    'return': (exit_price / entry_price - 1) * 100,
                    'duration_days': (current_date - entry_date).days if entry_date else 0,
                    'capital_after': capital
                }
                self.trades.append(trade)
                
                logger.debug(
                    f"SELL: {position} shares @ ${exit_price:.2f} on {current_date} | "
                    f"Profit: ${profit:.2f}"
                )
                
                # איפוס פוזיציה
                position = 0
                entry_price = 0
                entry_date = None
                stop_loss_price = None
                take_profit_price = None
        
        # שמירת היסטוריית תיק
        self.portfolio_history = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'cash': cash_history,
            'positions_value': position_history
        }, index=df.index)
        
        # חישוב מטריקות
        self.results = self._calculate_metrics(df)
        
        logger.info(
            f"Backtest completed: {len(self.trades)} trades, "
            f"Return: {self.results['total_return']:.2f}%"
        )
        
        return self.results
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """חישוב מדדי ביצועים מקיפים"""
        portfolio_values = self.portfolio_history['portfolio_value'].values
        final_value = portfolio_values[-1]
        
        # תשואה כוללת
        total_return = ((final_value / self.initial_capital) - 1) * 100
        
        # חישוב תשואות יומיות
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Sharpe Ratio (מניח 252 ימי מסחר בשנה)
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio (רק downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # מטריקות עסקאות
        if self.trades:
            winning_trades = [t for t in self.trades if t['profit'] > 0]
            losing_trades = [t for t in self.trades if t['profit'] < 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100
            
            avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
            
            # Profit Factor
            gross_profit = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average return per trade
            avg_return = np.mean([t['return'] for t in self.trades])
            
            # Average trade duration
            avg_duration = np.mean([t['duration_days'] for t in self.trades])
            
            # Expectancy
            win_prob = len(winning_trades) / len(self.trades)
            loss_prob = len(losing_trades) / len(self.trades)
            expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_return = 0
            avg_duration = 0
            expectancy = 0
        
        # Calculate Calmar Ratio
        if max_drawdown != 0:
            calmar = abs(total_return / max_drawdown)
        else:
            calmar = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['profit'] > 0]),
            'losing_trades': len([t for t in self.trades if t['profit'] < 0]),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_return_per_trade': avg_return,
            'avg_trade_duration_days': avg_duration,
            'expectancy': expectancy,
            'final_capital': final_value,
            'initial_capital': self.initial_capital,
            'portfolio_values': portfolio_values
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 12)):
        """
        ציור תוצאות הBacktest
        
        Args:
            figsize: גודל הפיגורה
        """
        if self.results is None:
            raise ValueError("No results to plot. Run backtest first.")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        axes[0, 0].plot(self.portfolio_history.index, self.portfolio_history['portfolio_value'],
                        label='Portfolio Value', linewidth=2, color='#2E86AB')
        axes[0, 0].axhline(y=self.initial_capital, color='gray', linestyle='--', 
                          label='Initial Capital', alpha=0.7)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Drawdown
        returns = self.portfolio_history['portfolio_value'].pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        axes[0, 1].set_title('Drawdown', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Returns Distribution
        returns_clean = returns.dropna()
        axes[1, 0].hist(returns_clean, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
        axes[1, 0].axvline(returns_clean.mean(), color='red', linestyle='--', 
                          label=f'Mean: {returns_clean.mean()*100:.3f}%')
        axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trade P&L
        if self.trades:
            trade_pnls = [t['profit'] for t in self.trades]
            colors = ['green' if p > 0 else 'red' for p in trade_pnls]
            axes[1, 1].bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.6)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            axes[1, 1].set_title('Trade P&L', fontweight='bold')
            axes[1, 1].set_xlabel('Trade #')
            axes[1, 1].set_ylabel('Profit/Loss ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Cumulative Returns
        cumulative_returns = (1 + returns).cumprod() - 1
        axes[2, 0].plot(cumulative_returns.index, cumulative_returns * 100, 
                       linewidth=2, color='#F18F01')
        axes[2, 0].fill_between(cumulative_returns.index, 0, cumulative_returns * 100, 
                               alpha=0.2, color='#F18F01')
        axes[2, 0].set_title('Cumulative Returns', fontweight='bold')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Cumulative Return (%)')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. Performance Metrics Table
        axes[2, 1].axis('off')
        metrics_text = f"""
        Performance Metrics
        {'='*40}
        Total Return: {self.results['total_return']:.2f}%
        Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
        Sortino Ratio: {self.results['sortino_ratio']:.2f}
        Max Drawdown: {self.results['max_drawdown']:.2f}%
        Calmar Ratio: {self.results['calmar_ratio']:.2f}
        
        Win Rate: {self.results['win_rate']:.2f}%
        Profit Factor: {self.results['profit_factor']:.2f}
        Total Trades: {self.results['total_trades']}
        Avg Win: ${self.results['avg_win']:.2f}
        Avg Loss: ${self.results['avg_loss']:.2f}
        Expectancy: ${self.results['expectancy']:.2f}
        
        Final Capital: ${self.results['final_capital']:,.2f}
        Initial Capital: ${self.results['initial_capital']:,.2f}
        """
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        קבלת DataFrame עם כל העסקאות
        
        Returns:
            DataFrame עם עסקאות
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def print_summary(self):
        """הדפסת סיכום תוצאות"""
        if self.results is None:
            print("No results available. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital:        ${self.results['initial_capital']:,.2f}")
        print(f"Final Capital:          ${self.results['final_capital']:,.2f}")
        print(f"Total Return:           {self.results['total_return']:.2f}%")
        print(f"Max Drawdown:           {self.results['max_drawdown']:.2f}%")
        print(f"\nSharpe Ratio:           {self.results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:          {self.results['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:           {self.results['calmar_ratio']:.2f}")
        print(f"\nTotal Trades:           {self.results['total_trades']}")
        print(f"Winning Trades:         {self.results['winning_trades']}")
        print(f"Losing Trades:          {self.results['losing_trades']}")
        print(f"Win Rate:               {self.results['win_rate']:.2f}%")
        print(f"Profit Factor:          {self.results['profit_factor']:.2f}")
        print(f"\nAverage Win:            ${self.results['avg_win']:.2f}")
        print(f"Average Loss:           ${self.results['avg_loss']:.2f}")
        print(f"Expectancy:             ${self.results['expectancy']:.2f}")
        print(f"Avg Trade Duration:     {self.results['avg_trade_duration_days']:.1f} days")
        print("="*60 + "\n")

