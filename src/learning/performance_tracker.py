"""
Performance Tracker - מעקב וניתוח ביצועים

מערכת שעוקבת אחר כל עסקה ולומדת מהביצועים.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    מערכת מעקב וניתוח ביצועים
    
    לומדת מהצלחות והפסדים וממליצה על שיפורים.
    
    Example:
        >>> tracker = PerformanceTracker()
        >>> tracker.log_trade({
        ...     'symbol': 'AAPL',
        ...     'strategy': 'MA_Crossover',
        ...     'pnl': 150.0,
        ...     'entry_price': 145.0,
        ...     'exit_price': 148.5
        ... })
        >>> recommendations = tracker.get_strategy_recommendations()
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        אתחול
        
        Args:
            db_path: נתיב למסד נתונים (אופציונלי)
        """
        self.db_path = db_path
        self.trades: List[Dict] = []
        self.daily_pnl: List[Dict] = []
        self.strategy_performance: Dict[str, Dict] = {}
        
        logger.info("PerformanceTracker initialized")
    
    def log_trade(self, trade: Dict):
        """
        רישום עסקה
        
        Args:
            trade: מילון עם פרטי עסקה
                - symbol: סימול
                - strategy: שם אסטרטגיה
                - action: BUY/SELL
                - quantity: כמות
                - entry_price: מחיר כניסה
                - exit_price: מחיר יציאה
                - pnl: רווח/הפסד
                - entry_time: זמן כניסה
                - exit_time: זמן יציאה
        """
        # הוספת timestamp אם לא קיים
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now().isoformat()
        
        self.trades.append(trade)
        self._update_strategy_stats(trade)
        
        pnl = trade.get('pnl', 0)
        symbol = trade.get('symbol', 'Unknown')
        strategy = trade.get('strategy', 'Unknown')
        
        logger.info(
            f"Trade logged: {strategy} | {symbol} | P&L: ${pnl:.2f} | "
            f"Total trades: {len(self.trades)}"
        )
    
    def _update_strategy_stats(self, trade: Dict):
        """עדכון סטטיסטיקות אסטרטגיה"""
        strategy_name = trade.get('strategy', 'Unknown')
        
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'current_streak': 0,
                'avg_duration_minutes': 0,
                'trades_by_symbol': {}
            }
        
        stats = self.strategy_performance[strategy_name]
        pnl = trade.get('pnl', 0)
        symbol = trade.get('symbol', 'Unknown')
        
        # עדכון מספרים בסיסיים
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        
        # עדכון לפי תוצאה
        if pnl > 0:
            stats['winning_trades'] += 1
            stats['current_streak'] = max(0, stats['current_streak']) + 1
            stats['consecutive_wins'] = max(stats['consecutive_wins'], stats['current_streak'])
            stats['best_trade'] = max(stats['best_trade'], pnl)
        elif pnl < 0:
            stats['losing_trades'] += 1
            stats['current_streak'] = min(0, stats['current_streak']) - 1
            stats['consecutive_losses'] = max(stats['consecutive_losses'], abs(stats['current_streak']))
            stats['worst_trade'] = min(stats['worst_trade'], pnl)
        
        # חישוב win rate
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        
        # חישוב avg win/loss
        strategy_trades = [t for t in self.trades if t.get('strategy') == strategy_name]
        wins = [t['pnl'] for t in strategy_trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in strategy_trades if t.get('pnl', 0) < 0]
        
        stats['avg_win'] = np.mean(wins) if wins else 0
        stats['avg_loss'] = np.mean(losses) if losses else 0
        
        # Profit Factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # עדכון עסקאות לפי סימול
        if symbol not in stats['trades_by_symbol']:
            stats['trades_by_symbol'][symbol] = {'count': 0, 'pnl': 0}
        stats['trades_by_symbol'][symbol]['count'] += 1
        stats['trades_by_symbol'][symbol]['pnl'] += pnl
    
    def get_strategy_recommendations(self) -> Dict[str, str]:
        """
        המלצות לשיפור אסטרטגיות
        
        Returns:
            מילון עם המלצות לכל אסטרטגיה
        """
        recommendations = {}
        
        for strategy_name, stats in self.strategy_performance.items():
            recs = []
            
            # בדיקת win rate
            if stats['win_rate'] < 0.4:
                recs.append("⚠️ Win rate נמוך - שקול לחדד תנאי כניסה או לשפר פילטרים")
            elif stats['win_rate'] > 0.6:
                recs.append("✅ Win rate מצוין - אסטרטגיה חזקה")
            
            # בדיקת profit factor
            if stats['profit_factor'] < 1.0:
                recs.append("🔴 Profit factor < 1 - האסטרטגיה מפסידה כסף!")
            elif stats['profit_factor'] < 1.5:
                recs.append("⚠️ Profit factor נמוך - שפר את יחס Win/Loss")
            elif stats['profit_factor'] > 2.0:
                recs.append("✅ Profit factor מעולה!")
            
            # בדיקת רצפי הפסדים
            if stats['consecutive_losses'] > 5:
                recs.append(f"⚠️ זוהו {stats['consecutive_losses']} הפסדים רצופים - שקול הפסקה זמנית או הפחתת גודל פוזיציה")
            
            # בדיקת התפלגות רווחים/הפסדים
            if abs(stats['worst_trade']) > abs(stats['best_trade']) * 2:
                recs.append("⚠️ הפסדים גדולים יחסית לרווחים - שפר stop-loss או שנה יחס risk/reward")
            
            # בדיקת avg win vs avg loss
            if stats['avg_win'] > 0 and stats['avg_loss'] < 0:
                win_loss_ratio = abs(stats['avg_win'] / stats['avg_loss'])
                if win_loss_ratio < 1.5:
                    recs.append(f"⚠️ יחס Win/Loss: {win_loss_ratio:.2f} - לא אופטימלי")
                elif win_loss_ratio > 2.0:
                    recs.append(f"✅ יחס Win/Loss: {win_loss_ratio:.2f} - מצוין!")
            
            # בדיקת מספר עסקאות
            if stats['total_trades'] < 20:
                recs.append(f"ℹ️ רק {stats['total_trades']} עסקאות - נדרש יותר data לניתוח מהימן")
            
            # אם אין המלצות, הכל בסדר
            if not recs:
                recs.append("✅ ביצועים תקינים - המשך למעקב")
            
            recommendations[strategy_name] = ' | '.join(recs)
        
        return recommendations
    
    def analyze_market_conditions(self) -> Dict:
        """
        ניתוח תנאי שוק אופטימליים
        
        מזהה באילו ימים/שעות/סימולים האסטרטגיות עובדות הכי טוב.
        
        Returns:
            מילון עם תובנות
        """
        if len(self.trades) < 30:
            return {"message": "לא מספיק נתונים לניתוח (נדרשות לפחות 30 עסקאות)"}
        
        df = pd.DataFrame(self.trades)
        
        # המרת timestamp ל-datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ניתוח לפי ימי שבוע
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        pnl_by_day = df.groupby('day_of_week')['pnl'].agg(['mean', 'sum', 'count'])
        
        days_map = {0: 'ראשון', 1: 'שני', 2: 'שלישי', 3: 'רביעי', 4: 'חמישי', 5: 'שישי', 6: 'שבת'}
        
        if len(pnl_by_day) > 0:
            best_day_idx = pnl_by_day['mean'].idxmax()
            worst_day_idx = pnl_by_day['mean'].idxmin()
            best_day = days_map[best_day_idx]
            worst_day = days_map[worst_day_idx]
        else:
            best_day = "לא ידוע"
            worst_day = "לא ידוע"
        
        # ניתוח לפי שעות (אם יש)
        df['hour'] = df['timestamp'].dt.hour
        pnl_by_hour = df.groupby('hour')['pnl'].agg(['mean', 'sum', 'count'])
        
        if len(pnl_by_hour) > 0:
            best_hour = pnl_by_hour['mean'].idxmax()
        else:
            best_hour = "לא ידוע"
        
        # ניתוח לפי סימולים
        if 'symbol' in df.columns:
            pnl_by_symbol = df.groupby('symbol')['pnl'].agg(['mean', 'sum', 'count'])
            if len(pnl_by_symbol) > 0:
                best_symbols = pnl_by_symbol.nlargest(5, 'sum')
                worst_symbols = pnl_by_symbol.nsmallest(3, 'sum')
            else:
                best_symbols = pd.DataFrame()
                worst_symbols = pd.DataFrame()
        else:
            best_symbols = pd.DataFrame()
            worst_symbols = pd.DataFrame()
        
        return {
            'best_trading_day': best_day,
            'worst_trading_day': worst_day,
            'best_trading_hour': f"{best_hour}:00" if best_hour != "לא ידוע" else best_hour,
            'best_symbols': best_symbols.index.tolist() if not best_symbols.empty else [],
            'worst_symbols': worst_symbols.index.tolist() if not worst_symbols.empty else [],
            'total_pnl': df['pnl'].sum(),
            'avg_pnl_per_trade': df['pnl'].mean(),
            'volatility': df['pnl'].std(),
            'total_trades': len(df)
        }
    
    def get_performance_summary(self, strategy_name: Optional[str] = None) -> str:
        """
        קבלת סיכום ביצועים כטקסט
        
        Args:
            strategy_name: שם אסטרטגיה (None = כל האסטרטגיות)
            
        Returns:
            מחרוזת עם סיכום
        """
        if strategy_name:
            if strategy_name not in self.strategy_performance:
                return f"Strategy '{strategy_name}' not found"
            
            strategies_to_show = {strategy_name: self.strategy_performance[strategy_name]}
        else:
            strategies_to_show = self.strategy_performance
        
        summary = []
        summary.append("="*70)
        summary.append("PERFORMANCE SUMMARY")
        summary.append("="*70)
        
        for name, stats in strategies_to_show.items():
            summary.append(f"\nStrategy: {name}")
            summary.append("-"*70)
            summary.append(f"Total Trades:        {stats['total_trades']}")
            summary.append(f"Win Rate:            {stats['win_rate']*100:.2f}%")
            summary.append(f"Total P&L:           ${stats['total_pnl']:.2f}")
            summary.append(f"")
            summary.append(f"Winning Trades:      {stats['winning_trades']}")
            summary.append(f"Average Win:         ${stats['avg_win']:.2f}")
            summary.append(f"Best Trade:          ${stats['best_trade']:.2f}")
            summary.append(f"Max Consecutive Wins: {stats['consecutive_wins']}")
            summary.append(f"")
            summary.append(f"Losing Trades:       {stats['losing_trades']}")
            summary.append(f"Average Loss:        ${stats['avg_loss']:.2f}")
            summary.append(f"Worst Trade:         ${stats['worst_trade']:.2f}")
            summary.append(f"Max Consecutive Losses: {stats['consecutive_losses']}")
            summary.append(f"")
            summary.append(f"Profit Factor:       {stats['profit_factor']:.2f}")
            summary.append("-"*70)
        
        summary.append("="*70)
        
        return "\n".join(summary)
    
    def export_to_csv(self, filename: str):
        """
        ייצוא עסקאות ל-CSV
        
        Args:
            filename: שם קובץ
        """
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {filename}")
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        קבלת DataFrame עם כל העסקאות
        
        Returns:
            DataFrame עם עסקאות
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)

