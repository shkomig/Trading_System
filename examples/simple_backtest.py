"""
דוגמה פשוטה להרצת backtest

דוגמה זו מראה איך להשתמש במערכת לבדיקת אסטרטגיה על נתונים היסטוריים.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ייבוא מודולי המערכת
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
from src.strategies.technical.momentum import MomentumStrategy
from src.backtesting.backtest_engine import BacktestEngine


def generate_sample_data(days=365):
    """
    יצירת נתונים לדוגמה
    
    Args:
        days: מספר ימים
        
    Returns:
        DataFrame עם נתוני OHLCV
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # סימולציה של מחירים עם טרנד ורעש
    base_price = 100
    trend = np.linspace(0, 20, days)
    noise = np.random.randn(days) * 2
    close_prices = base_price + trend + noise
    
    # יצירת OHLC
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(days) * 0.5,
        'high': close_prices + np.abs(np.random.randn(days)) * 1.5,
        'low': close_prices - np.abs(np.random.randn(days)) * 1.5,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    # וידוא שמחירים תקינים
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def run_simple_backtest():
    """הרצת backtest פשוט עם אסטרטגיית MA"""
    print("="*60)
    print("Simple Backtest Example - Moving Average Crossover")
    print("="*60)
    
    # 1. יצירת נתונים
    print("\n1. Generating sample data...")
    data = generate_sample_data(days=365)
    print(f"   Generated {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # 2. יצירת אסטרטגיה
    print("\n2. Creating Moving Average Crossover strategy...")
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    print(f"   Strategy: {strategy}")
    
    # 3. יצירת אותות
    print("\n3. Generating signals...")
    signals = strategy.generate_signals(data)
    num_buy = (signals == 1).sum()
    num_sell = (signals == -1).sum()
    print(f"   Buy signals: {num_buy}")
    print(f"   Sell signals: {num_sell}")
    
    # 4. הרצת backtest
    print("\n4. Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    results = engine.run(data, signals, position_size=0.5)
    
    # 5. הצגת תוצאות
    print("\n5. Backtest Results:")
    print("   " + "-"*56)
    engine.print_summary()
    
    # 6. ציור תוצאות
    print("\n6. Plotting results...")
    try:
        engine.plot_results()
    except Exception as e:
        print(f"   Could not plot results: {e}")
        print("   (Install matplotlib to see plots)")
    
    return results


def compare_strategies():
    """השוואה בין מספר אסטרטגיות"""
    print("\n" + "="*60)
    print("Strategy Comparison Example")
    print("="*60)
    
    # יצירת נתונים
    print("\nGenerating sample data...")
    data = generate_sample_data(days=500)
    
    # אסטרטגיות לבדיקה
    strategies = {
        'MA Crossover': MovingAverageCrossover(short_window=20, long_window=50),
        'RSI + MACD': RSI_MACD_Strategy(rsi_period=14, macd_fast=12, macd_slow=26),
        'Momentum': MomentumStrategy(lookback_period=14, threshold=0.02)
    }
    
    # הרצת backtest לכל אסטרטגיה
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        try:
            signals = strategy.generate_signals(data)
            
            engine = BacktestEngine(initial_capital=100000, commission=0.001)
            result = engine.run(data, signals, position_size=0.3)
            
            results[name] = result
            
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = None
    
    # סיכום השוואה
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        name: {
            'Total Return (%)': res['total_return'] if res else 0,
            'Sharpe Ratio': res['sharpe_ratio'] if res else 0,
            'Max Drawdown (%)': res['max_drawdown'] if res else 0,
            'Win Rate (%)': res['win_rate'] if res else 0,
            'Total Trades': res['total_trades'] if res else 0
        }
        for name, res in results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    # מציאת האסטרטגיה הטובה ביותר
    if comparison_df['Sharpe Ratio'].max() > 0:
        best_strategy = comparison_df['Sharpe Ratio'].idxmax()
        print(f"\n🏆 Best Strategy: {best_strategy}")
        print(f"   Sharpe Ratio: {comparison_df.loc[best_strategy, 'Sharpe Ratio']:.2f}")
    
    return comparison_df


def main():
    """פונקציה ראשית"""
    print("\n" + "="*60)
    print("TRADING SYSTEM - BACKTEST EXAMPLES")
    print("="*60)
    
    # הרצת backtest פשוט
    results = run_simple_backtest()
    
    # השוואת אסטרטגיות
    comparison = compare_strategies()
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Try running with real historical data")
    print("2. Optimize strategy parameters")
    print("3. Test with different time periods")
    print("4. Add risk management (stop-loss, position sizing)")
    print("5. Run paper trading with Interactive Brokers")


if __name__ == '__main__':
    main()

