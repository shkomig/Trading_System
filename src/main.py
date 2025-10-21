"""
Trading System - Main Entry Point

נקודת הכניסה הראשית למערכת המסחר האוטומטית.
"""

import sys
import os
import yaml
import logging
from pathlib import Path
import argparse

# הוספת נתיב המערכת
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.strategy_registry import get_registry, list_available_strategies
from src.backtesting.backtest_engine import BacktestEngine
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.kelly_criterion import KellyCriterion
from src.learning.performance_tracker import PerformanceTracker
from src.data.database import TradingDatabase

# הגדרת logging
def setup_logging(log_level: str = 'INFO'):
    """הגדרת logging למערכת"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # יצירת תיקיית logs אם לא קיימת
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler('logs/trading_system.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """
    טעינת קובץ הגדרות
    
    Args:
        config_path: נתיב לקובץ הגדרות
        
    Returns:
        מילון הגדרות
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """קבלת הגדרות ברירת מחדל"""
    return {
        'trading': {
            'max_positions': 5,
            'default_position_size': 0.2,
            'commission': 0.001,
            'slippage': 0.0005,
            'initial_capital': 100000
        },
        'risk_management': {
            'risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'kelly_fraction': 0.5,
            'max_leverage': 1.0
        },
        'strategies': {
            'enabled': ['ma_crossover', 'rsi_macd']
        },
        'logging': {
            'level': 'INFO'
        }
    }


def print_banner():
    """הדפסת באנר למערכת"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              📈 AUTOMATED TRADING SYSTEM 📈                          ║
    ║                                                                      ║
    ║              מערכת מסחר אוטומטית מתקדמת                             ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info(config: dict):
    """הדפסת מידע על המערכת"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    # אסטרטגיות זמינות
    strategies = list_available_strategies()
    print(f"\nAvailable Strategies ({len(strategies)}):")
    for strategy in strategies:
        print(f"  • {strategy}")
    
    # הגדרות מסחר
    print(f"\nTrading Configuration:")
    print(f"  Initial Capital:      ${config['trading']['initial_capital']:,}")
    print(f"  Max Positions:        {config['trading']['max_positions']}")
    print(f"  Position Size:        {config['trading']['default_position_size']*100:.1f}%")
    print(f"  Commission:           {config['trading']['commission']*100:.2f}%")
    
    # ניהול סיכונים
    print(f"\nRisk Management:")
    print(f"  Risk per Trade:       {config['risk_management']['risk_per_trade']*100:.1f}%")
    print(f"  Max Daily Loss:       {config['risk_management']['max_daily_loss']*100:.1f}%")
    print(f"  Kelly Fraction:       {config['risk_management']['kelly_fraction']}")
    
    print("="*70)


def run_backtest_mode(config: dict):
    """הרצת מצב backtest"""
    print("\n" + "="*70)
    print("BACKTEST MODE")
    print("="*70)
    
    print("\nThis mode allows you to test strategies on historical data.")
    print("For a complete backtest example, run: python examples/simple_backtest.py")
    
    # הצגת דוגמה קצרה
    print("\nExample:")
    print(">>> from src.strategies.technical.moving_average import MovingAverageCrossover")
    print(">>> strategy = MovingAverageCrossover(short_window=50, long_window=200)")
    print(">>> # Load your data...")
    print(">>> signals = strategy.generate_signals(data)")
    print(">>> engine = BacktestEngine(initial_capital=100000)")
    print(">>> results = engine.run(data, signals)")
    print(">>> engine.print_summary()")


def run_live_mode(config: dict):
    """הרצת מצב live (paper או real)"""
    print("\n" + "="*70)
    print("LIVE TRADING MODE")
    print("="*70)
    
    print("\n⚠️  IMPORTANT SAFETY NOTICE ⚠️")
    print("="*70)
    print("Live trading involves real financial risk!")
    print("• Always start with Paper Trading")
    print("• Test thoroughly before using real money")
    print("• Use proper risk management")
    print("• Monitor your positions regularly")
    print("="*70)
    
    print("\nTo connect to Interactive Brokers:")
    print("1. Open TWS or IB Gateway")
    print("2. Enable API connections in settings")
    print("3. Use port 7497 for Paper Trading")
    print("4. Use port 7496 for Live Trading")
    
    print("\nExample:")
    print(">>> from src.broker.ib_connector import IBConnector")
    print(">>> connector = IBConnector(host='127.0.0.1', port=7497, is_paper=True)")
    print(">>> connector.connect()")
    print(">>> # Your trading logic here...")
    print(">>> connector.disconnect()")


def main():
    """פונקציה ראשית"""
    # פרמטרים מ-command line
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live', 'info'], default='info',
                        help='Operation mode')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # הגדרת logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # טעינת הגדרות
    config = load_config(args.config)
    
    # הדפסת באנר
    print_banner()
    
    # הצגת מידע על המערכת
    if args.mode == 'info':
        print_system_info(config)
        
        print("\n" + "="*70)
        print("USAGE")
        print("="*70)
        print("\nTo run backtest:")
        print("  python src/main.py --mode backtest")
        print("\nTo run live trading:")
        print("  python src/main.py --mode live")
        print("\nFor examples:")
        print("  python examples/simple_backtest.py")
        print("\nFor full documentation:")
        print("  See README.md")
        print("="*70)
    
    elif args.mode == 'backtest':
        run_backtest_mode(config)
    
    elif args.mode == 'live':
        run_live_mode(config)
    
    logger.info(f"Trading System started in {args.mode} mode")


if __name__ == '__main__':
    main()

