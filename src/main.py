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
    ========================================================================
                                                                      
                   AUTOMATED TRADING SYSTEM                          
                                                                      
    ========================================================================
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


def run_production_mode(config: dict):
    """הרצת מצב production - אוטומציה מלאה עם TradingLoop"""
    import asyncio
    from src.broker.ib_connector import IBConnector
    from src.strategies.technical.moving_average import MovingAverageCrossover
    from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
    from src.risk_management.position_sizing import PositionSizer
    from src.monitoring.alert_manager import AlertManager
    from src.execution.order_executor import OrderExecutor
    from src.execution.position_manager import PositionManager
    from src.execution.trading_loop import TradingLoop
    from src.utils.market_hours import MarketHoursValidator

    logger = logging.getLogger(__name__)

    print("\n" + "="*70)
    print("PRODUCTION TRADING MODE")
    print("="*70)

    print("\n⚠️  AUTOMATED TRADING - FINAL WARNING ⚠️")
    print("="*70)
    print("This mode will run continuously and execute trades automatically!")
    print("• Ensure you have tested thoroughly with Paper Trading")
    print("• Monitor the system closely, especially in the first hours")
    print("• Daily loss limits and stop-losses will be enforced")
    print("• Press Ctrl+C to stop gracefully")
    print("="*70)

    # Get execution config
    exec_config = config.get('execution', {})
    broker_config = config.get('broker', {})

    print(f"\nConfiguration:")
    print(f"  Broker: {broker_config.get('host', '127.0.0.1')}:{broker_config.get('port', 7497)}")
    print(f"  Symbols: {exec_config.get('symbols', ['AAPL'])}")
    print(f"  Max Positions: {exec_config.get('max_positions', 5)}")
    print(f"  Max Position Value: ${exec_config.get('max_position_value', 10000):,}")
    print(f"  Stop Loss: {exec_config.get('stop_loss_pct', 0.05)*100}%")
    print(f"  Max Daily Loss: ${exec_config.get('max_daily_loss', 1000):,}")
    print(f"  Update Interval: {exec_config.get('update_interval', 60)}s")
    print(f"  Dry Run: {exec_config.get('dry_run', True)}")
    print()

    # Confirmation
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    print("\nStarting automated trading system...")
    print("(Press Ctrl+C to stop)\n")

    async def run_async():
        # Initialize components
        broker = IBConnector(
            host=broker_config.get('host', '127.0.0.1'),
            port=broker_config.get('port', 7497),
            is_paper=broker_config.get('is_paper', True)
        )

        if not broker.connect():
            logger.error("Failed to connect to broker")
            return

        # Create strategies
        symbols = exec_config.get('symbols', ['AAPL'])
        strategies = {}
        for symbol in symbols:
            strategies[symbol] = [
                MovingAverageCrossover(short_window=20, long_window=50),
                RSI_MACD_Strategy()
            ]

        # Initialize components
        risk_manager = PositionSizer(
            account_value=config['trading'].get('initial_capital', 100000)
        )

        alert_manager = AlertManager()
        market_validator = MarketHoursValidator()

        position_manager = PositionManager(
            broker=broker,
            max_positions=exec_config.get('max_positions', 5),
            alert_manager=alert_manager,
            enable_trailing_stops=True,
            trailing_stop_pct=exec_config.get('trailing_stop_pct', 0.05)
        )

        executor = OrderExecutor(
            broker=broker,
            risk_manager=risk_manager,
            alert_manager=alert_manager,
            max_position_value=exec_config.get('max_position_value', 10000.0),
            max_positions=exec_config.get('max_positions', 5),
            enable_stop_loss=True,
            stop_loss_pct=exec_config.get('stop_loss_pct', 0.05),
            dry_run=exec_config.get('dry_run', True)
        )

        # Create and start trading loop
        loop = TradingLoop(
            broker=broker,
            strategies=strategies,
            executor=executor,
            position_manager=position_manager,
            alert_manager=alert_manager,
            market_hours_validator=market_validator,
            data_buffer_size=exec_config.get('data_buffer_size', 200),
            update_interval=exec_config.get('update_interval', 60),
            use_realtime_bars=True,
            enable_trading=exec_config.get('enable_trading', True),
            max_daily_loss=exec_config.get('max_daily_loss', 1000.0)
        )

        try:
            await loop.start()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            loop.stop()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise

    # Run the async loop
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")


def main():
    """פונקציה ראשית"""
    # פרמטרים מ-command line
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--mode',
                        choices=['backtest', 'live', 'production', 'info'],
                        default='info',
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
        print("\nTo run live trading (manual):")
        print("  python src/main.py --mode live")
        print("\nTo run automated production trading:")
        print("  python src/main.py --mode production")
        print("\nFor quick demo:")
        print("  python demo_simple.py")
        print("\nFor examples:")
        print("  python examples/simple_backtest.py")
        print("\nFor full automation:")
        print("  python production_trader.py")
        print("\nFor full documentation:")
        print("  See README.md and workplan.md")
        print("="*70)

    elif args.mode == 'backtest':
        run_backtest_mode(config)

    elif args.mode == 'live':
        run_live_mode(config)

    elif args.mode == 'production':
        run_production_mode(config)

    logger.info(f"Trading System started in {args.mode} mode")


if __name__ == '__main__':
    main()

