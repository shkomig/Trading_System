"""
Production Trading System - Main Entry Point

This script implements a fully automated trading system with:
- Real-time data streaming from Interactive Brokers
- Continuous trading loop throughout market hours
- Automated signal execution
- Position and risk management
- Stop-loss and trailing stops
- Daily loss limits

IMPORTANT: Test with dry_run=True and Paper Trading before going live!
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.broker.ib_connector import IBConnector
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
from src.strategies.technical.momentum import MomentumStrategy
from src.risk_management.position_sizing import PositionSizer
from src.monitoring.alert_manager import AlertManager
from src.execution.order_executor import OrderExecutor
from src.execution.position_manager import PositionManager
from src.execution.trading_loop import TradingLoop
from src.utils.market_hours import MarketHoursValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main production trading function

    This sets up and runs the complete automated trading system.
    """

    logger.info("="*70)
    logger.info("PRODUCTION TRADING SYSTEM v2.0")
    logger.info("="*70)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Broker settings
    IB_HOST = '127.0.0.1'
    IB_PORT = 7497  # Paper trading (7496 for live)
    IS_PAPER = True

    # Trading universe
    SYMBOLS = ['AAPL', 'MSFT', 'TSLA']

    # Risk parameters
    MAX_POSITION_VALUE = 10000.0  # Max $10k per position
    MAX_POSITIONS = 5
    STOP_LOSS_PCT = 0.05  # 5% stop-loss
    TRAILING_STOP_PCT = 0.05  # 5% trailing stop
    MAX_DAILY_LOSS = 1000.0  # Max $1000 loss per day

    # Loop settings
    UPDATE_INTERVAL = 60  # Check every 60 seconds
    DATA_BUFFER_SIZE = 200  # Keep 200 bars in memory

    # Safety
    DRY_RUN = True  # SET TO FALSE FOR REAL TRADING!
    ENABLE_TRADING = True  # Master switch

    # ============================================================
    # INITIALIZATION
    # ============================================================

    try:
        # 1. Initialize broker connection
        logger.info("1/8 Initializing broker connection...")
        broker = IBConnector(
            host=IB_HOST,
            port=IB_PORT,
            is_paper=IS_PAPER
        )

        if not broker.connect():
            logger.error("Failed to connect to IB - ensure TWS/Gateway is running")
            return

        logger.info("✓ Connected to IB")

        # Get account info
        account_info = broker.get_account_info()
        if account_info:
            buying_power = account_info.get('BuyingPower', 'N/A')
            logger.info(f"  Buying Power: ${buying_power}")

        # 2. Initialize strategies
        logger.info("2/8 Initializing strategies...")

        ma_strategy = MovingAverageCrossover(short_window=20, long_window=50)
        rsi_strategy = RSI_MACD_Strategy()
        momentum_strategy = MomentumStrategy(lookback_period=20)

        strategies = {
            'AAPL': [ma_strategy, rsi_strategy],
            'MSFT': [ma_strategy],
            'TSLA': [momentum_strategy]
        }

        total_strategies = sum(len(strats) for strats in strategies.values())
        logger.info(f"✓ {total_strategies} strategies configured for {len(SYMBOLS)} symbols")

        # 3. Initialize risk management
        logger.info("3/8 Initializing risk management...")
        risk_manager = PositionSizer(account_value=100000)
        logger.info("✓ Risk manager ready")

        # 4. Initialize alert manager
        logger.info("4/8 Initializing alert manager...")
        alert_manager = AlertManager()
        logger.info("✓ Alert manager ready")

        # 5. Initialize market hours validator
        logger.info("5/8 Initializing market hours validator...")
        market_validator = MarketHoursValidator(
            avoid_first_minutes=10,  # Skip first 10 min
            avoid_last_minutes=10    # Skip last 10 min
        )

        status = market_validator.get_trading_status()
        logger.info(f"  Market Open: {status['is_market_open']}")
        logger.info(f"  Should Trade: {status['should_trade']}")

        # 6. Initialize position manager
        logger.info("6/8 Initializing position manager...")
        position_manager = PositionManager(
            broker=broker,
            max_positions=MAX_POSITIONS,
            alert_manager=alert_manager,
            enable_trailing_stops=True,
            trailing_stop_pct=TRAILING_STOP_PCT
        )
        logger.info("✓ Position manager ready")

        # Sync with existing positions
        position_manager.sync_with_broker()

        # 7. Initialize order executor
        logger.info("7/8 Initializing order executor...")
        executor = OrderExecutor(
            broker=broker,
            risk_manager=risk_manager,
            alert_manager=alert_manager,
            max_position_value=MAX_POSITION_VALUE,
            max_positions=MAX_POSITIONS,
            enable_stop_loss=True,
            stop_loss_pct=STOP_LOSS_PCT,
            execution_timeout=30,
            dry_run=DRY_RUN
        )

        if DRY_RUN:
            logger.warning("⚠️  DRY RUN MODE - Orders will be simulated")
        else:
            logger.info("✓ Order executor ready (LIVE MODE)")

        # 8. Initialize and start trading loop
        logger.info("8/8 Initializing trading loop...")
        loop = TradingLoop(
            broker=broker,
            strategies=strategies,
            executor=executor,
            position_manager=position_manager,
            alert_manager=alert_manager,
            market_hours_validator=market_validator,
            data_buffer_size=DATA_BUFFER_SIZE,
            update_interval=UPDATE_INTERVAL,
            use_realtime_bars=True,
            enable_trading=ENABLE_TRADING,
            max_daily_loss=MAX_DAILY_LOSS
        )

        logger.info("✓ Trading loop initialized")
        logger.info("")
        logger.info("="*70)
        logger.info("STARTING AUTOMATED TRADING")
        logger.info("="*70)
        logger.info(f"Symbols: {', '.join(SYMBOLS)}")
        logger.info(f"Strategies: {total_strategies}")
        logger.info(f"Max Positions: {MAX_POSITIONS}")
        logger.info(f"Max Position Value: ${MAX_POSITION_VALUE:,.2f}")
        logger.info(f"Stop Loss: {STOP_LOSS_PCT*100}%")
        logger.info(f"Max Daily Loss: ${MAX_DAILY_LOSS:,.2f}")
        logger.info(f"Update Interval: {UPDATE_INTERVAL}s")
        logger.info(f"Dry Run: {DRY_RUN}")
        logger.info("="*70)
        logger.info("")

        # Start the loop (runs until stopped)
        await loop.start()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("="*70)
        logger.info("SHUTDOWN REQUESTED (Ctrl+C)")
        logger.info("="*70)
        logger.info("Stopping trading loop...")

        if 'loop' in locals():
            loop.stop()

        logger.info("✓ Trading system stopped")

    except Exception as e:
        logger.error("="*70)
        logger.error("FATAL ERROR")
        logger.error("="*70)
        logger.error(f"{e}", exc_info=True)

        if alert_manager:
            alert_manager.send_alert(
                level="CRITICAL",
                message=f"🚨 Trading System Crashed: {str(e)}",
                channels=['telegram', 'email']
            )

        raise

    finally:
        logger.info("="*70)
        logger.info("CLEANUP")
        logger.info("="*70)

        # Final status
        if 'position_manager' in locals():
            summary = position_manager.get_portfolio_summary()
            logger.info(f"Final Positions: {summary['active_positions']}")
            logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")

        logger.info("Trading system exited")


if __name__ == '__main__':
    """
    Run the production trading system

    Usage:
        python production_trader.py

    To stop:
        Press Ctrl+C

    Safety:
        1. Always test with dry_run=True first
        2. Use Paper Trading (port 7497) before live
        3. Start with small position sizes
        4. Monitor logs closely for first week
    """

    # Run the async main function
    asyncio.run(main())
