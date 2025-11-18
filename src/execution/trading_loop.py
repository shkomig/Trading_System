"""
Trading Loop Module

Event-driven continuous trading loop that orchestrates:
- Real-time data streaming
- Strategy signal generation
- Order execution
- Position management
- Risk monitoring

Critical Fix: Provides the missing continuous trading capability
that transforms the system from a research platform to a production
automated trading system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, time as dt_time
from enum import Enum
from collections import deque
import pandas as pd

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """Trading loop states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class TradingLoop:
    """
    Continuous Event-Driven Trading Loop

    Orchestrates the complete automated trading workflow:

    1. Market hours validation
    2. Real-time data streaming from IB
    3. Data buffer management
    4. Strategy signal generation
    5. Order execution via OrderExecutor
    6. Position monitoring via PositionManager
    7. Stop-loss management
    8. Error handling and recovery

    Architecture:
        Real-Time Stream -> Data Buffer -> Strategy -> Signal
                                                         |
                                                         v
        Position Manager <- Order Executor <- Signal Decision

    Example:
        >>> loop = TradingLoop(
        ...     broker=ib_connector,
        ...     strategies={
        ...         'AAPL': [ma_strategy, rsi_strategy],
        ...         'MSFT': [ma_strategy]
        ...     },
        ...     executor=order_executor,
        ...     position_manager=pos_manager
        ... )
        >>> await loop.start()
        >>> # Loop runs continuously until stopped
        >>> loop.stop()
    """

    def __init__(
        self,
        broker,  # IBConnector
        strategies: Dict[str, List],  # {symbol: [strategy1, strategy2, ...]}
        executor,  # OrderExecutor
        position_manager,  # PositionManager
        alert_manager=None,  # AlertManager (optional)
        market_hours_validator=None,  # MarketHoursValidator (optional)
        data_buffer_size: int = 200,  # Number of bars to keep in buffer
        update_interval: int = 60,  # Seconds between strategy updates
        use_realtime_bars: bool = True,  # Use 5-sec bars vs tick data
        enable_trading: bool = True,  # Master switch for trading
        max_daily_loss: float = 1000.0  # Max daily loss limit
    ):
        """
        Initialize Trading Loop

        Args:
            broker: IBConnector instance
            strategies: Dict mapping symbols to list of strategies
            executor: OrderExecutor instance
            position_manager: PositionManager instance
            alert_manager: AlertManager for notifications
            market_hours_validator: Market hours checker
            data_buffer_size: How many bars to keep per symbol
            update_interval: Seconds between strategy evaluations
            use_realtime_bars: Use 5-sec bars (True) or tick data (False)
            enable_trading: Allow order execution
            max_daily_loss: Maximum daily loss before shutdown
        """
        self.broker = broker
        self.strategies = strategies
        self.executor = executor
        self.position_manager = position_manager
        self.alert_manager = alert_manager
        self.market_hours_validator = market_hours_validator

        self.data_buffer_size = data_buffer_size
        self.update_interval = update_interval
        self.use_realtime_bars = use_realtime_bars
        self.enable_trading = enable_trading
        self.max_daily_loss = max_daily_loss

        # State
        self.state = LoopState.STOPPED
        self.symbols: Set[str] = set(strategies.keys())

        # Data buffers: {symbol: deque of OHLCV dicts}
        self.data_buffers: Dict[str, deque] = {}
        for symbol in self.symbols:
            self.data_buffers[symbol] = deque(maxlen=data_buffer_size)

        # Current prices: {symbol: float}
        self.current_prices: Dict[str, float] = {}

        # Real-time subscriptions
        self.subscriptions = {}

        # Performance tracking
        self.start_time = None
        self.daily_pnl = 0.0
        self.total_signals_processed = 0
        self.total_orders_executed = 0

        logger.info(
            f"TradingLoop initialized: {len(self.symbols)} symbols, "
            f"buffer_size={data_buffer_size}, update_interval={update_interval}s"
        )

    async def start(self):
        """
        Start the trading loop

        This is the main entry point. The loop will run continuously
        until stop() is called or a fatal error occurs.
        """
        logger.info("="*70)
        logger.info("STARTING TRADING LOOP")
        logger.info("="*70)

        self.state = LoopState.STARTING
        self.start_time = datetime.now()

        try:
            # 1. Connect to broker
            if not self.broker.is_connected():
                logger.info("Connecting to broker...")
                if not self.broker.connect():
                    raise Exception("Failed to connect to broker")

            logger.info("✓ Connected to broker")

            # 2. Load historical data to fill buffers
            await self._initialize_data_buffers()

            # 3. Subscribe to real-time data
            await self._subscribe_realtime_data()

            # 4. Start main loop
            self.state = LoopState.RUNNING
            logger.info("✓ Trading loop RUNNING")

            if self.alert_manager:
                self.alert_manager.send_alert(
                    level="INFO",
                    message="🚀 Trading Loop Started",
                    channels=['telegram']
                )

            # Main event loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
            self.state = LoopState.ERROR

            if self.alert_manager:
                self.alert_manager.send_alert(
                    level="CRITICAL",
                    message=f"❌ Trading Loop Error: {str(e)}",
                    channels=['telegram', 'email']
                )

            raise

        finally:
            await self._cleanup()

    async def _main_loop(self):
        """
        Main trading loop

        Runs continuously, checking market hours and processing signals
        at regular intervals.
        """
        logger.info("Entering main trading loop...")

        while self.state == LoopState.RUNNING:
            try:
                loop_start = datetime.now()

                # 1. Check market hours
                if self.market_hours_validator:
                    if not self.market_hours_validator.is_market_open_now():
                        logger.debug("Market closed - sleeping")
                        await asyncio.sleep(60)
                        continue

                # 2. Check daily loss limit
                if self._check_daily_loss_limit():
                    logger.error("Daily loss limit exceeded - shutting down")
                    self.stop()
                    break

                # 3. Update position prices
                await self._update_position_prices()

                # 4. Check stop-losses
                await self._check_stop_losses()

                # 5. Update trailing stops
                await self._update_trailing_stops()

                # 6. Process strategies and generate signals
                await self._process_strategies()

                # 7. Log status
                self._log_status()

                # 8. Sleep until next update
                elapsed = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.update_interval - elapsed)

                logger.debug(
                    f"Loop iteration completed in {elapsed:.2f}s, "
                    f"sleeping {sleep_time:.2f}s"
                )

                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in main loop iteration: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(self.update_interval)

        logger.info("Exiting main trading loop")

    async def _initialize_data_buffers(self):
        """Load historical data to initialize buffers"""
        logger.info("Initializing data buffers with historical data...")

        for symbol in self.symbols:
            try:
                # Fetch last N days of historical data
                historical_data = self.broker.get_historical_data(
                    symbol=symbol,
                    duration='5 D',  # 5 days should be enough
                    bar_size='1 min'
                )

                if historical_data is not None and not historical_data.empty:
                    # Convert DataFrame to buffer format
                    for idx, row in historical_data.iterrows():
                        bar = {
                            'time': idx,
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        }
                        self.data_buffers[symbol].append(bar)

                    logger.info(
                        f"✓ {symbol}: Loaded {len(self.data_buffers[symbol])} bars"
                    )
                    self.current_prices[symbol] = historical_data['close'].iloc[-1]
                else:
                    logger.warning(f"✗ {symbol}: No historical data available")

            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")

    async def _subscribe_realtime_data(self):
        """Subscribe to real-time data streams"""
        logger.info("Subscribing to real-time data streams...")

        for symbol in self.symbols:
            try:
                if self.use_realtime_bars:
                    # Subscribe to 5-second bars
                    subscription = self.broker.subscribe_realtime_bars(
                        symbol=symbol,
                        callback=self._on_realtime_bar
                    )
                else:
                    # Subscribe to tick data
                    subscription = self.broker.subscribe_market_data(
                        symbol=symbol,
                        callback=self._on_market_tick
                    )

                if subscription:
                    self.subscriptions[symbol] = subscription
                    logger.info(f"✓ {symbol}: Subscribed to real-time data")
                else:
                    logger.warning(f"✗ {symbol}: Failed to subscribe")

            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")

    def _on_realtime_bar(self, symbol: str, bar: Dict):
        """
        Callback for real-time bar updates

        This is called every 5 seconds by IB for each symbol.
        """
        try:
            # Add to buffer
            self.data_buffers[symbol].append(bar)

            # Update current price
            self.current_prices[symbol] = bar['close']

            logger.debug(
                f"[{symbol}] New bar: ${bar['close']:.2f} (volume: {bar['volume']})"
            )

        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}")

    def _on_market_tick(self, symbol: str, ticker: Dict):
        """
        Callback for tick data updates

        This is called on every price tick.
        """
        try:
            # Update current price
            if ticker['last'] and ticker['last'] > 0:
                self.current_prices[symbol] = ticker['last']

            # Optionally aggregate ticks into bars
            # (implementation depends on requirements)

        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}")

    async def _process_strategies(self):
        """
        Process all strategies and execute signals

        This is the core of the trading logic.
        """
        for symbol in self.symbols:
            # Skip if we don't have enough data
            if len(self.data_buffers[symbol]) < 20:
                logger.debug(f"{symbol}: Insufficient data for strategy")
                continue

            # Get strategies for this symbol
            symbol_strategies = self.strategies.get(symbol, [])

            for strategy in symbol_strategies:
                try:
                    # Convert buffer to DataFrame for strategy
                    df = self._buffer_to_dataframe(symbol)

                    if df.empty:
                        continue

                    # Generate signal
                    signals = strategy.generate_signals(df)

                    if signals is None or signals.empty:
                        continue

                    # Get latest signal
                    latest_signal = int(signals.iloc[-1])
                    current_price = self.current_prices.get(symbol, df['close'].iloc[-1])

                    self.total_signals_processed += 1

                    # Skip HOLD signals
                    if latest_signal == 0:
                        logger.debug(
                            f"[{strategy.name}] {symbol}: HOLD @ ${current_price:.2f}"
                        )
                        continue

                    # Log signal
                    signal_type = "BUY" if latest_signal == 1 else "SELL"
                    logger.info(
                        f"[{strategy.name}] {symbol}: {signal_type} signal @ "
                        f"${current_price:.2f}"
                    )

                    # Execute signal if trading enabled
                    if self.enable_trading:
                        await self._execute_strategy_signal(
                            symbol=symbol,
                            signal=latest_signal,
                            price=current_price,
                            strategy_name=strategy.name
                        )
                    else:
                        logger.info("[DRY RUN] Trading disabled - signal not executed")

                except Exception as e:
                    logger.error(
                        f"Error processing strategy {strategy.name} for {symbol}: {e}",
                        exc_info=True
                    )

    async def _execute_strategy_signal(
        self,
        symbol: str,
        signal: int,
        price: float,
        strategy_name: str
    ):
        """Execute a strategy signal via OrderExecutor"""
        try:
            result = self.executor.execute_signal(
                symbol=symbol,
                signal=signal,
                current_price=price,
                strategy_name=strategy_name,
                position_manager=self.position_manager
            )

            if result.success:
                logger.info(
                    f"✓ Order executed: {result.action} {result.executed_quantity} "
                    f"{symbol} @ ${result.avg_fill_price:.2f}"
                )
                self.total_orders_executed += 1

                # Add position to manager if BUY
                if result.action == 'BUY' and result.status.value in ['filled', 'partially_filled']:
                    self.position_manager.add_position(
                        symbol=symbol,
                        quantity=result.executed_quantity,
                        entry_price=result.avg_fill_price,
                        strategy_name=strategy_name,
                        order_id=result.order_id,
                        stop_loss_price=result.avg_fill_price * 0.95  # 5% stop
                    )

                # Remove position if SELL
                elif result.action == 'SELL':
                    self.position_manager.close_position(
                        symbol=symbol,
                        exit_price=result.avg_fill_price,
                        reason=f"Strategy signal: {strategy_name}"
                    )

            else:
                logger.warning(
                    f"✗ Order failed: {result.error_message}"
                )

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _update_position_prices(self):
        """Update current prices for all positions"""
        try:
            self.position_manager.update_position_prices(self.current_prices)
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")

    async def _check_stop_losses(self):
        """Check and execute stop-losses"""
        try:
            triggered = self.position_manager.check_stop_losses()

            for symbol in triggered:
                logger.warning(f"⚠️  STOP-LOSS TRIGGERED: {symbol}")

                # Execute sell order
                position = self.position_manager.get_position(symbol)
                if position:
                    current_price = self.current_prices.get(symbol, 0.0)

                    result = self.executor.execute_signal(
                        symbol=symbol,
                        signal=-1,  # SELL
                        current_price=current_price,
                        strategy_name="STOP_LOSS",
                        position_manager=self.position_manager
                    )

                    if result.success:
                        logger.info(f"✓ Stop-loss executed for {symbol}")
                    else:
                        logger.error(f"✗ Stop-loss execution failed for {symbol}")

        except Exception as e:
            logger.error(f"Error checking stop-losses: {e}")

    async def _update_trailing_stops(self):
        """Update trailing stops for all positions"""
        try:
            updated = self.position_manager.update_trailing_stops()

            for symbol, new_stop in updated.items():
                logger.info(f"📈 Trailing stop updated: {symbol} -> ${new_stop:.2f}")

        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        try:
            total_pnl = self.position_manager.get_total_unrealized_pnl()
            total_pnl += self.position_manager.get_total_realized_pnl()

            if total_pnl < -abs(self.max_daily_loss):
                logger.error(
                    f"Daily loss limit exceeded: ${total_pnl:.2f} < "
                    f"-${self.max_daily_loss:.2f}"
                )

                if self.alert_manager:
                    self.alert_manager.send_alert(
                        level="CRITICAL",
                        message=f"🚨 DAILY LOSS LIMIT EXCEEDED: ${total_pnl:.2f}",
                        channels=['telegram', 'email']
                    )

                return True

            return False

        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return False

    def _buffer_to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert data buffer to DataFrame for strategy"""
        try:
            buffer = list(self.data_buffers[symbol])

            if not buffer:
                return pd.DataFrame()

            df = pd.DataFrame(buffer)
            df.set_index('time', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error converting buffer to DataFrame: {e}")
            return pd.DataFrame()

    def _log_status(self):
        """Log current status"""
        try:
            summary = self.position_manager.get_portfolio_summary()

            logger.info(
                f"STATUS: {summary['active_positions']} positions, "
                f"Exposure: ${summary['total_exposure']:.2f}, "
                f"P&L: ${summary['total_pnl']:.2f}, "
                f"Signals: {self.total_signals_processed}, "
                f"Orders: {self.total_orders_executed}"
            )

        except Exception as e:
            logger.error(f"Error logging status: {e}")

    def stop(self):
        """Stop the trading loop"""
        logger.info("Stopping trading loop...")
        self.state = LoopState.STOPPING

    async def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up trading loop...")

        try:
            # Unsubscribe from real-time data
            for symbol, subscription in self.subscriptions.items():
                try:
                    if self.use_realtime_bars:
                        self.broker.unsubscribe_realtime_bars(subscription)
                    else:
                        self.broker.unsubscribe_market_data(subscription)
                except Exception as e:
                    logger.error(f"Error unsubscribing {symbol}: {e}")

            # Close all positions (optional - may want to leave open)
            # for symbol in list(self.position_manager.get_all_positions().keys()):
            #     ...

            # Disconnect from broker (optional)
            # self.broker.disconnect()

            self.state = LoopState.STOPPED
            logger.info("✓ Trading loop stopped")

            if self.alert_manager:
                self.alert_manager.send_alert(
                    level="INFO",
                    message="🛑 Trading Loop Stopped",
                    channels=['telegram']
                )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Helper function to run the loop
def run_trading_loop(
    broker,
    strategies,
    executor,
    position_manager,
    **kwargs
):
    """
    Run trading loop (synchronous wrapper)

    Args:
        broker: IBConnector
        strategies: Dict of {symbol: [strategies]}
        executor: OrderExecutor
        position_manager: PositionManager
        **kwargs: Additional TradingLoop arguments

    Example:
        >>> run_trading_loop(
        ...     broker=ib_connector,
        ...     strategies={'AAPL': [ma_strategy]},
        ...     executor=executor,
        ...     position_manager=pm
        ... )
    """
    loop = TradingLoop(
        broker=broker,
        strategies=strategies,
        executor=executor,
        position_manager=position_manager,
        **kwargs
    )

    # Run async loop
    asyncio.run(loop.start())
