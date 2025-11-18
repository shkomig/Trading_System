"""
Unit Tests for TradingLoop

Tests the async trading loop functionality including initialization,
data buffer management, and stop-loss checking.

Note: Full integration tests require live broker connection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import pandas as pd
from collections import deque
from src.execution.trading_loop import TradingLoop, LoopState


class TestTradingLoop:
    """Test suite for TradingLoop"""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker"""
        broker = Mock()
        broker.is_connected.return_value = False
        broker.connect.return_value = True
        broker.get_historical_data.return_value = self._create_sample_data()
        broker.subscribe_realtime_bars.return_value = Mock()
        broker.unsubscribe_realtime_bars.return_value = None
        return broker

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy"""
        strategy = Mock()
        strategy.name = 'TestStrategy'
        strategy.generate_signals.return_value = pd.Series([0, 0, 1])  # BUY signal at end
        strategy.calculate_indicators.return_value = pd.DataFrame()
        return strategy

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor"""
        executor = Mock()
        from src.execution.order_executor import ExecutionResult, OrderStatus
        from datetime import datetime

        # Mock successful execution
        executor.execute_signal.return_value = ExecutionResult(
            success=True,
            order_id=12345,
            status=OrderStatus.FILLED,
            symbol='AAPL',
            action='BUY',
            quantity=100,
            executed_quantity=100,
            price=150.0,
            avg_fill_price=150.0,
            timestamp=datetime.now()
        )
        return executor

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager"""
        pm = Mock()
        pm.update_position_prices.return_value = None
        pm.check_stop_losses.return_value = []
        pm.update_trailing_stops.return_value = {}
        pm.get_portfolio_summary.return_value = {
            'active_positions': 0,
            'total_exposure': 0.0,
            'total_pnl': 0.0
        }
        return pm

    @pytest.fixture
    def loop(self, mock_broker, mock_strategy, mock_executor, mock_position_manager):
        """Create TradingLoop instance"""
        return TradingLoop(
            broker=mock_broker,
            strategies={'AAPL': [mock_strategy]},
            executor=mock_executor,
            position_manager=mock_position_manager,
            data_buffer_size=50,
            update_interval=1,  # 1 second for fast tests
            use_realtime_bars=True,
            enable_trading=False,  # Disable trading in tests
            max_daily_loss=1000.0
        )

    def _create_sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1min')
        return pd.DataFrame({
            'open': 150.0,
            'high': 151.0,
            'low': 149.0,
            'close': 150.5,
            'volume': 1000
        }, index=dates)

    def test_initialization(self, loop):
        """Test TradingLoop initialization"""
        assert loop.state == LoopState.STOPPED
        assert 'AAPL' in loop.symbols
        assert 'AAPL' in loop.data_buffers
        assert isinstance(loop.data_buffers['AAPL'], deque)
        assert loop.update_interval == 1
        assert loop.enable_trading is False

    def test_symbols_extraction(self, mock_broker, mock_strategy, mock_executor, mock_position_manager):
        """Test that symbols are extracted from strategies dict"""
        strategies = {
            'AAPL': [mock_strategy],
            'MSFT': [mock_strategy],
            'TSLA': [mock_strategy]
        }

        loop = TradingLoop(
            broker=mock_broker,
            strategies=strategies,
            executor=mock_executor,
            position_manager=mock_position_manager
        )

        assert len(loop.symbols) == 3
        assert 'AAPL' in loop.symbols
        assert 'MSFT' in loop.symbols
        assert 'TSLA' in loop.symbols

    def test_data_buffers_creation(self, loop):
        """Test data buffers are created for each symbol"""
        assert 'AAPL' in loop.data_buffers
        assert isinstance(loop.data_buffers['AAPL'], deque)
        assert loop.data_buffers['AAPL'].maxlen == 50

    def test_buffer_to_dataframe(self, loop):
        """Test converting buffer to DataFrame"""
        # Add some bars to buffer
        for i in range(10):
            bar = {
                'time': pd.Timestamp(f'2024-01-01 09:3{i}:00'),
                'open': 150.0 + i,
                'high': 151.0 + i,
                'low': 149.0 + i,
                'close': 150.5 + i,
                'volume': 1000 + i * 100
            }
            loop.data_buffers['AAPL'].append(bar)

        df = loop._buffer_to_dataframe('AAPL')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert df.index.name == 'time' or isinstance(df.index, pd.DatetimeIndex)

    def test_on_realtime_bar(self, loop):
        """Test real-time bar callback"""
        bar = {
            'time': pd.Timestamp('2024-01-01 09:30:00'),
            'open': 150.0,
            'high': 151.0,
            'low': 149.0,
            'close': 150.5,
            'volume': 1000
        }

        loop._on_realtime_bar('AAPL', bar)

        # Check bar was added to buffer
        assert len(loop.data_buffers['AAPL']) == 1
        assert loop.data_buffers['AAPL'][0] == bar

        # Check current price updated
        assert loop.current_prices['AAPL'] == 150.5

    def test_on_market_tick(self, loop):
        """Test market tick callback"""
        ticker = {
            'symbol': 'AAPL',
            'time': pd.Timestamp('2024-01-01 09:30:00'),
            'bid': 150.0,
            'ask': 150.1,
            'last': 150.05,
            'volume': 1000
        }

        loop._on_market_tick('AAPL', ticker)

        # Check current price updated
        assert loop.current_prices['AAPL'] == 150.05

    def test_stop_method(self, loop):
        """Test stop method"""
        loop.state = LoopState.RUNNING

        loop.stop()

        assert loop.state == LoopState.STOPPING

    @pytest.mark.asyncio
    async def test_check_daily_loss_limit_within_limit(self, loop, mock_position_manager):
        """Test daily loss limit check when within limit"""
        # Setup: small loss
        mock_position_manager.get_total_unrealized_pnl.return_value = -100.0
        mock_position_manager.get_total_realized_pnl.return_value = 0.0

        exceeded = loop._check_daily_loss_limit()

        assert exceeded is False

    @pytest.mark.asyncio
    async def test_check_daily_loss_limit_exceeded(self, loop, mock_position_manager):
        """Test daily loss limit check when exceeded"""
        # Setup: loss exceeds limit
        mock_position_manager.get_total_unrealized_pnl.return_value = -1500.0
        mock_position_manager.get_total_realized_pnl.return_value = 0.0

        exceeded = loop._check_daily_loss_limit()

        assert exceeded is True

    @pytest.mark.asyncio
    async def test_update_position_prices(self, loop, mock_position_manager):
        """Test updating position prices"""
        loop.current_prices = {'AAPL': 155.0, 'MSFT': 310.0}

        await loop._update_position_prices()

        mock_position_manager.update_position_prices.assert_called_once_with(
            {'AAPL': 155.0, 'MSFT': 310.0}
        )

    @pytest.mark.asyncio
    async def test_check_stop_losses_no_triggers(self, loop, mock_position_manager, mock_executor):
        """Test checking stop-losses when none are triggered"""
        mock_position_manager.check_stop_losses.return_value = []

        await loop._check_stop_losses()

        # Should not execute any orders
        mock_executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_stop_losses_with_triggers(self, loop, mock_position_manager, mock_executor):
        """Test checking stop-losses when some are triggered"""
        # Setup: AAPL stop-loss triggered
        mock_position_manager.check_stop_losses.return_value = ['AAPL']

        mock_position = Mock()
        mock_position_manager.get_position.return_value = mock_position

        loop.current_prices = {'AAPL': 140.0}

        await loop._check_stop_losses()

        # Should execute SELL order
        mock_executor.execute_signal.assert_called_once()
        call_args = mock_executor.execute_signal.call_args
        assert call_args[1]['symbol'] == 'AAPL'
        assert call_args[1]['signal'] == -1  # SELL

    @pytest.mark.asyncio
    async def test_update_trailing_stops(self, loop, mock_position_manager):
        """Test updating trailing stops"""
        mock_position_manager.update_trailing_stops.return_value = {
            'AAPL': 152.0
        }

        await loop._update_trailing_stops()

        mock_position_manager.update_trailing_stops.assert_called_once()

    def test_log_status(self, loop, mock_position_manager):
        """Test status logging doesn't crash"""
        loop.total_signals_processed = 10
        loop.total_orders_executed = 3

        # Should not raise exception
        loop._log_status()

    @pytest.mark.asyncio
    async def test_initialize_data_buffers(self, loop, mock_broker):
        """Test initializing data buffers with historical data"""
        await loop._initialize_data_buffers()

        # Should have called get_historical_data
        mock_broker.get_historical_data.assert_called()

        # Should have data in buffer
        assert len(loop.data_buffers['AAPL']) > 0

    @pytest.mark.asyncio
    async def test_subscribe_realtime_data(self, loop, mock_broker):
        """Test subscribing to real-time data"""
        await loop._subscribe_realtime_data()

        # Should have called subscribe
        mock_broker.subscribe_realtime_bars.assert_called()

        # Should have subscription
        assert 'AAPL' in loop.subscriptions


class TestLoopState:
    """Test LoopState enum"""

    def test_loop_state_values(self):
        """Test all LoopState enum values"""
        assert LoopState.STOPPED.value == 'stopped'
        assert LoopState.STARTING.value == 'starting'
        assert LoopState.RUNNING.value == 'running'
        assert LoopState.PAUSED.value == 'paused'
        assert LoopState.STOPPING.value == 'stopping'
        assert LoopState.ERROR.value == 'error'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
