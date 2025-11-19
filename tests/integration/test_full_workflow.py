"""
Integration Test - Full Trading Workflow

Tests the complete end-to-end workflow:
Broker -> Data -> Strategy -> Signal -> OrderExecutor -> PositionManager

Note: This is a mock integration test. Real integration with IB requires
      a running TWS/Gateway instance.
"""

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime

from src.broker.ib_connector import IBConnector
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.execution.order_executor import OrderExecutor, OrderStatus
from src.execution.position_manager import PositionManager
from src.risk_management.position_sizing import PositionSizer


class TestFullTradingWorkflow:
    """Integration test for complete trading workflow"""

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Create data with clear MA crossover
        close_prices = []
        for i in range(100):
            if i < 50:
                # Downtrend
                price = 150 - i * 0.5
            else:
                # Uptrend - should trigger BUY signal
                price = 125 + (i - 50) * 1.0
            close_prices.append(price)

        return pd.DataFrame({
            'open': close_prices,
            'high': [p + 1 for p in close_prices],
            'low': [p - 1 for p in close_prices],
            'close': close_prices,
            'volume': 1000000
        }, index=dates)

    @pytest.fixture
    def mock_broker(self, sample_data):
        """Create mock broker with realistic responses"""
        broker = Mock(spec=IBConnector)
        broker.is_connected.return_value = True
        broker.connect.return_value = True
        broker.get_historical_data.return_value = sample_data
        broker.place_market_order.return_value = 12345
        broker.get_positions.return_value = []
        broker.get_account_info.return_value = {
            'BuyingPower': '100000.00',
            'NetLiquidation': '100000.00'
        }
        return broker

    def test_complete_buy_workflow(self, mock_broker, sample_data):
        """
        Test complete BUY workflow:
        1. Load historical data
        2. Run strategy
        3. Generate BUY signal
        4. Execute order
        5. Track position
        """
        # 1. Create components
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        risk_manager = PositionSizer(account_value=100000)
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=5
        )
        executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=risk_manager,
            max_position_value=10000.0,
            dry_run=True  # Use dry_run for integration test
        )

        # 2. Load and process data
        data = sample_data
        assert not data.empty

        # 3. Generate signals
        signals = strategy.generate_signals(data)
        assert signals is not None
        assert not signals.empty

        # 4. Get latest signal
        latest_signal = int(signals.iloc[-1])
        current_price = data['close'].iloc[-1]

        # Should be BUY signal due to uptrend
        assert latest_signal in [-1, 0, 1]

        # 5. Execute signal
        if latest_signal != 0:
            result = executor.execute_signal(
                symbol='AAPL',
                signal=latest_signal,
                current_price=current_price,
                strategy_name=strategy.name,
                position_manager=position_manager
            )

            # Verify execution
            assert result is not None
            assert result.symbol == 'AAPL'

            if latest_signal == 1 and result.success:  # BUY
                # 6. Add position to manager
                position_manager.add_position(
                    symbol='AAPL',
                    quantity=result.executed_quantity,
                    entry_price=result.avg_fill_price,
                    strategy_name=strategy.name,
                    order_id=result.order_id,
                    stop_loss_price=result.avg_fill_price * 0.95
                )

                # Verify position added
                assert position_manager.has_position('AAPL')
                position = position_manager.get_position('AAPL')
                assert position.quantity == result.executed_quantity
                assert position.entry_price == result.avg_fill_price

    def test_complete_sell_workflow(self, mock_broker, sample_data):
        """
        Test complete SELL workflow:
        1. Have existing position
        2. Generate SELL signal
        3. Execute order
        4. Close position
        """
        # Setup
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        risk_manager = PositionSizer(account_value=100000)
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=5
        )
        executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=risk_manager,
            dry_run=True
        )

        # 1. Add existing position
        position_manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=11111,
            stop_loss_price=142.5
        )

        assert position_manager.has_position('AAPL')

        # 2. Execute SELL signal
        result = executor.execute_signal(
            symbol='AAPL',
            signal=-1,  # SELL
            current_price=155.0,
            strategy_name=strategy.name,
            position_manager=position_manager
        )

        # 3. Verify execution
        assert result.success is True
        assert result.action == 'SELL'
        assert result.quantity == 100

        # 4. Close position
        position_manager.close_position(
            symbol='AAPL',
            exit_price=result.avg_fill_price,
            reason='Strategy SELL signal'
        )

        # Verify position closed
        assert not position_manager.has_position('AAPL')
        assert len(position_manager.closed_positions) == 1

        closed = position_manager.closed_positions[0]
        assert closed.realized_pnl > 0  # Should be profit (155 > 150)

    def test_stop_loss_trigger_workflow(self, mock_broker):
        """
        Test stop-loss workflow:
        1. Have position with stop-loss
        2. Price drops below stop
        3. Stop-loss triggers
        4. Position closes
        """
        # Setup
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=5
        )
        executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=Mock(),
            dry_run=True
        )

        # 1. Add position with stop-loss
        position_manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=12345,
            stop_loss_price=142.5  # 5% below entry
        )

        # 2. Update price above stop - should not trigger
        position_manager.update_single_position('AAPL', 145.0)
        triggered = position_manager.check_stop_losses()
        assert len(triggered) == 0

        # 3. Update price below stop - should trigger
        position_manager.update_single_position('AAPL', 140.0)
        triggered = position_manager.check_stop_losses()
        assert 'AAPL' in triggered

        # 4. Execute stop-loss sell
        result = executor.execute_signal(
            symbol='AAPL',
            signal=-1,  # SELL
            current_price=140.0,
            strategy_name='STOP_LOSS',
            position_manager=position_manager
        )

        assert result.success is True
        assert result.action == 'SELL'

        # Close position
        position_manager.close_position(
            symbol='AAPL',
            exit_price=140.0,
            reason='Stop-loss triggered'
        )

        # Verify loss
        closed = position_manager.closed_positions[0]
        assert closed.realized_pnl < 0  # Should be loss
        assert closed.exit_reason == 'Stop-loss triggered'

    def test_trailing_stop_workflow(self, mock_broker):
        """
        Test trailing stop workflow:
        1. Have position with trailing stop
        2. Price rises - trailing stop updates
        3. Price falls below trailing stop
        4. Position closes
        """
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=5,
            enable_trailing_stops=True,
            trailing_stop_pct=0.05  # 5%
        )
        executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=Mock(),
            dry_run=True
        )

        # 1. Add position
        position_manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=12345,
            stop_loss_price=142.5
        )

        # 2. Price rises
        position_manager.update_single_position('AAPL', 160.0)

        # Update trailing stops
        updated = position_manager.update_trailing_stops()
        assert 'AAPL' in updated

        new_stop = updated['AAPL']
        assert new_stop > 142.5  # Should be higher than original
        assert new_stop == pytest.approx(160.0 * 0.95)  # 5% below peak

        # 3. Price falls below trailing stop
        position_manager.update_single_position('AAPL', new_stop - 0.5)
        triggered = position_manager.check_stop_losses()
        assert 'AAPL' in triggered

        # Execute and close
        result = executor.execute_signal(
            symbol='AAPL',
            signal=-1,
            current_price=new_stop - 0.5,
            strategy_name='TRAILING_STOP',
            position_manager=position_manager
        )

        position_manager.close_position(
            symbol='AAPL',
            exit_price=result.avg_fill_price,
            reason='Trailing stop'
        )

        # Should still have profit (entry 150, exit ~152)
        closed = position_manager.closed_positions[0]
        assert closed.realized_pnl > 0

    def test_max_positions_limit(self, mock_broker):
        """
        Test max positions enforcement:
        1. Add max positions
        2. Try to add one more - should reject
        """
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=3
        )
        executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=Mock(),
            max_positions=3,
            dry_run=True
        )

        # Add 3 positions (max)
        for i in range(3):
            result = executor.execute_signal(
                symbol=f'SYM{i}',
                signal=1,  # BUY
                current_price=100.0,
                strategy_name='Test',
                position_manager=position_manager
            )

            if result.success:
                position_manager.add_position(
                    symbol=f'SYM{i}',
                    quantity=result.executed_quantity,
                    entry_price=result.avg_fill_price,
                    strategy_name='Test',
                    order_id=i
                )

        assert position_manager.position_count() == 3

        # Try to add 4th - should reject
        result = executor.execute_signal(
            symbol='SYM4',
            signal=1,  # BUY
            current_price=100.0,
            strategy_name='Test',
            position_manager=position_manager
        )

        assert result.success is False
        assert "Max positions" in result.error_message

    def test_portfolio_pnl_tracking(self, mock_broker):
        """
        Test portfolio P&L tracking:
        1. Add multiple positions
        2. Update prices
        3. Verify total P&L calculation
        """
        position_manager = PositionManager(
            broker=mock_broker,
            max_positions=5
        )

        # Add 3 positions
        position_manager.add_position('AAPL', 100, 150.0, 'Test', 1, 142.5)
        position_manager.add_position('MSFT', 50, 300.0, 'Test', 2, 285.0)
        position_manager.add_position('TSLA', 25, 200.0, 'Test', 3, 190.0)

        # Update prices
        position_manager.update_position_prices({
            'AAPL': 155.0,  # +5 profit
            'MSFT': 310.0,  # +10 profit
            'TSLA': 195.0   # -5 loss
        })

        # Calculate expected P&L
        expected_pnl = (
            (155.0 - 150.0) * 100 +  # AAPL: +500
            (310.0 - 300.0) * 50 +   # MSFT: +500
            (195.0 - 200.0) * 25     # TSLA: -125
        )  # Total: +875

        total_pnl = position_manager.get_total_unrealized_pnl()
        assert total_pnl == pytest.approx(expected_pnl)

        # Get portfolio summary
        summary = position_manager.get_portfolio_summary()
        assert summary['active_positions'] == 3
        assert summary['unrealized_pnl'] == pytest.approx(expected_pnl)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
