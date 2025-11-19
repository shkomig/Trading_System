"""
Unit Tests for OrderExecutor

Tests the core order execution functionality with comprehensive
coverage of all execution scenarios.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from src.execution.order_executor import (
    OrderExecutor,
    OrderStatus,
    ExecutionResult
)


class TestOrderExecutor:
    """Test suite for OrderExecutor"""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker"""
        broker = Mock()
        broker.is_connected.return_value = True
        broker.place_market_order.return_value = 12345
        broker.get_positions.return_value = []
        return broker

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager"""
        return Mock()

    @pytest.fixture
    def mock_position_manager(self):
        """Create mock position manager"""
        pm = Mock()
        pm.has_position.return_value = False
        pm.position_count.return_value = 0
        return pm

    @pytest.fixture
    def executor(self, mock_broker, mock_risk_manager):
        """Create OrderExecutor instance for testing"""
        return OrderExecutor(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            max_position_value=10000.0,
            max_positions=5,
            enable_stop_loss=True,
            stop_loss_pct=0.05,
            dry_run=True  # Always use dry_run in tests
        )

    def test_initialization(self, executor):
        """Test OrderExecutor initialization"""
        assert executor.max_position_value == 10000.0
        assert executor.max_positions == 5
        assert executor.enable_stop_loss is True
        assert executor.stop_loss_pct == 0.05
        assert executor.dry_run is True
        assert len(executor.active_orders) == 0
        assert len(executor.execution_history) == 0

    def test_hold_signal(self, executor):
        """Test HOLD signal (signal=0) - should not execute"""
        result = executor.execute_signal(
            symbol='AAPL',
            signal=0,  # HOLD
            current_price=150.0,
            strategy_name='Test'
        )

        assert result.success is True
        assert result.action == 'HOLD'
        assert result.quantity == 0
        assert result.order_id is None
        assert result.status == OrderStatus.PENDING

    def test_buy_signal_success(self, executor, mock_position_manager):
        """Test successful BUY signal execution"""
        result = executor.execute_signal(
            symbol='AAPL',
            signal=1,  # BUY
            current_price=150.0,
            strategy_name='MA_Crossover',
            position_manager=mock_position_manager
        )

        assert result.success is True
        assert result.action == 'BUY'
        assert result.quantity > 0
        assert result.symbol == 'AAPL'
        assert result.status == OrderStatus.FILLED
        # In dry_run mode, should get fake order ID
        assert result.order_id == 999999

    def test_buy_signal_with_existing_position(self, executor, mock_position_manager):
        """Test BUY signal when position already exists - should reject"""
        mock_position_manager.has_position.return_value = True

        result = executor.execute_signal(
            symbol='AAPL',
            signal=1,  # BUY
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        assert result.success is False
        assert result.status == OrderStatus.REJECTED
        assert "Already have position" in result.error_message

    def test_buy_signal_max_positions_reached(self, executor, mock_position_manager):
        """Test BUY signal when max positions reached - should reject"""
        mock_position_manager.position_count.return_value = 5  # Max positions

        result = executor.execute_signal(
            symbol='AAPL',
            signal=1,  # BUY
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        assert result.success is False
        assert result.status == OrderStatus.REJECTED
        assert "Max positions" in result.error_message

    def test_sell_signal_success(self, executor, mock_position_manager):
        """Test successful SELL signal execution"""
        # Setup: have a position
        mock_position_manager.has_position.return_value = True
        mock_position = Mock()
        mock_position.quantity = 100
        mock_position_manager.get_position.return_value = mock_position

        result = executor.execute_signal(
            symbol='AAPL',
            signal=-1,  # SELL
            current_price=155.0,
            strategy_name='MA_Crossover',
            position_manager=mock_position_manager
        )

        assert result.success is True
        assert result.action == 'SELL'
        assert result.quantity == 100
        assert result.status == OrderStatus.FILLED

    def test_sell_signal_without_position(self, executor, mock_position_manager):
        """Test SELL signal without position - should reject"""
        mock_position_manager.has_position.return_value = False

        result = executor.execute_signal(
            symbol='AAPL',
            signal=-1,  # SELL
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        assert result.success is False
        assert result.status == OrderStatus.REJECTED
        assert "No position to sell" in result.error_message

    def test_invalid_signal(self, executor):
        """Test invalid signal value - should reject"""
        result = executor.execute_signal(
            symbol='AAPL',
            signal=99,  # Invalid
            current_price=150.0,
            strategy_name='Test'
        )

        assert result.success is False
        assert result.status == OrderStatus.REJECTED
        assert "Invalid signal" in result.error_message

    def test_position_sizing(self, executor):
        """Test position size calculation"""
        price = 150.0
        max_value = executor.max_position_value

        # Should calculate: max_value / price
        expected_quantity = int(max_value / price)

        quantity = executor._calculate_position_size(
            symbol='AAPL',
            price=price,
            data_context=None
        )

        assert quantity == expected_quantity
        assert quantity * price <= max_value

    def test_position_value_limit(self, executor, mock_position_manager):
        """Test that position value doesn't exceed max"""
        result = executor.execute_signal(
            symbol='AAPL',
            signal=1,  # BUY
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        if result.success:
            position_value = result.quantity * result.price
            assert position_value <= executor.max_position_value

    def test_execution_history_recording(self, executor, mock_position_manager):
        """Test that executions are recorded in history"""
        initial_count = len(executor.execution_history)

        executor.execute_signal(
            symbol='AAPL',
            signal=1,  # BUY
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        assert len(executor.execution_history) == initial_count + 1

        recorded = executor.execution_history[-1]
        assert recorded.symbol == 'AAPL'
        assert recorded.action == 'BUY'

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization"""
        result = ExecutionResult(
            success=True,
            order_id=12345,
            status=OrderStatus.FILLED,
            symbol='AAPL',
            action='BUY',
            quantity=100,
            executed_quantity=100,
            price=150.0,
            avg_fill_price=150.05,
            timestamp=datetime.now(),
            commission=1.0
        )

        result_dict = result.to_dict()

        assert result_dict['success'] is True
        assert result_dict['order_id'] == 12345
        assert result_dict['status'] == 'filled'
        assert result_dict['symbol'] == 'AAPL'
        assert result_dict['action'] == 'BUY'
        assert result_dict['quantity'] == 100
        assert result_dict['commission'] == 1.0

    def test_multiple_executions(self, executor, mock_position_manager):
        """Test multiple sequential executions"""
        # Execute BUY
        result1 = executor.execute_signal(
            symbol='AAPL',
            signal=1,
            current_price=150.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        # Setup for SELL
        mock_position_manager.has_position.return_value = True
        mock_position = Mock()
        mock_position.quantity = result1.quantity
        mock_position_manager.get_position.return_value = mock_position

        # Execute SELL
        result2 = executor.execute_signal(
            symbol='AAPL',
            signal=-1,
            current_price=155.0,
            strategy_name='Test',
            position_manager=mock_position_manager
        )

        assert result1.success is True
        assert result2.success is True
        assert len(executor.execution_history) == 2

    def test_real_mode_vs_dry_run(self, mock_broker, mock_risk_manager):
        """Test difference between real mode and dry_run"""
        # Dry run executor
        dry_executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            dry_run=True
        )

        # Real executor
        real_executor = OrderExecutor(
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            dry_run=False
        )

        assert dry_executor.dry_run is True
        assert real_executor.dry_run is False

        # In dry_run, should not call broker.place_market_order
        # (we can't easily test this without integration tests)

    def test_get_execution_history(self, executor, mock_position_manager):
        """Test getting execution history"""
        # Execute a few orders
        for i in range(3):
            executor.execute_signal(
                symbol=f'SYM{i}',
                signal=1,
                current_price=100.0 + i,
                strategy_name='Test',
                position_manager=mock_position_manager
            )

        history = executor.get_execution_history()
        assert len(history) == 3
        assert all(isinstance(r, ExecutionResult) for r in history)


class TestOrderStatus:
    """Test OrderStatus enum"""

    def test_order_status_values(self):
        """Test all OrderStatus enum values exist"""
        assert OrderStatus.PENDING.value == 'pending'
        assert OrderStatus.SUBMITTED.value == 'submitted'
        assert OrderStatus.FILLED.value == 'filled'
        assert OrderStatus.PARTIALLY_FILLED.value == 'partially_filled'
        assert OrderStatus.CANCELLED.value == 'cancelled'
        assert OrderStatus.REJECTED.value == 'rejected'
        assert OrderStatus.FAILED.value == 'failed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
