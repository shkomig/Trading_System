"""
Unit Tests for PositionManager

Tests position tracking, P&L calculation, stop-loss management,
and trailing stops.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from src.execution.position_manager import (
    PositionManager,
    Position,
    PositionState
)


class TestPosition:
    """Test suite for Position dataclass"""

    @pytest.fixture
    def position(self):
        """Create test position"""
        return Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            strategy_name='MA_Crossover',
            order_id=12345,
            state=PositionState.OPEN,
            current_price=150.0,
            stop_loss_price=142.5
        )

    def test_position_creation(self, position):
        """Test position initialization"""
        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.state == PositionState.OPEN
        assert position.stop_loss_price == 142.5

    def test_update_current_price(self, position):
        """Test updating current price"""
        position.update_current_price(155.0)

        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0  # (155-150) * 100

    def test_pnl_calculation_profit(self, position):
        """Test P&L calculation with profit"""
        position.update_current_price(160.0)

        assert position.unrealized_pnl == 1000.0  # (160-150) * 100
        assert position.get_pnl_pct() == pytest.approx(6.67, rel=0.01)

    def test_pnl_calculation_loss(self, position):
        """Test P&L calculation with loss"""
        position.update_current_price(145.0)

        assert position.unrealized_pnl == -500.0  # (145-150) * 100
        assert position.get_pnl_pct() == pytest.approx(-3.33, rel=0.01)

    def test_stop_loss_triggered(self, position):
        """Test stop-loss trigger detection"""
        position.stop_loss_price = 142.5

        # Above stop - not triggered
        position.update_current_price(145.0)
        assert position.is_stop_triggered() is False

        # At stop - triggered
        position.update_current_price(142.5)
        assert position.is_stop_triggered() is True

        # Below stop - triggered
        position.update_current_price(140.0)
        assert position.is_stop_triggered() is True

    def test_trailing_stop_calculation(self, position):
        """Test trailing stop calculation"""
        position.trailing_stop_enabled = True
        position.trailing_stop_pct = 0.05  # 5%

        # Update to higher price
        position.update_current_price(160.0)

        trailing_stop = position.calculate_trailing_stop()
        expected = 160.0 * 0.95  # 5% below highest price
        assert trailing_stop == pytest.approx(expected)

    def test_highest_price_tracking(self, position):
        """Test highest price tracking for trailing stop"""
        position.update_current_price(155.0)
        assert position.highest_price == 155.0

        position.update_current_price(160.0)
        assert position.highest_price == 160.0

        # Lower price shouldn't change highest
        position.update_current_price(158.0)
        assert position.highest_price == 160.0

    def test_trailing_stop_trigger(self, position):
        """Test trailing stop triggering"""
        position.trailing_stop_enabled = True
        position.trailing_stop_pct = 0.05

        # Price rises
        position.update_current_price(160.0)
        assert position.is_stop_triggered() is False

        # Price falls below trailing stop
        trailing_stop = position.calculate_trailing_stop()
        position.update_current_price(trailing_stop - 0.01)
        assert position.is_stop_triggered() is True

    def test_position_to_dict(self, position):
        """Test position serialization"""
        position.update_current_price(155.0)

        pos_dict = position.to_dict()

        assert pos_dict['symbol'] == 'AAPL'
        assert pos_dict['quantity'] == 100
        assert pos_dict['entry_price'] == 150.0
        assert pos_dict['current_price'] == 155.0
        assert pos_dict['state'] == 'open'
        assert 'unrealized_pnl' in pos_dict
        assert 'pnl_pct' in pos_dict


class TestPositionManager:
    """Test suite for PositionManager"""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker"""
        broker = Mock()
        broker.get_positions.return_value = []
        return broker

    @pytest.fixture
    def manager(self, mock_broker):
        """Create PositionManager instance"""
        return PositionManager(
            broker=mock_broker,
            max_positions=5,
            enable_trailing_stops=True,
            trailing_stop_pct=0.05
        )

    def test_initialization(self, manager):
        """Test PositionManager initialization"""
        assert manager.max_positions == 5
        assert manager.enable_trailing_stops is True
        assert manager.trailing_stop_pct == 0.05
        assert len(manager.positions) == 0
        assert len(manager.closed_positions) == 0

    def test_add_position(self, manager):
        """Test adding a position"""
        success = manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='MA_Crossover',
            order_id=12345,
            stop_loss_price=142.5
        )

        assert success is True
        assert 'AAPL' in manager.positions
        assert manager.positions['AAPL'].quantity == 100
        assert manager.positions['AAPL'].entry_price == 150.0

    def test_add_duplicate_position(self, manager):
        """Test adding duplicate position - should fail"""
        manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=1
        )

        # Try to add again
        success = manager.add_position(
            symbol='AAPL',
            quantity=50,
            entry_price=151.0,
            strategy_name='Test',
            order_id=2
        )

        assert success is False
        assert manager.positions['AAPL'].quantity == 100  # Original unchanged

    def test_max_positions_limit(self, manager):
        """Test max positions enforcement"""
        # Add max positions
        for i in range(5):
            manager.add_position(
                symbol=f'SYM{i}',
                quantity=100,
                entry_price=100.0,
                strategy_name='Test',
                order_id=i
            )

        # Try to add one more - should fail
        success = manager.add_position(
            symbol='EXTRA',
            quantity=100,
            entry_price=100.0,
            strategy_name='Test',
            order_id=999
        )

        assert success is False
        assert len(manager.positions) == 5

    def test_has_position(self, manager):
        """Test has_position check"""
        assert manager.has_position('AAPL') is False

        manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=1
        )

        assert manager.has_position('AAPL') is True
        assert manager.has_position('MSFT') is False

    def test_get_position(self, manager):
        """Test getting position by symbol"""
        manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=1
        )

        position = manager.get_position('AAPL')
        assert position is not None
        assert position.symbol == 'AAPL'
        assert position.quantity == 100

        # Non-existent position
        assert manager.get_position('MSFT') is None

    def test_position_count(self, manager):
        """Test position counting"""
        assert manager.position_count() == 0

        manager.add_position('SYM1', 100, 100.0, 'Test', 1)
        assert manager.position_count() == 1

        manager.add_position('SYM2', 100, 100.0, 'Test', 2)
        assert manager.position_count() == 2

    def test_update_position_prices(self, manager):
        """Test updating prices for all positions"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.add_position('MSFT', 50, 300.0, 'Test', 2)

        prices = {'AAPL': 155.0, 'MSFT': 310.0}
        manager.update_position_prices(prices)

        assert manager.positions['AAPL'].current_price == 155.0
        assert manager.positions['MSFT'].current_price == 310.0

    def test_update_single_position(self, manager):
        """Test updating single position price"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)

        manager.update_single_position('AAPL', 155.0)

        assert manager.positions['AAPL'].current_price == 155.0

    def test_check_stop_losses(self, manager):
        """Test stop-loss checking"""
        manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=1,
            stop_loss_price=142.5
        )

        # Price above stop - not triggered
        manager.update_single_position('AAPL', 145.0)
        triggered = manager.check_stop_losses()
        assert len(triggered) == 0

        # Price at/below stop - triggered
        manager.update_single_position('AAPL', 142.0)
        triggered = manager.check_stop_losses()
        assert 'AAPL' in triggered

    def test_close_position(self, manager):
        """Test closing a position"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)

        success = manager.close_position(
            symbol='AAPL',
            exit_price=155.0,
            reason='Strategy signal'
        )

        assert success is True
        assert 'AAPL' not in manager.positions
        assert len(manager.closed_positions) == 1

        closed = manager.closed_positions[0]
        assert closed.symbol == 'AAPL'
        assert closed.exit_price == 155.0
        assert closed.exit_reason == 'Strategy signal'
        assert closed.realized_pnl == 500.0  # (155-150) * 100

    def test_close_nonexistent_position(self, manager):
        """Test closing position that doesn't exist"""
        success = manager.close_position('AAPL', 155.0, 'Test')

        assert success is False

    def test_get_total_exposure(self, manager):
        """Test calculating total portfolio exposure"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.add_position('MSFT', 50, 300.0, 'Test', 2)

        manager.update_position_prices({'AAPL': 155.0, 'MSFT': 310.0})

        exposure = manager.get_total_exposure()
        expected = (100 * 155.0) + (50 * 310.0)
        assert exposure == expected

    def test_get_total_unrealized_pnl(self, manager):
        """Test total unrealized P&L calculation"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.add_position('MSFT', 50, 300.0, 'Test', 2)

        manager.update_position_prices({'AAPL': 155.0, 'MSFT': 310.0})

        pnl = manager.get_total_unrealized_pnl()
        expected = (100 * 5.0) + (50 * 10.0)  # 500 + 500
        assert pnl == expected

    def test_get_total_realized_pnl(self, manager):
        """Test total realized P&L from closed positions"""
        # Add and close positions
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.close_position('AAPL', 155.0, 'Win')

        manager.add_position('MSFT', 50, 300.0, 'Test', 2)
        manager.close_position('MSFT', 295.0, 'Loss')

        pnl = manager.get_total_realized_pnl()
        expected = 500.0 + (-250.0)  # Win + Loss
        assert pnl == expected

    def test_update_trailing_stops(self, manager):
        """Test trailing stop updates"""
        manager.add_position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            strategy_name='Test',
            order_id=1,
            stop_loss_price=142.5
        )

        # Price rises
        manager.update_single_position('AAPL', 160.0)

        updated = manager.update_trailing_stops()

        # Should have updated AAPL's stop
        assert 'AAPL' in updated
        new_stop = updated['AAPL']
        assert new_stop > 142.5  # Should be higher than original
        assert new_stop == pytest.approx(160.0 * 0.95)  # 5% trailing

    def test_portfolio_summary(self, manager):
        """Test portfolio summary generation"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.update_single_position('AAPL', 155.0)

        summary = manager.get_portfolio_summary()

        assert summary['active_positions'] == 1
        assert summary['total_exposure'] == 15500.0
        assert summary['unrealized_pnl'] == 500.0
        assert summary['realized_pnl'] == 0.0
        assert summary['total_pnl'] == 500.0
        assert len(summary['positions']) == 1

    def test_get_all_positions(self, manager):
        """Test getting all positions"""
        manager.add_position('AAPL', 100, 150.0, 'Test', 1)
        manager.add_position('MSFT', 50, 300.0, 'Test', 2)

        positions = manager.get_all_positions()

        assert len(positions) == 2
        assert 'AAPL' in positions
        assert 'MSFT' in positions


class TestPositionState:
    """Test PositionState enum"""

    def test_position_state_values(self):
        """Test all PositionState enum values"""
        assert PositionState.OPENING.value == 'opening'
        assert PositionState.OPEN.value == 'open'
        assert PositionState.CLOSING.value == 'closing'
        assert PositionState.CLOSED.value == 'closed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
