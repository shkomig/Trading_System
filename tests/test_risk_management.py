"""
Unit Tests for Risk Management
"""

import pytest
import numpy as np

from src.risk_management.kelly_criterion import KellyCriterion
from src.risk_management.position_sizing import PositionSizer, PositionSizeMethod
from src.risk_management.stop_loss_manager import StopLossManager, StopLossType


class TestKellyCriterion:
    """Test Kelly Criterion calculator"""
    
    def test_initialization(self):
        """Test Kelly calculator initialization"""
        kelly = KellyCriterion()
        assert kelly is not None
    
    def test_calculate_kelly(self):
        """Test Kelly calculation"""
        kelly = KellyCriterion()
        
        # Test with positive expectancy
        k = kelly.calculate(win_rate=0.6, avg_win=1000, avg_loss=500)
        assert 0 < k <= 1
    
    def test_negative_expectancy(self):
        """Test with negative expectancy"""
        kelly = KellyCriterion()
        
        # Low win rate with small wins
        k = kelly.calculate(win_rate=0.3, avg_win=500, avg_loss=1000)
        assert k == 0  # Should not recommend betting
    
    def test_from_trade_history(self):
        """Test calculation from trade history"""
        kelly = KellyCriterion()
        
        trade_results = [100, -50, 80, -40, 120, -60, 90]
        k = kelly.calculate_from_history(trade_results)
        
        assert isinstance(k, float)
        assert 0 <= k <= 1


class TestPositionSizer:
    """Test Position Sizer"""
    
    def test_initialization(self):
        """Test position sizer initialization"""
        sizer = PositionSizer(account_value=100000)
        assert sizer.account_value == 100000
    
    def test_fixed_fractional(self):
        """Test fixed fractional position sizing"""
        sizer = PositionSizer(account_value=100000)
        
        size = sizer.calculate_position_size(
            current_price=100,
            method=PositionSizeMethod.FIXED_FRACTIONAL,
            risk_per_trade=0.02
        )
        
        assert size > 0
        assert size * 100 <= 100000 * 0.02 * 5  # Reasonable bounds
    
    def test_kelly_method(self):
        """Test Kelly-based position sizing"""
        sizer = PositionSizer(account_value=100000)
        
        size = sizer.calculate_position_size(
            current_price=100,
            method=PositionSizeMethod.KELLY,
            kelly_fraction=0.5,
            win_rate=0.6,
            avg_win=1000,
            avg_loss=500
        )
        
        assert size > 0
    
    def test_risk_based(self):
        """Test risk-based position sizing"""
        sizer = PositionSizer(account_value=100000)
        
        size = sizer.calculate_position_size(
            current_price=100,
            method=PositionSizeMethod.RISK_BASED,
            risk_per_trade=0.02,
            stop_loss_pct=0.05
        )
        
        assert size > 0
        # Maximum loss should not exceed risk per trade
        max_loss = size * 100 * 0.05
        assert max_loss <= 100000 * 0.02
    
    def test_volatility_based(self):
        """Test volatility-based position sizing"""
        sizer = PositionSizer(account_value=100000)
        
        size = sizer.calculate_position_size(
            current_price=100,
            method=PositionSizeMethod.VOLATILITY_BASED,
            risk_per_trade=0.02,
            volatility=0.03
        )
        
        assert size > 0


class TestStopLossManager:
    """Test Stop Loss Manager"""
    
    def test_initialization(self):
        """Test stop loss manager initialization"""
        manager = StopLossManager()
        assert len(manager.active_stops) == 0
    
    def test_fixed_percentage_stop(self):
        """Test fixed percentage stop loss"""
        manager = StopLossManager()
        
        stop_price = manager.calculate_stop_loss(
            entry_price=100,
            stop_type=StopLossType.FIXED_PERCENTAGE,
            stop_percentage=0.05,
            is_long=True
        )
        
        assert stop_price == 95.0
    
    def test_fixed_percentage_short(self):
        """Test fixed percentage stop for short position"""
        manager = StopLossManager()
        
        stop_price = manager.calculate_stop_loss(
            entry_price=100,
            stop_type=StopLossType.FIXED_PERCENTAGE,
            stop_percentage=0.05,
            is_long=False
        )
        
        assert stop_price == 105.0
    
    def test_atr_stop(self):
        """Test ATR-based stop loss"""
        manager = StopLossManager()
        
        stop_price = manager.calculate_stop_loss(
            entry_price=100,
            stop_type=StopLossType.ATR,
            atr_value=2.0,
            atr_multiplier=2.0,
            is_long=True
        )
        
        assert stop_price == 96.0
    
    def test_trailing_stop(self):
        """Test trailing stop loss"""
        manager = StopLossManager()
        
        # Initial stop
        stop_price = manager.calculate_stop_loss(
            entry_price=100,
            stop_type=StopLossType.TRAILING,
            stop_percentage=0.05,
            is_long=True
        )
        
        assert stop_price == 95.0
        
        # Update with higher price (should trail up)
        new_stop = manager.update_trailing_stop(
            position_id="TEST_1",
            current_price=110,
            trailing_pct=0.05,
            is_long=True
        )
        
        assert new_stop > stop_price
    
    def test_check_stop_triggered(self):
        """Test stop loss trigger check"""
        manager = StopLossManager()
        
        manager.set_stop_loss("TEST_1", 95.0)
        
        # Price above stop - not triggered
        assert not manager.check_stop_triggered("TEST_1", 100.0, is_long=True)
        
        # Price below stop - triggered
        assert manager.check_stop_triggered("TEST_1", 94.0, is_long=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

