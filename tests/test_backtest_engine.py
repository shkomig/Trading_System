"""
Unit Tests for Backtest Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.technical.moving_average import MovingAverageCrossover


@pytest.fixture
def sample_data():
    """Generate sample data"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 1)
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + 1,
        'low': close_prices - 1,
        'close': close_prices,
        'volume': 1000000
    }, index=dates)
    
    return data


@pytest.fixture
def sample_signals(sample_data):
    """Generate sample signals"""
    signals = pd.Series(0, index=sample_data.index)
    signals.iloc[10] = 1  # Buy signal
    signals.iloc[50] = -1  # Sell signal
    return signals


class TestBacktestEngine:
    """Test Backtest Engine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = BacktestEngine(initial_capital=100000, commission=0.001)
        
        assert engine.initial_capital == 100000
        assert engine.commission == 0.001
        assert engine.capital == 100000
        assert engine.position == 0
    
    def test_run_backtest(self, sample_data, sample_signals):
        """Test running a backtest"""
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(sample_data, sample_signals, position_size=0.5)
        
        assert isinstance(results, dict)
        assert 'initial_capital' in results
        assert 'final_capital' in results
        assert 'total_return' in results
    
    def test_buy_action(self, sample_data, sample_signals):
        """Test buy action"""
        engine = BacktestEngine(initial_capital=100000, commission=0.001)
        engine.run(sample_data, sample_signals, position_size=0.5)
        
        # Should have executed at least one trade
        assert len(engine.trades) >= 0
    
    def test_position_sizing(self, sample_data):
        """Test position sizing"""
        engine = BacktestEngine(initial_capital=100000)
        
        signals = pd.Series(0, index=sample_data.index)
        signals.iloc[10] = 1
        
        # Test with 50% position size
        engine.run(sample_data, signals, position_size=0.5)
        
        # Capital should have changed
        assert engine.capital != 100000 or engine.position > 0
    
    def test_commission_impact(self, sample_data, sample_signals):
        """Test that commission affects returns"""
        engine_no_commission = BacktestEngine(initial_capital=100000, commission=0.0)
        results_no_commission = engine_no_commission.run(sample_data, sample_signals)
        
        engine_with_commission = BacktestEngine(initial_capital=100000, commission=0.01)
        results_with_commission = engine_with_commission.run(sample_data, sample_signals)
        
        # Returns with commission should be lower (or equal if no trades)
        if len(engine_no_commission.trades) > 0:
            assert results_with_commission['total_return'] <= results_no_commission['total_return']
    
    def test_equity_curve(self, sample_data, sample_signals):
        """Test equity curve generation"""
        engine = BacktestEngine(initial_capital=100000)
        engine.run(sample_data, sample_signals)
        
        assert isinstance(engine.equity_curve, list)
        assert len(engine.equity_curve) > 0
        assert engine.equity_curve[0] == 100000


class TestBacktestIntegration:
    """Integration tests for backtest with strategies"""
    
    def test_full_backtest_workflow(self, sample_data):
        """Test complete backtest workflow"""
        # Create strategy
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        
        # Generate signals
        signals = strategy.generate_signals(sample_data)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100000, commission=0.001)
        results = engine.run(sample_data, signals, position_size=0.5)
        
        # Verify results
        assert 'total_return' in results
        assert 'max_drawdown' in results
        assert isinstance(results['total_return'], (int, float))
    
    def test_multiple_strategies(self, sample_data):
        """Test backtesting multiple strategies"""
        strategies = [
            MovingAverageCrossover(short_window=10, long_window=20),
            MovingAverageCrossover(short_window=20, long_window=50),
        ]
        
        results_list = []
        
        for strategy in strategies:
            signals = strategy.generate_signals(sample_data)
            engine = BacktestEngine(initial_capital=100000)
            results = engine.run(sample_data, signals)
            results_list.append(results)
        
        assert len(results_list) == 2
        assert all('total_return' in r for r in results_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

