"""
Unit Tests for Trading Strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.technical.moving_average import MovingAverageCrossover, TripleMovingAverage
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
from src.strategies.technical.momentum import MomentumStrategy, DualMomentumStrategy


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(200) * 2)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(200) * 0.5,
        'high': close_prices + np.abs(np.random.randn(200)) * 2,
        'low': close_prices - np.abs(np.random.randn(200)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 200)
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


class TestMovingAverageCrossover:
    """Test Moving Average Crossover strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = MovingAverageCrossover(short_window=20, long_window=50)
        assert strategy.short_window == 20
        assert strategy.long_window == 50
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = MovingAverageCrossover(short_window=20, long_window=50)
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})
    
    def test_calculate_indicators(self, sample_data):
        """Test indicator calculation"""
        strategy = MovingAverageCrossover(short_window=20, long_window=50)
        indicators = strategy.calculate_indicators(sample_data)
        
        assert 'sma_short' in indicators.columns
        assert 'sma_long' in indicators.columns
        assert len(indicators) == len(sample_data)


class TestTripleMovingAverage:
    """Test Triple MA strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = TripleMovingAverage(fast_window=10, medium_window=20, slow_window=50)
        assert strategy.fast_window == 10
        assert strategy.medium_window == 20
        assert strategy.slow_window == 50
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = TripleMovingAverage(fast_window=10, medium_window=20, slow_window=50)
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)


class TestRSI_MACD_Strategy:
    """Test RSI + MACD strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = RSI_MACD_Strategy(rsi_period=14, rsi_overbought=70, rsi_oversold=30)
        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = RSI_MACD_Strategy()
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
    
    def test_calculate_indicators(self, sample_data):
        """Test indicator calculation"""
        strategy = RSI_MACD_Strategy()
        indicators = strategy.calculate_indicators(sample_data)
        
        assert 'rsi' in indicators.columns
        assert 'macd' in indicators.columns
        assert 'macd_signal' in indicators.columns


class TestMomentumStrategy:
    """Test Momentum strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = MomentumStrategy(lookback_period=20)
        assert strategy.lookback_period == 20
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = MomentumStrategy(lookback_period=20)
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
    
    def test_momentum_calculation(self, sample_data):
        """Test momentum calculation"""
        strategy = MomentumStrategy(lookback_period=20)
        indicators = strategy.calculate_indicators(sample_data)
        
        assert 'momentum' in indicators.columns
        assert 'momentum_signal' in indicators.columns


class TestDualMomentumStrategy:
    """Test Dual Momentum strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = DualMomentumStrategy(short_period=50, long_period=200)
        assert strategy.short_period == 50
        assert strategy.long_period == 200
    
    def test_generate_signals(self, sample_data):
        """Test signal generation"""
        strategy = DualMomentumStrategy()
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)


def test_all_strategies_consistency(sample_data):
    """Test that all strategies return consistent output format"""
    strategies = [
        MovingAverageCrossover(),
        TripleMovingAverage(),
        RSI_MACD_Strategy(),
        MomentumStrategy(),
        DualMomentumStrategy()
    ]
    
    for strategy in strategies:
        signals = strategy.generate_signals(sample_data)
        
        # Check output type
        assert isinstance(signals, pd.Series), f"{strategy.__class__.__name__} didn't return Series"
        
        # Check length
        assert len(signals) == len(sample_data), f"{strategy.__class__.__name__} length mismatch"
        
        # Check signal values
        assert set(signals.unique()).issubset({-1, 0, 1}), f"{strategy.__class__.__name__} invalid signals"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

