"""
Execution module for automated order execution and position management.

This module contains the core components required for production trading:
- OrderExecutor: Translates strategy signals into actual market orders
- TradingLoop: Continuous event-driven trading loop
- PositionManager: Tracks and manages active positions
"""

from .order_executor import OrderExecutor, OrderStatus, ExecutionResult
from .position_manager import PositionManager, Position, PositionState
from .trading_loop import TradingLoop, LoopState

__all__ = [
    'OrderExecutor',
    'OrderStatus',
    'ExecutionResult',
    'PositionManager',
    'Position',
    'PositionState',
    'TradingLoop',
    'LoopState'
]
