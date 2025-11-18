"""
Position Manager Module

Tracks and manages active trading positions with real-time P&L monitoring,
stop-loss management, and trailing stops.
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """Position lifecycle states"""
    OPENING = "opening"  # Order submitted but not filled
    OPEN = "open"  # Position active
    CLOSING = "closing"  # Exit order submitted
    CLOSED = "closed"  # Position fully exited


@dataclass
class Position:
    """
    Represents an active trading position

    Tracks all position details including entry, current P&L,
    stop-loss levels, and position state.
    """
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    strategy_name: str
    order_id: int

    # Current state
    state: PositionState = PositionState.OPEN
    current_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

    # Stop-loss management
    stop_loss_price: Optional[float] = None
    stop_loss_order_id: Optional[int] = None
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.05  # 5%
    highest_price: Optional[float] = None  # For trailing stop

    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0

    # Exit details
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    def update_current_price(self, price: float):
        """Update current price and recalculate P&L"""
        self.current_price = price
        self.last_update = datetime.now()

        # Update unrealized P&L
        self.unrealized_pnl = (price - self.entry_price) * self.quantity - self.commission

        # Update highest price for trailing stop
        if self.highest_price is None or price > self.highest_price:
            self.highest_price = price

    def get_pnl_pct(self) -> float:
        """Get P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100

    def calculate_trailing_stop(self) -> Optional[float]:
        """Calculate current trailing stop price"""
        if not self.trailing_stop_enabled or self.highest_price is None:
            return None

        trailing_stop = self.highest_price * (1 - self.trailing_stop_pct)
        return trailing_stop

    def is_stop_triggered(self) -> bool:
        """Check if stop-loss is triggered"""
        if self.current_price == 0:
            return False

        # Fixed stop-loss
        if self.stop_loss_price and self.current_price <= self.stop_loss_price:
            return True

        # Trailing stop
        trailing_stop = self.calculate_trailing_stop()
        if trailing_stop and self.current_price <= trailing_stop:
            return True

        return False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'strategy_name': self.strategy_name,
            'order_id': self.order_id,
            'state': self.state.value,
            'current_price': self.current_price,
            'last_update': self.last_update.isoformat(),
            'stop_loss_price': self.stop_loss_price,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'pnl_pct': self.get_pnl_pct(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason
        }


class PositionManager:
    """
    Manages all active trading positions

    Features:
    - Position tracking and lifecycle management
    - Real-time P&L monitoring
    - Stop-loss monitoring and execution
    - Trailing stop updates
    - Position limits enforcement
    - Performance analytics

    Example:
        >>> pm = PositionManager(broker=ib_connector, max_positions=5)
        >>> pm.add_position(
        ...     symbol='AAPL',
        ...     quantity=100,
        ...     entry_price=150.0,
        ...     strategy_name='MA_Crossover',
        ...     order_id=12345,
        ...     stop_loss_price=142.5
        ... )
        >>> pm.update_position_prices({'AAPL': 155.0})
        >>> if pm.check_stop_losses():
        ...     print("Stop-loss triggered!")
    """

    def __init__(
        self,
        broker,  # IBConnector instance
        max_positions: int = 5,
        alert_manager=None,
        enable_trailing_stops: bool = True,
        trailing_stop_pct: float = 0.05
    ):
        """
        Initialize Position Manager

        Args:
            broker: IBConnector for order placement
            max_positions: Maximum concurrent positions
            alert_manager: AlertManager for notifications
            enable_trailing_stops: Enable trailing stops
            trailing_stop_pct: Trailing stop percentage
        """
        self.broker = broker
        self.max_positions = max_positions
        self.alert_manager = alert_manager
        self.enable_trailing_stops = enable_trailing_stops
        self.trailing_stop_pct = trailing_stop_pct

        # Position tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []

        logger.info(
            f"PositionManager initialized: max_positions={max_positions}, "
            f"trailing_stops={enable_trailing_stops}"
        )

    def add_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        strategy_name: str,
        order_id: int,
        stop_loss_price: Optional[float] = None,
        stop_loss_order_id: Optional[int] = None
    ) -> bool:
        """
        Add new position

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price
            strategy_name: Strategy that opened position
            order_id: Order ID
            stop_loss_price: Stop-loss price (optional)
            stop_loss_order_id: Stop-loss order ID (optional)

        Returns:
            True if position added successfully
        """
        if symbol in self.positions:
            logger.warning(f"{symbol}: Position already exists")
            return False

        if len(self.positions) >= self.max_positions:
            logger.error(
                f"Cannot add position: max positions ({self.max_positions}) reached"
            )
            return False

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            strategy_name=strategy_name,
            order_id=order_id,
            stop_loss_price=stop_loss_price,
            stop_loss_order_id=stop_loss_order_id,
            trailing_stop_enabled=self.enable_trailing_stops,
            trailing_stop_pct=self.trailing_stop_pct,
            current_price=entry_price
        )

        self.positions[symbol] = position

        logger.info(
            f"Position added: {symbol} - {quantity} shares @ ${entry_price:.2f} "
            f"(Strategy: {strategy_name})"
        )

        # Send alert
        if self.alert_manager:
            self._send_position_alert(position, "OPENED")

        return True

    def update_position_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions

        Args:
            prices: Dict of {symbol: current_price}
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_current_price(prices[symbol])

    def update_single_position(self, symbol: str, price: float):
        """Update price for single position"""
        if symbol in self.positions:
            self.positions[symbol].update_current_price(price)

    def check_stop_losses(self) -> List[str]:
        """
        Check all positions for stop-loss triggers

        Returns:
            List of symbols with triggered stops
        """
        triggered = []

        for symbol, position in self.positions.items():
            if position.is_stop_triggered():
                logger.warning(
                    f"STOP-LOSS TRIGGERED: {symbol} @ ${position.current_price:.2f}"
                )
                triggered.append(symbol)

        return triggered

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "Manual close"
    ) -> bool:
        """
        Close a position

        Args:
            symbol: Symbol to close
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            True if closed successfully
        """
        if symbol not in self.positions:
            logger.error(f"{symbol}: Position not found")
            return False

        position = self.positions[symbol]
        position.state = PositionState.CLOSED
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason

        # Calculate realized P&L
        position.realized_pnl = (
            (exit_price - position.entry_price) * position.quantity
            - position.commission
        )

        logger.info(
            f"Position closed: {symbol} @ ${exit_price:.2f} - "
            f"P&L: ${position.realized_pnl:.2f} ({position.get_pnl_pct():.2f}%) - "
            f"Reason: {reason}"
        )

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        # Send alert
        if self.alert_manager:
            self._send_position_alert(position, "CLOSED")

        return True

    def has_position(self, symbol: str) -> bool:
        """Check if we have an active position in symbol"""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)

    def position_count(self) -> int:
        """Get number of active positions"""
        return len(self.positions)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions"""
        return self.positions.copy()

    def get_total_exposure(self) -> float:
        """Get total $ exposure across all positions"""
        total = sum(
            pos.current_price * pos.quantity
            for pos in self.positions.values()
        )
        return total

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions"""
        return sum(pos.realized_pnl for pos in self.closed_positions)

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'active_positions': len(self.positions),
            'total_exposure': self.get_total_exposure(),
            'unrealized_pnl': self.get_total_unrealized_pnl(),
            'realized_pnl': self.get_total_realized_pnl(),
            'total_pnl': self.get_total_unrealized_pnl() + self.get_total_realized_pnl(),
            'positions': [pos.to_dict() for pos in self.positions.values()]
        }

    def update_trailing_stops(self) -> Dict[str, float]:
        """
        Update trailing stops for all positions

        Returns:
            Dict of {symbol: new_stop_price}
        """
        updated = {}

        for symbol, position in self.positions.items():
            if not position.trailing_stop_enabled:
                continue

            new_stop = position.calculate_trailing_stop()

            if new_stop and (
                position.stop_loss_price is None
                or new_stop > position.stop_loss_price
            ):
                old_stop = position.stop_loss_price
                position.stop_loss_price = new_stop
                updated[symbol] = new_stop

                logger.info(
                    f"Trailing stop updated: {symbol} - "
                    f"${old_stop:.2f} -> ${new_stop:.2f}"
                )

                # Update stop order with broker
                # self._update_stop_order(symbol, new_stop)

        return updated

    def _send_position_alert(self, position: Position, event: str):
        """Send position alert"""
        if not self.alert_manager:
            return

        try:
            if event == "OPENED":
                emoji = "🟢"
                message = (
                    f"{emoji} POSITION OPENED\n"
                    f"Symbol: {position.symbol}\n"
                    f"Quantity: {position.quantity}\n"
                    f"Entry: ${position.entry_price:.2f}\n"
                    f"Strategy: {position.strategy_name}\n"
                    f"Stop-Loss: ${position.stop_loss_price:.2f}" if position.stop_loss_price else ""
                )
            elif event == "CLOSED":
                emoji = "🔴" if position.realized_pnl < 0 else "🟢"
                message = (
                    f"{emoji} POSITION CLOSED\n"
                    f"Symbol: {position.symbol}\n"
                    f"Exit: ${position.exit_price:.2f}\n"
                    f"P&L: ${position.realized_pnl:.2f} ({position.get_pnl_pct():.2f}%)\n"
                    f"Reason: {position.exit_reason}"
                )
            else:
                return

            self.alert_manager.send_alert(
                level="INFO",
                message=message,
                channels=['telegram']
            )
        except Exception as e:
            logger.error(f"Failed to send position alert: {e}")

    def sync_with_broker(self):
        """
        Sync positions with broker

        Useful for recovery after restart or manual position adjustments
        """
        try:
            broker_positions = self.broker.get_positions()

            for bp in broker_positions:
                symbol = bp['symbol']
                quantity = int(bp['position'])

                if quantity == 0:
                    continue

                if symbol not in self.positions:
                    logger.warning(
                        f"Found untracked position in broker: {symbol} - {quantity} shares"
                    )
                    # Could add it as unknown position
                    # self.add_position(
                    #     symbol=symbol,
                    #     quantity=quantity,
                    #     entry_price=bp['avgCost'],
                    #     strategy_name='MANUAL',
                    #     order_id=0
                    # )

        except Exception as e:
            logger.error(f"Failed to sync with broker: {e}")
