"""
Market Hours Validator

Validates trading hours and market open/close times.
Prevents trading during pre-market, after-hours, and holidays.
"""

import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Optional
import pytz

logger = logging.getLogger(__name__)


class MarketHoursValidator:
    """
    Validates market trading hours

    Prevents trading during:
    - Pre-market hours
    - After-hours
    - Weekends
    - Market holidays (basic)
    - First/last N minutes (optional)

    Example:
        >>> validator = MarketHoursValidator(timezone='America/New_York')
        >>> if validator.is_market_open_now():
        ...     print("Market is open!")
        >>> if validator.should_trade_now():
        ...     print("Safe to trade now")
    """

    def __init__(
        self,
        timezone: str = 'America/New_York',
        market_open: dt_time = dt_time(9, 30),
        market_close: dt_time = dt_time(16, 0),
        avoid_first_minutes: int = 10,  # Avoid first N minutes
        avoid_last_minutes: int = 10,  # Avoid last N minutes
        enable_pre_market: bool = False,
        enable_after_hours: bool = False
    ):
        """
        Initialize Market Hours Validator

        Args:
            timezone: Market timezone (e.g., 'America/New_York' for NYSE)
            market_open: Market open time (default: 9:30 AM)
            market_close: Market close time (default: 4:00 PM)
            avoid_first_minutes: Skip first N minutes after open
            avoid_last_minutes: Skip last N minutes before close
            enable_pre_market: Allow pre-market trading (4:00 AM - 9:30 AM)
            enable_after_hours: Allow after-hours trading (4:00 PM - 8:00 PM)
        """
        self.timezone = pytz.timezone(timezone)
        self.market_open = market_open
        self.market_close = market_close
        self.avoid_first_minutes = avoid_first_minutes
        self.avoid_last_minutes = avoid_last_minutes
        self.enable_pre_market = enable_pre_market
        self.enable_after_hours = enable_after_hours

        # Pre-market and after-hours times
        self.pre_market_open = dt_time(4, 0)
        self.after_hours_close = dt_time(20, 0)

        logger.info(
            f"MarketHoursValidator initialized: "
            f"open={market_open}, close={market_close}, "
            f"avoid_first={avoid_first_minutes}min, avoid_last={avoid_last_minutes}min"
        )

    def is_market_open_now(self) -> bool:
        """
        Check if market is currently open

        Returns:
            True if market is open for regular trading hours
        """
        now = datetime.now(self.timezone)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            logger.debug("Market closed: Weekend")
            return False

        # Check if holiday (basic check - could be enhanced with calendar)
        if self._is_market_holiday(now):
            logger.debug("Market closed: Holiday")
            return False

        # Check time
        current_time = now.time()

        # Regular hours
        if self.market_open <= current_time < self.market_close:
            return True

        # Pre-market
        if self.enable_pre_market and self.pre_market_open <= current_time < self.market_open:
            return True

        # After-hours
        if self.enable_after_hours and self.market_close <= current_time < self.after_hours_close:
            return True

        logger.debug(f"Market closed: Outside trading hours ({current_time})")
        return False

    def should_trade_now(self) -> bool:
        """
        Check if it's safe to trade now

        More conservative than is_market_open_now():
        - Avoids first N minutes after open (high volatility)
        - Avoids last N minutes before close (low liquidity)

        Returns:
            True if it's a good time to trade
        """
        if not self.is_market_open_now():
            return False

        now = datetime.now(self.timezone)
        current_time = now.time()

        # Calculate safe trading window
        safe_start = (
            datetime.combine(now.date(), self.market_open) +
            timedelta(minutes=self.avoid_first_minutes)
        ).time()

        safe_end = (
            datetime.combine(now.date(), self.market_close) -
            timedelta(minutes=self.avoid_last_minutes)
        ).time()

        # Check if in safe window
        if safe_start <= current_time < safe_end:
            return True
        else:
            logger.debug(
                f"Outside safe trading window: {safe_start} - {safe_end} "
                f"(current: {current_time})"
            )
            return False

    def get_next_market_open(self) -> datetime:
        """
        Get next market open time

        Returns:
            Datetime of next market open
        """
        now = datetime.now(self.timezone)
        next_open = now

        # If weekend, move to Monday
        if now.weekday() == 5:  # Saturday
            next_open = now + timedelta(days=2)
        elif now.weekday() == 6:  # Sunday
            next_open = now + timedelta(days=1)
        # If after market close, move to next day
        elif now.time() >= self.market_close:
            next_open = now + timedelta(days=1)

        # Set to market open time
        next_open = next_open.replace(
            hour=self.market_open.hour,
            minute=self.market_open.minute,
            second=0,
            microsecond=0
        )

        # Skip if holiday
        while self._is_market_holiday(next_open):
            next_open = next_open + timedelta(days=1)

        return next_open

    def get_next_market_close(self) -> datetime:
        """
        Get next market close time

        Returns:
            Datetime of next market close
        """
        now = datetime.now(self.timezone)

        # If before market close today, return today's close
        if now.time() < self.market_close and now.weekday() < 5:
            next_close = now.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0
            )
        else:
            # Return next trading day's close
            next_open = self.get_next_market_open()
            next_close = next_open.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute
            )

        return next_close

    def time_until_market_open(self) -> Optional[timedelta]:
        """
        Get time remaining until market opens

        Returns:
            Timedelta if market is closed, None if market is open
        """
        if self.is_market_open_now():
            return None

        next_open = self.get_next_market_open()
        now = datetime.now(self.timezone)

        return next_open - now

    def time_until_market_close(self) -> Optional[timedelta]:
        """
        Get time remaining until market closes

        Returns:
            Timedelta if market is open, None if market is closed
        """
        if not self.is_market_open_now():
            return None

        next_close = self.get_next_market_close()
        now = datetime.now(self.timezone)

        return next_close - now

    def _is_market_holiday(self, date: datetime) -> bool:
        """
        Check if date is a market holiday

        This is a basic implementation. For production, use a proper
        market calendar library like pandas_market_calendars.

        Args:
            date: Date to check

        Returns:
            True if holiday
        """
        # Basic US market holidays (2025)
        holidays = [
            datetime(2025, 1, 1),   # New Year's Day
            datetime(2025, 1, 20),  # MLK Day
            datetime(2025, 2, 17),  # Presidents' Day
            datetime(2025, 4, 18),  # Good Friday
            datetime(2025, 5, 26),  # Memorial Day
            datetime(2025, 7, 4),   # Independence Day
            datetime(2025, 9, 1),   # Labor Day
            datetime(2025, 11, 27), # Thanksgiving
            datetime(2025, 12, 25), # Christmas
        ]

        date_only = date.date()

        for holiday in holidays:
            if date_only == holiday.date():
                return True

        return False

    def get_trading_status(self) -> dict:
        """
        Get comprehensive trading status

        Returns:
            Dict with status information
        """
        now = datetime.now(self.timezone)
        is_open = self.is_market_open_now()
        should_trade = self.should_trade_now()

        status = {
            'current_time': now.isoformat(),
            'timezone': str(self.timezone),
            'is_market_open': is_open,
            'should_trade': should_trade,
            'is_weekend': now.weekday() >= 5,
            'is_holiday': self._is_market_holiday(now),
            'current_time_str': now.strftime('%Y-%m-%d %H:%M:%S %Z')
        }

        if is_open:
            time_until_close = self.time_until_market_close()
            if time_until_close:
                status['time_until_close'] = str(time_until_close)
                status['minutes_until_close'] = int(time_until_close.total_seconds() / 60)
        else:
            time_until_open = self.time_until_market_open()
            if time_until_open:
                status['time_until_open'] = str(time_until_open)
                status['minutes_until_open'] = int(time_until_open.total_seconds() / 60)
                status['next_market_open'] = self.get_next_market_open().isoformat()

        return status


# Simple fallback validator if pytz is not available
class SimpleMarketHoursValidator:
    """
    Simplified market hours validator without timezone support

    Use this if pytz is not installed.
    """

    def __init__(self):
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)

    def is_market_open_now(self) -> bool:
        """Check if market is open (basic check)"""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Time check
        current_time = now.time()
        if self.market_open <= current_time < self.market_close:
            return True

        return False

    def should_trade_now(self) -> bool:
        """Same as is_market_open_now for simple validator"""
        return self.is_market_open_now()
