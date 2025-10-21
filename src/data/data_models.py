"""
מודלי נתונים בסיסיים למערכת המסחר

מגדיר dataclasses וטיפוסים למבני נתונים שונים במערכת.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class OrderType(Enum):
    """סוגי פקודות"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderAction(Enum):
    """פעולות פקודה"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """סטטוס פקודה"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


class PositionSide(Enum):
    """צד פוזיציה"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class Trade:
    """
    מודל לעסקה בודדת
    
    Attributes:
        symbol: סימול המניה
        strategy: שם האסטרטגיה
        action: BUY או SELL
        quantity: כמות מניות
        entry_price: מחיר כניסה
        exit_price: מחיר יציאה
        entry_time: זמן כניסה
        exit_time: זמן יציאה
        pnl: רווח/הפסד
        commission: עמלות
        slippage: slippage
        notes: הערות נוספות
    """
    symbol: str
    strategy: str
    action: OrderAction
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    notes: str = ""
    
    def calculate_pnl(self) -> float:
        """חישוב רווח/הפסד"""
        if self.exit_price is None:
            return 0.0
        
        if self.action == OrderAction.BUY:
            pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL (short)
            pnl = (self.entry_price - self.exit_price) * self.quantity
        
        # ניכוי עמלות ו-slippage
        pnl -= (self.commission + self.slippage)
        
        self.pnl = pnl
        return pnl
    
    def duration_minutes(self) -> Optional[float]:
        """משך הזמן של העסקה בדקות"""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 60
    
    def to_dict(self) -> Dict:
        """המרה ל-dictionary"""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'action': self.action.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'commission': self.commission,
            'slippage': self.slippage,
            'notes': self.notes
        }


@dataclass
class Position:
    """
    מודל לפוזיציה פתוחה
    
    Attributes:
        symbol: סימול המניה
        side: LONG או SHORT
        quantity: כמות מניות
        entry_price: מחיר כניסה ממוצע
        current_price: מחיר נוכחי
        stop_loss: מחיר stop loss
        take_profit: מחיר take profit
        strategy: שם האסטרטגיה
        entry_time: זמן פתיחת הפוזיציה
    """
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """ערך שוק נוכחי"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """רווח/הפסד לא ממומש"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """אחוז רווח/הפסד לא ממומש"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
    
    def should_stop_loss(self) -> bool:
        """בדיקה האם צריך לסגור ב-stop loss"""
        if self.stop_loss is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """בדיקה האם צריך לסגור ב-take profit"""
        if self.take_profit is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit
    
    def to_dict(self) -> Dict:
        """המרה ל-dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percent': self.unrealized_pnl_percent,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy': self.strategy,
            'entry_time': self.entry_time.isoformat()
        }


@dataclass
class Order:
    """
    מודל לפקודת מסחר
    
    Attributes:
        symbol: סימול המניה
        action: BUY או SELL
        quantity: כמות
        order_type: סוג פקודה
        limit_price: מחיר לימיט (אם רלוונטי)
        stop_price: מחיר סטופ (אם רלוונטי)
        status: סטטוס הפקודה
        order_id: מזהה פקודה
        filled_quantity: כמות שמולאה
        avg_fill_price: מחיר ממוצע למילוי
        created_time: זמן יצירה
        filled_time: זמן מילוי
    """
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    filled_time: Optional[datetime] = None
    
    @property
    def is_filled(self) -> bool:
        """בדיקה האם הפקודה מולאה במלואה"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """בדיקה האם הפקודה פעילה"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    def to_dict(self) -> Dict:
        """המרה ל-dictionary"""
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'order_id': self.order_id,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_time': self.created_time.isoformat(),
            'filled_time': self.filled_time.isoformat() if self.filled_time else None
        }


@dataclass
class PerformanceMetrics:
    """
    מטריקות ביצועים
    
    Attributes:
        total_return: תשואה כוללת (%)
        sharpe_ratio: Sharpe Ratio
        sortino_ratio: Sortino Ratio
        max_drawdown: Max Drawdown (%)
        win_rate: אחוז ניצחונות
        profit_factor: Profit Factor
        total_trades: מספר עסקאות כולל
        avg_trade_duration: משך עסקה ממוצע (דקות)
        expectancy: Expectancy
    """
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    expectancy: float = 0.0
    
    def to_dict(self) -> Dict:
        """המרה ל-dictionary"""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'avg_trade_duration': self.avg_trade_duration,
            'expectancy': self.expectancy
        }


@dataclass
class AccountInfo:
    """
    מידע על חשבון מסחר
    
    Attributes:
        account_id: מזהה חשבון
        balance: יתרה
        equity: Equity
        buying_power: כוח קנייה
        margin_used: מרווח בשימוש
        positions_value: ערך פוזיציות
        cash: מזומן פנוי
        unrealized_pnl: רווח/הפסד לא ממומש
        realized_pnl: רווח/הפסד ממומש
    """
    account_id: str
    balance: float = 0.0
    equity: float = 0.0
    buying_power: float = 0.0
    margin_used: float = 0.0
    positions_value: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def margin_usage_percent(self) -> float:
        """אחוז שימוש במרווח"""
        if self.equity == 0:
            return 0.0
        return (self.margin_used / self.equity) * 100
    
    def to_dict(self) -> Dict:
        """המרה ל-dictionary"""
        return {
            'account_id': self.account_id,
            'balance': self.balance,
            'equity': self.equity,
            'buying_power': self.buying_power,
            'margin_used': self.margin_used,
            'positions_value': self.positions_value,
            'cash': self.cash,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'margin_usage_percent': self.margin_usage_percent
        }

