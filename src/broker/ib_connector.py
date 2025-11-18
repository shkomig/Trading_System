"""
חיבור ל-Interactive Brokers

מודול המתחבר ל-TWS או IB Gateway ומאפשר מסחר ואחזור נתונים.
"""

try:
    from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder, util
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False

import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IBConnector:
    """
    מחלקה לניהול חיבור ל-Interactive Brokers
    
    תומכת במסחר אמיתי ודמה (Paper Trading).
    
    Example:
        >>> connector = IBConnector(host='127.0.0.1', port=7497, is_paper=True)
        >>> if connector.connect():
        ...     data = connector.get_historical_data('AAPL', '1 Y', '1 day')
        ...     order_id = connector.place_market_order('AAPL', 100, 'BUY')
        ...     connector.disconnect()
    """
    
    def __init__(self,
                 host: str = '127.0.0.1',
                 port: int = 7497,
                 client_id: int = 1,
                 is_paper: bool = True):
        """
        אתחול חיבור ל-IB
        
        Args:
            host: כתובת TWS/Gateway
            port: פורט (7497 paper, 7496 live)
            client_id: מזהה לקוח
            is_paper: האם זה חשבון דמה
        """
        if not IB_INSYNC_AVAILABLE:
            logger.error("ib_insync not installed. Install with: pip install ib-insync")
            raise ImportError("ib_insync is required. Install with: pip install ib-insync")
        
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_paper = is_paper
        
        logger.info(f"IBConnector initialized (host={host}, port={port}, paper={is_paper})")
    
    def connect(self) -> bool:
        """
        התחברות ל-IB
        
        Returns:
            True אם ההתחברות הצליחה
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            account_type = "Paper Trading" if self.is_paper else "Live Trading"
            logger.info(f"Connected to IB {account_type} account")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            logger.error("Make sure TWS/IB Gateway is running and API is enabled")
            return False
    
    def disconnect(self):
        """ניתוק מ-IB"""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")
    
    def is_connected(self) -> bool:
        """בדיקת חיבור"""
        return self.ib.isConnected()
    
    def get_account_info(self) -> Dict:
        """
        קבלת מידע על החשבון
        
        Returns:
            מילון עם פרטי החשבון
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        account_values = self.ib.accountValues()
        info = {}
        
        for av in account_values:
            info[av.tag] = av.value
        
        logger.info(f"Retrieved account info: {len(info)} values")
        return info
    
    def get_positions(self) -> List[Dict]:
        """
        קבלת פוזיציות פתוחות
        
        Returns:
            רשימת פוזיציות
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return []
        
        positions = []
        
        for position in self.ib.positions():
            positions.append({
                'symbol': position.contract.symbol,
                'position': position.position,
                'avgCost': position.avgCost,
                'contract': position.contract
            })
        
        logger.info(f"Retrieved {len(positions)} positions")
        return positions
    
    def place_market_order(self,
                           symbol: str,
                           quantity: int,
                           action: str = 'BUY',
                           exchange: str = 'SMART') -> Optional[int]:
        """
        הגשת פקודת שוק
        
        Args:
            symbol: סימול המניה
            quantity: כמות
            action: 'BUY' או 'SELL'
            exchange: בורסה
            
        Returns:
            מזהה פקודה או None
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Market order placed: {action} {quantity} {symbol} @ Market")
            return trade.order.orderId
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None
    
    def place_limit_order(self,
                          symbol: str,
                          quantity: int,
                          limit_price: float,
                          action: str = 'BUY',
                          exchange: str = 'SMART') -> Optional[int]:
        """
        הגשת פקודת לימיט
        
        Args:
            symbol: סימול המניה
            quantity: כמות
            limit_price: מחיר לימיט
            action: 'BUY' או 'SELL'
            exchange: בורסה
            
        Returns:
            מזהה פקודה או None
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            order = LimitOrder(action, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Limit order placed: {action} {quantity} {symbol} @ ${limit_price}")
            return trade.order.orderId
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        ביטול פקודה
        
        Args:
            order_id: מזהה פקודה
            
        Returns:
            True אם הביטול הצליח
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return False
        
        try:
            # מציאת הפקודה
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True
            
            logger.warning(f"Order {order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_historical_data(self,
                            symbol: str,
                            duration: str = '1 Y',
                            bar_size: str = '1 day',
                            what_to_show: str = 'TRADES',
                            exchange: str = 'SMART') -> pd.DataFrame:
        """
        קבלת נתונים היסטוריים
        
        Args:
            symbol: סימול
            duration: משך זמן ('1 Y', '6 M', '1 W')
            bar_size: גודל נר ('1 day', '1 hour', '5 mins')
            what_to_show: סוג נתונים ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            exchange: בורסה
            
        Returns:
            DataFrame עם נתונים
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return pd.DataFrame()
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No historical data received for {symbol}")
                return pd.DataFrame()
            
            # המרה ל-DataFrame
            df = util.df(bars)
            df = df.set_index('date')
            
            # שינוי שמות עמודות לאותיות קטנות
            df.columns = df.columns.str.lower()
            
            logger.info(f"Retrieved {len(df)} bars of historical data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str, exchange: str = 'SMART') -> Optional[float]:
        """
        קבלת מחיר נוכחי
        
        Args:
            symbol: סימול
            exchange: בורסה
            
        Returns:
            מחיר נוכחי או None
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)  # המתנה לקבלת נתונים
            
            if ticker.marketPrice():
                price = ticker.marketPrice()
                logger.debug(f"Current price for {symbol}: ${price}")
                return price
            else:
                logger.warning(f"No price data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return None
    
    def get_open_orders(self) -> List[Dict]:
        """
        קבלת פקודות פתוחות
        
        Returns:
            רשימת פקודות
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return []
        
        orders = []
        
        for trade in self.ib.openTrades():
            orders.append({
                'orderId': trade.order.orderId,
                'symbol': trade.contract.symbol,
                'action': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'orderType': trade.order.orderType,
                'limitPrice': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                'status': trade.orderStatus.status
            })
        
        logger.info(f"Retrieved {len(orders)} open orders")
        return orders
    
    def subscribe_realtime_bars(
        self,
        symbol: str,
        callback: callable,
        bar_size: int = 5,
        what_to_show: str = 'TRADES',
        exchange: str = 'SMART'
    ):
        """
        Subscribe to real-time 5-second bars

        Critical Fix: Enables continuous real-time data streaming
        instead of periodic snapshots.

        Args:
            symbol: Stock symbol
            callback: Function to call when new bar arrives
                     callback(symbol: str, bar: dict)
            bar_size: Bar size in seconds (5 only)
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            exchange: Exchange

        Returns:
            RealTimeBarsList object (can be cancelled)

        Example:
            >>> def on_bar(symbol, bar):
            ...     print(f"{symbol}: {bar['close']}")
            >>> bars = connector.subscribe_realtime_bars('AAPL', on_bar)
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None

        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)

            # Request real-time bars
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=bar_size,
                whatToShow=what_to_show,
                useRTH=True  # Regular trading hours only
            )

            # Register callback
            def wrapper(bars_list, has_new_bar):
                """Wrapper to extract bar data and call user callback"""
                if has_new_bar and bars_list:
                    latest_bar = bars_list[-1]
                    bar_data = {
                        'time': latest_bar.time,
                        'open': latest_bar.open_,
                        'high': latest_bar.high,
                        'low': latest_bar.low,
                        'close': latest_bar.close,
                        'volume': latest_bar.volume,
                        'wap': latest_bar.wap,  # Weighted average price
                        'count': latest_bar.count
                    }
                    callback(symbol, bar_data)

            bars.updateEvent += wrapper

            logger.info(f"Subscribed to real-time bars for {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Failed to subscribe to real-time bars for {symbol}: {e}")
            return None

    def subscribe_market_data(
        self,
        symbol: str,
        callback: callable,
        exchange: str = 'SMART'
    ):
        """
        Subscribe to tick-by-tick market data

        Provides real-time price updates (ticks) as they occur.

        Args:
            symbol: Stock symbol
            callback: Function to call on price update
                     callback(symbol: str, ticker: dict)
            exchange: Exchange

        Returns:
            Ticker object

        Example:
            >>> def on_tick(symbol, ticker):
            ...     print(f"{symbol}: ${ticker['last']}")
            >>> ticker = connector.subscribe_market_data('AAPL', on_tick)
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None

        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)

            # Register callback
            def wrapper(ticker_obj):
                """Extract ticker data and call user callback"""
                ticker_data = {
                    'symbol': symbol,
                    'time': ticker_obj.time,
                    'bid': ticker_obj.bid,
                    'ask': ticker_obj.ask,
                    'last': ticker_obj.last,
                    'bid_size': ticker_obj.bidSize,
                    'ask_size': ticker_obj.askSize,
                    'last_size': ticker_obj.lastSize,
                    'volume': ticker_obj.volume,
                    'high': ticker_obj.high,
                    'low': ticker_obj.low,
                    'close': ticker_obj.close
                }
                callback(symbol, ticker_data)

            ticker.updateEvent += wrapper

            logger.info(f"Subscribed to market data for {symbol}")
            return ticker

        except Exception as e:
            logger.error(f"Failed to subscribe to market data for {symbol}: {e}")
            return None

    def unsubscribe_realtime_bars(self, bars):
        """Cancel real-time bars subscription"""
        try:
            if bars:
                self.ib.cancelRealTimeBars(bars)
                logger.info("Real-time bars subscription cancelled")
        except Exception as e:
            logger.error(f"Failed to unsubscribe real-time bars: {e}")

    def unsubscribe_market_data(self, ticker):
        """Cancel market data subscription"""
        try:
            if ticker:
                self.ib.cancelMktData(ticker.contract)
                logger.info("Market data subscription cancelled")
        except Exception as e:
            logger.error(f"Failed to unsubscribe market data: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

