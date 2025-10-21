# מדריך מקיף לבניית מערכת מסחר אוטומטית עם Cursor AI

## תוכן עניינים
1. [סקירה כללית](#סקירה-כללית)
2. [אדריכלות המערכת](#אדריכלות-המערכת)
3. [חיבור ל-Interactive Brokers](#חיבור-ל-interactive-brokers)
4. [אסטרטגיות מסחר](#אסטרטגיות-מסחר)
5. [למידת מכונה ו-Reinforcement Learning](#למידת-מכונה-ו-reinforcement-learning)
6. [Backtesting ו-Paper Trading](#backtesting-ו-paper-trading)
7. [Risk Management](#risk-management)
8. [פרומפטים מתקדמים ל-Cursor](#פרומפטים-מתקדמים-ל-cursor)
9. [מערכת למידה והתאמה](#מערכת-למידה-והתאמה)
10. [ממשק משתמש ותצוגה](#ממשק-משתמש-ותצוגה)
11. [פרוייקט התחלתי](#פרוייקט-התחלתי)

---

## סקירה כללית

### מטרת המערכת
מערכת מסחר אוטומטית מתקדמת המשלבת:
- **מסחר אמיתי ודמה** דרך Interactive Brokers Israel
- **אסטרטגיות מסחר מגוונות** (טכניות ולמידת מכונה)
- **למידה עצמית** מרווחים והפסדים
- **ניהול סיכונים מתקדם** עם Kelly Criterion
- **ממשק משתמש אינטואיטיבי** לניהול ומעקב
- **שימוש ב-API מקומיים** למודלי AI

### טכנולוגיות ליבה
- **Python 3.10+** - שפת התכנות הראשית
- **Interactive Brokers TWS API** - חיבור לברוקר
- **Pandas, NumPy** - עיבוד נתונים
- **Backtesting.py / Backtrader** - בדיקת אסטרטגיות
- **TensorFlow / PyTorch** - למידת מכונה
- **FastAPI / Streamlit** - ממשק משתמש
- **PostgreSQL / SQLite** - אחסון נתונים
- **Docker** - פריסה וניהול

---

## אדריכלות המערכת

### מבנה תיקיות מומלץ

```
trading_system/
├── src/
│   ├── broker/                  # חיבור ל-IB
│   │   ├── ib_connector.py
│   │   ├── order_manager.py
│   │   └── market_data.py
│   ├── strategies/              # אסטרטגיות מסחר
│   │   ├── technical/
│   │   │   ├── moving_average.py
│   │   │   ├── rsi_macd.py
│   │   │   └── bollinger_bands.py
│   │   ├── ml_based/
│   │   │   ├── lstm_predictor.py
│   │   │   ├── reinforcement_learning.py
│   │   │   └── ensemble_strategy.py
│   │   └── base_strategy.py
│   ├── backtesting/             # מנוע Backtesting
│   │   ├── backtest_engine.py
│   │   ├── performance_metrics.py
│   │   └── optimization.py
│   ├── risk_management/         # ניהול סיכונים
│   │   ├── position_sizing.py
│   │   ├── kelly_criterion.py
│   │   └── stop_loss_manager.py
│   ├── ml_models/               # מודלי למידת מכונה
│   │   ├── training/
│   │   ├── inference/
│   │   └── model_registry.py
│   ├── data/                    # ניהול נתונים
│   │   ├── data_fetcher.py
│   │   ├── data_processor.py
│   │   └── database.py
│   ├── learning/                # מערכת למידה
│   │   ├── performance_tracker.py
│   │   ├── strategy_optimizer.py
│   │   └── feedback_loop.py
│   └── ui/                      # ממשק משתמש
│       ├── dashboard.py
│       ├── api.py
│       └── components/
├── config/                      # קבצי הגדרות
│   ├── config.yaml
│   ├── strategies.yaml
│   └── risk_params.yaml
├── tests/                       # בדיקות
│   ├── unit/
│   ├── integration/
│   └── backtests/
├── docs/                        # תיעוד
├── data/                        # נתונים
│   ├── historical/
│   ├── live/
│   └── backtest_results/
├── models/                      # מודלים מאומנים
├── logs/                        # לוגים
├── .cursorrules               # כללים ל-Cursor
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## חיבור ל-Interactive Brokers

### התקנה והגדרה

#### 1. התקנת ספריות

```bash
pip install ib-insync ibapi pandas numpy
```

#### 2. הגדרת TWS / IB Gateway

**קובץ: `src/broker/ib_connector.py`**

```python
from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder
import logging
from typing import Optional, List
from datetime import datetime

class IBConnector:
    """
    מחלקה לניהול חיבור ל-Interactive Brokers
    תומכת במסחר אמיתי ודמה
    """
    
    def __init__(self, 
                 host: str = '127.0.0.1',
                 port: int = 7497,  # 7497 for paper, 7496 for live
                 client_id: int = 1,
                 is_paper: bool = True):
        """
        אתחול חיבור ל-IB
        
        Args:
            host: כתובת TWS/Gateway
            port: פורט (7497 דמה, 7496 אמיתי)
            client_id: מזהה לקוח
            is_paper: האם זה חשבון דמה
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_paper = is_paper
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """
        התחברות ל-IB
        
        Returns:
            True אם ההתחברות הצליחה
        """
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            account_type = "Paper" if self.is_paper else "Live"
            self.logger.info(f"Connected to IB {account_type} account")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """ניתוק מ-IB"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IB")
    
    def get_account_info(self) -> dict:
        """
        קבלת מידע על החשבון
        
        Returns:
            מילון עם פרטי החשבון
        """
        account_values = self.ib.accountValues()
        info = {}
        for av in account_values:
            info[av.tag] = av.value
        return info
    
    def get_positions(self) -> List[dict]:
        """
        קבלת פוזיציות פתוחות
        
        Returns:
            רשימת פוזיציות
        """
        positions = []
        for position in self.ib.positions():
            positions.append({
                'symbol': position.contract.symbol,
                'position': position.position,
                'avgCost': position.avgCost,
                'marketPrice': position.marketPrice,
                'marketValue': position.marketValue,
                'unrealizedPNL': position.unrealizedPNL
            })
        return positions
    
    def place_market_order(self, 
                          symbol: str, 
                          quantity: int, 
                          action: str = 'BUY') -> Optional[int]:
        """
        הגשת פקודת שוק
        
        Args:
            symbol: סימול המניה
            quantity: כמות
            action: BUY או SELL
            
        Returns:
            מזהה פקודה או None
        """
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            
            self.logger.info(f"Market order placed: {action} {quantity} {symbol}")
            return trade.order.orderId
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None
    
    def place_limit_order(self,
                         symbol: str,
                         quantity: int,
                         limit_price: float,
                         action: str = 'BUY') -> Optional[int]:
        """
        הגשת פקודת לימיט
        
        Args:
            symbol: סימול המניה
            quantity: כמות
            limit_price: מחיר לימיט
            action: BUY או SELL
            
        Returns:
            מזהה פקודה או None
        """
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            order = LimitOrder(action, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)
            
            self.logger.info(f"Limit order placed: {action} {quantity} {symbol} @ {limit_price}")
            return trade.order.orderId
        except Exception as e:
            self.logger.error(f"Failed to place limit order: {e}")
            return None
    
    def get_historical_data(self,
                           symbol: str,
                           duration: str = '1 Y',
                           bar_size: str = '1 day',
                           what_to_show: str = 'TRADES') -> pd.DataFrame:
        """
        קבלת נתונים היסטוריים
        
        Args:
            symbol: סימול
            duration: משך זמן ('1 Y', '6 M' וכו')
            bar_size: גודל נר ('1 day', '1 hour' וכו')
            what_to_show: סוג נתונים
            
        Returns:
            DataFrame עם נתונים
        """
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True
            )
            
            df = util.df(bars)
            return df
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
```

---

## אסטרטגיות מסחר

### אסטרטגיה בסיסית

**קובץ: `src/strategies/base_strategy.py`**

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional

class BaseStrategy(ABC):
    """
    מחלקת בסיס לכל האסטרטגיות
    """
    
    def __init__(self, name: str, params: Dict = None):
        """
        אתחול אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            params: פרמטרים
        """
        self.name = name
        self.params = params or {}
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.trades = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        יצירת אותות מסחר
        
        Args:
            data: נתוני שוק
            
        Returns:
            Series עם אותות: 1 (קנה), -1 (מכור), 0 (החזק)
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב אינדיקטורים טכניים
        
        Args:
            data: נתוני שוק
            
        Returns:
            DataFrame עם אינדיקטורים
        """
        pass
    
    def update_performance(self, trade_result: Dict):
        """
        עדכון מדדי ביצועים
        
        Args:
            trade_result: תוצאת עסקה
        """
        self.trades.append(trade_result)
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """חישוב מדדי ביצועים"""
        if not self.trades:
            return
            
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'profit_factor': self._calculate_profit_factor()
        }
    
    def _calculate_profit_factor(self) -> float:
        """חישוב Profit Factor"""
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
```

### אסטרטגיית Moving Average Crossover

**קובץ: `src/strategies/technical/moving_average.py`**

```python
import pandas as pd
import numpy as np
from ..base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """
    אסטרטגיית חציית ממוצעים נעים
    קונה כאשר MA קצר חוצה מעל MA ארוך
    מוכר כאשר MA קצר חוצה מתחת ל-MA ארוך
    """
    
    def __init__(self, 
                 short_window: int = 50, 
                 long_window: int = 200,
                 use_ema: bool = False):
        """
        אתחול
        
        Args:
            short_window: חלון ממוצע קצר
            long_window: חלון ממוצע ארוך
            use_ema: האם להשתמש ב-EMA במקום SMA
        """
        params = {
            'short_window': short_window,
            'long_window': long_window,
            'use_ema': use_ema
        }
        super().__init__('MA_Crossover', params)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב ממוצעים נעים"""
        df = data.copy()
        
        if self.params['use_ema']:
            df['short_ma'] = df['close'].ewm(span=self.params['short_window']).mean()
            df['long_ma'] = df['close'].ewm(span=self.params['long_window']).mean()
        else:
            df['short_ma'] = df['close'].rolling(window=self.params['short_window']).mean()
            df['long_ma'] = df['close'].rolling(window=self.params['long_window']).mean()
            
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """יצירת אותות מסחר"""
        df = self.calculate_indicators(data)
        
        # אתחול עמודת אותות
        signals = pd.Series(0, index=df.index)
        
        # זיהוי חציות
        df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
        df['signal'] = df['position'].diff()
        
        # אותות קנייה ומכירה
        signals[df['signal'] == 2] = 1   # קנה
        signals[df['signal'] == -2] = -1  # מכור
        
        return signals
```

### אסטרטגיית RSI + MACD + Bollinger Bands

**קובץ: `src/strategies/technical/rsi_macd.py`**

```python
import pandas as pd
import numpy as np
from ..base_strategy import BaseStrategy

class RSI_MACD_Strategy(BaseStrategy):
    """
    אסטרטגיה משולבת של RSI, MACD ו-Bollinger Bands
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: float = 2.0):
        """אתחול עם פרמטרים"""
        params = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bb_period': bb_period,
            'bb_std': bb_std
        }
        super().__init__('RSI_MACD_BB', params)
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """חישוב RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.Series) -> tuple:
        """חישוב MACD"""
        fast = data.ewm(span=self.params['macd_fast']).mean()
        slow = data.ewm(span=self.params['macd_slow']).mean()
        
        macd = fast - slow
        signal = macd.ewm(span=self.params['macd_signal']).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series) -> tuple:
        """חישוב Bollinger Bands"""
        ma = data.rolling(window=self.params['bb_period']).mean()
        std = data.rolling(window=self.params['bb_period']).std()
        
        upper = ma + (std * self.params['bb_std'])
        lower = ma - (std * self.params['bb_std'])
        
        return upper, ma, lower
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב כל האינדיקטורים"""
        df = data.copy()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """יצירת אותות מסחר מבוססי כללים משולבים"""
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        # תנאי קנייה: RSI נמוך + MACD חיובי + מחיר מתחת ל-BB התחתון
        buy_condition = (
            (df['rsi'] < self.params['rsi_oversold']) &
            (df['macd'] > df['macd_signal']) &
            (df['close'] < df['bb_lower'])
        )
        
        # תנאי מכירה: RSI גבוה + MACD שלילי + מחיר מעל ל-BB העליון
        sell_condition = (
            (df['rsi'] > self.params['rsi_overbought']) &
            (df['macd'] < df['macd_signal']) &
            (df['close'] > df['bb_upper'])
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
```

---

## למידת מכונה ו-Reinforcement Learning

### LSTM למניבוי מחירים

**קובץ: `src/strategies/ml_based/lstm_predictor.py`**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from ..base_strategy import BaseStrategy

class LSTMPredictor(BaseStrategy):
    """
    אסטרטגיה מבוססת LSTM לחיזוי מחירים
    """
    
    def __init__(self,
                 lookback_period: int = 60,
                 prediction_horizon: int = 1,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2):
        """אתחול"""
        params = {
            'lookback_period': lookback_period,
            'prediction_horizon': prediction_horizon,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate
        }
        super().__init__('LSTM_Predictor', params)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self, input_shape: tuple):
        """בניית מודל LSTM"""
        model = Sequential([
            LSTM(self.params['lstm_units'], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.params['dropout_rate']),
            
            LSTM(self.params['lstm_units'], 
                 return_sequences=False),
            Dropout(self.params['dropout_rate']),
            
            Dense(25, activation='relu'),
            Dense(self.params['prediction_horizon'])
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """הכנת נתונים לאימון"""
        # נרמול
        scaled_data = self.scaler.fit_transform(data[['close']].values)
        
        X, y = [], []
        lookback = self.params['lookback_period']
        horizon = self.params['prediction_horizon']
        
        for i in range(lookback, len(scaled_data) - horizon + 1):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i:i+horizon])
        
        return np.array(X), np.array(y).reshape(-1, horizon)
    
    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """אימון המודל"""
        X, y = self.prepare_data(data)
        
        # פיצול train/validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """חיזוי מחירים עתידיים"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        scaled_data = self.scaler.transform(data[['close']].values)
        lookback = self.params['lookback_period']
        
        X = scaled_data[-lookback:].reshape(1, lookback, 1)
        prediction_scaled = self.model.predict(X)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return prediction
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב חיזויים"""
        df = data.copy()
        
        if self.is_trained and len(data) >= self.params['lookback_period']:
            prediction = self.predict(data)
            df['predicted_price'] = prediction[0][0]
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """יצירת אותות מבוססי חיזוי"""
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        if 'predicted_price' in df.columns:
            # קנה אם החיזוי גבוה מהמחיר הנוכחי ב-1%+
            buy_condition = (df['predicted_price'] / df['close'] - 1) > 0.01
            # מכור אם החיזוי נמוך מהמחיר הנוכחי ב-1%+
            sell_condition = (df['close'] / df['predicted_price'] - 1) > 0.01
            
            signals[buy_condition] = 1
            signals[sell_condition] = -1
        
        return signals
```

### Deep Reinforcement Learning (DQN)

**קובץ: `src/strategies/ml_based/reinforcement_learning.py`**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from ..base_strategy import BaseStrategy

class DQNTradingAgent(BaseStrategy):
    """
    אסטרטגיית מסחר מבוססת Deep Q-Network
    """
    
    def __init__(self,
                 state_size: int = 10,
                 action_size: int = 3,  # Hold, Buy, Sell
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 2000):
        """אתחול"""
        params = {
            'state_size': state_size,
            'action_size': action_size,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'epsilon_min': epsilon_min,
            'memory_size': memory_size
        }
        super().__init__('DQN_Agent', params)
        
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        
    def _build_model(self):
        """בניית רשת עצבית"""
        model = Sequential([
            Dense(64, input_dim=self.params['state_size'], activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.params['action_size'], activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='mse'
        )
        
        return model
    
    def get_state(self, data: pd.DataFrame, t: int, window_size: int = 10):
        """
        יצירת State מנתוני שוק
        
        Args:
            data: נתוני שוק
            t: זמן נוכחי
            window_size: גודל חלון
            
        Returns:
            State vector
        """
        if t < window_size:
            # Pad with zeros if not enough data
            state = np.zeros(window_size)
            available_data = data['close'].iloc[:t+1].pct_change().fillna(0).values
            state[-len(available_data):] = available_data
        else:
            state = data['close'].iloc[t-window_size:t].pct_change().fillna(0).values
        
        return state.reshape(1, -1)
    
    def act(self, state: np.ndarray) -> int:
        """
        בחירת פעולה (exploration vs exploitation)
        
        Args:
            state: State נוכחי
            
        Returns:
            Action (0: Hold, 1: Buy, 2: Sell)
        """
        if np.random.rand() <= self.params['epsilon']:
            # Exploration: פעולה אקראית
            return random.randrange(self.params['action_size'])
        
        # Exploitation: פעולה מבוססת Q-values
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """שמירת חוויה בזיכרון"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """
        למידה מחוויות קודמות (Experience Replay)
        
        Args:
            batch_size: גודל batch
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.params['gamma'] * \
                         np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.params['epsilon'] > self.params['epsilon_min']:
            self.params['epsilon'] *= self.params['epsilon_decay']
    
    def train_on_historical_data(self, 
                                 data: pd.DataFrame, 
                                 initial_balance: float = 10000,
                                 episodes: int = 100):
        """
        אימון על נתונים היסטוריים
        
        Args:
            data: נתוני שוק
            initial_balance: יתרה התחלתית
            episodes: מספר episodes
        """
        for episode in range(episodes):
            balance = initial_balance
            inventory = []
            total_profit = 0
            
            for t in range(len(data) - 1):
                state = self.get_state(data, t)
                action = self.act(state)
                
                # ביצוע פעולה
                current_price = data['close'].iloc[t]
                next_price = data['close'].iloc[t + 1]
                reward = 0
                
                if action == 1:  # Buy
                    if balance >= current_price:
                        inventory.append(current_price)
                        balance -= current_price
                        
                elif action == 2 and len(inventory) > 0:  # Sell
                    bought_price = inventory.pop(0)
                    profit = next_price - bought_price
                    total_profit += profit
                    balance += next_price
                    reward = profit
                
                next_state = self.get_state(data, t + 1)
                done = (t == len(data) - 2)
                
                self.remember(state, action, reward, next_state, done)
                
                if len(self.memory) > 32:
                    self.replay(32)
            
            print(f"Episode {episode + 1}/{episodes} - "
                  f"Total Profit: ${total_profit:.2f} - "
                  f"Epsilon: {self.params['epsilon']:.4f}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב indicators (לא נדרש ל-DQN)"""
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """יצירת אותות מסחר"""
        signals = pd.Series(0, index=data.index)
        
        for t in range(len(data)):
            state = self.get_state(data, t)
            action = self.act(state)
            
            if action == 1:  # Buy
                signals.iloc[t] = 1
            elif action == 2:  # Sell
                signals.iloc[t] = -1
        
        return signals
```

---

## Backtesting ו-Paper Trading

### מנוע Backtesting

**קובץ: `src/backtesting/backtest_engine.py`**

```python
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    """
    מנוע Backtesting מתקדם
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        אתחול
        
        Args:
            initial_capital: הון התחלתי
            commission: עמלות (0.1% = 0.001)
            slippage: slippage (0.05% = 0.0005)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.results = None
        self.trades = []
        
    def run(self, 
            data: pd.DataFrame,
            signals: pd.Series,
            position_size: float = 1.0) -> Dict:
        """
        הרצת Backtest
        
        Args:
            data: נתוני שוק
            signals: אותות מסחר (1: קנה, -1: מכור, 0: החזק)
            position_size: גודל פוזיציה (0-1)
            
        Returns:
            תוצאות Backtest
        """
        df = data.copy()
        df['signal'] = signals
        
        # אתחול
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = []
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # עדכון ערך תיק
            current_value = capital + (position * current_price)
            portfolio_value.append(current_value)
            
            # ביצוע עסקאות
            if signal == 1 and position == 0:  # קנה
                # חישוב מספר מניות
                shares = int((capital * position_size) / current_price)
                if shares > 0:
                    cost = shares * current_price
                    cost_with_fees = cost * (1 + self.commission + self.slippage)
                    
                    if cost_with_fees <= capital:
                        position = shares
                        entry_price = current_price
                        capital -= cost_with_fees
                        
                        self.trades.append({
                            'date': df.index[i],
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'capital': capital
                        })
            
            elif signal == -1 and position > 0:  # מכור
                revenue = position * current_price
                revenue_after_fees = revenue * (1 - self.commission - self.slippage)
                
                profit = revenue_after_fees - (position * entry_price)
                capital += revenue_after_fees
                
                self.trades.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'profit': profit,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
        
        # חישוב מדדים
        df['portfolio_value'] = portfolio_value
        returns = pd.Series(portfolio_value).pct_change()
        
        self.results = self._calculate_metrics(df, returns)
        return self.results
    
    def _calculate_metrics(self, df: pd.DataFrame, returns: pd.Series) -> Dict:
        """חישוב מדדי ביצועים"""
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Sharpe Ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win Rate
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len([t for t in self.trades if 'profit' in t]) * 100 if self.trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_capital': df['portfolio_value'].iloc[-1],
            'portfolio_values': df['portfolio_value'].values
        }
    
    def plot_results(self):
        """ציור תוצאות"""
        if self.results is None:
            raise ValueError("No results to plot. Run backtest first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity Curve
        axes[0, 0].plot(self.results['portfolio_values'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        returns = pd.Series(self.results['portfolio_values']).pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Returns Distribution
        axes[1, 0].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Performance Metrics
        metrics_text = f"""
        Total Return: {self.results['total_return']:.2f}%
        Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
        Sortino Ratio: {self.results['sortino_ratio']:.2f}
        Max Drawdown: {self.results['max_drawdown']:.2f}%
        Win Rate: {self.results['win_rate']:.2f}%
        Total Trades: {self.results['total_trades']}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
```

---

## Risk Management

### Kelly Criterion ו-Position Sizing

**קובץ: `src/risk_management/kelly_criterion.py`**

```python
import numpy as np
from typing import List, Dict

class KellyCriterion:
    """
    יישום Kelly Criterion לניהול גודל פוזיציות
    """
    
    def __init__(self, 
                 kelly_fraction: float = 0.5,
                 max_position_size: float = 0.25):
        """
        אתחול
        
        Args:
            kelly_fraction: אחוז Kelly (0.5 = Half Kelly)
            max_position_size: גודל מקסימלי של פוזיציה (25% = 0.25)
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        
    def calculate_from_trades(self, trades: List[Dict]) -> float:
        """
        חישוב Kelly מתוצאות עסקאות
        
        Args:
            trades: רשימת עסקאות עם 'pnl'
            
        Returns:
            Kelly fraction מומלץ
        """
        if not trades:
            return 0.0
        
        # חישוב win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # חישוב avg win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1
        
        # Kelly formula
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # חישוב fractional Kelly
        kelly_fraction = kelly * self.kelly_fraction
        
        # הגבלה למקסימום
        return min(max(kelly_fraction, 0), self.max_position_size)
    
    def calculate_from_sharpe(self, 
                             mean_return: float,
                             std_return: float,
                             risk_free_rate: float = 0.02) -> float:
        """
        חישוב Kelly מ-Sharpe Ratio
        
        Args:
            mean_return: תשואה ממוצעת שנתית
            std_return: סטיית תקן שנתית
            risk_free_rate: ריבית חסרת סיכון
            
        Returns:
            Kelly fraction מומלץ
        """
        if std_return == 0:
            return 0.0
        
        excess_return = mean_return - risk_free_rate
        kelly = excess_return / (std_return ** 2)
        
        kelly_fraction = kelly * self.kelly_fraction
        return min(max(kelly_fraction, 0), self.max_position_size)
    
    def calculate_position_size(self,
                               account_value: float,
                               kelly_fraction: float) -> float:
        """
        חישוב גודל פוזיציה בדולרים
        
        Args:
            account_value: ערך חשבון
            kelly_fraction: Kelly fraction
            
        Returns:
            גודל פוזיציה בדולרים
        """
        return account_value * kelly_fraction


class PositionSizer:
    """
    מחלקה לניהול גודל פוזיציות
    """
    
    def __init__(self,
                 method: str = 'kelly',
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5):
        """
        אתחול
        
        Args:
            method: שיטה ('kelly', 'fixed_fractional', 'volatility_based')
            risk_per_trade: סיכון לעסקה (2% = 0.02)
            max_positions: מספר מקסימלי של פוזיציות
        """
        self.method = method
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.kelly = KellyCriterion()
        
    def calculate(self,
                 account_value: float,
                 entry_price: float,
                 stop_loss: float,
                 trades_history: List[Dict] = None,
                 volatility: float = None) -> int:
        """
        חישוב מספר מניות
        
        Args:
            account_value: ערך חשבון
            entry_price: מחיר כניסה
            stop_loss: stop loss
            trades_history: היסטוריית עסקאות
            volatility: volatility (לשיטה מבוססת volatility)
            
        Returns:
            מספר מניות
        """
        if self.method == 'fixed_fractional':
            return self._fixed_fractional(account_value, entry_price)
        
        elif self.method == 'kelly' and trades_history:
            kelly_fraction = self.kelly.calculate_from_trades(trades_history)
            position_value = self.kelly.calculate_position_size(account_value, kelly_fraction)
            return int(position_value / entry_price)
        
        elif self.method == 'volatility_based' and volatility:
            return self._volatility_based(account_value, entry_price, volatility)
        
        else:
            # ברירת מחדל: סיכון קבוע
            return self._risk_based(account_value, entry_price, stop_loss)
    
    def _fixed_fractional(self, account_value: float, entry_price: float) -> int:
        """שיטת fractional קבוע"""
        position_value = account_value * self.risk_per_trade
        return int(position_value / entry_price)
    
    def _risk_based(self, 
                   account_value: float,
                   entry_price: float,
                   stop_loss: float) -> int:
        """שיטה מבוססת סיכון"""
        risk_amount = account_value * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        return int(risk_amount / risk_per_share)
    
    def _volatility_based(self,
                         account_value: float,
                         entry_price: float,
                         volatility: float) -> int:
        """שיטה מבוססת volatility"""
        # ככל ש-volatility גבוה יותר, גודל פוזיציה קטן יותר
        adjusted_risk = self.risk_per_trade / (1 + volatility)
        position_value = account_value * adjusted_risk
        return int(position_value / entry_price)
```

---

## פרומפטים מתקדמים ל-Cursor

### קובץ .cursorrules

**קובץ: `.cursorrules`**

```markdown
# Trading System Development Rules for Cursor AI

## Project Context
You are building a professional algorithmic trading system with the following requirements:
- Interactive Brokers API integration for real and paper trading
- Multiple trading strategies (technical and ML-based)
- Backtesting engine with performance metrics
- Risk management with Kelly Criterion
- Learning system that improves from historical performance
- Clean, modular, production-ready code

## Code Style Guidelines

### Python Best Practices
- Use Python 3.10+ features
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Write comprehensive docstrings (Google style)
- Keep functions under 50 lines when possible
- Use meaningful variable names (not single letters except in loops)

### Architecture Principles
- Follow SOLID principles
- Use dependency injection
- Implement proper error handling with custom exceptions
- Use logging instead of print statements
- Write unit tests for critical functions
- Separate business logic from infrastructure

### Trading-Specific Rules
1. **Never hardcode credentials** - always use environment variables
2. **Always validate market data** before using it
3. **Include proper risk checks** before placing orders
4. **Log all trading decisions** with reasoning
5. **Handle connection failures** gracefully
6. **Implement circuit breakers** for excessive losses
7. **Use paper trading mode** by default during development

## File Organization
- Keep each strategy in a separate file
- Use `base_strategy.py` as parent class for all strategies
- Store configuration in YAML files, not in code
- Keep API keys in `.env` file (never commit this)
- Use `config/` directory for all configuration files

## When Creating New Features

### For Trading Strategies
```python
class NewStrategy(BaseStrategy):
    """
    [Strategy description]
    
    Attributes:
        [List key attributes]
    
    Methods:
        calculate_indicators: Compute technical indicators
        generate_signals: Generate buy/sell signals
        update_performance: Track strategy performance
    """
```

### For ML Models
- Always split data into train/validation/test
- Use proper normalization/scaling
- Implement early stopping
- Save model checkpoints
- Track training metrics
- Validate predictions before trading

### For Risk Management
- Calculate position size based on account value
- Never risk more than 2% per trade by default
- Implement stop-loss for every position
- Check total portfolio exposure before new trades
- Use Kelly Criterion with fractional sizing (0.25-0.5)

## Testing Requirements

### Unit Tests
- Test each strategy's signal generation
- Test risk management calculations
- Mock IB API calls in tests
- Test edge cases (empty data, extreme values)

### Integration Tests
- Test full trading workflow
- Test backtesting engine with known data
- Verify order placement flow

### Backtesting
- Test strategies on multiple time periods
- Include transaction costs
- Test during different market conditions
- Walk-forward optimization

## Error Handling

### Critical Errors (Stop Trading)
- Connection loss to broker
- Invalid order parameters
- Insufficient funds
- Risk limit breach

### Warnings (Log and Continue)
- Missing data points
- Strategy signal conflicts
- Minor API errors

## Performance Requirements
- Backtest execution: < 1 minute for 1 year daily data
- Real-time signal generation: < 100ms
- Order placement: < 500ms
- Database queries: < 100ms

## Documentation
- Update README.md with new features
- Document API changes
- Include example usage for new components
- Maintain changelog

## Security
- Never log sensitive data (API keys, account numbers)
- Use encrypted storage for credentials
- Validate all user inputs
- Implement rate limiting for API calls

## When I Ask You To...

### "Create a new strategy"
1. Create file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement all abstract methods
4. Add configuration to `config/strategies.yaml`
5. Write unit tests
6. Add example usage in docstring

### "Optimize a strategy"
1. Run backtest with parameter ranges
2. Use walk-forward optimization
3. Check for overfitting (out-of-sample test)
4. Compare multiple metrics (Sharpe, Sortino, Win Rate)
5. Visualize results

### "Add ML model"
1. Create in `src/ml_models/`
2. Implement train/predict methods
3. Add data preprocessing pipeline
4. Include model evaluation metrics
5. Save/load model functionality
6. Add to strategy registry

### "Debug an issue"
1. Check logs first
2. Verify data integrity
3. Test with paper trading
4. Add detailed logging if needed
5. Create minimal reproducible example

## Code Review Checklist
Before submitting code, ensure:
- [ ] Type hints on all functions
- [ ] Docstrings with examples
- [ ] Error handling implemented
- [ ] Logging added
- [ ] Tests written and passing
- [ ] No hardcoded values
- [ ] Configuration externalized
- [ ] Performance acceptable
- [ ] Security reviewed

## Remember
- Safety first: Always validate before trading
- Code quality matters: Write for maintainability
- Test thoroughly: Bugs can cost money
- Document well: Future you will thank you
- Stay modular: Easy to extend and modify
```

### פרומפטים ספציפיים ל-Cursor

#### פרומפט ליצירת אסטרטגיה חדשה

```
Create a new trading strategy called [STRATEGY_NAME] that:
1. Inherits from BaseStrategy
2. Uses the following indicators: [LIST_INDICATORS]
3. Generates signals based on: [LOGIC]
4. Includes proper docstrings and type hints
5. Has configurable parameters
6. Implements risk management
7. Includes example usage

File path: src/strategies/[category]/[strategy_name].py

Make sure to:
- Follow the .cursorrules guidelines
- Add comprehensive error handling
- Include logging for all decisions
- Write it in Hebrew comments for better understanding
```

#### פרומפט לאופטימיזציה

```
Optimize the [STRATEGY_NAME] strategy:
1. Create parameter grid for optimization
2. Run walk-forward backtests
3. Analyze results across multiple metrics:
   - Sharpe Ratio
   - Sortino Ratio  
   - Max Drawdown
   - Win Rate
   - Profit Factor
4. Check for overfitting using out-of-sample data
5. Generate visualization of results
6. Recommend best parameters with justification

Use the backtesting engine in src/backtesting/backtest_engine.py
```

#### פרומפט למודל ML

```
Create a machine learning model for [TASK]:
1. Model type: [LSTM/GRU/Transformer/RL]
2. Input features: [LIST_FEATURES]
3. Output: [PREDICTION_TYPE]
4. Training pipeline with proper data splitting
5. Evaluation metrics
6. Model serialization (save/load)
7. Integration with BaseStrategy

Requirements:
- Use TensorFlow/PyTorch
- Implement early stopping
- Add learning rate scheduling
- Track training history
- Validate predictions before use in trading
- Include example training script

File path: src/ml_models/[model_name].py
```

#### פרומפט לממשק משתמש

```
Create a Streamlit dashboard for the trading system with:

Pages:
1. Overview - Portfolio summary, P&L, active strategies
2. Strategies - List of strategies with performance metrics
3. Backtesting - Run backtests with parameter selection
4. Risk Management - Current exposures, risk metrics
5. Trade History - Detailed trade log with filters
6. Live Trading - Real-time monitoring and controls

Features:
- Real-time updates
- Interactive charts (use Plotly)
- Strategy comparison tools
- Risk alerts
- Performance analytics
- Configuration management

File: src/ui/dashboard.py

Use components from src/ui/components/ for reusability
```

---

## מערכת למידה והתאמה

### Performance Tracker

**קובץ: `src/learning/performance_tracker.py`**

```python
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

class PerformanceTracker:
    """
    מערכת מעקב וניתוח ביצועים
    לומדת מהצלחות והפסדים
    """
    
    def __init__(self, db_path: str = 'data/performance.db'):
        """אתחול"""
        self.db_path = db_path
        self.trades = []
        self.daily_pnl = []
        self.strategy_performance = {}
        
    def log_trade(self, trade: Dict):
        """
        רישום עסקה
        
        Args:
            trade: מילון עם פרטי עסקה
                - strategy: שם אסטרטגיה
                - symbol: סימול
                - action: BUY/SELL
                - quantity: כמות
                - price: מחיר
                - pnl: רווח/הפסד
                - timestamp: זמן
        """
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self._update_strategy_stats(trade)
        self._save_to_db()
    
    def _update_strategy_stats(self, trade: Dict):
        """עדכון סטטיסטיקות אסטרטגיה"""
        strategy_name = trade['strategy']
        
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'current_streak': 0
            }
        
        stats = self.strategy_performance[strategy_name]
        pnl = trade.get('pnl', 0)
        
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['winning_trades'] += 1
            stats['current_streak'] = max(0, stats['current_streak']) + 1
            stats['consecutive_wins'] = max(stats['consecutive_wins'], stats['current_streak'])
            stats['best_trade'] = max(stats['best_trade'], pnl)
        elif pnl < 0:
            stats['current_streak'] = min(0, stats['current_streak']) - 1
            stats['consecutive_losses'] = max(stats['consecutive_losses'], abs(stats['current_streak']))
            stats['worst_trade'] = min(stats['worst_trade'], pnl)
        
        stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        
        # חישוב avg win/loss
        wins = [t['pnl'] for t in self.trades if t['strategy'] == strategy_name and t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['strategy'] == strategy_name and t['pnl'] < 0]
        
        stats['avg_win'] = np.mean(wins) if wins else 0
        stats['avg_loss'] = np.mean(losses) if losses else 0
        
        # Profit Factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def get_strategy_recommendations(self) -> Dict[str, str]:
        """
        המלצות לשיפור אסטרטגיות
        
        Returns:
            מילון עם המלצות לכל אסטרטגיה
        """
        recommendations = {}
        
        for strategy_name, stats in self.strategy_performance.items():
            recs = []
            
            # בדיקת win rate
            if stats['win_rate'] < 0.4:
                recs.append("Win rate נמוך - שקול לחדד תנאי כניסה")
            
            # בדיקת profit factor
            if stats['profit_factor'] < 1.5:
                recs.append("Profit factor נמוך - שפר את יחס Win/Loss")
            
            # בדיקת רצפי הפסדים
            if stats['consecutive_losses'] > 5:
                recs.append(f"זוהו {stats['consecutive_losses']} הפסדים רצופים - שקול הפסקה זמנית")
            
            # בדיקת התפלגות רווחים/הפסדים
            if abs(stats['worst_trade']) > abs(stats['best_trade']) * 2:
                recs.append("הפסדים גדולים יחסית - שפר stop-loss")
            
            # המלצות חיוביות
            if stats['win_rate'] > 0.6:
                recs.append("Win rate מצוין - שקול הגדלת position size")
            
            if stats['profit_factor'] > 2.0:
                recs.append("Profit factor מעולה - אסטרטגיה חזקה")
            
            recommendations[strategy_name] = ' | '.join(recs) if recs else "ביצועים תקינים"
        
        return recommendations
    
    def analyze_market_conditions(self) -> Dict:
        """
        ניתוח תנאי שוק אופטימליים
        
        Returns:
            מילון עם תובנות
        """
        if len(self.trades) < 50:
            return {"message": "לא מספיק נתונים לניתוח"}
        
        df = pd.DataFrame(self.trades)
        
        # ניתוח לפי ימי שבוע
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        pnl_by_day = df.groupby('day_of_week')['pnl'].agg(['mean', 'sum', 'count'])
        best_day = pnl_by_day['mean'].idxmax()
        
        # ניתוח לפי שעות (אם יש)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        pnl_by_hour = df.groupby('hour')['pnl'].agg(['mean', 'sum', 'count'])
        best_hour = pnl_by_hour['mean'].idxmax()
        
        # ניתוח לפי סימולים
        pnl_by_symbol = df.groupby('symbol')['pnl'].agg(['mean', 'sum', 'count'])
        best_symbols = pnl_by_symbol.nlargest(5, 'sum')
        
        days_map = {0: 'ראשון', 1: 'שני', 2: 'שלישי', 3: 'רביעי', 4: 'חמישי', 5: 'שישי', 6: 'שבת'}
        
        return {
            'best_trading_day': days_map[best_day],
            'best_trading_hour': f"{best_hour}:00",
            'best_symbols': best_symbols.index.tolist(),
            'total_pnl': df['pnl'].sum(),
            'avg_pnl_per_trade': df['pnl'].mean(),
            'volatility': df['pnl'].std()
        }
    
    def _save_to_db(self):
        """שמירה למסד נתונים"""
        # יישום שמירה ל-SQLite או PostgreSQL
        pass
```

---

## ממשק משתמש ותצוגה

### Streamlit Dashboard

**קובץ: `src/ui/dashboard.py`**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# הגדרת עמוד
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ הגדרות")
    
    # בחירת מצב
    trading_mode = st.radio(
        "מצב מסחר",
        ["Paper Trading", "Live Trading"],
        help="Paper Trading = מסחר דמה, Live Trading = מסחר אמיתי"
    )
    
    # בחירת אסטרטגיות
    st.subheader("אסטרטגיות פעילות")
    strategies = {
        "MA Crossover": st.checkbox("MA Crossover", value=True),
        "RSI + MACD": st.checkbox("RSI + MACD", value=True),
        "LSTM Predictor": st.checkbox("LSTM Predictor"),
        "DQN Agent": st.checkbox("DQN Agent")
    }
    
    # הגדרות סיכון
    st.subheader("ניהול סיכונים")
    risk_per_trade = st.slider("סיכון לעסקה (%)", 0.5, 5.0, 2.0, 0.5)
    max_positions = st.number_input("מקס פוזיציות", 1, 20, 5)
    
    # כפתורי פעולה
    st.divider()
    if st.button("🚀 התחל מסחר", use_container_width=True):
        st.success("המערכת החלה למסחר!")
    
    if st.button("⏸️ השהה", use_container_width=True):
        st.warning("המערכת הושהתה")
    
    if st.button("🛑 עצור הכל", use_container_width=True, type="primary"):
        st.error("כל הפעילות נעצרה")

# Main Dashboard
st.title("📊 מערכת מסחר אוטומטית")
st.caption(f"מצב: {trading_mode} | עדכון אחרון: {datetime.now().strftime('%H:%M:%S')}")

# Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="ערך תיק",
        value="$125,430",
        delta="$5,430 (4.5%)"
    )

with col2:
    st.metric(
        label="רווח היום",
        value="$1,234",
        delta="0.98%"
    )

with col3:
    st.metric(
        label="פוזיציות פתוחות",
        value="3",
        delta="-1"
    )

with col4:
    st.metric(
        label="Win Rate",
        value="64%",
        delta="2%"
    )

with col5:
    st.metric(
        label="Sharpe Ratio",
        value="1.85",
        delta="0.15"
    )

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 סקירה כללית",
    "🎯 אסטרטגיות",
    "🔬 Backtesting",
    "⚠️ ניהול סיכונים",
    "📜 היסטוריית עסקאות"
])

with tab1:
    # Equity Curve
    st.subheader("Equity Curve")
    
    # נתונים לדוגמה
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    portfolio_values = 100000 * (1 + pd.Series(range(len(dates))) * 0.0005 + 
                                 pd.Series(range(len(dates))).apply(lambda x: np.random.randn() * 0.01))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00ff00', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title="תאריך",
        yaxis_title="ערך ($)",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # פוזיציות נוכחיות
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("פוזיציות פתוחות")
        positions_data = {
            'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'Shares': [100, 50, 75],
            'Avg Price': [175.50, 140.30, 380.20],
            'Current Price': [178.30, 142.10, 382.50],
            'P&L': ['+$280', '+$90', '+$172.50'],
            'P&L %': ['+1.6%', '+1.3%', '+0.6%']
        }
        st.dataframe(positions_data, use_container_width=True)
    
    with col2:
        st.subheader("התפלגות תיק")
        allocation = pd.DataFrame({
            'Asset': ['מניות', 'קאש', 'אופציות'],
            'Value': [95000, 25000, 5000]
        })
        
        fig_pie = px.pie(
            allocation,
            values='Value',
            names='Asset',
            hole=0.4
        )
        fig_pie.update_layout(height=300, template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("ביצועי אסטרטגיות")
    
    # טבלת ביצועים
    strategy_performance = pd.DataFrame({
        'אסטרטגיה': ['MA Crossover', 'RSI + MACD', 'LSTM Predictor', 'DQN Agent'],
        'סטטוס': ['🟢 פעיל', '🟢 פעיל', '🟡 לומד', '🔴 כבוי'],
        'עסקאות': [45, 32, 12, 0],
        'Win Rate': ['67%', '62%', '58%', '-'],
        'Total P&L': ['$4,230', '$3,120', '$890', '-'],
        'Sharpe': [1.85, 1.62, 1.23, '-'],
        'Max DD': ['-8.5%', '-12.3%', '-6.7%', '-']
    })
    
    st.dataframe(
        strategy_performance,
        use_container_width=True,
        hide_index=True
    )
    
    # גרף השוואה
    st.subheader("השוואת ביצועים")
    
    fig_compare = go.Figure()
    
    for strategy in ['MA Crossover', 'RSI + MACD', 'LSTM Predictor']:
        returns = np.cumsum(np.random.randn(100) * 0.02 + 0.001)
        fig_compare.add_trace(go.Scatter(
            x=list(range(100)),
            y=returns,
            mode='lines',
            name=strategy
        ))
    
    fig_compare.update_layout(
        height=400,
        xaxis_title="עסקאות",
        yaxis_title="תשואה מצטברת (%)",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)

with tab3:
    st.subheader("🔬 Backtesting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_strategy = st.selectbox(
            "אסטרטגיה",
            ["MA Crossover", "RSI + MACD", "LSTM Predictor", "DQN Agent"]
        )
    
    with col2:
        backtest_period = st.selectbox(
            "תקופה",
            ["1 חודש", "3 חודשים", "6 חודשים", "1 שנה", "3 שנים"]
        )
    
    with col3:
        initial_capital = st.number_input(
            "הון התחלתי ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )
    
    if st.button("🚀 הרץ Backtest", use_container_width=True):
        with st.spinner("מריץ backtest..."):
            # סימולציה של backtest
            import time
            time.sleep(2)
            
            st.success("Backtest הושלם!")
            
            # תוצאות
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", "45.3%", "12.1%")
            with col2:
                st.metric("Sharpe Ratio", "1.87", "0.23")
            with col3:
                st.metric("Max Drawdown", "-12.5%", "3.2%")
            with col4:
                st.metric("Win Rate", "64%", "5%")

with tab4:
    st.subheader("⚠️ ניהול סיכונים")
    
    # מדדי סיכון
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "חשיפה כוללת",
            "68%",
            help="אחוז התיק בסיכון"
        )
    
    with col2:
        st.metric(
            "Kelly Fraction",
            "0.42",
            help="מקדם Kelly מומלץ"
        )
    
    with col3:
        st.metric(
            "VaR (95%)",
            "$2,340",
            help="Value at Risk"
        )
    
    # התראות
    st.subheader("התראות")
    
    alerts = [
        {"level": "🟡", "message": "חשיפה גבוהה ל-AAPL (35% מהתיק)"},
        {"level": "🟢", "message": "Sharpe Ratio משתפר - 1.85"},
        {"level": "🔴", "message": "רצף של 3 הפסדים ב-MA Crossover"}
    ]
    
    for alert in alerts:
        st.info(f"{alert['level']} {alert['message']}")

with tab5:
    st.subheader("📜 היסטוריית עסקאות")
    
    # פילטרים
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_strategy = st.multiselect(
            "אסטרטגיה",
            ["הכל", "MA Crossover", "RSI + MACD", "LSTM"],
            default=["הכל"]
        )
    
    with col2:
        filter_symbol = st.multiselect(
            "סימול",
            ["הכל", "AAPL", "GOOGL", "MSFT"],
            default=["הכל"]
        )
    
    with col3:
        filter_date = st.date_input(
            "מתאריך",
            value=datetime.now() - timedelta(days=30)
        )
    
    # טבלת עסקאות
    trades = pd.DataFrame({
        'תאריך': pd.date_range(end=datetime.now(), periods=20, freq='D'),
        'אסטרטגיה': np.random.choice(['MA Crossover', 'RSI + MACD'], 20),
        'סימול': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 20),
        'פעולה': np.random.choice(['BUY', 'SELL'], 20),
        'מחיר': np.random.uniform(100, 200, 20).round(2),
        'כמות': np.random.randint(10, 100, 20),
        'P&L': np.random.uniform(-500, 1000, 20).round(2)
    })
    
    # צביעת P&L
    def color_pnl(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    
    styled_trades = trades.style.applymap(
        color_pnl,
        subset=['P&L']
    )
    
    st.dataframe(styled_trades, use_container_width=True, height=400)
    
    # סטטיסטיקות
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("סה״כ עסקאות", len(trades))
    with col2:
        st.metric("עסקאות רווחיות", len(trades[trades['P&L'] > 0]))
    with col3:
        st.metric("סה״כ P&L", f"${trades['P&L'].sum():.2f}")

# Footer
st.divider()
st.caption("© 2025 Advanced Trading System | Powered by Interactive Brokers & AI")
```

---

## פרוייקט התחלתי

### מבנה פרוייקט מינימלי

```bash
# יצירת מבנה תיקיות
mkdir -p trading_system/{src/{broker,strategies/{technical,ml_based},backtesting,risk_management,ml_models,data,learning,ui},config,tests,docs,data,models,logs}

# יצירת קבצים בסיסיים
touch trading_system/{requirements.txt,.env,.gitignore,.cursorrules,README.md,docker-compose.yml}
touch trading_system/config/{config.yaml,strategies.yaml,risk_params.yaml}
```

### requirements.txt

```
# Core
python>=3.10
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0

# Interactive Brokers
ib-insync>=0.9.86
ibapi>=9.81.1

# ML & Data Science
tensorflow>=2.13.0
scikit-learn>=1.3.0
torch>=2.0.0

# Backtesting
backtesting>=0.3.3
ta-lib>=0.4.28

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
streamlit>=1.28.0

# Database
psycopg2-binary>=2.9.6
sqlalchemy>=2.0.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
requests>=2.31.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### config.yaml

```yaml
# Trading System Configuration

broker:
  provider: "interactive_brokers"
  host: "127.0.0.1"
  paper_port: 7497
  live_port: 7496
  client_id: 1
  default_mode: "paper"  # paper or live

trading:
  max_positions: 5
  default_position_size: 0.2  # 20% of portfolio
  commission: 0.001  # 0.1%
  slippage: 0.0005   # 0.05%

risk_management:
  risk_per_trade: 0.02  # 2%
  max_daily_loss: 0.05  # 5%
  kelly_fraction: 0.5   # Half Kelly
  max_leverage: 1.0     # No leverage
  
strategies:
  enabled:
    - "MA_Crossover"
    - "RSI_MACD"
  
  MA_Crossover:
    short_window: 50
    long_window: 200
    use_ema: false
  
  RSI_MACD:
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9

data:
  default_timeframe: "1d"
  historical_period: "1y"
  update_interval: 60  # seconds
  
logging:
  level: "INFO"
  file: "logs/trading.log"
  max_size: "100MB"
  backup_count: 5

database:
  type: "sqlite"  # or "postgresql"
  path: "data/trading.db"
```

---

## סיכום והמלצות

### שלבי פיתוח מומלצים

1. **שלב 1: תשתית בסיסית (שבוע 1-2)**
   - הקמת מבנה פרוייקט
   - חיבור ל-IB Paper Trading
   - מנוע Backtesting בסיסי

2. **שלב 2: אסטרטגיות טכניות (שבוע 3-4)**
   - יישום 2-3 אסטרטגיות טכניות
   - Backtesting מקיף
   - אופטימיזציה

3. **שלב 3: Risk Management (שבוע 5)**
   - Kelly Criterion
   - Position Sizing
   - Stop Loss Management

4. **שלב 4: למידת מכונה (שבוע 6-8)**
   - LSTM Predictor
   - אימון ובדיקה
   - אינטגרציה למערכת

5. **שלב 5: RL Agent (שבוע 9-10)**
   - DQN Implementation
   - אימון על נתונים היסטוריים
   - Fine-tuning

6. **שלב 6: ממשק משתמש (שבוע 11)**
   - Streamlit Dashboard
   - ויזואליזציה
   - ניהול ומעקב

7. **שלב 7: למידה והתאמה (שבוע 12)**
   - Performance Tracker
   - Feedback Loop
   - אופטימיזציה אוטומטית

8. **שלב 8: בדיקות ופריסה (שבוע 13-14)**
   - בדיקות מקיפות
   - Paper Trading ממושך
   - מעבר ל-Live Trading (בזהירות!)

### טיפים חשובים

1. **התחל קטן** - אל תנסה ליישם הכל בבת אחת
2. **בדוק היטב** - Backtest על תקופות שונות ותנאי שוק שונים
3. **Paper Trade ארוך** - לפחות 3 חודשים לפני Live
4. **למד מטעויות** - עקוב אחר כל עסקה ונתח
5. **נהל סיכונים** - אל תסכן יותר מ-2% לעסקה
6. **גיוון** - אל תסמוך על אסטרטגיה אחת
7. **המשך ללמוד** - השוק משתנה, המערכת צריכה להשתנות איתו
8. **תיעוד** - תעד כל החלטה וכל שינוי

### משאבים נוספים

- **Interactive Brokers API**: https://interactivebrokers.github.io/
- **Backtesting.py**: https://kernc.github.io/backtesting.py/
- **Machine Learning for Trading**: https://github.com/stefan-jansen/machine-learning-for-trading
- **QuantConnect**: https://www.quantconnect.com/
- **Cursor Docs**: https://docs.cursor.com/

---

---

## אינטגרציה עם מודלים מקומיים

### שימוש ב-Ollama לחיזויים מקומיים

**קובץ: `src/ml_models/local_llm_analyzer.py`**

```python
import requests
import json
from typing import Dict, List
import pandas as pd

class LocalLLMAnalyzer:
    """
    אינטגרציה עם מודלי LLM מקומיים (Ollama, LM Studio וכו')
    לניתוח סנטימנט וקבלת החלטות
    """
    
    def __init__(self, 
                 api_url: str = "http://localhost:11434/api/generate",
                 model: str = "llama3"):
        """
        אתחול
        
        Args:
            api_url: כתובת API של המודל המקומי
            model: שם המודל
        """
        self.api_url = api_url
        self.model = model
        
    def analyze_market_sentiment(self, 
                                 news_data: List[str],
                                 symbol: str) -> Dict:
        """
        ניתוח סנטימנט של חדשות שוק
        
        Args:
            news_data: רשימת כותרות חדשות
            symbol: סימול המניה
            
        Returns:
            ניתוח סנטימנט וציון
        """
        prompt = f"""
        Analyze the following news headlines for {symbol} and determine market sentiment:
        
        Headlines:
        {chr(10).join([f"- {news}" for news in news_data])}
        
        Provide analysis in JSON format:
        {{
            "sentiment": "bullish/neutral/bearish",
            "confidence": 0-100,
            "reasoning": "brief explanation",
            "key_points": ["point1", "point2"],
            "recommendation": "buy/hold/sell"
        }}
        """
        
        response = self._query_model(prompt)
        
        try:
            # ניסיון לחלץ JSON מהתשובה
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            return result
        except:
            return {
                "sentiment": "neutral",
                "confidence": 50,
                "reasoning": "Failed to parse response",
                "key_points": [],
                "recommendation": "hold"
            }
    
    def generate_trading_strategy(self,
                                  market_data: pd.DataFrame,
                                  indicators: Dict) -> str:
        """
        יצירת אסטרטגיית מסחר מבוססת AI
        
        Args:
            market_data: נתוני שוק
            indicators: אינדיקטורים טכניים
            
        Returns:
            המלצת אסטרטגיה
        """
        # הכנת סיכום נתונים
        summary = f"""
        Recent Price Action:
        - Current Price: {market_data['close'].iloc[-1]:.2f}
        - 5-day Change: {((market_data['close'].iloc[-1] / market_data['close'].iloc[-5]) - 1) * 100:.2f}%
        - Volume Trend: {"Increasing" if market_data['volume'].iloc[-1] > market_data['volume'].iloc[-5] else "Decreasing"}
        
        Technical Indicators:
        - RSI: {indicators.get('rsi', 'N/A')}
        - MACD: {indicators.get('macd', 'N/A')}
        - Moving Averages: {indicators.get('ma', 'N/A')}
        """
        
        prompt = f"""
        Based on this market data, suggest a trading strategy:
        
        {summary}
        
        Provide:
        1. Entry strategy
        2. Exit strategy
        3. Risk management
        4. Position sizing recommendation
        """
        
        return self._query_model(prompt)
    
    def explain_trade_decision(self, 
                              trade_data: Dict) -> str:
        """
        הסבר החלטת מסחר באופן מובן לאדם
        
        Args:
            trade_data: נתוני העסקה
            
        Returns:
            הסבר מפורט
        """
        prompt = f"""
        Explain this trading decision in simple terms:
        
        Symbol: {trade_data['symbol']}
        Action: {trade_data['action']}
        Quantity: {trade_data['quantity']}
        Price: ${trade_data['price']}
        Strategy: {trade_data['strategy']}
        Indicators: {trade_data.get('indicators', {})}
        
        Provide a clear, concise explanation of why this trade was made.
        """
        
        return self._query_model(prompt)
    
    def _query_model(self, prompt: str) -> str:
        """שליחת שאילתה למודל"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error querying model: {str(e)}"


class LocalModelOrchestrator:
    """
    מנהל מודלים מקומיים מרובים
    """
    
    def __init__(self):
        """אתחול"""
        self.models = {
            'sentiment': LocalLLMAnalyzer(model='llama3'),
            'technical': LocalLLMAnalyzer(model='mistral'),
            'risk': LocalLLMAnalyzer(model='phi')
        }
        
    def get_comprehensive_analysis(self,
                                   symbol: str,
                                   market_data: pd.DataFrame,
                                   news: List[str]) -> Dict:
        """
        ניתוח מקיף ממספר מודלים
        
        Args:
            symbol: סימול
            market_data: נתוני שוק
            news: חדשות
            
        Returns:
            ניתוח משולב
        """
        # ניתוח סנטימנט
        sentiment = self.models['sentiment'].analyze_market_sentiment(news, symbol)
        
        # ניתוח טכני
        indicators = self._calculate_indicators(market_data)
        technical = self.models['technical'].generate_trading_strategy(
            market_data, indicators
        )
        
        # הערכת סיכון
        risk_analysis = self._assess_risk(market_data, sentiment)
        
        return {
            'sentiment_analysis': sentiment,
            'technical_analysis': technical,
            'risk_assessment': risk_analysis,
            'final_recommendation': self._synthesize_recommendation(
                sentiment, technical, risk_analysis
            )
        }
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """חישוב אינדיקטורים"""
        return {
            'rsi': data['rsi'].iloc[-1] if 'rsi' in data else None,
            'macd': data['macd'].iloc[-1] if 'macd' in data else None,
            'ma': data['close'].rolling(20).mean().iloc[-1]
        }
    
    def _assess_risk(self, data: pd.DataFrame, sentiment: Dict) -> str:
        """הערכת סיכון"""
        volatility = data['close'].pct_change().std() * 100
        
        risk_level = "low" if volatility < 1 else "medium" if volatility < 2 else "high"
        
        return f"Volatility: {volatility:.2f}%, Risk Level: {risk_level}, Sentiment Confidence: {sentiment['confidence']}%"
    
    def _synthesize_recommendation(self,
                                   sentiment: Dict,
                                   technical: str,
                                   risk: str) -> str:
        """סינתזה של כל הניתוחים להמלצה סופית"""
        if sentiment['sentiment'] == 'bullish' and sentiment['confidence'] > 70:
            return "Strong Buy Signal"
        elif sentiment['sentiment'] == 'bearish' and sentiment['confidence'] > 70:
            return "Strong Sell Signal"
        else:
            return "Hold - Mixed Signals"
```

---

## מערכת ניטור ואלרטים

### Alert Manager

**קובץ: `src/monitoring/alert_manager.py`**

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Callable
from datetime import datetime
import logging
from enum import Enum

class AlertLevel(Enum):
    """רמות חומרת התראה"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    
class AlertChannel(Enum):
    """ערוצי התראה"""
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    SLACK = "slack"
    LOG = "log"

class AlertManager:
    """
    מערכת ניהול התראות מתקדמת
    """
    
    def __init__(self, config: Dict):
        """
        אתחול
        
        Args:
            config: הגדרות (email, telegram, slack וכו')
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_rules = []
        self.alert_history = []
        
    def add_rule(self, 
                 name: str,
                 condition: Callable,
                 level: AlertLevel,
                 channels: List[AlertChannel],
                 cooldown_minutes: int = 60):
        """
        הוספת כלל התראה
        
        Args:
            name: שם הכלל
            condition: פונקציה שמחזירה True אם צריך להתריע
            level: רמת חומרה
            channels: ערוצי התראה
            cooldown_minutes: זמן המתנה בין התראות
        """
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'level': level,
            'channels': channels,
            'cooldown_minutes': cooldown_minutes,
            'last_triggered': None
        })
    
    def check_alerts(self, context: Dict):
        """
        בדיקת כל כללי ההתראה
        
        Args:
            context: הקשר נוכחי (מחירים, פוזיציות וכו')
        """
        for rule in self.alert_rules:
            try:
                # בדיקת cooldown
                if self._is_in_cooldown(rule):
                    continue
                
                # בדיקת תנאי
                if rule['condition'](context):
                    self._trigger_alert(rule, context)
                    rule['last_triggered'] = datetime.now()
            except Exception as e:
                self.logger.error(f"Error checking rule {rule['name']}: {e}")
    
    def _is_in_cooldown(self, rule: Dict) -> bool:
        """בדיקה אם הכלל ב-cooldown"""
        if rule['last_triggered'] is None:
            return False
        
        elapsed = (datetime.now() - rule['last_triggered']).total_seconds() / 60
        return elapsed < rule['cooldown_minutes']
    
    def _trigger_alert(self, rule: Dict, context: Dict):
        """הפעלת התראה"""
        alert = {
            'timestamp': datetime.now(),
            'rule_name': rule['name'],
            'level': rule['level'],
            'context': context
        }
        
        self.alert_history.append(alert)
        
        # שליחה לכל הערוצים
        for channel in rule['channels']:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email_alert(rule, context)
                elif channel == AlertChannel.TELEGRAM:
                    self._send_telegram_alert(rule, context)
                elif channel == AlertChannel.LOG:
                    self._log_alert(rule, context)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_email_alert(self, rule: Dict, context: Dict):
        """שליחת התראה במייל"""
        if 'email' not in self.config:
            return
        
        config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = config['from']
        msg['To'] = config['to']
        msg['Subject'] = f"[{rule['level'].value}] Trading Alert: {rule['name']}"
        
        body = f"""
        Alert: {rule['name']}
        Level: {rule['level'].value}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Context:
        {json.dumps(context, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()
    
    def _send_telegram_alert(self, rule: Dict, context: Dict):
        """שליחת התראה ל-Telegram"""
        if 'telegram' not in self.config:
            return
        
        config = self.config['telegram']
        
        message = f"""
        🚨 *{rule['level'].value}*: {rule['name']}
        
        Time: {datetime.now().strftime('%H:%M:%S')}
        
        Details:
        ```
        {json.dumps(context, indent=2)}
        ```
        """
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': config['chat_id'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        requests.post(url, json=payload)
    
    def _log_alert(self, rule: Dict, context: Dict):
        """רישום התראה ללוג"""
        log_method = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.CRITICAL: self.logger.critical
        }[rule['level']]
        
        log_method(f"ALERT: {rule['name']} | {context}")


# דוגמאות לכללי התראה נפוצים

def create_standard_alerts(alert_manager: AlertManager):
    """יצירת כללי התראה סטנדרטיים"""
    
    # התראה על הפסד יומי גדול
    alert_manager.add_rule(
        name="Daily Loss Threshold",
        condition=lambda ctx: ctx.get('daily_pnl', 0) < -5000,
        level=AlertLevel.CRITICAL,
        channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM, AlertChannel.LOG],
        cooldown_minutes=30
    )
    
    # התראה על חשיפה גבוהה
    alert_manager.add_rule(
        name="High Portfolio Exposure",
        condition=lambda ctx: ctx.get('total_exposure', 0) > 0.8,
        level=AlertLevel.WARNING,
        channels=[AlertChannel.LOG, AlertChannel.EMAIL],
        cooldown_minutes=120
    )
    
    # התראה על רצף הפסדים
    alert_manager.add_rule(
        name="Consecutive Losses",
        condition=lambda ctx: ctx.get('consecutive_losses', 0) >= 5,
        level=AlertLevel.WARNING,
        channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM],
        cooldown_minutes=240
    )
    
    # התראה על ביצועים מצוינים
    alert_manager.add_rule(
        name="Exceptional Performance",
        condition=lambda ctx: ctx.get('daily_return', 0) > 0.05,
        level=AlertLevel.INFO,
        channels=[AlertChannel.TELEGRAM],
        cooldown_minutes=360
    )
    
    # התראה על בעיית חיבור
    alert_manager.add_rule(
        name="Connection Lost",
        condition=lambda ctx: not ctx.get('is_connected', True),
        level=AlertLevel.CRITICAL,
        channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM, AlertChannel.LOG],
        cooldown_minutes=5
    )
```

---

## בדיקות (Testing)

### Unit Tests

**קובץ: `tests/test_strategies.py`**

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy

@pytest.fixture
def sample_data():
    """יצירת נתוני דוגמה"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # סימולציה של מחירים
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(len(dates)) * 0.5,
        'high': prices + np.abs(np.random.randn(len(dates))),
        'low': prices - np.abs(np.random.randn(len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    df.set_index('date', inplace=True)
    return df

class TestMovingAverageCrossover:
    """בדיקות לאסטרטגיית MA Crossover"""
    
    def test_initialization(self):
        """בדיקת אתחול"""
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        
        assert strategy.name == 'MA_Crossover'
        assert strategy.params['short_window'] == 10
        assert strategy.params['long_window'] == 20
    
    def test_calculate_indicators(self, sample_data):
        """בדיקת חישוב אינדיקטורים"""
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        
        df = strategy.calculate_indicators(sample_data)
        
        assert 'short_ma' in df.columns
        assert 'long_ma' in df.columns
        assert not df['short_ma'].isna().all()
        assert not df['long_ma'].isna().all()
    
    def test_generate_signals(self, sample_data):
        """בדיקת יצירת אותות"""
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        
        signals = strategy.generate_signals(sample_data)
        
        assert len(signals) == len(sample_data)
        assert signals.isin([0, 1, -1]).all()
        assert (signals == 1).any() or (signals == -1).any()  # לפחות אות אחד
    
    def test_signals_logic(self, sample_data):
        """בדיקת לוגיקת האותות"""
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        
        df = strategy.calculate_indicators(sample_data)
        signals = strategy.generate_signals(sample_data)
        
        # בדיקה שאותות מופיעים בזמן חציות
        for i in range(11, len(df)):
            if signals.iloc[i] == 1:  # אות קנייה
                # MA קצר צריך להיות מעל MA ארוך
                assert df['short_ma'].iloc[i] > df['long_ma'].iloc[i]
            elif signals.iloc[i] == -1:  # אות מכירה
                # MA קצר צריך להיות מתחת ל-MA ארוך
                assert df['short_ma'].iloc[i] < df['long_ma'].iloc[i]

class TestRSI_MACD_Strategy:
    """בדיקות לאסטרטגיית RSI + MACD"""
    
    def test_rsi_calculation(self, sample_data):
        """בדיקת חישוב RSI"""
        strategy = RSI_MACD_Strategy()
        
        rsi = strategy.calculate_rsi(sample_data['close'], period=14)
        
        assert not rsi.isna().all()
        assert (rsi >= 0).all() and (rsi <= 100).all()
    
    def test_macd_calculation(self, sample_data):
        """בדיקת חישוב MACD"""
        strategy = RSI_MACD_Strategy()
        
        macd, signal, hist = strategy.calculate_macd(sample_data['close'])
        
        assert not macd.isna().all()
        assert not signal.isna().all()
        assert len(macd) == len(sample_data)
    
    def test_combined_signals(self, sample_data):
        """בדיקת אותות משולבים"""
        strategy = RSI_MACD_Strategy(
            rsi_oversold=30,
            rsi_overbought=70
        )
        
        signals = strategy.generate_signals(sample_data)
        
        # בדיקה שיש איזון סביר בין אותות
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        assert buy_signals > 0 or sell_signals > 0

@pytest.fixture
def backtest_data():
    """נתונים לבדיקת backtesting"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 2),
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df

class TestBacktestEngine:
    """בדיקות למנוע Backtesting"""
    
    def test_simple_backtest(self, backtest_data):
        """בדיקת backtest בסיסי"""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001
        )
        
        # אותות פשוטים
        signals = pd.Series(0, index=backtest_data.index)
        signals.iloc[10] = 1   # קנה
        signals.iloc[50] = -1  # מכור
        
        results = engine.run(backtest_data, signals)
        
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert results['total_trades'] > 0
    
    def test_no_trades(self, backtest_data):
        """בדיקה ללא עסקאות"""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine()
        signals = pd.Series(0, index=backtest_data.index)  # אין אותות
        
        results = engine.run(backtest_data, signals)
        
        assert results['total_trades'] == 0
        assert results['final_capital'] == engine.initial_capital

class TestKellyCriterion:
    """בדיקות ל-Kelly Criterion"""
    
    def test_kelly_calculation(self):
        """בדיקת חישוב Kelly"""
        from src.risk_management.kelly_criterion import KellyCriterion
        
        kelly = KellyCriterion(kelly_fraction=0.5)
        
        # עסקאות דוגמה
        trades = [
            {'pnl': 100}, {'pnl': 150}, {'pnl': -50},
            {'pnl': 200}, {'pnl': -80}, {'pnl': 120}
        ]
        
        fraction = kelly.calculate_from_trades(trades)
        
        assert 0 <= fraction <= kelly.max_position_size
    
    def test_kelly_with_losses_only(self):
        """בדיקה עם הפסדים בלבד"""
        from src.risk_management.kelly_criterion import KellyCriterion
        
        kelly = KellyCriterion()
        
        trades = [{'pnl': -100}, {'pnl': -50}, {'pnl': -80}]
        
        fraction = kelly.calculate_from_trades(trades)
        
        assert fraction == 0.0  # אין להמליץ על מסחר
```

---

## Docker Deployment

### Dockerfile

**קובץ: `Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .cursorrules .

# Create necessary directories
RUN mkdir -p logs data models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Jerusalem

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "-m", "src.main"]
```

### docker-compose.yml

**קובץ: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  trading-system:
    build: .
    container_name: trading_system
    restart: unless-stopped
    environment:
      - TRADING_MODE=paper
      - IB_HOST=host.docker.internal
      - IB_PORT=7497
      - DATABASE_URL=postgresql://trader:password@postgres:5432/trading_db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
      - ./config:/app/config
    ports:
      - "8000:8000"
      - "8501:8501"  # Streamlit
    depends_on:
      - postgres
      - redis
    networks:
      - trading_network

  postgres:
    image: postgres:15-alpine
    container_name: trading_postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=trader
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=trading_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - trading_network

  redis:
    image: redis:7-alpine
    container_name: trading_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - trading_network

  ollama:
    image: ollama/ollama:latest
    container_name: trading_ollama
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - trading_network

volumes:
  postgres_data:
  redis_data:
  ollama_data:

networks:
  trading_network:
    driver: bridge
```

### הרצת המערכת עם Docker

```bash
# בניית התמונה
docker-compose build

# הרצה
docker-compose up -d

# צפייה בלוגים
docker-compose logs -f trading-system

# עצירה
docker-compose down

# עצירה עם מחיקת volumes
docker-compose down -v
```

---

## תרחישי שימוש מלאים

### תרחיש 1: הפעלת מסחר אוטומטי מלא

**קובץ: `examples/full_trading_example.py`**

```python
"""
דוגמה מלאה להפעלת מערכת מסחר אוטומטית
"""

import asyncio
from datetime import datetime
from src.broker.ib_connector import IBConnector
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.kelly_criterion import KellyCriterion
from src.monitoring.alert_manager import AlertManager, AlertLevel, AlertChannel
from src.learning.performance_tracker import PerformanceTracker
import logging

# הגדרת logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

class TradingSystem:
    """
    מערכת מסחר אוטומטית מלאה
    """
    
    def __init__(self, config: dict):
        """אתחול המערכת"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # התחברות לברוקר
        self.broker = IBConnector(
            host=config['broker']['host'],
            port=config['broker']['paper_port'],
            is_paper=True
        )
        
        # אסטרטגיות
        self.strategies = {
            'ma_crossover': MovingAverageCrossover(
                short_window=50,
                long_window=200
            ),
            'rsi_macd': RSI_MACD_Strategy()
        }
        
        # ניהול סיכונים
        self.position_sizer = PositionSizer(
            method='kelly',
            risk_per_trade=0.02
        )
        
        self.kelly = KellyCriterion(kelly_fraction=0.5)
        
        # התראות
        self.alert_manager = AlertManager(config.get('alerts', {}))
        self._setup_alerts()
        
        # מעקב ביצועים
        self.performance_tracker = PerformanceTracker()
        
        # משתנים פנימיים
        self.positions = {}
        self.account_value = 0
        self.is_running = False
    
    def _setup_alerts(self):
        """הגדרת כללי התראה"""
        # הפסד יומי
        self.alert_manager.add_rule(
            name="Daily Loss Limit",
            condition=lambda ctx: ctx.get('daily_pnl', 0) < -5000,
            level=AlertLevel.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.LOG]
        )
        
        # רווח גדול
        self.alert_manager.add_rule(
            name="Big Win",
            condition=lambda ctx: ctx.get('trade_pnl', 0) > 1000,
            level=AlertLevel.INFO,
            channels=[AlertChannel.LOG]
        )
    
    async def start(self):
        """הפעלת המערכת"""
        self.logger.info("Starting Trading System...")
        
        # התחברות
        if not self.broker.connect():
            self.logger.error("Failed to connect to broker")
            return
        
        self.logger.info("Connected to broker")
        
        # קבלת מידע חשבון
        account_info = self.broker.get_account_info()
        self.account_value = float(account_info.get('NetLiquidation', 0))
        
        self.logger.info(f"Account Value: ${self.account_value:,.2f}")
        
        self.is_running = True
        
        # לולאת מסחר ראשית
        await self.trading_loop()
    
    async def trading_loop(self):
        """לולאת מסחר ראשית"""
        symbols = self.config.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
        
        while self.is_running:
            try:
                for symbol in symbols:
                    await self.process_symbol(symbol)
                
                # בדיקת התראות
                context = self._get_context()
                self.alert_manager.check_alerts(context)
                
                # המתנה לפני הפעם הבאה
                await asyncio.sleep(60)  # כל דקה
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
    
    async def process_symbol(self, symbol: str):
        """עיבוד סימול בודד"""
        try:
            # קבלת נתונים
            data = self.broker.get_historical_data(
                symbol,
                duration='1 Y',
                bar_size='1 day'
            )
            
            if data.empty:
                return
            
            # יצירת אותות מכל האסטרטגיות
            signals = {}
            for name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signals(data)
                    signals[name] = signal.iloc[-1] if len(signal) > 0 else 0
                except Exception as e:
                    self.logger.error(f"Error in strategy {name}: {e}")
                    signals[name] = 0
            
            # קבלת החלטה משולבת
            decision = self._aggregate_signals(signals)
            
            # ביצוע עסקה
            if decision != 0:
                await self.execute_trade(symbol, decision, data)
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
    def _aggregate_signals(self, signals: dict) -> int:
        """
        צבירת אותות מאסטרטגיות שונות
        
        Returns:
            1: קנה, -1: מכור, 0: החזק
        """
        # וטו - אם יש signal חזק ממספיק אסטרטגיות
        buy_votes = sum(1 for s in signals.values() if s == 1)
        sell_votes = sum(1 for s in signals.values() if s == -1)
        
        threshold = len(signals) // 2  # רוב
        
        if buy_votes > threshold:
            return 1
        elif sell_votes > threshold:
            return -1
        else:
            return 0
    
    async def execute_trade(self, symbol: str, signal: int, data):
        """ביצוע עסקה"""
        try:
            current_price = data['close'].iloc[-1]
            
            # בדיקה אם יש כבר פוזיציה
            has_position = symbol in self.positions
            
            if signal == 1 and not has_position:
                # קנייה
                await self._buy(symbol, current_price)
                
            elif signal == -1 and has_position:
                # מכירה
                await self._sell(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def _buy(self, symbol: str, price: float):
        """קנייה"""
        # חישוב גודל פוזיציה
        stop_loss = price * 0.95  # 5% stop loss
        
        quantity = self.position_sizer.calculate(
            account_value=self.account_value,
            entry_price=price,
            stop_loss=stop_loss,
            trades_history=self.performance_tracker.trades
        )
        
        if quantity == 0:
            return
        
        # הגשת פקודה
        order_id = self.broker.place_market_order(
            symbol=symbol,
            quantity=quantity,
            action='BUY'
        )
        
        if order_id:
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'stop_loss': stop_loss,
                'entry_time': datetime.now()
            }
            
            self.logger.info(f"BUY: {quantity} {symbol} @ ${price:.2f}")
    
    async def _sell(self, symbol: str, price: float):
        """מכירה"""
        position = self.positions.get(symbol)
        if not position:
            return
        
        quantity = position['quantity']
        
        # הגשת פקודה
        order_id = self.broker.place_market_order(
            symbol=symbol,
            quantity=quantity,
            action='SELL'
        )
        
        if order_id:
            # חישוב רווח/הפסד
            pnl = (price - position['entry_price']) * quantity
            
            # רישום עסקה
            trade = {
                'symbol': symbol,
                'strategy': 'combined',
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now()
            }
            
            self.performance_tracker.log_trade(trade)
            
            # הסרת פוזיציה
            del self.positions[symbol]
            
            self.logger.info(f"SELL: {quantity} {symbol} @ ${price:.2f} | P&L: ${pnl:.2f}")
    
    def _get_context(self) -> dict:
        """קבלת context נוכחי להתראות"""
        positions_value = sum(
            p['quantity'] * p['entry_price'] 
            for p in self.positions.values()
        )
        
        return {
            'account_value': self.account_value,
            'positions_value': positions_value,
            'total_exposure': positions_value / self.account_value if self.account_value > 0 else 0,
            'num_positions': len(self.positions),
            'daily_pnl': self._calculate_daily_pnl(),
            'is_connected': self.broker.ib.isConnected()
        }
    
    def _calculate_daily_pnl(self) -> float:
        """חישוב P&L יומי"""
        # חישוב מהעסקאות של היום
        today_trades = [
            t for t in self.performance_tracker.trades
            if t['exit_time'].date() == datetime.now().date()
        ]
        
        return sum(t['pnl'] for t in today_trades)
    
    def stop(self):
        """עצירת המערכת"""
        self.logger.info("Stopping Trading System...")
        self.is_running = False
        
        # סגירת כל הפוזיציות
        for symbol in list(self.positions.keys()):
            # קבלת מחיר נוכחי
            data = self.broker.get_historical_data(symbol, '1 D', '1 hour')
            if not data.empty:
                current_price = data['close'].iloc[-1]
                asyncio.run(self._sell(symbol, current_price))
        
        # ניתוק מהברוקר
        self.broker.disconnect()
        
        # הצגת סיכום
        self._print_summary()
    
    def _print_summary(self):
        """הדפסת סיכום ביצועים"""
        print("\n" + "="*50)
        print("TRADING SYSTEM SUMMARY")
        print("="*50)
        
        for strategy_name, stats in self.performance_tracker.strategy_performance.items():
            print(f"\n{strategy_name}:")
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']*100:.2f}%")
            print(f"  Total P&L: ${stats['total_pnl']:.2f}")
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        
        # המלצות
        print("\nRecommendations:")
        recommendations = self.performance_tracker.get_strategy_recommendations()
        for strategy, rec in recommendations.items():
            print(f"  {strategy}: {rec}")
        
        print("\n" + "="*50)


# הרצה
async def main():
    config = {
        'broker': {
            'host': '127.0.0.1',
            'paper_port': 7497
        },
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'alerts': {
            'email': {
                'from': 'trading@example.com',
                'to': 'trader@example.com',
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email',
                'password': 'your_password'
            }
        }
    }
    
    system = TradingSystem(config)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        system.stop()

if __name__ == '__main__':
    asyncio.run(main())
```

---

## FAQ ופתרון בעיות

### שאלות נפוצות

**Q: איך מתחילים עם Paper Trading?**
A: 
1. פתח חשבון Paper Trading ב-Interactive Brokers
2. הורד והתקן TWS או IB Gateway
3. הגדר פורט 7497 
4. הרץ `python examples/full_trading_example.py`

**Q: איך יודעים אם אסטרטגיה טובה?**
A: בדוק את:
- Win Rate > 50%
- Sharpe Ratio > 1.0
- Profit Factor > 1.5
- Max Drawdown < 20%
- ביצועים עקביים על תקופות שונות

**Q: כמה זמן לאמן מודל ML?**
A: תלוי במורכבות:
- LSTM פשוט: 30-60 דקות
- DQN: 2-5 שעות
- Ensemble: 5-10 שעות

**Q: איך להימנע מ-Overfitting?**
A:
- השתמש ב-Walk-Forward Optimization
- פיצול נכון: 60% train, 20% validation, 20% test
- Early Stopping
- Regularization
- בדיקה על Out-of-Sample data

### בעיות נפוצות ופתרונות

**בעיה: "Connection refused" ל-IB**
```python
# פתרון:
# 1. ודא ש-TWS/Gateway פועלים
# 2. בדוק הגדרות API ב-TWS:
#    - Edit -> Global Configuration -> API -> Settings
#    - סמן "Enable ActiveX and Socket Clients"
#    - סמן "Allow connections from localhost"
#    - פורט 7497 (paper) או 7496 (live)
```

**בעיה: "No market data"**
```python
# פתרון:
# 1. ודא שיש מנוי לנתוני שוק (Market Data Subscriptions)
# 2. בדוק אם השוק פתוח
# 3. השתמש ב-delayed data בזמן פיתוח
```

**בעיה: מודל ML לא משתפר**
```python
# פתרון:
# 1. בדוק את איכות הנתונים
# 2. נסה Feature Engineering שונה
# 3. הגדל את Dataset
# 4. נסה ארכיטקטורה שונה
# 5. כוונן Hyperparameters
```

**בעיה: Backtest מהיר מדי לעומת Live**
```python
# פתרון:
# הוסף Slippage ועמלות:
from src.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine(
    commission=0.002,  # 0.2%
    slippage=0.001     # 0.1%
)
```

---

**בהצלחה בבניית מערכת המסחר! 🚀📈**

זכור: מסחר כולל סיכון. השתמש תמיד ב-Paper Trading תחילה!
