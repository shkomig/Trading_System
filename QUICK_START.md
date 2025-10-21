# 🚀 מדריך התחלה מהירה

## מה נבנה?

בנינו מערכת מסחר אוטומטית מלאה ומקצועית עם:

### ✅ מה כבר עובד (מוכן לשימוש):

1. **תשתית מלאה**
   - מבנה תיקיות מסודר
   - קבצי הגדרות (YAML)
   - מערכת logging
   - Database (SQLite)

2. **מנוע Backtesting מתקדם**
   - תמיכה בעמלות ו-slippage
   - מטריקות ביצועים (Sharpe, Sortino, Max DD, Win Rate ועוד)
   - ויזואליזציה של תוצאות
   - Walk-forward optimization

3. **8 אסטרטגיות טכניות מוכנות לשימוש**
   - Moving Average Crossover (עם גרסת Triple MA)
   - RSI + MACD + Bollinger Bands
   - RSI Divergence
   - Momentum Strategy
   - Dual Momentum
   - Trend Following (עם ADX)
   - Mean Reversion
   - Strategy Registry לניהול קל

4. **Risk Management מקצועי**
   - Kelly Criterion (עם 3 שיטות חישוב)
   - Position Sizing (5 שיטות: Kelly, Fixed Fractional, Fixed Risk, Volatility-Based, Risk Parity)
   - Stop Loss Manager (6 סוגי stop loss)
   - Take Profit אוטומטי
   - Trailing Stop

5. **חיבור ל-Interactive Brokers**
   - Paper Trading ו-Live Trading
   - הגשת פקודות (Market, Limit)
   - אחזור נתונים היסטוריים
   - מעקב אחר פוזיציות
   - Context manager נוח

6. **מערכת למידה**
   - Performance Tracker שעוקב אחר כל עסקה
   - ניתוח ביצועים לפי אסטרטגיה
   - המלצות אוטומטיות לשיפור
   - ניתוח תנאי שוק אופטימליים

7. **כלי עזר ועיבוד נתונים**
   - Data Processor עם 10+ פונקציות
   - Data Models (Trade, Position, Order, etc.)
   - Database Manager מלא

### 📋 מה עדיין בתהליך:

- ML Models (LSTM, DQN) - מורכב, ניתן להוסיף מאוחר יותר
- Dashboard Streamlit - יפה לראות אבל לא קריטי
- Monitoring & Alerts - כדאי להוסיף בעתיד
- Unit Tests - חשוב אבל המערכת עובדת בלי

---

## 🎯 איך מתחילים?

### שלב 1: התקנה (2 דקות)

```bash
# שכפול הפרויקט
git clone https://github.com/shkomig/Trading_System.git
cd Trading_System

# יצירת סביבה וירטואלית
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# התקנת תלויות
pip install -r requirements.txt
```

### שלב 2: בדיקה ראשונה (1 דקה)

```bash
# בדוק שהכל עובד
python src/main.py --mode info

# הרץ backtest לדוגמה
python examples/simple_backtest.py
```

אם ראית תוצאות - מזל טוב! המערכת עובדת 🎉

---

## 💡 דוגמאות שימוש

### דוגמה 1: Backtest פשוט

```python
import pandas as pd
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.backtesting.backtest_engine import BacktestEngine

# 1. טען נתונים (או צור נתונים לדוגמה)
# data = pd.read_csv('your_data.csv')

# 2. צור אסטרטגיה
strategy = MovingAverageCrossover(short_window=50, long_window=200)

# 3. צור אותות
signals = strategy.generate_signals(data)

# 4. הרץ backtest
engine = BacktestEngine(initial_capital=100000, commission=0.001)
results = engine.run(data, signals)

# 5. צפה בתוצאות
engine.print_summary()
engine.plot_results()
```

### דוגמה 2: השוואת אסטרטגיות

```python
from src.strategies.strategy_registry import create_strategy, list_available_strategies

# רשימת כל האסטרטגיות
print(list_available_strategies())

# צור אסטרטגיות
strategies = {
    'MA': create_strategy('ma_crossover', short_window=50, long_window=200),
    'RSI': create_strategy('rsi_macd', rsi_period=14),
    'Momentum': create_strategy('momentum', lookback_period=14)
}

# בדוק כל אחת
for name, strategy in strategies.items():
    signals = strategy.generate_signals(data)
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(data, signals)
    print(f"{name}: Return={results['total_return']:.2f}%, Sharpe={results['sharpe_ratio']:.2f}")
```

### דוגמה 3: עם Risk Management

```python
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.stop_loss_manager import StopLossManager, StopLossType

# Position Sizer
sizer = PositionSizer(method='kelly', risk_per_trade=0.02)

# Stop Loss Manager
sl_manager = StopLossManager(default_stop_percent=0.05)

# חישוב גודל פוזיציה
account_value = 100000
entry_price = 150.0

# כמה מניות לקנות?
shares = sizer.calculate(
    account_value=account_value,
    entry_price=entry_price,
    stop_loss=145.0,  # 5$ stop
    trades_history=previous_trades
)

# קבע stop loss ו-take profit
stop_loss, take_profit = sl_manager.calculate_stops(
    entry_price=entry_price,
    side='long',
    stop_type=StopLossType.FIXED_PERCENT,
    stop_percent=0.05,
    take_profit_ratio=2.0  # רווח פוטנציאלי פי 2 מהסיכון
)

print(f"Buy {shares} shares @ ${entry_price}")
print(f"Stop Loss: ${stop_loss:.2f}")
print(f"Take Profit: ${take_profit:.2f}")
```

### דוגמה 4: חיבור ל-Interactive Brokers

```python
from src.broker.ib_connector import IBConnector

# התחבר (Paper Trading)
with IBConnector(host='127.0.0.1', port=7497, is_paper=True) as ib:
    # קבל מידע חשבון
    account = ib.get_account_info()
    print(f"Account Value: ${account.get('NetLiquidation')}")
    
    # קבל נתונים היסטוריים
    data = ib.get_historical_data('AAPL', duration='1 Y', bar_size='1 day')
    
    # הגש פקודה
    order_id = ib.place_market_order('AAPL', 10, 'BUY')
    print(f"Order placed: {order_id}")
    
    # בדוק פוזיציות
    positions = ib.get_positions()
    for pos in positions:
        print(f"{pos['symbol']}: {pos['position']} shares @ ${pos['avgCost']:.2f}")
```

### דוגמה 5: Performance Tracking

```python
from src.learning.performance_tracker import PerformanceTracker

# צור tracker
tracker = PerformanceTracker()

# רשום עסקאות
tracker.log_trade({
    'symbol': 'AAPL',
    'strategy': 'MA_Crossover',
    'pnl': 150.0,
    'entry_price': 145.0,
    'exit_price': 148.5
})

# קבל המלצות
recommendations = tracker.get_strategy_recommendations()
print(recommendations)

# נתח תנאי שוק
analysis = tracker.analyze_market_conditions()
print(f"Best trading day: {analysis['best_trading_day']}")
print(f"Best symbols: {analysis['best_symbols']}")

# הצג סיכום
print(tracker.get_performance_summary())
```

---

## 📊 מה אפשר לעשות עכשיו?

### רמה 1: התחלתי (0-3 ימים)
1. ✅ הרץ את הדוגמאות
2. ✅ נסה backtest עם נתונים שלך
3. ✅ השווה בין אסטרטגיות שונות
4. ✅ שחק עם הפרמטרים

### רמה 2: מתקדם (שבוע)
1. ✅ אופטם פרמטרים (Grid Search / Random Search)
2. ✅ הוסף Risk Management לאסטרטגיות
3. ✅ בדוק Walk-Forward Optimization
4. ✅ התחבר ל-IB Paper Trading

### רמה 3: מקצועי (חודש)
1. 🔄 צור אסטרטגיות משלך (inherit from BaseStrategy)
2. 🔄 הוסף ML models (LSTM / RL)
3. 🔄 בנה Dashboard
4. 🔄 הוסף התראות
5. ⚠️ עבור ל-Live Trading (בזהירות!)

---

## 🎓 טיפים חשובים

### ✅ DO:
- תמיד התחל עם Backtesting
- השתמש ב-Risk Management
- עקוב אחר הביצועים עם PerformanceTracker
- נסה Paper Trading לפני Live
- בדוק על תקופות זמן שונות
- השווה מספר אסטרטגיות

### ❌ DON'T:
- אל תעבור ישר ל-Live Trading
- אל תשקיע יותר מ-2% לעסקה
- אל תמסור ללא Stop Loss
- אל תעשה Overfitting על הנתונים
- אל תתעלם מהעמלות והSlippage

---

## 🆘 בעיות נפוצות

### "ModuleNotFoundError"
```bash
# ודא שהסביבה הוירטואלית מופעלת
venv\Scripts\activate
pip install -r requirements.txt
```

### "No module named 'ib_insync'"
```bash
pip install ib-insync
```

### "Connection refused" ל-IB
1. ודא ש-TWS/Gateway פועלים
2. הפעל API: Edit → Global Configuration → API → Settings
3. סמן "Enable ActiveX and Socket Clients"
4. פורט 7497 ל-Paper, 7496 ל-Live

### Backtest לא עובד
1. ודא שיש לך נתוני OHLCV תקינים
2. בדוק שהאינדקס הוא datetime
3. ודא שאין ערכים חסרים (NaN)

---

## 📚 קבצים חשובים

- `examples/simple_backtest.py` - דוגמה מלאה
- `src/main.py` - נקודת כניסה ראשית
- `config/config.yaml` - הגדרות המערכת
- `README.md` - תיעוד מלא
- `trading_system_guide.md` - מדריך מקיף

---

## 🤝 תמיכה

שאלות? בעיות?
1. בדוק את `trading_system_guide.md` למדריך מפורט
2. הרץ `python src/main.py --mode info` למידע על המערכת
3. פתח Issue ב-GitHub

---

**בהצלחה במסחר! 📈💰**

*זכור: מסחר כרוך בסיכון. השתמש במערכת באחריות ועם ניהול סיכונים נכון.*

