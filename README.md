# 📈 מערכת מסחר אוטומטית מתקדמת

מערכת מסחר אוטומטית מקצועית עם למידת מכונה, Backtesting, ו-Risk Management מתקדם.

## ✨ תכונות עיקריות

- 🤖 **אסטרטגיות מסחר מגוונות** - טכניות ומבוססות למידת מכונה
- 📊 **Backtesting מתקדם** - בדיקה מקיפה עם מטריקות ביצועים
- 💰 **Risk Management** - Kelly Criterion, Position Sizing, Stop Loss
- 🔗 **Interactive Brokers Integration** - חיבור ל-Paper/Live Trading
- 🧠 **למידת מכונה** - LSTM Predictor, DQN Agent
- 📈 **Dashboard אינטראקטיבי** - ממשק Streamlit מלא
- 🚨 **מערכת התראות** - Email, Telegram, Logs
- 🤖 **Local LLM Integration** - ניתוח סנטימנט והמלצות

## 📁 מבנה הפרויקט

```
trading_system/
├── src/
│   ├── broker/              # חיבור ל-Interactive Brokers
│   ├── strategies/          # אסטרטגיות מסחר
│   │   ├── technical/       # אסטרטגיות טכניות
│   │   └── ml_based/        # אסטרטגיות ML
│   ├── backtesting/         # מנוע Backtesting
│   ├── risk_management/     # ניהול סיכונים
│   ├── ml_models/           # מודלי למידת מכונה
│   ├── data/                # ניהול נתונים
│   ├── learning/            # מערכת למידה
│   ├── monitoring/          # ניטור והתראות
│   └── ui/                  # ממשק משתמש
├── config/                  # קבצי הגדרות
├── tests/                   # בדיקות
├── examples/                # דוגמאות שימוש
├── data/                    # נתונים
├── models/                  # מודלים מאומנים
└── logs/                    # לוגים
```

## 🚀 התקנה והתחלה מהירה

### דרישות מקדימות

- Python 3.10 או גבוה יותר
- pip
- Git

### התקנה מהירה

```bash
# 1. שכפול הפרויקט
git clone https://github.com/shkomig/Trading_System.git
cd Trading_System

# 2. יצירת סביבה וירטואלית
python -m venv venv

# 3. הפעלת הסביבה
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. התקנת תלויות
pip install -r requirements.txt

# 5. בדיקה שהכל עובד
python src/main.py --mode info
```

### הרצה ראשונה

```bash
# הרצת דוגמה מלאה
python examples/simple_backtest.py
```

זהו! המערכת מוכנה לשימוש 🎉

## 📖 שימוש מהיר

### הרצת Backtest

```python
from src.backtesting.backtest_engine import BacktestEngine
from src.strategies.technical.moving_average import MovingAverageCrossover
import pandas as pd

# טעינת נתונים
data = pd.read_csv('data/historical/AAPL.csv', index_col='date', parse_dates=True)

# יצירת אסטרטגיה
strategy = MovingAverageCrossover(short_window=50, long_window=200)

# יצירת אותות
signals = strategy.generate_signals(data)

# הרצת backtest
engine = BacktestEngine(initial_capital=100000, commission=0.001)
results = engine.run(data, signals)

# הצגת תוצאות
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

# ציור גרפים
engine.plot_results()
```

### הוספת אסטרטגיה חדשה

```python
from src.strategies.base_strategy import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    """אסטרטגיה מותאמת אישית"""
    
    def __init__(self, param1=10, param2=20):
        params = {'param1': param1, 'param2': param2}
        super().__init__('MyStrategy', params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב אינדיקטורים"""
        df = data.copy()
        # הוסף אינדיקטורים כאן
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """יצירת אותות מסחר"""
        df = self.calculate_indicators(data)
        signals = pd.Series(0, index=df.index)
        # לוגיקת האותות כאן
        return signals
```

### הרצת Dashboard

```bash
streamlit run src/ui/dashboard.py
```

### חיבור ל-Interactive Brokers

```python
from src.broker.ib_connector import IBConnector

# חיבור (Paper Trading)
broker = IBConnector(host='127.0.0.1', port=7497, is_paper=True)
broker.connect()

# קבלת מידע חשבון
account_info = broker.get_account_info()
print(f"Account Value: ${account_info['NetLiquidation']}")

# קבלת נתונים היסטוריים
data = broker.get_historical_data('AAPL', duration='1 Y', bar_size='1 day')

# הגשת פקודה
order_id = broker.place_market_order('AAPL', 100, 'BUY')
```

## 🧪 הרצת בדיקות

```bash
# כל הבדיקות
pytest tests/

# עם כיסוי
pytest --cov=src tests/

# בדיקות ספציפיות
pytest tests/unit/test_strategies.py
```

## 📊 אסטרטגיות זמינות

### אסטרטגיות טכניות
- **MA Crossover** - חציית ממוצעים נעים
- **RSI + MACD + BB** - אסטרטגיה משולבת
- **Momentum** - מבוססת מומנטום

### אסטרטגיות ML
- **LSTM Predictor** - חיזוי מחירים עם LSTM
- **DQN Agent** - למידת חיזוק עמוקה

## ⚙️ הגדרות

ערוך את `config/config.yaml` לשינוי הגדרות:

```yaml
trading:
  max_positions: 5
  default_position_size: 0.2

risk_management:
  risk_per_trade: 0.02
  max_daily_loss: 0.05
  kelly_fraction: 0.5

strategies:
  enabled:
    - "MA_Crossover"
    - "RSI_MACD"
```

## 📈 דוגמאות

### דוגמה מלאה למערכת מסחר

```bash
python examples/full_trading_example.py
```

### אימון מודל LSTM

```bash
python examples/train_lstm.py
```

### אופטימיזציית אסטרטגיה

```bash
python examples/optimize_strategy.py
```

## 🔒 אבטחה

- **אל תשתף** את קובץ `.env` או credentials
- השתמש ב-**Paper Trading** בזמן פיתוח
- **אמת נתונים** לפני שימוש
- **הגדר stop-loss** לכל פוזיציה

## ⚠️ אזהרה

**מסחר כרוך בסיכון משמעותי. אל תמסור יותר ממה שאתה יכול להרשות לעצמך להפסיד.**

- תמיד התחל עם **Paper Trading**
- בדוק אסטרטגיות היטב עם **Backtesting**
- השתמש ב-**Risk Management** מתאים
- עקוב אחר הביצועים והתאם לפי הצורך

## 📚 תיעוד נוסף

- [מדריך למתחילים](docs/getting_started.md)
- [תיעוד API](docs/api.md)
- [יצירת אסטרטגיות](docs/creating_strategies.md)
- [Risk Management](docs/risk_management.md)

## 🤝 תרומה

תרומות מתקבלות בברכה! אנא:
1. צור Fork
2. צור Branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. פתח Pull Request

## 📝 רישיון

הפרויקט הזה הוא לשימוש חינוכי בלבד.

## 💬 תמיכה

לשאלות ותמיכה:
- פתח Issue ב-GitHub
- בדוק את ה-FAQ במדריך

---

**בהצלחה במסחר! 🚀**

*זכור: מסחר אוטומטי דורש בדיקה, ניטור והתאמה מתמדת.*

