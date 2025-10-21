"""
הדגמת אסטרטגיה חיה עם Interactive Brokers Paper Trading
מערכת מנתחת מניות בזמן אמת ונותנת המלצות מסחר
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.broker.ib_connector import IBConnector
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.strategies.technical.rsi_macd import RSI_MACD_Strategy
from src.strategies.technical.momentum import MomentumStrategy
import time

def analyze_symbol_with_strategy(connector, symbol, strategy, strategy_name):
    """ניתוח מניה עם אסטרטגיה ספציפית"""
    
    print(f"\n{'='*70}")
    print(f"מנתח: {symbol} עם אסטרטגיה: {strategy_name}")
    print('='*70)
    
    # קבל נתונים היסטוריים
    print(f"\n[1] מקבל נתונים היסטוריים של {symbol}...")
    data = connector.get_historical_data(symbol, '6 M', '1 day')
    
    if data is None or data.empty:
        print(f"[!] לא הצלחתי לקבל נתונים עבור {symbol}")
        return None
    
    print(f"[OK] התקבלו {len(data)} ימי מסחר")
    
    # הרץ אסטרטגיה
    print(f"\n[2] מריץ אסטרטגיה: {strategy_name}...")
    try:
        indicators = strategy.calculate_indicators(data)
        signals = strategy.generate_signals(data)
        
        # המידע האחרון
        last_signal = signals.iloc[-1]
        last_price = data['close'].iloc[-1]
        last_date = data.index[-1]
        
        # חישוב שינוי מחיר
        if len(data) > 1:
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
            pct_change = (price_change / data['close'].iloc[-2]) * 100
        else:
            price_change = 0
            pct_change = 0
        
        # הצג תוצאות
        print(f"\n[OK] ניתוח הושלם!")
        print(f"\n>> מידע על {symbol}:")
        print(f"   תאריך: {last_date.strftime('%Y-%m-%d')}")
        print(f"   מחיר נוכחי: ${last_price:.2f}")
        print(f"   שינוי יומי: {price_change:+.2f} ({pct_change:+.2f}%)")
        
        # פענוח האות
        signal_names = {-1: 'SELL (מכור)', 0: 'HOLD (החזק)', 1: 'BUY (קנה)'}
        signal_name = signal_names.get(int(last_signal), 'UNKNOWN')
        
        print(f"\n>> אות מ-{strategy_name}:")
        print(f"   >>> {signal_name} <<<")
        
        # המלצה פרטנית
        if last_signal == 1:  # Buy
            quantity = int(5000 / last_price)  # $5000 שווי
            total_cost = quantity * last_price
            
            print(f"\n[!] המלצת קנייה!")
            print(f"   כמות מומלצת: {quantity} מניות")
            print(f"   שווי כולל: ${total_cost:.2f}")
            print(f"   עמלה משוערת (0.1%): ${total_cost * 0.001:.2f}")
            
            # הצג אינדיקטורים
            if 'short_ma' in indicators.columns and 'long_ma' in indicators.columns:
                short_ma = indicators['short_ma'].iloc[-1]
                long_ma = indicators['long_ma'].iloc[-1]
                print(f"\n   אינדיקטורים:")
                print(f"   MA קצר: ${short_ma:.2f}")
                print(f"   MA ארוך: ${long_ma:.2f}")
                print(f"   מרחק: {((short_ma - long_ma) / long_ma * 100):+.2f}%")
            
        elif last_signal == -1:  # Sell
            print(f"\n[!] המלצת מכירה!")
            print(f"   אם יש לך פוזיציה - שקול לצאת")
            
        else:  # Hold
            print(f"\n[OK] HOLD - אין פעולה מומלצת כרגע")
            print(f"   המשך לעקוב אחר המניה")
        
        # סיכון
        print(f"\n>> ניהול סיכונים:")
        stop_loss_pct = 0.05  # 5%
        stop_loss_price = last_price * (1 - stop_loss_pct)
        print(f"   Stop Loss מומלץ: ${stop_loss_price:.2f} (-{stop_loss_pct*100}%)")
        
        return {
            'symbol': symbol,
            'price': last_price,
            'signal': int(last_signal),
            'signal_name': signal_name,
            'strategy': strategy_name
        }
        
    except Exception as e:
        print(f"[!] שגיאה בהרצת האסטרטגיה: {e}")
        return None


def main():
    """הרצה ראשית"""
    
    print("\n" + "="*70)
    print(">>> הדגמת אסטרטגיה חיה - IB Paper Trading <<<")
    print("="*70)
    print("\n[!] אזהרה: זוהי הדגמה בלבד!")
    print("    המערכת תנתח מניות ותיתן המלצות")
    print("    אבל לא תבצע עסקאות אוטומטית")
    print("    הסר הערות בקוד כדי לבצע עסקאות בפועל (Paper Trading)")
    
    # התחבר ל-IB
    print("\n" + "="*70)
    print("[*] מתחבר ל-Interactive Brokers...")
    print("="*70)
    
    connector = IBConnector(host='127.0.0.1', port=7497, is_paper=True)
    
    if not connector.connect():
        print("\n[!] החיבור נכשל!")
        print("    וודא ש-IB Gateway/TWS פועל עם API מופעל")
        return
    
    print("[OK] מחובר ל-IB Paper Trading!")
    
    # הצג מידע חשבון
    account_info = connector.get_account_info()
    if account_info:
        buying_power = account_info.get('BuyingPower', 'N/A')
        print(f"\n[*] כוח קנייה זמין: ${buying_power}")
    
    # רשימת מניות לניתוח
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    
    # אסטרטגיות לבדיקה
    strategies = [
        (MovingAverageCrossover(short_window=20, long_window=50), 
         "MA Crossover (20/50)"),
        (RSI_MACD_Strategy(rsi_period=14, rsi_overbought=70, rsi_oversold=30),
         "RSI + MACD"),
        (MomentumStrategy(lookback_period=20),
         "Momentum")
    ]
    
    # נתח כל מניה
    results = []
    
    print("\n\n" + "="*70)
    print(">>> מתחיל ניתוח מניות <<<")
    print("="*70)
    
    for symbol in symbols:
        print(f"\n{'#'*70}")
        print(f"# ניתוח: {symbol}")
        print(f"{'#'*70}")
        
        for strategy, strategy_name in strategies:
            result = analyze_symbol_with_strategy(
                connector, 
                symbol, 
                strategy, 
                strategy_name
            )
            
            if result:
                results.append(result)
            
            time.sleep(0.5)  # המתנה קצרה בין בקשות
    
    # סיכום
    print("\n\n" + "="*70)
    print(">>> סיכום המלצות <<<")
    print("="*70)
    
    buy_signals = [r for r in results if r['signal'] == 1]
    sell_signals = [r for r in results if r['signal'] == -1]
    hold_signals = [r for r in results if r['signal'] == 0]
    
    print(f"\n[*] סה\"כ ניתוחים: {len(results)}")
    print(f"    אותות קנייה (BUY): {len(buy_signals)}")
    print(f"    אותות מכירה (SELL): {len(sell_signals)}")
    print(f"    החזק (HOLD): {len(hold_signals)}")
    
    if buy_signals:
        print(f"\n>> המלצות קנייה חזקות:")
        for sig in buy_signals:
            print(f"   * {sig['symbol']}: ${sig['price']:.2f} - {sig['strategy']}")
    
    if sell_signals:
        print(f"\n>> המלצות מכירה:")
        for sig in sell_signals:
            print(f"   * {sig['symbol']}: ${sig['price']:.2f} - {sig['strategy']}")
    
    # הצעה לפעולה
    print("\n\n" + "="*70)
    print(">>> מה הלאה? <<<")
    print("="*70)
    
    if buy_signals:
        print("\n[!] נמצאו אותות קנייה!")
        print("    אופציות:")
        print("    1. עיין בניתוח מפורט למעלה")
        print("    2. בדוק את המניות ב-Dashboard")
        print("    3. הסר הערות בקוד לביצוע עסקה (Paper Trading)")
        print("\n    דוגמת קוד לביצוע עסקה:")
        best_buy = buy_signals[0]
        print(f"    # order_id = connector.place_market_order('{best_buy['symbol']}', 10, 'BUY')")
        print(f"    # print(f'הזמנה בוצעה: {{order_id}}')")
    else:
        print("\n[OK] אין אותות קנייה כרגע")
        print("    המשך לעקוב אחר השוק")
    
    # ניתוק
    print("\n\n" + "="*70)
    print("[*] מתנתק מ-IB...")
    connector.disconnect()
    print("[OK] ההדגמה הושלמה!")
    print("="*70 + "\n")
    
    print("\n>> תודה שהשתמשת במערכת המסחר!")
    print(">> כל הניתוחים נשמרו ללוגים")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] הופסק על ידי המשתמש")
    except Exception as e:
        print(f"\n[!] שגיאה: {e}")
        import traceback
        traceback.print_exc()

