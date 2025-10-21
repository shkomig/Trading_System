"""
בדיקת חיבור חי ל-Interactive Brokers Paper Trading
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.broker.ib_connector import IBConnector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_live_connection():
    """בדיקת חיבור חי ל-IB"""
    
    print("\n" + "="*70)
    print(">>> בדיקת חיבור ל-Interactive Brokers Paper Trading")
    print("="*70)
    
    # יצירת חיבור
    connector = IBConnector(
        host='127.0.0.1',
        port=7497,  # Paper Trading
        client_id=1,
        is_paper=True
    )
    
    print("\n[1] מנסה להתחבר ל-IB Gateway/TWS על פורט 7497...")
    
    if not connector.connect():
        print("\n[!] החיבור נכשל!")
        print("\n>> וודא ש:")
        print("  * IB Gateway/TWS פועל")
        print("  * API מופעל (Configure -> API -> Settings)")
        print("  * הפורט הוא 7497")
        print("  * 'Enable ActiveX and Socket Clients' מסומן")
        return False
    
    print("[OK] החיבור הצליח!")
    
    # מידע על החשבון
    print("\n[2] מקבל מידע על החשבון...")
    account_info = connector.get_account_info()
    
    if account_info:
        print("\n>> פרטי החשבון Paper Trading:")
        for key, value in list(account_info.items())[:10]:
            print(f"  * {key}: {value}")
    
    # נתונים היסטוריים
    print("\n[3] מנסה לקבל נתונים היסטוריים של AAPL...")
    
    data = connector.get_historical_data(
        symbol='AAPL',
        duration='1 M',
        bar_size='1 day'
    )
    
    if data is not None and not data.empty:
        print(f"[OK] התקבלו {len(data)} ימי מסחר")
        print(f"\n>> מחירי AAPL אחרונים:")
        print(f"  תאריך: {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  פתיחה: ${data['open'].iloc[-1]:.2f}")
        print(f"  סגירה: ${data['close'].iloc[-1]:.2f}")
        print(f"  גבוה: ${data['high'].iloc[-1]:.2f}")
        print(f"  נמוך: ${data['low'].iloc[-1]:.2f}")
        print(f"  נפח: {data['volume'].iloc[-1]:,.0f}")
        
        # חישוב שינוי
        if len(data) > 1:
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
            pct_change = (price_change / data['close'].iloc[-2]) * 100
            print(f"  שינוי: {price_change:+.2f} ({pct_change:+.2f}%)")
    else:
        print("[!] לא הצלחתי לקבל נתונים")
    
    # רשימת פוזיציות
    print("\n[4] בודק פוזיציות פתוחות...")
    positions = connector.get_positions()
    
    if positions:
        print(f"\n>> יש {len(positions)} פוזיציות פתוחות:")
        for pos in positions:
            print(f"  * {pos.get('symbol')}: {pos.get('position')} מניות")
    else:
        print("[OK] אין פוזיציות פתוחות (חשבון נקי)")
    
    # סיום
    print("\n[5] מתנתק...")
    connector.disconnect()
    print("[OK] נותק בהצלחה")
    
    print("\n" + "="*70)
    print(">>> בדיקה הושלמה בהצלחה!")
    print("="*70)
    print("\n>> המערכת שלך מחוברת ל-IB Paper Trading ועובדת!")
    print("\n>> כעת אפשר:")
    print("  1. להריץ Backtest עם נתונים אמיתיים מ-IB")
    print("  2. לנסות אסטרטגיה בפועל (Paper Trading)")
    print("  3. להמשיך בפיתוח תכונות נוספות")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_live_connection()
        if not success:
            print("\n[!] החיבור נכשל. נסה שוב או בדוק את ההגדרות.")
    except KeyboardInterrupt:
        print("\n\n[!] הופסק על ידי המשתמש")
    except Exception as e:
        print(f"\n[!] שגיאה: {e}")
        import traceback
        traceback.print_exc()
