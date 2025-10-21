"""
ניהול מסד נתונים למערכת המסחר

כולל פונקציות לשמירה וטעינה של עסקאות, פוזיציות ונתונים היסטוריים.
"""

import sqlite3
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TradingDatabase:
    """
    מנהל מסד נתונים למערכת מסחר
    
    תומך ב-SQLite כברירת מחדל.
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        אתחול חיבור למסד נתונים
        
        Args:
            db_path: נתיב למסד הנתונים
        """
        self.db_path = db_path
        
        # יצירת תיקייה אם לא קיימת
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._create_tables()
        
        logger.info(f"Database initialized at {db_path}")
    
    def connect(self):
        """יצירת חיבור למסד נתונים"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """סגירת חיבור"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _create_tables(self):
        """יצירת טבלאות"""
        self.connect()
        
        cursor = self.conn.cursor()
        
        # טבלת עסקאות
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                pnl REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # טבלת פוזיציות
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL DEFAULT 0,
                stop_loss REAL,
                take_profit REAL,
                strategy TEXT,
                entry_time TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # טבלת ביצועי אסטרטגיות
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                date TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                metrics_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, date)
            )
        """)
        
        # טבלת ביצועים יומיים
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                starting_balance REAL NOT NULL,
                ending_balance REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                daily_return REAL DEFAULT 0,
                num_trades INTEGER DEFAULT 0,
                num_positions INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # טבלת נתונים היסטוריים
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(symbol, date)
            )
        """)
        
        # אינדקסים לביצועים
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, date)
        """)
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def save_trade(self, trade: Dict) -> int:
        """
        שמירת עסקה
        
        Args:
            trade: מילון עם פרטי עסקה
            
        Returns:
            מזהה העסקה
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                symbol, strategy, action, quantity, entry_price, exit_price,
                entry_time, exit_time, pnl, commission, slippage, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get('symbol'),
            trade.get('strategy'),
            trade.get('action'),
            trade.get('quantity'),
            trade.get('entry_price'),
            trade.get('exit_price'),
            trade.get('entry_time'),
            trade.get('exit_time'),
            trade.get('pnl', 0),
            trade.get('commission', 0),
            trade.get('slippage', 0),
            trade.get('notes', '')
        ))
        
        self.conn.commit()
        trade_id = cursor.lastrowid
        
        logger.info(f"Trade saved with ID: {trade_id}")
        return trade_id
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        קבלת עסקאות
        
        Args:
            symbol: סינון לפי סימול
            strategy: סינון לפי אסטרטגיה
            start_date: תאריך התחלה
            end_date: תאריך סיום
            limit: מגבלת מספר תוצאות
            
        Returns:
            DataFrame עם עסקאות
        """
        self.connect()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY entry_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def save_position(self, position: Dict) -> int:
        """
        שמירת פוזיציה
        
        Args:
            position: מילון עם פרטי פוזיציה
            
        Returns:
            מזהה הפוזיציה
        """
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO positions (
                symbol, side, quantity, entry_price, current_price,
                stop_loss, take_profit, strategy, entry_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.get('symbol'),
            position.get('side'),
            position.get('quantity'),
            position.get('entry_price'),
            position.get('current_price', 0),
            position.get('stop_loss'),
            position.get('take_profit'),
            position.get('strategy'),
            position.get('entry_time')
        ))
        
        self.conn.commit()
        position_id = cursor.lastrowid
        
        logger.info(f"Position saved with ID: {position_id}")
        return position_id
    
    def update_position(self, position_id: int, updates: Dict):
        """
        עדכון פוזיציה
        
        Args:
            position_id: מזהה פוזיציה
            updates: מילון עם עדכונים
        """
        self.connect()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [position_id]
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            UPDATE positions 
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, values)
        
        self.conn.commit()
        logger.info(f"Position {position_id} updated")
    
    def delete_position(self, position_id: int):
        """מחיקת פוזיציה"""
        self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM positions WHERE id = ?", (position_id,))
        self.conn.commit()
        
        logger.info(f"Position {position_id} deleted")
    
    def get_positions(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        קבלת פוזיציות פתוחות
        
        Args:
            symbol: סינון לפי סימול
            
        Returns:
            DataFrame עם פוזיציות
        """
        self.connect()
        
        query = "SELECT * FROM positions"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def save_strategy_performance(self, strategy_name: str, metrics: Dict):
        """
        שמירת ביצועי אסטרטגיה
        
        Args:
            strategy_name: שם אסטרטגיה
            metrics: מטריקות ביצועים
        """
        self.connect()
        
        date = datetime.now().date().isoformat()
        metrics_json = json.dumps(metrics)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO strategy_performance (
                strategy_name, date, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, profit_factor, sharpe_ratio, max_drawdown, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_name,
            date,
            metrics.get('total_trades', 0),
            metrics.get('winning_trades', 0),
            metrics.get('losing_trades', 0),
            metrics.get('total_pnl', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0),
            metrics_json
        ))
        
        self.conn.commit()
        logger.info(f"Strategy performance saved for {strategy_name}")
    
    def get_strategy_performance(
        self,
        strategy_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        קבלת ביצועי אסטרטגיות
        
        Args:
            strategy_name: שם אסטרטגיה (None = כל האסטרטגיות)
            start_date: תאריך התחלה
            end_date: תאריך סיום
            
        Returns:
            DataFrame עם ביצועים
        """
        self.connect()
        
        query = "SELECT * FROM strategy_performance WHERE 1=1"
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def save_market_data(self, symbol: str, data: pd.DataFrame):
        """
        שמירת נתונים היסטוריים
        
        Args:
            symbol: סימול
            data: DataFrame עם נתוני OHLCV
        """
        self.connect()
        
        data = data.copy()
        data['symbol'] = symbol
        data['date'] = data.index.astype(str)
        
        data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']].to_sql(
            'market_data',
            self.conn,
            if_exists='append',
            index=False
        )
        
        self.conn.commit()
        logger.info(f"Market data saved for {symbol}: {len(data)} rows")
    
    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        קבלת נתונים היסטוריים
        
        Args:
            symbol: סימול
            start_date: תאריך התחלה
            end_date: תאריך סיום
            
        Returns:
            DataFrame עם נתוני OHLCV
        """
        self.connect()
        
        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        return df
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

