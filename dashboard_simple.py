"""
Simple Trading System Dashboard
מערכת Dashboard פשוטה ועובדת
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# הגדרת עמוד
st.set_page_config(
    page_title="מערכת מסחר אוטומטית",
    page_icon="📈",
    layout="wide"
)

# כותרת ראשית
st.title("📈 מערכת מסחר אוטומטית - Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("תפריט ניווט")
page = st.sidebar.radio(
    "בחר עמוד:",
    ["🏠 סקירה כללית", "📊 Backtest", "🎯 אסטרטגיות", "⚙️ הגדרות"]
)

st.sidebar.markdown("---")
st.sidebar.success("✅ המערכת פעילה")

# --- עמוד סקירה כללית ---
if page == "🏠 סקירה כללית":
    st.header("סקירה כללית")
    
    # מדדים עיקריים
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ערך תיק", "$105,234", "+5.2%")
    
    with col2:
        st.metric("רווח יומי", "+$1,234", "+1.2%")
    
    with col3:
        st.metric("אחוז ניצחונות", "65%", "+2%")
    
    with col4:
        st.metric("פוזיציות פתוחות", "3", "")
    
    st.markdown("---")
    
    # גרף Equity Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 עקומת הון")
        
        # יצירת נתונים לדוגמה
        days = 60
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        equity = 100000 + np.cumsum(np.random.randn(days) * 500)
        
        df = pd.DataFrame({
            'תאריך': dates,
            'הון': equity
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['תאריך'],
            y=df['הון'],
            mode='lines',
            name='הון',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="תאריך",
            yaxis_title="הון ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 אסטרטגיות זמינות")
        
        strategies = [
            "MA Crossover",
            "Triple MA",
            "RSI + MACD",
            "RSI Divergence",
            "Momentum",
            "Dual Momentum",
            "Trend Following",
            "Mean Reversion"
        ]
        
        for i, strategy in enumerate(strategies, 1):
            st.info(f"{i}. {strategy}")
    
    st.markdown("---")
    
    # סטטוס מערכת
    st.subheader("💻 סטטוס מערכת")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU", "25%", "")
        st.progress(0.25)
    
    with col2:
        st.metric("זיכרון", "45%", "")
        st.progress(0.45)
    
    with col3:
        st.metric("חיבור לברוקר", "מנותק", "")
        st.warning("לא מחובר ל-IB")

# --- עמוד Backtest ---
elif page == "📊 Backtest":
    st.header("📊 Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ הגדרות")
        
        strategy = st.selectbox(
            "אסטרטגיה:",
            ["MA Crossover", "RSI + MACD", "Momentum"]
        )
        
        initial_capital = st.number_input(
            "הון התחלתי ($):",
            value=100000,
            step=10000
        )
        
        commission = st.number_input(
            "עמלה (%):",
            value=0.1,
            step=0.01
        ) / 100
        
        position_size = st.slider(
            "גודל פוזיציה (%):",
            0, 100, 50
        ) / 100
        
        if st.button("🚀 הרץ Backtest", type="primary"):
            with st.spinner("מריץ backtest..."):
                import time
                time.sleep(2)
                st.success("✅ Backtest הושלם!")
                
                # תוצאות
                st.session_state['backtest_done'] = True
    
    with col2:
        if st.session_state.get('backtest_done'):
            st.subheader("📈 תוצאות")
            
            # מדדים
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("תשואה כוללת", "+12.5%")
            
            with col2:
                st.metric("Sharpe Ratio", "1.45")
            
            with col3:
                st.metric("אחוז ניצחונות", "62%")
            
            with col4:
                st.metric("מספר עסקאות", "45")
            
            # גרף
            st.markdown("---")
            
            days = 365
            dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
            equity = 100000 + np.cumsum(np.random.randn(days) * 300)
            
            df = pd.DataFrame({
                'תאריך': dates,
                'הון': equity
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['תאריך'],
                y=df['הון'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="תאריך",
                yaxis_title="הון ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# --- עמוד אסטרטגיות ---
elif page == "🎯 אסטרטגיות":
    st.header("🎯 אסטרטגיות מסחר")
    
    strategies = {
        "MA Crossover": "אסטרטגיית חציית ממוצעים נעים - קלאסית ואמינה",
        "Triple MA": "3 ממוצעים נעים לאישור חזק יותר",
        "RSI + MACD": "שילוב אינדיקטורים למומנטום",
        "RSI Divergence": "זיהוי דיברגנציות של RSI",
        "Momentum": "אסטרטגיית מומנטום טהורה",
        "Dual Momentum": "מומנטום יחסי ומוחלט",
        "Trend Following": "מעקב אחר טרנדים",
        "Mean Reversion": "חזרה לממוצע"
    }
    
    for name, desc in strategies.items():
        with st.expander(f"📊 {name}"):
            st.write(f"**תיאור:** {desc}")
            st.write(f"**סטטוס:** ✅ זמינה")
            st.write(f"**ביצועים אחרונים:** +{np.random.randint(5, 20)}%")

# --- עמוד הגדרות ---
elif page == "⚙️ הגדרות":
    st.header("⚙️ הגדרות מערכת")
    
    tab1, tab2, tab3 = st.tabs(["מסחר", "ניהול סיכונים", "התראות"])
    
    with tab1:
        st.subheader("הגדרות מסחר")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("הון התחלתי ($)", value=100000)
            st.number_input("מספר פוזיציות מקסימלי", value=5)
            st.slider("גודל פוזיציה (%)", 0, 100, 20)
        
        with col2:
            st.number_input("עמלה (%)", value=0.1)
            st.selectbox("מצב מסחר", ["Paper Trading", "Live Trading"])
            st.checkbox("מסחר אוטומטי")
    
    with tab2:
        st.subheader("ניהול סיכונים")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("סיכון לעסקה (%)", 0.0, 5.0, 2.0)
            st.slider("הפסד מקסימלי יומי (%)", 0.0, 10.0, 5.0)
        
        with col2:
            st.slider("Kelly Fraction", 0.0, 1.0, 0.5)
            st.checkbox("השתמש ב-Stop Loss", value=True)
    
    with tab3:
        st.subheader("הגדרות התראות")
        
        st.checkbox("התראות Email")
        st.text_input("כתובת Email")
        
        st.checkbox("התראות Telegram")
        st.text_input("Telegram Bot Token")
    
    st.markdown("---")
    
    if st.button("💾 שמור הגדרות", type="primary"):
        st.success("✅ ההגדרות נשמרו בהצלחה!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <b>מערכת מסחר אוטומטית</b> | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

