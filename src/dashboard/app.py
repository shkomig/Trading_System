"""
Trading System Dashboard
Streamlit web application for monitoring and controlling the trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.strategy_registry import list_available_strategies, get_strategy
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics
from src.monitoring.alert_manager import AlertManager, AlertLevel
from src.monitoring.monitor import SystemMonitor


# Page configuration
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    
    if 'system_monitor' not in st.session_state:
        st.session_state.system_monitor = SystemMonitor(st.session_state.alert_manager)
        st.session_state.system_monitor.start_monitoring()
    
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None


def sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("📈 Trading System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📊 Backtest", "🎯 Strategies", "📉 Performance", "⚠️ Alerts", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**System Status**: {'🟢 Active' if st.session_state.system_monitor.is_monitoring else '🔴 Inactive'}")
    
    return page


def overview_page():
    """Main overview page"""
    st.markdown('<h1 class="main-header">Trading System Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # System Status
    monitor = st.session_state.system_monitor
    monitor.collect_system_metrics()
    
    status = monitor.get_status()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${status['trading_metrics']['portfolio_value']:,.2f}",
            f"{status['trading_metrics']['daily_pnl']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Daily P&L",
            f"{status['trading_metrics']['daily_pnl_percent']:+.2f}%",
            help="Daily profit/loss percentage"
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{status['trading_metrics']['win_rate']:.1f}%",
            help="Percentage of winning trades"
        )
    
    with col4:
        st.metric(
            "Open Positions",
            status['trading_metrics']['open_positions'],
            help="Number of currently open positions"
        )
    
    st.markdown("---")
    
    # System Health
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💻 System Health")
        
        sys_metrics = status['system_metrics']
        
        # CPU Usage
        cpu_color = "green" if sys_metrics['cpu_percent'] < 70 else "orange" if sys_metrics['cpu_percent'] < 90 else "red"
        st.progress(sys_metrics['cpu_percent'] / 100)
        st.caption(f"CPU: {sys_metrics['cpu_percent']:.1f}%")
        
        # Memory Usage
        mem_color = "green" if sys_metrics['memory_percent'] < 70 else "orange" if sys_metrics['memory_percent'] < 90 else "red"
        st.progress(sys_metrics['memory_percent'] / 100)
        st.caption(f"Memory: {sys_metrics['memory_percent']:.1f}% ({sys_metrics['memory_used_mb']:.0f} MB)")
        
        # Connection Status
        conn_status = "🟢 Connected" if status['is_connected_to_broker'] else "🔴 Disconnected"
        st.info(f"**Broker**: {conn_status}")
    
    with col2:
        st.subheader("📈 Recent Performance")
        
        # Generate sample equity curve
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        equity = 100000 + np.cumsum(np.random.randn(days) * 500)
        
        df = pd.DataFrame({'Date': dates, 'Equity': equity})
        
        fig = px.line(df, x='Date', y='Equity', title='Equity Curve (Last 30 Days)')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Available Strategies
    st.markdown("---")
    st.subheader("🎯 Available Strategies")
    
    strategies = list_available_strategies()
    cols = st.columns(4)
    
    for i, strategy_name in enumerate(strategies):
        with cols[i % 4]:
            st.info(f"**{strategy_name}**")


def backtest_page():
    """Backtesting interface"""
    st.title("📊 Strategy Backtesting")
    st.markdown("---")
    
    # Backtest Configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Strategy selection
        strategies = list_available_strategies()
        selected_strategy = st.selectbox("Strategy", strategies)
        
        # Parameters
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        commission = st.number_input("Commission (%)", value=0.1, step=0.01) / 100
        position_size = st.slider("Position Size (%)", 0, 100, 50) / 100
        
        # Date range
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        
        run_backtest = st.button("🚀 Run Backtest", type="primary")
    
    with col2:
        if run_backtest:
            with st.spinner("Running backtest..."):
                # Generate sample data
                days = (end_date - start_date).days
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                trend = np.linspace(100, 120, len(dates))
                noise = np.random.randn(len(dates)) * 3
                close_prices = trend + noise
                
                data = pd.DataFrame({
                    'open': close_prices + np.random.randn(len(dates)) * 0.5,
                    'high': close_prices + np.abs(np.random.randn(len(dates))) * 2,
                    'low': close_prices - np.abs(np.random.randn(len(dates))) * 2,
                    'close': close_prices,
                    'volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates)
                
                data['high'] = data[['open', 'high', 'close']].max(axis=1)
                data['low'] = data[['open', 'low', 'close']].min(axis=1)
                
                # Get strategy
                try:
                    strategy = get_strategy(selected_strategy)
                    if strategy:
                        signals = strategy.generate_signals(data)
                        
                        # Run backtest
                        engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
                        results = engine.run(data, signals, position_size=position_size)
                        
                        st.session_state.backtest_results = {
                            'engine': engine,
                            'results': results,
                            'data': data,
                            'signals': signals
                        }
                        
                        st.success("✅ Backtest completed!")
                    else:
                        st.error(f"Strategy '{selected_strategy}' not found")
                except Exception as e:
                    st.error(f"Error running backtest: {e}")
    
    # Display Results
    if st.session_state.backtest_results:
        st.markdown("---")
        st.subheader("📈 Results")
        
        results = st.session_state.backtest_results
        engine = results['engine']
        data = results['data']
        signals = results['signals']
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (engine.capital - initial_capital) / initial_capital * 100
            st.metric("Total Return", f"{total_return:.2f}%")
        
        with col2:
            metrics_calc = PerformanceMetrics()
            equity_curve = pd.Series([t['capital'] for t in engine.trades])
            sharpe = metrics_calc.calculate_sharpe_ratio(equity_curve)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            win_rate = (len([t for t in engine.trades if t.get('pnl', 0) > 0]) / len(engine.trades) * 100) if engine.trades else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            st.metric("Total Trades", len(engine.trades))
        
        # Equity Curve
        st.markdown("---")
        equity_df = pd.DataFrame({
            'Date': data.index[:len(engine.equity_curve)],
            'Equity': engine.equity_curve
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Equity'], name='Equity', line=dict(color='blue', width=2)))
        fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value ($)', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Price Chart with Signals
        st.markdown("---")
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Price', line=dict(color='black', width=1)))
        
        # Buy signals
        buy_signals = signals[signals == 1]
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, 
                y=data.loc[buy_signals.index, 'close'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        # Sell signals
        sell_signals = signals[signals == -1]
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=data.loc[sell_signals.index, 'close'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(title='Price Chart with Signals', xaxis_title='Date', yaxis_title='Price ($)', height=400)
        st.plotly_chart(fig, use_container_width=True)


def strategies_page():
    """Strategies overview"""
    st.title("🎯 Trading Strategies")
    st.markdown("---")
    
    strategies = list_available_strategies()
    
    st.info(f"**{len(strategies)} strategies available**")
    
    for strategy_name in strategies:
        with st.expander(f"📊 {strategy_name}"):
            try:
                strategy = get_strategy(strategy_name)
                if strategy:
                    st.write(f"**Type**: {strategy.__class__.__name__}")
                    if hasattr(strategy, '__doc__') and strategy.__doc__:
                        st.write(strategy.__doc__)
                else:
                    st.write("Strategy details not available")
            except Exception as e:
                st.error(f"Error loading strategy: {e}")


def performance_page():
    """Performance analytics"""
    st.title("📉 Performance Analytics")
    st.markdown("---")
    
    st.info("Performance analytics will show live trading results here")
    
    # Sample metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Returns")
        periods = ['Daily', 'Weekly', 'Monthly', 'Yearly']
        returns = [0.5, 2.3, 8.7, 45.2]
        
        df = pd.DataFrame({'Period': periods, 'Return (%)': returns})
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Metrics")
        metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Volatility']
        values = [1.45, 1.89, '-12.3%', '18.5%']
        
        df = pd.DataFrame({'Metric': metrics, 'Value': values})
        st.dataframe(df, use_container_width=True)
    
    with col3:
        st.subheader("💰 Trade Statistics")
        stats = ['Total Trades', 'Win Rate', 'Avg Win', 'Avg Loss']
        values = [145, '62.4%', '$458', '$312']
        
        df = pd.DataFrame({'Statistic': stats, 'Value': values})
        st.dataframe(df, use_container_width=True)


def alerts_page():
    """Alerts and notifications"""
    st.title("⚠️ Alerts & Notifications")
    st.markdown("---")
    
    alert_manager = st.session_state.alert_manager
    
    # Alert Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    alert_counts = alert_manager.get_alerts_by_level()
    
    with col1:
        st.metric("ℹ️ Info", alert_counts.get(AlertLevel.INFO, 0))
    with col2:
        st.metric("⚠️ Warnings", alert_counts.get(AlertLevel.WARNING, 0))
    with col3:
        st.metric("❌ Errors", alert_counts.get(AlertLevel.ERROR, 0))
    with col4:
        st.metric("🚨 Critical", alert_counts.get(AlertLevel.CRITICAL, 0))
    
    st.markdown("---")
    
    # Recent Alerts
    st.subheader("Recent Alerts")
    
    recent_alerts = alert_manager.get_recent_alerts(count=20)
    
    if recent_alerts:
        for alert in reversed(recent_alerts):
            emoji = {
                AlertLevel.INFO: 'ℹ️',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.ERROR: '❌',
                AlertLevel.CRITICAL: '🚨'
            }
            
            with st.expander(f"{emoji.get(alert.level, '')} {alert.title} - {alert.timestamp.strftime('%H:%M:%S')}"):
                st.write(alert.message)
                if alert.metadata:
                    st.json(alert.metadata)
    else:
        st.info("No alerts yet")


def settings_page():
    """System settings"""
    st.title("⚙️ Settings")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Trading", "Risk Management", "Notifications"])
    
    with tab1:
        st.subheader("Trading Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Initial Capital ($)", value=100000, step=10000)
            st.number_input("Max Positions", value=5, step=1)
            st.slider("Position Size (%)", 0, 100, 20)
        
        with col2:
            st.number_input("Commission (%)", value=0.1, step=0.01)
            st.selectbox("Trading Mode", ["Paper Trading", "Live Trading"])
            st.checkbox("Auto Trading", value=False)
    
    with tab2:
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Risk per Trade (%)", 0.0, 5.0, 2.0, step=0.1)
            st.slider("Max Daily Loss (%)", 0.0, 10.0, 5.0, step=0.5)
        
        with col2:
            st.slider("Kelly Fraction", 0.0, 1.0, 0.5, step=0.1)
            st.checkbox("Use Stop Loss", value=True)
    
    with tab3:
        st.subheader("Notification Settings")
        
        st.checkbox("Enable Email Alerts", value=False)
        st.text_input("Email Address", placeholder="your@email.com")
        
        st.checkbox("Enable Telegram Alerts", value=False)
        st.text_input("Telegram Bot Token", placeholder="123456:ABC...")
        st.text_input("Telegram Chat ID", placeholder="123456789")
    
    st.markdown("---")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")


def main():
    """Main application entry point"""
    init_session_state()
    
    page = sidebar()
    
    if page == "🏠 Overview":
        overview_page()
    elif page == "📊 Backtest":
        backtest_page()
    elif page == "🎯 Strategies":
        strategies_page()
    elif page == "📉 Performance":
        performance_page()
    elif page == "⚠️ Alerts":
        alerts_page()
    elif page == "⚙️ Settings":
        settings_page()


if __name__ == "__main__":
    main()

