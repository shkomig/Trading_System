"""
Simple Trading System Demo

This is a beginner-friendly demo that shows how to:
1. Connect to Interactive Brokers Paper Trading
2. Run a simple strategy
3. Execute trades automatically (in dry-run mode)

IMPORTANT: This demo uses DRY_RUN mode - no real orders will be placed!

Requirements:
- IB TWS or Gateway running
- API enabled (Configure -> API -> Settings)
- Paper Trading account (port 7497)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.broker.ib_connector import IBConnector
from src.strategies.technical.moving_average import MovingAverageCrossover
from src.execution.order_executor import OrderExecutor
from src.execution.position_manager import PositionManager
from src.risk_management.position_sizing import PositionSizer

def main():
    """Run simple trading demo"""

    print("=" * 70)
    print("   SIMPLE TRADING SYSTEM DEMO")
    print("=" * 70)
    print()
    print("This demo will:")
    print("  1. Connect to IB Paper Trading")
    print("  2. Fetch historical data for AAPL")
    print("  3. Run Moving Average Crossover strategy")
    print("  4. Show you how signals would be executed")
    print()
    print("⚠️  DRY RUN MODE - No real orders will be placed!")
    print("=" * 70)
    print()

    # =============================================
    # STEP 1: Connect to Interactive Brokers
    # =============================================
    print("[STEP 1] Connecting to Interactive Brokers...")
    print("          (Make sure TWS/Gateway is running!)")

    broker = IBConnector(
        host='127.0.0.1',
        port=7497,  # Paper Trading port
        is_paper=True
    )

    if not broker.connect():
        print()
        print("❌ Connection FAILED!")
        print()
        print("Troubleshooting:")
        print("  1. Is TWS or IB Gateway running?")
        print("  2. Is API enabled? (Configure -> API -> Settings)")
        print("  3. Is Socket Port 7497 configured?")
        print("  4. Is 'localhost' allowed in Trusted IPs?")
        print()
        return

    print("✅ Connected successfully!")
    print()

    # Show account info
    account_info = broker.get_account_info()
    if account_info:
        buying_power = account_info.get('BuyingPower', 'N/A')
        print(f"   Account Buying Power: ${buying_power}")

    print()

    # =============================================
    # STEP 2: Get Historical Data
    # =============================================
    print("[STEP 2] Fetching historical data for AAPL...")

    symbol = 'AAPL'
    data = broker.get_historical_data(
        symbol=symbol,
        duration='6 M',  # 6 months
        bar_size='1 day'
    )

    if data is None or data.empty:
        print(f"❌ Failed to get data for {symbol}")
        broker.disconnect()
        return

    print(f"✅ Retrieved {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Latest close: ${data['close'].iloc[-1]:.2f}")
    print()

    # =============================================
    # STEP 3: Create and Run Strategy
    # =============================================
    print("[STEP 3] Running Moving Average Crossover strategy...")
    print("          (Short MA: 20 days, Long MA: 50 days)")

    strategy = MovingAverageCrossover(
        short_window=20,
        long_window=50
    )

    # Calculate indicators
    indicators = strategy.calculate_indicators(data)

    # Generate signals
    signals = strategy.generate_signals(data)

    if signals is None or signals.empty:
        print("❌ Failed to generate signals")
        broker.disconnect()
        return

    # Get latest signal
    latest_signal = int(signals.iloc[-1])
    current_price = data['close'].iloc[-1]

    signal_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
    signal_name = signal_names.get(latest_signal, 'UNKNOWN')

    print(f"✅ Strategy signal: {signal_name}")
    print()

    # Show indicator values
    if 'short_ma' in indicators.columns:
        short_ma = indicators['short_ma'].iloc[-1]
        long_ma = indicators['long_ma'].iloc[-1]
        print(f"   Indicators:")
        print(f"     Short MA (20): ${short_ma:.2f}")
        print(f"     Long MA (50):  ${long_ma:.2f}")
        print(f"     Current Price: ${current_price:.2f}")
        print()

    # =============================================
    # STEP 4: Setup Execution Components
    # =============================================
    print("[STEP 4] Setting up execution components...")

    # Position Manager
    position_manager = PositionManager(
        broker=broker,
        max_positions=5,
        enable_trailing_stops=True,
        trailing_stop_pct=0.05  # 5% trailing stop
    )

    # Risk Manager
    risk_manager = PositionSizer(account_value=100000)

    # Order Executor (DRY RUN mode!)
    executor = OrderExecutor(
        broker=broker,
        risk_manager=risk_manager,
        max_position_value=10000.0,  # Max $10k per position
        max_positions=5,
        enable_stop_loss=True,
        stop_loss_pct=0.05,  # 5% stop-loss
        dry_run=True  # IMPORTANT: Dry run mode!
    )

    print("✅ Execution components ready")
    print("   ⚠️  DRY RUN MODE - Orders will be simulated")
    print()

    # =============================================
    # STEP 5: Execute Signal
    # =============================================
    print("[STEP 5] Executing signal...")

    if latest_signal == 0:
        print("   Signal is HOLD - no action needed")
    else:
        result = executor.execute_signal(
            symbol=symbol,
            signal=latest_signal,
            current_price=current_price,
            strategy_name=strategy.name,
            position_manager=position_manager
        )

        print()
        print("   Execution Result:")
        print("   " + "-" * 50)

        if result.success:
            print(f"   ✅ Order executed successfully!")
            print(f"      Action: {result.action}")
            print(f"      Symbol: {result.symbol}")
            print(f"      Quantity: {result.executed_quantity} shares")
            print(f"      Price: ${result.avg_fill_price:.2f}")
            print(f"      Total Value: ${result.executed_quantity * result.avg_fill_price:,.2f}")
            print(f"      Order ID: {result.order_id}")
            print()

            if result.action == 'BUY':
                stop_loss = result.avg_fill_price * 0.95
                print(f"      Stop-Loss would be set at: ${stop_loss:.2f} (-5%)")
        else:
            print(f"   ❌ Order failed: {result.error_message}")

    print()

    # =============================================
    # STEP 6: Summary
    # =============================================
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("What you learned:")
    print("  ✅ How to connect to IB Paper Trading")
    print("  ✅ How to fetch historical data")
    print("  ✅ How to run a trading strategy")
    print("  ✅ How signals are generated")
    print("  ✅ How orders would be executed")
    print()
    print("Next steps:")
    print("  1. Review the code in demo_simple.py")
    print("  2. Try modifying strategy parameters")
    print("  3. Test with different symbols")
    print("  4. When ready, run: python production_trader.py")
    print()
    print("⚠️  Remember: Always test thoroughly with Paper Trading")
    print("    before attempting live trading!")
    print()

    # Cleanup
    broker.disconnect()
    print("✅ Disconnected from IB")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
