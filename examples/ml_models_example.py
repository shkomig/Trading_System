"""
ML Models Example
Demonstrates how to use LSTM and DQN for trading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.ml_models.lstm_predictor import LSTMPredictor
from src.ml_models.dqn_agent import DQNAgent, TradingEnvironment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(days: int = 500) -> pd.DataFrame:
    """Generate sample price data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate trending prices with noise
    trend = np.linspace(100, 150, days)
    seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, days))
    noise = np.random.randn(days) * 5
    close_prices = trend + seasonal + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(days) * 0.5,
        'high': close_prices + np.abs(np.random.randn(days)) * 2,
        'low': close_prices - np.abs(np.random.randn(days)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    # Ensure high/low are correct
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def demo_lstm_predictor():
    """Demonstrate LSTM price prediction"""
    print("\n" + "="*70)
    print("LSTM PRICE PREDICTOR DEMO")
    print("="*70)
    
    # Generate data
    logger.info("Generating sample data...")
    data = generate_sample_data(days=500)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Create predictor
    predictor = LSTMPredictor(
        sequence_length=60,
        features=['close', 'volume'],
        hidden_size=50,
        num_layers=2
    )
    
    # Check if TensorFlow is available
    if predictor.model is None:
        logger.warning("TensorFlow not installed. Skipping LSTM demo.")
        print("\nInstall TensorFlow to use LSTM:")
        print("  pip install tensorflow")
        return
    
    # Train
    logger.info("Training LSTM model...")
    training_results = predictor.train(
        train_data,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    print(f"\nTraining Results:")
    print(f"  Final Loss: {training_results.get('final_loss', 0):.6f}")
    print(f"  Final Val Loss: {training_results.get('final_val_loss', 0):.6f}")
    print(f"  Epochs Trained: {training_results.get('epochs_trained', 0)}")
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = predictor.evaluate(test_data)
    
    print(f"\nEvaluation Metrics:")
    print(f"  RMSE: ${metrics.get('rmse', 0):.2f}")
    print(f"  MAE: ${metrics.get('mae', 0):.2f}")
    print(f"  MAPE: {metrics.get('mape', 0):.2f}%")
    print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.2f}%")
    
    # Predict next 5 days
    logger.info("Predicting next 5 days...")
    future_predictions = predictor.predict_next(test_data, steps=5)
    
    print(f"\nNext 5 Days Predictions:")
    current_price = test_data['close'].iloc[-1]
    print(f"  Current Price: ${current_price:.2f}")
    for i, pred in enumerate(future_predictions, 1):
        change = ((pred - current_price) / current_price) * 100
        print(f"  Day +{i}: ${pred:.2f} ({change:+.2f}%)")
    
    # Save model
    model_path = 'models/lstm_demo'
    predictor.save(model_path)
    logger.info(f"Model saved to {model_path}")


def demo_dqn_agent():
    """Demonstrate DQN reinforcement learning agent"""
    print("\n" + "="*70)
    print("DQN TRADING AGENT DEMO")
    print("="*70)
    
    # Generate data
    logger.info("Generating sample data...")
    data = generate_sample_data(days=300)
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_balance=10000,
        commission=0.001,
        lookback_window=10
    )
    
    # Create agent
    state_size = 10 + 3  # lookback_window + portfolio features
    agent = DQNAgent(
        state_size=state_size,
        action_size=3,  # hold, buy, sell
        memory_size=2000,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    # Check if TensorFlow is available
    if agent.model is None:
        logger.warning("TensorFlow not installed. Skipping DQN demo.")
        print("\nInstall TensorFlow to use DQN:")
        print("  pip install tensorflow")
        return
    
    # Train agent
    logger.info("Training DQN agent (this may take a few minutes)...")
    print("\nTraining Progress:")
    
    results = agent.train(
        env=env,
        episodes=50,
        max_steps=len(data) - 10,
        verbose=10
    )
    
    # Show training summary
    print(f"\nTraining Summary:")
    print(f"  Episodes: {len(results)}")
    print(f"  Avg Reward (last 10): {np.mean([r['total_reward'] for r in results[-10:]]):.4f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    
    # Evaluate agent
    logger.info("Evaluating agent...")
    eval_metrics = agent.evaluate(env, episodes=10)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Avg Reward: {eval_metrics['avg_reward']:.4f}")
    print(f"  Avg Portfolio Value: ${eval_metrics['avg_portfolio_value']:.2f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.2f}%")
    
    # Test single episode
    logger.info("Running test episode...")
    state = env.reset()
    total_reward = 0
    actions_taken = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    
    done = False
    while not done:
        action = agent.act(state, training=False)
        action_name = agent.get_action_name(action)
        actions_taken[action_name] += 1
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
    
    print(f"\nTest Episode Results:")
    print(f"  Initial Balance: $10,000.00")
    print(f"  Final Portfolio Value: ${info['portfolio_value']:.2f}")
    print(f"  Return: {(info['portfolio_value'] / 10000 - 1) * 100:.2f}%")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"\n  Actions Taken:")
    for action, count in actions_taken.items():
        print(f"    {action}: {count}")
    
    # Save agent
    model_path = 'models/dqn_demo'
    agent.save(model_path)
    logger.info(f"Agent saved to {model_path}")


def main():
    """Run all ML demos"""
    print("\n" + "="*70)
    print("  ML MODELS FOR TRADING - DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how to use:")
    print("  1. LSTM for price prediction")
    print("  2. DQN for reinforcement learning trading")
    print("\nNote: Requires TensorFlow installation")
    print("="*70)
    
    # Demo LSTM
    try:
        demo_lstm_predictor()
    except Exception as e:
        logger.error(f"LSTM demo failed: {e}")
    
    # Demo DQN
    try:
        demo_dqn_agent()
    except Exception as e:
        logger.error(f"DQN demo failed: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  - Integrate ML models with your trading strategies")
    print("  - Train on real market data")
    print("  - Combine LSTM predictions with DQN decisions")
    print("  - Use ensemble methods for better predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

