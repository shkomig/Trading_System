"""
DQN Trading Agent
Deep Q-Network for reinforcement learning based trading
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import logging
from pathlib import Path
import json
import random

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network Agent for algorithmic trading
    
    Uses reinforcement learning to learn optimal trading actions
    (buy, sell, hold) based on market state.
    
    Attributes:
        state_size: Dimension of the state space
        action_size: Number of possible actions (3: buy, sell, hold)
        memory: Replay buffer for experience replay
        gamma: Discount factor for future rewards
        epsilon: Exploration rate
        model: Neural network for Q-value approximation
        
    Example:
        >>> agent = DQNAgent(state_size=10, action_size=3)
        >>> for episode in range(1000):
        ...     state = env.reset()
        ...     action = agent.act(state)
        ...     next_state, reward, done = env.step(action)
        ...     agent.remember(state, action, reward, next_state, done)
        ...     agent.replay()
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        memory_size: int = 2000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state representation
            action_size: Number of actions (default 3: buy, sell, hold)
            memory_size: Size of replay buffer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model = None
        self.target_model = None
        self.training_history = []
        
        self._build_model()
        
    def _build_model(self):
        """Build the neural network for Q-value approximation"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Main model
            self.model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            
            self.model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )
            
            # Target model (for stable training)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            
            logger.info("DQN model initialized successfully")
            
        except ImportError:
            logger.warning("TensorFlow not installed. Using dummy model.")
            self.model = None
            self.target_model = None
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Action index (0=hold, 1=buy, 2=sell)
        """
        if self.model is None:
            return random.randrange(self.action_size)
        
        # Exploration
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        state = state.reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> Optional[float]:
        """
        Train on batch from replay memory
        
        Returns:
            Average loss from training batch
        """
        if len(self.memory) < self.batch_size:
            return None
        
        if self.model is None:
            return None
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Compute Q targets
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(
            states,
            targets,
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def update_target_model(self):
        """Update target model with weights from main model"""
        if self.model is not None and self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())
            logger.debug("Target model updated")
    
    def train_episode(
        self,
        env,
        max_steps: int = 1000
    ) -> Dict[str, Any]:
        """
        Train agent for one episode
        
        Args:
            env: Trading environment
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with episode results
        """
        state = env.reset()
        total_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Choose action
            action = self.act(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Remember
            self.remember(state, action, reward, next_state, done)
            
            # Train
            loss = self.replay()
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update target model periodically
        if len(self.training_history) % 10 == 0:
            self.update_target_model()
        
        episode_result = {
            'total_reward': total_reward,
            'steps': step + 1,
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.epsilon,
            'final_portfolio_value': info.get('portfolio_value', 0)
        }
        
        self.training_history.append(episode_result)
        
        return episode_result
    
    def train(
        self,
        env,
        episodes: int = 100,
        max_steps: int = 1000,
        verbose: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Train agent for multiple episodes
        
        Args:
            env: Trading environment
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            verbose: Print progress every N episodes
            
        Returns:
            List of episode results
        """
        logger.info(f"Starting training for {episodes} episodes...")
        
        results = []
        
        for episode in range(episodes):
            result = self.train_episode(env, max_steps)
            results.append(result)
            
            if verbose > 0 and (episode + 1) % verbose == 0:
                avg_reward = np.mean([r['total_reward'] for r in results[-verbose:]])
                logger.info(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )
        
        logger.info("Training completed")
        return results
    
    def evaluate(
        self,
        env,
        episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            env: Trading environment
            episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating agent for {episodes} episodes...")
        
        rewards = []
        portfolio_values = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
            
            rewards.append(total_reward)
            portfolio_values.append(info.get('portfolio_value', 0))
        
        metrics = {
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'avg_portfolio_value': float(np.mean(portfolio_values)),
            'success_rate': float(np.mean([r > 0 for r in rewards]) * 100)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, path: str):
        """
        Save agent to disk
        
        Args:
            path: Directory path to save agent
        """
        if self.model is None:
            logger.warning("Model not initialized. Nothing to save.")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.model.save(path / 'dqn_model.h5')
        self.target_model.save(path / 'dqn_target_model.h5')
        
        # Save config
        config = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'training_episodes': len(self.training_history)
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(path / 'training_history.csv', index=False)
        
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent from disk
        
        Args:
            path: Directory path containing saved agent
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Agent path not found: {path}")
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        
        # Load models
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path / 'dqn_model.h5')
            self.target_model = load_model(path / 'dqn_target_model.h5')
        except ImportError:
            logger.error("TensorFlow not installed. Cannot load models.")
            return
        
        # Load training history
        history_file = path / 'training_history.csv'
        if history_file.exists():
            history_df = pd.read_csv(history_file)
            self.training_history = history_df.to_dict('records')
        
        logger.info(f"Agent loaded from {path}")
    
    def get_action_name(self, action: int) -> str:
        """
        Convert action index to name
        
        Args:
            action: Action index
            
        Returns:
            Action name
        """
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_names.get(action, 'UNKNOWN')


class TradingEnvironment:
    """
    Simple trading environment for DQN training
    
    This environment simulates a trading scenario where an agent
    can buy, sell, or hold a position based on market data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        commission: float = 0.001,
        lookback_window: int = 10
    ):
        """
        Initialize trading environment
        
        Args:
            data: Market data DataFrame (must have 'close' column)
            initial_balance: Starting cash balance
            commission: Trading commission rate
            lookback_window: Number of historical steps in state
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window
        
        self.current_step = lookback_window
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Get historical prices
        prices = self.data['close'].iloc[
            self.current_step - self.lookback_window:self.current_step
        ].values
        
        # Normalize prices
        prices_normalized = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Add portfolio information
        current_price = self.data['close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        portfolio_features = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            portfolio_value / self.initial_balance
        ])
        
        state = np.concatenate([prices_normalized, portfolio_features])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        current_price = self.data['close'].iloc[self.current_step]
        
        # Execute action
        if action == 1:  # Buy
            shares_to_buy = self.balance // (current_price * (1 + self.commission))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.commission)
                self.balance -= cost
                self.shares_held += shares_to_buy
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.commission)
                self.balance += revenue
                self.total_shares_sold += self.shares_held
                self.total_sales_value += revenue
                self.shares_held = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        portfolio_value = self.balance + self.shares_held * current_price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.lookback_window + 3)
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price
        }
        
        return next_state, reward, done, info

