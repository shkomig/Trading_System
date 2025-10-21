"""
LSTM Price Predictor
Long Short-Term Memory neural network for price prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM-based price predictor for time series forecasting
    
    This class provides a complete LSTM implementation for predicting
    future prices based on historical data. Supports training, prediction,
    and model persistence.
    
    Attributes:
        sequence_length: Number of time steps to use for prediction
        features: List of feature names to use
        model: The LSTM model (TensorFlow/PyTorch)
        scaler: Data scaler for normalization
        is_trained: Whether the model has been trained
        
    Example:
        >>> predictor = LSTMPredictor(sequence_length=60, features=['close', 'volume'])
        >>> predictor.train(train_data, epochs=50)
        >>> predictions = predictor.predict(test_data)
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: Optional[list] = None,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM Predictor
        
        Args:
            sequence_length: Number of time steps for each sequence
            features: List of feature names to use (default: ['close'])
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.features = features or ['close']
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
        self._setup_model()
        
    def _setup_model(self):
        """Initialize the LSTM model architecture"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler
            
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            self.model = Sequential([
                LSTM(
                    self.hidden_size,
                    return_sequences=True if self.num_layers > 1 else False,
                    input_shape=(self.sequence_length, len(self.features))
                ),
                Dropout(self.dropout),
                *[layer for _ in range(self.num_layers - 1) 
                  for layer in [
                      LSTM(self.hidden_size, return_sequences=(_ < self.num_layers - 2)),
                      Dropout(self.dropout)
                  ]],
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            logger.info(f"LSTM model initialized with {self.num_layers} layers")
            
        except ImportError:
            logger.warning("TensorFlow not installed. Using dummy model.")
            self.model = None
            
    def _prepare_data(
        self,
        data: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training/prediction
        
        Args:
            data: DataFrame with OHLCV data
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (X, y) arrays ready for LSTM
        """
        # Extract features
        feature_data = data[self.features].values
        
        # Scale data
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(feature_data)
        else:
            scaled_data = self.scaler.transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict first feature (usually 'close')
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            data: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Dictionary with training history and metrics
        """
        if self.model is None:
            logger.error("Model not initialized. Cannot train.")
            return {}
        
        logger.info(f"Training LSTM on {len(data)} samples...")
        
        # Prepare data
        X, y = self._prepare_data(data, fit_scaler=True)
        
        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
        
        # Setup callbacks
        callbacks = []
        if early_stopping:
            try:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
                callbacks.append(early_stop)
            except ImportError:
                pass
        
        # Train model
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            logger.info("Training completed successfully")
            
            return {
                'history': history.history,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}
    
    def predict(
        self,
        data: pd.DataFrame,
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            data: DataFrame with features
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Array of predictions (and optionally confidence intervals)
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Returning zeros.")
            return np.zeros(len(data) - self.sequence_length)
        
        # Prepare data
        X, _ = self._prepare_data(data, fit_scaler=False)
        
        # Make predictions
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.concatenate([predictions_scaled, np.zeros((len(predictions_scaled), len(self.features)-1))], axis=1)
        )[:, 0]
        
        return predictions
    
    def predict_next(
        self,
        recent_data: pd.DataFrame,
        steps: int = 1
    ) -> np.ndarray:
        """
        Predict next N time steps
        
        Args:
            recent_data: Recent data (at least sequence_length rows)
            steps: Number of steps ahead to predict
            
        Returns:
            Array of future predictions
        """
        if not self.is_trained:
            logger.warning("Model not trained. Returning zeros.")
            return np.zeros(steps)
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        predictions = []
        current_sequence = recent_data.tail(self.sequence_length)[self.features].values
        current_sequence = self.scaler.transform(current_sequence)
        
        for _ in range(steps):
            # Predict next value
            X = current_sequence.reshape(1, self.sequence_length, len(self.features))
            next_pred = self.model.predict(X, verbose=0)[0]
            
            predictions.append(next_pred)
            
            # Update sequence
            new_row = np.zeros(len(self.features))
            new_row[0] = next_pred
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((len(predictions), len(self.features)-1))], axis=1)
        )[:, 0]
        
        return predictions
    
    def evaluate(
        self,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        X, y_true = self._prepare_data(test_data, fit_scaler=False)
        
        # Get predictions
        y_pred_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform
        y_true_original = self.scaler.inverse_transform(
            np.concatenate([y_true.reshape(-1, 1), np.zeros((len(y_true), len(self.features)-1))], axis=1)
        )[:, 0]
        
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([y_pred_scaled, np.zeros((len(y_pred_scaled), len(self.features)-1))], axis=1)
        )[:, 0]
        
        # Calculate metrics
        mse = np.mean((y_true_original - y_pred_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_original - y_pred_original))
        mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
        
        # Direction accuracy
        direction_true = np.diff(y_true_original) > 0
        direction_pred = np.diff(y_pred_original) > 0
        direction_accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Directory path to save model
        """
        if not self.is_trained:
            logger.warning("Model not trained. Nothing to save.")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path / 'lstm_model.h5')
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, path / 'scaler.pkl')
        
        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'features': self.features,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Directory path containing saved model
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.sequence_length = config['sequence_length']
        self.features = config['features']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        self.is_trained = config['is_trained']
        
        # Load model
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path / 'lstm_model.h5')
        except ImportError:
            logger.error("TensorFlow not installed. Cannot load model.")
            return
        
        # Load scaler
        import joblib
        self.scaler = joblib.load(path / 'scaler.pkl')
        
        logger.info(f"Model loaded from {path}")

