import os
import time
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from TradingEnvironment import TradingEnvironment

# Set up logging
import os
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "rl_training.log"))
    ]
)
logger = logging.getLogger("RLTraining")


class TradingTrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress and adapting hyperparameters"""
    
    def __init__(self, check_freq=1000, save_path=None, verbose=1, training_callback=None, 
                 enable_auto_tuning=True, patience=5, min_reward_improvement=0.05,
                 stop_flag_callback=None):
        super(TradingTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        self.training_callback = training_callback
        self.episode_count = 0
        
        # External stop flag callback
        self.stop_flag_callback = stop_flag_callback
        
        # Auto-tuning parameters
        self.enable_auto_tuning = enable_auto_tuning
        self.patience = patience  # How many checks to wait before tuning
        self.min_reward_improvement = min_reward_improvement  # Minimum improvement to reset patience
        self.stagnation_counter = 0
        self.last_mean_reward = -np.inf
        self.tuning_attempts = 0
        self.max_tuning_attempts = 5
        
        # Hyperparameter ranges
        self.lr_range = (0.00001, 0.001)
        self.batch_size_range = (32, 256)
        self.ent_coef_range = (0.0, 0.1)
        
        # Progress tracking
        self.mean_rewards_history = []
        self.timestamps = []
        self.start_time = time.time()
        
        logger.info("Auto-tuning of hyperparameters is enabled" if enable_auto_tuning else "Auto-tuning of hyperparameters is disabled")
    
    def _on_step(self) -> bool:
        # Comprobar si se debe detener el entrenamiento primero
        if self.stop_flag_callback and self.stop_flag_callback():
            logger.info("Stop flag detected. Training will be interrupted.")
            return False  # Esto detiene el entrenamiento
            
        if self.n_calls % self.check_freq == 0:
            # Get current training stats
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0
            
            # Record progress for analysis
            self.mean_rewards_history.append(mean_reward)
            self.timestamps.append(time.time() - self.start_time)
            
            logger.info(f"Training step: {self.n_calls}")
            logger.info(f"Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
            
            # Auto-tune hyperparameters if enabled and needed
            if self.enable_auto_tuning:
                self._check_and_tune_hyperparameters(mean_reward)
            
            # Track best reward (but don't save intermediate models)
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                logger.info(f"New best reward: {mean_reward:.2f}")
            
            # Call the training callback if provided
            if self.training_callback is not None:
                # Extract the most recent episode info if available
                episode_reward = 0
                episode_length = 0
                
                if len(self.model.ep_info_buffer) > 0:
                    # Increment episode count for each completed episode
                    old_ep_count = self.episode_count
                    self.episode_count = len(self.model.ep_info_buffer)
                    
                    # Get the latest episode info only if there's a new one
                    if self.episode_count > old_ep_count:
                        episode_reward = self.model.ep_info_buffer[-1]["r"]
                        episode_length = self.model.ep_info_buffer[-1]["l"]
                
                # Get win rate from the environment if available
                win_rate = 0.0
                if hasattr(self.model, 'get_env'):
                    env = self.model.get_env()
                    if hasattr(env, 'venv') and hasattr(env.venv, 'envs') and len(env.venv.envs) > 0:
                        if hasattr(env.venv.envs[0].unwrapped, 'successful_trades') and hasattr(env.venv.envs[0].unwrapped, 'trade_count'):
                            if env.venv.envs[0].unwrapped.trade_count > 0:
                                win_rate = env.venv.envs[0].unwrapped.successful_trades / env.venv.envs[0].unwrapped.trade_count
                
                # Calculate training progress as percentage
                total_timesteps = self.model.num_timesteps
                target_timesteps = self.model._total_timesteps
                progress_percent = min(int((total_timesteps / target_timesteps) * 100), 100)
                
                # Create metrics dict
                metrics = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'mean_reward': mean_reward,
                    'win_rate': win_rate,
                    'episode': self.episode_count,
                    'hyperparams': self._get_current_hyperparams(),
                    'progress': progress_percent
                }
                
                # Call the callback with metrics
                self.training_callback(metrics)
        
        return True
        
    def _check_and_tune_hyperparameters(self, current_mean_reward):
        """Check if learning has stagnated and tune hyperparameters if needed"""
        # Check if reward is improving
        reward_improvement = current_mean_reward - self.last_mean_reward
        
        # Update last mean reward
        self.last_mean_reward = current_mean_reward
        
        # Check for stagnation
        if reward_improvement < self.min_reward_improvement:
            self.stagnation_counter += 1
            logger.info(f"Learning might be stagnating: {self.stagnation_counter}/{self.patience} checks with minimal improvement")
        else:
            # Reset counter if we see good improvement
            self.stagnation_counter = 0
            logger.info(f"Learning is progressing well, reward improved by {reward_improvement:.4f}")
        
        # Tune hyperparameters if stagnation exceeds patience threshold
        if self.stagnation_counter >= self.patience:
            if self.tuning_attempts < self.max_tuning_attempts:
                logger.info(f"Detected learning stagnation, tuning hyperparameters (attempt {self.tuning_attempts + 1}/{self.max_tuning_attempts})")
                self._tune_hyperparameters()
                self.stagnation_counter = 0
                self.tuning_attempts += 1
            else:
                logger.info(f"Reached maximum tuning attempts ({self.max_tuning_attempts}), continuing with current parameters")
    
    def _tune_hyperparameters(self):
        """Adapt hyperparameters based on training progress"""
        # Get current parameters
        params = self._get_current_hyperparams()
        
        # Analyze reward history to determine adaptations
        if len(self.mean_rewards_history) >= 3:
            reward_trend = np.polyfit(range(len(self.mean_rewards_history[-3:])), 
                                       self.mean_rewards_history[-3:], 1)[0]
            
            # Strategy 1: If rewards are consistently decreasing, reduce learning rate
            if reward_trend < 0:
                new_lr = max(params['learning_rate'] * 0.5, self.lr_range[0])
                logger.info(f"Decreasing learning rate: {params['learning_rate']:.6f} -> {new_lr:.6f}")
                self.model.learning_rate = new_lr
            
            # Strategy 2: If rewards are flat, try increasing batch size
            elif abs(reward_trend) < 0.001:
                # Increase batch size to stabilize learning (must be power of 2)
                current_batch = params.get('batch_size', 64)
                new_batch = min(current_batch * 2, self.batch_size_range[1])
                
                if hasattr(self.model, 'batch_size') and new_batch != current_batch:
                    logger.info(f"Increasing batch size: {current_batch} -> {new_batch}")
                    self.model.batch_size = new_batch
                
                # Add entropy to encourage exploration
                if hasattr(self.model, 'ent_coef'):
                    current_ent = params.get('ent_coef', 0.0)
                    new_ent = min(current_ent + 0.01, self.ent_coef_range[1])
                    if new_ent != current_ent:
                        logger.info(f"Increasing entropy coefficient: {current_ent:.3f} -> {new_ent:.3f}")
                        self.model.ent_coef = new_ent
            
            # Strategy 3: If rewards are slowly improving, subtle adjustments
            else:
                # Fine tuning - slight increase in learning rate if progress is slow but positive
                new_lr = min(params['learning_rate'] * 1.2, self.lr_range[1])
                logger.info(f"Fine-tuning learning rate: {params['learning_rate']:.6f} -> {new_lr:.6f}")
                self.model.learning_rate = new_lr
    
    def _get_current_hyperparams(self):
        """Extract current hyperparameters from the model"""
        params = {}
        
        # Get common parameters
        if hasattr(self.model, 'learning_rate'):
            params['learning_rate'] = self.model.learning_rate
        
        # PPO specific parameters
        if hasattr(self.model, 'batch_size'):
            params['batch_size'] = self.model.batch_size
        
        if hasattr(self.model, 'n_steps'):
            params['n_steps'] = self.model.n_steps
        
        if hasattr(self.model, 'ent_coef'):
            params['ent_coef'] = self.model.ent_coef
        
        if hasattr(self.model, 'clip_range'):
            if callable(self.model.clip_range):
                params['clip_range'] = self.model.clip_range(1)  # Approximation
            else:
                params['clip_range'] = self.model.clip_range
        
        return params
    

class DataLoader:
    """Class to load and prepare financial data for RL training"""
    
    def __init__(self, data_dir="./data", min_bars=6000):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory to store/load data files
            min_bars: Minimum number of bars to load
        """
        self.data_dir = data_dir
        self.min_bars = min_bars
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_csv_data(self, file_path):
        """Load data from a CSV file"""
        # Load CSV with lowercase column names to ensure compatibility
        df = pd.read_csv(file_path, parse_dates=True)
        
        # Convert all column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Log the available columns for debugging
        logging.debug(f"CSV columns: {list(df.columns)}")
        
        # Identify timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() and 'value' not in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        else:
            logging.warning("No timestamp column found in CSV. Using row index instead.")
        
        # Make sure we have all required columns
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                # Try to find a column with case differences
                case_variants = [c for c in df.columns if c.lower() == col.lower()]
                if case_variants:
                    df[col] = df[case_variants[0]]
                else:
                    logging.error(f"Required column '{col}' not found in CSV.")
                    raise ValueError(f"Required column '{col}' not found in CSV.")
        
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Copy the dataframe to avoid modifying the original
        df = df.copy()
        
        # Check if required indicators already exist in the data
        calculate_ema_short = 'ema_short' not in df.columns
        calculate_ema_long = 'ema_long' not in df.columns
        calculate_atr = 'atr' not in df.columns
        
        # Calculate EMAs if needed
        if calculate_ema_short:
            df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        
        if calculate_ema_long:
            df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Calculate ATR (Average True Range) if needed
        if calculate_atr:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
        
        # Simple calculation for ADX (Directional Movement Index)
        # For simplicity, we'll use a very basic proxy for ADX
        if 'adx' not in df.columns:
            df['up_move'] = df['high'].diff()
            df['down_move'] = df['low'].diff().abs()
            
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # Simplified ADX
            df['plus_di'] = 100 * df['plus_dm'].rolling(window=14).mean() / df['atr'].replace(0, 0.001)
            df['minus_di'] = 100 * df['minus_dm'].rolling(window=14).mean() / df['atr'].replace(0, 0.001)
            
            # Avoid division by zero
            sum_di = df['plus_di'] + df['minus_di']
            df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / sum_di.replace(0, 0.001)
            df['adx'] = df['dx'].rolling(window=14).mean()
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    
    def prepare_train_test_data(self, data=None, csv_file=None, test_ratio=0.2):
        """
        Prepare training and testing datasets
        
        Args:
            data: Pandas DataFrame with financial data (optional)
            csv_file: Path to CSV file with financial data (optional)
            test_ratio: Ratio of data to use for testing
            
        Returns:
            train_data, test_data as pandas DataFrames
        """
        if data is None and csv_file is None:
            # No synthetic data generation - require real data
            raise ValueError("No data provided. Please extract data from NinjaTrader or provide a DataFrame.")
        elif csv_file is not None:
            # Load data from CSV
            data = self.load_csv_data(csv_file)
            
            # Check if we have enough data
            if len(data) < self.min_bars:
                logger.warning(f"CSV data has only {len(data)} bars, which is less than the recommended {self.min_bars} bars.")
                logger.warning("Training results may be poor with insufficient data.")
                
                # If we have extremely small dataset, raise an error
                if len(data) < 100:  # Minimum practical size for any meaningful training
                    logger.error(f"Dataset size ({len(data)} bars) is too small for effective training. Need at least 100 bars.")
                    raise ValueError(f"Dataset size ({len(data)} bars) is too small for effective training")
            
            # Add technical indicators if not present
            required_columns = ['ema_short', 'ema_long', 'atr', 'adx']
            if not all(col in data.columns for col in required_columns):
                data = self.add_technical_indicators(data)
        
        # Sort by index (timestamp) to ensure chronological order
        data.sort_index(inplace=True)
        
        # Ensure we have enough data for window_size
        window_size = 10  # Default window size used in TradingEnvironment
        if len(data) < window_size * 3:
            logger.error(f"Dataset has only {len(data)} bars, which is less than {window_size * 3} (3x window_size).")
            logger.error("Please provide more data or reduce window_size in TradingEnvironment")
            raise ValueError(f"Insufficient data for training with window_size={window_size}")
        
        # Split into train and test sets, ensuring minimum sizes for both
        min_train_size = max(window_size * 5, 100)  # At least 5x window_size or 100 bars
        min_test_size = max(window_size * 2, 50)    # At least 2x window_size or 50 bars
        
        # Adjust test_ratio if needed to ensure minimum sizes
        if len(data) * test_ratio < min_test_size:
            test_ratio = min_test_size / len(data)
            logger.warning(f"Adjusted test_ratio to {test_ratio:.2f} to ensure minimum test set size")
        
        if len(data) * (1 - test_ratio) < min_train_size:
            # If we can't satisfy both constraints, prioritize training set
            test_ratio = max(0.1, 1 - (min_train_size / len(data)))
            logger.warning(f"Adjusted test_ratio to {test_ratio:.2f} to ensure minimum training set size")
        
        split_idx = int(len(data) * (1 - test_ratio))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Prepared data with {len(train_data)} training bars and {len(test_data)} testing bars")
        
        return train_data, test_data


class TrainingManager:
    """Manager class for RL model training and evaluation"""
    
    def __init__(self, models_dir="./models", logs_dir="./logs"):
        """
        Initialize the training manager
        
        Args:
            models_dir: Directory to save trained models
            logs_dir: Directory to save training logs
        """
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        
        # Create necessary directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Set default parameters
        self.train_params = {
            'algo': 'PPO',
            'policy': 'MlpPolicy',
            'learning_rate': 0.0003,
            'batch_size': 64,
            'n_steps': 256,  # Reduced from 2048 to better handle small datasets
            'total_timesteps': 100_000,  # Reduced from 1M to complete training faster with small datasets
            'device': 'auto'
        }
        
        self.env_params = {
            'initial_balance': 100000.0,
            'commission': 0.0001,
            'slippage': 0.0001,
            'window_size': 5,  # Reduced from 10 to handle smaller datasets
            'reward_scaling': 0.01,
            'position_size': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    
    def create_env(self, data, is_train=True):
        """Create a trading environment with the provided data"""
        # Set max steps based on whether we're training or not
        max_steps = None if is_train else len(data)
        
        def _init():
            env = TradingEnvironment(
                df=data,
                initial_balance=self.env_params['initial_balance'],
                max_steps=max_steps,
                window_size=self.env_params['window_size'],
                commission=self.env_params['commission'],
                slippage=self.env_params['slippage'],
                reward_scaling=self.env_params['reward_scaling'],
                position_size=self.env_params['position_size'],
                stop_loss_pct=self.env_params['stop_loss_pct'],
                take_profit_pct=self.env_params['take_profit_pct']
            )
            
            # Wrap with Monitor for training stats
            env = Monitor(env)
            return env
        
        # Use DummyVecEnv as SB3 requires vectorized environments
        vec_env = DummyVecEnv([_init])
        
        # Normalize observations and rewards for both training and evaluation
        # But only update normalization stats during training
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            training=is_train
        )
        
        return vec_env
    
    def create_model(self, env):
        """Create RL model based on selected algorithm"""
        # Choose device (CPU/GPU)
        if self.train_params['device'] == 'auto':
            # Let Stable Baselines decide based on availability
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # Use the specified device
            device_name = self.train_params['device']
            
        device = torch.device(device_name)
        logger.info(f"Using device: {device_name}")
        
        # Create model based on selected algorithm
        if self.train_params['algo'] == 'PPO':
            model = PPO(
                self.train_params['policy'],
                env,
                learning_rate=self.train_params['learning_rate'],
                n_steps=self.train_params['n_steps'],
                batch_size=self.train_params['batch_size'],
                verbose=1,
                device=device
            )
        elif self.train_params['algo'] == 'A2C':
            model = A2C(
                self.train_params['policy'],
                env,
                learning_rate=self.train_params['learning_rate'],
                n_steps=self.train_params['n_steps'],
                verbose=1,
                device=device
            )
        elif self.train_params['algo'] == 'DQN':
            model = DQN(
                self.train_params['policy'],
                env,
                learning_rate=self.train_params['learning_rate'],
                batch_size=self.train_params['batch_size'],
                verbose=1,
                device=device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.train_params['algo']}")
        
        return model
    
    def train(self, train_data, test_data=None, training_callback=None, enable_auto_tuning=True, 
              instrument_name="Unknown", continue_training=False, existing_model_path=None, 
              existing_vec_normalize_path=None, stop_flag_callback=None, **kwargs):
        
        # Check if the dataset is too small and log a warning
        min_recommended_size = 100  # Minimum recommended bars for training
        if len(train_data) < min_recommended_size:
            logger.warning(f"WARNING: Your training dataset only has {len(train_data)} bars. "
                          f"This is much smaller than the recommended minimum of {min_recommended_size} bars. "
                          f"Training results may be poor due to insufficient data.")
        """
        Train the RL model with adaptive hyperparameter tuning
        
        Args:
            train_data: DataFrame with training data
            test_data: Optional DataFrame with testing data for evaluation
            training_callback: Optional callback function to report training progress
            enable_auto_tuning: Whether to automatically tune hyperparameters during training
            instrument_name: Name of the financial instrument being trained on
            continue_training: Whether to continue training from an existing model
            existing_model_path: Path to existing model for continued training
            existing_vec_normalize_path: Path to existing VecNormalize stats for continued training
            **kwargs: Optional parameters to override defaults
            
        Returns:
            Trained model
        """
        # Update parameters with any provided kwargs
        self.train_params.update({k: v for k, v in kwargs.items() if k in self.train_params})
        self.env_params.update({k: v for k, v in kwargs.items() if k in self.env_params})
        
        logger.info(f"Starting training for instrument: {instrument_name}")
        logger.info(f"Algorithm: {self.train_params['algo']}")
        logger.info(f"Training parameters: {self.train_params}")
        logger.info(f"Environment parameters: {self.env_params}")
        logger.info(f"Continue training: {continue_training}")
        
        # Create training environment
        train_env = self.create_env(train_data, is_train=True)
        
        # Create or load model
        if continue_training and existing_model_path and os.path.exists(existing_model_path):
            try:
                # Load the existing model
                logger.info(f"Loading existing model from {existing_model_path}")
                model = self.load_model(existing_model_path)
                
                # Load normalization stats if available
                if existing_vec_normalize_path and os.path.exists(existing_vec_normalize_path):
                    logger.info(f"Loading existing normalization stats from {existing_vec_normalize_path}")
                    train_env = VecNormalize.load(existing_vec_normalize_path, train_env)
                    train_env.training = True  # Make sure we're in training mode
                
                # Update the model's environment
                model.set_env(train_env)
                logger.info("Successfully loaded existing model and environment")
            except Exception as e:
                logger.error(f"Error loading existing model: {e}")
                logger.info("Creating new model instead")
                model = self.create_model(train_env)
        else:
            # Create a new model
            model = self.create_model(train_env)
        
        # Create test environment if test data provided
        eval_callback = None
        if test_data is not None:
            # Create evaluation environment with the same normalization but without updating stats
            test_env = self.create_env(test_data, is_train=False)
            
            # Custom evaluation callback (without saving intermediate models)
            class CustomEvalCallback(BaseCallback):
                def __init__(self, eval_env, eval_freq=10000, log_path=None, deterministic=True, verbose=1):
                    super().__init__(verbose)
                    self.eval_env = eval_env
                    self.eval_freq = eval_freq
                    self.log_path = log_path
                    self.deterministic = deterministic
                    self.best_mean_reward = -np.inf
                    
                def _on_step(self):
                    if self.n_calls % self.eval_freq == 0:
                        # Evaluate the model without trying to sync normalization
                        mean_reward, _ = evaluate_policy(
                            self.model, 
                            self.eval_env, 
                            n_eval_episodes=2,
                            deterministic=self.deterministic
                        )
                        
                        if self.verbose > 0:
                            logger.info(f"Eval num_timesteps={self.num_timesteps}, " 
                                    f"episode_reward={mean_reward:.2f}")
                    
                    return True
            
            # Create our custom callback (without saving intermediate models)
            eval_callback = CustomEvalCallback(
                test_env,
                log_path=self.logs_dir,
                eval_freq=10000,
                deterministic=True
            )
        
        # Training callback for logging and hyperparameter tuning (no model saving during training)
        training_callback_obj = TradingTrainingCallback(
            check_freq=5000,
            save_path=None,  # Don't save intermediate models
            verbose=1,
            training_callback=training_callback,
            enable_auto_tuning=enable_auto_tuning,
            stop_flag_callback=stop_flag_callback  # Pasar el callback para verificar detención
        )
        
        # Create callback list
        callbacks = [training_callback_obj]
        if eval_callback:
            callbacks.append(eval_callback)
        
        # Start training
        start_time = time.time()
        
        # Modificamos learn para manejar la detención externa
        try:
            # Añadimos la opción reset_num_timesteps=False para que no resetee el contador
            # Esto es importante para mantener el progreso correcto
            model.learn(
                total_timesteps=self.train_params['total_timesteps'],
                callback=callbacks
            )
            logger.info("Training completed normally")
        except Exception as e:
            # Capturar excepciones que puedan ocurrir por detener el entrenamiento
            logger.warning(f"Training interrupted: {e}")
            
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model with timestamp and instrument name
        timestamp = datetime.now().strftime("%m-%d-%y_%H%M%S")
        model_filename = f"{instrument_name}_{timestamp}"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save(model_path)
        
        # Also save the VecNormalize stats with matching name
        vec_norm_path = os.path.join(self.models_dir, f"{model_filename}_vecnorm.pkl")
        train_env.save(vec_norm_path)
        
        logger.info(f"Model saved as {model_path}")
        logger.info(f"Normalization stats saved as {vec_norm_path}")
        
        return model, train_env, model_path, vec_norm_path
    
    def load_model(self, model_path, vec_norm_path=None):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
            vec_norm_path: Optional path to the saved VecNormalize stats
            
        Returns:
            Loaded model
        """
        # Extract algorithm from the model file
        algo_name = None
        if 'PPO' in model_path:
            algo_name = 'PPO'
        elif 'A2C' in model_path:
            algo_name = 'A2C'
        elif 'DQN' in model_path:
            algo_name = 'DQN'
        else:
            # Default to PPO if can't determine
            algo_name = 'PPO'
        
        # Choose the right algorithm class
        if algo_name == 'PPO':
            model = PPO.load(model_path)
        elif algo_name == 'A2C':
            model = A2C.load(model_path)
        elif algo_name == 'DQN':
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")
        
        return model
    
    def backtest(self, model, test_data, vec_normalize_path=None):
        """
        Backtest a trained model on test data
        
        Args:
            model: Trained RL model
            test_data: DataFrame with test data
            vec_normalize_path: Optional path to saved VecNormalize stats
            
        Returns:
            Performance metrics
        """
        logger.info("Starting backtest...")
        
        # Create test environment
        test_env = self.create_env(test_data, is_train=False)
        
        # Load normalization stats if provided
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            test_env = VecNormalize.load(vec_normalize_path, test_env)
            # Don't update normalization stats during testing
            test_env.training = False
            test_env.norm_reward = False
        
        # Run backtest
        obs, _ = test_env.reset()
        done = False
        cumulative_reward = 0
        results = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            
            # Extract info from environment (first sub-env in VecEnv)
            info = info[0]
            cumulative_reward += reward[0]
            
            # Store results
            results.append({
                'step': info.get('step', 0),
                'balance': info.get('balance', 0),
                'position': info.get('current_position', 0),
                'reward': reward[0],
                'cumulative_reward': cumulative_reward,
                'trade_count': info.get('trade_count', 0),
                'win_rate': info.get('win_rate', 0),
                'total_pnl': info.get('total_pnl', 0)
            })
            
            # Check if all environments are done
            done = done.all() if hasattr(done, 'all') else done
        
        # Extract final stats
        final_balance = results[-1]['balance']
        total_pnl = results[-1]['total_pnl']
        trade_count = results[-1]['trade_count']
        win_rate = results[-1]['win_rate']
        
        # Calculate performance metrics
        initial_balance = self.env_params['initial_balance']
        roi = (final_balance / initial_balance - 1) * 100
        
        logger.info(f"Backtest completed.")
        logger.info(f"Initial balance: ${initial_balance:.2f}")
        logger.info(f"Final balance: ${final_balance:.2f}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"ROI: {roi:.2f}%")
        logger.info(f"Total trades: {trade_count}")
        logger.info(f"Win rate: {win_rate*100:.2f}%")
        
        # Create performance DataFrame
        performance_df = pd.DataFrame(results)
        
        return performance_df
    
    def save_backtest_results(self, performance_df, filename="backtest_results.csv"):
        """Save backtest results to a CSV file"""
        save_path = os.path.join(self.logs_dir, filename)
        performance_df.to_csv(save_path, index=False)
        logger.info(f"Backtest results saved to {save_path}")
    
    def plot_backtest_results(self, performance_df):
        """Plot backtest results"""
        plt.figure(figsize=(15, 12))
        
        # Plot equity curve
        plt.subplot(3, 1, 1)
        plt.plot(performance_df['balance'])
        plt.title('Equity Curve')
        plt.grid(True)
        
        # Plot cumulative reward
        plt.subplot(3, 1, 2)
        plt.plot(performance_df['cumulative_reward'])
        plt.title('Cumulative Reward')
        plt.grid(True)
        
        # Plot trade count and win rate
        plt.subplot(3, 1, 3)
        ax1 = plt.gca()
        ax1.plot(performance_df['trade_count'], color='blue', label='Trade Count')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Number of Trades', color='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(performance_df['win_rate'], color='green', label='Win Rate')
        ax2.set_ylabel('Win Rate', color='green')
        
        plt.title('Trading Statistics')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, 'backtest_results.png'))
        plt.show()


def main():
    """Main function to run the training pipeline"""
    # Initialize data loader
    data_loader = DataLoader(min_bars=6000)
    
    # Load or generate data
    # Uncomment the following line to load from CSV
    # train_data, test_data = data_loader.prepare_train_test_data(csv_file="path/to/your/data.csv")
    
    # Or use synthetic data for testing
    train_data, test_data = data_loader.prepare_train_test_data()
    
    # Initialize training manager
    training_manager = TrainingManager()
    
    # Set training parameters
    train_params = {
        'algo': 'PPO',  # Options: 'PPO', 'A2C', 'DQN'
        'policy': 'MlpPolicy',
        'learning_rate': 0.0003,
        'batch_size': 64,
        'n_steps': 2048,
        'total_timesteps': 500_000  # Reduced for testing, use 1M+ for better results
    }
    
    # Set environment parameters
    env_params = {
        'initial_balance': 100000.0,
        'commission': 0.0001,
        'slippage': 0.0001,
        'window_size': 10,
        'reward_scaling': 0.01,
        'position_size': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }
    
    # Train the model
    model, train_env = training_manager.train(
        train_data, test_data,
        **train_params, **env_params
    )
    
    # Backtest the model
    vec_norm_path = os.path.join(training_manager.models_dir, "vec_normalize_final.pkl")
    performance_df = training_manager.backtest(model, test_data, vec_norm_path)
    
    # Save and plot results
    training_manager.save_backtest_results(performance_df)
    training_manager.plot_backtest_results(performance_df)


if __name__ == "__main__":
    main()