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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rl_training.log")
    ]
)
logger = logging.getLogger("RLTraining")


class TradingTrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, check_freq=1000, save_path=None, verbose=1, training_callback=None):
        super(TradingTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        self.training_callback = training_callback
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current training stats
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]) if len(self.model.ep_info_buffer) > 0 else 0
            
            logger.info(f"Training step: {self.n_calls}")
            logger.info(f"Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
            
            # Save best model
            if self.save_path is not None and mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.save_path, f"best_model_step_{self.n_calls}"))
                logger.info(f"New best model saved with reward: {mean_reward:.2f}")
            
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
                
                # Create metrics dict
                metrics = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'mean_reward': mean_reward,
                    'win_rate': win_rate,
                    'episode': self.episode_count
                }
                
                # Call the callback with metrics
                self.training_callback(metrics)
        
        return True
    

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
        df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
        
        # Identify timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Copy the dataframe to avoid modifying the original
        df = df.copy()
        
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Calculate ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Simple calculation for ADX (Directional Movement Index)
        # For simplicity, we'll use a very basic proxy for ADX
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].diff().abs()
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Simplified ADX
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=14).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=14).mean() / df['atr']
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
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
                logger.warning(f"CSV data has only {len(data)} bars, which is less than the recommended {self.min_bars}. " +
                              f"Using available data without synthetic supplement.")
            
            # Add technical indicators if not present
            required_columns = ['ema_short', 'ema_long', 'atr', 'adx']
            if not all(col in data.columns for col in required_columns):
                data = self.add_technical_indicators(data)
        
        # Sort by index (timestamp) to ensure chronological order
        data.sort_index(inplace=True)
        
        # Split into train and test sets
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
            'n_steps': 2048,
            'total_timesteps': 1_000_000,
            'device': 'auto'
        }
        
        self.env_params = {
            'initial_balance': 100000.0,
            'commission': 0.0001,
            'slippage': 0.0001,
            'window_size': 10,
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
    
    def train(self, train_data, test_data=None, training_callback=None, **kwargs):
        """
        Train the RL model
        
        Args:
            train_data: DataFrame with training data
            test_data: Optional DataFrame with testing data for evaluation
            training_callback: Optional callback function to report training progress
            **kwargs: Optional parameters to override defaults
            
        Returns:
            Trained model
        """
        # Update parameters with any provided kwargs
        self.train_params.update({k: v for k, v in kwargs.items() if k in self.train_params})
        self.env_params.update({k: v for k, v in kwargs.items() if k in self.env_params})
        
        logger.info(f"Starting training with algorithm: {self.train_params['algo']}")
        logger.info(f"Training parameters: {self.train_params}")
        logger.info(f"Environment parameters: {self.env_params}")
        
        # Create training environment
        train_env = self.create_env(train_data, is_train=True)
        
        # Create model
        model = self.create_model(train_env)
        
        # Create test environment if test data provided
        eval_callback = None
        if test_data is not None:
            # Create evaluation environment with the same normalization but without updating stats
            test_env = self.create_env(test_data, is_train=False)
            
            # Instead of using EvalCallback directly, use a custom function
            # to avoid synchronization issues
            class CustomEvalCallback(BaseCallback):
                def __init__(self, eval_env, eval_freq=10000, best_model_save_path=None, 
                            log_path=None, deterministic=True, verbose=1):
                    super().__init__(verbose)
                    self.eval_env = eval_env
                    self.eval_freq = eval_freq
                    self.best_model_save_path = best_model_save_path
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
                        
                        # Save best model
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.best_model_save_path is not None:
                                self.model.save(os.path.join(
                                    self.best_model_save_path, 
                                    f"best_model_step_{self.num_timesteps}"
                                ))
                                logger.info(f"New best model saved with reward: {mean_reward:.2f}")
                    
                    return True
            
            # Create our custom callback
            eval_callback = CustomEvalCallback(
                test_env,
                best_model_save_path=self.models_dir,
                log_path=self.logs_dir,
                eval_freq=10000,
                deterministic=True
            )
        
        # Training callback for logging
        training_callback_obj = TradingTrainingCallback(
            check_freq=5000,
            save_path=self.models_dir,
            verbose=1,
            training_callback=training_callback  # Pass the callback function
        )
        
        # Create callback list
        callbacks = [training_callback_obj]
        if eval_callback:
            callbacks.append(eval_callback)
        
        # Start training
        start_time = time.time()
        model.learn(
            total_timesteps=self.train_params['total_timesteps'],
            callback=callbacks
        )
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        final_model_path = os.path.join(self.models_dir, "final_model")
        model.save(final_model_path)
        
        # Also save the VecNormalize stats
        vec_norm_path = os.path.join(self.models_dir, "vec_normalize_final.pkl")
        train_env.save(vec_norm_path)
        
        logger.info(f"Final model saved to {final_model_path}")
        
        return model, train_env
    
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