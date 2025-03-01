import socket
import threading
import time
import json
import numpy as np
import pandas as pd
import logging
import queue
import os
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Import Stable Baselines 3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our custom environment and training modules
from src.TradingEnvironment import TradingEnvironment
from src.TrainingManager import DataLoader, TrainingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rl_trading_agent.log")
    ]
)
logger = logging.getLogger("RLTradingAgent")

class MarketData:
    """Class to store and process market data"""
    
    def __init__(self, max_history: int = 6000):
        self.max_history = max_history
        self.data = pd.DataFrame(columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'ema_short', 'ema_long', 'atr', 'adx'
        ])
        
    def add_data(self, data_point: Dict):
        """Add a new data point to the market data"""
        # Convert to DataFrame row and append
        new_row = pd.DataFrame([data_point])
        
        # Append to existing data
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        # Keep only the most recent data points
        if len(self.data) > self.max_history:
            self.data = self.data.iloc[-self.max_history:]
    
    def get_last_n_bars(self, n: int = 100) -> pd.DataFrame:
        """Get the last n bars of data"""
        return self.data.tail(n)
    
    def get_latest_features(self) -> np.ndarray:
        """Extract features from the latest data for the RL agent"""
        if len(self.data) < 10:  # Need at least 10 bars for meaningful features
            return None
        
        # Get recent data
        recent_data = self.data.tail(10).copy()
        
        # Extract basic price features
        close_prices = recent_data['close'].values
        
        # Normalize features to recent volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        
        # Extract technical indicators
        ema_short = recent_data['ema_short'].values[-1]
        ema_long = recent_data['ema_long'].values[-1]
        atr = recent_data['atr'].values[-1]
        adx = recent_data['adx'].values[-1]
        
        # Calculate moving averages
        if len(close_prices) >= 5:
            ma5 = np.mean(close_prices[-5:])
        else:
            ma5 = close_prices[-1]
            
        if len(close_prices) >= 10:
            ma10 = np.mean(close_prices[-10:])
        else:
            ma10 = close_prices[-1]
        
        # Create feature vector: price momentum, volatility, technical indicators
        current_price = close_prices[-1]
        price_norm = current_price * 0.01  # Scale factor for normalization
        
        features = np.array([
            (ema_short - current_price) / price_norm,  # Distance from short EMA
            (ema_long - current_price) / price_norm,   # Distance from long EMA
            (ema_short - ema_long) / price_norm,       # EMA crossover
            atr / price_norm,                          # Normalized ATR
            adx / 100.0,                               # Normalized ADX
            (ma5 - current_price) / price_norm,        # Distance from MA5
            (ma10 - current_price) / price_norm,       # Distance from MA10
            volatility * 100                           # Recent volatility
        ])
        
        return features

    def prepare_for_gym_env(self, window_size: int = 10) -> pd.DataFrame:
        """Prepare data for the Gym environment for prediction"""
        if len(self.data) < window_size:
            return None
        
        # Make sure data has all required columns
        required_columns = ['open', 'high', 'low', 'close', 'ema_short', 'ema_long', 'atr', 'adx']
        for col in required_columns:
            if col not in self.data.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # Sort by timestamp if available
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.sort_values('timestamp', inplace=True)
            self.data.set_index('timestamp', inplace=True)
        
        return self.data


class RLAgent:
    """Reinforcement Learning agent for trading decisions"""
    
    def __init__(self, model_path: str = None, vec_normalize_path: str = None):
        self.model = None
        self.vec_env = None
        self.model_path = model_path
        self.vec_normalize_path = vec_normalize_path
        
        # RL model hyperparameters
        self.window_size = 10
        self.position_size = 1.0
        self.default_stop_loss = 10  # in ticks
        self.default_take_profit = 20  # in ticks
        
        # State trackers
        self.current_position = 0  # -1 for short, 0 for flat, 1 for long
        self.position_entry_price = 0.0
        self.trade_count = 0
        self.successful_trades = 0
        
        # Performance tracking
        self.performance_history = []
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, vec_normalize_path)
        else:
            # Fallback to rule-based model if no RL model available
            logger.info("No RL model provided, using rule-based fallback")
            self.use_rule_based_fallback()
        
        logger.info("RL Agent initialized")
    
    def load_model(self, model_path: str, vec_normalize_path: str = None):
        """Load a trained model from disk"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Determine algorithm type from filename
            if 'PPO' in model_path:
                self.model = PPO.load(model_path)
            elif 'A2C' in model_path:
                self.model = A2C.load(model_path)
            elif 'DQN' in model_path:
                self.model = DQN.load(model_path)
            else:
                # Default to PPO
                self.model = PPO.load(model_path)
            
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Extract parameters from model
            self.window_size = getattr(self.model, 'window_size', 10)
            
            # Load the vectorized environment normalization stats if available
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                logger.info(f"Loading VecNormalize stats from {vec_normalize_path}")
                # We'll create and load the VecNormalize when needed in get_action
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.use_rule_based_fallback()
            return False
    
    def use_rule_based_fallback(self):
        """Initialize a rule-based model as fallback"""
        # Simple rule-based weights for decision making
        self.weights = np.array([
            0.5,    # ema_short - price
            0.3,    # ema_long - price
            0.8,    # ema_short - ema_long
            -0.2,   # atr
            0.4,    # adx
            0.3,    # ma5 - price
            0.1,    # ma10 - price
            -0.5    # volatility
        ])
        
        logger.info("Rule-based model initialized as fallback")
    
    def create_env_for_prediction(self, data: pd.DataFrame) -> gym.Env:
        """Create a Gym environment for model prediction"""
        if data is None or len(data) < self.window_size:
            logger.warning("Not enough data to create a prediction environment")
            return None
        
        # Create bare minimum environment for prediction
        env = TradingEnvironment(
            df=data,
            window_size=self.window_size,
            max_steps=1  # We only need one step for prediction
        )
        
        # Wrap in DummyVecEnv for SB3 compatibility
        vec_env = DummyVecEnv([lambda: env])
        
        # Load normalization stats if available
        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False  # Don't update stats during prediction
            vec_env.norm_reward = False
        
        return vec_env
    
    def get_action_rule_based(self, features: np.ndarray) -> Tuple[int, int, float, float, float]:
        """Get action using rule-based approach (fallback method)"""
        if features is None:
            return 0, 0, 0.0, 0.0, 0.0
        
        # Simple linear combination of features as a trading signal
        signal_strength = np.dot(features, self.weights)
        
        # Determine trading action
        if signal_strength > 0.5:
            trade_signal = 1  # Long
        elif signal_strength < -0.5:
            trade_signal = -1  # Short
        else:
            trade_signal = 0  # Flat
            
        # Only change position if significant change or no position
        if self.current_position != 0 and trade_signal == self.current_position:
            # Hold current position
            return 0, 0, 0.0, 0.0, 0.0
        
        # EMA choice based on signal
        if trade_signal == 1:
            ema_choice = 1  # Short EMA for long positions
        elif trade_signal == -1:
            ema_choice = 2  # Long EMA for short positions
        else:
            ema_choice = 0
        
        # Position size based on signal strength
        position_size = min(2.0, max(0.5, abs(signal_strength))) * self.position_size
        
        # Adjust stop loss and take profit based on signal strength
        volatility_factor = min(2.0, max(0.5, abs(features[-1])))
        stop_loss = self.default_stop_loss * volatility_factor
        take_profit = self.default_take_profit * volatility_factor
        
        # Update current position
        self.current_position = trade_signal
        
        return trade_signal, ema_choice, position_size, stop_loss, take_profit
    
    def get_action_rl(self, market_data: MarketData) -> Tuple[int, int, float, float, float]:
        """Get trading action from the RL model"""
        if self.model is None:
            logger.warning("RL model not loaded, falling back to rule-based method")
            features = market_data.get_latest_features()
            return self.get_action_rule_based(features)
        
        try:
            # Prepare data for the environment
            data = market_data.prepare_for_gym_env(window_size=self.window_size)
            if data is None or len(data) < self.window_size:
                logger.warning("Not enough data for RL prediction")
                features = market_data.get_latest_features()
                return self.get_action_rule_based(features)
            
            # Create environment for prediction
            vec_env = self.create_env_for_prediction(data)
            if vec_env is None:
                logger.warning("Failed to create prediction environment")
                features = market_data.get_latest_features()
                return self.get_action_rule_based(features)
            
            # Reset environment and get observation
            obs, _ = vec_env.reset()
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Convert action to trade signal
            # Action space: 0 = Hold, 1 = Buy, 2 = Sell
            if action[0] == 1:
                trade_signal = 1  # Long
            elif action[0] == 2:
                trade_signal = -1  # Short
            else:
                trade_signal = 0  # Hold
            
            # Only change position if different from current
            if self.current_position != 0 and trade_signal == self.current_position:
                return 0, 0, 0.0, 0.0, 0.0
            
            # EMA choice based on signal
            if trade_signal == 1:
                ema_choice = 1  # Short EMA for long positions
            elif trade_signal == -1:
                ema_choice = 2  # Long EMA for short positions
            else:
                ema_choice = 0
            
            # Get the current ATR for volatility measurement
            current_atr = data['atr'].iloc[-1]
            current_price = data['close'].iloc[-1]
            atr_pct = current_atr / current_price
            
            # Scale position size, stop loss, and take profit based on ATR
            position_size = 1.0  # Base position size
            
            # Stop loss and take profit in ticks
            stop_loss = max(5, int(self.default_stop_loss * atr_pct * 1000))
            take_profit = max(10, int(self.default_take_profit * atr_pct * 1000))
            
            # Update current position
            self.current_position = trade_signal
            
            logger.info(f"RL model prediction: Signal={trade_signal}, EMA={ema_choice}, Size={position_size:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
            
            return trade_signal, ema_choice, position_size, stop_loss, take_profit
        
        except Exception as e:
            logger.error(f"Error in RL prediction: {e}")
            features = market_data.get_latest_features()
            return self.get_action_rule_based(features)
    
    def get_action(self, market_data: MarketData) -> Tuple[int, int, float, float, float]:
        """Determine trading action based on current market data"""
        if self.model is not None:
            return self.get_action_rl(market_data)
        else:
            features = market_data.get_latest_features()
            return self.get_action_rule_based(features)
    
    def update_model(self, reward: float):
        """Update the agent's model based on reward (would be used in online learning)"""
        # In a real online RL implementation, this would update the model based on reward
        # For now, we'll just track performance
        self.trade_count += 1
        if reward > 0:
            self.successful_trades += 1
        
        win_rate = self.successful_trades / max(1, self.trade_count)
        self.performance_history.append({
            'timestamp': datetime.now(),
            'reward': reward,
            'win_rate': win_rate
        })
        
        logger.info(f"Trade completed with reward: {reward:.2f}, win rate: {win_rate:.2f}")


class NinjaTraderInterface:
    """Interface for communicating with NinjaTrader strategy"""
    
    def __init__(self, server_ip: str = '127.0.0.1', data_port: int = 5000, order_port: int = 5001,
                 model_path: str = None, vec_normalize_path: str = None):
        self.server_ip = server_ip
        self.data_port = data_port
        self.order_port = order_port
        
        # Market data storage
        self.market_data = MarketData(max_history=6000)  # Increased for RL
        
        # Communication state
        self.data_receiver_running = False
        self.order_sender_running = False
        self.data_receiver_thread = None
        self.order_sender_thread = None
        self.order_sender_socket = None
        
        # Order queue
        self.order_queue = queue.Queue()
        
        # Initialize RL agent with model if provided
        self.rl_agent = RLAgent(model_path, vec_normalize_path)
        
        logger.info(f"NinjaTrader interface initialized (IP: {server_ip}, Data Port: {data_port}, Order Port: {order_port})")
    
    def start(self):
        """Start the interface"""
        # Start data receiver thread
        self.data_receiver_running = True
        self.data_receiver_thread = threading.Thread(target=self.data_receiver_loop)
        self.data_receiver_thread.daemon = True
        self.data_receiver_thread.start()
        
        # Start order sender thread
        self.order_sender_running = True
        self.order_sender_thread = threading.Thread(target=self.order_sender_loop)
        self.order_sender_thread.daemon = True
        self.order_sender_thread.start()
        
        logger.info("NinjaTrader interface started")
    
    def stop(self):
        """Stop the interface"""
        self.data_receiver_running = False
        self.order_sender_running = False
        
        if self.data_receiver_thread and self.data_receiver_thread.is_alive():
            self.data_receiver_thread.join(timeout=2.0)
        
        if self.order_sender_thread and self.order_sender_thread.is_alive():
            self.order_sender_thread.join(timeout=2.0)
        
        if self.order_sender_socket:
            try:
                self.order_sender_socket.close()
            except:
                pass
        
        logger.info("NinjaTrader interface stopped")
    
    def data_receiver_loop(self):
        """Loop to receive market data from NinjaTrader"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.server_ip, self.data_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)  # 1 second timeout for accept()
            
            logger.info(f"Data receiver listening on {self.server_ip}:{self.data_port}")
            
            while self.data_receiver_running:
                try:
                    client_socket, address = server_socket.accept()
                    client_socket.settimeout(1.0)  # 1 second timeout for recv()
                    logger.info(f"Data connection established from {address}")
                    
                    buffer = ""
                    
                    while self.data_receiver_running:
                        try:
                            data = client_socket.recv(4096).decode('ascii')
                            if not data:
                                break
                            
                            buffer += data
                            
                            # Process complete lines
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                self.process_data(line)
                        
                        except socket.timeout:
                            continue
                        except Exception as e:
                            logger.error(f"Error receiving data: {e}")
                            break
                    
                    client_socket.close()
                    logger.info("Data connection closed")
                
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    time.sleep(1.0)
        
        except Exception as e:
            logger.error(f"Data receiver error: {e}")
        
        finally:
            server_socket.close()
            logger.info("Data receiver stopped")
    
    def process_data(self, data_line: str):
        """Process a line of market data from NinjaTrader"""
        try:
            # Expected format: open,high,low,close,emaShort,emaLong,atr,adx,timestamp
            parts = data_line.strip().split(',')
            
            if len(parts) < 9:
                logger.warning(f"Invalid data format: {data_line}")
                return
            
            # Parse data
            data_point = {
                'open': float(parts[0]),
                'high': float(parts[1]),
                'low': float(parts[2]),
                'close': float(parts[3]),
                'ema_short': float(parts[4]),
                'ema_long': float(parts[5]),
                'atr': float(parts[6]),
                'adx': float(parts[7]),
                'timestamp': parts[8]
            }
            
            # Add to market data
            self.market_data.add_data(data_point)
            
            # Get trading decision from RL agent
            trade_signal, ema_choice, position_size, stop_loss, take_profit = self.rl_agent.get_action(self.market_data)
            
            # Only queue non-zero signals
            if trade_signal != 0:
                # Queue order for sending
                order = {
                    'trade_signal': trade_signal,
                    'ema_choice': ema_choice,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                self.order_queue.put(order)
                logger.info(f"RL agent decision: {order}")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def order_sender_loop(self):
        """Loop to send trading signals to NinjaTrader"""
        while self.order_sender_running:
            try:
                # Check if there are orders to send
                if self.order_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # Get next order
                order = self.order_queue.get(block=False)
                
                # Try to send order
                self.send_order(order)
                
                # Mark task as done
                self.order_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in order sender loop: {e}")
                time.sleep(1.0)
    
    def send_order(self, order: Dict):
        """Send a trading signal to NinjaTrader"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Create socket if not exists or is closed
                if self.order_sender_socket is None:
                    self.order_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.order_sender_socket.settimeout(3.0)  # 3 second timeout
                    self.order_sender_socket.connect((self.server_ip, self.order_port))
                    logger.info(f"Order sender connected to {self.server_ip}:{self.order_port}")
                
                # Format message
                message = f"{order['trade_signal']},{order['ema_choice']},{order['position_size']:.2f},{order['stop_loss']:.2f},{order['take_profit']:.2f}\n"
                
                # Send message
                self.order_sender_socket.sendall(message.encode('ascii'))
                logger.info(f"Order sent: {message.strip()}")
                
                # Successfully sent, return
                return
            
            except Exception as e:
                logger.error(f"Error sending order (attempt {attempt+1}/{max_retries}): {e}")
                
                # Close socket for reconnection
                try:
                    if self.order_sender_socket:
                        self.order_sender_socket.close()
                except:
                    pass
                
                self.order_sender_socket = None
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"Failed to send order after {max_retries} attempts: {order}")


def main():
    """Main function to run the RL trading agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Trading Agent')
    parser.add_argument('--train', action='store_true', help='Train a new model instead of running the agent')
    parser.add_argument('--port', type=int, default=5000, help='Port for the data receiver (default: 5000)')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address to listen on (default: 127.0.0.1)')
    parser.add_argument('--model', type=str, default='models/final_model.zip', help='Path to model file for prediction')
    parser.add_argument('--vec-norm', type=str, default='models/vec_normalize_final.pkl', help='Path to VecNormalize stats')
    parser.add_argument('--data', type=str, help='Path to CSV data file for training')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total timesteps for training')
    
    args = parser.parse_args()
    
    try:
        # Training mode
        if args.train:
            logger.info("Starting training mode")
            
            # Initialize data loader
            data_loader = DataLoader(min_bars=6000)
            
            # Load or generate data
            if args.data and os.path.exists(args.data):
                logger.info(f"Loading data from {args.data}")
                train_data, test_data = data_loader.prepare_train_test_data(csv_file=args.data)
            else:
                logger.info("Generating synthetic data for training")
                train_data, test_data = data_loader.prepare_train_test_data()
            
            # Initialize training manager
            training_manager = TrainingManager()
            
            # Set training parameters
            train_params = {
                'algo': 'PPO',
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'batch_size': 64,
                'n_steps': 2048,
                'total_timesteps': args.timesteps
            }
            
            # Train the model
            model, train_env = training_manager.train(
                train_data, test_data, **train_params
            )
            
            # Backtest the model
            vec_norm_path = os.path.join(training_manager.models_dir, "vec_normalize_final.pkl")
            performance_df = training_manager.backtest(model, test_data, vec_norm_path)
            
            # Save and plot results
            training_manager.save_backtest_results(performance_df)
            logger.info("Training and backtesting completed")
            
        else:
            # Running mode - connect to NinjaTrader
            logger.info("Starting RL Trading Agent in execution mode")
            
            # Check for model file
            if not os.path.exists(args.model):
                logger.warning(f"Model file not found: {args.model}")
                logger.info("Will use rule-based fallback strategy")
            
            # Create NinjaTrader interface
            nt_interface = NinjaTraderInterface(
                server_ip=args.ip,
                data_port=args.port,
                order_port=args.port + 1,
                model_path=args.model,
                vec_normalize_path=args.vec_norm
            )
            
            # Start interface
            nt_interface.start()
            
            # Main loop - keep running until interrupted
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
            
            # Stop interface
            nt_interface.stop()
            
            logger.info("RL Trading Agent stopped")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()