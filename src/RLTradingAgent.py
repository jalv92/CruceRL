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
    
    def save_to_file(self, filename: str) -> bool:
        """Save market data to a file for training"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save data to CSV file
            self.data.to_csv(filename)
            logger.info(f"Market data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving market data to file: {e}")
            return False

class RLAgent:
    """Reinforcement Learning agent for trading decisions"""
    
    def __init__(self, model_path: str = None, vec_normalize_path: str = None):
        """Initialize the RL agent with optional model loading"""
        self.model = None
        self.vec_normalize = None
        # TODO: Implement model loading if paths are provided
        if model_path and vec_normalize_path:
            try:
                # Placeholder for loading the model and normalization
                self.model = PPO.load(model_path)
                self.vec_normalize = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: TradingEnvironment()]))
                logger.info(f"Loaded model from {model_path} and normalization from {vec_normalize_path}")
            except Exception as e:
                logger.error(f"Error loading model or normalization: {e}")
                self.model = None
                self.vec_normalize = None
    
    def get_action(self, market_data: MarketData) -> Tuple[int, str, float, float, float]:
        """Get trading action based on market data"""
        # Placeholder implementation returning no trade
        # Returns: trade_signal, ema_choice, position_size, stop_loss, take_profit
        return 0, 'short', 1.0, 0.0, 0.0  # No trade signal as default

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
        
        # Historical data extraction
        self.historical_data = MarketData(max_history=100000)  # Larger capacity for historical data
        self.is_extracting_data = False
        self.extraction_complete = False
        self.extraction_callback = None
        self.extraction_progress = 0
        self.total_bars_to_extract = 0
        self.extracted_bars_count = 0
        
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
            # Check if this is a special message
            if data_line.startswith("EXTRACTION_START:"):
                # Format: EXTRACTION_START:total_bars
                total_bars = int(data_line.split(':')[1])
                self.start_data_extraction(total_bars)
                return
            elif data_line.startswith("EXTRACTION_PROGRESS:"):
                # Format: EXTRACTION_PROGRESS:current:total:percent
                parts = data_line.split(':')
                current = int(parts[1])
                total = int(parts[2])
                percent = float(parts[3])
                self.update_extraction_progress(current, total, percent)
                return
            elif data_line == "EXTRACTION_COMPLETE":
                self.complete_data_extraction()
                return
            elif data_line.startswith("HISTORICAL:"):
                # Format: HISTORICAL:open,high,low,close,ema_short,ema_long,atr,adx,timestamp
                parts = data_line[len("HISTORICAL:"):].strip().split(',')
                self.process_historical_data(parts)
                return
            
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
    
    def start_data_extraction(self, total_bars: int):
        """Start historical data extraction process"""
        logger.info(f"Starting historical data extraction: {total_bars} bars")
        self.is_extracting_data = True
        self.extraction_complete = False
        self.total_bars_to_extract = total_bars
        self.extracted_bars_count = 0
        self.extraction_progress = 0
        
        # Create new empty historical data container
        self.historical_data = MarketData(max_history=total_bars + 1000)  # Add buffer
    
    def update_extraction_progress(self, current: int, total: int, percent: float):
        """Update extraction progress"""
        self.extracted_bars_count = current
        self.total_bars_to_extract = total
        self.extraction_progress = percent
        logger.info(f"Historical data extraction progress: {percent:.1f}% ({current}/{total})")
        
        # Call callback if provided
        if self.extraction_callback:
            self.extraction_callback(current, total, percent)
    
    def complete_data_extraction(self):
        """Complete historical data extraction"""
        logger.info("Historical data extraction completed")
        self.extraction_complete = True
        self.is_extracting_data = False
        
        # Save data to training file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/extracted_data_{timestamp}.csv"
        self.historical_data.save_to_file(filename)
        
        # Call callback if provided
        if self.extraction_callback:
            self.extraction_callback(self.extracted_bars_count, self.total_bars_to_extract, 100.0, filename)
    
    def process_historical_data(self, parts: List[str]):
        """Process historical data bar"""
        if len(parts) < 9:
            logger.warning(f"Invalid historical data format")
            return
        
        try:
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
            
            # Add to historical data
            self.historical_data.add_data(data_point)
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
    
    def request_historical_data(self, callback=None):
        """Request historical data from NinjaTrader"""
        self.extraction_callback = callback
        
        # Reset extraction state
        self.is_extracting_data = False
        self.extraction_complete = False
        self.extraction_progress = 0
        
        # Send command to NinjaTrader to start extracting data
        command = "EXTRACT_HISTORICAL_DATA\n"
        self.send_order_command(command)
        
        logger.info("Historical data extraction request sent")
        return True
    
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
    
    def send_order_command(self, command: str):
        """Send a command to NinjaTrader through the order channel"""
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
                
                # Send command
                self.order_sender_socket.sendall(command.encode('ascii'))
                logger.info(f"Command sent: {command.strip()}")
                
                # Successfully sent, return
                return True
            
            except Exception as e:
                logger.error(f"Error sending command (attempt {attempt+1}/{max_retries}): {e}")
                
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
        
        logger.error(f"Failed to send command after {max_retries} attempts")
        return False

def main():
    """Main function to run the RL trading agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Trading Agent')
    parser.add_argument('--train', action='store_true', help='Train a new model instead of running the agent')
    parser.add_argument('--port', type=int, default=5000, help='Port for the data receiver (default: 5000)')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address to listen on (default: 127.0.0.1)')
    parser.add_argument('--model', type=str, default='models/final_model.zip', help='Path to model file for prediction')
    parser.add_argument('--vec-norm', type=str, default='models/vec_normalize_final.pkl', help='Path to VecNormalize stats')
    parser.add_argument('--extract', action='store_true', help='Extract historical data from NinjaTrader')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total timesteps for training')
    
    args = parser.parse_args()
    
    try:
        # Extract historical data mode
        if args.extract:
            logger.info("Starting historical data extraction mode")
            
            # Create NinjaTrader interface
            nt_interface = NinjaTraderInterface(
                server_ip=args.ip,
                data_port=args.port,
                order_port=args.port + 1
            )
            
            # Start interface
            nt_interface.start()
            
            # Request historical data extraction
            def extraction_callback(current, total, percent, filename=None):
                if filename:
                    logger.info(f"Data extraction complete. Data saved to {filename}")
                else:
                    logger.info(f"Extraction progress: {percent:.1f}% ({current}/{total})")
            
            nt_interface.request_historical_data(callback=extraction_callback)
            
            # Keep running until interrupted
            try:
                while not nt_interface.extraction_complete:
                    time.sleep(1.0)
                
                # Wait a bit more to ensure all data is processed
                time.sleep(3.0)
                
                logger.info("Historical data extraction completed successfully")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
            
            # Stop interface
            nt_interface.stop()
            
        # Training mode
        elif args.train:
            logger.info("Starting training mode")
            
            # Initialize data loader
            data_loader = DataLoader(min_bars=6000)
            
            # Look for extracted data files
            data_dir = "data"
            extracted_files = [f for f in os.listdir(data_dir) if f.startswith("extracted_data_") and f.endswith(".csv")]
            
            if not extracted_files:
                logger.error("No extracted data files found. Please extract data from NinjaTrader first.")
                return False
            
            # Sort by date (newest first)
            extracted_files.sort(reverse=True)
            latest_data_file = os.path.join(data_dir, extracted_files[0])
            
            logger.info(f"Using latest extracted data: {latest_data_file}")
            
            # Load data
            train_data, test_data = data_loader.prepare_train_test_data(csv_file=latest_data_file)
            
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