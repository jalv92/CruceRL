import socket
import threading
import time
import json
import numpy as np
import pandas as pd
import logging
import queue
import os
import select
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Importación de tkinter messagebox (con manejo de error por si no está disponible)
try:
    from tkinter import messagebox
except ImportError:
    # Crear un reemplazo simple si tkinter no está disponible
    class MessageboxStub:
        @staticmethod
        def showinfo(title, message):
            print(f"[INFO] {title}: {message}")
    messagebox = MessageboxStub()

# Import Stable Baselines 3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our custom environment and training modules
from src.TradingEnvironment import TradingEnvironment
from src.TrainingManager import DataLoader, TrainingManager

# Configure logging
import os

# Asegurar que el directorio logs existe
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "rl_trading_agent.log"))
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
        # Convert to DataFrame row
        # Corregir advertencia de pandas evitando concat con un DataFrame vacío
        if self.data.empty:
            # Si el DataFrame está vacío, simplemente crear uno nuevo con el data_point
            self.data = pd.DataFrame([data_point])
        else:
            # Si ya tiene datos, entonces usar concat normalmente
            new_row = pd.DataFrame([data_point])
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
        self.profit_loss = 0.0  # Initialize profit_loss attribute
        self.current_position = 0  # Track current position: -1 (short), 0 (none), 1 (long)
        self.position_size = 0  # Track position size
        self.trades_count = 0  # Track number of trades
        self.trade_count = 0  # This was missing - alias for trades_count for compatibility 
        self.winning_trades = 0  # Track winning trades
        self.successful_trades = 0  # Track successful trades
        self.entry_price = 0.0  # Track entry price for P&L calculation
        
        # Load model if paths are provided
        if model_path and vec_normalize_path:
            try:
                # Load the model and normalization
                self.model = PPO.load(model_path)
                self.vec_normalize = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: TradingEnvironment()]))
                logger.info(f"Loaded model from {model_path} and normalization from {vec_normalize_path}")
            except Exception as e:
                logger.error(f"Error loading model or normalization: {e}")
                self.model = None
                self.vec_normalize = None
    
    def get_action(self, market_data: MarketData) -> Tuple[int, str, float, float, float]:
        """Get trading action based on market data
        
        Args:
            market_data: MarketData object containing recent price data
            
        Returns:
            Tuple containing:
                - trade_signal: -1 (sell), 0 (no trade), 1 (buy)
                - ema_choice: 'short', 'long', or 'both' (which EMA to follow)
                - position_size: size of position to take (1.0 = 100%)
                - stop_loss: stop loss level (0.0 = no stop loss)
                - take_profit: take profit level (0.0 = no take profit)
        """
        if self.model is None:
            # No model loaded, return no trade signal
            return 0, 'short', 1.0, 0.0, 0.0
        
        try:
            # Get market data features for model input
            features = market_data.get_latest_features()
            
            if features is None:
                logger.warning("Not enough market data for features")
                return 0, 'short', 1.0, 0.0, 0.0
            
            # Normalize data if needed
            if self.vec_normalize:
                # Reshape for VecEnv (expects batch dimension)
                features_batch = np.array([features])
                normalized_features = self.vec_normalize.normalize_obs(features_batch)
                features = normalized_features[0]  # Take first (only) element
            
            # Get action from model
            action, _ = self.model.predict(features, deterministic=True)
            
            # Convert continuous action to discrete trade signal
            # Assuming model output is in range [-1, 1] or similar
            if isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else action[0]
            
            # Threshold for trade decisions
            threshold = 0.2  # Require stronger signal
            
            if action > threshold:
                trade_signal = 1  # Buy
            elif action < -threshold:
                trade_signal = -1  # Sell
            else:
                trade_signal = 0  # No trade
            
            # Default values for other parameters
            ema_choice = 'short' if action > 0 else 'long'
            position_size = 1.0  # Full size
            
            # Calculate dynamic stop loss based on ATR
            atr_value = market_data.data['atr'].iloc[-1] if 'atr' in market_data.data.columns else 0.01
            stop_loss = atr_value * 2.0  # 2x ATR
            take_profit = atr_value * 3.0  # 3x ATR (risk:reward 1:1.5)
            
            return trade_signal, ema_choice, position_size, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            return 0, 'short', 1.0, 0.0, 0.0  # Default to no trade on error
        
    def update_profit_loss(self, trade_data: Dict):
        """Update profit and loss tracking based on trade data"""
        try:
            if 'pnl' in trade_data:
                self.profit_loss += trade_data['pnl']
            
            if 'action' in trade_data:
                action = trade_data['action']
                
                # Update position tracking based on action
                if 'Enter Long' in action:
                    self.current_position = 1
                    self.entry_price = trade_data.get('entry_price', 0.0)
                    self.position_size = trade_data.get('size', 1)
                elif 'Enter Short' in action:
                    self.current_position = -1
                    self.entry_price = trade_data.get('entry_price', 0.0)
                    self.position_size = trade_data.get('size', 1)
                elif 'Exit' in action or 'SL/TP' in action:
                    # Track completed trade
                    self.trades_count += 1
                    self.trade_count = self.trades_count  # Keep both variables in sync
                    if trade_data.get('pnl', 0.0) > 0:
                        self.successful_trades += 1
                    
                    self.current_position = 0
                    self.position_size = 0
                    self.entry_price = 0.0
                
            logger.info(f"P&L updated: {self.profit_loss}, Position: {self.current_position}")
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
    
    def get_trading_stats(self) -> Dict:
        """Get current trading statistics"""
        win_rate = self.successful_trades / max(1, self.trades_count)
        
        return {
            'total_pnl': self.profit_loss,
            'current_position': self.current_position,
            'trades_count': self.trades_count,
            'win_rate': win_rate
        }

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
        self.data_client_running = False
        self.order_client_running = False
        self.data_client_thread = None
        self.order_client_thread = None
        self.data_client_socket = None
        self.order_client_socket = None
        self.connection_status = False  # Track connection status
        self.last_heartbeat_time = time.time()
        self.connection_timeout = 10  # 10 seconds timeout for connection
        
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
        
        # Auto trading state
        self.auto_trading_enabled = False
        
        # Initialize RL agent with model if provided
        self.rl_agent = RLAgent(model_path, vec_normalize_path)
        
        logger.info(f"NinjaTrader interface initialized (IP: {server_ip}, Data Port: {data_port}, Order Port: {order_port})")
    
    def is_connected(self):
        """Check if the connection to NinjaTrader is active and healthy"""
        # Consider connected if at least data connection is active
        data_connected = self.data_client_socket is not None
        
        # Verify time since last heartbeat
        heartbeat_valid = (time.time() - self.last_heartbeat_time <= self.connection_timeout)
        
        # Update connection status for reports
        self.connection_status = data_connected and heartbeat_valid
        
        if not heartbeat_valid and data_connected:
            logger.warning("Connection timeout - no heartbeat received recently")
        
        return self.connection_status
    
    def set_auto_trading(self, enabled: bool):
        """Enable or disable auto trading"""
        self.auto_trading_enabled = enabled
        logger.info(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
        print(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
    
    def start(self):
        """Start the interface by connecting to NinjaTrader servers"""
        try:
            # Initialize communication threads
            logger.info(f"Connecting to NinjaTrader at {self.server_ip}:{self.data_port}...")
            
            # Start data client connection thread
            self.data_client_running = True
            self.data_client_thread = threading.Thread(target=self.data_client_loop)
            self.data_client_thread.daemon = True
            self.data_client_thread.start()
            
            # Start order client connection thread
            self.order_client_running = True
            self.order_client_thread = threading.Thread(target=self.order_client_loop)
            self.order_client_thread.daemon = True
            self.order_client_thread.start()
            
            # Wait for max 5 connection attempts
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                logger.info(f"Esperando conexión... intento {attempt}/{max_attempts}")
                time.sleep(2)
                
                if self.is_connected():
                    logger.info("Connected to NinjaTrader servers")
                    return True
            
            # If we get here, connection failed
            logger.error("Failed to connect to NinjaTrader servers")
            return False
            
        except Exception as e:
            logger.error(f"Error starting NinjaTrader interface: {e}")
            self.connection_status = False
            return False
    
    def stop(self):
        """Stop the interface"""
        logger.info("Desconectando de NinjaTrader...")
        
        self.data_client_running = False
        self.order_client_running = False
        
        # Close data socket
        if self.data_client_socket:
            try:
                self.data_client_socket.close()
                self.data_client_socket = None
            except:
                pass
                
        # Close order socket
        if self.order_client_socket:
            try:
                self.order_client_socket.close()
                self.order_client_socket = None
            except:
                pass
                
        # Wait for threads to terminate
        if self.data_client_thread and self.data_client_thread.is_alive():
            self.data_client_thread.join(timeout=2.0)
        
        if self.order_client_thread and self.order_client_thread.is_alive():
            self.order_client_thread.join(timeout=2.0)
        
        logger.info("NinjaTrader interface stopped")

    def cancel_extraction(self):
        """Cancela la extracción de datos en curso"""
        if self.is_extracting_data:
            logger.info("Cancelando extracción de datos históricos")
            self.is_extracting_data = False
            self.extraction_complete = False
            # Enviar mensaje de cancelación si es necesario
            try:
                self.send_order_command("EXTRACTION_CANCEL\n")
            except Exception as e:
                logger.error(f"Error al cancelar extracción: {e}")
                pass
    
    def data_client_loop(self):
        """Client loop to connect to NinjaTrader data server"""
        logger.info(f"Starting data client to connect to NinjaTrader at {self.server_ip}:{self.data_port}")
        
        # In this implementation, Python acts as CLIENT connecting to NinjaTrader's server
        max_reconnect_attempts = 30  # Maximum number of connection attempts
        reconnect_attempts = 0
        reconnect_delay = 2  # seconds
        
        while self.data_client_running and reconnect_attempts < max_reconnect_attempts:
            try:
                # Create a new socket for each connection attempt
                if self.data_client_socket:
                    try:
                        self.data_client_socket.close()
                    except:
                        pass
                    self.data_client_socket = None
                
                # Create TCP client socket
                self.data_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.data_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.data_client_socket.settimeout(5.0)  # 5 second timeout for connections
                
                # Attempt to connect to NinjaTrader server
                logger.info(f"Attempting to connect to NinjaTrader data server at {self.server_ip}:{self.data_port}...")
                self.data_client_socket.connect((self.server_ip, self.data_port))
                
                # Set shorter timeout for operations
                self.data_client_socket.settimeout(1.0)
                
                # Connected successfully
                logger.info(f"Connected to NinjaTrader data server at {self.server_ip}:{self.data_port}")
                self.connection_status = True
                self.last_heartbeat_time = time.time()
                reconnect_attempts = 0  # Reset attempts on successful connection
                
                # Send connection information to NinjaTrader
                self.data_client_socket.sendall(f"CONNECT:Python_Client\n".encode('ascii'))
                
                # Process data from the server
                buffer = ""
                
                # Main data reception loop
                while self.data_client_running and self.data_client_socket:
                    try:
                        # Check if there's data to read
                        ready_to_read, _, _ = select.select([self.data_client_socket], [], [], 0.1)
                        
                        if ready_to_read:
                            # Receive data from the server
                            data = self.data_client_socket.recv(4096)
                            
                            if not data:
                                # Connection closed by server
                                logger.warning("NinjaTrader data server closed the connection")
                                break
                                
                            # Update heartbeat time
                            self.last_heartbeat_time = time.time()
                            
                            # Process received data
                            buffer += data.decode('ascii', errors='replace')
                            
                            # Process complete messages
                            lines = buffer.split('\n')
                            buffer = lines[-1]  # Keep what's after the last newline
                            
                            for line in lines[:-1]:  # Process complete lines
                                if line.strip():  # Ignore empty lines
                                    self.process_data_message(line)
                        
                        # Send heartbeats to keep connection alive
                        if time.time() - self.last_heartbeat_time > 5:  # Every 5 seconds
                            try:
                                self.data_client_socket.sendall("PING\n".encode('ascii'))
                                self.last_heartbeat_time = time.time()
                            except Exception as e:
                                logger.error(f"Error sending heartbeat: {e}")
                                break
                    
                    except socket.timeout:
                        # Timeout is normal, continue
                        continue
                    except Exception as e:
                        logger.error(f"Error in data client loop: {e}")
                        break
                        
                # Connection lost or closed, clean up
                if self.data_client_socket:
                    try:
                        self.data_client_socket.close()
                    except:
                        pass
                    self.data_client_socket = None
                
                logger.warning("Disconnected from NinjaTrader data server, will attempt to reconnect...")
                self.connection_status = False
                
                # Increment reconnection attempts and delay before retry
                reconnect_attempts += 1
                retry_delay = min(30, reconnect_delay * reconnect_attempts)  # Exponential backoff up to 30 seconds
                logger.info(f"Retrying connection in {retry_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Error in data client loop: {e}")
                
                # Clean up socket
                if self.data_client_socket:
                    try:
                        self.data_client_socket.close()
                    except:
                        pass
                    self.data_client_socket = None
                
                # Increment reconnection attempts and delay before retry
                reconnect_attempts += 1
                retry_delay = min(30, reconnect_delay * reconnect_attempts)
                logger.info(f"Retrying connection in {retry_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(retry_delay)
            
            # Maximum number of reconnection attempts reached
            if reconnect_attempts >= max_reconnect_attempts:
                logger.error("Maximum number of data connection attempts reached. Giving up.")
                break
        
        logger.info("Data client thread stopped")
    
    def order_client_loop(self):
        """Client loop to connect to NinjaTrader order server"""
        logger.info(f"Starting order client to connect to NinjaTrader at {self.server_ip}:{self.order_port}")
        
        # In this implementation, Python acts as CLIENT connecting to NinjaTrader's server
        max_reconnect_attempts = 30  # Maximum number of connection attempts
        reconnect_attempts = 0
        reconnect_delay = 2  # seconds
        
        while self.order_client_running and reconnect_attempts < max_reconnect_attempts:
            try:
                # Create a new socket for each connection attempt
                if self.order_client_socket:
                    try:
                        self.order_client_socket.close()
                    except:
                        pass
                    self.order_client_socket = None
                
                # Create TCP client socket
                self.order_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.order_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.order_client_socket.settimeout(5.0)  # 5 second timeout for connections
                
                # Attempt to connect to NinjaTrader server
                logger.info(f"Attempting to connect to NinjaTrader order server at {self.server_ip}:{self.order_port}...")
                self.order_client_socket.connect((self.server_ip, self.order_port))
                
                # Set shorter timeout for operations
                self.order_client_socket.settimeout(1.0)
                
                # Connected successfully
                logger.info(f"Connected to NinjaTrader order server at {self.server_ip}:{self.order_port}")
                
                # Send connection information to NinjaTrader
                self.order_client_socket.sendall(f"CONNECT:Python_Order_Client\n".encode('ascii'))
                
                # Process data from the server
                buffer = ""
                
                # Main data reception loop
                while self.order_client_running and self.order_client_socket:
                    try:
                        # Check if there's data to read
                        ready_to_read, _, _ = select.select([self.order_client_socket], [], [], 0.1)
                        
                        if ready_to_read:
                            # Receive data from the server
                            data = self.order_client_socket.recv(4096)
                            
                            if not data:
                                # Connection closed by server
                                logger.warning("NinjaTrader order server closed the connection")
                                break
                                
                            # Process received data
                            buffer += data.decode('ascii', errors='replace')
                            
                            # Process complete messages
                            lines = buffer.split('\n')
                            buffer = lines[-1]  # Keep what's after the last newline
                            
                            for line in lines[:-1]:  # Process complete lines
                                if line.strip():  # Ignore empty lines
                                    self.process_order_message(line)
                        
                        # Check if there are pending orders to send
                        if not self.order_queue.empty():
                            order = self.order_queue.get()
                            try:
                                self.order_client_socket.sendall((order + "\n").encode('ascii'))
                                logger.debug(f"Sent order: {order}")
                            except Exception as e:
                                logger.error(f"Error sending order: {e}")
                                # Put the order back in the queue
                                self.order_queue.put(order)
                                break
                        
                        # Send heartbeats to keep connection alive
                        if time.time() - self.last_heartbeat_time > 5:  # Every 5 seconds
                            try:
                                self.order_client_socket.sendall("PING\n".encode('ascii'))
                            except Exception as e:
                                logger.error(f"Error sending heartbeat: {e}")
                                break
                    
                    except socket.timeout:
                        # Timeout is normal, continue
                        continue
                    except Exception as e:
                        logger.error(f"Error in order client loop: {e}")
                        break
                        
                # Connection lost or closed, clean up
                if self.order_client_socket:
                    try:
                        self.order_client_socket.close()
                    except:
                        pass
                    self.order_client_socket = None
                
                logger.warning("Disconnected from NinjaTrader order server, will attempt to reconnect...")
                
                # Increment reconnection attempts and delay before retry
                reconnect_attempts += 1
                retry_delay = min(30, reconnect_delay * reconnect_attempts)  # Exponential backoff up to 30 seconds
                logger.info(f"Retrying connection in {retry_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Error connecting to NinjaTrader order server: {e}")
                
                # Clean up socket
                if self.order_client_socket:
                    try:
                        self.order_client_socket.close()
                    except:
                        pass
                    self.order_client_socket = None
                
                # Increment reconnection attempts and delay before retry
                reconnect_attempts += 1
                retry_delay = min(30, reconnect_delay * reconnect_attempts)
                logger.info(f"Retrying connection in {retry_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(retry_delay)
            
            # Maximum number of reconnection attempts reached
            if reconnect_attempts >= max_reconnect_attempts:
                logger.error("Maximum number of order connection attempts reached. Giving up.")
                break
        
        logger.info("Order client thread stopped")
                        
    # Este método se ha refactorizado y mejorado en otro lugar del código
    # para evitar duplicados. La nueva versión se encuentra más abajo.
            
    def process_order_message(self, message):
        """Process an order message received from NinjaTrader"""
        try:
            # Check for special message types
            if message.startswith("ORDER_SERVER_READY"):
                # Server welcome message
                logger.info("NinjaTrader order server is ready")
                return
                
            if message.startswith("PING"):
                # Respond to ping with pong
                try:
                    self.order_client_socket.sendall("PONG\n".encode('ascii'))
                except Exception as e:
                    logger.error(f"Error sending PONG response: {e}")
                return
                
            if message.startswith("PONG"):
                # Heartbeat response
                logger.debug("Received PONG heartbeat from order server")
                self.last_heartbeat_time = time.time()
                self.connection_status = True
                return
                
            if message.startswith("TRADE_EXECUTED:"):
                # Process trade execution
                parts = message[15:].split(",")  # Remove "TRADE_EXECUTED:" prefix
                if len(parts) >= 5:
                    trade_data = {
                        'action': parts[0],
                        'entry_price': float(parts[1]) if parts[1] else 0,
                        'exit_price': float(parts[2]) if parts[2] else 0,
                        'pnl': float(parts[3]) if parts[3] else 0,
                        'size': int(parts[4]) if parts[4] else 0
                    }
                    
                    # Update profit/loss tracking
                    self.rl_agent.update_profit_loss(trade_data)
                    logger.info(f"Trade executed: {trade_data['action']}, P&L: {trade_data['pnl']}")
                return
                
            if message.startswith("ORDER_CONFIRMED:"): 
                logger.info(f"Order confirmed: {message}")
                return
                
            # Handle extraction-related responses
            if message.startswith("EXTRACTION_"):
                # Forward to the data processing function
                self.process_data_message(message)
                return
                
            # Log any other messages
            logger.info(f"Received order message: {message}")
            
        except Exception as e:
            logger.error(f"Error processing order message: {e}")
            logger.error(f"Message content: {message}")
                        
    def send_trading_action(self, signal, ema_choice, position_size, stop_loss, take_profit):
        """Send a trading action to NinjaTrader"""
        try:
            # Format the order command
            order_command = f"{signal},{ema_choice},{position_size},{stop_loss},{take_profit}"
            
            # Send the command (will be placed in the queue if not connected)
            self.order_queue.put(order_command)
            
            logger.info(f"Queued trading action: {order_command}")
        except Exception as e:
            logger.error(f"Error queueing trading action: {e}")
            
    def send_order_command(self, command):
        """Send a command to NinjaTrader via the order connection"""
        try:
            if not self.order_client_socket or not self.order_client_socket.fileno() > 0:
                logger.warning("No order connection available")
                # Queue the command to send when connection is restored
                if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                    self.order_queue.put(command)
                return False
                
            # Ensure command ends with newline
            if not command.endswith('\n'):
                command += '\n'
                
            # Send command
            self.order_client_socket.sendall(command.encode('ascii'))
            
            # Log non-trivial commands
            if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                logger.debug(f"Sent command: {command.strip()}")
                
            return True
        except Exception as e:
            logger.error(f"Error sending order command: {e}")
            
            # Queue the command for retry
            if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                self.order_queue.put(command)
                
            return False
            
    def request_historical_data(self, callback=None):
        """Request historical data from NinjaTrader"""
        self.extraction_callback = callback
        self.is_extracting_data = True
        self.extraction_complete = False
        self.historical_data = MarketData(max_history=100000)  # Reset historical data
        
        logger.info("Requesting historical data from NinjaTrader...")
        
        # Send extraction request
        result = self.send_order_command("EXTRACT_HISTORICAL_DATA\n")
        logger.info(f"Send extraction command result: {result}")
        return result
    
    def process_data_message(self, message):
        """Process a data message received from NinjaTrader"""
        try:
            # Check for special message types
            if message.startswith("PONG"):
                # Heartbeat response
                logger.debug("Received PONG heartbeat")
                self.last_heartbeat_time = time.time()
                self.connection_status = True
                return
                
            if message.startswith("SERVER_READY") or message.startswith("SERVER_INFO"):
                # Server information or welcome message
                logger.info(f"NinjaTrader server message: {message}")
                return
                
            if message.startswith("PING"):
                # Respond to ping with pong
                try:
                    self.data_client_socket.sendall("PONG\n".encode('ascii'))
                    self.last_heartbeat_time = time.time()
                except Exception as e:
                    logger.error(f"Error sending PONG response: {e}")
                return
                
            # Si el mensaje tiene formato HISTORICAL:datos, extraer solo la parte de datos
            if message.startswith("HISTORICAL:"):
                data_str = message[10:]  # Remove "HISTORICAL:" prefix
                self.process_market_data(data_str, is_historical=True)
                return
                
            if message.startswith("EXTRACTION_"):
                logger.info(f"Received extraction message: {message}")
                # Handle extraction-related messages
                if message.startswith("EXTRACTION_START"):
                    # Extract the total bars count and instrument name
                    parts = message.split(":")
                    if len(parts) > 1:
                        self.total_bars_to_extract = int(parts[1])
                        self.extracted_bars_count = 0
                        self.is_extracting_data = True
                        self.extraction_complete = False
                        
                        # Extract instrument name if available
                        instrument_name = "Unknown"
                        if len(parts) > 2:
                            instrument_name = parts[2].strip()
                        
                        # Reset historical data buffer for this new extraction
                        self.historical_data = MarketData(max_history=100000)
                        
                        logger.info(f"Starting extraction of {self.total_bars_to_extract} bars for {instrument_name}")
                        
                        # Notify callback if available
                        if self.extraction_callback:
                            self.extraction_callback(0, self.total_bars_to_extract, 0, None, instrument_name)
                    
                elif message.startswith("CHECK_EXISTING_DATA"):
                    # Check if we have existing data for an instrument
                    parts = message.split(":")
                    if len(parts) > 1:
                        instrument_name = parts[1].strip()
                        
                        # Check for existing data files
                        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
                        existing_files = []
                        latest_timestamp = None
                        
                        if os.path.exists(data_dir):
                            for file in os.listdir(data_dir):
                                if file.startswith(f"{instrument_name}_") and file.endswith(".csv"):
                                    file_path = os.path.join(data_dir, file)
                                    existing_files.append(file_path)
                        
                        if existing_files:
                            # We have existing data, load the most recent file to check last timestamp
                            existing_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                            latest_file = existing_files[0]
                            
                            try:
                                df = pd.read_csv(latest_file)
                                if 'timestamp' in df.columns and len(df) > 0:
                                    # Convert to datetime if it's not already
                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                    latest_timestamp = df['timestamp'].max()
                                    
                                    logger.info(f"Found existing data for {instrument_name} up to {latest_timestamp}")
                                    
                                    # Tell NinjaTrader to start extraction from this date
                                    response = f"EXISTING_DATA_FOUND:{instrument_name}:{latest_timestamp}\n"
                                else:
                                    logger.info(f"Existing data file found but no valid timestamp column")
                                    response = f"NO_EXISTING_DATA:{instrument_name}\n"
                            except Exception as e:
                                logger.error(f"Error reading existing data file: {e}")
                                response = f"NO_EXISTING_DATA:{instrument_name}\n"
                        else:
                            logger.info(f"No existing data found for {instrument_name}")
                            response = f"NO_EXISTING_DATA:{instrument_name}\n"
                        
                        # Send response back to NinjaTrader using order client socket
                        try:
                            self.send_order_command(response)
                        except Exception as e:
                            logger.error(f"Error sending data existence response: {e}")
                    
                elif message.startswith("EXTRACTION_PROGRESS"):
                    # Update extraction progress
                    parts = message.split(":")
                    if len(parts) > 3:
                        self.extracted_bars_count = int(parts[1])
                        self.total_bars_to_extract = int(parts[2])
                        # Ensure we have a valid progress percentage
                        try:
                            progress_percent = float(parts[3])
                        except (ValueError, IndexError):
                            # Calculate it ourselves if parsing fails
                            progress_percent = (self.extracted_bars_count / max(1, self.total_bars_to_extract)) * 100.0
                        
                        logger.info(f"Extraction progress: {progress_percent:.1f}% ({self.extracted_bars_count}/{self.total_bars_to_extract})")
                        
                        # Notify callback if available
                        if self.extraction_callback:
                            self.extraction_callback(
                                self.extracted_bars_count,
                                self.total_bars_to_extract,
                                progress_percent
                            )
                    
                elif message.startswith("EXTRACTION_COMPLETE"):
                    # Extraction completed
                    self.is_extracting_data = False
                    self.extraction_complete = True
                    
                    # Extract instrument name from message if available
                    instrument_name = "Unknown"
                    parts = message.split(":")
                    if len(parts) > 1:
                        instrument_name = parts[1].strip()
                    
                    logger.info(f"Extraction complete for {instrument_name}")
                    
                    # Check if we already have data for this instrument
                    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
                    os.makedirs(data_dir, exist_ok=True)  # Ensure data directory exists
                    
                    existing_files = []
                    if os.path.exists(data_dir):
                        for file in os.listdir(data_dir):
                            if file.startswith(f"{instrument_name}_") and file.endswith(".csv"):
                                existing_files.append(os.path.join(data_dir, file))
                    
                    # Generate a filename based on whether we're appending or creating new
                    timestamp = datetime.now().strftime("%m-%d-%y_%H%M%S")
                    
                    if existing_files and len(existing_files) > 0:
                        # Sort by modification time to get the most recent file
                        existing_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                        latest_file = existing_files[0]
                        
                        # Try to load existing data
                        try:
                            existing_data = pd.read_csv(latest_file)
                            logger.info(f"Found existing data file {latest_file} with {len(existing_data)} bars")
                            
                            # Merge existing data with new data
                            merged_data = pd.concat([existing_data, self.historical_data.data], ignore_index=True)
                            
                            # Remove duplicates if any (based on timestamp)
                            if 'timestamp' in merged_data.columns:
                                merged_data.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                                
                            # Sort by timestamp if available
                            if 'timestamp' in merged_data.columns:
                                merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
                                merged_data.sort_values('timestamp', inplace=True)
                            
                            # Save back to the same file
                            merged_data.to_csv(latest_file, index=False)
                            filename = latest_file
                            logger.info(f"Updated existing data file with {len(merged_data)} total bars")
                            
                            # Notify callback with the filename
                            if self.extraction_callback:
                                self.extraction_callback(
                                    self.total_bars_to_extract,
                                    self.total_bars_to_extract,
                                    100.0,
                                    filename,
                                    instrument_name
                                )
                            
                        except Exception as e:
                            logger.error(f"Error merging with existing data: {e}, creating new file instead")
                            # Fall back to creating a new file
                            filename = f"data/extracted_data_{timestamp}.csv"
                            if self.historical_data.save_to_file(filename):
                                if self.extraction_callback:
                                    self.extraction_callback(
                                        self.total_bars_to_extract,
                                        self.total_bars_to_extract,
                                        100.0,
                                        filename,
                                        instrument_name
                                    )
                    else:
                        # No existing file, create a new one
                        filename = f"data/extracted_data_{timestamp}.csv"
                        if self.historical_data.save_to_file(filename):
                            # Notify callback with the filename
                            if self.extraction_callback:
                                self.extraction_callback(
                                    self.total_bars_to_extract,
                                    self.total_bars_to_extract,
                                    100.0,
                                    filename,
                                    instrument_name
                                )
                            logger.info(f"Saved extracted data to {filename}")
                        else:
                            logger.error("Failed to save extracted data")
                        
                return
                
            if message.startswith("TRADE_EXECUTED:"):
                # Process trade execution
                parts = message[15:].split(",")  # Remove "TRADE_EXECUTED:" prefix
                if len(parts) >= 5:
                    trade_data = {
                        'action': parts[0],
                        'entry_price': float(parts[1]) if parts[1] else 0,
                        'exit_price': float(parts[2]) if parts[2] else 0,
                        'pnl': float(parts[3]) if parts[3] else 0,
                        'size': int(parts[4]) if parts[4] else 0
                    }
                    
                    # Update profit/loss tracking
                    self.rl_agent.update_profit_loss(trade_data)
                    logger.info(f"Trade executed: {trade_data['action']}, P&L: {trade_data['pnl']}")
                return
            
            # Default: process as market data
            self.process_market_data(message)
                
        except Exception as e:
            logger.error(f"Error processing data message: {e}")
            logger.error(f"Message content: {message}")
    
    def process_market_data(self, data_str, is_historical=False):
        """Process market data received from NinjaTrader"""
        try:
            # Parse CSV data
            fields = data_str.strip().split(',')
            if len(fields) < 9:
                logger.warning(f"Invalid market data format: {data_str}")
                return
                
            # Extract data fields
            data_point = {
                'open': float(fields[0]),
                'high': float(fields[1]),
                'low': float(fields[2]),
                'close': float(fields[3]),
                'ema_short': float(fields[4]),
                'ema_long': float(fields[5]),
                'atr': float(fields[6]),
                'adx': float(fields[7]),
                'timestamp': fields[8]
            }
            
            # Add to appropriate data store
            if is_historical:
                self.historical_data.add_data(data_point)
                
                # Log progress occasionally (every 100 bars)
                if self.historical_data.data.shape[0] % 100 == 0:
                    logger.info(f"Received {self.historical_data.data.shape[0]} historical bars")
            else:
                self.market_data.add_data(data_point)
                
                # Check if we should make a trading decision
                if self.auto_trading_enabled and hasattr(self, 'rl_agent') and self.rl_agent:
                    # Get action from RL agent
                    action = self.rl_agent.get_action(self.market_data)
                    if action and action[0] != 0:  # If there's a trade signal
                        self.send_trading_action(*action)
                        
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            logger.error(f"Data string: {data_str}")
            
    def send_trading_action(self, signal, ema_choice, position_size, stop_loss, take_profit):
        """Send a trading action to NinjaTrader"""
        try:
            # Format the order command
            order_command = f"{signal},{ema_choice},{position_size},{stop_loss},{take_profit}\n"
            
            # Send the command
            self.send_order_command(order_command)
            
            logger.info(f"Sent trading action: {order_command.strip()}")
        except Exception as e:
            logger.error(f"Error sending trading action: {e}")
            
    # The order_sender_loop and process_order_confirmation methods have been removed
    # since Python now acts only as a client connecting to NinjaTrader's servers
    # The send_order_command method is already implemented correctly above
    
            
    # Este método fue movido arriba y mejorado, esta duplicación ya no es necesaria