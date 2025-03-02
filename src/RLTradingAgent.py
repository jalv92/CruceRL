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
        """Get trading action based on market data"""
        # Placeholder implementation returning no trade
        # Returns: trade_signal, ema_choice, position_size, stop_loss, take_profit
        return 0, 'short', 1.0, 0.0, 0.0  # No trade signal as default
        
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
        self.data_receiver_running = False
        self.order_sender_running = False
        self.data_receiver_thread = None
        self.order_sender_thread = None
        self.data_receiver_socket = None
        self.order_sender_socket = None
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
        # Check if the last heartbeat was received recently
        if not self.connection_status:
            return False
            
        # If no heartbeat received for 10 seconds, consider connection lost
        if time.time() - self.last_heartbeat_time > self.connection_timeout:
            self.connection_status = False
            logger.warning("Connection timeout - no heartbeat received")
            return False
        
        # Conexiones activas = conectado
        if (self.data_receiver_socket and self.order_sender_socket):
            return True
            
        return False
    
    def set_auto_trading(self, enabled: bool):
        """Enable or disable auto trading"""
        self.auto_trading_enabled = enabled
        logger.info(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
        print(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
    
    def start(self):
        """Start the interface"""
        try:
            # Inicializar los hilos de comunicación
            self.data_receiver_running = True
            self.data_receiver_thread = threading.Thread(target=self.data_receiver_loop)
            self.data_receiver_thread.daemon = True
            self.data_receiver_thread.start()
            
            self.order_sender_running = True
            self.order_sender_thread = threading.Thread(target=self.order_sender_loop)
            self.order_sender_thread.daemon = True
            self.order_sender_thread.start()
            
            # Actualizar el estado de conexión
            self.last_heartbeat_time = time.time()
            logger.info("NinjaTrader interface started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting NinjaTrader interface: {e}")
            self.connection_status = False
            return False
    
    def stop(self):
        """Stop the interface"""
        self.data_receiver_running = False
        self.order_sender_running = False
        
        # Cerrar data socket
        if self.data_receiver_socket:
            try:
                self.data_receiver_socket.close()
                self.data_receiver_socket = None
            except:
                pass
                
        # Cerrar order socket
        if self.order_sender_socket:
            try:
                self.order_sender_socket.close()
                self.order_sender_socket = None
            except:
                pass
                
        # Esperar a que terminen los hilos
        if self.data_receiver_thread and self.data_receiver_thread.is_alive():
            self.data_receiver_thread.join(timeout=2.0)
        
        if self.order_sender_thread and self.order_sender_thread.is_alive():
            self.order_sender_thread.join(timeout=2.0)
        
        logger.info("NinjaTrader interface stopped")

    def cancel_extraction(self):
        """Cancela la extracción de datos en curso"""
        if self.is_extracting_data:
            logger.info("Cancelando extracción de datos históricos")
            self.is_extracting_data = False
            self.extraction_complete = False
            # Enviar mensaje de cancelación si es necesario
            try:
                if self.order_sender_socket and self.order_sender_socket.fileno() != -1:
                    self.send_order_command("EXTRACTION_CANCEL\n")
            except:
                pass
    
    def data_receiver_loop(self):
        """Loop to receive market data from NinjaTrader"""
        logger.info(f"Starting data receiver to connect to NinjaTrader at {self.server_ip}:{self.data_port}")
        
        # En esta implementación, Python actúa como SERVIDOR en el puerto data_port
        # NinjaTrader se conectará a este puerto
        
        # Crear socket servidor
        server_socket = None
        client_socket = None
        
        try:
            # Configurar el socket servidor
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.data_port))
            server_socket.listen(1)  # Solo aceptar una conexión
            server_socket.settimeout(1.0)  # Timeout para aceptar conexiones
            
            logger.info(f"Data receiver listening on {self.server_ip}:{self.data_port}")
            
            reconnect_attempts = 0
            max_reconnect_attempts = 30  # Mayor tiempo de espera
            
            # Bucle principal para aceptar conexiones
            while self.data_receiver_running and reconnect_attempts < max_reconnect_attempts:
                try:
                    # Esperar a que se conecte NinjaTrader
                    client_socket, addr = server_socket.accept()
                    client_socket.settimeout(1.0)  # Timeout para recepción de datos
                    
                    self.data_receiver_socket = client_socket
                    logger.info(f"Client connected to data receiver from {addr}")
                    self.connection_status = True
                    self.last_heartbeat_time = time.time()
                    reconnect_attempts = 0  # Resetear intentos
                    
                    # Procesar datos recibidos
                    buffer = ""
                    
                    # Bucle para recibir datos
                    while self.data_receiver_running:
                        try:
                            # Recibir datos
                            data = client_socket.recv(4096)
                            if not data:
                                # Conexión cerrada
                                logger.warning("Client disconnected from data receiver")
                                break
                                
                            # Actualizar el tiempo del último heartbeat
                            self.last_heartbeat_time = time.time()
                            
                            # Procesar datos recibidos
                            buffer += data.decode('ascii', errors='replace')
                            
                            # Procesar líneas completas
                            lines = buffer.split('\n')
                            buffer = lines[-1]  # Mantener lo que queda después del último newline
                            
                            for line in lines[:-1]:  # Procesar líneas completas
                                if line.strip():  # Ignorar líneas vacías
                                    self.process_data_message(line)
                        
                        except socket.timeout:
                            # Timeout en recv - verificar tiempo del último heartbeat
                            if time.time() - self.last_heartbeat_time > self.connection_timeout:
                                logger.warning(f"Data receiver heartbeat timeout ({self.connection_timeout}s)")
                                break
                            continue
                        except socket.error as e:
                            logger.error(f"Data receiver socket error: {e}")
                            break
                        except Exception as e:
                            logger.error(f"Data receiver error: {e}")
                            break
                    
                    # Cerrar socket del cliente
                    if client_socket:
                        try:
                            client_socket.close()
                        except:
                            pass
                        
                        self.data_receiver_socket = None
                    
                    # Marcar como desconectado
                    self.connection_status = False
                    
                except socket.timeout:
                    # Timeout al aceptar conexiones (normal)
                    reconnect_attempts += 1
                    if reconnect_attempts % 5 == 0:  # Log cada 5 intentos
                        logger.info(f"Waiting for NinjaTrader to connect to data port... ({reconnect_attempts}/{max_reconnect_attempts})")
                    continue
                except Exception as e:
                    logger.error(f"Data receiver accept error: {e}")
                    reconnect_attempts += 1
                    time.sleep(1.0)
            
            if reconnect_attempts >= max_reconnect_attempts:
                logger.error("No se pudo establecer conexión con NinjaTrader")
                
        except Exception as e:
            logger.error(f"Data receiver initialization error: {e}")
        finally:
            # Cerrar socket servidor
            if server_socket:
                try:
                    server_socket.close()
                except:
                    pass
            
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
                
            self.data_receiver_socket = None
        
        logger.info("Data receiver stopped")
    
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
                
            if message.startswith("EXTRACTION_"):
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
                        
                        # Send response back to NinjaTrader
                        if self.order_sender_socket and self.order_sender_socket.fileno() != -1:
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
                        progress_percent = float(parts[3])
                        
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
                    
                    # Check if we already have data for this instrument
                    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
                    existing_files = []
                    if os.path.exists(data_dir):
                        for file in os.listdir(data_dir):
                            if file.startswith(f"{instrument_name}_") and file.endswith(".csv"):
                                existing_files.append(os.path.join(data_dir, file))
                    
                    # Generate a filename based on whether we're appending or creating new
                    timestamp = datetime.now().strftime("%m-%d-%y_%H%M%S")
                    
                    if existing_files:
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
                            filename = f"data/{instrument_name}_{timestamp}.csv"
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
                        filename = f"data/{instrument_name}_{timestamp}.csv"
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
                        else:
                            logger.error("Failed to save extracted data")
                        
                return
                
            if message.startswith("HISTORICAL:"):
                # Process historical data
                data_str = message[10:]  # Remove "HISTORICAL:" prefix
                self.process_market_data(data_str, is_historical=True)
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
            
    def order_sender_loop(self):
        """Loop to handle orders communication with NinjaTrader"""
        # En esta nueva implementación, Python actúa como SERVIDOR en el puerto order_port
        # NinjaTrader se conectará a este puerto como cliente
        
        # Crear socket servidor
        server_socket = None
        client_socket = None
        
        try:
            # Configurar el socket servidor
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.order_port))
            server_socket.listen(1)  # Solo aceptar una conexión
            server_socket.settimeout(1.0)  # Timeout para aceptar conexiones
            
            logger.info(f"Order handler listening on {self.server_ip}:{self.order_port}")
            
            reconnect_attempts = 0
            max_reconnect_attempts = 30  # Mayor tiempo de espera
            
            # Bucle principal para aceptar conexiones
            while self.order_sender_running and reconnect_attempts < max_reconnect_attempts:
                try:
                    # Esperar a que se conecte NinjaTrader
                    client_socket, addr = server_socket.accept()
                    client_socket.settimeout(1.0)  # Timeout para recepción de datos
                    
                    self.order_sender_socket = client_socket
                    logger.info(f"Client connected to order handler from {addr}")
                    self.connection_status = True
                    self.last_heartbeat_time = time.time()
                    reconnect_attempts = 0  # Resetear intentos
                    
                    # Bucle para recibir confirmaciones y enviar órdenes
                    buffer = ""
                    heartbeat_interval = 2.0  # Enviar heartbeat cada 2 segundos
                    last_heartbeat = time.time()
                    
                    while self.order_sender_running:
                        try:
                            # Verificar si es momento de enviar heartbeat
                            if time.time() - last_heartbeat >= heartbeat_interval:
                                # Enviar heartbeat 
                                self.send_order_command("PING\n")
                                last_heartbeat = time.time()
                            
                            # Verificar si hay datos para recibir
                            if client_socket.fileno() != -1:
                                ready_to_read, _, _ = select.select([client_socket], [], [], 0.1)
                                if ready_to_read:
                                    # Recibir datos (confirmaciones o respuestas)
                                    data = client_socket.recv(4096)
                                    if not data:
                                        # Conexión cerrada
                                        logger.warning("Client disconnected from order handler")
                                        break
                                    
                                    # Actualizar el tiempo del último heartbeat
                                    self.last_heartbeat_time = time.time()
                                    
                                    # Procesar datos recibidos
                                    buffer += data.decode('ascii', errors='replace')
                                    
                                    # Procesar líneas completas
                                    lines = buffer.split('\n')
                                    buffer = lines[-1]  # Mantener lo que queda después del último newline
                                    
                                    for line in lines[:-1]:  # Procesar líneas completas
                                        if line.strip():  # Ignorar líneas vacías
                                            self.process_order_confirmation(line)
                            
                            # Verificar si hay órdenes pendientes para enviar
                            while not self.order_queue.empty():
                                order = self.order_queue.get_nowait()
                                self.send_order_command(order)
                            
                            # Pequeña pausa para no saturar CPU
                            time.sleep(0.05)
                            
                            # Verificar si la conexión ha estado demasiado tiempo sin actividad
                            if time.time() - self.last_heartbeat_time > self.connection_timeout:
                                logger.warning("Order handler heartbeat timeout")
                                break
                        
                        except socket.timeout:
                            # Timeout en recv - verificar tiempo del último heartbeat
                            if time.time() - self.last_heartbeat_time > self.connection_timeout:
                                logger.warning(f"Order handler heartbeat timeout ({self.connection_timeout}s)")
                                break
                            continue
                        except socket.error as e:
                            logger.error(f"Order handler socket error: {e}")
                            break
                        except Exception as e:
                            logger.error(f"Order handler error: {e}")
                            break
                    
                    # Cerrar socket del cliente
                    if client_socket:
                        try:
                            client_socket.close()
                        except:
                            pass
                        
                        self.order_sender_socket = None
                    
                    # Marcar como desconectado
                    self.connection_status = False
                    
                except socket.timeout:
                    # Timeout al aceptar conexiones (normal)
                    reconnect_attempts += 1
                    if reconnect_attempts % 5 == 0:  # Log cada 5 intentos
                        logger.info(f"Waiting for NinjaTrader to connect to order port... ({reconnect_attempts}/{max_reconnect_attempts})")
                    continue
                except Exception as e:
                    logger.error(f"Order handler accept error: {e}")
                    reconnect_attempts += 1
                    time.sleep(1.0)
            
            if reconnect_attempts >= max_reconnect_attempts:
                logger.error("No se pudo establecer conexión con NinjaTrader (order port)")
                
        except Exception as e:
            logger.error(f"Order handler initialization error: {e}")
        finally:
            # Cerrar socket servidor
            if server_socket:
                try:
                    server_socket.close()
                except:
                    pass
            
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
                
            self.order_sender_socket = None
        
        logger.info("Order handler stopped")
    
    def process_order_confirmation(self, message):
        """Process a confirmation message received from NinjaTrader"""
        try:
            # Check for special message types
            if message.startswith("PONG") or message.startswith("HEARTBEAT"):
                # Heartbeat response
                logger.debug("Received heartbeat confirmation")
                self.last_heartbeat_time = time.time()
                self.connection_status = True
                return
                
            if message.startswith("ORDER_CONFIRMED:"):
                # Order confirmation received
                logger.info(f"Order confirmed: {message}")
                # Aquí podrías procesar los detalles específicos de la confirmación
                return
                
            if message.startswith("ERROR:"):
                # Error message received
                logger.error(f"Error from NinjaTrader: {message}")
                # Aquí podrías manejar tipos específicos de errores
                return
                
            if message.startswith("TRADE_EXECUTED:"):
                # Trade executed notification
                logger.info(f"Trade executed: {message}")
                # Procesar la información de la ejecución
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
                return
                
            # Default: log unknown message
            logger.info(f"Received message from NinjaTrader: {message}")
            
        except Exception as e:
            logger.error(f"Error processing order confirmation: {e}")
    
    def send_order_command(self, command):
        """Send a command to NinjaTrader via the order connection"""
        try:
            if not self.order_sender_socket:
                logger.warning("No order connection available")
                # Encolar el comando para enviarlo cuando se restablezca la conexión
                if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                    self.order_queue.put(command)
                return False
                
            # Asegurar que el comando termina con newline
            if not command.endswith('\n'):
                command += '\n'
                
            # Enviar comando
            self.order_sender_socket.sendall(command.encode('ascii'))
            
            # Actualizar tiempo del último heartbeat enviado
            self.last_heartbeat_time = time.time()
            
            # Log detallado para comandos no triviales
            if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                logger.debug(f"Sent command: {command.strip()}")
                
            return True
        except socket.error as e:
            logger.error(f"Socket error sending order command: {e}")
            # Marcar la conexión como perdida
            self.connection_status = False
            
            # Encolar el comando para reintento posterior
            if not command.startswith("HEARTBEAT") and not command.startswith("PING"):
                self.order_queue.put(command)
                
            return False
        except Exception as e:
            logger.error(f"Error sending order command: {e}")
            return False
            
    def request_historical_data(self, callback=None):
        """Request historical data from NinjaTrader"""
        self.extraction_callback = callback
        self.is_extracting_data = True
        self.extraction_complete = False
        self.historical_data = MarketData(max_history=100000)  # Reset historical data
        
        # Send extraction request
        return self.send_order_command("EXTRACT_HISTORICAL_DATA\n")