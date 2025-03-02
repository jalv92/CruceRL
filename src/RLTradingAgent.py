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
        
        # Intentar probar la conexión directamente con un socket
        try:
            # Crear socket de prueba
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)  # Timeout de 1 segundo
            result = test_socket.connect_ex((self.server_ip, self.data_port))
            test_socket.close()
            
            # Si result es 0, la conexión fue exitosa
            if result == 0:
                return True
            else:
                logger.warning(f"Connection test failed with code {result}")
                return False
        except Exception as e:
            logger.warning(f"Connection test error: {e}")
            return False
            
        return True
    
    def set_auto_trading(self, enabled: bool):
        """Enable or disable auto trading"""
        self.auto_trading_enabled = enabled
        logger.info(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
        print(f"Auto Trading switched {'ON' if enabled else 'OFF'}")
    
    def start(self):
        """Start the interface"""
        try:
            # NinjaTrader actúa como servidor en los puertos data_port (para enviar datos) y
            # order_port (para recibir órdenes). En esta implementación, Python es el cliente.
            
            # Primero, inicializar los hilos que manejarán la comunicación
            self.data_receiver_running = True
            self.data_receiver_thread = threading.Thread(target=self.data_receiver_loop)
            self.data_receiver_thread.daemon = True
            self.data_receiver_thread.start()
            
            self.order_sender_running = True
            self.order_sender_thread = threading.Thread(target=self.order_sender_loop)
            self.order_sender_thread.daemon = True
            self.order_sender_thread.start()
            
            # Actualizar el estado de conexión - la conexión real la establecerá
            # el data_receiver_loop y el order_sender_loop
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
        
        # En esta implementación, Python es el CLIENTE que se conecta al puerto data_port de NinjaTrader
        # NinjaTrader actúa como SERVIDOR en ese puerto
        
        reconnect_delay = 5  # segundos entre intentos de reconexión
        last_attempt = time.time() - reconnect_delay  # para intentar la conexión inmediatamente
        
        while self.data_receiver_running:
            # Verificar tiempo entre reconexiones
            if time.time() - last_attempt < reconnect_delay:
                time.sleep(0.5)
                continue
                
            # Marcar tiempo de intento
            last_attempt = time.time()
                
            client_socket = None
            try:
                # Crear un nuevo socket para conectar con NinjaTrader
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)  # 3 segundos de timeout para conectar
                
                # Conectar al servidor de datos de NinjaTrader
                logger.info(f"Attempting to connect to NinjaTrader at {self.server_ip}:{self.data_port}...")
                client_socket.connect((self.server_ip, self.data_port))
                
                # Ajustar timeout para recepción de datos
                client_socket.settimeout(1.0)
                
                # Conexión establecida
                logger.info(f"Data connection established to NinjaTrader at {self.server_ip}:{self.data_port}")
                self.connection_status = True
                self.last_heartbeat_time = time.time()
                
                # Procesar datos recibidos
                buffer = ""
                
                while self.data_receiver_running:
                    try:
                        # Recibir datos
                        data = client_socket.recv(4096)
                        if not data:
                            # Conexión cerrada
                            logger.warning("NinjaTrader connection closed")
                            break
                            
                        # Actualizar el tiempo del último heartbeat
                        self.last_heartbeat_time = time.time()
                        
                        # Procesar datos recibidos
                        buffer += data.decode('ascii')
                        
                        # Procesar líneas completas
                        lines = buffer.split('\n')
                        buffer = lines[-1]  # Mantener lo que queda después del último newline
                        
                        for line in lines[:-1]:  # Procesar líneas completas
                            if line.strip():  # Ignorar líneas vacías
                                self.process_data_message(line)
                    
                    except socket.timeout:
                        # Timeout en recv - normal, seguir intentando
                        continue
                    except Exception as e:
                        # Error en la conexión
                        logger.error(f"Data receiver error: {e}")
                        break
                
                # Cerrar socket si hay algún problema
                if client_socket:
                    try:
                        client_socket.close()
                    except:
                        pass
                
                # Marcar como desconectado
                self.connection_status = False
                logger.warning("Data receiver disconnected, will attempt to reconnect")
                
            except socket.timeout:
                # Timeout al conectar
                logger.warning(f"Connection timeout to NinjaTrader at {self.server_ip}:{self.data_port}")
                self.connection_status = False
            except ConnectionRefusedError:
                # Conexión rechazada
                logger.warning(f"Connection refused by NinjaTrader at {self.server_ip}:{self.data_port}")
                self.connection_status = False
            except Exception as e:
                # Otro error
                logger.error(f"Data receiver error: {e}")
                self.connection_status = False
                
            # Cerrar socket si ocurrió un error antes de la conexión
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
                    
            # Esperar antes de intentar reconectar
            time.sleep(1.0)
        
        logger.info("Data receiver stopped")
    
    def process_data_message(self, message):
        """Process a data message received from NinjaTrader"""
        try:
            # Check for special message types
            if message.startswith("PONG"):
                # Heartbeat response
                return
                
            if message.startswith("EXTRACTION_"):
                # Handle extraction-related messages
                if message.startswith("EXTRACTION_START"):
                    # Extract the total bars count
                    parts = message.split(":")
                    if len(parts) > 1:
                        self.total_bars_to_extract = int(parts[1])
                        self.extracted_bars_count = 0
                        self.is_extracting_data = True
                        self.extraction_complete = False
                        
                        logger.info(f"Starting extraction of {self.total_bars_to_extract} bars")
                        
                        # Notify callback if available
                        if self.extraction_callback:
                            self.extraction_callback(0, self.total_bars_to_extract, 0)
                    
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
                    
                    # Generate a timestamp for the filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/extracted_data_{timestamp}.csv"
                    
                    # Save the data to a file
                    if self.historical_data.save_to_file(filename):
                        # Notify callback with the filename
                        if self.extraction_callback:
                            self.extraction_callback(
                                self.total_bars_to_extract,
                                self.total_bars_to_extract,
                                100.0,
                                filename
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
                return
            
            # Default: process as market data
            self.process_market_data(message)
                
        except Exception as e:
            logger.error(f"Error processing data message: {e}")
    
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
        """Loop to send orders to NinjaTrader"""
        reconnect_delay = 5  # segundos entre intentos de reconexión
        last_attempt = time.time() - reconnect_delay  # para intentar la conexión inmediatamente
        
        while self.order_sender_running:
            # Verificar tiempo entre reconexiones
            if time.time() - last_attempt < reconnect_delay:
                time.sleep(0.5)
                continue
                
            # Marcar tiempo de intento
            last_attempt = time.time()
                
            try:
                # Crear socket para conectar al servidor de órdenes de NinjaTrader
                self.order_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.order_sender_socket.settimeout(3.0)  # 3 segundos de timeout
                
                # Conectar al servidor de órdenes de NinjaTrader
                logger.info(f"Connecting to order port at {self.server_ip}:{self.order_port}...")
                self.order_sender_socket.connect((self.server_ip, self.order_port))
                
                # Ajustar timeout para envío/recepción
                self.order_sender_socket.settimeout(1.0)
                
                logger.info(f"Order sender connected to {self.server_ip}:{self.order_port}")
                
                # Enviar ping inicial para verificar conexión
                self.order_sender_socket.sendall(b"PING\n")
                
                # Mantener conexión activa
                while self.order_sender_running:
                    try:
                        # Enviar heartbeat periódicamente
                        self.order_sender_socket.sendall(b"HEARTBEAT\n")
                        time.sleep(5.0)  # Esperar 5 segundos
                    except:
                        logger.warning("Order sender connection lost")
                        break
                        
            except ConnectionRefusedError:
                logger.warning(f"Order port connection refused at {self.server_ip}:{self.order_port}")
            except Exception as e:
                logger.error(f"Order sender error: {e}")
            finally:
                # Cerrar el socket si está abierto
                if self.order_sender_socket:
                    try:
                        self.order_sender_socket.close()
                    except:
                        pass
                    self.order_sender_socket = None
                
            # Esperar antes de reintentar
            time.sleep(1.0)
            
        logger.info("Order sender stopped")
    
    def send_order_command(self, command):
        """Send a command to NinjaTrader via the order connection"""
        try:
            if not self.order_sender_socket:
                logger.warning("No order connection available")
                return False
                
            # Enviar comando
            self.order_sender_socket.sendall(command.encode('ascii'))
            return True
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