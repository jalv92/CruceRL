import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext, ttk
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import os
import queue
from datetime import datetime
import sys
import torch

# Asegurarnos de que el directorio actual está en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar nuestros módulos custom
try:
    from src.TradingEnvironment import TradingEnvironment
    from src.TrainingManager import DataLoader, TrainingManager
    from src.RLTradingAgent import RLAgent, NinjaTraderInterface, MarketData
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    MODULES_AVAILABLE = False

# ---------------------------------------------------------------------
# Configuración del logging
# ---------------------------------------------------------------------
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        self.log_queue.put(record)

log_queue = queue.Queue()
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
queue_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.addHandler(queue_handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("TradingSystemGUI")

# Asegurar que el directorio logs existe
import os
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configuración adicional para loggear a un archivo
file_handler = logging.FileHandler(os.path.join(logs_dir, "rl_trading_system.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# ---------------------------------------------------------------------
# Esquema de colores
# ---------------------------------------------------------------------
COLORS = {
    'bg_very_dark': '#1e1e1e',
    'bg_dark': '#252526',
    'bg_medium': '#333333',
    'bg_light': '#3f3f46',
    'fg_white': '#ffffff',
    'fg_light': '#f0f0f0',
    'fg_medium': '#cccccc',
    'accent': '#0078d7',
    'accent_hover': '#1c97ea',
    'accent_pressed': '#005a9e',
    'green': '#6aaa64',
    'red': '#d9534f',
    'yellow': '#ffc107',
    'border': '#3f3f46',
    'selected': '#264f78',
}

# ---------------------------------------------------------------------
# Clase base para paneles
# ---------------------------------------------------------------------
class BasePanel(tk.Frame):
    def __init__(self, parent, title="Panel", **kwargs):
        super().__init__(parent, bg=COLORS['bg_dark'], **kwargs)

        # Título en la parte superior
        self.title_label = tk.Label(
            self, text=title,
            font=('Segoe UI', 14, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_dark'],
            anchor="w", padx=10, pady=5
        )
        self.title_label.pack(side=tk.TOP, fill=tk.X)

        # Separador
        self.separator = tk.Frame(self, height=1, bg=COLORS['border'])
        self.separator.pack(side=tk.TOP, fill=tk.X)

        # Contenedor principal donde irán los widgets específicos de cada panel
        self.main_container = tk.Frame(self, bg=COLORS['bg_dark'])
        self.main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

# ---------------------------------------------------------------------
# Panel de estadísticas
# ---------------------------------------------------------------------
class StatsPanel(BasePanel):
    def __init__(self, parent):
        super().__init__(parent, title="Trading Statistics")
        self.stats = {
            'balance': 100000.0,
            'total_pnl': 0.0,
            'trades_count': 0,
            'win_rate': 0.0,
            'current_position': 'None',
            'training_progress': 0
        }
        self.setup_ui()
    
    def setup_ui(self):
        container = self.main_container
        # Usamos grid para las etiquetas dentro de container
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        label_style = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'font': ('Segoe UI', 11),
            'anchor': 'w'
        }
        value_style = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_white'],
            'font': ('Segoe UI', 11, 'bold'),
            'anchor': 'e'
        }

        # Filas con grid
        tk.Label(container, text="Account Balance:", **label_style)\
            .grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.balance_var = tk.StringVar(value="$100,000.00")
        tk.Label(container, textvariable=self.balance_var, **value_style)\
            .grid(row=0, column=1, sticky='e', padx=10, pady=5)

        tk.Label(container, text="Total P&L:", **label_style)\
            .grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.pnl_var = tk.StringVar(value="$0.00")
        self.pnl_label = tk.Label(container, textvariable=self.pnl_var, **value_style)
        self.pnl_label.grid(row=1, column=1, sticky='e', padx=10, pady=5)

        tk.Label(container, text="Total Trades:", **label_style)\
            .grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.trades_var = tk.StringVar(value="0")
        tk.Label(container, textvariable=self.trades_var, **value_style)\
            .grid(row=2, column=1, sticky='e', padx=10, pady=5)

        tk.Label(container, text="Win Rate:", **label_style)\
            .grid(row=3, column=0, sticky='w', padx=10, pady=5)
        self.winrate_var = tk.StringVar(value="0.0%")
        tk.Label(container, textvariable=self.winrate_var, **value_style)\
            .grid(row=3, column=1, sticky='e', padx=10, pady=5)

        tk.Label(container, text="Current Position:", **label_style)\
            .grid(row=4, column=0, sticky='w', padx=10, pady=5)
        self.position_var = tk.StringVar(value="None")
        self.position_label = tk.Label(container, textvariable=self.position_var, **value_style)
        self.position_label.grid(row=4, column=1, sticky='e', padx=10, pady=5)

        tk.Label(container, text="Connection Status:", **label_style)\
            .grid(row=5, column=0, sticky='w', padx=10, pady=5)
        self.connection_var = tk.StringVar(value="Disconnected")
        self.connection_label = tk.Label(
            container, textvariable=self.connection_var,
            fg=COLORS['red'], bg=COLORS['bg_dark'],
            font=('Segoe UI', 11, 'bold'), anchor='e'
        )
        self.connection_label.grid(row=5, column=1, sticky='e', padx=10, pady=5)

    def update_stats(self, stats):
        self.stats.update(stats)
        # Actualizar cada valor
        self.balance_var.set(f"${self.stats['balance']:,.2f}")
        pnl = self.stats['total_pnl']
        self.pnl_var.set(f"${pnl:,.2f}")

        # Color según PnL
        if pnl > 0:
            self.pnl_label.configure(fg=COLORS['green'])
        elif pnl < 0:
            self.pnl_label.configure(fg=COLORS['red'])
        else:
            self.pnl_label.configure(fg=COLORS['fg_white'])

        self.trades_var.set(f"{self.stats['trades_count']}")
        self.winrate_var.set(f"{self.stats['win_rate'] * 100:.1f}%")

        position = self.stats['current_position']
        self.position_var.set(position)
        if position == 'Long':
            self.position_label.configure(fg=COLORS['green'])
        elif position == 'Short':
            self.position_label.configure(fg=COLORS['red'])
        else:
            self.position_label.configure(fg=COLORS['fg_white'])

    def update_connection(self, connected):
        if connected:
            self.connection_var.set("Connected")
            self.connection_label.configure(fg=COLORS['green'])
        else:
            self.connection_var.set("Disconnected")
            self.connection_label.configure(fg=COLORS['red'])

# ---------------------------------------------------------------------
# Panel de operaciones recientes
# ---------------------------------------------------------------------
class TradesPanel(BasePanel):
    def __init__(self, parent):
        super().__init__(parent, title="Recent Trades")
        self.trades = []
        self.setup_ui()
    
    def setup_ui(self):
        # Usaremos pack para todo dentro de self.main_container
        container = self.main_container

        table_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        header_frame = tk.Frame(table_frame, bg=COLORS['bg_medium'])
        header_frame.pack(fill=tk.X, side=tk.TOP)

        self.col_widths = {'Time': 80, 'Action': 120, 'Price': 100, 'PnL': 100}
        header_style = {
            'bg': COLORS['bg_medium'],
            'fg': COLORS['fg_white'],
            'font': ('Segoe UI', 10, 'bold'),
            'height': 2
        }

        # Cabeceras
        tk.Label(header_frame, text="Time", width=self.col_widths['Time']//8, **header_style).pack(side=tk.LEFT)
        tk.Label(header_frame, text="Action", width=self.col_widths['Action']//8, **header_style).pack(side=tk.LEFT)
        tk.Label(header_frame, text="Price", width=self.col_widths['Price']//8, **header_style).pack(side=tk.LEFT)
        tk.Label(header_frame, text="P&L", width=self.col_widths['PnL']//8, **header_style).pack(side=tk.LEFT)

        # Canvas con scrollbar
        self.canvas = tk.Canvas(table_frame, bg=COLORS['bg_dark'], bd=0, highlightthickness=0, height=300)
        scrollbar = tk.Scrollbar(table_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS['bg_dark'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Ajustar ancho del canvas al cambiar tamaño
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self.rows = []
    
    def _on_canvas_configure(self, event):
        # Ajustar ancho del frame interno
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def update_trades(self, trades_list):
        """Actualiza la lista completa de operaciones"""
        # Limpiar filas existentes
        for row in self.rows:
            row.destroy()
        self.rows = []
        self.trades = []
        
        # Añadir nuevas operaciones
        for trade in trades_list:
            self.add_trade(trade)
    
    def add_trade(self, trade):
        time_str = trade.get('timestamp', '')
        if isinstance(time_str, pd.Timestamp):
            time_str = time_str.strftime('%H:%M:%S')
        elif isinstance(time_str, datetime):
            time_str = time_str.strftime('%H:%M:%S')

        action = trade.get('action', '')
        
        if 'Enter' in action:
            price = trade.get('entry_price', 0.0)
        else:
            price = trade.get('exit_price', 0.0)

        pnl = trade.get('pnl', 0.0)

        # Colores de fondo y texto según acción
        if 'Enter Long' in action:
            bg_color = COLORS['green']
            fg_color = COLORS['bg_very_dark']
        elif 'Enter Short' in action:
            bg_color = COLORS['red']
            fg_color = COLORS['bg_very_dark']
        elif 'Exit' in action or 'SL/TP' in action:
            bg_color = COLORS['accent']
            fg_color = COLORS['bg_very_dark']
        else:
            bg_color = COLORS['bg_dark']
            fg_color = COLORS['fg_white']

        pnl_color = (COLORS['green'] if pnl > 0
                     else COLORS['red'] if pnl < 0
                     else fg_color)

        # Crear fila
        row_frame = tk.Frame(self.scrollable_frame, bg=bg_color)
        row_frame.pack(fill=tk.X, pady=1)

        tk.Label(
            row_frame, text=time_str,
            width=self.col_widths['Time']//8,
            bg=bg_color, fg=fg_color, font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)

        tk.Label(
            row_frame, text=action,
            width=self.col_widths['Action']//8,
            bg=bg_color, fg=fg_color, font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)

        tk.Label(
            row_frame, text=f"${price:,.2f}",
            width=self.col_widths['Price']//8,
            bg=bg_color, fg=fg_color, font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)

        pnl_text = f"${pnl:,.2f}" if pnl != 0 else "-"
        tk.Label(
            row_frame, text=pnl_text,
            width=self.col_widths['PnL']//8,
            bg=bg_color, fg=pnl_color, font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)

        self.rows.append(row_frame)
        # Limitar a 50 filas
        if len(self.rows) > 50:
            row_to_remove = self.rows.pop(0)
            row_to_remove.destroy()

        # Desplazar scroll al final
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

        # Almacenar
        self.trades.append(trade)

# ---------------------------------------------------------------------
# Panel de gráficos actualizado para mostrar métricas de entrenamiento
# ---------------------------------------------------------------------
class ChartPanel(BasePanel):
    def __init__(self, parent):
        super().__init__(parent, title="Market Data & Training Visualization")
        self.price_data = []
        self.performance_data = {
            'balance': [100000.0],
            'timestamps': [pd.Timestamp.now()]
        }
        
        # Métricas de entrenamiento
        self.training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_timestamps': [],
            'mean_rewards': [],
            'win_rates': []
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Crear notebook para tener pestañas
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pestaña de datos de mercado
        self.market_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(self.market_frame, text="Market Data")
        
        # Pestaña de entrenamiento
        self.training_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(self.training_frame, text="Training Metrics")
        
        # Configuración de gráficos para datos de mercado
        self.market_figure = plt.Figure(figsize=(8, 6), dpi=100, facecolor=COLORS['bg_dark'])
        
        # Subplot de precio
        self.price_ax = self.market_figure.add_subplot(2, 1, 1)
        self.price_ax.set_facecolor(COLORS['bg_dark'])
        self.price_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.price_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.price_ax.set_title('Price Chart', color=COLORS['fg_white'])

        # Subplot de balance
        self.perf_ax = self.market_figure.add_subplot(2, 1, 2)
        self.perf_ax.set_facecolor(COLORS['bg_dark'])
        self.perf_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.perf_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.perf_ax.set_title('Account Balance', color=COLORS['fg_white'])

        self.market_figure.tight_layout(pad=2)

        # FigureCanvas para datos de mercado
        self.market_canvas = FigureCanvasTkAgg(self.market_figure, self.market_frame)
        self.market_canvas_widget = self.market_canvas.get_tk_widget()
        self.market_canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Configuración de gráficos para métricas de entrenamiento
        self.training_figure = plt.Figure(figsize=(8, 6), dpi=100, facecolor=COLORS['bg_dark'])
        
        # Subplot de recompensas de episodio
        self.rewards_ax = self.training_figure.add_subplot(2, 2, 1)
        self.rewards_ax.set_facecolor(COLORS['bg_dark'])
        self.rewards_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.rewards_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.rewards_ax.set_title('Episode Rewards', color=COLORS['fg_white'])
        self.rewards_ax.set_ylabel('Reward', color=COLORS['fg_light'])
        
        # Subplot de duración de episodio
        self.lengths_ax = self.training_figure.add_subplot(2, 2, 2)
        self.lengths_ax.set_facecolor(COLORS['bg_dark'])
        self.lengths_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.lengths_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.lengths_ax.set_title('Episode Length', color=COLORS['fg_white'])
        self.lengths_ax.set_ylabel('Steps', color=COLORS['fg_light'])
        
        # Subplot de recompensas promedio
        self.mean_rewards_ax = self.training_figure.add_subplot(2, 2, 3)
        self.mean_rewards_ax.set_facecolor(COLORS['bg_dark'])
        self.mean_rewards_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.mean_rewards_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.mean_rewards_ax.set_title('Mean Reward', color=COLORS['fg_white'])
        self.mean_rewards_ax.set_xlabel('Episode', color=COLORS['fg_light'])
        self.mean_rewards_ax.set_ylabel('Mean Reward', color=COLORS['fg_light'])
        
        # Subplot de tasa de victorias
        self.win_rate_ax = self.training_figure.add_subplot(2, 2, 4)
        self.win_rate_ax.set_facecolor(COLORS['bg_dark'])
        self.win_rate_ax.tick_params(colors=COLORS['fg_light'])
        for spine in self.win_rate_ax.spines.values():
            spine.set_color(COLORS['border'])
        self.win_rate_ax.set_title('Win Rate', color=COLORS['fg_white'])
        self.win_rate_ax.set_xlabel('Episode', color=COLORS['fg_light'])
        self.win_rate_ax.set_ylabel('Win Rate %', color=COLORS['fg_light'])
        
        self.training_figure.tight_layout(pad=2)
        
        # FigureCanvas para métricas de entrenamiento
        self.training_canvas = FigureCanvasTkAgg(self.training_figure, self.training_frame)
        self.training_canvas_widget = self.training_canvas.get_tk_widget()
        self.training_canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_price_chart(self, data):
        self.price_data = data
        if not data:
            return

        self.price_ax.clear()

        timestamps = [d.get('timestamp') for d in data]
        closes = [d.get('close') for d in data]
        self.price_ax.plot(timestamps, closes, color=COLORS['accent'], linewidth=1.5)

        # Marcar trades
        for point in data:
            if 'action' in point:
                action = point['action']
                timestamp = point['timestamp']
                price = point.get('entry_price', point.get('exit_price', 0))
                if 'Enter Long' in action:
                    self.price_ax.scatter(timestamp, price, color=COLORS['green'], marker='^', s=100)
                elif 'Enter Short' in action:
                    self.price_ax.scatter(timestamp, price, color=COLORS['red'], marker='v', s=100)
                elif 'Exit' in action or 'SL/TP' in action:
                    self.price_ax.scatter(timestamp, price, color=COLORS['accent'], marker='o', s=80)

        self.price_ax.set_facecolor(COLORS['bg_dark'])
        self.price_ax.tick_params(colors=COLORS['fg_light'])
        self.price_ax.set_title('Price Chart', color=COLORS['fg_white'])
        self.price_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        self.price_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
        for spine in self.price_ax.spines.values():
            spine.set_color(COLORS['border'])

        self.market_canvas.draw()

    def update_performance_chart(self, balance, timestamp):
        self.performance_data['balance'].append(balance)
        self.performance_data['timestamps'].append(timestamp)

        # Limitar a 1000 puntos
        if len(self.performance_data['balance']) > 1000:
            self.performance_data['balance'] = self.performance_data['balance'][-1000:]
            self.performance_data['timestamps'] = self.performance_data['timestamps'][-1000:]

        self.perf_ax.clear()
        self.perf_ax.plot(
            self.performance_data['timestamps'],
            self.performance_data['balance'],
            color=COLORS['green'],
            linewidth=1.5
        )

        self.perf_ax.set_facecolor(COLORS['bg_dark'])
        self.perf_ax.tick_params(colors=COLORS['fg_light'])
        self.perf_ax.set_title('Account Balance', color=COLORS['fg_white'])
        self.perf_ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
        self.perf_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        self.perf_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
        for spine in self.perf_ax.spines.values():
            spine.set_color(COLORS['border'])

        self.market_canvas.draw()
    
    def update_training_metrics(self, training_metrics):
        """
        Actualiza los gráficos de métricas de entrenamiento
        
        Args:
            training_metrics: Dict con métricas de entrenamiento
                - episode_reward: recompensa del episodio actual
                - episode_length: longitud del episodio actual
                - mean_reward: recompensa media de todos los episodios
                - win_rate: tasa de victorias
                - episode: número de episodio actual
        """
        # Actualizar datos
        self.training_data['episode_rewards'].append(training_metrics.get('episode_reward', 0))
        self.training_data['episode_lengths'].append(training_metrics.get('episode_length', 0))
        self.training_data['episode_timestamps'].append(datetime.now())
        self.training_data['mean_rewards'].append(training_metrics.get('mean_reward', 0))
        self.training_data['win_rates'].append(training_metrics.get('win_rate', 0) * 100)  # Convertir a porcentaje
        
        # Obtener el número de episodio actual
        current_episode = training_metrics.get('episode', len(self.training_data['episode_rewards']))
        
        # Generar eje x basado en el número de episodios acumulados
        episodes = list(range(1, len(self.training_data['episode_rewards']) + 1))
        
        # Actualizar gráficos solo si tenemos datos
        if len(episodes) > 0:
            # Recompensas de episodio
            self.rewards_ax.clear()
            self.rewards_ax.plot(episodes, self.training_data['episode_rewards'], 
                                color=COLORS['accent'], linewidth=1.5)
            self.rewards_ax.set_facecolor(COLORS['bg_dark'])
            self.rewards_ax.tick_params(colors=COLORS['fg_light'])
            self.rewards_ax.set_title('Episode Rewards', color=COLORS['fg_white'])
            self.rewards_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
            for spine in self.rewards_ax.spines.values():
                spine.set_color(COLORS['border'])
            
            # Longitud de episodio
            self.lengths_ax.clear()
            self.lengths_ax.plot(episodes, self.training_data['episode_lengths'], 
                            color=COLORS['yellow'], linewidth=1.5)
            self.lengths_ax.set_facecolor(COLORS['bg_dark'])
            self.lengths_ax.tick_params(colors=COLORS['fg_light'])
            self.lengths_ax.set_title('Episode Length', color=COLORS['fg_white'])
            self.lengths_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
            for spine in self.lengths_ax.spines.values():
                spine.set_color(COLORS['border'])
            
            # Recompensas promedio
            self.mean_rewards_ax.clear()
            self.mean_rewards_ax.plot(episodes, self.training_data['mean_rewards'], 
                                    color=COLORS['green'], linewidth=1.5)
            self.mean_rewards_ax.set_facecolor(COLORS['bg_dark'])
            self.mean_rewards_ax.tick_params(colors=COLORS['fg_light'])
            self.mean_rewards_ax.set_title('Mean Reward', color=COLORS['fg_white'])
            self.mean_rewards_ax.set_xlabel('Episode', color=COLORS['fg_light'])
            self.mean_rewards_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
            for spine in self.mean_rewards_ax.spines.values():
                spine.set_color(COLORS['border'])
            
            # Tasa de victorias
            self.win_rate_ax.clear()
            self.win_rate_ax.plot(episodes, self.training_data['win_rates'], 
                                color=COLORS['red'], linewidth=1.5)
            self.win_rate_ax.set_facecolor(COLORS['bg_dark'])
            self.win_rate_ax.tick_params(colors=COLORS['fg_light'])
            self.win_rate_ax.set_title('Win Rate', color=COLORS['fg_white'])
            self.win_rate_ax.set_xlabel('Episode', color=COLORS['fg_light'])
            self.win_rate_ax.set_ylabel('Win Rate %', color=COLORS['fg_light'])
            self.win_rate_ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['border'])
            for spine in self.win_rate_ax.spines.values():
                spine.set_color(COLORS['border'])
        
        # Actualizar canvas
        self.training_figure.tight_layout(pad=2)
        self.training_canvas.draw()
        
        # Cambiar a la pestaña de entrenamiento si estamos en modo entrenamiento
        self.notebook.select(1)  # Índice 1 corresponde a la pestaña de entrenamiento

# ---------------------------------------------------------------------
# Diálogo de configuración de entrenamiento
# ---------------------------------------------------------------------
class TrainingConfigDialog(tk.Toplevel):
    def __init__(self, parent, config_panel):
        super().__init__(parent)
        self.parent = parent
        self.config_panel = config_panel
        self.title("Training Configuration")
        self.geometry("700x600")  # Aumentamos el tamaño para asegurar que todo es visible
        self.minsize(650, 550)    # Aumentamos el tamaño mínimo
        self.configure(background=COLORS['bg_dark'])
        self.transient(parent)
        self.grab_set()
        
        if os.path.exists("icon.ico"):
            self.iconbitmap("icon.ico")
        
        # Frame principal para contener todo
        main_frame = tk.Frame(self, bg=COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título principal
        title_label = tk.Label(
            main_frame,
            text="Configuración de Entrenamiento",
            font=('Segoe UI', 14, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_dark']
        )
        title_label.pack(pady=(0, 10))
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Inicializar variables de configuración
        self.init_config_vars()
        
        # Recreamos todas las pestañas aquí en lugar de usar el panel existente
        # Pestaña 1: Parámetros básicos del algoritmo
                
        basic_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(basic_frame, text="Algoritmo")
        
        # Usar los valores del panel original
        self.setup_algo_tab(basic_frame)
        
        # Pestaña 2: Parámetros del entorno
        env_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(env_frame, text="Entorno")
        
        self.setup_env_tab(env_frame)
        
        # Pestaña 3: Sistema de recompensas
        reward_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(reward_frame, text="Recompensas")
        
        self.setup_reward_tab(reward_frame)
        
        # Pestaña 4: Presets de trading
        presets_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(presets_frame, text="Presets")
        
        self.setup_presets_tab(presets_frame)
        
        # Botones en la parte inferior
        button_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        button_frame.pack(fill=tk.X, pady=10)
        
        ok_button = tk.Button(
            button_frame,
            text="OK",
            command=self.on_ok,
            bg=COLORS['green'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['green'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=5
        )
        ok_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.destroy,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=5
        )
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Centrar
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def init_config_vars(self):
        """Inicializa las variables de configuración del diálogo"""
        try:
            # Algoritmo - Intentamos obtener valores del panel de configuración
            if hasattr(self.config_panel, 'algo_var'):
                self.algo_var = tk.StringVar(value=self.config_panel.algo_var.get())
                self.timesteps_var = tk.StringVar(value=self.config_panel.timesteps_var.get())
                self.learning_rate_var = tk.StringVar(value=self.config_panel.learning_rate_var.get()) 
                self.batch_size_var = tk.StringVar(value=self.config_panel.batch_size_var.get())
                self.n_steps_var = tk.StringVar(value=self.config_panel.n_steps_var.get())
                self.device_var = tk.StringVar(value=self.config_panel.device_var.get())
                self.auto_tune_var = tk.BooleanVar(value=self.config_panel.auto_tune_var.get())
                
                # Entorno
                self.balance_var = tk.StringVar(value=self.config_panel.balance_var.get())
                self.commission_var = tk.StringVar(value=self.config_panel.commission_var.get())
                self.slippage_var = tk.StringVar(value=self.config_panel.slippage_var.get())
                self.window_size_var = tk.StringVar(value=self.config_panel.window_size_var.get())
                self.position_size_var = tk.StringVar(value=self.config_panel.position_size_var.get())
                self.stop_loss_var = tk.StringVar(value=self.config_panel.stop_loss_var.get())
                self.take_profit_var = tk.StringVar(value=self.config_panel.take_profit_var.get())
                
                # Recompensas
                self.reward_scaling_var = tk.StringVar(value=self.config_panel.reward_scaling_var.get())
                self.inactivity_penalty_var = tk.StringVar(value=self.config_panel.inactivity_penalty_var.get())
                self.bankruptcy_penalty_var = tk.StringVar(value=self.config_panel.bankruptcy_penalty_var.get())
                self.drawdown_factor_var = tk.StringVar(value=self.config_panel.drawdown_factor_var.get())
                self.win_rate_bonus_var = tk.StringVar(value=self.config_panel.win_rate_bonus_var.get())
                self.normalize_rewards_var = tk.BooleanVar(value=self.config_panel.normalize_rewards_var.get())
                self.capital_efficiency_bonus_var = tk.StringVar(value=self.config_panel.capital_efficiency_bonus_var.get())
                self.time_decay_factor_var = tk.StringVar(value=self.config_panel.time_decay_factor_var.get())
        except (AttributeError, TypeError) as e:
            # Si hay algún error, inicializar con valores por defecto
            # Algoritmo
            self.algo_var = tk.StringVar(value="PPO")
            self.timesteps_var = tk.StringVar(value="500000")
            self.learning_rate_var = tk.StringVar(value="0.0003")
            self.batch_size_var = tk.StringVar(value="64")
            self.n_steps_var = tk.StringVar(value="2048")
            self.device_var = tk.StringVar(value="auto")
            self.auto_tune_var = tk.BooleanVar(value=True)
            
            # Entorno
            self.balance_var = tk.StringVar(value="100000.0")
            self.commission_var = tk.StringVar(value="0.0001")
            self.slippage_var = tk.StringVar(value="0.0001")
            self.window_size_var = tk.StringVar(value="10")
            self.position_size_var = tk.StringVar(value="0.1")
            self.stop_loss_var = tk.StringVar(value="0.02")
            self.take_profit_var = tk.StringVar(value="0.04")
            
            # Recompensas
            self.reward_scaling_var = tk.StringVar(value="0.05")
            self.inactivity_penalty_var = tk.StringVar(value="-0.00005")
            self.bankruptcy_penalty_var = tk.StringVar(value="-1.0")
            self.drawdown_factor_var = tk.StringVar(value="0.2")
            self.win_rate_bonus_var = tk.StringVar(value="0.0005")
            self.normalize_rewards_var = tk.BooleanVar(value=True)
            self.capital_efficiency_bonus_var = tk.StringVar(value="0.0")
            self.time_decay_factor_var = tk.StringVar(value="0.0")
    
    def setup_algo_tab(self, frame):
        """Configurar pestaña de algoritmo"""
        # Estilo para etiquetas
        label_style = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'font': ('Segoe UI', 10),
            'anchor': 'w'
        }
        
        # Estilo para entradas
        entry_style = {
            'bg': COLORS['bg_medium'],
            'fg': COLORS['fg_white'],
            'insertbackground': COLORS['fg_white'],
            'relief': tk.FLAT,
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': COLORS['border']
        }
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        
        # Filas de configuración básica
        row = 0
        
        # Algoritmo
        tk.Label(frame, text="Algorithm:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        algo_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        algo_frame.grid(row=row, column=1, sticky='ew', padx=10, pady=5)
        
        for algo in ['PPO', 'A2C', 'DQN']:
            rb = tk.Radiobutton(
                algo_frame, text=algo, variable=self.algo_var, value=algo,
                bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
                selectcolor=COLORS['bg_medium'], 
                activebackground=COLORS['bg_dark'],
                activeforeground=COLORS['accent']
            )
            rb.pack(side=tk.LEFT, padx=10)
        
        row += 1
        
        # Timesteps
        tk.Label(frame, text="Timesteps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        timesteps_entry = tk.Entry(frame, textvariable=self.timesteps_var, width=10, **entry_style)
        timesteps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Learning rate
        tk.Label(frame, text="Learning Rate:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        lr_entry = tk.Entry(frame, textvariable=self.learning_rate_var, width=10, **entry_style)
        lr_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Batch size
        tk.Label(frame, text="Batch Size:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        batch_entry = tk.Entry(frame, textvariable=self.batch_size_var, width=10, **entry_style)
        batch_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # N steps
        tk.Label(frame, text="N Steps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        n_steps_entry = tk.Entry(frame, textvariable=self.n_steps_var, width=10, **entry_style)
        n_steps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Device
        tk.Label(frame, text="Device:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        device_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        device_frame.grid(row=row, column=1, sticky='ew', padx=10, pady=5)
        
        for device in ['auto', 'cpu', 'cuda']:
            rb = tk.Radiobutton(
                device_frame, text=device, variable=self.device_var, value=device,
                bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
                selectcolor=COLORS['bg_medium'], 
                activebackground=COLORS['bg_dark'],
                activeforeground=COLORS['accent']
            )
            rb.pack(side=tk.LEFT, padx=10)
        
        row += 1
        
        # Checkbox para Auto-tuning - NUEVO
        auto_tune_check = tk.Checkbutton(
            frame, 
            text="Auto-tune Hyperparameters", 
            variable=self.auto_tune_var,
            bg=COLORS['bg_dark'],
            fg=COLORS['fg_light'],
            selectcolor=COLORS['bg_medium'],
            activebackground=COLORS['bg_dark'],
            activeforeground=COLORS['accent']
        )
        auto_tune_check.grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=5)
    
    def setup_env_tab(self, frame):
        """Configurar pestaña de entorno"""
        # Estilo para etiquetas
        label_style = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'font': ('Segoe UI', 10),
            'anchor': 'w'
        }
        
        # Estilo para entradas
        entry_style = {
            'bg': COLORS['bg_medium'],
            'fg': COLORS['fg_white'],
            'insertbackground': COLORS['fg_white'],
            'relief': tk.FLAT,
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': COLORS['border']
        }
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)
        
        # Campos de configuración de entorno
        row = 0
        
        # Balance inicial
        tk.Label(frame, text="Initial Balance:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        balance_entry = tk.Entry(frame, textvariable=self.balance_var, width=10, **entry_style)
        balance_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Window size
        tk.Label(frame, text="Window Size:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        window_entry = tk.Entry(frame, textvariable=self.window_size_var, width=10, **entry_style)
        window_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Comisión
        tk.Label(frame, text="Commission:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        commission_entry = tk.Entry(frame, textvariable=self.commission_var, width=10, **entry_style)
        commission_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Position size
        tk.Label(frame, text="Position Size:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        position_entry = tk.Entry(frame, textvariable=self.position_size_var, width=10, **entry_style)
        position_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Slippage
        tk.Label(frame, text="Slippage:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        slippage_entry = tk.Entry(frame, textvariable=self.slippage_var, width=10, **entry_style)
        slippage_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Stop loss
        tk.Label(frame, text="Stop Loss %:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        stop_entry = tk.Entry(frame, textvariable=self.stop_loss_var, width=10, **entry_style)
        stop_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Take profit
        tk.Label(frame, text="Take Profit %:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        profit_entry = tk.Entry(frame, textvariable=self.take_profit_var, width=10, **entry_style)
        profit_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
    
    def setup_reward_tab(self, frame):
        """Configurar pestaña de recompensas"""
        # Estilo para etiquetas
        label_style = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'font': ('Segoe UI', 10),
            'anchor': 'w'
        }
        
        # Estilo para entradas
        entry_style = {
            'bg': COLORS['bg_medium'],
            'fg': COLORS['fg_white'],
            'insertbackground': COLORS['fg_white'],
            'relief': tk.FLAT,
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': COLORS['border']
        }
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)
        
        row = 0
        
        # Factor de escalado de recompensa
        tk.Label(frame, text="Reward Scaling:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        reward_scaling_entry = tk.Entry(frame, textvariable=self.reward_scaling_var, width=10, **entry_style)
        reward_scaling_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Penalización por inactividad
        tk.Label(frame, text="Inactivity Penalty:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        inactivity_entry = tk.Entry(frame, textvariable=self.inactivity_penalty_var, width=10, **entry_style)
        inactivity_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Penalización por quiebra
        tk.Label(frame, text="Bankruptcy Penalty:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        bankruptcy_entry = tk.Entry(frame, textvariable=self.bankruptcy_penalty_var, width=10, **entry_style)
        bankruptcy_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Factor de drawdown
        tk.Label(frame, text="Drawdown Factor:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        drawdown_entry = tk.Entry(frame, textvariable=self.drawdown_factor_var, width=10, **entry_style)
        drawdown_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Bonificación por win rate
        tk.Label(frame, text="Win Rate Bonus:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        win_rate_entry = tk.Entry(frame, textvariable=self.win_rate_bonus_var, width=10, **entry_style)
        win_rate_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Bonus por eficiencia de capital
        tk.Label(frame, text="Capital Efficiency:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        capital_entry = tk.Entry(frame, textvariable=self.capital_efficiency_bonus_var, width=10, **entry_style)
        capital_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Factor de descuento temporal
        tk.Label(frame, text="Time Decay Factor:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        time_decay_entry = tk.Entry(frame, textvariable=self.time_decay_factor_var, width=10, **entry_style)
        time_decay_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Normalización de recompensas
        normalize_check = tk.Checkbutton(
            frame, 
            text="Normalize Rewards", 
            variable=self.normalize_rewards_var,
            bg=COLORS['bg_dark'],
            fg=COLORS['fg_light'],
            selectcolor=COLORS['bg_medium'],
            activebackground=COLORS['bg_dark'],
            activeforeground=COLORS['accent']
        )
        normalize_check.grid(row=row, column=2, columnspan=2, sticky='w', padx=10, pady=5)
    
    def setup_presets_tab(self, frame):
        """Configurar pestaña de presets"""
        # Añadir presets para diferentes estilos de trading
        preset_title = tk.Label(
            frame, 
            text="Presets de Trading",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent']
        )
        preset_title.pack(pady=(10, 15))
        
        # Frame para los botones de preset
        presets_buttons_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        presets_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_style = {
            'font': ('Segoe UI', 10),
            'relief': tk.FLAT,
            'bd': 0,
            'pady': 8,
            'padx': 15,
        }
        
        # Botón para preset de trading de alta frecuencia
        hft_button = tk.Button(
            presets_buttons_frame,
            text="Trading de Alta Frecuencia",
            command=lambda: self.apply_preset_and_update("hft"),
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            **button_style
        )
        hft_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Botón para preset de trading de posición
        position_button = tk.Button(
            presets_buttons_frame,
            text="Trading de Posición",
            command=lambda: self.apply_preset_and_update("position"),
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            **button_style
        )
        position_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Botón para preset de trading de tendencia
        trend_button = tk.Button(
            presets_buttons_frame,
            text="Trading de Tendencia",
            command=lambda: self.apply_preset_and_update("trend"),
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            **button_style
        )
        trend_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Clonar el widget de descripción del preset
        description_text = tk.Text(
            frame,
            height=10,
            wrap=tk.WORD,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_light'],
            font=('Segoe UI', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        description_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        description_text.insert(tk.END, "Seleccione un preset para ver su descripción y aplicar automáticamente los parámetros recomendados para ese estilo de trading.")
        
        # Asociar este widget con el original para que ambos se actualicen
        self.preset_description = description_text
    
    def apply_preset_and_update(self, preset_type):
        """Aplica el preset y actualiza todos los widgets del diálogo"""
        # Aplicar el preset en el panel
        self.config_panel.apply_preset(preset_type)
        
        # Actualizar todos los widgets del diálogo
        self.update_all_fields()
        
        # Cambiar a pestaña relevante para mostrar cambios
        if preset_type == "hft" or preset_type == "position" or preset_type == "trend":
            # Cambiar a la pestaña "Entorno" o "Recompensas" para ver los cambios
            self.notebook.select(1)  # Índice 1 = pestaña "Entorno"
        
        # Mensaje de confirmación
        messagebox.showinfo("Preset Aplicado", 
                          f"Se ha aplicado el preset de {preset_type.replace('hft', 'Trading de Alta Frecuencia').replace('position', 'Trading de Posición').replace('trend', 'Trading de Tendencia')}.\n\nLos valores han sido actualizados en todas las pestañas.")
    
    def update_all_fields(self):
        """Actualiza todos los campos del diálogo con los valores del panel"""
        # Algoritmo
        self.algo_var.set(self.config_panel.algo_var.get())
        self.timesteps_var.set(self.config_panel.timesteps_var.get())
        self.learning_rate_var.set(self.config_panel.learning_rate_var.get()) 
        self.batch_size_var.set(self.config_panel.batch_size_var.get())
        self.n_steps_var.set(self.config_panel.n_steps_var.get())
        self.device_var.set(self.config_panel.device_var.get())
        self.auto_tune_var.set(self.config_panel.auto_tune_var.get())
        
        # Entorno
        self.balance_var.set(self.config_panel.balance_var.get())
        self.commission_var.set(self.config_panel.commission_var.get())
        self.slippage_var.set(self.config_panel.slippage_var.get())
        self.window_size_var.set(self.config_panel.window_size_var.get())
        self.position_size_var.set(self.config_panel.position_size_var.get())
        self.stop_loss_var.set(self.config_panel.stop_loss_var.get())
        self.take_profit_var.set(self.config_panel.take_profit_var.get())
        
        # Recompensas
        self.reward_scaling_var.set(self.config_panel.reward_scaling_var.get())
        self.inactivity_penalty_var.set(self.config_panel.inactivity_penalty_var.get())
        self.bankruptcy_penalty_var.set(self.config_panel.bankruptcy_penalty_var.get())
        self.drawdown_factor_var.set(self.config_panel.drawdown_factor_var.get())
        self.win_rate_bonus_var.set(self.config_panel.win_rate_bonus_var.get())
        self.normalize_rewards_var.set(self.config_panel.normalize_rewards_var.get())
        self.capital_efficiency_bonus_var.set(self.config_panel.capital_efficiency_bonus_var.get())
        self.time_decay_factor_var.set(self.config_panel.time_decay_factor_var.get())
    
    def on_ok(self):
        # Sincronizar valores del diálogo con el panel de configuración
        # Algoritmo
        self.config_panel.algo_var.set(self.algo_var.get())
        self.config_panel.timesteps_var.set(self.timesteps_var.get())
        self.config_panel.learning_rate_var.set(self.learning_rate_var.get())
        self.config_panel.batch_size_var.set(self.batch_size_var.get())
        self.config_panel.n_steps_var.set(self.n_steps_var.get())
        self.config_panel.device_var.set(self.device_var.get())
        self.config_panel.auto_tune_var.set(self.auto_tune_var.get())
        
        # Entorno
        self.config_panel.balance_var.set(self.balance_var.get())
        self.config_panel.commission_var.set(self.commission_var.get())
        self.config_panel.slippage_var.set(self.slippage_var.get())
        self.config_panel.window_size_var.set(self.window_size_var.get())
        self.config_panel.position_size_var.set(self.position_size_var.get())
        self.config_panel.stop_loss_var.set(self.stop_loss_var.get())
        self.config_panel.take_profit_var.set(self.take_profit_var.get())
        
        # Recompensas
        self.config_panel.reward_scaling_var.set(self.reward_scaling_var.get())
        self.config_panel.inactivity_penalty_var.set(self.inactivity_penalty_var.get())
        self.config_panel.bankruptcy_penalty_var.set(self.bankruptcy_penalty_var.get())
        self.config_panel.drawdown_factor_var.set(self.drawdown_factor_var.get())
        self.config_panel.win_rate_bonus_var.set(self.win_rate_bonus_var.get())
        self.config_panel.normalize_rewards_var.set(self.normalize_rewards_var.get())
        self.config_panel.capital_efficiency_bonus_var.set(self.capital_efficiency_bonus_var.get())
        self.config_panel.time_decay_factor_var.set(self.time_decay_factor_var.get())
        
        # Validar los parámetros
        params = self.config_panel.get_training_params()
        if params is not None:
            self.destroy()

# ---------------------------------------------------------------------
# Panel de logs
# ---------------------------------------------------------------------
class LogPanel(tk.Frame):
    def __init__(self, parent, log_queue, height=6):
        super().__init__(parent, bg=COLORS['bg_dark'])
        self.log_queue = log_queue
        self.height = height
        self.setup_ui()
        self.update_logs()
    
    def setup_ui(self):
        # Frame de título
        title_frame = tk.Frame(self, bg=COLORS['bg_dark'])
        title_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Título
        self.title_label = tk.Label(
            title_frame, 
            text="System Logs",
            font=('Segoe UI', 12, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_dark'],
            anchor="w", padx=10, pady=5
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Botón para limpiar logs
        self.clear_button = tk.Button(
            title_frame,
            text="Clear",
            command=self.clear_logs,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 9),
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=2
        )
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Frame para separador
        separator = tk.Frame(self, height=1, bg=COLORS['border'])
        separator.pack(fill=tk.X, pady=(0,5))
        
        # Frame de texto
        self.log_text = scrolledtext.ScrolledText(
            self, 
            height=self.height,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_light'],
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        # Configurar colores de tag
        self.log_text.tag_configure('INFO', foreground='white')
        self.log_text.tag_configure('WARNING', foreground='#ffc107')
        self.log_text.tag_configure('ERROR', foreground='#d9534f')
        self.log_text.tag_configure('DEBUG', foreground='#6c757d')
        self.log_text.tag_configure('CRITICAL', foreground='#d9534f', font=('Consolas', 9, 'bold'))
    
    def clear_logs(self):
        """Limpia el área de logs"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
    
    def update_logs(self):
        """Actualiza los logs con los mensajes de la cola"""
        try:
            while True:
                record = self.log_queue.get_nowait()
                
                # Solo formatear si hay mensajes en la cola
                msg = self.format_record(record)
                
                # Insertar en el área de texto
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, msg + '\n', record.levelname)
                
                # Desplazar a la última línea
                self.log_text.see(tk.END)
                self.log_text.configure(state='disabled')
                
                # Marcar como procesado
                self.log_queue.task_done()
        except queue.Empty:
            # Si la cola está vacía, programar la próxima actualización
            self.after(100, self.update_logs)
    
    def format_record(self, record):
        """Formatea un registro de log para mostrar"""
        if hasattr(record, 'message'):
            message = record.message
        else:
            message = record.getMessage()
            
        # Formato timestamp hora:minuto:segundo
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Retornar formato simple sin mostrar nivel
        return f"[{timestamp}] {message}"

# ---------------------------------------------------------------------
# Diálogo de ayuda
# ---------------------------------------------------------------------
class HelpDialog(tk.Toplevel):
    def __init__(self, parent, title="RL Trading System - Help", help_text=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")
        self.minsize(600, 500)
        self.configure(background=COLORS['bg_very_dark'])
        self.transient(parent)
        self.grab_set()
        if os.path.exists("icon.ico"):
            self.iconbitmap("icon.ico")
        self.help_text = help_text
        self.setup_ui()

        # Centrar
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def load_help_file(self, file_name):
        """Carga el contenido de un archivo de ayuda"""
        # Buscar en múltiples ubicaciones potenciales
        try:
            # 1. Directorio raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 2. Carpeta doc
            doc_dir = os.path.join(base_dir, "doc")
            
            # Posibles rutas para el archivo
            possible_paths = [
                os.path.join(base_dir, file_name),
                os.path.join(doc_dir, file_name),
                os.path.join(os.getcwd(), file_name),
                os.path.join(os.getcwd(), "doc", file_name)
            ]
            
            # Intentar cada ruta
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        # Intentar con UTF-8 primero
                        with open(path, 'r', encoding='utf-8') as f:
                            logger.info(f"Archivo de ayuda cargado desde: {path}")
                            return f.read()
                    except UnicodeDecodeError:
                        # Si falla, intentar con latin-1
                        with open(path, 'r', encoding='latin-1') as f:
                            logger.info(f"Archivo de ayuda cargado desde: {path} (con codificación latin-1)")
                            return f.read()
            
            # Si no se encuentra, mostrar mensaje informativo
            logger.error(f"No se pudo encontrar el archivo de ayuda {file_name}")
            return f"Archivo de ayuda no encontrado: {file_name}\n\n" + \
                f"Se buscó en las siguientes rutas:\n" + \
                "\n".join(possible_paths)
                
        except Exception as e:
            logger.error(f"Error al cargar archivo de ayuda {file_name}: {e}")
            return f"Error loading help file: {e}"
    
    def setup_ui(self):
        # Frame principal
        self.main_frame = tk.Frame(self, bg=COLORS['bg_very_dark'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(
            self.main_frame,
            text="RL Trading System Help",
            bg=COLORS['bg_very_dark'],
            fg=COLORS['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(side=tk.TOP, anchor='w', pady=10)
        
        # Estilos para el texto de ayuda
        text_config = {
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'font': ('Segoe UI', 10),
            'relief': tk.FLAT,
            'bd': 0,
            'highlightthickness': 0,
            'padx': 10,
            'pady': 10
        }
        
        # Si se proporcionó un texto de ayuda específico, usarlo
        if self.help_text:
            help_frame = tk.Frame(self.main_frame, bg=COLORS['bg_dark'])
            help_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            help_text_widget = scrolledtext.ScrolledText(help_frame, **text_config)
            help_text_widget.pack(fill=tk.BOTH, expand=True)
            help_text_widget.insert(tk.END, self.help_text)
            help_text_widget.configure(state='disabled')
        else:
            # Crear notebook para pestañas de ayuda
            help_notebook = ttk.Notebook(self.main_frame)
            help_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Pestaña 1: Interfaz General
            interface_frame = tk.Frame(help_notebook, bg=COLORS['bg_dark'])
            interface_text = scrolledtext.ScrolledText(interface_frame, **text_config)
            interface_text.pack(fill=tk.BOTH, expand=True)
            interface_content = self.load_help_file("interface_help.txt")
            interface_text.insert(tk.END, interface_content)
            interface_text.configure(state='disabled')
            help_notebook.add(interface_frame, text="Interface")
            
            # Pestaña 2: Controles
            controls_frame = tk.Frame(help_notebook, bg=COLORS['bg_dark'])
            controls_text = scrolledtext.ScrolledText(controls_frame, **text_config)
            controls_text.pack(fill=tk.BOTH, expand=True)
            controls_content = self.load_help_file("controls_help.txt")
            controls_text.insert(tk.END, controls_content)
            controls_text.configure(state='disabled')
            help_notebook.add(controls_frame, text="Controls")
            
            # Pestaña 3: Gráficos
            charts_frame = tk.Frame(help_notebook, bg=COLORS['bg_dark'])
            charts_text = scrolledtext.ScrolledText(charts_frame, **text_config)
            charts_text.pack(fill=tk.BOTH, expand=True)
            charts_content = self.load_help_file("charts_help.txt")
            charts_text.insert(tk.END, charts_content)
            charts_text.configure(state='disabled')
            help_notebook.add(charts_frame, text="Charts")
            
            # Pestaña 4: Trading
            trading_frame = tk.Frame(help_notebook, bg=COLORS['bg_dark'])
            trading_text = scrolledtext.ScrolledText(trading_frame, **text_config)
            trading_text.pack(fill=tk.BOTH, expand=True)
            trading_content = self.load_help_file("trading_help.txt")
            trading_text.insert(tk.END, trading_content)
            trading_text.configure(state='disabled')
            help_notebook.add(trading_frame, text="Trading")

        close_button = tk.Button(
            self.main_frame,
            text="Close",
            command=self.destroy,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=5
        )
        close_button.pack(side=tk.BOTTOM, pady=10)