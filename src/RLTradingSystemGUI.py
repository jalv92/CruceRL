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

# Configuración adicional para loggear a un archivo
file_handler = logging.FileHandler("rl_trading_system.log")
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
        
        # Generar eje x basado en el número de episodio
        episode = training_metrics.get('episode', len(self.training_data['episode_rewards']))
        episodes = list(range(1, episode + 1))
        
        # Actualizar gráficos
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
# Panel de configuración de entrenamiento
# ---------------------------------------------------------------------
class TrainingConfigPanel(BasePanel):
    def __init__(self, parent):
        super().__init__(parent, title="Training Configuration")
        self.setup_ui()
        
    def setup_ui(self):
        container = self.main_container
        
        # Usamos grid para organizar los controles
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=3)
        
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
        
        # Variables
        self.algo_var = tk.StringVar(value="PPO")
        self.timesteps_var = tk.StringVar(value="500000")
        self.learning_rate_var = tk.StringVar(value="0.0003")
        self.batch_size_var = tk.StringVar(value="64")
        self.n_steps_var = tk.StringVar(value="2048")
        self.device_var = tk.StringVar(value="auto")
        
        # Filas de configuración
        row = 0
        
        # Algoritmo
        tk.Label(container, text="Algorithm:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        algo_frame = tk.Frame(container, bg=COLORS['bg_dark'])
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
        tk.Label(container, text="Timesteps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        timesteps_entry = tk.Entry(container, textvariable=self.timesteps_var, width=10, **entry_style)
        timesteps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Learning rate
        tk.Label(container, text="Learning Rate:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        lr_entry = tk.Entry(container, textvariable=self.learning_rate_var, width=10, **entry_style)
        lr_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Batch size
        tk.Label(container, text="Batch Size:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        batch_entry = tk.Entry(container, textvariable=self.batch_size_var, width=10, **entry_style)
        batch_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # N steps
        tk.Label(container, text="N Steps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        n_steps_entry = tk.Entry(container, textvariable=self.n_steps_var, width=10, **entry_style)
        n_steps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Device
        tk.Label(container, text="Device:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        device_frame = tk.Frame(container, bg=COLORS['bg_dark'])
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
        
        # Separador
        separator = tk.Frame(container, height=1, bg=COLORS['border'])
        separator.grid(row=row, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        
        row += 1
        
        # Variables de entorno
        self.env_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        self.env_frame.grid(row=row, column=0, columnspan=2, sticky='nsew', padx=10, pady=5)
        
        self.env_frame.columnconfigure(0, weight=1)
        self.env_frame.columnconfigure(1, weight=1)
        self.env_frame.columnconfigure(2, weight=1)
        
        # Título
        env_title = tk.Label(
            self.env_frame, text="Environment Settings",
            bg=COLORS['bg_dark'], fg=COLORS['accent'],
            font=('Segoe UI', 10, 'bold')
        )
        env_title.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 5))
        
        # Variables de entorno
        self.balance_var = tk.StringVar(value="100000.0")
        self.commission_var = tk.StringVar(value="0.0001")
        self.slippage_var = tk.StringVar(value="0.0001")
        self.window_size_var = tk.StringVar(value="10")
        self.reward_scaling_var = tk.StringVar(value="0.01")
        self.position_size_var = tk.StringVar(value="0.1")
        self.stop_loss_var = tk.StringVar(value="0.02")
        self.take_profit_var = tk.StringVar(value="0.04")
        
        # Balance inicial
        tk.Label(self.env_frame, text="Initial Balance:", **label_style).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        balance_entry = tk.Entry(self.env_frame, textvariable=self.balance_var, width=10, **entry_style)
        balance_entry.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Comisión
        tk.Label(self.env_frame, text="Commission:", **label_style).grid(row=2, column=0, sticky='w', padx=10, pady=5)
        commission_entry = tk.Entry(self.env_frame, textvariable=self.commission_var, width=10, **entry_style)
        commission_entry.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        
        # Slippage
        tk.Label(self.env_frame, text="Slippage:", **label_style).grid(row=3, column=0, sticky='w', padx=10, pady=5)
        slippage_entry = tk.Entry(self.env_frame, textvariable=self.slippage_var, width=10, **entry_style)
        slippage_entry.grid(row=3, column=1, sticky='w', padx=10, pady=5)
        
        # Window size
        tk.Label(self.env_frame, text="Window Size:", **label_style).grid(row=1, column=2, sticky='w', padx=10, pady=5)
        window_entry = tk.Entry(self.env_frame, textvariable=self.window_size_var, width=10, **entry_style)
        window_entry.grid(row=1, column=3, sticky='w', padx=10, pady=5)
        
        # Reward scaling
        tk.Label(self.env_frame, text="Reward Scaling:", **label_style).grid(row=2, column=2, sticky='w', padx=10, pady=5)
        reward_entry = tk.Entry(self.env_frame, textvariable=self.reward_scaling_var, width=10, **entry_style)
        reward_entry.grid(row=2, column=3, sticky='w', padx=10, pady=5)
        
        # Position size
        tk.Label(self.env_frame, text="Position Size:", **label_style).grid(row=3, column=2, sticky='w', padx=10, pady=5)
        position_entry = tk.Entry(self.env_frame, textvariable=self.position_size_var, width=10, **entry_style)
        position_entry.grid(row=3, column=3, sticky='w', padx=10, pady=5)
        
        # Stop loss
        tk.Label(self.env_frame, text="Stop Loss %:", **label_style).grid(row=4, column=0, sticky='w', padx=10, pady=5)
        stop_entry = tk.Entry(self.env_frame, textvariable=self.stop_loss_var, width=10, **entry_style)
        stop_entry.grid(row=4, column=1, sticky='w', padx=10, pady=5)
        
        # Take profit
        tk.Label(self.env_frame, text="Take Profit %:", **label_style).grid(row=4, column=2, sticky='w', padx=10, pady=5)
        profit_entry = tk.Entry(self.env_frame, textvariable=self.take_profit_var, width=10, **entry_style)
        profit_entry.grid(row=4, column=3, sticky='w', padx=10, pady=5)
    
    def show(self):
        """Muestra el panel de configuración en una ventana separada"""
        self.dialog = TrainingConfigDialog(self.winfo_toplevel(), self)
    
    def get_training_params(self):
        """Obtener los parámetros de entrenamiento como dict"""
        try:
            return {
                'algo': self.algo_var.get(),
                'total_timesteps': int(self.timesteps_var.get()),
                'learning_rate': float(self.learning_rate_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'n_steps': int(self.n_steps_var.get()),
                'device': self.device_var.get(),
                'initial_balance': float(self.balance_var.get()),
                'commission': float(self.commission_var.get()),
                'slippage': float(self.slippage_var.get()),
                'window_size': int(self.window_size_var.get()),
                'reward_scaling': float(self.reward_scaling_var.get()),
                'position_size': float(self.position_size_var.get()),
                'stop_loss_pct': float(self.stop_loss_var.get()),
                'take_profit_pct': float(self.take_profit_var.get())
            }
        except ValueError as e:
            messagebox.showerror("Invalid parameters", f"Invalid value: {e}")
            return None

# ---------------------------------------------------------------------
# Panel de logs
# ---------------------------------------------------------------------
class LogPanel(BasePanel):
    def __init__(self, parent, log_queue):
        super().__init__(parent, title="System Logs")
        self.log_queue = log_queue
        self.running = False
        self.setup_ui()
        self.start_log_monitor()
    
    def setup_ui(self):
        self.log_text = scrolledtext.ScrolledText(
            self.main_container,
            wrap=tk.WORD,
            height=8,
            font=('Consolas', 9),
            bg=COLORS['bg_very_dark'],
            fg=COLORS['fg_light'],
            insertbackground=COLORS['fg_light'],
            selectbackground=COLORS['selected'],
            selectforeground=COLORS['fg_white'],
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state='disabled')

        self.log_text.tag_configure('INFO', foreground=COLORS['fg_light'])
        self.log_text.tag_configure('DEBUG', foreground=COLORS['accent'])
        self.log_text.tag_configure('WARNING', foreground=COLORS['yellow'])
        self.log_text.tag_configure('ERROR', foreground=COLORS['red'])
        self.log_text.tag_configure('CRITICAL', background=COLORS['red'], foreground=COLORS['fg_white'])

    def start_log_monitor(self):
        self.running = True
        self.log_thread = threading.Thread(target=self.process_log_queue)
        self.log_thread.daemon = True
        self.log_thread.start()

    def process_log_queue(self):
        while self.running:
            try:
                record = self.log_queue.get(block=True, timeout=0.1)
                self.add_log(record)
            except queue.Empty:
                pass

    def add_log(self, record):
        log_time = time.strftime('%H:%M:%S', time.localtime(record.created))
        log_msg = f"[{log_time}] {record.levelname}: {record.getMessage()}\n"
        level_tag = record.levelname
        self.after(0, self._update_log, log_msg, level_tag)

    def _update_log(self, message, level_tag):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message, level_tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def stop(self):
        self.running = False
        if hasattr(self, 'log_thread') and self.log_thread.is_alive():
            self.log_thread.join()

# ---------------------------------------------------------------------
# Panel de control
# ---------------------------------------------------------------------
class ControlPanel(BasePanel):
    def __init__(self, parent, on_start, on_pause, on_stop, on_connect, on_train_config, on_extract_data):
        super().__init__(parent, title="Control Panel")
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_stop = on_stop
        self.on_connect = on_connect
        self.on_train_config = on_train_config
        self.on_extract_data = on_extract_data
        self.paused = False
        self.progress_value = 0
        self.setup_ui()
    
    def setup_ui(self):
        container = self.main_container

        # Frame para la conexión
        server_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        server_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            server_frame, text="Server IP:",
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0,5), pady=5)

        self.ip_var = tk.StringVar(value="127.0.0.1")
        ip_entry = tk.Entry(
            server_frame, textvariable=self.ip_var,
            bg=COLORS['bg_medium'], fg=COLORS['fg_white'],
            insertbackground=COLORS['fg_white'], relief=tk.FLAT,
            bd=0, width=15, highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        ip_entry.pack(side=tk.LEFT, padx=(0,10), pady=5)

        tk.Label(
            server_frame, text="Data Port:",
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0,5), pady=5)

        self.data_port_var = tk.StringVar(value="5000")
        data_port_entry = tk.Entry(
            server_frame, textvariable=self.data_port_var,
            bg=COLORS['bg_medium'], fg=COLORS['fg_white'],
            insertbackground=COLORS['fg_white'], relief=tk.FLAT,
            bd=0, width=6, highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        data_port_entry.pack(side=tk.LEFT, padx=(0,10), pady=5)

        self.connect_button = tk.Button(
            server_frame, text="Connect",
            command=self._on_connect_click,
            bg=COLORS['accent'], fg=COLORS['fg_white'],
            activebackground=COLORS['accent_hover'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT, bd=0, padx=10
        )
        self.connect_button.pack(side=tk.LEFT, padx=(0,0), pady=5)

        # Nuevo botón para extraer datos
        self.extract_data_button = tk.Button(
            server_frame, text="Extract Data",
            command=self._on_extract_data_click,
            bg=COLORS['bg_medium'], fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT, bd=0, padx=10
        )
        self.extract_data_button.pack(side=tk.LEFT, padx=(10,0), pady=5)

        # Frame para el modo de operación
        mode_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        mode_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            mode_frame, text="Operation Mode:",
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="server")
        mode_options = ['train', 'backtest', 'server']
        
        # Usar ttk.Combobox para el menú desplegable
        self.mode_menu = ttk.Combobox(
            mode_frame, 
            textvariable=self.mode_var,
            values=mode_options,
            state="readonly",
            width=10
        )
        self.mode_menu.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón de configuración de entrenamiento
        self.train_config_button = tk.Button(
            mode_frame, text="Configure Training",
            command=self._on_train_config_click,
            bg=COLORS['bg_medium'], fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT, bd=0, padx=10
        )
        self.train_config_button.pack(side=tk.LEFT, padx=(10,0), pady=5)

        # Barra de progreso
        progress_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_canvas = tk.Canvas(
            progress_frame, height=20,
            bg=COLORS['bg_medium'],
            highlightthickness=0, bd=0
        )
        self.progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))

        self.progress_text = tk.StringVar(value="0%")
        tk.Label(
            progress_frame, textvariable=self.progress_text,
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0,5))

        self.progress_rect = self.progress_canvas.create_rectangle(0, 0, 0, 20, fill=COLORS['accent'], width=0)

        # Frame de botones
        buttons_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        buttons_frame.pack(fill=tk.X, pady=10)

        for i in range(4):
            buttons_frame.columnconfigure(i, weight=1)

        self.start_button = tk.Button(
            buttons_frame, text="Start",
            command=self.on_start_click,
            bg=COLORS['green'], fg=COLORS['fg_white'],
            activebackground="#7ebe77",
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT, bd=0, padx=10, pady=5
        )
        self.start_button.grid(row=0, column=0, padx=5, sticky='ew')

        self.pause_button = tk.Button(
            buttons_frame, text="Pause",
            command=self.on_pause_click,
            bg=COLORS['yellow'], fg=COLORS['bg_very_dark'],
            activebackground="#ffdb4a",
            activeforeground=COLORS['bg_very_dark'],
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT, bd=0, padx=10, pady=5,
            state='disabled'
        )
        self.pause_button.grid(row=0, column=1, padx=5, sticky='ew')

        self.stop_button = tk.Button(
            buttons_frame, text="Stop",
            command=self.on_stop_click,
            bg=COLORS['red'], fg=COLORS['fg_white'],
            activebackground="#ff6b68",
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT, bd=0, padx=10, pady=5,
            state='disabled'
        )
        self.stop_button.grid(row=0, column=2, padx=5, sticky='ew')

        # Auto trading
        auto_frame = tk.Frame(buttons_frame, bg=COLORS['bg_dark'])
        auto_frame.grid(row=0, column=3, padx=5, sticky='ew')

        tk.Label(
            auto_frame, text="Auto Trading:",
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0,5))

        self.switch_var = tk.BooleanVar(value=False)
        self.switch_canvas = tk.Canvas(
            auto_frame, width=40, height=20,
            bg=COLORS['bg_medium'], highlightthickness=0, bd=0
        )
        self.switch_canvas.pack(side=tk.LEFT)

        self.switch_bg = self.switch_canvas.create_rectangle(
            0, 0, 40, 20,
            fill=COLORS['bg_medium'], width=0, outline=""
        )
        self.switch_handle = self.switch_canvas.create_oval(
            2, 2, 18, 18,
            fill=COLORS['bg_light'], width=0, outline=""
        )

        self.switch_canvas.bind("<Button-1>", self.toggle_switch)
        
        # Evento para cambio de modo
        self.mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)
        
        # Actualizar visibilidad de botones según el modo inicial
        self._update_button_visibility()

    def _on_connect_click(self):
        ip = self.ip_var.get()
        try:
            data_port = int(self.data_port_var.get())
        except ValueError:
            messagebox.showerror("Invalid Port", "Data port must be a valid number.")
            return
        # Corregir esta línea para pasar los argumentos
        self.on_connect(ip, data_port)  # Añadir los parámetros ip y data_port
    
    def _on_train_config_click(self):
        self.on_train_config()
    
    def _on_extract_data_click(self):
        self.on_extract_data()
    
    def _on_mode_change(self, event=None):
        self._update_button_visibility()
    
    def _update_button_visibility(self):
        mode = self.mode_var.get()
        
        # Mostrar/ocultar botones según el modo
        if mode == "train" or mode == "backtest":
            self.train_config_button.config(state="normal")
            self.extract_data_button.config(state="normal")
        else:
            self.train_config_button.config(state="disabled")
            # Dejar el botón de extraer datos siempre disponible
            self.extract_data_button.config(state="normal")

    def on_start_click(self):
        mode = self.mode_var.get()
        self.on_start(mode)
        self.start_button.configure(state='disabled')
        self.pause_button.configure(state='normal')
        self.stop_button.configure(state='normal')

    def on_pause_click(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.configure(text="Resume")
        else:
            self.pause_button.configure(text="Pause")
        self.on_pause(self.paused)

    def on_stop_click(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the process?"):
            self.on_stop()
            self.start_button.configure(state='normal')
            self.pause_button.configure(state='disabled')
            self.stop_button.configure(state='disabled')
            self.paused = False
            self.pause_button.configure(text="Pause")

    def on_switch_toggle(self):
        state = "ON" if self.switch_var.get() else "OFF"
        logger.info(f"Auto Trading switched {state}")
    
    def toggle_switch(self, event):
        if self.switch_var.get():
            # Actualmente está ON, lo apagamos
            self.switch_var.set(False)
            self.switch_canvas.itemconfig(self.switch_bg, fill=COLORS['bg_medium'])
            self.switch_canvas.coords(self.switch_handle, 2, 2, 18, 18)
        else:
            # Actualmente está OFF, lo encendemos
            self.switch_var.set(True)
            self.switch_canvas.itemconfig(self.switch_bg, fill=COLORS['accent'])
            self.switch_canvas.coords(self.switch_handle, 22, 2, 38, 18)

            # Luego llamamos a on_switch_toggle para registrar en logs, etc.
            self.on_switch_toggle()

    def update_progress(self, value):
        self.progress_value = value
        self.progress_text.set(f"{int(value)}%")
        width = self.progress_canvas.winfo_width()
        progress_width = (value / 100) * width
        self.progress_canvas.coords(self.progress_rect, 0, 0, progress_width, 20)

    def reset_progress(self):
        self.progress_value = 0
        self.progress_text.set("0%")
        self.progress_canvas.coords(self.progress_rect, 0, 0, 0, 20)

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
                    with open(path, 'r', encoding='utf-8') as f:
                        logger.info(f"Archivo de ayuda cargado desde: {path}")
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
        main_frame = tk.Frame(self, bg=COLORS['bg_very_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(
            main_frame,
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
            help_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
            help_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            help_text_widget = scrolledtext.ScrolledText(help_frame, **text_config)
            help_text_widget.pack(fill=tk.BOTH, expand=True)
            help_text_widget.insert(tk.END, self.help_text)
            help_text_widget.configure(state='disabled')
        else:
            # Crear notebook para pestañas de ayuda
            help_notebook = ttk.Notebook(main_frame)
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
            main_frame,
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

# ---------------------------------------------------------------------
# Diálogo de configuración de entrenamiento
# ---------------------------------------------------------------------
class TrainingConfigDialog(tk.Toplevel):
    def __init__(self, parent, config_panel):
        super().__init__(parent)
        self.parent = parent
        self.config_panel = config_panel
        self.title("Training Configuration")
        self.geometry("600x500")
        self.minsize(500, 400)
        self.configure(background=COLORS['bg_dark'])
        self.transient(parent)
        self.grab_set()
        
        if os.path.exists("icon.ico"):
            self.iconbitmap("icon.ico")
            
        # Añadir el panel al diálogo
        self.config_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botones en la parte inferior
        button_frame = tk.Frame(self, bg=COLORS['bg_dark'])
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
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
    
    def on_ok(self):
        # Validar los parámetros
        params = self.config_panel.get_training_params()
        if params is not None:
            self.destroy()