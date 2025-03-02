import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Asegurar que src está en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos del proyecto
try:
    from src.RLTradingSystemGUI import (
        COLORS, BasePanel, StatsPanel, TradesPanel, ChartPanel, 
        LogPanel, HelpDialog
    )
    from src.training_config import TrainingConfigPanel
    from src.control_panel import ControlPanel
    from src.TrainingManager import DataLoader, TrainingManager
    from src.RLTradingAgent import NinjaTraderInterface, RLAgent, MarketData
    from src.TradingEnvironment import TradingEnvironment
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    MODULES_AVAILABLE = False

# Configurar logging
logger = logging.getLogger("MainGUI")

class LoginFrame(tk.Frame):
    """Pantalla de inicio de sesión"""
    
    def __init__(self, parent, on_login_success):
        super().__init__(parent, bg=COLORS['bg_very_dark'])
        self.parent = parent
        self.on_login_success = on_login_success
        self.setup_ui()
        
    def setup_ui(self):
        # Logo o título
        logo_frame = tk.Frame(self, bg=COLORS['bg_very_dark'])
        logo_frame.pack(pady=(100, 50))
        
        title = tk.Label(
            logo_frame, 
            text="RL Trading System",
            font=('Segoe UI', 36, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_very_dark']
        )
        title.pack()
        
        subtitle = tk.Label(
            logo_frame, 
            text="Trading con Reinforcement Learning para NinjaTrader 8",
            font=('Segoe UI', 14),
            fg=COLORS['fg_light'],
            bg=COLORS['bg_very_dark']
        )
        subtitle.pack(pady=(10, 0))
        
        # Panel de inicio de sesión
        login_panel = tk.Frame(
            self, 
            bg=COLORS['bg_dark'],
            highlightbackground=COLORS['border'],
            highlightthickness=1,
            padx=40,
            pady=40
        )
        login_panel.pack(pady=30)
        
        # Estilo para etiquetas
        label_style = {
            'font': ('Segoe UI', 12),
            'bg': COLORS['bg_dark'],
            'fg': COLORS['fg_light'],
            'anchor': 'w'
        }
        
        # Estilo para entradas
        entry_style = {
            'font': ('Segoe UI', 12),
            'bg': COLORS['bg_medium'],
            'fg': COLORS['fg_white'],
            'insertbackground': COLORS['fg_white'],
            'relief': tk.FLAT,
            'bd': 0,
            'width': 25,
            'highlightthickness': 1,
            'highlightbackground': COLORS['border'],
            'selectbackground': COLORS['selected']
        }
        
        # Campo usuario
        tk.Label(login_panel, text="Usuario", **label_style).pack(anchor='w', pady=(0, 5))
        self.username_entry = tk.Entry(login_panel, **entry_style)
        self.username_entry.pack(pady=(0, 20))
        self.username_entry.insert(0, "trader")  # Valor predeterminado
        
        # Campo contraseña
        tk.Label(login_panel, text="Contraseña", **label_style).pack(anchor='w', pady=(0, 5))
        self.password_entry = tk.Entry(login_panel, show="•", **entry_style)
        self.password_entry.pack(pady=(0, 30))
        self.password_entry.insert(0, "jav92")  # Valor predeterminado
        
        # Botón iniciar sesión
        login_button = tk.Button(
            login_panel,
            text="Iniciar Sesión",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['accent'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['accent_hover'],
            activeforeground=COLORS['fg_white'],
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=10,
            command=self.validate_login
        )
        login_button.pack()
        
        # Vincular tecla Enter
        self.username_entry.bind('<Return>', lambda event: self.password_entry.focus())
        self.password_entry.bind('<Return>', lambda event: self.validate_login())
        
        # Enfoque inicial
        self.username_entry.focus()
    
    def validate_login(self):
        """Validar credenciales de inicio de sesión"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        expected_username = "trader"
        expected_password = "jav92"
        
        if username == expected_username and password == expected_password:
            self.animate_transition()
        else:
            messagebox.showerror("Error de inicio de sesión", "Usuario o contraseña incorrectos")
    
    def animate_transition(self):
        """Animar la transición de inicio de sesión a la aplicación principal"""
        # Obtener dimensiones de la ventana
        width = self.winfo_width()
        height = self.winfo_height()
        center_x = width // 2
        center_y = height // 2
        
        # Crear un canvas para la animación
        canvas = tk.Canvas(
            self, 
            width=width, 
            height=height, 
            bg=COLORS['bg_very_dark'],
            highlightthickness=0
        )
        canvas.place(x=0, y=0)
        
        # Crear un óvalo en el centro
        oval = canvas.create_oval(
            center_x - 5, center_y - 5, 
            center_x + 5, center_y + 5, 
            fill=COLORS['accent'], 
            outline=COLORS['accent']
        )
        
        # Variable para asegurar que el callback solo se ejecute una vez
        self._animation_finished = False
        
        # Función de animación
        def animate_zoom():
            nonlocal oval
            
            try:
                for i in range(30):  # 30 frames de animación
                    if not canvas.winfo_exists():  # Verificar si el canvas aún existe
                        return
                        
                    size = (i + 1) * 20  # Aumentar tamaño gradualmente
                    canvas.coords(
                        oval, 
                        center_x - size, center_y - size, 
                        center_x + size, center_y + size
                    )
                    
                    # Ajustar la opacidad
                    opacity = min(1.0, 0.5 + i * 0.02)
                    color = COLORS['accent']
                    canvas.itemconfig(oval, fill=color, outline=color)
                    
                    self.update()  # Actualizar la ventana
                    time.sleep(0.02)  # Pequeña pausa
                
                if not canvas.winfo_exists():  # Verificar si el canvas aún existe
                    return
                    
                # Llenar completamente la pantalla
                canvas.create_rectangle(0, 0, width, height, fill=COLORS['accent'], outline=COLORS['accent'])
                self.update()
                time.sleep(0.3)
                
                # Evitar múltiples llamadas al callback
                if not self._animation_finished:
                    self._animation_finished = True
                    # Llamar a la función de éxito después de la animación en el hilo principal
                    self.after(100, self.on_login_success)
            except Exception as e:
                print(f"Error en animación: {e}")
        
        # Iniciar animación en un hilo separado
        threading.Thread(target=animate_zoom, daemon=True).start()


class MainApplication(tk.Frame):
    """Aplicación principal después del inicio de sesión"""

    
    def __init__(self, parent):
        super().__init__(parent, bg=COLORS['bg_very_dark'])
        self.parent = parent
        
        # Estado del sistema
        self.running = False
        self.paused = False
        self.connected = False
        self.worker_thread = None
        self.nt_interface = None
        
        # Inicializar objetos de entrenamiento/trading
        self.initialize_objects()
        
        # Configurar la interfaz
        self.setup_ui()
    
    def toggle_pause(self, paused):
        """Pausa o reanuda el proceso actual"""
        self.paused = paused
        logger.info(f"Proceso {'pausado' if paused else 'reanudado'}")
        
        # Si estamos en modo de entrenamiento, podría necesitar lógica adicional
        if hasattr(self, 'training_manager') and self.training_manager:
            # Aquí podría ir lógica específica para pausar el entrenamiento
            pass
            
    def toggle_auto_trading(self):
        """Activa o desactiva el auto trading"""
        if not self.nt_interface:
            messagebox.showerror("Error", "Debe conectarse a NinjaTrader antes de activar el trading automático")
            # Revertir el cambio en el UI si no hay conexión
            self.control_panel.switch_var.set(False)
            self.control_panel.switch_canvas.itemconfig(self.control_panel.switch_bg, fill=COLORS['bg_medium'])
            self.control_panel.switch_canvas.coords(self.control_panel.switch_handle, 2, 2, 18, 18)
            return
            
        enabled = self.control_panel.switch_var.get()
        
        # Configurar el estado de auto trading en la interfaz
        self.nt_interface.set_auto_trading(enabled)
        
        logger.info(f"Auto Trading {'activado' if enabled else 'desactivado'}")
        
    def initialize_objects(self):
        """Inicializar objetos del sistema"""
        if MODULES_AVAILABLE:
            try:
                self.data_loader = DataLoader(min_bars=6000)
                self.training_manager = TrainingManager()
                self.market_data = MarketData()
                logger.info("Módulos de trading inicializados correctamente")
            except Exception as e:
                logger.error(f"Error al inicializar los módulos de trading: {e}")
                messagebox.showerror("Error", f"Error al inicializar los módulos de trading: {e}")
    
    def setup_ui(self):
        """Configurar la interfaz principal"""
        # Panel principal
        main_panel = tk.Frame(self, bg=COLORS['bg_very_dark'])
        main_panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Título
        title_label = tk.Label(
            main_panel,
            text="RL Trading System Dashboard",
            font=('Segoe UI', 22, 'bold'),
            fg=COLORS['accent'],
            bg=COLORS['bg_very_dark']
        )
        title_label.pack(side=tk.TOP, anchor='w', pady=10)
        
        # Contenedor de paneles (grid)
        self.panel_container = tk.Frame(main_panel, bg=COLORS['bg_very_dark'])
        self.panel_container.pack(fill=tk.BOTH, expand=True)
        
        self.panel_container.columnconfigure(0, weight=2)  # 2/3
        self.panel_container.columnconfigure(1, weight=1)  # 1/3
        self.panel_container.rowconfigure(0, weight=3)     # parte superior
        self.panel_container.rowconfigure(1, weight=1)     # parte inferior
        
        # Panel de configuración (no visible directamente)
        self.train_config_panel = TrainingConfigPanel(self)
        
        # Panel de gráficos (col=0, row=0)
        self.chart_panel = ChartPanel(self.panel_container)
        self.chart_panel.grid(row=0, column=0, sticky='nsew', padx=(0,5), pady=(0,5))
        
        # Panel derecho
        right_frame = tk.Frame(self.panel_container, bg=COLORS['bg_very_dark'])
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5,0), pady=(0,5))
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=3)
        
        # Panel de estadísticas (arriba a la derecha)
        self.stats_panel = StatsPanel(right_frame)
        self.stats_panel.grid(row=0, column=0, sticky='nsew', pady=(0,5))
        
        # Panel de operaciones (abajo a la derecha)
        self.trades_panel = TradesPanel(right_frame)
        self.trades_panel.grid(row=1, column=0, sticky='nsew')
        
        # Panel de control (parte inferior)
        self.control_panel = ControlPanel(
            self.panel_container,
            on_start=self.start_process,
            on_pause=self.toggle_pause,
            on_stop=self.stop_process,
            on_connect=self.connect_to_ninjatrader,
            on_disconnect=self.disconnect_from_ninjatrader,
            on_train_config=self.show_training_config
        )
        # Extender la funcionalidad para manejar el toggle de autotrading
        self.control_panel.on_switch_toggle = self.toggle_auto_trading
        self.control_panel.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(5,0))
        
        # Panel de logs (parte inferior de todo)
        from src.RLTradingSystemGUI import log_queue
        self.log_panel = LogPanel(main_panel, log_queue)
        self.log_panel.pack(fill=tk.X, expand=False, pady=(5,0))
        
        # Menú
        self.setup_menu()
        
        # Inicializar datos de simulación
        self.initialize_simulation()
        
    # The extract_data_from_ninjatrader method and cancel_extraction method
    # have been removed as data extraction is now handled by the NinjaTrader indicator
            
    def stop_process(self):
        """Detiene el proceso actual"""
        self.running = False
        logger.info("Deteniendo proceso...")
        
        # Si hay un hilo en ejecución, esperar a que termine
        if self.worker_thread and self.worker_thread.is_alive():
            # No bloquear la UI mientras esperamos
            pass
            
        logger.info("Proceso detenido correctamente")
        
    def connect_to_ninjatrader(self, ip, port):
        """Conecta con NinjaTrader"""
        try:
            # Si ya existe una interfaz, detenerla primero
            if self.nt_interface:
                self.nt_interface.stop()
                self.nt_interface = None
            
            logger.info(f"Conectando a NinjaTrader en {ip}:{port}...")
            
            # Importar las funciones del archivo run_trading_system
            import run_trading_system
            
            # Asegurarse de que no estamos en modo GUI para evitar bucles recursivos
            run_trading_system.gui_mode = False
            
            # Iniciar el servidor en segundo plano (siempre usar 127.0.0.1)
            server_started = run_trading_system.start_server_in_background(
                ip='127.0.0.1',  # Forzar a usar loopback para consistencia
                data_port=port,
                order_port=port + 1
            )
            
            # Obtener la referencia global al servidor
            self.nt_interface = run_trading_system.nt_interface
            
            if not server_started or not self.nt_interface:
                logger.error("Conexión inicial fallida")
                self.stats_panel.update_connection(False)
                messagebox.showerror("Error de Conexión", 
                                "No se pudo establecer conexión con NinjaTrader.\n\n"
                                "Posibles causas:\n"
                                "- NinjaTrader no está ejecutándose\n"
                                "- La Estrategia no está activa en NinjaTrader\n"
                                "- IP o puerto incorrectos")
                
                # Resetear el estado de los botones para permitir intentar conectar de nuevo
                self.control_panel.connect_button.configure(state='normal')
                self.control_panel.disconnect_button.configure(state='disabled')
                
                # Liberar recursos
                run_trading_system.stop_server()
                self.nt_interface = None
                return
            
            # Verificar la conexión después de un breve retraso
            self.parent.after(3000, lambda: self.check_connection_status(ip, port))
            
        except Exception as e:
            logger.error(f"Error en conexión: {e}")
            self.stats_panel.update_connection(False)
            messagebox.showerror("Error de Conexión", f"No se pudo conectar a NinjaTrader: {e}")
            
            # Resetear el estado de los botones para permitir intentar conectar de nuevo
            self.control_panel.connect_button.configure(state='normal')
            self.control_panel.disconnect_button.configure(state='disabled')
            
    def check_connection_status(self, ip, port):
        """Verificar el estado de la conexión después de un intento"""
        # Dar más tiempo para la conexión inicial (especialmente en sistemas lentos)
        retry_count = 0
        max_retries = 5
        
        # Función de verificación con reintentos
        def check_connection_with_retry():
            nonlocal retry_count
            
            if not self.nt_interface:
                return False
                
            # Verificar si la conexión está activa
            is_connected = self.nt_interface.is_connected()
            
            # Si no está conectado pero aún tenemos reintentos, programar otro intento
            if not is_connected and retry_count < max_retries:
                retry_count += 1
                logger.info(f"Esperando conexión... intento {retry_count}/{max_retries}")
                # Programar próximo intento en 2 segundos
                self.parent.after(2000, check_connection_with_retry)
                return None  # Resultado pendiente
                
            return is_connected
            
        # Iniciar verificación con reintentos
        connection_result = check_connection_with_retry()
        
        # Si el resultado es None, significa que estamos en proceso de reintentos
        if connection_result is None:
            return
            
        # Procesar el resultado final
        if connection_result:
            # La conexión fue exitosa
            self.connected = True
            self.stats_panel.update_connection(True)
            logger.info(f"Conexión exitosa con NinjaTrader en {ip}:{port}")
            messagebox.showinfo("Conexión", f"Conectado a NinjaTrader en {ip}:{port}")
            
            # Iniciar recolección de estadísticas
            self.update_stats_periodically()
        else:
            # La conexión falló o no está activa
            self.connected = False
            self.stats_panel.update_connection(False)
            
            messagebox.showerror("Error de Conexión", 
                            "No se pudo establecer conexión con NinjaTrader.\n\n"
                            "Posibles causas:\n"
                            "- NinjaTrader no está ejecutándose\n"
                            "- La Estrategia no está activa en NinjaTrader\n"
                            "- IP o puerto incorrectos")
            
            logger.error("No se pudo establecer conexión con NinjaTrader")
            
            # Detener la interfaz para limpiar recursos
            if self.nt_interface:
                self.nt_interface.stop()
                self.nt_interface = None
                
            # Resetear el estado de los botones para permitir intentar conectar de nuevo
            self.control_panel.connect_button.configure(state='normal')
            self.control_panel.disconnect_button.configure(state='disabled')
    
    def update_stats_periodically(self):
        """Actualiza las estadísticas de trading periódicamente"""
        if self.nt_interface and self.connected:
            # Obtener estadísticas del agente RL
            stats = self.nt_interface.rl_agent.get_trading_stats()
            
            # Actualizar panel de estadísticas
            self.stats_panel.update_stats(stats)
            
            # Programar la próxima actualización
            self.parent.after(1000, self.update_stats_periodically)  # Cada 1 segundo
            
    def disconnect_from_ninjatrader(self):
        """Desconecta de NinjaTrader"""
        try:
            if self.nt_interface:
                logger.info("Desconectando de NinjaTrader...")
                
                # Usar la función global para detener el servidor
                import run_trading_system
                run_trading_system.stop_server()
                
                self.nt_interface = None
                self.connected = False
                self.stats_panel.update_connection(False)
                logger.info("Desconexión exitosa de NinjaTrader")
                messagebox.showinfo("Desconexión", "Desconectado de NinjaTrader")
            else:
                logger.warning("No hay conexión activa con NinjaTrader")
        except Exception as e:
            logger.error(f"Error al desconectar: {e}")
            messagebox.showerror("Error de Desconexión", f"Error al desconectar de NinjaTrader: {e}")
            
    def setup_menu(self):
        """Configura el menú de la aplicación"""
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Importar datos...", command=self.import_data)
        file_menu.add_command(label="Exportar resultados...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.parent.destroy)
        
        # Menú Herramientas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Configuración de entrenamiento", command=self.show_training_config)
        tools_menu.add_command(label="Visualizar modelo", command=self.visualize_model)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Interfaz", command=lambda: self.show_help("interface"))
        help_menu.add_command(label="Gráficos", command=lambda: self.show_help("charts"))
        help_menu.add_command(label="Controles", command=lambda: self.show_help("controls"))
        help_menu.add_command(label="Trading", command=lambda: self.show_help("trading"))
        help_menu.add_command(label="Parámetros de Recompensa", command=lambda: self.show_help("rewards"))
        help_menu.add_separator()
        help_menu.add_command(label="Acerca de", command=self.show_about)
    
    def initialize_simulation(self):
        """Inicializa datos de simulación para la interfaz"""
        # Inicializar datos de precio simulados
        price_data = []
        for i in range(100):
            # Simular datos OHLC
            base = 100 + i * 0.1 + np.random.normal(0, 0.5)
            data_point = {
                'timestamp': datetime.now() - pd.Timedelta(minutes=100-i),
                'open': base,
                'high': base + abs(np.random.normal(0, 0.3)),
                'low': base - abs(np.random.normal(0, 0.3)),
                'close': base + np.random.normal(0, 0.2)
            }
            price_data.append(data_point)
        
        # Actualizar gráfico de precios
        self.chart_panel.update_price_chart(price_data)
        
        # Inicializar datos de rendimiento simulados
        for i in range(50):
            balance = 100000 + np.cumsum(np.random.normal(0, 100, i+1))[-1]
            timestamp = datetime.now() - pd.Timedelta(minutes=50-i)
            self.chart_panel.update_performance_chart(balance, timestamp)
        
        # Inicializar estadísticas
        stats = {
            'balance': 100000.0,
            'total_pnl': 0.0,
            'trades_count': 0,
            'win_rate': 0.0,
            'current_position': 'None'
        }
        self.stats_panel.update_stats(stats)
        
        # Inicializar operaciones simuladas
        trades = []
        self.trades_panel.update_trades(trades)
    
    # Métodos para el menú
    def import_data(self):
        """Importa datos desde un archivo"""
        file_path = filedialog.askopenfilename(
            title="Importar datos",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            # Lógica para importar datos
            logger.info(f"Importando datos desde: {file_path}")
            messagebox.showinfo("Importar", f"Datos importados correctamente: {file_path}")
    
    def export_results(self):
        """Exporta resultados a un archivo"""
        file_path = filedialog.asksaveasfilename(
            title="Exportar resultados",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
            defaultextension=".csv"
        )
        if file_path:
            # Lógica para exportar resultados
            logger.info(f"Exportando resultados a: {file_path}")
            messagebox.showinfo("Exportar", f"Resultados exportados correctamente: {file_path}")
    
    def show_training_config(self):
        """Muestra la ventana de configuración de entrenamiento"""
        # La ventana ya está creada, solo mostrarla
        self.train_config_panel.show()
    
    def visualize_model(self):
        """Visualiza la estructura del modelo"""
        model_path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[("Archivos de modelo", "*.zip"), ("Todos los archivos", "*.*")]
        )
        if model_path:
            messagebox.showinfo("Visualizar modelo", "Funcionalidad en desarrollo")
    
    def show_help(self, topic):
        """Muestra la ayuda según el tema seleccionado"""
        help_file = f"doc/{topic}_help.txt"
        if os.path.exists(help_file):
            try:
                # Intentar con UTF-8 primero
                with open(help_file, 'r', encoding='utf-8') as f:
                    help_text = f.read()
                help_dialog = HelpDialog(self, f"Ayuda - {topic.capitalize()}", help_text)
            except UnicodeDecodeError:
                # Si falla, intentar con latin-1 que es más permisivo
                try:
                    with open(help_file, 'r', encoding='latin-1') as f:
                        help_text = f.read()
                    help_dialog = HelpDialog(self, f"Ayuda - {topic.capitalize()}", help_text)
                except Exception as e:
                    messagebox.showerror("Error de lectura", f"No se pudo leer el archivo de ayuda: {e}")
        else:
            messagebox.showinfo("Ayuda", "Tema de ayuda no disponible")
    
    # En src/MainGUI.py - método show_about modificado
    def show_about(self):
        """Muestra información sobre la aplicación"""
        about_text = """RL Trading System v1.0
        
    Sistema de trading basado en Reinforcement Learning para NinjaTrader 8.

    Desarrollado por: Javier Lora
    Contacto: jvlora@hublai.com
    """
        messagebox.showinfo("Acerca de", about_text)
    
    def start_process(self, mode):
        """Inicia un proceso según el modo seleccionado"""
        if self.running:
            return
        
        self.running = True
        self.paused = False
        logger.info(f"Iniciando proceso en modo: {mode}")
        
        if mode == 'train':
            self.start_training()
        elif mode == 'backtest':
            self.start_backtesting()
        elif mode == 'server':
            self.start_server_mode()
    
    def start_training(self):
        """Inicia el entrenamiento de un modelo"""
        # Verificar si hay archivos de datos
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        
        if not data_files:
            messagebox.showerror("No Data Available", "No data files found. Please use the RLDataExtractor indicator in NinjaTrader to export data first.")
            self.running = False
            return
        
        # Ordenar por fecha (más reciente primero)
        data_files.sort(reverse=True)
        latest_data_file = os.path.join(data_dir, data_files[0])
        
        # Obtener parámetros
        params = self.train_config_panel.get_training_params()
        if params is None:
            self.running = False
            return
        
        # Crear función para entrenar en un hilo separado
        def training_thread():
            try:
                logger.info("Iniciando entrenamiento...")
                self.control_panel.start_button.configure(state='disabled')
                self.control_panel.pause_button.configure(state='normal')
                self.control_panel.stop_button.configure(state='normal')
                
                # Cargar datos del archivo extraído
                logger.info(f"Usando datos extraídos: {latest_data_file}")
                train_data, test_data = self.data_loader.prepare_train_test_data(csv_file=latest_data_file)
                
                logger.info(f"Datos cargados: {len(train_data)} registros de entrenamiento, {len(test_data)} de prueba")
                
                # Extraer auto-tuning explícitamente
                enable_auto_tuning = params.pop('enable_auto_tuning', True)
                logger.info(f"Auto-tuning of hyperparameters: {'Enabled' if enable_auto_tuning else 'Disabled'}")
                
                # Definir función de callback para actualizar la interfaz
                def update_training_ui(metrics):
                    # Actualizar interfaz en el hilo principal
                    self.parent.after(0, lambda: self.chart_panel.update_training_metrics(metrics))
                    
                    # Actualizar también la barra de progreso si hay total_timesteps disponible
                    if 'episode' in metrics and params.get('total_timesteps'):
                        progress = min(100, (metrics['episode'] * 100) / (params.get('total_timesteps') / 5000))
                        self.parent.after(0, lambda: self.control_panel.update_progress(progress))
                
                # Iniciar entrenamiento con el callback
                model, train_env = self.training_manager.train(
                    train_data, test_data,
                    training_callback=update_training_ui,
                    enable_auto_tuning=enable_auto_tuning,  # Pasar explícitamente
                    **params
                )
                
                # Verificar si se detuvo manualmente
                if not self.running:
                    logger.info("Entrenamiento detenido manualmente")
                    return
                
                # Realizar backtesting
                logger.info("Realizando backtesting...")
                vec_norm_path = os.path.join(self.training_manager.models_dir, "vec_normalize_final.pkl")
                performance_df = self.training_manager.backtest(model, test_data, vec_norm_path)
                
                # Mostrar resultados
                self.training_manager.plot_backtest_results(performance_df)
                
                logger.info("Entrenamiento completado exitosamente")
                
                # Actualizar UI
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
                
                # Mensaje al usuario
                messagebox.showinfo("Entrenamiento", "El entrenamiento ha finalizado correctamente")
                
                self.running = False
                
            except Exception as e:
                logger.error(f"Error en entrenamiento: {e}")
                messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")
                self.running = False
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
        
        # Iniciar en un hilo separado
        self.worker_thread = threading.Thread(target=training_thread)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def start_backtesting(self, csv_path=None):
        """Inicia el backtesting de un modelo"""
        # Si no se proporciona ruta de CSV, usar los datos más recientes
        if not csv_path:
            data_dir = "data"
            data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            
            if not data_files:
                messagebox.showerror("No Data Available", "No data files found. Please use the RLDataExtractor indicator in NinjaTrader to export data first.")
                self.running = False
                return
            
            # Verificar si debemos pedir al usuario seleccionar un archivo CSV
            should_select_file = len(data_files) > 1
            
            if should_select_file:
                file_path = filedialog.askopenfilename(
                    title="Seleccionar datos para backtesting",
                    filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
                    initialdir=data_dir
                )
                if not file_path:
                    self.running = False
                    return
                csv_path = file_path
            else:
                # Ordenar por fecha (más reciente primero)
                data_files.sort(reverse=True)
                csv_path = os.path.join(data_dir, data_files[0])
        
        # Obtener modelo
        model_path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[("Archivos de modelo", "*.zip"), ("Todos los archivos", "*.*")]
        )
        if not model_path:
            self.running = False
            return
        
        # Crear función para backtesting en un hilo separado
        def backtesting_thread():
            try:
                logger.info("Iniciando backtesting...")
                self.control_panel.start_button.configure(state='disabled')
                self.control_panel.pause_button.configure(state='normal')
                self.control_panel.stop_button.configure(state='normal')
                
                # Cargar datos
                _, test_data = self.data_loader.prepare_train_test_data(csv_file=csv_path, test_ratio=1.0)
                logger.info(f"Datos cargados: {len(test_data)} registros")
                
                # Cargar modelo
                model = self.training_manager.load_model(model_path)
                
                # Iniciar backtesting
                model_dir = os.path.dirname(model_path)
                vec_normalize_path = os.path.join(model_dir, "vec_normalize_final.pkl")
                if not os.path.exists(vec_normalize_path):
                    vec_normalize_path = None
                
                # Realizar backtesting
                performance_df = self.training_manager.backtest(model, test_data, vec_normalize_path)
                
                # Verificar si se detuvo manualmente
                if not self.running:
                    logger.info("Backtesting detenido manualmente")
                    return
                
                # Mostrar resultados
                self.training_manager.plot_backtest_results(performance_df)
                
                logger.info("Backtesting completado exitosamente")
                
                # Actualizar UI
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
                
                # Mensaje al usuario
                messagebox.showinfo("Backtesting", "El backtesting ha finalizado correctamente")
                
                self.running = False
                
            except Exception as e:
                logger.error(f"Error en backtesting: {e}")
                messagebox.showerror("Error", f"Error durante el backtesting: {e}")
                self.running = False
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
        
        # Iniciar en un hilo separado
        self.worker_thread = threading.Thread(target=backtesting_thread)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def start_server_mode(self):
        """Inicia el modo servidor para conectarse a NinjaTrader"""
        if not self.connected:
            messagebox.showwarning("No conectado", "Debe conectarse a NinjaTrader primero")
            self.running = False
            return
        
        # Preguntar si desea usar un modelo
        use_model = messagebox.askyesno("Modelo", "¿Desea cargar un modelo de RL para trading?")
        model_path = None
        
        if use_model:
            # Seleccionar modelo
            file_path = filedialog.askopenfilename(
                title="Seleccionar modelo",
                filetypes=[("Archivos de modelo", "*.zip"), ("Todos los archivos", "*.*")]
            )
            if file_path:
                model_path = file_path
        
        # Crear función para el modo servidor en un hilo separado
        def server_thread():
            try:
                logger.info("Iniciando modo servidor...")
                self.control_panel.start_button.configure(state='disabled')
                self.control_panel.pause_button.configure(state='normal')
                self.control_panel.stop_button.configure(state='normal')
                
                # Detener el servidor actual si existe
                import run_trading_system
                run_trading_system.stop_server()
                
                # Determinar ruta de normalización
                vec_normalize_path = None
                if model_path:
                    model_dir = os.path.dirname(model_path)
                    possible_vec_norm = os.path.join(model_dir, "vec_normalize_final.pkl")
                    if os.path.exists(possible_vec_norm):
                        vec_normalize_path = possible_vec_norm
                
                # Iniciar el servidor en segundo plano con el modelo
                run_trading_system.start_server_in_background(
                    ip=self.control_panel.ip_var.get(),
                    data_port=int(self.control_panel.data_port_var.get()),
                    order_port=int(self.control_panel.data_port_var.get()) + 1,
                    model_path=model_path,
                    vec_normalize_path=vec_normalize_path
                )
                
                # Obtener la referencia global al servidor
                self.nt_interface = run_trading_system.nt_interface
                
                # Bucle principal mientras se ejecuta
                while self.running:
                    # Verificar pausa
                    if self.paused:
                        time.sleep(0.5)
                        continue
                    
                    # Actualizar estadísticas si hay datos disponibles
                    if hasattr(self.nt_interface, 'market_data') and len(self.nt_interface.market_data.data) > 0:
                        # Actualizar gráfico de precios
                        recent_data = self.nt_interface.market_data.data.tail(100).copy()
                        
                        if 'timestamp' not in recent_data.columns:
                            recent_data['timestamp'] = pd.date_range(
                                end=pd.Timestamp.now(), 
                                periods=len(recent_data), 
                                freq='1min'
                            )
                        
                        price_data = []
                        for idx, row in recent_data.iterrows():
                            data_point = {
                                'timestamp': row['timestamp'] if 'timestamp' in row else pd.Timestamp.now(),
                                'open': row['open'] if 'open' in row else 0,
                                'high': row['high'] if 'high' in row else 0,
                                'low': row['low'] if 'low' in row else 0,
                                'close': row['close'] if 'close' in row else 0
                            }
                            price_data.append(data_point)
                        
                        self.chart_panel.update_price_chart(price_data)
                    
                    # Actualizar estadísticas si hay datos del agente
                    if hasattr(self.nt_interface, 'rl_agent'):
                        agent = self.nt_interface.rl_agent
                        
                        # Actualizar panel de estadísticas
                        stats = {
                            'balance': 100000.0 + agent.profit_loss,
                            'total_pnl': agent.profit_loss,
                            'trades_count': agent.trade_count,
                            'win_rate': agent.successful_trades / max(1, agent.trade_count),
                            'current_position': 'Long' if agent.current_position == 1 else 
                                               'Short' if agent.current_position == -1 else 'None'
                        }
                        self.stats_panel.update_stats(stats)
                        
                        # Actualizar gráfico de rendimiento
                        self.chart_panel.update_performance_chart(
                            100000.0 + agent.profit_loss, 
                            pd.Timestamp.now()
                        )
                    
                    time.sleep(1.0)
                
                # Detener interfaz al finalizar
                run_trading_system.stop_server()
                
                logger.info("Modo servidor finalizado")
                
                # Actualizar UI
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
                
            except Exception as e:
                logger.error(f"Error en modo servidor: {e}")
                messagebox.showerror("Error", f"Error en modo servidor: {e}")
                self.running = False
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
        
        # Iniciar en un hilo separado
        self.worker_thread = threading.Thread(target=server_thread)
        self.worker_thread.daemon = True
        self.worker_thread.start()


class Application(tk.Tk):
    """Aplicación principal"""
    
    def __init__(self):
        super().__init__()
        self.title("RL Trading System")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.configure(background=COLORS['bg_very_dark'])
        
        # Icono de la aplicación
        if os.path.exists("icon.ico"):
            self.iconbitmap("icon.ico")
        
        # Inicialmente mostrar pantalla de inicio de sesión
        self.show_login()
    
    def show_login(self):
        """Muestra la pantalla de inicio de sesión"""
        # Limpiar ventana
        for widget in self.winfo_children():
            widget.destroy()
        
        # Mostrar pantalla de inicio de sesión
        self.login_frame = LoginFrame(self, self.on_login_success)
        self.login_frame.pack(fill=tk.BOTH, expand=True)
    
    def on_login_success(self):
        """Función llamada cuando el inicio de sesión es exitoso"""
        # Limpiar ventana
        for widget in self.winfo_children():
            widget.destroy()
        
        # Mostrar aplicación principal
        self.main_app = MainApplication(self)
        self.main_app.pack(fill=tk.BOTH, expand=True)


# Punto de entrada principal
def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
