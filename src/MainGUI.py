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
        TrainingConfigPanel, ControlPanel, LogPanel, HelpDialog
    )
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
            on_train_config=self.show_training_config,
            on_extract_data=self.extract_data_from_ninjatrader
        )
        self.control_panel.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(5,0))
        
        # Panel de logs (parte inferior de todo)
        from src.RLTradingSystemGUI import log_queue
        self.log_panel = LogPanel(main_panel, log_queue)
        self.log_panel.pack(fill=tk.X, expand=False, pady=(5,0))
        
        # Menú
        self.setup_menu()
        
        # Inicializar datos de simulación
        self.initialize_simulation()
        
    def extract_data_from_ninjatrader(self):
        """Extrae datos históricos de NinjaTrader"""
        if not self.connected:
            messagebox.showwarning("No conectado", "Debe conectarse a NinjaTrader primero")
            return
        
        # Mostrar ventana de progreso
        progress_window = tk.Toplevel(self.parent)
        progress_window.title("Extrayendo Datos")
        progress_window.geometry("400x180")  # Aumentado para acomodar mejor el botón
        progress_window.resizable(False, False)
        progress_window.transient(self.parent)
        progress_window.grab_set()
        
        # Configurar contenido de la ventana
        frame = tk.Frame(progress_window, bg=COLORS['bg_dark'], padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Etiqueta de mensaje
        message_label = tk.Label(
            frame, 
            text="Extrayendo datos históricos de NinjaTrader...",
            font=('Segoe UI', 12),
            fg=COLORS['fg_white'],
            bg=COLORS['bg_dark']
        )
        message_label.pack(pady=(0, 10))
        
        # Barra de progreso
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            frame, 
            variable=progress_var,
            maximum=100.0,
            mode='determinate',
            length=360
        )
        progress_bar.pack(pady=(0, 10))
        
        # Etiqueta de progreso
        progress_text = tk.StringVar(value="0%")
        progress_label = tk.Label(
            frame,
            textvariable=progress_text,
            font=('Segoe UI', 10),
            fg=COLORS['fg_white'],
            bg=COLORS['bg_dark']
        )
        progress_label.pack(pady=(0, 10))
        
        # Botón de cancelar - MEJORADO
        cancel_button = tk.Button(
            frame,
            text="Cancelar",
            command=lambda: self.cancel_extraction(progress_window),
            bg=COLORS['red'],
            fg=COLORS['fg_white'],
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=8,
            width=15  # Ancho fijo para mejor visibilidad
        )
        cancel_button.pack(pady=(5, 0))
        
        # Callback para actualizar progreso
        def update_extraction_progress(current, total, percent, filename=None):
            if filename:
                # Extracción completa
                progress_var.set(100.0)
                progress_text.set("100% - Completado")
                message_label.config(text=f"Datos extraídos correctamente:")
                
                # Añadir etiqueta para mostrar el archivo
                file_label = tk.Label(
                    frame,
                    text=f"Guardado en: {filename}",
                    font=('Segoe UI', 9),
                    fg=COLORS['accent'],
                    bg=COLORS['bg_dark']
                )
                file_label.pack(pady=(5, 0))
                
                # Cambiar a botón cerrar
                cancel_button.config(
                    text="Cerrar",
                    command=progress_window.destroy,
                    bg=COLORS['bg_medium']
                )                
                # Mostrar mensaje en log
                logger.info(f"Datos históricos extraídos correctamente: {filename}")
            else:
                # Actualizar progreso
                progress_var.set(percent)
                progress_text.set(f"{percent:.1f}% ({current}/{total})")
        
        # Actualizar UI
        progress_window.update()
        
        # Iniciar proceso de extracción
        def extraction_thread():
            try:
                self.nt_interface.request_historical_data(callback=update_extraction_progress)
                
                # Esperar a que se complete la extracción
                while not self.nt_interface.extraction_complete and not self.nt_interface.is_extracting_data:
                    time.sleep(0.1)
                    
                # Esperar a que se complete la extracción
                while self.nt_interface.is_extracting_data:
                    time.sleep(0.1)
                
                # Si la ventana fue cerrada, asegurarse de que no intentemos actualizar
                if not progress_window.winfo_exists():
                    return
                
                # Verificar si se completó correctamente
                if not self.nt_interface.extraction_complete:
                    message_label.config(text="Error al extraer datos")
                    progress_text.set("Error")
                    
                    # Cambiar a botón cerrar
                    cancel_button.config(
                        text="Cerrar",
                        command=progress_window.destroy,
                        bg=COLORS['bg_medium']
                    )            
            except Exception as e:
                logger.error(f"Error en extracción de datos: {e}")
                
                # Si la ventana fue cerrada, asegurarse de que no intentemos actualizar
                if not progress_window.winfo_exists():
                    return
                
                message_label.config(text=f"Error: {str(e)}")
                progress_text.set("Error")
                
                # Cambiar a botón cerrar
                cancel_button.config(
                    text="Cerrar",
                    command=progress_window.destroy,
                    bg=COLORS['bg_medium']
                )        
        # Iniciar extracción en un hilo separado
        extraction_thread = threading.Thread(target=extraction_thread)
        extraction_thread.daemon = True
        extraction_thread.start()

    # Método nuevo para cancelar la extracción
    def cancel_extraction(self, progress_window):
        """Cancela la extracción de datos"""
        if self.nt_interface:
            self.nt_interface.cancel_extraction()
        progress_window.destroy()
            
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
            
            # Crear interfaz
            self.nt_interface = NinjaTraderInterface(
                server_ip=ip,
                data_port=port,
                order_port=port + 1
            )
            
            # Iniciar interfaz (esto inicia los sockets y threads de comunicación)
            self.nt_interface.start()
            
            self.connected = True
            self.stats_panel.update_connection(True)
            logger.info(f"Conexión exitosa con NinjaTrader en {ip}:{port}")
            messagebox.showinfo("Conexión", f"Conectado a NinjaTrader en {ip}:{port}")
            
        except Exception as e:
            logger.error(f"Error en conexión: {e}")
            self.stats_panel.update_connection(False)
            messagebox.showerror("Error de Conexión", f"No se pudo conectar a NinjaTrader: {e}")
            
    def disconnect_from_ninjatrader(self):
        """Desconecta de NinjaTrader"""
        try:
            if self.nt_interface:
                logger.info("Desconectando de NinjaTrader...")
                self.nt_interface.stop()
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
            with open(help_file, 'r') as f:
                help_text = f.read()
            help_dialog = HelpDialog(self, f"Ayuda - {topic.capitalize()}", help_text)
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
        # Verificar si hay datos extraídos
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        extracted_files = [f for f in os.listdir(data_dir) if f.startswith("extracted_data_") and f.endswith(".csv")]
        
        if not extracted_files:
            messagebox.showerror("No Data Available", "No extracted data found. Please extract data from NinjaTrader first.")
            self.running = False
            return
        
        # Ordenar por fecha (más reciente primero)
        extracted_files.sort(reverse=True)
        latest_data_file = os.path.join(data_dir, extracted_files[0])
        
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
    
    def start_backtesting(self):
        """Inicia el backtesting de un modelo"""
        # Verificar si hay datos extraídos
        data_dir = "data"
        extracted_files = [f for f in os.listdir(data_dir) if f.startswith("extracted_data_") and f.endswith(".csv")]
        
        if not extracted_files:
            messagebox.showerror("No Data Available", "No extracted data found. Please extract data from NinjaTrader first.")
            self.running = False
            return
        
        # Ordenar por fecha (más reciente primero)
        extracted_files.sort(reverse=True)
        latest_data_file = os.path.join(data_dir, extracted_files[0])
        
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
                _, test_data = self.data_loader.prepare_train_test_data(csv_file=latest_data_file, test_ratio=1.0)
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
                self.control_panel.start_button
                self.control_panel.start_button.configure(state='normal')
                self.control_panel.pause_button.configure(state='disabled')
                self.control_panel.stop_button.configure(state='disabled')
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
                
                # Reiniciar interfaz con el modelo
                if self.nt_interface:
                    self.nt_interface.stop()
                
                # Determinar ruta de normalización
                vec_normalize_path = None
                if model_path:
                    model_dir = os.path.dirname(model_path)
                    possible_vec_norm = os.path.join(model_dir, "vec_normalize_final.pkl")
                    if os.path.exists(possible_vec_norm):
                        vec_normalize_path = possible_vec_norm
                
                # Crear interfaz con NinjaTrader
                self.nt_interface = NinjaTraderInterface(
                    server_ip=self.control_panel.ip_var.get(),
                    data_port=int(self.control_panel.data_port_var.get()),
                    order_port=int(self.control_panel.data_port_var.get()) + 1,
                    model_path=model_path,
                    vec_normalize_path=vec_normalize_path
                )
                
                # Iniciar interfaz
                self.nt_interface.start()
                
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
                if self.nt_interface:
                    self.nt_interface.stop()
                
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
    
    def start_backtesting(self, csv_path=None):
        """Inicia el backtesting de un modelo"""
        if not csv_path:
            file_path = filedialog.askopenfilename(
                title="Seleccionar datos para backtesting",
                filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
            )
            if not file_path:
                self.running = False
                return
            csv_path = file_path
        
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
                
                # Reiniciar interfaz con el modelo
                if self.nt_interface:
                    self.nt_interface.stop()
                
                # Determinar ruta de normalización
                vec_normalize_path = None
                if model_path:
                    model_dir = os.path.dirname(model_path)
                    possible_vec_norm = os.path.join(model_dir, "vec_normalize_final.pkl")
                    if os.path.exists(possible_vec_norm):
                        vec_normalize_path = possible_vec_norm
                
                # Crear interfaz con NinjaTrader
                self.nt_interface = NinjaTraderInterface(
                    server_ip=self.control_panel.ip_var.get(),
                    data_port=int(self.control_panel.data_port_var.get()),
                    order_port=int(self.control_panel.data_port_var.get()) + 1,
                    model_path=model_path,
                    vec_normalize_path=vec_normalize_path
                )
                
                # Iniciar interfaz
                self.nt_interface.start()
                
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
                if self.nt_interface:
                    self.nt_interface.stop()
                
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
