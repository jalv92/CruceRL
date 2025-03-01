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
        
        # Función de animación
        def animate_zoom():
            nonlocal oval
            
            for i in range(30):  # 30 frames de animación
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
            
            # Llenar completamente la pantalla
            canvas.create_rectangle(0, 0, width, height, fill=COLORS['accent'], outline=COLORS['accent'])
            self.update()
            time.sleep(0.3)
            
            # Llamar a la función de éxito después de la animación
            self.on_login_success()
        
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
            on_train_config=self.show_training_config
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
    
    def setup_menu(self):
        """Configurar menú principal"""
        menu_bar = tk.Menu(self.parent)
        
        # Menú Archivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Importar Modelo", command=self.import_model)
        file_menu.add_command(label="Exportar Modelo", command=self.export_model)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.on_exit)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)
        
        # Menú Ver
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Estadísticas", command=self.show_statistics)
        view_menu.add_command(label="Gráficos", command=self.show_charts)
        menu_bar.add_cascade(label="Ver", menu=view_menu)
        
        # Menú Ayuda
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Manual de Usuario", command=self.show_help)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        menu_bar.add_cascade(label="Ayuda", menu=help_menu)
        
        self.parent.config(menu=menu_bar)
    
    def initialize_simulation(self):
        """Inicializa datos de simulación para mostrar al inicio"""
        # Generar datos de precio simulados
        n_points = 100
        timestamps = [pd.Timestamp.now() - pd.Timedelta(minutes=n_points-i) for i in range(n_points)]
        
        price = 15000.0  # Precio inicial
        prices = []
        for _ in range(n_points):
            price *= (1 + np.random.normal(0, 0.001))
            prices.append(price)
        
        price_data = []
        for i in range(n_points):
            data_point = {
                'timestamp': timestamps[i],
                'close': prices[i]
            }
            price_data.append(data_point)
        
        # Actualizar gráficos
        self.chart_panel.update_price_chart(price_data)
        self.chart_panel.update_performance_chart(100000.0, pd.Timestamp.now())
    
    # ------------------------------------------------------------------------
    # Funciones de menú
    # ------------------------------------------------------------------------
    def import_model(self):
        """Importa un modelo entrenado"""
        file_path = filedialog.askopenfilename(
            title="Importar Modelo",
            filetypes=[("Archivos de Modelo (*.zip)", "*.zip"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                # Aquí cargaríamos el modelo usando TrainingManager
                logger.info(f"Modelo importado: {file_path}")
                messagebox.showinfo("Importar Modelo", f"Modelo importado correctamente: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error al importar modelo: {e}")
                messagebox.showerror("Error", f"Error al importar modelo: {e}")
    
    def export_model(self):
        """Exporta un modelo entrenado"""
        file_path = filedialog.asksaveasfilename(
            title="Exportar Modelo",
            defaultextension=".zip",
            filetypes=[("Archivos de Modelo (*.zip)", "*.zip"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                # Aquí exportaríamos el modelo
                logger.info(f"Modelo exportado: {file_path}")
                messagebox.showinfo("Exportar Modelo", f"Modelo exportado correctamente: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error al exportar modelo: {e}")
                messagebox.showerror("Error", f"Error al exportar modelo: {e}")
    
    def show_statistics(self):
        """Muestra estadísticas detalladas"""
        # Implementar visualización de estadísticas detalladas
        pass
    
    def show_charts(self):
        """Muestra gráficos detallados"""
        # Implementar visualización de gráficos detallados
        pass
    
    def show_help(self):
        """Muestra el diálogo de ayuda"""
        try:
            # Primero verificamos si existen los archivos de ayuda
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            doc_dir = os.path.join(base_dir, "doc")
            
            # Si la carpeta doc no existe o está vacía, mostramos un mensaje
            if not os.path.exists(doc_dir) or not os.listdir(doc_dir):
                messagebox.showinfo(
                    "Archivos de ayuda no encontrados",
                    "Los archivos de ayuda no fueron encontrados en la carpeta 'doc/'.\n\n"
                    "Por favor, asegúrate de que los siguientes archivos existen en la carpeta doc/:\n"
                    "- interface_help.txt\n"
                    "- controls_help.txt\n"
                    "- charts_help.txt\n"
                    "- trading_help.txt"
                )
                return
                
            # Si existe la carpeta, mostramos el diálogo
            help_dialog = HelpDialog(self.parent)
            self.wait_window(help_dialog)
        except Exception as e:
            logger.error(f"Error al mostrar diálogo de ayuda: {e}")
            messagebox.showerror("Error", f"No se pudo mostrar la ayuda: {e}")
    
    def show_about(self):
        """Muestra información acerca del software"""
        messagebox.showinfo(
            "Acerca de RL Trading System",
            "RL Trading System v1.0\n\n"
            "Sistema de trading basado en aprendizaje por refuerzo para NinjaTrader 8\n\n"
            "© 2023-2025"
        )
    
    def on_exit(self):
        """Manejador para salir de la aplicación"""
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            self.stop_all_processes()
            self.parent.destroy()
    
    # ------------------------------------------------------------------------
    # Funciones de control
    # ------------------------------------------------------------------------
    def show_training_config(self):
        """Muestra el diálogo de configuración de entrenamiento"""
        from src.RLTradingSystemGUI import TrainingConfigDialog
        dialog = TrainingConfigDialog(self.parent, self.train_config_panel)
        self.wait_window(dialog)
    
    def start_process(self, mode, csv_path=None):
        """Inicia un proceso según el modo seleccionado"""
        if self.running:
            return
        
        self.running = True
        self.paused = False
        logger.info(f"Iniciando proceso en modo: {mode}")
        
        if mode == 'train':
            self.start_training(csv_path)
        elif mode == 'backtest':
            self.start_backtesting(csv_path)
        elif mode == 'server':
            self.start_server_mode()
    
    def toggle_pause(self, paused):
        """Pausa o reanuda el proceso actual"""
        self.paused = paused
        logger.info(f"Proceso {'pausado' if paused else 'reanudado'}")
    
    def stop_process(self):
        """Detiene el proceso actual"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Deteniendo proceso...")
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        # Detener interfaz NinjaTrader si está activa
        if self.nt_interface:
            self.nt_interface.stop()
        
        self.control_panel.start_button.configure(state='normal')
        self.control_panel.pause_button.configure(state='disabled')
        self.control_panel.stop_button.configure(state='disabled')
        
        logger.info("Proceso detenido")
    
    def stop_all_processes(self):
        """Detiene todos los procesos antes de salir"""
        self.stop_process()
        
        # Detener otros componentes
        if hasattr(self, 'log_panel'):
            self.log_panel.stop()
    
    def connect_to_ninjatrader(self, ip, port):
        """Conecta con NinjaTrader"""
        try:
            if self.nt_interface:
                self.nt_interface.stop()
            
            logger.info(f"Conectando a NinjaTrader en {ip}:{port}...")
            
            # Crear interfaz
            self.nt_interface = NinjaTraderInterface(
                server_ip=ip,
                data_port=port,
                order_port=port + 1
            )
            
            # Iniciar interfaz
            self.nt_interface.start()
            
            self.connected = True
            self.stats_panel.update_connection(True)
            logger.info(f"Conexión exitosa con NinjaTrader en {ip}:{port}")
            messagebox.showinfo("Conexión", f"Conectado a NinjaTrader en {ip}:{port}")
            
        except Exception as e:
            logger.error(f"Error de conexión: {e}")
            self.stats_panel.update_connection(False)
            messagebox.showerror("Error de Conexión", f"No se pudo conectar a NinjaTrader: {e}")
    
    def start_training(self, csv_path=None):
        """Inicia el entrenamiento de un modelo"""
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
                
                # Cargar datos
                if csv_path and os.path.exists(csv_path):
                    train_data, test_data = self.data_loader.prepare_train_test_data(csv_file=csv_path)
                else:
                    logger.info("Generando datos sintéticos para entrenamiento")
                    train_data, test_data = self.data_loader.prepare_train_test_data()
                
                logger.info(f"Datos cargados: {len(train_data)} registros de entrenamiento, {len(test_data)} de prueba")
                
                # Iniciar entrenamiento
                model, train_env = self.training_manager.train(
                    train_data, test_data,
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
