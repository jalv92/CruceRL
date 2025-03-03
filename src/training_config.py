import tkinter as tk
from tkinter import ttk, messagebox

# Importar los colores desde el archivo original
from src.RLTradingSystemGUI import COLORS, BasePanel

# Crear una definición local de TrainingConfigDialog en este archivo
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
        
        import os
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
        
        # Inicializar variables
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
        # Algoritmo
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
        
        # Descripción del preset seleccionado
        self.preset_description = tk.Text(
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
        self.preset_description.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preset_description.insert(tk.END, "Seleccione un preset para ver su descripción y aplicar automáticamente los parámetros recomendados para ese estilo de trading.")
    
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
        # Check for invalid values
        try:
            # Validate numeric values for critical parameters
            float(self.learning_rate_var.get())
            int(self.timesteps_var.get())
            int(self.batch_size_var.get())
            int(self.n_steps_var.get())
            float(self.reward_scaling_var.get())
            float(self.inactivity_penalty_var.get())
            float(self.window_size_var.get())
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Please enter valid numeric values: {e}")
            return
            
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

# Definir la clase TrainingConfigPanel como una clase independiente
class TrainingConfigPanel(BasePanel):
    def __init__(self, parent):
        super().__init__(parent, title="Training Configuration")
        self.setup_ui()
        
    def get_optimized_defaults(self):
        """Return optimized default hyperparameters"""
        return {
            # Algo settings
            'algo': 'PPO',
            'timesteps': '500000',
            'learning_rate': '0.0001',  # Lower learning rate for stability
            'batch_size': '128',        # Larger batch size
            'n_steps': '1024',          # Shorter steps for faster updates
            'device': 'auto',
            'auto_tune': True,
            
            # Environment settings
            'balance': '100000.0',
            'commission': '0.0001',
            'slippage': '0.0001',
            'window_size': '5',         # Smaller window for easier learning
            'position_size': '0.1',
            'stop_loss': '0.02',
            'take_profit': '0.04',
            
            # Reward settings
            'reward_scaling': '0.05',
            'inactivity_penalty': '-0.00002',  # Reduced penalty
            'bankruptcy_penalty': '-1.0',
            'drawdown_factor': '0.1',          # Lower penalty
            'win_rate_bonus': '0.001',
            'normalize_rewards': True,
            'capital_efficiency_bonus': '0.0001',
            'time_decay_factor': '0.0',
        }
        
    def setup_ui(self):
        container = self.main_container
        
        # Get optimized defaults
        defaults = self.get_optimized_defaults()
        
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
        
        # Use defaults to initialize variables
        self.algo_var = tk.StringVar(value=defaults['algo'])
        self.timesteps_var = tk.StringVar(value=defaults['timesteps'])
        self.learning_rate_var = tk.StringVar(value=defaults['learning_rate'])
        self.batch_size_var = tk.StringVar(value=defaults['batch_size'])
        self.n_steps_var = tk.StringVar(value=defaults['n_steps'])
        self.device_var = tk.StringVar(value=defaults['device'])
        self.auto_tune_var = tk.BooleanVar(value=defaults['auto_tune'])
        
        # Environment variables
        self.balance_var = tk.StringVar(value=defaults['balance'])
        self.commission_var = tk.StringVar(value=defaults['commission'])
        self.slippage_var = tk.StringVar(value=defaults['slippage'])
        self.window_size_var = tk.StringVar(value=defaults['window_size'])
        self.position_size_var = tk.StringVar(value=defaults['position_size']) 
        self.stop_loss_var = tk.StringVar(value=defaults['stop_loss'])
        self.take_profit_var = tk.StringVar(value=defaults['take_profit'])
        
        # Reward variables
        self.reward_scaling_var = tk.StringVar(value=defaults['reward_scaling'])
        self.inactivity_penalty_var = tk.StringVar(value=defaults['inactivity_penalty'])
        self.bankruptcy_penalty_var = tk.StringVar(value=defaults['bankruptcy_penalty'])
        self.drawdown_factor_var = tk.StringVar(value=defaults['drawdown_factor'])
        self.win_rate_bonus_var = tk.StringVar(value=defaults['win_rate_bonus'])
        self.normalize_rewards_var = tk.BooleanVar(value=defaults['normalize_rewards'])
        self.capital_efficiency_bonus_var = tk.StringVar(value=defaults['capital_efficiency_bonus'])
        self.time_decay_factor_var = tk.StringVar(value=defaults['time_decay_factor'])
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pestaña 1: Parámetros básicos del algoritmo
        basic_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(basic_frame, text="Algoritmo")
        
        basic_frame.columnconfigure(0, weight=1)
        basic_frame.columnconfigure(1, weight=3)
        
        # Filas de configuración básica
        row = 0
        
        # Algoritmo
        tk.Label(basic_frame, text="Algorithm:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        algo_frame = tk.Frame(basic_frame, bg=COLORS['bg_dark'])
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
        tk.Label(basic_frame, text="Timesteps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        timesteps_entry = tk.Entry(basic_frame, textvariable=self.timesteps_var, width=10, **entry_style)
        timesteps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Learning rate
        tk.Label(basic_frame, text="Learning Rate:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        lr_entry = tk.Entry(basic_frame, textvariable=self.learning_rate_var, width=10, **entry_style)
        lr_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Batch size
        tk.Label(basic_frame, text="Batch Size:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        batch_entry = tk.Entry(basic_frame, textvariable=self.batch_size_var, width=10, **entry_style)
        batch_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # N steps
        tk.Label(basic_frame, text="N Steps:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        n_steps_entry = tk.Entry(basic_frame, textvariable=self.n_steps_var, width=10, **entry_style)
        n_steps_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Device
        tk.Label(basic_frame, text="Device:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        device_frame = tk.Frame(basic_frame, bg=COLORS['bg_dark'])
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
        
        # Checkbox para Auto-tuning
        auto_tune_check = tk.Checkbutton(
            basic_frame, 
            text="Auto-tune Hyperparameters", 
            variable=self.auto_tune_var,
            bg=COLORS['bg_dark'],
            fg=COLORS['fg_light'],
            selectcolor=COLORS['bg_medium'],
            activebackground=COLORS['bg_dark'],
            activeforeground=COLORS['accent']
        )
        auto_tune_check.grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=5)        
        
        # Pestaña 2: Parámetros del entorno
        env_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(env_frame, text="Entorno")
        
        env_frame.columnconfigure(0, weight=1)
        env_frame.columnconfigure(1, weight=1)
        env_frame.columnconfigure(2, weight=1)
        env_frame.columnconfigure(3, weight=1)
        
        # Añadir campos para el entorno
        row = 0
        
        # Balance inicial
        tk.Label(env_frame, text="Initial Balance:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        balance_entry = tk.Entry(env_frame, textvariable=self.balance_var, width=10, **entry_style)
        balance_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Window size
        tk.Label(env_frame, text="Window Size:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        window_entry = tk.Entry(env_frame, textvariable=self.window_size_var, width=10, **entry_style)
        window_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Comisión
        tk.Label(env_frame, text="Commission:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        commission_entry = tk.Entry(env_frame, textvariable=self.commission_var, width=10, **entry_style)
        commission_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Position size
        tk.Label(env_frame, text="Position Size:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        position_entry = tk.Entry(env_frame, textvariable=self.position_size_var, width=10, **entry_style)
        position_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Slippage
        tk.Label(env_frame, text="Slippage:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        slippage_entry = tk.Entry(env_frame, textvariable=self.slippage_var, width=10, **entry_style)
        slippage_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Stop loss
        tk.Label(env_frame, text="Stop Loss %:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        stop_entry = tk.Entry(env_frame, textvariable=self.stop_loss_var, width=10, **entry_style)
        stop_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Take profit
        tk.Label(env_frame, text="Take Profit %:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        profit_entry = tk.Entry(env_frame, textvariable=self.take_profit_var, width=10, **entry_style)
        profit_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        # Pestaña 3: Sistema de recompensas
        reward_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(reward_frame, text="Recompensas")
        
        reward_frame.columnconfigure(0, weight=1)
        reward_frame.columnconfigure(1, weight=1)
        reward_frame.columnconfigure(2, weight=1)
        reward_frame.columnconfigure(3, weight=1)
        
        row = 0
        
        # Factor de escalado de recompensa
        tk.Label(reward_frame, text="Reward Scaling:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        reward_scaling_entry = tk.Entry(reward_frame, textvariable=self.reward_scaling_var, width=10, **entry_style)
        reward_scaling_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Penalización por inactividad
        tk.Label(reward_frame, text="Inactivity Penalty:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        inactivity_entry = tk.Entry(reward_frame, textvariable=self.inactivity_penalty_var, width=10, **entry_style)
        inactivity_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Penalización por quiebra
        tk.Label(reward_frame, text="Bankruptcy Penalty:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        bankruptcy_entry = tk.Entry(reward_frame, textvariable=self.bankruptcy_penalty_var, width=10, **entry_style)
        bankruptcy_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Factor de drawdown
        tk.Label(reward_frame, text="Drawdown Factor:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        drawdown_entry = tk.Entry(reward_frame, textvariable=self.drawdown_factor_var, width=10, **entry_style)
        drawdown_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Bonificación por win rate
        tk.Label(reward_frame, text="Win Rate Bonus:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        win_rate_entry = tk.Entry(reward_frame, textvariable=self.win_rate_bonus_var, width=10, **entry_style)
        win_rate_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Bonus por eficiencia de capital
        tk.Label(reward_frame, text="Capital Efficiency:", **label_style).grid(row=row, column=2, sticky='w', padx=10, pady=5)
        capital_entry = tk.Entry(reward_frame, textvariable=self.capital_efficiency_bonus_var, width=10, **entry_style)
        capital_entry.grid(row=row, column=3, sticky='w', padx=10, pady=5)
        
        row += 1
        
        # Factor de descuento temporal
        tk.Label(reward_frame, text="Time Decay Factor:", **label_style).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        time_decay_entry = tk.Entry(reward_frame, textvariable=self.time_decay_factor_var, width=10, **entry_style)
        time_decay_entry.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        
        # Normalización de recompensas
        normalize_check = tk.Checkbutton(
            reward_frame, 
            text="Normalize Rewards", 
            variable=self.normalize_rewards_var,
            bg=COLORS['bg_dark'],
            fg=COLORS['fg_light'],
            selectcolor=COLORS['bg_medium'],
            activebackground=COLORS['bg_dark'],
            activeforeground=COLORS['accent']
        )
        normalize_check.grid(row=row, column=2, columnspan=2, sticky='w', padx=10, pady=5)
        
        # Pestaña 4: Presets de trading
        presets_frame = tk.Frame(self.notebook, bg=COLORS['bg_dark'])
        self.notebook.add(presets_frame, text="Presets")
        
        # Añadir presets para diferentes estilos de trading
        preset_title = tk.Label(
            presets_frame, 
            text="Presets de Trading",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent']
        )
        preset_title.pack(pady=(10, 15))
        
        # Frame para los botones de preset
        presets_buttons_frame = tk.Frame(presets_frame, bg=COLORS['bg_dark'])
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
            command=lambda: self.apply_preset("hft"),
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
            command=lambda: self.apply_preset("position"),
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
            command=lambda: self.apply_preset("trend"),
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_white'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['fg_white'],
            **button_style
        )
        trend_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Descripción del preset seleccionado
        self.preset_description = tk.Text(
            presets_frame,
            height=10,
            wrap=tk.WORD,
            bg=COLORS['bg_medium'],
            fg=COLORS['fg_light'],
            font=('Segoe UI', 9),
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.preset_description.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preset_description.insert(tk.END, "Seleccione un preset para ver su descripción y aplicar automáticamente los parámetros recomendados para ese estilo de trading.")
        self.preset_description.config(state=tk.DISABLED)

    def apply_preset(self, preset_type):
        """Aplica un preset predefinido para un estilo de trading específico"""
        self.preset_description.config(state=tk.NORMAL)
        self.preset_description.delete(1.0, tk.END)
        
        if preset_type == "hft":
            # Trading de Alta Frecuencia - Valores mejorados para el aprendizaje
            self.algo_var.set("PPO")
            self.learning_rate_var.set("0.0001")
            self.batch_size_var.set("128")
            self.n_steps_var.set("512")
            self.timesteps_var.set("600000")
            
            # Parámetros de entorno
            self.window_size_var.set("5")
            self.commission_var.set("0.0002")
            self.slippage_var.set("0.0002")
            self.position_size_var.set("0.05")
            self.stop_loss_var.set("0.01")
            self.take_profit_var.set("0.02")
            
            # Parámetros de recompensa
            self.reward_scaling_var.set("0.05")
            self.inactivity_penalty_var.set("-0.00002")
            self.time_decay_factor_var.set("0.001")
            self.drawdown_factor_var.set("0.1")
            
            description = """Trading de Alta Frecuencia

        Configuración optimizada para estrategias de trading de alta frecuencia que buscan capitalizar movimientos pequeños y rápidos del mercado.

        Características principales:
        - Penalización menor por inactividad para evitar sobre-trading
        - Factor de descuento temporal para favorecer operaciones más cortas
        - Stops y targets más ajustados
        - Posiciones más pequeñas para gestionar el riesgo
        - Mayor factor de escalado de recompensa"""
        
            self.preset_description.insert(tk.END, description)
            self.preset_description.config(state=tk.DISABLED)
            
        elif preset_type == "position":
            # Trading de Posición - Valores mejorados para el aprendizaje
            self.algo_var.set("PPO")
            self.learning_rate_var.set("0.00005")
            self.batch_size_var.set("256")
            self.n_steps_var.set("1024")
            self.timesteps_var.set("700000")
            
            # Parámetros de entorno
            self.window_size_var.set("8")
            self.commission_var.set("0.0001")
            self.slippage_var.set("0.0001")
            self.position_size_var.set("0.1")
            self.stop_loss_var.set("0.03")
            self.take_profit_var.set("0.06")
            
            # Parámetros de recompensa
            self.reward_scaling_var.set("0.03")
            self.inactivity_penalty_var.set("-0.00001")
            self.win_rate_bonus_var.set("0.005")
            self.drawdown_factor_var.set("0.1")
            
            description = """Trading de Posición

        Configuración optimizada para estrategias de trading de posición que mantienen operaciones durante periodos más largos buscando movimientos de precio mayores.

        Características principales:
        - Penalización muy pequeña por inactividad para permitir gran paciencia
        - Mayor bonificación por win rate
        - Stops y targets más amplios
        - Tamaño de ventana más grande para capturar movimientos de más largo plazo
        - Tasa de aprendizaje reducida para más estabilidad"""
        
            self.preset_description.insert(tk.END, description)
            self.preset_description.config(state=tk.DISABLED)
            
        elif preset_type == "trend":
            # Trading de Tendencia - Valores mejorados para el aprendizaje
            self.algo_var.set("PPO")
            self.learning_rate_var.set("0.0001")
            self.batch_size_var.set("128")
            self.n_steps_var.set("1024")
            self.timesteps_var.set("600000")
            
            # Parámetros de entorno
            self.window_size_var.set("6")
            self.commission_var.set("0.0001")
            self.slippage_var.set("0.0001")
            self.position_size_var.set("0.15")
            self.stop_loss_var.set("0.03")
            self.take_profit_var.set("0.09")
            
            # Parámetros de recompensa
            self.reward_scaling_var.set("0.04")
            self.inactivity_penalty_var.set("-0.00001")
            self.win_rate_bonus_var.set("0.001")
            self.drawdown_factor_var.set("0.05")
            self.normalize_rewards_var.set(True)
            
            description = """Trading de Tendencia

        Configuración optimizada para estrategias de trading que buscan identificar y seguir tendencias de mercado.

        Características principales:
        - Ratio de riesgo/recompensa elevado (1:3)
        - Posiciones mayores cuando se identifica una tendencia
        - Factor de drawdown reducido para permitir drawdowns en operaciones de tendencia
        - Normalización de recompensas para estabilidad en operaciones largas
        - Tasa de aprendizaje ajustada para aprender patrones de tendencia"""
        
            self.preset_description.insert(tk.END, description)
            self.preset_description.config(state=tk.DISABLED)
    
    def show(self):
        """Muestra el panel de configuración en una ventana separada"""
        try:
            self.dialog = TrainingConfigDialog(self.winfo_toplevel(), self)
        except Exception as e:
            import tkinter as tk
            from tkinter import messagebox
            messagebox.showerror("Error al abrir configuración", f"Se produjo un error al abrir la ventana de configuración: {e}")
            import logging
            logging.error(f"Error al abrir la ventana de configuración: {e}", exc_info=True)
    
    def get_training_params(self):
        """Obtener los parámetros de entrenamiento como dict"""
        try:
            # Parámetros básicos del algoritmo
            params = {
                'algo': self.algo_var.get(),
                'total_timesteps': int(self.timesteps_var.get()),
                'learning_rate': float(self.learning_rate_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'n_steps': int(self.n_steps_var.get()),
                'device': self.device_var.get(),
                'enable_auto_tuning': bool(self.auto_tune_var.get()),
            }
            
            # Parámetros del entorno
            params.update({
                'initial_balance': float(self.balance_var.get()),
                'commission': float(self.commission_var.get()),
                'slippage': float(self.slippage_var.get()),
                'window_size': int(self.window_size_var.get()),
                'position_size': float(self.position_size_var.get()),
                'stop_loss_pct': float(self.stop_loss_var.get()),
                'take_profit_pct': float(self.take_profit_var.get()),
            })
            
            # Parámetros del sistema de recompensas
            params.update({
                'reward_scaling': float(self.reward_scaling_var.get()),
                'inactivity_penalty': float(self.inactivity_penalty_var.get()),
                'bankruptcy_penalty': float(self.bankruptcy_penalty_var.get()),
                'drawdown_factor': float(self.drawdown_factor_var.get()),
                'win_rate_bonus': float(self.win_rate_bonus_var.get()),
                'normalize_rewards': bool(self.normalize_rewards_var.get()),
                'capital_efficiency_bonus': float(self.capital_efficiency_bonus_var.get()),
                'time_decay_factor': float(self.time_decay_factor_var.get()),
            })
            
            return params
        except ValueError as e:
            messagebox.showerror("Invalid parameters", f"Invalid value: {e}")
            return None