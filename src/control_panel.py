import tkinter as tk
from tkinter import ttk

# Importamos colores y clases base
from src.RLTradingSystemGUI import COLORS, BasePanel

class ControlPanel(BasePanel):
    def __init__(self, parent, on_start, on_pause, on_stop, on_connect, on_disconnect, on_train_config, on_extract_data):
        super().__init__(parent, title="Control Panel")
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_stop = on_stop
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_train_config = on_train_config
        self.on_extract_data = on_extract_data
        self.paused = False
        self.progress_value = 0
        self.tooltip_windows = {}  # Para guardar referencias a las ventanas de tooltip
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

        # Frame para botones de conexión
        conn_buttons_frame = tk.Frame(server_frame, bg=COLORS['bg_dark'])
        conn_buttons_frame.pack(side=tk.LEFT, padx=(5,0), pady=5)
        
        self.connect_button = tk.Button(
            conn_buttons_frame, text="Connect",
            command=self._on_connect_click,
            bg=COLORS['accent'], fg=COLORS['fg_white'],
            activebackground=COLORS['accent_hover'],
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT, bd=0, padx=10
        )
        self.connect_button.pack(side=tk.LEFT, padx=(0,5), pady=5)
        
        # Nuevo botón de desconexión
        self.disconnect_button = tk.Button(
            conn_buttons_frame, text="Disconnect",
            command=self._on_disconnect_click,
            bg=COLORS['red'], fg=COLORS['fg_white'],
            activebackground='#ff6b68',
            activeforeground=COLORS['fg_white'],
            font=('Segoe UI', 10),
            relief=tk.FLAT, bd=0, padx=10,
            state='disabled'  # Inicialmente deshabilitado
        )
        self.disconnect_button.pack(side=tk.LEFT, padx=(0,10), pady=5)

        # Campo para definir la cantidad de barras a extraer
        tk.Label(
            server_frame, text="Bars to Extract:",
            bg=COLORS['bg_dark'], fg=COLORS['fg_light'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(10,5), pady=5)
        
        self.bars_to_extract_var = tk.StringVar(value="5000")
        self.bars_to_extract_entry = tk.Entry(
            server_frame, textvariable=self.bars_to_extract_var,
            bg=COLORS['bg_medium'], fg=COLORS['fg_white'],
            insertbackground=COLORS['fg_white'], relief=tk.FLAT,
            bd=0, width=6, highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        self.bars_to_extract_entry.pack(side=tk.LEFT, padx=(0,10), pady=5)
        # Agregar tooltip para el campo
        self._create_tooltip(self.bars_to_extract_entry, "Cantidad de barras históricas a extraer")
        
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
        self.extract_data_button.pack(side=tk.LEFT, padx=(0,0), pady=5)

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
        # Solo deshabilitamos el botón temporalmente mientras se intenta la conexión
        # El estado final se configurará en MainGUI.check_connection_status
        self.connect_button.configure(state='disabled')
        self.disconnect_button.configure(state='normal')
        self.on_connect(ip, data_port)
    
    def _on_train_config_click(self):
        self.on_train_config()
    
    def _on_extract_data_click(self):
        try:
            bars_to_extract = int(self.bars_to_extract_var.get())
            self.on_extract_data(bars_to_extract)
        except ValueError:
            # Si el valor no es un número válido, usar el valor predeterminado
            self.bars_to_extract_var.set("5000")
            self.on_extract_data(5000)
    
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
        print(f"Auto Trading switched {state}")
    
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
            
        # Notificar al callback de cambio de estado
        self.on_switch_toggle()

    def _on_disconnect_click(self):
        """Desconectar de NinjaTrader"""
        if hasattr(self, 'on_disconnect') and self.on_disconnect:
            self.on_disconnect()
            self.connect_button.configure(state='normal')
            self.disconnect_button.configure(state='disabled')

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
        
    def _create_tooltip(self, widget, text):
        """Crea un tooltip para el widget dado"""
        def enter(event):
            # Crear una ventana flotante para el tooltip
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Eliminar el tooltip si ya existe
            self._hide_tooltip()
            
            # Crear ventana
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)  # Sin bordes ni título
            tw.wm_geometry(f"+{x}+{y}")
            
            # Crear una etiqueta con el texto del tooltip
            label = tk.Label(
                tw, text=text, justify=tk.LEFT,
                background=COLORS['bg_dark'], 
                foreground=COLORS['fg_light'],
                relief=tk.SOLID, borderwidth=1,
                font=("Segoe UI", 9),
                padx=5, pady=2
            )
            label.pack()
            
            # Guardar referencia para poder destruirlo después
            self.tooltip_windows[widget] = tw
        
        def leave(event):
            self._hide_tooltip()
        
        # Vincular eventos de ratón
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        
    def _hide_tooltip(self):
        """Oculta el tooltip actual"""
        for tooltip in self.tooltip_windows.values():
            try:
                tooltip.destroy()
            except:
                pass
        self.tooltip_windows.clear()