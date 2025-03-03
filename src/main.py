#!/usr/bin/env python
"""
RLTrading - Sistema de trading con aprendizaje por refuerzo
===========================================================

Este es el script principal para entrenar, evaluar y ejecutar el sistema de trading
basado en aprendizaje por refuerzo (RL) para NinjaTrader.

Uso:
    python main.py train --data [archivo_csv] --output [directorio] --timesteps [pasos]
    python main.py backtest --model [ruta_modelo] --data [archivo_csv]
    python main.py run --model [ruta_modelo] --ip [dirección_ip] --port [puerto]
    python main.py gui (o sin argumentos para iniciar la interfaz gráfica)
"""

import os
# Solucionar conflicto de bibliotecas OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
# Asegurar que el directorio logs existe
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "rl_trading.log"))
    ]
)
logger = logging.getLogger("RLTrading")

def check_dependencies():
    """Verifica que todas las dependencias necesarias estén instaladas"""
    try:
        import gymnasium
        import stable_baselines3
        import torch
        logger.info("Todas las dependencias necesarias están instaladas.")
        return True
    except ImportError as e:
        logger.error(f"Falta una dependencia: {e}")
        logger.error("Por favor, instala todas las dependencias requeridas con:")
        logger.error("pip install gymnasium stable-baselines3 torch pandas matplotlib")
        return False

def train_model(args):
    """Entrena un nuevo modelo RL con los datos proporcionados"""
    from src.TrainingManager import DataLoader, TrainingManager
    
    # Crea directorios de salida si no existen
    models_dir = os.path.join(args.output, "models")
    logs_dir = os.path.join(args.output, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Inicializa el cargador de datos
    data_loader = DataLoader(min_bars=6000)
    
    # Carga o genera datos
    if args.data and os.path.exists(args.data):
        logger.info(f"Cargando datos desde {args.data}")
        train_data, test_data = data_loader.prepare_train_test_data(csv_file=args.data)
    else:
        logger.info("Generando datos sintéticos para entrenamiento")
        train_data, test_data = data_loader.prepare_train_test_data()
    
    # Inicializa el gestor de entrenamiento
    training_manager = TrainingManager(models_dir=models_dir, logs_dir=logs_dir)
    
    # Configura parámetros de entrenamiento
    train_params = {
        'algo': args.algo,
        'policy': 'MlpPolicy',
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'total_timesteps': args.timesteps
    }
    
    # Configura parámetros del entorno
    env_params = {
        'initial_balance': args.initial_balance,
        'commission': args.commission,
        'slippage': args.slippage,
        'window_size': args.window_size,
        'reward_scaling': args.reward_scaling,
        'position_size': args.position_size,
        'stop_loss_pct': args.stop_loss_pct if args.stop_loss_pct > 0 else None,
        'take_profit_pct': args.take_profit_pct if args.take_profit_pct > 0 else None
    }
    
    # Entrena el modelo
    logger.info("Iniciando entrenamiento del modelo RL...")
    model, train_env, model_path, vec_norm_path = training_manager.train(
        train_data, test_data,
        **train_params, **env_params
    )
    
    # Realiza backtesting
    vec_norm_path = os.path.join(models_dir, "vec_normalize_final.pkl")
    performance_df = training_manager.backtest(model, test_data, vec_norm_path)
    
    # Guarda y visualiza resultados
    training_manager.save_backtest_results(performance_df)
    training_manager.plot_backtest_results(performance_df)
    
    logger.info(f"Entrenamiento completado. El modelo se ha guardado en {models_dir}/final_model.zip")
    
    return True

def backtest_model(args):
    """Evalúa un modelo existente usando datos históricos"""
    from src.TrainingManager import DataLoader, TrainingManager
    
    # Verifica que exista el modelo
    if not os.path.exists(args.model):
        logger.error(f"No se encuentra el modelo: {args.model}")
        return False
    
    # Inicializa el cargador de datos
    data_loader = DataLoader(min_bars=6000)
    
    # Carga o genera datos
    if args.data and os.path.exists(args.data):
        logger.info(f"Cargando datos desde {args.data}")
        _, test_data = data_loader.prepare_train_test_data(csv_file=args.data, test_ratio=1.0)
    else:
        logger.error("Se requiere un archivo de datos para realizar backtesting")
        return False
    
    # Inicializa el gestor de entrenamiento
    training_manager = TrainingManager()
    
    # Carga el modelo
    logger.info(f"Cargando modelo desde {args.model}")
    model = training_manager.load_model(args.model)
    
    # Ruta al archivo VecNormalize si existe
    model_dir = os.path.dirname(args.model)
    vec_normalize_path = os.path.join(model_dir, "vec_normalize_final.pkl")
    if not os.path.exists(vec_normalize_path):
        vec_normalize_path = None
        logger.warning("No se encontró un archivo de normalización. El rendimiento puede verse afectado.")
    
    # Realiza backtesting
    logger.info("Iniciando backtesting...")
    performance_df = training_manager.backtest(model, test_data, vec_normalize_path)
    
    # Guarda y visualiza resultados
    training_manager.save_backtest_results(performance_df, filename=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    training_manager.plot_backtest_results(performance_df)
    
    logger.info("Backtesting completado.")
    
    return True

def run_trading_system(args):
    """Ejecuta el sistema de trading en tiempo real conectado a NinjaTrader"""
    from src.RLTradingAgent import NinjaTraderInterface
    
    # Verifica que exista el modelo si se proporciona
    if args.model and not os.path.exists(args.model):
        logger.warning(f"No se encuentra el modelo: {args.model}. Se utilizará el modo de reglas básicas.")
        model_path = None
    else:
        model_path = args.model
    
    # Ruta al archivo VecNormalize si existe
    if model_path:
        model_dir = os.path.dirname(model_path)
        vec_normalize_path = os.path.join(model_dir, "vec_normalize_final.pkl")
        if not os.path.exists(vec_normalize_path):
            vec_normalize_path = None
            logger.warning("No se encontró un archivo de normalización. El rendimiento puede verse afectado.")
    else:
        vec_normalize_path = None
    
    # Inicializa la interfaz de NinjaTrader
    logger.info(f"Conectando con NinjaTrader en {args.ip}:{args.port}")
    nt_interface = NinjaTraderInterface(
        server_ip=args.ip,
        data_port=args.port,
        order_port=args.port + 1,
        model_path=model_path,
        vec_normalize_path=vec_normalize_path
    )
    
    # Inicia la interfaz
    nt_interface.start()
    
    # Bucle principal - mantener ejecución hasta ser interrumpido
    try:
        logger.info("Sistema de trading RL iniciado. Presiona Ctrl+C para detener.")
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupción recibida, cerrando conexiones...")
    finally:
        # Detiene la interfaz
        nt_interface.stop()
        logger.info("Sistema de trading RL detenido.")
    
    return True

def start_gui():
    """Inicia la interfaz gráfica de usuario"""
    try:
        import tkinter as tk
        from tkinter import messagebox
        import src.RLTradingSystemGUI as gui
        
        logger.info("Iniciando interfaz gráfica del sistema de trading RL...")
        
        # Crear ventana principal
        root = tk.Tk()
        root.title("RL Trading System")
        root.geometry("1200x800")
        root.minsize(1000, 700)
        
        # Configurar colores
        colors = {
            'bg_very_dark': '#1e1e1e',
            'bg_dark': '#252526',
            'bg_medium': '#333333',
            'accent': '#0078d7',
            'fg_white': '#ffffff',
            'fg_light': '#f0f0f0'
        }
        root.configure(background=colors['bg_very_dark'])
        
        # Contenedor principal
        main_container = tk.Frame(root, bg=colors['bg_very_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Título
        title = tk.Label(
            main_container, 
            text="RL Trading System", 
            font=('Segoe UI', 24, 'bold'),
            fg=colors['accent'],
            bg=colors['bg_very_dark']
        )
        title.pack(pady=20)
        
        # Mensaje de bienvenida
        welcome = tk.Label(
            main_container, 
            text="Sistema de trading con aprendizaje por refuerzo para NinjaTrader 8",
            font=('Segoe UI', 14),
            fg=colors['fg_light'],
            bg=colors['bg_very_dark']
        )
        welcome.pack(pady=10)
        
        # Panel de botones
        button_frame = tk.Frame(main_container, bg=colors['bg_very_dark'])
        button_frame.pack(pady=30)
        
        button_style = {
            'font': ('Segoe UI', 12),
            'width': 20, 
            'height': 2,
            'bd': 0, 
            'relief': tk.FLAT
        }
        
        # Botones para cada modo
        train_btn = tk.Button(
            button_frame, 
            text="Entrenar Modelo", 
            bg=colors['bg_medium'],
            fg=colors['fg_white'],
            command=lambda: messagebox.showinfo("Entrenamiento", "Para entrenar un modelo, use el comando:\npython run_trading_system.py train --data ruta/a/datos.csv"),
            **button_style
        )
        train_btn.grid(row=0, column=0, padx=10, pady=10)
        
        backtest_btn = tk.Button(
            button_frame, 
            text="Backtesting", 
            bg=colors['bg_medium'],
            fg=colors['fg_white'],
            command=lambda: messagebox.showinfo("Backtesting", "Para realizar backtesting, use el comando:\npython run_trading_system.py backtest --model ruta/a/modelo.zip --data ruta/a/datos.csv"),
            **button_style
        )
        backtest_btn.grid(row=0, column=1, padx=10, pady=10)
        
        server_btn = tk.Button(
            button_frame, 
            text="Conectar a NinjaTrader", 
            bg=colors['accent'],
            fg=colors['fg_white'],
            command=lambda: messagebox.showinfo("Modo Servidor", "Para conectar con NinjaTrader, use el comando:\npython run_trading_system.py run --model ruta/a/modelo.zip"),
            **button_style
        )
        server_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Mensaje de estado
        status = tk.Label(
            main_container, 
            text="Sistema listo para usar. Seleccione una operación o use los comandos de línea.",
            font=('Segoe UI', 10, 'italic'),
            fg=colors['fg_light'],
            bg=colors['bg_very_dark']
        )
        status.pack(pady=20)
        
        # Iniciar la aplicación
        root.mainloop()
        return True
    except Exception as e:
        logger.error(f"Error al iniciar la interfaz gráfica: {e}")
        import traceback
        traceback.print_exc()
        return False



def main():
    """Función principal"""
    # Verificar dependencias
    if not check_dependencies():
        return 1
    
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description='Sistema de Trading con RL para NinjaTrader')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Subcomando: train
    train_parser = subparsers.add_parser('train', help='Entrenar un nuevo modelo')
    train_parser.add_argument('--data', type=str, help='Ruta al archivo CSV con datos históricos')
    train_parser.add_argument('--output', type=str, default='./output', help='Directorio de salida para modelos y logs')
    train_parser.add_argument('--timesteps', type=int, default=500000, help='Número total de pasos de entrenamiento')
    train_parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'A2C', 'DQN'], help='Algoritmo RL a utilizar')
    train_parser.add_argument('--learning-rate', type=float, default=0.0003, help='Tasa de aprendizaje')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Tamaño del lote')
    train_parser.add_argument('--n-steps', type=int, default=2048, help='Número de pasos por actualización (PPO/A2C)')
    train_parser.add_argument('--initial-balance', type=float, default=100000.0, help='Balance inicial')
    train_parser.add_argument('--commission', type=float, default=0.0001, help='Comisión por operación')
    train_parser.add_argument('--slippage', type=float, default=0.0001, help='Deslizamiento por operación')
    train_parser.add_argument('--window-size', type=int, default=10, help='Tamaño de ventana de observación')
    train_parser.add_argument('--reward-scaling', type=float, default=0.01, help='Factor de escalado de recompensa')
    train_parser.add_argument('--position-size', type=float, default=0.1, help='Tamaño de posición como % del balance')
    train_parser.add_argument('--stop-loss-pct', type=float, default=0.02, help='Porcentaje de stop loss (0 para deshabilitar)')
    train_parser.add_argument('--take-profit-pct', type=float, default=0.04, help='Porcentaje de take profit (0 para deshabilitar)')
    
    # Subcomando: backtest
    backtest_parser = subparsers.add_parser('backtest', help='Evaluar un modelo existente con datos históricos')
    backtest_parser.add_argument('--model', type=str, required=True, help='Ruta al archivo del modelo')
    backtest_parser.add_argument('--data', type=str, required=True, help='Ruta al archivo CSV con datos históricos para test')
    
    # Subcomando: run
    run_parser = subparsers.add_parser('run', help='Ejecutar el sistema de trading en tiempo real')
    run_parser.add_argument('--model', type=str, help='Ruta al archivo del modelo (opcional)')
    run_parser.add_argument('--ip', type=str, default='127.0.0.1', help='Dirección IP para conectar con NinjaTrader')
    run_parser.add_argument('--port', type=int, default=5000, help='Puerto para la conexión de datos')
    
    # Subcomando: gui
    gui_parser = subparsers.add_parser('gui', help='Iniciar la interfaz gráfica de usuario')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar el comando correspondiente
    if args.command == 'gui':
        if start_gui():
            return 0
        return 1
    elif args.command is None:
        # Si no se proporciona ningún comando, iniciar la interfaz gráfica por defecto
        logger.info("No se especificó ningún comando. Iniciando interfaz gráfica...")
        if start_gui():
            return 0
        return 1
    elif args.command == 'train':
        if train_model(args):
            return 0
        return 1
    elif args.command == 'backtest':
        if backtest_model(args):
            return 0
        return 1
    elif args.command == 'run':
        if run_trading_system(args):
            return 0
        return 1
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())