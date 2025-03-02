#!/usr/bin/env python
import os
import sys
import time
import logging
import threading

# Solucionar conflicto de bibliotecas OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "trading_system.log"))
    ]
)
logger = logging.getLogger("TradingSystem")

# Variable global para el servidor
nt_interface = None
gui_mode = False  # Flag para indicar si estamos en modo GUI

def start_server_in_background(ip='127.0.0.1', data_port=5000, order_port=5001, model_path=None, vec_normalize_path=None):
    """Inicia los servidores para comunicación con NinjaTrader en segundo plano"""
    
    from src.RLTradingAgent import NinjaTraderInterface
    
    # Asegurar que el directorio de logs existe
    os.makedirs("logs", exist_ok=True)
    
    global nt_interface
    
    # Detener interfaz existente si hay alguna
    if nt_interface:
        nt_interface.stop()
        nt_interface = None
    
    # Inicializar la interfaz de NinjaTrader
    import time
    
    # Pequeño retraso para asegurarse de que cualquier proceso anterior haya liberado los puertos
    logger.info(f"Esperando 2 segundos para asegurar puertos libres...")
    time.sleep(2)
    
    nt_interface = NinjaTraderInterface(
        server_ip=ip,
        data_port=data_port,
        order_port=order_port,
        model_path=model_path,
        vec_normalize_path=vec_normalize_path
    )
    
    # Iniciar la interfaz
    if nt_interface.start():
        logger.info("Servidores iniciados correctamente. Esperando conexiones de NinjaTrader...")
        
        # Activar trading automático si se proporcionó un modelo
        if model_path and vec_normalize_path:
            nt_interface.set_auto_trading(True)
            logger.info("Trading automático ACTIVADO")
        
        return True
    else:
        logger.error("Error al iniciar los servidores")
        return False

def stop_server():
    """Detiene los servidores para comunicación con NinjaTrader"""
    global nt_interface
    
    if nt_interface:
        nt_interface.stop()
        nt_interface = None
        logger.info("Servidores detenidos")
        return True
    return False

# Verificar si se ejecuta como script principal o se importa como módulo
if __name__ == "__main__":
    # Si hay argumentos, usar main.py para modo comando
    if len(sys.argv) > 1:
        from src.main import main
        sys.exit(main())
    else:
        # Si no hay argumentos, iniciar la GUI completa
        gui_mode = True
        from src.MainGUI import main
        main()
else:
    # Si se importa como módulo, no iniciar automáticamente la GUI
    logger.info("Módulo run_trading_system importado")