# RL Trading System para NinjaTrader 8

## Descripción
Sistema de trading basado en aprendizaje por refuerzo (RL) que se integra con NinjaTrader 8. Capaz de entrenar modelos, realizar backtesting y ejecutar operaciones en tiempo real.

## Estructura del proyecto
- **src/**: Código fuente del sistema
- **doc/**: Documentación y archivos de ayuda
- **data/**: Datos para entrenamiento y backtesting
- **models/**: Modelos RL entrenados
- **logs/**: Archivos de registro
- **integrations/**: Archivos para integración con NinjaTrader
- **output/**: Resultados de entrenamiento y backtesting

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```bash
# Iniciar la interfaz gráfica
python run_trading_system.py

# Entrenar un nuevo modelo
python run_trading_system.py train --data path/to/data.csv --output ./output

# Ejecutar backtesting
python run_trading_system.py backtest --model path/to/model.zip --data path/to/data.csv

# Conectarse a NinjaTrader en modo servidor
python run_trading_system.py run --model path/to/model.zip
```

## Integración con NinjaTrader 8
1. Importar `integrations/RLTrading.cs` en NinjaTrader 8
2. Configurar la estrategia en un gráfico
3. Iniciar el sistema en modo servidor

## Parámetros personalizables
Consulta la documentación en la carpeta `doc/` para conocer todos los parámetros disponibles.

## Limpieza y Mantenimiento

Para mantener el proyecto organizado:

1. Use `make clean` para eliminar archivos temporales como logs y bytecode.
2. Ejecute `./cleanup_files.sh` para eliminar scripts y archivos temporales que se usaron durante el desarrollo.

## Estructura de Archivos

- `run_trading_system.py`: Punto de entrada principal para todas las operaciones
- `src/main.py`: Implementación principal de las funcionalidades
- `src/TradingEnvironment.py`: Entorno para aprendizaje por refuerzo
- `src/TrainingManager.py`: Gestión de entrenamiento y evaluación de modelos
- `src/RLTradingAgent.py`: Implementación del agente e interfaz con NinjaTrader
- `src/RLTradingSystemGUI.py`: Componentes de la interfaz gráfica

## Solución de Problemas

Si encuentra errores relacionados con OpenMP, esto puede deberse a conflictos entre bibliotecas. 
La solución ya está implementada en el código (configuración de `KMP_DUPLICATE_LIB_OK=TRUE`).

Para otros problemas, revise los archivos de log en la carpeta `logs/`.
