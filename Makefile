.PHONY: run train backtest server clean setup help

# Variables
PYTHON = python
DATA_DIR = ./data
MODELS_DIR = ./models
OUTPUT_DIR = ./output
LOG_DIR = ./logs

help:
	@echo "RL Trading System Makefile"
	@echo "=========================="
	@echo "Comandos disponibles:"
	@echo "  make run       - Iniciar la interfaz gráfica"
	@echo "  make train     - Entrenar un nuevo modelo"
	@echo "  make backtest  - Realizar backtesting con el último modelo"
	@echo "  make server    - Iniciar en modo servidor"
	@echo "  make clean     - Limpiar archivos temporales y logs"
	@echo "  make setup     - Instalar dependencias"
	@echo ""
	@echo "Variables personalizables:"
	@echo "  DATA_FILE=ruta/al/archivo.csv   - Especificar archivo de datos"
	@echo "  MODEL=ruta/al/modelo.zip        - Especificar modelo a utilizar"
	@echo "  IP=127.0.0.1                    - Dirección IP del servidor"
	@echo "  PORT=5000                       - Puerto para la conexión"
	@echo "  TIMESTEPS=500000                - Pasos de entrenamiento"
	@echo "  ALGO=PPO                        - Algoritmo (PPO, A2C, DQN)"

run:
	$(PYTHON) run_trading_system.py

train:
	@if [ "$(DATA_FILE)" \!= "" ]; then \
		$(PYTHON) run_trading_system.py train --data $(DATA_FILE) --output $(OUTPUT_DIR) $(if $(TIMESTEPS),--timesteps $(TIMESTEPS),) $(if $(ALGO),--algo $(ALGO),); \
	else \
		$(PYTHON) run_trading_system.py train --output $(OUTPUT_DIR) $(if $(TIMESTEPS),--timesteps $(TIMESTEPS),) $(if $(ALGO),--algo $(ALGO),); \
	fi

backtest:
	@if [ "$(MODEL)" \!= "" ] && [ "$(DATA_FILE)" \!= "" ]; then \
		$(PYTHON) run_trading_system.py backtest --model $(MODEL) --data $(DATA_FILE); \
	else \
		echo "Se requiere especificar modelo y datos con MODEL= y DATA_FILE="; \
		exit 1; \
	fi

server:
	@if [ "$(MODEL)" \!= "" ]; then \
		$(PYTHON) run_trading_system.py run --model $(MODEL) $(if $(IP),--ip $(IP),) $(if $(PORT),--port $(PORT),); \
	else \
		$(PYTHON) run_trading_system.py run $(if $(IP),--ip $(IP),) $(if $(PORT),--port $(PORT),); \
	fi

clean:
	rm -f $(LOG_DIR)/*.log
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

setup:
	pip install -r requirements.txt
