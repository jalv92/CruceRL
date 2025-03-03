# 🤖 Sistema de Trading RL para NinjaTrader 8 🚀

## 📋 Descripción

El Sistema de Trading RL es una potente plataforma de trading que utiliza Aprendizaje por Refuerzo (RL) para crear, probar y ejecutar estrategias de trading automatizadas para NinjaTrader 8. El sistema aprende patrones óptimos de trading a partir de datos históricos y puede ejecutar operaciones en tiempo real con mínima intervención humana.

## ✨ Características Principales

- 🧠 **Trading basado en Aprendizaje por Refuerzo** - El sistema aprende de los datos del mercado para tomar decisiones de trading
- 📊 **Visualización avanzada de datos** - Seguimiento del rendimiento con gráficos claros e interactivos
- 🔄 **Ejecución de operaciones en tiempo real** - Integración directa con NinjaTrader 8
- 🧪 **Backtesting completo** - Prueba tus modelos en datos históricos antes de arriesgar dinero real
- ⚙️ **Sistemas de recompensa personalizables** - Ajusta cómo la IA evalúa el rendimiento del trading
- 💼 **Múltiples estilos de trading** - Elige entre presets de alta frecuencia, posición o seguimiento de tendencia
- 🔍 **Métricas de rendimiento detalladas** - Seguimiento de balance, P&L, win rate y más

## 🛠️ Requisitos Previos

Antes de instalar, asegúrate de tener:

- Windows 10 o posterior (requerido para NinjaTrader 8)
- NinjaTrader 8 instalado (la versión demo gratuita es suficiente para pruebas)
- Privilegios de administrador en tu ordenador

## 📥 Guía de Instalación Completa (Para Principiantes)

### Paso 1: Instalar Python (si no está instalado)

1. Ve a [Python.org](https://www.python.org/downloads/) y descarga la última versión de Python 3.8+
2. Ejecuta el instalador, y asegúrate de marcar ✅ "Añadir Python al PATH"
3. Haz clic en "Instalar Ahora" y espera a que finalice la instalación
4. Verifica la instalación abriendo el Símbolo del Sistema y escribiendo:
   ```
   python --version
   ```
   Deberías ver la versión de Python mostrada.

### Paso 2: Descargar el Sistema de Trading RL

1. Descarga el sistema desde [GitHub](https://github.com/yourusername/rl-trading-system) o el enlace proporcionado
2. Extrae el archivo ZIP a una ubicación que puedas encontrar fácilmente (p.ej., `C:\RLTrading`)

### Paso 3: Instalar Paquetes Requeridos

1. Abre el Símbolo del Sistema como Administrador
2. Navega a la carpeta RLTrading:
   ```
   cd C:\RLTrading
   ```
3. Instala todos los paquetes requeridos:
   ```
   pip install -r requirements.txt
   ```
   Esto instalará todas las bibliotecas necesarias como numpy, pandas, matplotlib, gymnasium, stable-baselines3 y torch.

### Paso 4: Configurar la Integración con NinjaTrader

1. Abre NinjaTrader 8
2. Ve a Herramientas → Importar → Complemento NinjaScript...
3. Navega a tu carpeta de RLTrading y selecciona:
   - `integrations/RLExecutor.cs` para trading en vivo
   - `integrations/RLLink.cs` para el conector de datos
4. Después de importar, los indicadores deberían aparecer en tu plataforma NinjaTrader

### Paso 5: Iniciar el Sistema de Trading RL

1. Abre el Símbolo del Sistema
2. Navega a tu carpeta de RLTrading:
   ```
   cd C:\RLTrading
   ```
3. Inicia la aplicación:
   ```
   python run_trading_system.py
   ```
4. Inicia sesión con las credenciales predeterminadas:
   - Usuario: `trader`
   - Contraseña: `jav92`

## 🚀 Guía de Inicio Rápido

### 📈 Extracción de Datos de Trading desde NinjaTrader

1. Abre NinjaTrader y muestra el gráfico que quieres usar para el entrenamiento
2. Añade el indicador RLExecutor a tu gráfico
3. Haz clic en el botón "Extract Data" en el Sistema de Trading RL
4. Los datos se guardarán en la carpeta `data` en tu directorio RLTrading

### 🏋️ Entrenando Tu Primer Modelo

1. En el Sistema de Trading RL, haz clic en "Configure Training"
2. Selecciona tu preset de entrenamiento preferido (Alta Frecuencia, Posición o Tendencia)
3. Haz clic en "OK" para aceptar la configuración
4. Selecciona el modo "train" del menú desplegable
5. Haz clic en "Start" para comenzar a entrenar tu modelo
6. El progreso del entrenamiento se mostrará en el panel de gráficos

### 🧪 Ejecutando un Backtest

1. Selecciona el modo "backtest" del menú desplegable
2. Haz clic en "Start" para abrir el diálogo de selección de modelo
3. Elige tu modelo entrenado
4. Selecciona el archivo de datos para el backtesting
5. El sistema ejecutará el backtest y mostrará los resultados en el gráfico de rendimiento

### 💹 Trading en Vivo

1. Asegúrate de que NinjaTrader 8 esté ejecutándose con el indicador RLExecutor en tu gráfico
2. En el Sistema de Trading RL, introduce los detalles de conexión:
   - IP del Servidor: `127.0.0.1` (predeterminado)
   - Puerto de Datos: `5000` (predeterminado)
3. Haz clic en "Connect" para establecer una conexión con NinjaTrader
4. Selecciona el modo "server" del menú desplegable
5. Haz clic en "Start" para iniciar el modo servidor
6. Activa el interruptor "Auto Trading" para habilitar el trading automatizado

## 📱 Guía de la Interfaz de Usuario

### Panel de Gráficos

El Panel de Gráficos muestra dos gráficos principales:
- **Gráfico de Precios (Superior)**: Muestra la evolución del precio del mercado con señales de compra (triángulos verdes), señales de venta (triángulos rojos) y señales de salida (círculos azules)
- **Gráfico de Balance de Cuenta (Inferior)**: Muestra tu balance de cuenta a lo largo del tiempo

### Panel de Estadísticas

Muestra métricas clave de rendimiento:
- Balance de la Cuenta
- P&L Total (Ganancias y Pérdidas)
- Número de Operaciones
- Tasa de Éxito
- Posición Actual
- Estado de Conexión

### Panel de Operaciones

Lista la actividad de trading reciente con:
- Hora
- Acción (Compra/Venta/Salida)
- Precio
- P&L

### Panel de Control

Gestiona la operación del sistema con:
- Configuración de Conexión (IP/Puerto)
- Modo de Operación (Entrenamiento/Backtest/Servidor)
- Botones de Control (Iniciar/Pausar/Detener)
- Interruptor de Trading Automático

## ⚙️ Configuración Avanzada

### Configuración de Aprendizaje por Refuerzo

Configura parámetros de entrenamiento:
- **Algoritmo**: Elige entre algoritmos PPO, A2C o DQN
- **Pasos de Tiempo**: Número de pasos para el entrenamiento (mayor = mejor aprendizaje pero más lento)
- **Tasa de Aprendizaje**: Qué tan rápido se adapta el modelo a nueva información
- **Tamaño de Lote**: Cantidad de datos procesados en cada paso de entrenamiento
- **Dispositivo**: Selecciona entre entrenamiento en CPU o GPU

### Configuración del Sistema de Recompensas

Ajusta cómo la IA evalúa el rendimiento del trading:
- **Escalado de Recompensa**: Controla la magnitud general de las recompensas
- **Penalización por Inactividad**: Desalienta la espera excesiva
- **Factor de Drawdown**: Penalización por disminuciones en el valor de la cuenta
- **Bonificación por Tasa de Éxito**: Recompensa estrategias con alta precisión

### Presets de Trading

Elige entre configuraciones preestablecidas:
1. **Trading de Alta Frecuencia**:
   - Optimizado para operaciones a corto plazo
   - Valores de stop-loss y take-profit más pequeños
   - Mayor factor de decaimiento temporal
   
2. **Trading de Posición**:
   - Diseñado para posiciones a más largo plazo
   - Menor penalización por inactividad
   - Mayor bonificación por tasa de éxito
   
3. **Trading de Tendencia**:
   - Configurado para seguir tendencias del mercado
   - Mayor ratio riesgo/recompensa (1:3)
   - Recompensas normalizadas para estabilidad

## ❓ Solución de Problemas

### Problemas de Conexión

Si tienes problemas para conectarte a NinjaTrader:
- Asegúrate de que NinjaTrader 8 esté en ejecución
- Verifica que el indicador RLExecutor esté añadido a tu gráfico
- Comprueba que la configuración de IP y puerto coincida
- Intenta reiniciar tanto NinjaTrader como el Sistema de Trading RL

### Errores de OpenMP

Si encuentras errores relacionados con OpenMP, esto suele deberse a conflictos de bibliotecas. La solución ya está implementada en el código mediante la configuración `KMP_DUPLICATE_LIB_OK=TRUE`, pero si los problemas persisten:

1. Abre el Símbolo del Sistema como Administrador
2. Ejecuta:
   ```
   set KMP_DUPLICATE_LIB_OK=TRUE
   ```
3. Inicia la aplicación desde el mismo Símbolo del Sistema

### Problemas de Entrenamiento

Si tu modelo no está aprendiendo de manera efectiva:
- Asegúrate de tener datos suficientes (se recomiendan al menos 100 días de trading)
- Prueba una configuración de preset diferente
- Ajusta la tasa de aprendizaje o el tamaño de lote
- Revisa los componentes de recompensa en los logs para identificar problemas

## 📞 Soporte y Recursos

Para ayuda y orientación:
- Revisa la documentación en la carpeta `doc`
- Consulta los archivos de ayuda accesibles desde la barra de menú (Ayuda → Interfaz/Gráficos/Controles/Trading)
- Envía problemas a nuestro repositorio de GitHub
- Únete a nuestro foro comunitario para discusiones

## 📊 Estructura del Proyecto

```
RL-Trading-System/
├── src/               # Código fuente
├── doc/               # Documentación y archivos de ayuda
├── data/              # Datos de trading para entrenamiento/backtesting
├── models/            # Modelos RL guardados
├── logs/              # Logs de la aplicación
├── integrations/      # Archivos de integración con NinjaTrader
├── output/            # Resultados de entrenamiento y backtest
└── requirements.txt   # Paquetes de Python requeridos
```

---

💡 **Recuerda**: El trading implica riesgo financiero. Siempre prueba tus modelos exhaustivamente en backtesting y trading en papel antes de usarlos con dinero real.

¡Feliz Trading! 📈