#!/bin/bash

# Script para eliminar archivos temporales y obsoletos

echo "Este script eliminará los siguientes archivos temporales:"
echo "- find_gui_class.py (script de análisis temporal)"
echo "- find_main_class.py (script de análisis temporal)"
echo "- fix_main.py (script de corrección temporal)"
echo "- fix_main_import.py (script de corrección temporal)"
echo "- ttk_fix.py (corrección para problemas de importación)"
echo "- update_main.py (script de actualización temporal)"
echo "- update_main_imports.py (script de actualización temporal)"
echo "- ClaudeTAI.py (script temporal para pruebas)"
echo "- IntegratedTradingSystemGUI.py (reemplazado por nueva implementación)"
echo "- CreateSynteticDATA.py (funcionalidad duplicada en TrainingManager)"
echo "- update_imports.py (script de utilidad temporal)"
echo "- cleanup_temporary_files.py (este script ahora es obsoleto)"

echo -e "\nTambién moverá los archivos de log de la raíz a la carpeta logs/"

echo -e "\nUsage: ./cleanup_files.sh"
echo "NOTE: Execute this script only when you are sure you don't need these files anymore."

# Crear la carpeta logs si no existe
mkdir -p logs

# Mover archivos de log a la carpeta logs
for log_file in rl_*.log; do
    if [ -f "$log_file" ]; then
        echo "Moviendo $log_file a logs/"
        mv "$log_file" logs/
    fi
done

echo "Limpieza completada."
