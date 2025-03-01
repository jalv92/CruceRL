#\!/usr/bin/env python
import os
import sys

# Solucionar conflicto de bibliotecas OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Añadir la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Verificar si se pasan argumentos
if len(sys.argv) > 1:
    # Si hay argumentos, usar main.py para modo comando
    from src.main import main
    sys.exit(main())
else:
    # Si no hay argumentos, iniciar la GUI completa
    from src.MainGUI import main
    main()
