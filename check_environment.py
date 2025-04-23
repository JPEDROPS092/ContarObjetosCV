import sys
import os
import platform

def check_environment():
    print("=== Verificação do Ambiente ===")
    print(f"Sistema Operacional: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Não definido')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Não definido')}")
    print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', 'Não definido')}")
    print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', 'Não definido')}")
    print(f"GDK_BACKEND: {os.environ.get('GDK_BACKEND', 'Não definido')}")
    
    try:
        import tkinter as tk
        print("Tkinter: OK")
    except ImportError as e:
        print(f"Tkinter: ERRO - {e}")
    
    try:
        import cv2
        print(f"OpenCV: OK (versão {cv2.__version__})")
    except ImportError as e:
        print(f"OpenCV: ERRO - {e}")
    
    try:
        from PIL import Image
        print(f"Pillow: OK (versão {Image.__version__})")
    except ImportError as e:
        print(f"Pillow: ERRO - {e}")
    
    try:
        import matplotlib
        print(f"Matplotlib: OK (versão {matplotlib.__version__})")
    except ImportError as e:
        print(f"Matplotlib: ERRO - {e}")

if __name__ == "__main__":
    check_environment() 