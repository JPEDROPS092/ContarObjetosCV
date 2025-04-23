#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IndustriaCount - Sistema de Contagem de Objetos em Linha de Montagem
Aplicação principal para iniciar o sistema
"""

import os
import sys
import tkinter as tk
from src.ui.app import ObjectCounterApp

def check_dependencies():
    """Verifica se todas as dependências necessárias estão instaladas"""
    try:
        import cv2
        import numpy
        import PIL
        import matplotlib
        
        # Verificar se PyTorch/Ultralytics está disponível
        try:
            import torch
            import ultralytics
            torch_available = True
        except ImportError:
            torch_available = False
            print("AVISO: PyTorch/Ultralytics não encontrado. A funcionalidade de detecção será limitada.")
        
        # Verificar outras dependências opcionais
        try:
            import rich
            import requests
            import tqdm
        except ImportError:
            print("AVISO: Algumas dependências opcionais não foram encontradas. A funcionalidade de download de modelos pode ser limitada.")
        
        return True
    except ImportError as e:
        print(f"ERRO: Dependência não encontrada: {e}")
        print("Por favor, instale todas as dependências necessárias com: pip install -r requirements.txt")
        return False

def setup_environment():
    """Configura o ambiente para a aplicação"""
    # Adicionar o diretório raiz ao path para importações relativas
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Criar diretórios necessários se não existirem
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Configurar variáveis de ambiente
    os.environ['PYTHONUNBUFFERED'] = '1'  # Saída não bufferizada

def main():
    """Função principal para iniciar a aplicação"""
    print("Iniciando IndustriaCount...")
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Configurar ambiente
    setup_environment()
    
    # Iniciar a aplicação
    root = tk.Tk()
    app = ObjectCounterApp(root, "IndustriaCount - Sistema de Contagem de Objetos")
    root.mainloop()

if __name__ == "__main__":
    main()
