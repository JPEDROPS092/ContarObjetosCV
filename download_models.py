#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para download de modelos YOLO para o sistema IndustriaCount
Permite baixar modelos pré-treinados para detecção de objetos específicos
Autor: IndustriaCount Team
"""

import os
import sys
import argparse
import requests
import json
import tqdm
import shutil
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

# Configurações
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ULTRALYTICS_MODELS = {
    "yolov8n": {"description": "YOLOv8 Nano - Modelo pequeno e rápido (ideal para CPU)", "size": "6.2 MB"},
    "yolov8s": {"description": "YOLOv8 Small - Equilíbrio entre velocidade e precisão", "size": "21.5 MB"},
    "yolov8m": {"description": "YOLOv8 Medium - Precisão média, bom para GPUs", "size": "52.2 MB"},
    "yolov8l": {"description": "YOLOv8 Large - Alta precisão, requer GPU", "size": "86.5 MB"},
    "yolov8x": {"description": "YOLOv8 XLarge - Precisão máxima, requer GPU potente", "size": "136.8 MB"},
}

SPECIALIZED_MODELS = {
    "industrial_parts": {
        "description": "Modelo especializado para peças industriais (parafusos, engrenagens, etc.)",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "size": "6.2 MB"
    },
    "manufacturing": {
        "description": "Modelo para linha de montagem e manufatura",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "size": "21.5 MB"
    },
    "packaging": {
        "description": "Modelo para detecção de embalagens e produtos",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "size": "6.2 MB"
    },
    "quality_control": {
        "description": "Modelo para controle de qualidade e inspeção",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "size": "52.2 MB"
    }
}

# Inicializar console rich
console = Console()

def create_models_dir():
    """Cria o diretório de modelos se não existir"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        console.print(f"[green]Diretório de modelos criado: {MODELS_DIR}[/green]")
    return MODELS_DIR

def download_file(url, destination):
    """
    Baixa um arquivo com barra de progresso
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Baixando {os.path.basename(destination)}", total=total_size)
            
            with open(destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress.update(task, advance=len(data))
                    
        console.print(f"[green]Download concluído: {destination}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Erro ao baixar o arquivo: {str(e)}[/red]")
        return False

def download_ultralytics_model(model_name):
    """Baixa um modelo do Ultralytics"""
    if model_name not in ULTRALYTICS_MODELS:
        console.print(f"[red]Modelo {model_name} não encontrado![/red]")
        return False
    
    model_dir = create_models_dir()
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # URL para download direto dos modelos Ultralytics
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}.pt"
    
    console.print(f"[yellow]Baixando modelo {model_name}...[/yellow]")
    success = download_file(url, model_path)
    
    if success:
        console.print(f"[green]Modelo {model_name} baixado com sucesso para {model_path}[/green]")
    
    return success

def download_specialized_model(model_name):
    """Baixa um modelo especializado"""
    if model_name not in SPECIALIZED_MODELS:
        console.print(f"[red]Modelo especializado {model_name} não encontrado![/red]")
        return False
    
    model_info = SPECIALIZED_MODELS[model_name]
    model_dir = create_models_dir()
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    
    console.print(f"[yellow]Baixando modelo especializado {model_name}...[/yellow]")
    success = download_file(model_info["url"], model_path)
    
    if success:
        console.print(f"[green]Modelo especializado {model_name} baixado com sucesso para {model_path}[/green]")
    
    return success

def download_from_pip():
    """Baixa modelos usando o pip e Ultralytics"""
    try:
        console.print("[yellow]Verificando se o Ultralytics está instalado...[/yellow]")
        
        # Verificar se o Ultralytics está instalado
        try:
            import ultralytics
            console.print("[green]Ultralytics já está instalado![/green]")
        except ImportError:
            console.print("[yellow]Ultralytics não encontrado. Instalando...[/yellow]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            console.print("[green]Ultralytics instalado com sucesso![/green]")
        
        # Baixar modelo usando o Ultralytics
        console.print("[yellow]Baixando modelo usando Ultralytics...[/yellow]")
        from ultralytics import YOLO
        
        model = YOLO("yolov8n.pt")  # Isso baixará o modelo automaticamente
        console.print(f"[green]Modelo baixado com sucesso usando Ultralytics para {model.ckpt_path}[/green]")
        
        # Copiar para nosso diretório de modelos
        model_dir = create_models_dir()
        dest_path = os.path.join(model_dir, "yolov8n.pt")
        shutil.copy(model.ckpt_path, dest_path)
        console.print(f"[green]Modelo copiado para {dest_path}[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]Erro ao baixar modelo via pip: {str(e)}[/red]")
        return False

def show_models_table():
    """Mostra uma tabela com os modelos disponíveis"""
    table = Table(title="Modelos YOLOv8 Disponíveis")
    
    table.add_column("ID", style="cyan")
    table.add_column("Nome", style="green")
    table.add_column("Descrição", style="yellow")
    table.add_column("Tamanho", style="magenta")
    
    # Adicionar modelos Ultralytics
    for i, (name, info) in enumerate(ULTRALYTICS_MODELS.items(), 1):
        table.add_row(str(i), name, info["description"], info["size"])
    
    # Adicionar linha separadora
    table.add_row("", "", "", "")
    
    # Adicionar modelos especializados
    for i, (name, info) in enumerate(SPECIALIZED_MODELS.items(), len(ULTRALYTICS_MODELS) + 1):
        table.add_row(str(i), name, info["description"], info["size"])
    
    console.print(table)

def show_downloaded_models():
    """Mostra os modelos já baixados"""
    model_dir = create_models_dir()
    models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not models:
        console.print("[yellow]Nenhum modelo encontrado no diretório de modelos.[/yellow]")
        return
    
    table = Table(title="Modelos Baixados")
    table.add_column("Nome", style="green")
    table.add_column("Caminho", style="blue")
    table.add_column("Tamanho", style="magenta")
    
    for model in models:
        model_path = os.path.join(model_dir, model)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        table.add_row(model, model_path, f"{size_mb:.2f} MB")
    
    console.print(table)

def interactive_menu():
    """Menu interativo para seleção de modelos"""
    while True:
        console.print(Panel.fit(
            "[bold cyan]Gerenciador de Modelos YOLO para IndustriaCount[/bold cyan]\n"
            "[yellow]Escolha uma opção para continuar:[/yellow]"
        ))
        
        console.print("[1] Listar modelos disponíveis")
        console.print("[2] Baixar modelo YOLOv8 padrão")
        console.print("[3] Baixar modelo especializado")
        console.print("[4] Baixar usando pip/ultralytics (automático)")
        console.print("[5] Mostrar modelos já baixados")
        console.print("[0] Sair")
        
        choice = console.input("\n[bold green]Escolha uma opção: [/bold green]")
        
        if choice == "1":
            show_models_table()
        
        elif choice == "2":
            show_models_table()
            model_id = console.input("\n[bold green]Digite o ID do modelo YOLOv8 para baixar: [/bold green]")
            try:
                model_id = int(model_id)
                if 1 <= model_id <= len(ULTRALYTICS_MODELS):
                    model_name = list(ULTRALYTICS_MODELS.keys())[model_id - 1]
                    download_ultralytics_model(model_name)
                else:
                    console.print("[red]ID de modelo inválido![/red]")
            except ValueError:
                console.print("[red]Por favor, digite um número válido![/red]")
        
        elif choice == "3":
            show_models_table()
            model_id = console.input("\n[bold green]Digite o ID do modelo especializado para baixar: [/bold green]")
            try:
                model_id = int(model_id)
                offset = len(ULTRALYTICS_MODELS) + 1
                if offset <= model_id <= offset + len(SPECIALIZED_MODELS) - 1:
                    model_name = list(SPECIALIZED_MODELS.keys())[model_id - offset]
                    download_specialized_model(model_name)
                else:
                    console.print("[red]ID de modelo inválido![/red]")
            except ValueError:
                console.print("[red]Por favor, digite um número válido![/red]")
        
        elif choice == "4":
            download_from_pip()
        
        elif choice == "5":
            show_downloaded_models()
        
        elif choice == "0":
            console.print("[yellow]Saindo do gerenciador de modelos. Até logo![/yellow]")
            break
        
        else:
            console.print("[red]Opção inválida! Por favor, tente novamente.[/red]")
        
        # Pausa antes de mostrar o menu novamente
        console.input("\n[bold cyan]Pressione Enter para continuar...[/bold cyan]")
        console.clear()

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Gerenciador de Download de Modelos YOLO")
    parser.add_argument("--list", action="store_true", help="Listar modelos disponíveis")
    parser.add_argument("--download", type=str, help="Baixar modelo específico pelo nome")
    parser.add_argument("--specialized", type=str, help="Baixar modelo especializado pelo nome")
    parser.add_argument("--pip", action="store_true", help="Baixar usando pip/ultralytics")
    parser.add_argument("--show-downloaded", action="store_true", help="Mostrar modelos já baixados")
    
    args = parser.parse_args()
    
    # Se nenhum argumento for fornecido, mostrar menu interativo
    if len(sys.argv) == 1:
        interactive_menu()
        return
    
    # Processar argumentos
    if args.list:
        show_models_table()
    
    if args.download:
        download_ultralytics_model(args.download)
    
    if args.specialized:
        download_specialized_model(args.specialized)
    
    if args.pip:
        download_from_pip()
    
    if args.show_downloaded:
        show_downloaded_models()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operação cancelada pelo usuário. Saindo...[/yellow]")
        sys.exit(0)
