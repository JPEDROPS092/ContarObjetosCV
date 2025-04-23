#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para exportação de relatórios em PDF
"""

import os
import datetime
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
from PIL import Image

def create_pdf_report(stats, frame_image=None, plot_image=None, output_path=None):
    """
    Cria um relatório PDF com as estatísticas de contagem usando matplotlib
    
    Args:
        stats: Objeto ObjectStatistics com os dados
        frame_image: Imagem do último frame processado (opcional)
        plot_image: Imagem do gráfico de contagem (opcional)
        output_path: Caminho para salvar o PDF (opcional)
        
    Returns:
        str: Caminho do arquivo PDF gerado
    """
    # Se não for especificado um caminho, criar um com timestamp
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"relatorio_contagem_{timestamp}.pdf"
    
    # Criar diretório temporário para imagens
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    # Criar PDF usando matplotlib
    with PdfPages(output_path) as pdf:
        # Página de título
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, "Relatório de Contagem de Objetos", 
                 fontsize=24, ha='center', va='center')
        
        # Informações gerais
        info_text = f"Data do Relatório: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        
        if stats.start_time:
            info_text += f"Início da Contagem: {stats.start_time.strftime('%d/%m/%Y %H:%M:%S')}\n"
            info_text += f"Fim da Contagem: {stats.last_update_time.strftime('%d/%m/%Y %H:%M:%S')}\n\n"
            
            # Calcular duração
            duration_seconds = (stats.last_update_time - stats.start_time).total_seconds()
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            info_text += f"Duração: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n"
        
        info_text += f"Total de Objetos Contados: {stats.total_counted}\n\n"
        
        if stats.count_rate > 0:
            info_text += f"Taxa de Contagem: {stats.count_rate:.1f} objetos/minuto\n"
            info_text += f"Taxa de Contagem: {stats.count_rate * 60:.1f} objetos/hora\n"
        
        plt.text(0.5, 0.5, info_text, fontsize=12, ha='center', va='center', 
                 transform=plt.gca().transAxes, linespacing=1.5)
        
        # Adicionar rodapé
        plt.text(0.5, 0.05, f"Gerado por IndustriaCount - {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                 fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
        
        # Salvar página
        pdf.savefig()
        plt.close()
        
        # Adicionar imagem do último frame, se disponível
        if frame_image is not None:
            # Converter para RGB se necessário
            if len(frame_image.shape) == 3 and frame_image.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_image
                
            plt.figure(figsize=(8.5, 11))
            plt.title("Última Imagem Processada", fontsize=16)
            plt.imshow(frame_rgb)
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # Estatísticas por classe
        if stats.counts_per_class:
            plt.figure(figsize=(8.5, 11))
            plt.subplot(2, 1, 1)
            plt.title("Contagem por Classe", fontsize=16)
            
            # Criar tabela de contagem por classe
            classes = []
            counts = []
            percentages = []
            
            for cls, count in sorted(stats.counts_per_class.items(), key=lambda x: x[1], reverse=True):
                classes.append(cls)
                counts.append(count)
                percentage = 100.0 * count / stats.total_counted if stats.total_counted > 0 else 0
                percentages.append(f"{percentage:.1f}%")
            
            # Limitar a 10 classes para melhor visualização
            if len(classes) > 10:
                classes = classes[:10]
                counts = counts[:10]
                percentages = percentages[:10]
            
            # Criar tabela como texto
            table_text = "Classe | Quantidade | Porcentagem\n"
            table_text += "-------|------------|------------\n"
            
            for i in range(len(classes)):
                table_text += f"{classes[i]} | {counts[i]} | {percentages[i]}\n"
            
            plt.text(0.1, 0.5, table_text, fontsize=10, va='center', 
                     transform=plt.gca().transAxes, family='monospace')
            
            # Criar gráfico de barras para classes
            plt.subplot(2, 1, 2)
            plt.bar(classes, counts, color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel('Quantidade')
            plt.title('Distribuição de Classes')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Salvar página
            pdf.savefig()
            plt.close()
        
        # Estatísticas por cor (se disponível)
        if any(stats.counts_per_color_per_class.values()):
            plt.figure(figsize=(8.5, 11))
            plt.subplot(2, 1, 1)
            plt.title("Contagem por Cor", fontsize=16)
            
            # Consolidar contagem por cor (todas as classes)
            color_counts = {}
            for class_colors in stats.counts_per_color_per_class.values():
                for color, count in class_colors.items():
                    if color not in color_counts:
                        color_counts[color] = 0
                    color_counts[color] += count
            
            # Criar tabela como texto
            colors = []
            counts = []
            percentages = []
            
            for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
                colors.append(color)
                counts.append(count)
                percentage = 100.0 * count / stats.total_counted if stats.total_counted > 0 else 0
                percentages.append(f"{percentage:.1f}%")
            
            # Criar tabela como texto
            table_text = "Cor | Quantidade | Porcentagem\n"
            table_text += "-----|------------|------------\n"
            
            for i in range(len(colors)):
                table_text += f"{colors[i]} | {counts[i]} | {percentages[i]}\n"
            
            plt.text(0.1, 0.5, table_text, fontsize=10, va='center', 
                     transform=plt.gca().transAxes, family='monospace')
            
            # Criar gráfico de pizza para cores
            plt.subplot(2, 1, 2)
            plt.pie(counts, labels=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Distribuição por Cor')
            
            # Salvar página
            pdf.savefig()
            plt.close()
        
        # Gráfico de contagem ao longo do tempo
        if plot_image is not None:
            # Converter para RGB se necessário
            if len(plot_image.shape) == 3 and plot_image.shape[2] == 3:
                plot_rgb = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
            else:
                plot_rgb = plot_image
                
            plt.figure(figsize=(8.5, 11))
            plt.title("Contagem ao Longo do Tempo", fontsize=16)
            plt.imshow(plot_rgb)
            plt.axis('off')
            pdf.savefig()
            plt.close()
    
    return output_path


def capture_plot_image(fig):
    """
    Captura uma imagem do gráfico matplotlib
    
    Args:
        fig: Figura matplotlib
        
    Returns:
        numpy.ndarray: Imagem do gráfico em formato OpenCV (BGR)
    """
    # Salvar temporariamente
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        fig.savefig(tmp.name)
        # Ler com OpenCV
        img = cv2.imread(tmp.name)
    return img
