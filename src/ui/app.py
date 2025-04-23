#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aplicação principal do sistema IndustriaCount
Interface gráfica para contagem de objetos em linha de montagem
"""

import os
import sys
import time
import datetime
import traceback
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib
matplotlib.use("TkAgg")  # Set backend before other matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from collections import defaultdict, deque

# Adicionar o diretório raiz ao path para importações relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar módulos do projeto
from src.core.model_manager import ModelManager, TORCH_AVAILABLE
from src.core.statistics import ObjectStatistics
from src.core.tracker import ObjectTracker
from src.utils.image_processing import (
    get_object_color, check_line_crossing, draw_bounding_box,
    LINE_COLOR, LINE_THICKNESS, TRACKING_DOT_COLOR, COUNTING_TEXT_COLOR
)
from src.utils.pdf_exporter import create_pdf_report, capture_plot_image
from src.ui.ui_components import create_ui

# Configurações padrão
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models/yolov8n.pt")
DEFAULT_CONFIDENCE = 0.45

class ObjectCounterApp:
    """
    Aplicação principal para contagem de objetos em linha de montagem
    """
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x900")  # Increased size for new elements
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        # --- State Variables ---
        self.cap = None
        self.is_processing_active = False
        self.current_source_path = None
        self.current_mode = "None"  # None, Webcam, Video, Image
        self.display_width = 640
        self.display_height = 480
        self.last_original_frame = None
        self.last_processed_frame = None  # Store the frame *with* drawings
        self.last_display_scale = 1.0
        self.last_display_offsets = (0, 0)
        self.last_display_dims = (0, 0)
        
        # --- Performance Optimization ---
        self.skip_frames = tk.IntVar(value=0)  # Número de frames para pular (0 = processar todos)
        self.frame_count = 0  # Contador de frames para controle de pulo
        self.process_resolution = tk.DoubleVar(value=1.0)  # Escala de resolução para processamento (1.0 = resolução original)
        self.fps_history = deque(maxlen=30)  # Para calcular FPS médio
        self.last_frame_time = time.time()
        self.current_fps = 0

        # --- Model Management ---
        self.model_manager = ModelManager(self.log_to_console)
        self.confidence_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.selected_classes = {}  # class_id -> BooleanVar (checkbox state)
        self.classes_to_detect = None  # Lista de IDs de classes para detectar (None = todas)

        # --- Line Crossing & Tracking ---
        self.line_start = None  # (x, y) in *original* frame coordinates
        self.line_end = None
        self.drawing_line = False
        self.enable_line_drawing = tk.BooleanVar(value=False)
        self.tracker = ObjectTracker(self.log_to_console)
        self.crossing_direction = tk.StringVar(value="Esquerda -> Direita")  # Options: "Esquerda -> Direita", "Direita -> Esquerda", "Cima -> Baixo", "Baixo -> Cima"

        # --- Statistics ---
        self.stats = ObjectStatistics()

        # --- Plotting Settings ---
        self.plot_data = defaultdict(lambda: deque(maxlen=100))  # class -> deque[(time, count)]
        self.fig = None
        self.ax = None
        self.canvas = None
        self.selected_plot_class = tk.StringVar(value="Total")  # Class to display on plot

        # --- Create UI Elements ---
        self._create_ui()

        # --- Final Setup ---
        self.delay = 30  # ms between frames
        self.setup_plot()
        self.set_placeholder_image()
        self.update_status_label()
        self.update_stats_display()  # Show initial stats message
        self.update_plot_class_options()  # Populate plot dropdown
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log_to_console("Aplicação iniciada. Carregue um modelo e uma fonte de vídeo/webcam.")

        # Try to load default model if it exists
        if TORCH_AVAILABLE and os.path.exists(DEFAULT_MODEL_PATH):
            self.model_manager.load_model(DEFAULT_MODEL_PATH)
            self._update_model_status_label()
            self.update_plot_class_options()
        else:
            self.log_to_console(f"Modelo padrão '{DEFAULT_MODEL_PATH}' não encontrado ou PyTorch/Ultralytics indisponível.")
            
    def _create_ui(self):
        """Cria a interface do usuário utilizando o módulo ui_components"""
        create_ui(self)

    def log_to_console(self, message, is_error=False):
        """Adiciona uma mensagem ao console da aplicação"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = "[ERRO]" if is_error else "[INFO]"
        formatted_message = f"{timestamp} {prefix} {message}\n"
        
        # Ensure thread safety when updating UI
        self.window.after(0, self._update_console, formatted_message)
        
        # Also print to standard output for debugging
        print(formatted_message.strip())
        
        if is_error:
            print(traceback.format_exc())
    
    def _update_console(self, message):
        """Atualiza o widget de console com uma nova mensagem (thread-safe)"""
        self.console_output.configure(state=tk.NORMAL)
        self.console_output.insert(tk.END, message)
        self.console_output.see(tk.END)  # Auto-scroll to bottom
        self.console_output.configure(state=tk.DISABLED)
    
    def update_status_label(self):
        """Atualiza o rótulo de status com informações atuais"""
        status_text = f"Status: {'Processando' if self.is_processing_active else 'Parado'}"
        
        if self.current_mode != "None":
            status_text += f" | Fonte: {self.current_mode}"
            
            if self.current_mode == "Video" and self.cap is not None:
                # Get current frame position and total frames
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress = current_frame / total_frames * 100
                    status_text += f" | Progresso: {progress:.1f}%"
        
        # Adicionar informação de FPS
        if self.is_processing_active:
            status_text += f" | FPS: {self.current_fps:.1f}"
            
        self.status_label.config(text=status_text)
    
    def update_stats_display(self):
        """Atualiza a exibição de estatísticas"""
        summary = self.stats.get_summary()
        
        # Update the statistics text widget
        self.stats_output.configure(state=tk.NORMAL)
        self.stats_output.delete(1.0, tk.END)
        self.stats_output.insert(tk.END, summary)
        self.stats_output.configure(state=tk.DISABLED)
        
        # Update the count label
        self.count_label.config(text=f"Total Contado: {self.stats.total_counted}")
        
        # Atualizar estatísticas em tempo real
        self.rate_label.config(text=f"{self.stats.count_rate:.1f} obj/min")
        
        # Atualizar informações da última classe e cor contadas
        if self.stats.last_counted_class and self.stats.last_counted_color:
            self.last_class_label.config(text=self.stats.last_counted_class)
            self.last_color_label.config(text=self.stats.last_counted_color)
    
    def _update_model_status_label(self):
        """Atualiza o rótulo de status do modelo"""
        if self.model_manager.current_model:
            model_name = os.path.basename(self.model_manager.model_path)
            self.model_status_label.config(text=f"Modelo: {model_name}")
            # Atualizar a lista de classes disponíveis
            self.update_class_checkboxes()
        else:
            self.model_status_label.config(text="Modelo: Nenhum carregado")
    
    def _update_confidence_label(self, *args):
        """Atualiza o rótulo de confiança quando o slider é movido"""
        self.confidence_label.config(text=f"{self.confidence_threshold.get():.2f}")
    
    def _on_toggle_draw_mode(self):
        """Manipula a alternância do modo de desenho da linha"""
        if self.enable_line_drawing.get():
            self.log_to_console("Modo de desenho de linha ativado. Clique e arraste para desenhar a linha de contagem.")
        else:
            self.log_to_console("Modo de desenho de linha desativado.")
    
    def set_placeholder_image(self):
        """Define uma imagem de espaço reservado quando nenhum vídeo está sendo processado"""
        # Create a blank image with text
        width, height = self.display_width, self.display_height
        img = Image.new('RGB', (width, height), color=(64, 64, 64))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text
        text = "Nenhuma fonte de vídeo carregada"
        text_width = draw.textlength(text, font=font)
        position = ((width - text_width) // 2, height // 2 - 30)
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        # Add instructions
        instructions = "Use os botões acima para carregar uma fonte"
        instructions_width = draw.textlength(instructions, font=font)
        position = ((width - instructions_width) // 2, height // 2 + 10)
        draw.text(position, instructions, fill=(200, 200, 200), font=font)
        
        # Convert to PhotoImage and display
        self.placeholder_image = ImageTk.PhotoImage(image=img)
        self.display_label.config(image=self.placeholder_image)
        self.display_label.image = self.placeholder_image  # Keep a reference
    
    def setup_plot(self):
        """Configura o gráfico de contagem acumulada"""
        # Create figure and axis
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Style the plot
        self.ax.set_title("Contagem Acumulada (Tempo Real)")
        self.ax.set_xlabel("Tempo")
        self.ax.set_ylabel("Contagem")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurar timer para atualização automática do gráfico em tempo real
        self._plot_update_interval = 500  # ms (0.5 segundos para atualizações mais frequentes)
        
        # Iniciar animação para atualizações suaves
        self.ani = animation.FuncAnimation(
            self.fig, 
            self._animate_plot, 
            interval=self._plot_update_interval,
            blit=False
        )
        
        # Iniciar atualizações
        self._schedule_plot_update()
    
    def _schedule_plot_update(self):
        """Agenda a próxima atualização do gráfico"""
        if self.is_processing_active:
            # A animação já está cuidando das atualizações visuais
            pass
        self.window.after(self._plot_update_interval, self._schedule_plot_update)
    
    def _animate_plot(self, i):
        """Função de animação para atualização suave do gráfico"""
        if not self.is_processing_active:
            return
        
        self._update_plot_data()
        return self.ax,
    
    def _update_plot_data(self):
        """Atualiza os dados do gráfico sem redesenhar completamente"""
        if not self.ax:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        # Get selected class
        selected_class = self.selected_plot_class.get()
        
        # Prepare data for plotting
        if selected_class == "Total":
            # Sum all classes for total count
            timestamps = []
            counts = []
            
            # Get all unique timestamps across all classes
            all_timestamps = set()
            for class_data in self.stats.time_series_counts.values():
                for ts, _ in class_data:
                    all_timestamps.add(ts)
            
            # Sort timestamps
            all_timestamps = sorted(all_timestamps)
            
            # Calculate cumulative count at each timestamp
            cumulative_count = 0
            for ts in all_timestamps:
                # Find the count at this timestamp
                for class_name, data_points in self.stats.time_series_counts.items():
                    for data_ts, count in data_points:
                        if data_ts == ts:
                            # Use the difference from previous count to avoid double counting
                            if len(timestamps) > 0:
                                # Find previous count for this class
                                prev_count = 0
                                for prev_ts, prev_cnt in data_points:
                                    if prev_ts < ts:
                                        prev_count = prev_cnt
                                        break
                                cumulative_count += (count - prev_count)
                            else:
                                cumulative_count += count
                
                timestamps.append(ts)
                counts.append(cumulative_count)
        else:
            # Plot specific class data
            if selected_class in self.stats.time_series_counts:
                data = self.stats.time_series_counts[selected_class]
                timestamps = [point[0] for point in data]
                counts = [point[1] for point in data]
            else:
                timestamps = []
                counts = []
        
        # Plot the data if we have any
        if timestamps:
            # Criar um gráfico mais informativo com cores e marcadores
            self.ax.plot(timestamps, counts, marker='o', linestyle='-', markersize=4, 
                         color='#3366CC', linewidth=2, alpha=0.8)
            
            # Adicionar área sombreada sob a curva para destacar o volume
            self.ax.fill_between(timestamps, 0, counts, color='#3366CC', alpha=0.2)
            
            # Adicionar rótulos nos pontos de dados mais recentes
            if len(timestamps) > 0:
                # Mostrar o valor mais recente
                last_ts = timestamps[-1]
                last_count = counts[-1]
                self.ax.annotate(f'{last_count}', 
                                xy=(last_ts, last_count),
                                xytext=(10, 0),
                                textcoords='offset points',
                                ha='left', va='center',
                                fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
            
            # Format x-axis as time
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.fig.autofmt_xdate(rotation=30)  # Rotate date labels
            
            # Set y-axis to start from 0 with um pouco de margem superior
            if counts:
                max_count = max(counts)
                self.ax.set_ylim(bottom=0, top=max_count * 1.15)
            else:
                self.ax.set_ylim(bottom=0)
            
            # Add grid
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set title based on selected class with taxa de contagem
            if self.stats.count_rate > 0:
                self.ax.set_title(f"Contagem em Tempo Real - {selected_class} ({self.stats.count_rate:.1f} obj/min)")
            else:
                self.ax.set_title(f"Contagem em Tempo Real - {selected_class}")
                
            # Adicionar uma linha de tendência se houver dados suficientes
            if len(timestamps) > 2:
                try:
                    # Converter timestamps para números para regressão linear
                    x_numeric = mdates.date2num(timestamps)
                    z = np.polyfit(x_numeric, counts, 1)
                    p = np.poly1d(z)
                    self.ax.plot(timestamps, p(x_numeric), "r--", alpha=0.7, linewidth=1)
                except:
                    # Ignorar erros na linha de tendência
                    pass
        else:
            self.ax.set_title("Sem dados para exibir")
        
        # Update the canvas
        self.ax.set_xlabel("Tempo")
        self.ax.set_ylabel("Contagem")
        
    def update_plot(self, *args):
        """Atualiza o gráfico com os dados mais recentes (chamado manualmente)"""
        self._update_plot_data()
        self.canvas.draw_idle()  # Redesenha o canvas apenas quando necessário
    
    def update_plot_class_options(self):
        """Atualiza as opções de classe para o gráfico"""
        # Always include "Total" option
        options = ["Total"]
        
        # Add class names from the model if available
        if self.model_manager.class_names:
            options.extend(self.model_manager.class_names.values())
        
        # Update the combobox
        self.combo_plot_class['values'] = options
        
        # Set to "Total" if current selection is not in options
        if self.selected_plot_class.get() not in options:
            self.selected_plot_class.set("Total")
    
    def on_closing(self):
        """Manipula o evento de fechamento da janela"""
        if self.is_processing_active:
            self.stop_processing()
        
        if self.cap is not None:
            self.cap.release()
        
        self.window.destroy()
        
    def reset_statistics(self):
        """Reseta todas as estatísticas"""
        if messagebox.askyesno("Confirmar Reset", "Tem certeza que deseja resetar todas as estatísticas?"):
            self.stats.reset()
            self.update_stats_display()
            self.update_plot()
            self.log_to_console("Estatísticas resetadas.")
            
    # --- Video Processing Methods ---
    
    def toggle_webcam(self):
        """Alterna a captura da webcam ligada/desligada"""
        if self.is_processing_active:
            self.stop_processing()
            return
            
        # Try to open the webcam
        try:
            self.cap = cv2.VideoCapture(0)  # Default webcam
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
                return
                
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Update state
            self.current_mode = "Webcam"
            self.current_source_path = "Webcam 0"
            self.btn_toggle_process.config(text="Parar Webcam")
            self.btn_start_stop_monitor.config(state=tk.NORMAL)
            
            # Start processing
            self.start_processing()
            
        except Exception as e:
            self.log_to_console(f"Erro ao iniciar webcam: {e}", is_error=True)
    
    def load_file(self):
        """Carrega um arquivo de vídeo ou imagem"""
        if self.is_processing_active:
            self.stop_processing()
            
        # Ask for file
        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo",
            filetypes=[
                ("Arquivos de Vídeo", "*.mp4 *.avi *.mov *.mkv"),
                ("Imagens", "*.jpg *.jpeg *.png"),
                ("Todos os Arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
            
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        is_image = file_ext in ['.jpg', '.jpeg', '.png']
        
        try:
            if is_image:
                # Load as image
                self.current_mode = "Image"
                self.current_source_path = file_path
                
                # Load image using OpenCV
                self.last_original_frame = cv2.imread(file_path)
                if self.last_original_frame is None:
                    raise ValueError(f"Não foi possível carregar a imagem: {file_path}")
                    
                # Update UI
                self.btn_toggle_process.config(text="Processar Imagem")
                self.btn_start_stop_monitor.config(state=tk.DISABLED)  # No monitoring for images
                
                # Display the image
                self.display_frame(self.last_original_frame)
                self.log_to_console(f"Imagem carregada: {os.path.basename(file_path)}")
                
            else:
                # Load as video
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise ValueError(f"Não foi possível abrir o vídeo: {file_path}")
                    
                # Update state
                self.current_mode = "Video"
                self.current_source_path = file_path
                self.btn_toggle_process.config(text="Parar Vídeo")
                self.btn_start_stop_monitor.config(state=tk.NORMAL)
                
                # Start processing
                self.start_processing()
                self.log_to_console(f"Vídeo carregado: {os.path.basename(file_path)}")
                
        except Exception as e:
            self.log_to_console(f"Erro ao carregar arquivo: {e}", is_error=True)
    
    def start_processing(self):
        """Inicia o processamento de vídeo"""
        if not self.model_manager.current_model:
            messagebox.showwarning("Aviso", "Nenhum modelo carregado. Carregue um modelo primeiro.")
            return
            
        self.is_processing_active = True
        self.update_status_label()
        self.process_next_frame()
    
    def stop_processing(self):
        """Para o processamento de vídeo"""
        self.is_processing_active = False
        self.update_status_label()
        
        # Update button text based on current mode
        if self.current_mode == "Webcam":
            self.btn_toggle_process.config(text="Iniciar Webcam")
        elif self.current_mode == "Video":
            self.btn_toggle_process.config(text="Iniciar Vídeo")
        elif self.current_mode == "Image":
            self.btn_toggle_process.config(text="Processar Imagem")
            
        # Release video capture if needed
        if self.cap is not None and self.current_mode != "Image":
            self.cap.release()
            self.cap = None
            
        self.log_to_console("Processamento parado.")
    
    def process_next_frame(self):
        """Processa o próximo frame do vídeo"""
        if not self.is_processing_active:
            return
            
        # Calcular FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
            self.current_fps = sum(self.fps_history) / len(self.fps_history)
            
        # Get frame from video source
        if self.current_mode == "Image":
            # For images, just use the last loaded frame
            frame = self.last_original_frame.copy()
        else:
            # Para vídeo/webcam, implementar pulo de frames para melhorar desempenho
            skip = self.skip_frames.get()
            
            # Pular frames se necessário
            for _ in range(skip):
                if self.cap.isOpened():
                    self.cap.read()  # Ler e descartar frames
            
            # Ler o próximo frame para processamento
            ret, frame = self.cap.read()
            if not ret:
                # End of video or error
                if self.current_mode == "Video":
                    self.log_to_console("Fim do vídeo alcançado.")
                else:
                    self.log_to_console("Erro ao capturar frame da webcam.", is_error=True)
                self.stop_processing()
                return
        
        # Redimensionar frame para processamento mais rápido, se necessário
        scale = self.process_resolution.get()
        if scale < 1.0 and frame is not None:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Processar o frame redimensionado
            processed_frame_resized = self.process_frame(frame_resized)
            
            # Redimensionar de volta para tamanho original para exibição
            processed_frame = cv2.resize(processed_frame_resized, (w, h))
        else:
            # Processar na resolução original
            processed_frame = self.process_frame(frame)
        
        # Store original frame
        self.last_original_frame = frame.copy()
        
        # Display the processed frame
        self.display_frame(processed_frame)
        
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Update statistics display periodically (a cada 10 frames)
        if self.frame_count % 10 == 0:
            self.tracker.increment_frame()
            self.update_stats_display()
            self.update_status_label()
        
        # Schedule next frame processing - usar after_idle para priorizar processamento
        if self.is_processing_active and self.current_mode != "Image":
            self.window.after_idle(self.process_next_frame)
    
    def process_frame(self, frame):
        """
        Processa um frame para detecção, rastreamento e contagem de objetos
        
        Args:
            frame: O frame BGR a ser processado
            
        Returns:
            O frame processado com visualizações
        """
        # Make a copy for drawing
        display_frame = frame.copy()
        
        # Get current confidence threshold
        conf_threshold = self.confidence_threshold.get()
        
        # Run object detection
        detections = self.model_manager.detect_objects(frame, conf_threshold, self.classes_to_detect)
        
        # Process detections and update tracker
        current_objects_count = 0  # Contador de objetos na imagem atual
        objects_by_class = defaultdict(int)  # Contagem por classe na imagem atual
        
        if detections:
            # Extract centroids and details from detections
            centroids = {}  # temp_id -> (x, y)
            details = {}    # temp_id -> {'class': class_name, 'confidence': conf}
            boxes = {}      # temp_id -> (x1, y1, x2, y2)
            
            for i, (box, class_id, confidence) in enumerate(detections):
                # Get box coordinates
                x1, y1, x2, y2 = box
                
                # Calculate centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Get class name
                class_name = self.model_manager.class_names.get(class_id, f"classe_{class_id}")
                
                # Store in dictionaries with temporary ID
                temp_id = f"temp_{i}"
                centroids[temp_id] = (cx, cy)
                details[temp_id] = {'class': class_name, 'confidence': confidence}
                boxes[temp_id] = (x1, y1, x2, y2)
                
                # Incrementar contadores
                current_objects_count += 1
                objects_by_class[class_name] += 1
            
            # Update tracker with new detections
            self.tracker.update(centroids, details, boxes)
        else:
            # No detections, update tracker with empty lists
            self.tracker.update({}, {}, {})
        
        # Draw counting line if defined
        if self.line_start and self.line_end:
            cv2.line(display_frame, self.line_start, self.line_end, LINE_COLOR, LINE_THICKNESS)
            
            # Adicionar setas para indicar a direção de contagem
            direction = self.crossing_direction.get()
            self._draw_direction_arrows(display_frame, self.line_start, self.line_end, direction)
        
        # Variável para controlar se houve alguma contagem neste frame
        object_counted_this_frame = False
        
        # Process tracked objects
        for obj_id, obj_data in self.tracker.get_objects().items():
            # Skip if object has no current position
            if obj_data['last_frame'] < self.tracker.frame_count:
                continue
                
            # Get object data
            centroid = obj_data['centroid']
            prev_centroid = obj_data['prev_centroid']
            class_name = obj_data['class']
            box = obj_data['box']
            is_counted = obj_data['counted']
            
            # Skip if no box (shouldn't happen for current frame objects)
            if box is None:
                continue
                
            # Determine object color if not already counted
            if not is_counted and self.line_start and self.line_end:
                # Check if object crossed the line
                if prev_centroid:  # Need previous position to check crossing
                    crossed, direction = check_line_crossing(
                        prev_centroid, centroid, self.line_start, self.line_end
                    )
                    
                    # If crossed in the right direction, count it
                    count_it = False
                    
                    # Check if direction matches user selection
                    user_direction = self.crossing_direction.get()
                    if (user_direction == "Esquerda -> Direita" and direction == "LR") or \
                       (user_direction == "Direita -> Esquerda" and direction == "RL") or \
                       (user_direction == "Cima -> Baixo" and direction == "TB") or \
                       (user_direction == "Baixo -> Cima" and direction == "BT"):
                        count_it = True
                    
                    if crossed and count_it and not is_counted:
                        # Identify color
                        color_name = get_object_color(frame, box)
                        
                        # Update statistics
                        timestamp = datetime.datetime.now()
                        self.stats.update(class_name, color_name, timestamp)
                        
                        # Mark as counted
                        self.tracker.mark_as_counted(obj_id)
                        
                        # Log the count
                        self.log_to_console(f"Contado: {class_name} ({color_name})")
                        
                        # Update is_counted for drawing
                        is_counted = True
                        
                        # Marcar que houve contagem neste frame
                        object_counted_this_frame = True
                        
                        # Atualizar estatísticas em tempo real
                        self.window.after(1, self.update_stats_display)
            
            # Draw bounding box and label
            if box:
                # Get color if not already determined
                color_name = "Desconhecida"
                if not is_counted:
                    color_name = get_object_color(frame, box)
                
                # Draw box with label
                draw_bounding_box(
                    display_frame, box, class_name, 
                    obj_data.get('confidence', 0.0), 
                    color_name, is_counted
                )
        
        # Draw count text
        count_text = f"Total Contado: {self.stats.total_counted}"
        cv2.putText(display_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, COUNTING_TEXT_COLOR, 2, cv2.LINE_AA)
        
        # Mostrar número de objetos na imagem atual
        objects_text = f"Objetos na imagem: {current_objects_count}"
        cv2.putText(display_frame, objects_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Mostrar contagem por classe na imagem atual (até 3 classes mais frequentes)
        if objects_by_class:
            sorted_classes = sorted(objects_by_class.items(), key=lambda x: x[1], reverse=True)
            y_pos = 110
            for i, (class_name, count) in enumerate(sorted_classes[:3]):  # Mostrar apenas as 3 principais classes
                class_text = f"{class_name}: {count}"
                cv2.putText(display_frame, class_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (50, 50, 255), 2, cv2.LINE_AA)
                y_pos += 30
        
        # Se houve contagem neste frame, atualizar o gráfico imediatamente
        if object_counted_this_frame:
            self.window.after(1, self.update_plot)
        
        # Store processed frame
        self.last_processed_frame = display_frame
        
        return display_frame
        
    def _draw_direction_arrows(self, frame, line_start, line_end, direction_text):
        """
        Desenha setas indicando a direção de contagem na linha
        
        Args:
            frame: Frame para desenhar
            line_start, line_end: Pontos de início e fim da linha
            direction_text: Texto indicando a direção ("Esquerda -> Direita", etc.)
        """
        # Calcular o ponto médio da linha
        mid_x = (line_start[0] + line_end[0]) // 2
        mid_y = (line_start[1] + line_end[1]) // 2
        
        # Calcular o vetor da linha
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        # Normalizar o vetor
        length = max(1, np.sqrt(dx*dx + dy*dy))
        dx, dy = dx/length, dy/length
        
        # Calcular o vetor perpendicular (normal)
        nx, ny = -dy, dx
        
        # Tamanho da seta
        arrow_size = 30
        
        # Desenhar seta baseada na direção selecionada
        if direction_text == "Esquerda -> Direita":
            # Seta da esquerda para a direita (perpendicular à linha)
            start_point = (int(mid_x - nx * arrow_size), int(mid_y - ny * arrow_size))
            end_point = (int(mid_x + nx * arrow_size), int(mid_y + ny * arrow_size))
            cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2, tipLength=0.3)
            
        elif direction_text == "Direita -> Esquerda":
            # Seta da direita para a esquerda (perpendicular à linha)
            start_point = (int(mid_x + nx * arrow_size), int(mid_y + ny * arrow_size))
            end_point = (int(mid_x - nx * arrow_size), int(mid_y - ny * arrow_size))
            cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2, tipLength=0.3)
            
        elif direction_text == "Cima -> Baixo":
            # Seta de cima para baixo (perpendicular à linha)
            start_point = (int(mid_x - nx * arrow_size), int(mid_y - ny * arrow_size))
            end_point = (int(mid_x + nx * arrow_size), int(mid_y + ny * arrow_size))
            cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2, tipLength=0.3)
            
        elif direction_text == "Baixo -> Cima":
            # Seta de baixo para cima (perpendicular à linha)
            start_point = (int(mid_x + nx * arrow_size), int(mid_y + ny * arrow_size))
            end_point = (int(mid_x - nx * arrow_size), int(mid_y - ny * arrow_size))
            cv2.arrowedLine(frame, start_point, end_point, (0, 200, 255), 2, tipLength=0.3)
    
    def display_frame(self, frame):
        """
        Exibe um frame na interface gráfica
        
        Args:
            frame: O frame BGR a ser exibido
        """
        # Convert to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get display dimensions
        display_width = self.display_label.winfo_width()
        display_height = self.display_label.winfo_height()
        
        # Ensure we have valid dimensions
        if display_width <= 1 or display_height <= 1:
            display_width = self.display_width
            display_height = self.display_height
        
        # Calculate scale to fit frame in display area while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        width_scale = display_width / frame_width
        height_scale = display_height / frame_height
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized_frame)
        
        # Create PhotoImage
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update label
        self.display_label.config(image=photo)
        self.display_label.image = photo  # Keep a reference
        
        # Store display information for coordinate conversion
        self.last_display_scale = scale
        self.last_display_dims = (new_width, new_height)
        
        # Calculate offsets for centering
        x_offset = (display_width - new_width) // 2
        y_offset = (display_height - new_height) // 2
        self.last_display_offsets = (x_offset, y_offset)
    
    # --- Line Drawing Methods ---
    
    def on_line_draw_start(self, event):
        """Inicia o desenho da linha quando o botão do mouse é pressionado"""
        if not self.enable_line_drawing.get() or self.last_original_frame is None:
            return
            
        # Convert display coordinates to original frame coordinates
        orig_x, orig_y = self._display_to_original_coords(event.x, event.y)
        
        # Start drawing line
        self.drawing_line = True
        self.line_start = (orig_x, orig_y)
        self.line_end = (orig_x, orig_y)  # Initially same as start
        
        # Redraw frame with the new line
        if self.last_processed_frame is not None:
            self._redraw_with_line()
    
    def on_line_draw_motion(self, event):
        """Atualiza a linha enquanto o mouse é movido"""
        if not self.drawing_line or not self.enable_line_drawing.get():
            return
            
        # Convert display coordinates to original frame coordinates
        orig_x, orig_y = self._display_to_original_coords(event.x, event.y)
        
        # Update end point
        self.line_end = (orig_x, orig_y)
        
        # Redraw frame with the updated line
        if self.last_processed_frame is not None:
            self._redraw_with_line()
    
    def on_line_draw_end(self, event):
        """Finaliza o desenho da linha quando o botão do mouse é liberado"""
        if not self.drawing_line or not self.enable_line_drawing.get():
            return
            
        # Convert display coordinates to original frame coordinates
        orig_x, orig_y = self._display_to_original_coords(event.x, event.y)
        
        # Update end point
        self.line_end = (orig_x, orig_y)
        
        # Finish drawing
        self.drawing_line = False
        
        # Redraw frame with the final line
        if self.last_processed_frame is not None:
            self._redraw_with_line()
            
        self.log_to_console(f"Linha de contagem definida: {self.line_start} a {self.line_end}")
    
    def _redraw_with_line(self):
        """Redesenha o frame com a linha atual"""
        # Make a copy of the processed frame
        frame_with_line = self.last_processed_frame.copy()
        
        # Draw the line
        cv2.line(frame_with_line, self.line_start, self.line_end, LINE_COLOR, LINE_THICKNESS)
        
        # Display the updated frame
        self.display_frame(frame_with_line)
    
    def _display_to_original_coords(self, display_x, display_y):
        """
        Converte coordenadas da tela para coordenadas do frame original
        
        Args:
            display_x, display_y: Coordenadas na tela
            
        Returns:
            tuple: (x, y) no frame original
        """
        if self.last_original_frame is None:
            return (0, 0)
            
        # Get offsets and scale
        x_offset, y_offset = self.last_display_offsets
        scale = self.last_display_scale
        
        # Adjust for offset
        adjusted_x = display_x - x_offset
        adjusted_y = display_y - y_offset
        
        # Convert to original coordinates
        orig_x = int(adjusted_x / scale)
        orig_y = int(adjusted_y / scale)
        
        # Ensure within bounds of original frame
        frame_height, frame_width = self.last_original_frame.shape[:2]
        orig_x = max(0, min(orig_x, frame_width - 1))
        orig_y = max(0, min(orig_y, frame_height - 1))
        
        return (orig_x, orig_y)
    
    # --- Class Selection Methods ---
    
    def update_class_checkboxes(self):
        """Atualiza a lista de checkboxes de classes baseado no modelo carregado"""
        # Limpar checkboxes existentes
        for widget in self.classes_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Resetar dicionário de classes selecionadas
        self.selected_classes = {}
        
        # Adicionar checkbox para cada classe do modelo
        if self.model_manager.class_names:
            for class_id, class_name in self.model_manager.class_names.items():
                var = tk.BooleanVar(value=True)  # Inicialmente todas selecionadas
                self.selected_classes[class_id] = var
                
                checkbox = ttk.Checkbutton(
                    self.classes_scrollable_frame, 
                    text=f"{class_name} (ID: {class_id})",
                    variable=var
                )
                checkbox.pack(anchor="w", padx=5, pady=2)
    
    def select_all_classes(self):
        """Seleciona todas as classes disponíveis"""
        for var in self.selected_classes.values():
            var.set(True)
    
    def clear_class_selection(self):
        """Limpa a seleção de classes"""
        for var in self.selected_classes.values():
            var.set(False)
    
    def apply_class_selection(self):
        """Aplica a seleção atual de classes para detecção"""
        # Obter IDs das classes selecionadas
        selected_ids = [class_id for class_id, var in self.selected_classes.items() if var.get()]
        
        # Se nenhuma classe selecionada, detectar todas
        if not selected_ids:
            self.classes_to_detect = None
            self.log_to_console("Detectando todas as classes disponíveis.")
        else:
            self.classes_to_detect = selected_ids
            class_names = [self.model_manager.class_names[class_id] for class_id in selected_ids]
            self.log_to_console(f"Detectando apenas as classes: {', '.join(class_names)}")
    
    # --- Model Management Methods ---
    
    def load_model_dialog(self):
        """Abre um diálogo para carregar um modelo YOLO"""
        model_path = filedialog.askopenfilename(
            title="Selecione um Modelo YOLO",
            filetypes=[
                ("PyTorch Model", "*.pt"),
                ("ONNX Model", "*.onnx"),
                ("Todos os Arquivos", "*.*")
            ]
        )
        
        if not model_path:
            return  # User cancelled
            
        # Try to load the model
        try:
            success = self.model_manager.load_model(model_path)
            if success:
                self._update_model_status_label()
                self.update_plot_class_options()
                self.log_to_console(f"Modelo carregado: {os.path.basename(model_path)}")
            else:
                messagebox.showerror("Erro", "Falha ao carregar o modelo.")
        except Exception as e:
            self.log_to_console(f"Erro ao carregar modelo: {e}", is_error=True)
            messagebox.showerror("Erro", f"Falha ao carregar o modelo: {e}")
    
    # --- Monitoring Methods ---
    
    def toggle_monitoring(self):
        """Alterna o monitoramento de estatísticas ligado/desligado"""
        if not self.stats.start_time:
            # Start monitoring
            self.stats.start_monitoring()
            self.btn_start_stop_monitor.config(text="Parar Monitoramento")
            self.log_to_console("Monitoramento iniciado.")
        else:
            # Stop monitoring
            self.stats.stop_monitoring()
            self.btn_start_stop_monitor.config(text="Iniciar Monitoramento")
            self.log_to_console("Monitoramento parado.")
            
        # Update statistics display
        self.update_stats_display()

    def export_statistics(self):
        """Exporta as estatísticas para um arquivo JSON"""
        if self.stats.total_counted == 0:
            messagebox.showinfo("Informação", "Não há estatísticas para exportar.")
            return
            
        # Pedir ao usuário onde salvar o arquivo
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"estatisticas_{timestamp}.json"
        
        filepath = filedialog.asksaveasfilename(
            title="Exportar Estatísticas",
            defaultextension=".json",
            initialfile=default_filename,
            filetypes=[("JSON", "*.json"), ("Todos os Arquivos", "*.*")]
        )
        
        if not filepath:
            return  # Usuário cancelou
            
        # Exportar para o arquivo selecionado
        success, message = self.stats.export_to_json(filepath)
        
        if success:
            messagebox.showinfo("Sucesso", message)
        else:
            messagebox.showerror("Erro", message)
    
    def export_pdf_report(self):
        """Exporta um relatório completo em PDF com estatísticas e gráficos"""
        if self.stats.total_counted == 0:
            messagebox.showinfo("Informação", "Não há estatísticas para exportar.")
            return
        
        # Pedir ao usuário onde salvar o arquivo
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"relatorio_{timestamp}.pdf"
        
        filepath = filedialog.asksaveasfilename(
            title="Exportar Relatório PDF",
            defaultextension=".pdf",
            initialfile=default_filename,
            filetypes=[("PDF", "*.pdf"), ("Todos os Arquivos", "*.*")]
        )
        
        if not filepath:
            return  # Usuário cancelou
        
        try:
            # Capturar imagem do último frame processado
            frame_image = None
            if self.last_processed_frame is not None:
                frame_image = self.last_processed_frame.copy()
            
            # Capturar imagem do gráfico atual
            plot_image = None
            if self.fig is not None:
                # Forçar atualização do gráfico com dados mais recentes
                self.update_plot()
                plot_image = capture_plot_image(self.fig)
            
            # Mostrar diálogo de progresso
            progress_window = tk.Toplevel(self.window)
            progress_window.title("Gerando PDF")
            progress_window.geometry("300x100")
            progress_window.transient(self.window)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="Gerando relatório PDF...").pack(pady=10)
            progress = ttk.Progressbar(progress_window, mode="indeterminate")
            progress.pack(fill=tk.X, padx=20, pady=10)
            progress.start()
            
            # Atualizar a interface
            self.window.update_idletasks()
            
            # Criar o relatório PDF em uma thread separada para não bloquear a interface
            def generate_pdf():
                try:
                    output_path = create_pdf_report(self.stats, frame_image, plot_image, filepath)
                    progress_window.destroy()
                    messagebox.showinfo("Sucesso", f"Relatório PDF gerado com sucesso em:\n{output_path}")
                except Exception as e:
                    progress_window.destroy()
                    messagebox.showerror("Erro", f"Falha ao gerar relatório PDF: {e}")
                    self.log_to_console(f"Erro ao gerar PDF: {e}", is_error=True)
            
            # Iniciar geração em thread separada
            self.window.after(100, generate_pdf)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar relatório PDF: {e}")
            self.log_to_console(f"Erro ao gerar PDF: {e}", is_error=True)
    
# --- Main Function ---

def main():
    """Função principal para iniciar a aplicação"""
    root = tk.Tk()
    app = ObjectCounterApp(root, "IndustriaCount - Sistema de Contagem de Objetos")
    root.mainloop()

if __name__ == "__main__":
    main()
