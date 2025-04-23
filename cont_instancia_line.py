import sys
import os

print("=== Informações do Sistema ===")
print(f"Python Version: {sys.version}")
print(f"DISPLAY: {os.environ.get('DISPLAY', 'Não definido')}")
print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', 'Não definido')}")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', 'Não definido')}")
print(f"GDK_BACKEND: {os.environ.get('GDK_BACKEND', 'Não definido')}")

try:
    import tkinter as tk
    print("Tkinter importado com sucesso")
except Exception as e:
    print(f"Erro ao importar tkinter: {e}")
    sys.exit(1)

try:
    from tkinter import ttk, messagebox, filedialog, colorchooser
    print("Módulos do tkinter importados com sucesso")
except Exception as e:
    print(f"Erro ao importar módulos do tkinter: {e}")
    sys.exit(1)

import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os
import datetime
from collections import deque, defaultdict
import traceback
import json
import math # For line crossing calculation

print("\n=== Versões das Bibliotecas ===")
print(f"Python version: {sys.version}")
print(f"Tkinter version: {tk.TkVersion}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Pillow version: {Image.__version__}")

# Import matplotlib after setting the backend
import matplotlib
matplotlib.use("TkAgg")  # Set backend before other matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print(f"Matplotlib version: {matplotlib.__version__}")

# --- Model Loading ---
# Try loading PyTorch/Torchvision, but make it optional if not installed
# Users will need to install these manually: pip install torch torchvision torchaudio ultralytics
try:
    import torch
    import torchvision
    from ultralytics import YOLO # Use the official ultralytics package
    TORCH_AVAILABLE = True
    print("PyTorch, Torchvision, and Ultralytics loaded successfully.")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch/Torchvision/Ultralytics not found. Object detection features will be disabled. Error: {e}")
    print("Please install them: pip install torch torchvision torchaudio ultralytics")
    # Define dummy classes/functions if torch isn't available to avoid NameErrors later
    class DummyModel:
        def __init__(self): pass
        def __call__(self, *args, **kwargs): return [] # Return empty list
    YOLO = lambda *args, **kwargs: DummyModel() # Dummy YOLO returns DummyModel

# --- Predefined Color Ranges (Name: [Lower HSV], [Upper HSV]) ---
# Kept for color *identification* after detection
PREDEFINED_COLORS = {
    "Vermelho": ([170, 100, 100], [10, 255, 255]),
    "Laranja": ([5, 100, 100], [20, 255, 255]),
    "Amarelo": ([22, 100, 100], [38, 255, 255]),
    "Verde": ([40, 70, 70], [85, 255, 255]),
    "Ciano": ([85, 100, 100], [100, 255, 255]),
    "Azul": ([100, 100, 100], [130, 255, 255]),
    "Violeta/Roxo": ([130, 80, 80], [160, 255, 255]),
    "Rosa/Magenta": ([160, 80, 80], [175, 255, 255]),
    "Branco": ([0, 0, 200], [179, 40, 255]),
    "Cinza": ([0, 0, 50], [179, 50, 200]),
    "Preto": ([0, 0, 0], [179, 255, 60]),
    "Desconhecida": ([0, 0, 0], [0, 0, 0]) # Fallback
}
PREDEFINED_COLOR_NAMES = list(PREDEFINED_COLORS.keys())

# --- Configuration ---
DEFAULT_MODEL_PATH = "yolov8n.pt" # Default to YOLOv8 Nano
DEFAULT_CONFIDENCE = 0.45
MAX_TRACKING_DISTANCE = 75 # Max pixels a centroid can move between frames to be considered the same object
MAX_FRAMES_DISAPPEARED = 10 # How many frames an object can be missing before its track is deleted
LINE_COLOR = (0, 255, 255) # Yellow for the counting line
LINE_THICKNESS = 2
TRACKING_DOT_COLOR = (255, 0, 0) # Blue dots for tracked centroids
COUNTING_TEXT_COLOR = (0, 255, 0) # Green for count text

class ModelManager:
    """Handles loading and running object detection models with efficient resource management."""
    def __init__(self, app_log_func):
        self.log = app_log_func
        self.current_model = None
        self.model_path = None
        self.model_type = None  # e.g., "YOLOv8"
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.log(f"Usando device: {self.device}")
        self.class_names = []  # To store class names from the loaded model
        self._last_inference_time = 0
        self._inference_times = deque(maxlen=10)  # Track last 10 inference times for performance monitoring

    def load_model(self, model_path):
        """Loads a detection model from the specified path with proper error handling."""
        if not TORCH_AVAILABLE:
            self.log("Erro: PyTorch/Ultralytics não instalado. Não é possível carregar o modelo.", is_error=True)
            return False
            
        if not os.path.exists(model_path):
            self.log(f"Erro: Arquivo do modelo não encontrado em '{model_path}'.", is_error=True)
            return False

        try:
            # Release resources from previous model if any
            if self.current_model is not None:
                self.log("Liberando recursos do modelo anterior...")
                # Force garbage collection to release CUDA memory if applicable
                if hasattr(self.current_model, 'cpu'):
                    try:
                        self.current_model.cpu()
                    except Exception:
                        pass
                self.current_model = None
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.log(f"Carregando modelo '{os.path.basename(model_path)}' em {self.device}...")
            
            # Currently supports YOLO models via ultralytics package
            if model_path.endswith(".pt"):
                self.current_model = YOLO(model_path)
                self.current_model.to(self.device)
                
                # Perform a dummy inference to initialize/warm up the model
                start_time = time.time()
                _ = self.current_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                warmup_time = time.time() - start_time
                
                self.model_path = model_path
                self.model_type = "YOLOv8"  # Assume YOLOv8/v5 for .pt files
                
                # Get class names from the model
                self.class_names = self.current_model.names if hasattr(self.current_model, 'names') else [f'classe_{i}' for i in range(100)]
                
                self.log(f"Modelo {self.model_type} carregado com sucesso. Classes: {len(self.class_names)}")
                self.log(f"Tempo de warmup: {warmup_time:.3f}s")
                return True
            else:
                self.log(f"Erro: Tipo de modelo não suportado (somente arquivos .pt YOLO são suportados no momento).", is_error=True)
                self.current_model = None
                return False
                
        except Exception as e:
            self.log(f"Erro ao carregar modelo: {e}", is_error=True)
            traceback.print_exc()
            self.current_model = None
            return False

    def detect_objects(self, frame, confidence_threshold=0.5, classes_to_detect=None):
        """Performs object detection on a frame with performance tracking."""
        if self.current_model is None or not TORCH_AVAILABLE:
            return [], []  # No model or no torch

        detections = []  # List to store [x1, y1, x2, y2, conf, class_id]
        names = []  # List to store class names corresponding to detections

        try:
            start_time = time.time()
            
            if self.model_type == "YOLOv8":
                # Create a copy of the frame to avoid modifying the original
                # Note: Ultralytics YOLO automatically handles BGR->RGB and normalization
                results = self.current_model.predict(
                    source=frame.copy(), 
                    conf=confidence_threshold, 
                    classes=classes_to_detect, 
                    verbose=False
                )

                if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    
                    # Check if we have any detections
                    if len(boxes) > 0:
                        # Get boxes, confidences, and class IDs
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
                        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []

                        # Process each detection
                        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
                            # Convert to integers for drawing
                            x1, y1, x2, y2 = map(int, box[:4])
                            
                            # Add to results
                            detections.append([x1, y1, x2, y2, float(conf), int(cls_id)])
                            
                            # Get class name
                            try:
                                class_name = self.class_names[int(cls_id)]
                            except (IndexError, KeyError):
                                class_name = f"classe_{int(cls_id)}"
                            names.append(class_name)

            # Track inference time
            inference_time = time.time() - start_time
            self._inference_times.append(inference_time)
            self._last_inference_time = inference_time
            
            # Log performance occasionally (every 50th frame or if very slow)
            if len(detections) > 0 and (inference_time > 0.1 or random.random() < 0.02):
                avg_time = sum(self._inference_times) / len(self._inference_times)
                self.log(f"Detecção: {len(detections)} objetos em {inference_time:.3f}s (média: {avg_time:.3f}s)")

        except Exception as e:
            self.log(f"Erro durante a detecção: {e}", is_error=True)
            # Only print traceback for unexpected errors
            if not isinstance(e, (torch.cuda.OutOfMemoryError, RuntimeError)):
                traceback.print_exc()
            return [], []

        return detections, names
    
    def get_inference_stats(self):
        """Returns statistics about inference performance."""
        if not self._inference_times:
            return {"last": 0, "avg": 0, "min": 0, "max": 0}
            
        return {
            "last": self._last_inference_time,
            "avg": sum(self._inference_times) / len(self._inference_times),
            "min": min(self._inference_times),
            "max": max(self._inference_times)
        }
        
    def cleanup(self):
        """Releases resources used by the model."""
        if self.current_model is not None:
            try:
                if hasattr(self.current_model, 'cpu'):
                    self.current_model.cpu()
                self.current_model = None
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.log("Recursos do modelo liberados com sucesso.")
            except Exception as e:
                self.log(f"Erro ao liberar recursos do modelo: {e}", is_error=True)

class ObjectStatistics:
    """Stores and manages statistics about detected and counted objects."""
    def __init__(self):
        self.counts_per_class = defaultdict(int)
        self.counts_per_color_per_class = defaultdict(lambda: defaultdict(int))
        self.time_series_counts = defaultdict(list) # class -> [(timestamp, cumulative_count)]
        self.total_counted = 0
        self.start_time = None
        self.last_update_time = None

    def start_monitoring(self):
        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time
        print("Monitoring started at:", self.start_time) # Debug

    def stop_monitoring(self):
        self.last_update_time = datetime.datetime.now()
        print("Monitoring stopped at:", self.last_update_time) # Debug

    def update(self, class_name, color_name, timestamp):
        """Records a counted object."""
        if self.start_time is None: # Ensure monitoring has started
            self.start_monitoring()

        self.total_counted += 1
        self.counts_per_class[class_name] += 1
        self.counts_per_color_per_class[class_name][color_name] += 1
        self.last_update_time = timestamp

        # Add to time series for plotting cumulative count of this class
        cumulative_count = self.counts_per_class[class_name]
        self.time_series_counts[class_name].append((timestamp, cumulative_count))

    def get_summary(self):
        """Returns a formatted string summary of the statistics."""
        if self.start_time is None:
            return "Monitoramento não iniciado."

        duration_seconds = (self.last_update_time - self.start_time).total_seconds()
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))

        summary = f"--- Resumo Estatístico ({duration_str}) ---\n"
        summary += f"Total Contado: {self.total_counted}\n"

        if self.counts_per_class:
            summary += "\nContagem por Classe:\n"
            for cls, count in sorted(self.counts_per_class.items()):
                summary += f"  - {cls}: {count}\n"

        if self.counts_per_color_per_class:
            summary += "\nContagem por Cor (dentro da Classe):\n"
            for cls, color_counts in sorted(self.counts_per_color_per_class.items()):
                if color_counts: # Only show classes with color info
                    summary += f"  - {cls}:\n"
                    for color, count in sorted(color_counts.items()):
                        summary += f"    - {color}: {count}\n"
        summary += "--------------------------------------"
        return summary

    def reset(self):
        """Resets all statistics."""
        self.counts_per_class.clear()
        self.counts_per_color_per_class.clear()
        self.time_series_counts.clear()
        self.total_counted = 0
        self.start_time = None
        self.last_update_time = None
        print("Statistics reset.") # Debug

class ObjectCounterApp: # Renamed from PileDetectorApp
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x900") # Increased size for new elements
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        # --- State Variables ---
        self.cap = None
        self.is_processing_active = False
        self.current_source_path = None
        self.current_mode = "None" # None, Webcam, Video, Image
        self.display_width = 640
        self.display_height = 480
        self.last_original_frame = None
        self.last_processed_frame = None # Store the frame *with* drawings
        self.last_display_scale = 1.0
        self.last_display_offsets = (0, 0)
        self.last_display_dims = (0, 0)

        # --- Model Management ---
        self.model_manager = ModelManager(self.log_to_console)
        self.confidence_threshold = tk.DoubleVar(value=DEFAULT_CONFIDENCE)

        # --- Line Crossing & Tracking ---
        self.line_start = None # (x, y) in *original* frame coordinates
        self.line_end = None
        self.drawing_line = False
        self.enable_line_drawing = tk.BooleanVar(value=False)
        self.next_object_id = 0
        self.tracked_objects = {} # id -> {'centroid':(x,y), 'last_frame': frame_count, 'counted': False, 'prev_centroid': (x,y), 'class': name}
        self.frame_count = 0
        self.crossing_direction = tk.StringVar(value="Esquerda -> Direita") # Options: "Esquerda -> Direita", "Direita -> Esquerda", "Cima -> Baixo", "Baixo -> Cima"

        # --- Statistics ---
        self.stats = ObjectStatistics()

        # --- Color Identification Settings (using predefined) ---
        # No separate HSV settings needed for detection anymore, use predefined for classification

        # --- Plotting Settings ---
        self.plot_data = defaultdict(lambda: deque(maxlen=100)) # class -> deque[(time, count)]
        self.fig = None; self.ax = None; self.canvas = None
        self.selected_plot_class = tk.StringVar(value="Total") # Class to display on plot

        # --- Tkinter Widgets ---
        main_frame = ttk.Frame(window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=3) # Video display area
        main_frame.columnconfigure(1, weight=1) # Controls/Stats area
        main_frame.rowconfigure(0, weight=1) # Make display row expand

        # === Left Column: Display ===
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)

        self.display_label = ttk.Label(display_frame, background='grey', cursor="crosshair")
        self.display_label.grid(row=0, column=0, pady=0, sticky="nsew")
        self.display_label.bind("<ButtonPress-1>", self.on_line_draw_start)
        self.display_label.bind("<B1-Motion>", self.on_line_draw_motion)
        self.display_label.bind("<ButtonRelease-1>", self.on_line_draw_end)

        # === Right Column: Controls & Info ===
        controls_info_frame = ttk.Frame(main_frame)
        controls_info_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        controls_info_frame.columnconfigure(0, weight=1)
        # Configure rows for proportional space allocation
        controls_info_frame.rowconfigure(0, weight=0) # Source Controls
        controls_info_frame.rowconfigure(1, weight=0) # Model Config
        controls_info_frame.rowconfigure(2, weight=0) # Line Config
        controls_info_frame.rowconfigure(3, weight=0) # Status/Counts
        controls_info_frame.rowconfigure(4, weight=2) # Statistics Display
        controls_info_frame.rowconfigure(5, weight=2) # Plot
        controls_info_frame.rowconfigure(6, weight=1) # Console

        # --- 1. Source Controls ---
        source_frame = ttk.LabelFrame(controls_info_frame, text="Fonte e Controle", padding="5")
        source_frame.grid(row=0, column=0, sticky="new", pady=(0, 5))
        source_frame.columnconfigure(0, weight=1); source_frame.columnconfigure(1, weight=1)
        self.btn_toggle_process = ttk.Button(source_frame, text="Iniciar Webcam", command=self.toggle_webcam)
        self.btn_toggle_process.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
        self.btn_load_file = ttk.Button(source_frame, text="Carregar Arquivo", command=self.load_file)
        self.btn_load_file.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=2)
        self.btn_start_stop_monitor = ttk.Button(source_frame, text="Iniciar Monitoramento", command=self.toggle_monitoring, state=tk.DISABLED)
        self.btn_start_stop_monitor.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)


        # --- 2. Model Configuration ---
        model_frame = ttk.LabelFrame(controls_info_frame, text="Configuração do Modelo", padding="5")
        model_frame.grid(row=1, column=0, sticky="new", pady=5)
        model_frame.columnconfigure(1, weight=1)
        ttk.Button(model_frame, text="Carregar Modelo (.pt)", command=self.load_model_dialog).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.model_status_label = ttk.Label(model_frame, text="Modelo: Nenhum carregado", anchor="w", relief=tk.SUNKEN, padding=(3, 1))
        self.model_status_label.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(model_frame, text="Confiança Mín.:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.confidence_scale = ttk.Scale(model_frame, from_=0.1, to=0.95, orient=tk.HORIZONTAL, variable=self.confidence_threshold, command=self._update_confidence_label)
        self.confidence_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.confidence_label = ttk.Label(model_frame, text=f"{self.confidence_threshold.get():.2f}", width=5)
        self.confidence_label.grid(row=1, column=2, sticky="w", padx=5, pady=2)


        # --- 3. Line Configuration ---
        line_frame = ttk.LabelFrame(controls_info_frame, text="Linha de Contagem", padding="5")
        line_frame.grid(row=2, column=0, sticky="new", pady=5)
        line_frame.columnconfigure(1, weight=1)
        self.chk_draw_line = ttk.Checkbutton(line_frame, text="Habilitar Desenho da Linha", variable=self.enable_line_drawing, command=self._on_toggle_draw_mode)
        self.chk_draw_line.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Label(line_frame, text="Direção Contagem:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.combo_direction = ttk.Combobox(line_frame, textvariable=self.crossing_direction,
                                            values=["Esquerda -> Direita", "Direita -> Esquerda", "Cima -> Baixo", "Baixo -> Cima"],
                                            state="readonly", width=18)
        self.combo_direction.grid(row=1, column=1, sticky="ew", padx=5, pady=2)


        # --- 4. Status & Counts ---
        status_frame = ttk.Frame(controls_info_frame) # No label frame needed
        status_frame.grid(row=3, column=0, sticky="new", pady=5)
        status_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status_frame, text="Status: Parado", anchor=tk.W)
        self.status_label.grid(row=0, column=0, sticky="ew", padx=5)
        self.count_label = ttk.Label(status_frame, text="Total Contado: 0", anchor=tk.W, font=('Arial', 10, 'bold'))
        self.count_label.grid(row=1, column=0, sticky="ew", padx=5)


        # --- 5. Statistics Display ---
        stats_frame = ttk.LabelFrame(controls_info_frame, text="Estatísticas", padding="5")
        stats_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        stats_frame.columnconfigure(0, weight=1); stats_frame.rowconfigure(0, weight=1)
        self.stats_output = tk.Text(stats_frame, height=6, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1, font=("Courier New", 8))
        self.stats_output.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_output.yview)
        stats_scrollbar.grid(row=0, column=1, sticky="ns", pady=(0,5))
        self.stats_output['yscrollcommand'] = stats_scrollbar.set
        ttk.Button(stats_frame, text="Resetar Estatísticas", command=self.reset_statistics).grid(row=1, column=0, columnspan=2, pady=(0,2))


        # --- 6. Plotting Frame ---
        plot_outer_frame = ttk.LabelFrame(controls_info_frame, text="Contagem Acumulada por Classe", padding="5")
        plot_outer_frame.grid(row=5, column=0, sticky="nsew", pady=5)
        plot_outer_frame.columnconfigure(0, weight=1); plot_outer_frame.rowconfigure(1, weight=1)
        # Dropdown to select class for plotting
        plot_controls_frame = ttk.Frame(plot_outer_frame)
        plot_controls_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(plot_controls_frame, text="Exibir Classe:").pack(side=tk.LEFT, padx=(0, 5))
        self.combo_plot_class = ttk.Combobox(plot_controls_frame, textvariable=self.selected_plot_class, state="readonly", width=15)
        self.combo_plot_class.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.combo_plot_class.bind("<<ComboboxSelected>>", self.update_plot) # Update plot on selection change
        # Plot Canvas Area
        self.plot_canvas_frame = ttk.Frame(plot_outer_frame) # Frame to hold the canvas
        self.plot_canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_canvas_frame.columnconfigure(0, weight=1); self.plot_canvas_frame.rowconfigure(0, weight=1)


        # --- 7. Console ---
        console_frame = ttk.LabelFrame(controls_info_frame, text="Mensagens", padding="5")
        console_frame.grid(row=6, column=0, sticky="nsew", pady=(5, 0))
        console_frame.columnconfigure(0, weight=1); console_frame.rowconfigure(0, weight=1)
        self.console_output = tk.Text(console_frame, height=4, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1);
        self.console_output.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        console_scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.console_output.yview);
        console_scrollbar.grid(row=0, column=1, sticky="ns", pady=(0,5))
        self.console_output['yscrollcommand'] = console_scrollbar.set

        # --- Final Setup ---
        self.delay = 30 # ms between frames
        self.setup_plot()
        self.set_placeholder_image()
        self.update_status_label()
        self.update_stats_display() # Show initial stats message
        self.update_plot_class_options() # Populate plot dropdown
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.log_to_console("Aplicação iniciada. Carregue um modelo e uma fonte de vídeo/webcam.")

        # Try to load default model if it exists
        if TORCH_AVAILABLE and os.path.exists(DEFAULT_MODEL_PATH):
            self.model_manager.load_model(DEFAULT_MODEL_PATH)
            self._update_model_status_label()
            self.update_plot_class_options()
        else:
             self.log_to_console(f"Modelo padrão '{DEFAULT_MODEL_PATH}' não encontrado ou PyTorch/Ultralytics indisponível.")


    # --- Logging and UI Updates ---

    def log_to_console(self, message, is_error=False):
        """Logs messages to the console widget and optionally prints."""
        prefix = "[ERRO] " if is_error else ""
        full_message = f"{prefix}{message}"
        print(f"[{time.strftime('%H:%M:%S')}] {full_message}") # Always print for debugging

        if hasattr(self, 'console_output') and self.console_output and self.console_output.winfo_exists():
            try:
                self.console_output.config(state=tk.NORMAL)
                timestamp = time.strftime("%H:%M:%S")
                self.console_output.insert(tk.END, f"[{timestamp}] {full_message}\n")
                self.console_output.see(tk.END)
                self.console_output.config(state=tk.DISABLED)
            except tk.TclError: pass # Ignore if widget destroyed
            except Exception as e: print(f"Error logging to console widget: {e}")

    def set_placeholder_image(self):
        """Displays a gray placeholder on the video label."""
        try:
            placeholder = Image.new('RGB', (self.display_width, self.display_height), color=(128, 128, 128))
            # Add text to placeholder
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(placeholder)
            try:
                # Try loading a system font; adjust path/name if needed
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            text = "Sem Vídeo / Câmera Inativa"
            # Deprecated way, use textbbox for newer Pillow versions if needed
            # text_width, text_height = draw.textsize(text, font=font) # Deprecated
            # For Pillow >= 10.0.0
            try:
                bbox = draw.textbbox((0,0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError: # Fallback for older Pillow
                 text_width, text_height = draw.textsize(text, font=font)

            position = ((self.display_width - text_width) // 2, (self.display_height - text_height) // 2)
            draw.text(position, text, fill=(220, 220, 220), font=font)

            imgtk = ImageTk.PhotoImage(image=placeholder)
            if hasattr(self,'display_label') and self.display_label.winfo_exists():
                 self.display_label.imgtk = imgtk
                 self.display_label.configure(image=imgtk, anchor=tk.CENTER)

            self.last_original_frame = None
            self.last_processed_frame = None
            self.last_display_dims = (0,0)
            self.last_display_offsets = (0,0)
            self.last_display_scale = 1.0
            # Reset line drawing on placeholder
            self.line_start, self.line_end = None, None

            # Keep stats, but reset current count display if needed
            # self.update_count_label() # Count label updated via stats now
            # Maybe clear plot? Depends on desired behavior when source stops.
            # self.plot_data.clear()
            # self.update_plot()

        except Exception as e:
            self.log_to_console(f"Erro ao definir placeholder: {e}", is_error=True)

    def update_status_label(self):
        if hasattr(self, 'status_label') and self.status_label and self.status_label.winfo_exists():
            status_text = f"Status: {self.current_mode}"
            if self.is_processing_active:
                status_text += " (Processando)"
            elif self.current_mode == "None":
                status_text = "Status: Parado"
            elif self.current_mode == "Image":
                fname = os.path.basename(str(self.current_source_path)) if self.current_source_path else 'N/A'
                status_text += f" (Imagem Carregada: {fname})"
            elif self.current_mode == "Video" or self.current_mode == "Webcam":
                 source_name = "Webcam" if self.current_mode == "Webcam" else (os.path.basename(str(self.current_source_path)) if self.current_source_path and isinstance(self.current_source_path, str) and os.path.exists(self.current_source_path) else 'Fonte Desconhecida')
                 status_text += f" (Pronto: {source_name})"
            else:
                 status_text += " (Parado)"

            # Add monitoring status
            if self.stats.start_time and not self.is_processing_active: # Monitoring on but source stopped
                 status_text += " [Monitoramento Ativo]"
            elif self.stats.start_time and self.is_processing_active: # Both running
                 status_text += " [Monitoramento Ativo]"
            elif not self.stats.start_time:
                 status_text += " [Monitoramento Parado]"


            try: self.status_label.config(text=status_text)
            except tk.TclError: pass

    def update_count_label(self):
        """Updates the label showing the total count from statistics."""
        if hasattr(self,'count_label') and self.count_label and self.count_label.winfo_exists():
            try:
                self.count_label.config(text=f"Total Contado: {self.stats.total_counted}")
            except tk.TclError: pass

    def update_stats_display(self):
        """Updates the statistics text widget."""
        if hasattr(self,'stats_output') and self.stats_output and self.stats_output.winfo_exists():
            try:
                summary = self.stats.get_summary()
                self.stats_output.config(state=tk.NORMAL)
                self.stats_output.delete('1.0', tk.END)
                self.stats_output.insert(tk.END, summary)
                self.stats_output.config(state=tk.DISABLED)
            except tk.TclError: pass
            except Exception as e:
                self.log_to_console(f"Erro ao atualizar display de estatísticas: {e}", is_error=True)

    def _update_model_status_label(self):
        """Updates the label showing the loaded model's status."""
        if hasattr(self, 'model_status_label') and self.model_status_label.winfo_exists():
            if self.model_manager.current_model:
                model_name = os.path.basename(self.model_manager.model_path)
                num_classes = len(self.model_manager.class_names)
                text = f"Modelo: {model_name} ({num_classes} classes)"
            else:
                text = "Modelo: Nenhum carregado"
            try: self.model_status_label.config(text=text)
            except tk.TclError: pass

    def _update_confidence_label(self, event=None):
        """Updates the label next to the confidence slider."""
        if hasattr(self, 'confidence_label') and self.confidence_label.winfo_exists():
            try: self.confidence_label.config(text=f"{self.confidence_threshold.get():.2f}")
            except tk.TclError: pass

    def display_cv2_image(self, cv2_image):
        """Resizes, centers, and displays a CV2 image (BGR) in the Tkinter label."""
        # Store the processed frame (which includes drawings) for potential display
        self.last_processed_frame = cv2_image

        if cv2_image is None:
            self.log_to_console("Erro: Tentativa de exibir imagem nula.", is_error=True)
            self.set_placeholder_image()
            self.last_original_frame = None # Also clear original if processed is bad
            return

        try:
            h_orig, w_orig = cv2_image.shape[:2]
            if h_orig == 0 or w_orig == 0:
                self.log_to_console(f"Aviso: Dimensões inválidas da imagem ({w_orig}x{h_orig}). Exibindo placeholder.", is_error=True)
                self.set_placeholder_image()
                self.last_original_frame = None
                return

            # Calculate scaling factor to fit within display_width/height
            # Use the actual label size if available for better fitting
            label_w = self.display_label.winfo_width()
            label_h = self.display_label.winfo_height()
            if label_w <= 1 or label_h <= 1: # Fallback if label not rendered yet
                target_w, target_h = self.display_width, self.display_height
            else:
                 target_w, target_h = label_w, label_h

            scale = min(target_w / w_orig, target_h / h_orig, 1.0)

            new_w = max(1, int(w_orig * scale))
            new_h = max(1, int(h_orig * scale))

            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_img = cv2.resize(cv2_image, (new_w, new_h), interpolation=interpolation)

            # Create background (use display label's actual background if possible)
            try: bg_color_str = self.display_label.cget('background')
            except: bg_color_str = 'grey' # Fallback
            bg_color_rgb = self.display_label.winfo_rgb(bg_color_str) # returns tuple of 16-bit values
            bg_color_pil = tuple(c // 256 for c in bg_color_rgb) # Convert to 8-bit RGB

            background = Image.new('RGB', (target_w, target_h), color=bg_color_pil)

            # Convert resized BGR to RGB PIL Image
            rgb_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            pil_resized = Image.fromarray(rgb_resized)

            # Calculate offsets to center the image on the background
            x_offset = max(0, (target_w - new_w) // 2)
            y_offset = max(0, (target_h - new_h) // 2)

            # Paste the resized image onto the background
            background.paste(pil_resized, (x_offset, y_offset))

            # Store display parameters for click/line calculations (relative to the *background* size)
            self.last_display_scale = scale
            self.last_display_offsets = (x_offset, y_offset) # Offset within the background/label
            self.last_display_dims = (new_w, new_h) # Dimensions of the *scaled image* on the background

            # Convert final image to Tkinter format
            imgtk = ImageTk.PhotoImage(image=background)

            if hasattr(self,'display_label') and self.display_label.winfo_exists():
                self.display_label.imgtk = imgtk
                self.display_label.configure(image=imgtk, anchor=tk.CENTER)

        except cv2.error as cv_err:
             self.log_to_console(f"Erro OpenCV ao exibir imagem: {cv_err}", is_error=True)
             self.set_placeholder_image()
             self.last_original_frame = None
        except Exception as e:
            self.log_to_console(f"Erro inesperado ao exibir imagem: {e}", is_error=True)
            traceback.print_exc()
            self.set_placeholder_image()
            self.last_original_frame = None

    # --- Source & Processing Control ---

    def stop_processing(self, source_ended_naturally=False):
        """Stops video feed processing and releases resources."""
        was_active = self.is_processing_active
        if was_active:
            self.log_to_console(f"Parando processamento ({self.current_mode})...")
            self.is_processing_active = False

            if self.cap:
                try:
                    self.cap.release()
                    self.log_to_console("Fonte de captura liberada.")
                except Exception as e:
                    self.log_to_console(f"Erro ao liberar captura: {e}", is_error=True)
                self.cap = None

            # Don't clear original frame if we stopped a video, user might want to draw line
            # self.last_original_frame = None

            # Clear tracked objects when source stops
            self.tracked_objects.clear()
            self.next_object_id = 0
            self.frame_count = 0

        # Update button text based on last mode
        next_btn_text = "Iniciar Webcam"
        next_btn_cmd = self.toggle_webcam
        last_path = str(self.current_source_path) if self.current_source_path is not None else None

        if self.current_mode == "Video" and last_path and os.path.exists(last_path):
            next_btn_text = "Reiniciar Vídeo"
            next_btn_cmd = lambda p=self.current_source_path: self.start_video_processing(p)
        # If webcam stopped, button reverts correctly via toggle_webcam logic

        # Reset toggle button state safely
        if hasattr(self, 'btn_toggle_process') and self.btn_toggle_process.winfo_exists():
            try: self.btn_toggle_process.config(text=next_btn_text, command=next_btn_cmd)
            except tk.TclError: pass

        # Don't reset stats when source stops, only via explicit button or app close
        # self.stats.reset()
        # self.update_stats_display()
        # self.update_count_label()

        # Update status label to reflect stopped state but potentially active monitoring
        self.update_status_label()
        # Enable monitoring button if a source was active
        if was_active and hasattr(self,'btn_start_stop_monitor') and self.btn_start_stop_monitor.winfo_exists():
            try: self.btn_start_stop_monitor.config(state=tk.NORMAL)
            except tk.TclError: pass


    def toggle_webcam(self):
        """Starts or stops the webcam feed."""
        if self.is_processing_active and self.current_mode == "Webcam":
            self.stop_processing()
            # Keep mode as None or let start_video handle it? Stop_processing does this.
            self.set_placeholder_image() # Show placeholder after stopping webcam
        else:
            # Stop any other processing first
            if self.is_processing_active:
                self.stop_processing()

            self.set_placeholder_image() # Clear display before starting
            self.current_mode = "Webcam"
            self.current_source_path = 0 # Default webcam index
            # Button text/command updated inside start_video_processing on success/fail
            self.log_to_console("Tentando iniciar Webcam...")
            self.update_status_label()
            self.start_video_processing(0)

    def load_file(self):
        """Loads a video or image file."""
        if self.is_processing_active:
             self.stop_processing()

        filepath = filedialog.askopenfilename(
            title="Selecionar Vídeo ou Imagem",
            filetypes=[
                ("Arquivos de Mídia", "*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos os Arquivos", "*.*")
            ]
        )

        btn = getattr(self, 'btn_toggle_process', None)

        if not filepath:
            self.log_to_console("Nenhum arquivo selecionado.")
            self.current_mode = "None"
            self.set_placeholder_image()
            if btn and btn.winfo_exists(): # Reset button if dialog cancelled
                 try: btn.config(text="Iniciar Webcam", command=self.toggle_webcam)
                 except tk.TclError: pass
            self.update_status_label()
            return

        self.current_source_path = filepath
        _, file_extension = os.path.splitext(filepath)
        file_extension = file_extension.lower()

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Reset stats when loading a new file
        self.reset_statistics()

        if file_extension in image_extensions:
            self.log_to_console(f"Arquivo de imagem selecionado: {os.path.basename(filepath)}")
            self.current_mode = "Image"
            # Reset button to default state for images (as they don't 'run')
            if btn and btn.winfo_exists():
                 try: btn.config(text="Iniciar Webcam", command=self.toggle_webcam)
                 except tk.TclError: pass
            self.process_static_image(filepath)
            self.update_status_label() # Update status after processing

        elif file_extension in video_extensions:
            self.log_to_console(f"Arquivo de vídeo selecionado: {os.path.basename(filepath)}")
            self.current_mode = "Video"
            # Set button to "Iniciar Vídeo" - command set in start_video_processing
            if btn and btn.winfo_exists():
                try: btn.config(text="Iniciar Vídeo", command=lambda p=filepath: self.start_video_processing(p))
                except tk.TclError: pass
            # Try to load the first frame as a preview
            temp_cap = cv2.VideoCapture(filepath)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    self.last_original_frame = frame # Store for potential line drawing before start
                    self.display_cv2_image(frame)
                else:
                    self.set_placeholder_image()
                temp_cap.release()
            else:
                 self.set_placeholder_image()
            self.log_to_console(f"Vídeo '{os.path.basename(filepath)}' pronto. Clique em 'Iniciar Vídeo'.")
            self.update_status_label() # Show status as Video (Pronto)

        else:
            messagebox.showerror("Erro de Arquivo", f"Formato de arquivo não suportado: {file_extension}")
            self.current_mode = "None"
            self.current_source_path = None
            self.set_placeholder_image()
            if btn and btn.winfo_exists(): # Reset button on error
                 try: btn.config(text="Iniciar Webcam", command=self.toggle_webcam)
                 except tk.TclError: pass
            self.update_status_label()

    def process_static_image(self, filepath):
        """Loads, processes (detects objects), and displays a static image."""
        if not self.model_manager.current_model:
            messagebox.showwarning("Modelo Ausente", "Carregue um modelo de detecção antes de processar uma imagem.")
            self.set_placeholder_image() # Show placeholder if no model
            return
        try:
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Erro de Leitura", f"Não foi possível ler o arquivo de imagem: {filepath}")
                self.current_mode = "None"; self.current_source_path = None; self.set_placeholder_image(); self.update_status_label()
                return

            self.log_to_console(f"Processando imagem estática: {os.path.basename(filepath)}")
            self.last_original_frame = img.copy() # Store original before drawing
            self.frame_count = 0 # Reset frame count for static image

            # Perform detection (no tracking needed for static)
            detections, names = self.model_manager.detect_objects(self.last_original_frame, self.confidence_threshold.get())

            # Draw detections and get colors
            processed_img = self.last_original_frame.copy()
            object_details = [] # Store details for potential stats (though less useful for static)
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                label = names[i]
                # Draw bounding box
                color = PREDEFINED_COLORS.get(label[:3].upper(), (0, 200, 0)) # Simple color based on first 3 letters? Or fixed. Use green.
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                # Get color requires HSV conversion and logic... simplified for static image example
                # dominant_color_name = self.get_object_color(self.last_original_frame, (x1, y1, x2, y2))
                dominant_color_name = "N/A" # Placeholder for static
                # Draw label
                text = f"{label} ({conf:.2f}) {dominant_color_name}"
                cv2.putText(processed_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
                object_details.append({'class': label, 'color': dominant_color_name})

            # Display the processed image
            self.display_cv2_image(processed_img)

            # Update stats (simple count for static image)
            self.stats.reset() # Reset for the static image
            for obj in object_details:
                 self.stats.update(obj['class'], obj['color'], datetime.datetime.now())
            self.update_count_label()
            self.update_stats_display()

            # No plot update for static image as it's not time-series

            self.log_to_console(f"{len(detections)} objeto(s) detectado(s) na imagem.")
            # Enable monitoring button after processing static image
            if hasattr(self,'btn_start_stop_monitor') and self.btn_start_stop_monitor.winfo_exists():
                try: self.btn_start_stop_monitor.config(state=tk.NORMAL)
                except tk.TclError: pass


        except FileNotFoundError:
             messagebox.showerror("Erro de Arquivo", f"Arquivo não encontrado: {filepath}")
             self.current_mode = "None"; self.current_source_path = None; self.set_placeholder_image(); self.update_status_label()
        except cv2.error as cv_err:
            messagebox.showerror("Erro OpenCV", f"Erro ao processar imagem: {cv_err}")
            self.current_mode = "None"; self.current_source_path = None; self.set_placeholder_image(); self.update_status_label()
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Erro ao processar imagem estática: {e}")
            traceback.print_exc()
            self.current_mode = "None"; self.current_source_path = None; self.set_placeholder_image(); self.update_status_label()

    def start_video_processing(self, source):
        """Starts processing a video file or webcam feed."""
        if not self.model_manager.current_model:
            messagebox.showwarning("Modelo Ausente", "Carregue um modelo de detecção antes de iniciar o vídeo/webcam.")
            return
        if self.is_processing_active:
            self.log_to_console("Processamento já está ativo.", is_error=True)
            return

        self.log_to_console(f"Iniciando {self.current_mode} da fonte: {source}")
        device_index = source
        btn = getattr(self, 'btn_toggle_process', None)
        monitor_btn = getattr(self, 'btn_start_stop_monitor', None)

        try:
            # Specific handling for webcam index 0 or 1
            if isinstance(source, int):
                 # Try DSHOW first on Windows, then default
                 cap_options = [(source, cv2.CAP_DSHOW), (1-source, cv2.CAP_DSHOW), (source,), (1-source,)] if os.name == 'nt' else [(source,), (1-source,)]
                 for i, args in enumerate(cap_options):
                     self.cap = cv2.VideoCapture(*args)
                     if self.cap and self.cap.isOpened():
                         device_index = args[0]
                         self.log_to_console(f"Webcam aberta com sucesso (Índice {device_index}, Tentativa {i+1}).")
                         break
            else: # File path
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {source}")
                self.cap = cv2.VideoCapture(source)
                device_index = source

            if not self.cap or not self.cap.isOpened():
                raise IOError(f"Não foi possível abrir a fonte: {source}")

            self.current_source_path = device_index # Store the actual working source

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log_to_console(f"{self.current_mode} iniciado ({width}x{height}).")

            # Update state and UI
            self.is_processing_active = True
            self.frame_count = 0
            self.tracked_objects.clear() # Clear tracks for new stream
            self.next_object_id = 0
            # Don't clear plot data automatically, keep history unless reset
            # self.plot_data.clear()
            # self.update_plot()

            # Update buttons
            if btn and btn.winfo_exists():
                stop_text = "Parar Webcam" if self.current_mode == "Webcam" else "Parar Vídeo"
                stop_cmd = self.toggle_webcam if self.current_mode == "Webcam" else self.stop_processing
                try: btn.config(text=stop_text, command=stop_cmd)
                except tk.TclError: pass

            # Enable monitoring button now that processing is starting
            if monitor_btn and monitor_btn.winfo_exists():
                 try: monitor_btn.config(state=tk.NORMAL)
                 except tk.TclError: pass
                 # Automatically start monitoring if not already started? Optional.
                 # if not self.stats.start_time:
                 #    self.toggle_monitoring()


            self.update_status_label()
            self.update_feed() # Start the loop

        except (FileNotFoundError, IOError, cv2.error, Exception) as e:
            messagebox.showerror("Erro na Fonte", f"Falha ao iniciar a fonte '{source}': {e}")
            traceback.print_exc()
            if self.cap: self.cap.release(); self.cap = None
            self.is_processing_active = False
            # Reset mode/path only if it wasn't a file not found error for a specific file
            if not isinstance(e, FileNotFoundError) or isinstance(source, int):
                self.current_mode = "None"
                self.current_source_path = None
            self.set_placeholder_image()
            # Reset button to default on failure
            if btn and btn.winfo_exists():
                 try: btn.config(text="Iniciar Webcam", command=self.toggle_webcam)
                 except tk.TclError: pass
            # Disable monitoring button on failure
            if monitor_btn and monitor_btn.winfo_exists():
                 try: monitor_btn.config(state=tk.DISABLED)
                 except tk.TclError: pass

            self.update_status_label()

    def update_feed(self):
        """Reads a frame, processes it, and schedules the next update."""
        if not self.is_processing_active or not self.cap or not self.cap.isOpened():
            if self.is_processing_active: # If it was supposed to be running
                 self.log_to_console("Erro: Fonte de vídeo tornou-se indisponível.", is_error=True)
                 self.stop_processing(source_ended_naturally=False)
                 self.set_placeholder_image()
            return

        try:
            ret, frame = self.cap.read()

            if ret:
                self.last_original_frame = frame.copy() # Store current frame
                self.frame_count += 1

                # --- Core Logic: Detection, Tracking, Crossing, Coloring ---
                processed_frame = self.last_original_frame.copy()

                # 1. Object Detection
                detections, names = self.model_manager.detect_objects(
                    processed_frame, self.confidence_threshold.get()
                )

                # 2. Prepare current frame detections for tracker
                current_centroids = {} # id -> (x, y)
                current_boxes = {} # id -> (x1, y1, x2, y2)
                current_details = {} # id -> {'class': name, 'conf': conf}
                temp_id_counter = -1 # Temporary IDs for current frame detections
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls_id = det
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_centroids[temp_id_counter] = (cx, cy)
                    current_boxes[temp_id_counter] = (x1, y1, x2, y2)
                    current_details[temp_id_counter] = {'class': names[i], 'conf': conf}
                    temp_id_counter -= 1

                # 3. Update Tracker
                self.update_tracker(current_centroids, current_details, current_boxes)

                # 4. Check Line Crossing & Draw
                timestamp = datetime.datetime.now()
                crossing_occurred_this_frame = False
                for obj_id, data in list(self.tracked_objects.items()): # Iterate over copy for safe deletion
                    if not data['counted'] and data['centroid'] and data['prev_centroid']:
                        crossed, direction = self.check_line_crossing(data['prev_centroid'], data['centroid'], self.line_start, self.line_end)
                        if crossed:
                            # Check if crossing direction matches user setting
                            desired_direction = self.crossing_direction.get()
                            match = False
                            if desired_direction == "Esquerda -> Direita" and direction == "LR": match = True
                            elif desired_direction == "Direita -> Esquerda" and direction == "RL": match = True
                            elif desired_direction == "Cima -> Baixo" and direction == "TB": match = True
                            elif desired_direction == "Baixo -> Cima" and direction == "BT": match = True

                            if match:
                                # Mark as counted
                                data['counted'] = True
                                class_name = data['class']
                                box = data.get('box') # Get the latest box associated with the track

                                # 5. Get Color (only if crossing)
                                color_name = "Desconhecida"
                                if box:
                                    try:
                                        color_name = self.get_object_color(self.last_original_frame, box)
                                    except Exception as color_e:
                                        self.log_to_console(f"Erro ao obter cor para obj {obj_id}: {color_e}", is_error=True)

                                # 6. Update Statistics (if monitoring is active)
                                if self.stats.start_time:
                                    self.stats.update(class_name, color_name, timestamp)
                                    crossing_occurred_this_frame = True

                                # Visual feedback for counting
                                self.log_to_console(f"Objeto {obj_id} ({class_name}, {color_name}) cruzou a linha ({direction}).")
                                # Flash line or draw object differently?
                                if self.line_start and self.line_end:
                                    cv2.line(processed_frame, self._orig_to_display(self.line_start), self._orig_to_display(self.line_end), (0, 0, 255), LINE_THICKNESS + 2) # Flash red


                    # Draw tracked object info
                    if data['centroid'] and data['box']:
                        cx, cy = data['centroid']
                        x1, y1, x2, y2 = data['box']
                        disp_cx, disp_cy = self._orig_to_display((cx, cy))
                        disp_x1, disp_y1 = self._orig_to_display((x1, y1))
                        disp_x2, disp_y2 = self._orig_to_display((x2, y2))

                        # Draw box (maybe different color if counted?)
                        box_color = (255, 100, 0) if data['counted'] else (0, 200, 0) # Orange if counted, Green otherwise
                        cv2.rectangle(processed_frame, (disp_x1, disp_y1), (disp_x2, disp_y2), box_color, 1)
                        # Draw centroid
                        cv2.circle(processed_frame, (disp_cx, disp_cy), 3, TRACKING_DOT_COLOR, -1)
                        # Draw ID and Class
                        label_text = f"ID:{obj_id} {data['class']}"
                        cv2.putText(processed_frame, label_text, (disp_x1, disp_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

                # 7. Draw Counting Line
                if self.line_start and self.line_end:
                     # Draw line on the *processed_frame* in display coordinates
                     p1_disp = self._orig_to_display(self.line_start)
                     p2_disp = self._orig_to_display(self.line_end)
                     if p1_disp and p2_disp: # Ensure conversion was successful
                         cv2.line(processed_frame, p1_disp, p2_disp, LINE_COLOR, LINE_THICKNESS)


                # 8. Update UI Elements (Stats, Count, Plot) if crossing occurred
                if crossing_occurred_this_frame:
                    self.update_count_label()
                    self.update_stats_display()
                    self.update_plot()
                    self.update_plot_class_options() # Refresh classes if a new one was counted

                # 9. Display the final processed frame
                self.display_cv2_image(processed_frame)

                # Schedule next update
                try: self.window.after(self.delay, self.update_feed)
                except tk.TclError:
                    self.log_to_console("Janela fechada durante o loop de atualização.")
                    if self.is_processing_active: self.stop_processing(source_ended_naturally=False)

            else: # End of video file or read error
                self.is_processing_active = False
                is_video_file = (self.current_mode == "Video")
                log_msg = "Fim do arquivo de vídeo." if is_video_file else "Falha na leitura do frame da webcam."
                self.log_to_console(log_msg)
                self.stop_processing(source_ended_naturally=is_video_file)
                # Keep last frame displayed for video files, show placeholder for webcam error
                if not is_video_file:
                    self.set_placeholder_image()
                self.update_status_label()

        except cv2.error as cv_err:
            self.log_to_console(f"Erro OpenCV durante atualização: {cv_err}", is_error=True)
            self.stop_processing(source_ended_naturally=False)
            self.set_placeholder_image()
        except Exception as e:
            self.log_to_console(f"Erro inesperado na atualização: {e}", is_error=True)
            traceback.print_exc()
            self.stop_processing(source_ended_naturally=False)
            self.set_placeholder_image()

    # --- Model Loading ---
    def load_model_dialog(self):
        """Opens a dialog to select a model file (.pt)."""
        if not TORCH_AVAILABLE:
             messagebox.showerror("Erro", "PyTorch/Ultralytics não está instalado. Não é possível carregar modelos.")
             return

        filepath = filedialog.askopenfilename(
            title="Selecionar Arquivo de Modelo YOLO (.pt)",
            filetypes=[("PyTorch Models", "*.pt"), ("Todos os Arquivos", "*.*")]
        )
        if filepath:
            if self.model_manager.load_model(filepath):
                messagebox.showinfo("Sucesso", f"Modelo '{os.path.basename(filepath)}' carregado.")
                self._update_model_status_label()
                self.update_plot_class_options() # Update plot dropdown with new classes
            else:
                messagebox.showerror("Erro", f"Falha ao carregar modelo de '{filepath}'. Verifique o console.")
                self._update_model_status_label()
        else:
            self.log_to_console("Nenhum arquivo de modelo selecionado.")

    # --- Line Drawing ---

    def _display_to_orig(self, point_disp):
        """Converts coordinates from display label space to original frame space."""
        if self.last_original_frame is None or self.last_display_scale == 0:
            return None # Cannot convert without original frame context

        disp_x, disp_y = point_disp
        offset_x, offset_y = self.last_display_offsets
        scale = self.last_display_scale

        # Check if the point is within the scaled image area on the display
        img_area_x1 = offset_x
        img_area_y1 = offset_y
        img_area_x2 = offset_x + self.last_display_dims[0]
        img_area_y2 = offset_y + self.last_display_dims[1]

        if not (img_area_x1 <= disp_x < img_area_x2 and img_area_y1 <= disp_y < img_area_y2):
            # Click was outside the actual image, maybe on the border/padding
            # Optionally clamp or return None. Returning None is safer.
             return None

        # Calculate original coordinates
        orig_x = int((disp_x - offset_x) / scale)
        orig_y = int((disp_y - offset_y) / scale)

        # Clamp to original image bounds (important!)
        h_orig, w_orig = self.last_original_frame.shape[:2]
        orig_x = max(0, min(orig_x, w_orig - 1))
        orig_y = max(0, min(orig_y, h_orig - 1))

        return (orig_x, orig_y)

    def _orig_to_display(self, point_orig):
        """Converts coordinates from original frame space to display label space."""
        if point_orig is None or self.last_display_scale == 0:
             return None

        orig_x, orig_y = point_orig
        offset_x, offset_y = self.last_display_offsets
        scale = self.last_display_scale

        disp_x = int(orig_x * scale + offset_x)
        disp_y = int(orig_y * scale + offset_y)

        # No clamping needed here, as it's converting *to* display coords

        return (disp_x, disp_y)


    def _on_toggle_draw_mode(self):
        """Handles the checkbutton for enabling/disabling line drawing."""
        if self.enable_line_drawing.get():
            self.log_to_console("Modo de desenho de linha HABILITADO. Clique e arraste na imagem.")
            self.display_label.config(cursor="crosshair")
        else:
            self.log_to_console("Modo de desenho de linha DESABILITADO.")
            self.display_label.config(cursor="") # Reset cursor
            self.drawing_line = False # Ensure drawing stops if checkbox unchecked mid-drag


    def on_line_draw_start(self, event):
        """Starts drawing the line if draw mode is enabled."""
        if not self.enable_line_drawing.get():
            return # Ignore clicks if drawing is not enabled

        if self.last_original_frame is None:
            self.log_to_console("Carregue uma imagem ou vídeo antes de desenhar a linha.")
            return

        orig_point = self._display_to_orig((event.x, event.y))
        if orig_point:
            self.line_start = orig_point
            self.line_end = orig_point # Initialize end point to start point
            self.drawing_line = True
            self.log_to_console(f"Início da linha: {self.line_start} (Original Coords)")
            # Draw initial point or line immediately on the *last processed* frame if available
            self._redraw_display_with_line()


    def on_line_draw_motion(self, event):
        """Updates the line end point while dragging."""
        if not self.drawing_line or not self.enable_line_drawing.get():
            return

        orig_point = self._display_to_orig((event.x, event.y))
        if orig_point:
             self.line_end = orig_point
             # Redraw the display with the updated line
             self._redraw_display_with_line()


    def on_line_draw_end(self, event):
        """Finalizes the line drawing."""
        if not self.drawing_line or not self.enable_line_drawing.get():
            # If draw mode was disabled mid-draw, just reset state
            if self.drawing_line:
                 self.drawing_line = False
                 self.line_start = None # Clear potential partial line
                 self.line_end = None
                 self._redraw_display_with_line() # Redraw without the line
            return

        orig_point = self._display_to_orig((event.x, event.y))
        if orig_point:
            self.line_end = orig_point
            self.log_to_console(f"Fim da linha: {self.line_end} (Original Coords)")
        else:
            # If mouse released outside image, use the last valid point
            self.log_to_console(f"Fim da linha (último ponto válido): {self.line_end} (Original Coords)")

        self.drawing_line = False

        # Final redraw
        self._redraw_display_with_line()

        # Optional: Automatically disable drawing mode after line is drawn?
        # self.enable_line_drawing.set(False)
        # self._on_toggle_draw_mode()


    def _redraw_display_with_line(self):
         """Helper to redraw the current frame (original or last processed) with the current line."""
         # Decide which frame to draw on: original if not processing, last processed if processing
         frame_to_draw_on = None
         if self.is_processing_active and self.last_processed_frame is not None:
             # If processing, redraw on the latest processed frame to include detections etc.
             # However, drawing the line *during* processing is handled in update_feed.
             # This redraw is mainly for static images or when drawing while paused.
             # Let's use the last original frame for consistency when *manually* drawing.
             frame_to_draw_on = self.last_original_frame
         elif self.last_original_frame is not None:
              # Use original if not processing or no processed frame available
              frame_to_draw_on = self.last_original_frame
         else:
              # No frame available, do nothing
              return

         display_frame = frame_to_draw_on.copy()

         # Draw the line if points are valid
         if self.line_start and self.line_end:
             # Convert original line coords to display coords for drawing
             p1_disp = self._orig_to_display(self.line_start)
             p2_disp = self._orig_to_display(self.line_end)
             if p1_disp and p2_disp:
                 cv2.line(display_frame, p1_disp, p2_disp, LINE_COLOR, LINE_THICKNESS)
                 # Draw circles at endpoints for clarity during drawing
                 if self.drawing_line or self.enable_line_drawing.get():
                      cv2.circle(display_frame, p1_disp, 4, (0,0,255), -1)
                      cv2.circle(display_frame, p2_disp, 4, (0,0,255), -1)


         # Display the frame with the line
         self.display_cv2_image(display_frame)


    # --- Tracking Logic ---

    def update_tracker(self, current_centroids, current_details, current_boxes):
        """Updates object tracks based on current detections."""
        if not current_centroids:
            # No detections this frame, decrement disappearance count for existing tracks
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['last_frame'] -= 1
                if self.tracked_objects[obj_id]['last_frame'] < -MAX_FRAMES_DISAPPEARED:
                    # self.log_to_console(f"Removendo track {obj_id} (desaparecido).")
                    del self.tracked_objects[obj_id]
            return

        # If no existing tracks, register all new detections
        if not self.tracked_objects:
            for temp_id, centroid in current_centroids.items():
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'prev_centroid': None, # No previous yet
                    'last_frame': self.frame_count,
                    'counted': False,
                    'class': current_details[temp_id]['class'],
                    'box': current_boxes[temp_id]
                }
                self.next_object_id += 1
            return

        # --- Matching existing tracks with current detections ---
        existing_ids = list(self.tracked_objects.keys())
        existing_centroids = [self.tracked_objects[id]['centroid'] for id in existing_ids]
        current_temp_ids = list(current_centroids.keys())
        current_cents = list(current_centroids.values())

        # Calculate distance matrix (Euclidean distance)
        # Requires scipy if available, otherwise manual calculation
        try:
            from scipy.spatial import distance
            dist_matrix = distance.cdist(np.array(existing_centroids), np.array(current_cents))
        except ImportError:
            # Manual distance calculation
            dist_matrix = np.zeros((len(existing_centroids), len(current_cents)))
            for i, ec in enumerate(existing_centroids):
                 for j, cc in enumerate(current_cents):
                     dist_matrix[i, j] = math.sqrt((ec[0]-cc[0])**2 + (ec[1]-cc[1])**2)


        # --- Simple Greedy Matching (can be improved with Hungarian algorithm) ---
        matched_indices = set()
        matched_existing_ids = set()

        # Iterate through rows (existing tracks) and find the best match (minimum distance)
        # This is a simplified approach. A more robust method uses algorithms like Hungarian.
        # Get rows sorted by the minimum distance in that row
        rows = dist_matrix.min(axis=1).argsort()
        # Get columns sorted by the minimum distance in that col (for unmatched current detections later)
        cols = dist_matrix.min(axis=0).argsort()

        # Match existing tracks to current detections
        for row in rows:
            if row in matched_existing_ids: continue # Already matched this existing track

            # Find the closest current detection to this existing track
            best_col = -1
            min_dist = MAX_TRACKING_DISTANCE

            for col in range(dist_matrix.shape[1]):
                 if col not in matched_indices: # If this current detection isn't matched yet
                      d = dist_matrix[row, col]
                      if d < min_dist:
                           min_dist = d
                           best_col = col

            if best_col != -1:
                 # Found a match within threshold
                 existing_id = existing_ids[row]
                 current_temp_id = current_temp_ids[best_col]

                 # Update track
                 self.tracked_objects[existing_id]['prev_centroid'] = self.tracked_objects[existing_id]['centroid'] # Store previous position
                 self.tracked_objects[existing_id]['centroid'] = current_centroids[current_temp_id]
                 self.tracked_objects[existing_id]['last_frame'] = self.frame_count
                 self.tracked_objects[existing_id]['box'] = current_boxes[current_temp_id] # Update box
                 # Keep class and counted status

                 matched_indices.add(best_col)
                 matched_existing_ids.add(row)


        # --- Handle Unmatched Detections (Register New Tracks) ---
        unmatched_current_cols = set(range(len(current_temp_ids))) - matched_indices
        for col in unmatched_current_cols:
            temp_id = current_temp_ids[col]
            self.tracked_objects[self.next_object_id] = {
                 'centroid': current_centroids[temp_id],
                 'prev_centroid': None,
                 'last_frame': self.frame_count,
                 'counted': False,
                 'class': current_details[temp_id]['class'],
                 'box': current_boxes[temp_id]
            }
            # self.log_to_console(f"Registrando novo track {self.next_object_id} ({current_details[temp_id]['class']}).")
            self.next_object_id += 1

        # --- Handle Unmatched Existing Tracks (Decrement Disappearance Count) ---
        unmatched_existing_rows = set(range(len(existing_ids))) - matched_existing_ids
        for row in unmatched_existing_rows:
            existing_id = existing_ids[row]
            self.tracked_objects[existing_id]['last_frame'] -= 1
            # Keep previous centroid, clear current? No, keep last known centroid.
            # self.tracked_objects[existing_id]['centroid'] = None # Mark as currently unseen?
            self.tracked_objects[existing_id]['box'] = None # No current box
            if self.tracked_objects[existing_id]['last_frame'] < -MAX_FRAMES_DISAPPEARED:
                # self.log_to_console(f"Removendo track {existing_id} (desaparecido).")
                del self.tracked_objects[existing_id]


    # --- Line Crossing Logic ---

    def _check_line_intersection(self, p1, p2, p3, p4):
        """Checks if line segment p1p2 intersects line segment p3p4."""
        # Using vector cross product method
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases (points are collinear) - Check if they lie on segment
        # Not strictly needed for simple crossing, but included for completeness
        def on_segment(p, q, r):
             return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                     q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True

        return False

    def check_line_crossing(self, prev_point, current_point, line_start, line_end):
        """
        Checks if the movement from prev_point to current_point crosses the line segment.
        Returns (crossed, direction) where direction is 'LR', 'RL', 'TB', 'BT', or None.
        Points must be in original frame coordinates.
        """
        if not prev_point or not current_point or not line_start or not line_end:
            return False, None

        # Check for intersection between the object's movement vector and the counting line
        intersects = self._check_line_intersection(prev_point, current_point, line_start, line_end)

        if not intersects:
            return False, None

        # Determine direction relative to the line's normal or orientation
        line_dx = line_end[0] - line_start[0]
        line_dy = line_end[1] - line_start[1]
        move_dx = current_point[0] - prev_point[0]
        move_dy = current_point[1] - prev_point[1]

        # Calculate cross product of line vector and movement vector start relative to line start
        # This indicates which side the movement started on relative to the line's direction
        # cross_product = line_dx * (prev_point[1] - line_start[1]) - line_dy * (prev_point[0] - line_start[0])
        # A simpler approach for horizontal/vertical bias:
        direction = None
        if abs(line_dx) > abs(line_dy): # More horizontal line
            if move_dx > 0: direction = "LR" # Moving right
            elif move_dx < 0: direction = "RL" # Moving left
        elif abs(line_dy) > abs(line_dx): # More vertical line
             if move_dy > 0: direction = "TB" # Moving down (Top to Bottom)
             elif move_dy < 0: direction = "BT" # Moving up (Bottom to Top)
        else: # Diagonal - use cross product or relative position
             # Let's determine based on relative position to line center for simplicity here
             line_center_x = (line_start[0] + line_end[0]) / 2
             line_center_y = (line_start[1] + line_end[1]) / 2
             if current_point[0] > line_center_x and prev_point[0] <= line_center_x: direction = "LR" # Approx Left to Right
             elif current_point[0] < line_center_x and prev_point[0] >= line_center_x: direction = "RL" # Approx Right to Left
             elif current_point[1] > line_center_y and prev_point[1] <= line_center_y: direction = "TB" # Approx Top to Bottom
             elif current_point[1] < line_center_y and prev_point[1] >= line_center_y: direction = "BT" # Approx Bottom to Top


        return True, direction


    # --- Color Identification ---

    def get_object_color(self, frame, box):
        """
        Identifies the dominant color within the bounding box using predefined HSV ranges.
        Returns the name of the matched color.
        """
        x1, y1, x2, y2 = box
        # Ensure coordinates are within frame bounds and valid
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

        if x1 >= x2 or y1 >= y2:
            return "Desconhecida" # Invalid box

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "Desconhecida"

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        dominant_color_name = "Desconhecida"
        max_pixels = 0

        for color_name, (lower, upper) in PREDEFINED_COLORS.items():
            if color_name == "Desconhecida": continue # Skip fallback

            lower_np = np.array(lower)
            upper_np = np.array(upper)

            # Handle hue wrap-around for mask creation
            if lower_np[0] > upper_np[0]:
                mask1 = cv2.inRange(hsv_roi, np.array([lower_np[0], lower_np[1], lower_np[2]]), np.array([179, upper_np[1], upper_np[2]]))
                mask2 = cv2.inRange(hsv_roi, np.array([0, lower_np[1], lower_np[2]]), np.array([upper_np[0], upper_np[1], upper_np[2]]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_roi, lower_np, upper_np)

            # Count non-zero pixels in the mask
            pixel_count = cv2.countNonZero(mask)

            # Simple dominance: color with most pixels wins
            # (More sophisticated methods exist, e.g., histograms, clustering)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color_name = color_name

        # Add a threshold? Only identify if a significant portion is that color?
        # total_pixels = (x2 - x1) * (y2 - y1)
        # if max_pixels / total_pixels < 0.1: # Example: require at least 10% match
        #     return "Desconhecida"

        return dominant_color_name


    # --- Statistics Control ---
    def toggle_monitoring(self):
        """Starts or stops the statistics monitoring period."""
        monitor_btn = getattr(self, 'btn_start_stop_monitor', None)
        if not monitor_btn: return

        if self.stats.start_time:
            # Stop monitoring
            self.stats.stop_monitoring()
            self.log_to_console("Monitoramento de estatísticas PARADO.")
            try: monitor_btn.config(text="Iniciar Monitoramento")
            except tk.TclError: pass
            # Keep stats, just mark stop time
        else:
            # Start monitoring
            self.stats.start_monitoring()
            self.log_to_console("Monitoramento de estatísticas INICIADO.")
            try: monitor_btn.config(text="Parar Monitoramento")
            except tk.TclError: pass

        self.update_stats_display()
        self.update_status_label() # Reflect monitoring state change

    def reset_statistics(self):
        """Resets all collected statistics and the plot."""
        if messagebox.askyesno("Resetar Estatísticas", "Tem certeza que deseja apagar todas as estatísticas e contagens?"):
            self.log_to_console("Resetando estatísticas...")
            self.stats.reset()
            self.tracked_objects.clear() # Also clear tracks as counts are reset
            self.next_object_id = 0
            # Reset 'counted' flag for any currently tracked objects if needed?
            # Clearning tracked_objects handles this.

            # Reset plot
            self.plot_data.clear()
            self.selected_plot_class.set("Total")
            self.update_plot_class_options()
            self.update_plot()

            # Update UI
            self.update_count_label()
            self.update_stats_display()
            self.log_to_console("Estatísticas resetadas.")

            # Reset monitoring button state if monitoring was active
            monitor_btn = getattr(self, 'btn_start_stop_monitor', None)
            if monitor_btn and monitor_btn.winfo_exists():
                try:
                     monitor_btn.config(text="Iniciar Monitoramento")
                     # Re-enable if a source is loaded/ready? Or disable until process starts?
                     if self.current_mode != "None": monitor_btn.config(state=tk.NORMAL)
                     else: monitor_btn.config(state=tk.DISABLED)
                except tk.TclError: pass
            self.update_status_label()


    # --- Plotting ---
    def setup_plot(self):
        """Initializes the Matplotlib plot."""
        try:
            plt.style.use('dark_background')
            self.fig = Figure(figsize=(5, 2.5), dpi=90) # Adjusted size slightly
            self.fig.patch.set_facecolor('#2b2b2b')
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#3c3f41')

            # Apply styling (called again in update_plot)
            self._style_plot_axes()

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame) # Place in its dedicated frame
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.grid(row=0, column=0, sticky="nsew") # Fill the dedicated frame
            self.fig.tight_layout(pad=0.5) # Add padding
            self.canvas.draw()
            self.log_to_console("Plot inicializado.")
        except Exception as e:
            self.log_to_console(f"Erro ao inicializar o plot: {e}", is_error=True)
            messagebox.showerror("Erro de Plot", f"Falha ao inicializar o gráfico: {e}")

    def _style_plot_axes(self):
         """Applies common styling to plot axes."""
         if not self.ax: return
         class_name = self.selected_plot_class.get()
         title = f"Contagem Acumulada ({class_name})"
         self.ax.set_title(title, color='white', fontsize=10)
         self.ax.set_xlabel("Tempo", color='lightgrey', fontsize=8)
         self.ax.set_ylabel("Contagem", color='lightgrey', fontsize=8)
         self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
         self.ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
         self.ax.tick_params(axis='x', colors='lightgrey', rotation=25, labelsize=7)
         self.ax.tick_params(axis='y', colors='lightgrey', labelsize=7)
         self.ax.spines['top'].set_color('grey'); self.ax.spines['bottom'].set_color('grey')
         self.ax.spines['left'].set_color('grey'); self.ax.spines['right'].set_color('grey')
         self.ax.grid(True, linestyle='--', color='grey', alpha=0.4)
         self.ax.set_facecolor('#3c3f41') # Reapply facecolor

    def update_plot_class_options(self):
        """Updates the dropdown menu for selecting which class to plot."""
        if not hasattr(self, 'combo_plot_class') or not self.combo_plot_class.winfo_exists():
            return

        # Get all classes that have been counted + "Total"
        counted_classes = list(self.stats.counts_per_class.keys())
        options = ["Total"] + sorted(counted_classes)

        current_selection = self.selected_plot_class.get()

        try:
            self.combo_plot_class['values'] = options
            # Keep current selection if still valid, otherwise default to "Total"
            if current_selection not in options:
                self.selected_plot_class.set("Total")
                # self.update_plot() # Update plot if selection changed
        except tk.TclError: pass

    def update_plot(self, event=None):
        """Updates the plot with current time-series data."""
        if not hasattr(self, 'ax') or not self.ax or not hasattr(self, 'canvas') or not self.canvas: return

        try:
            self.ax.clear()
            class_to_plot = self.selected_plot_class.get()
            plot_title = f"Contagem Acumulada ({class_to_plot})"
            data_to_plot = None
            color = 'cyan'

            if class_to_plot == "Total":
                # Aggregate data for total count
                all_times = set()
                temp_total_data = defaultdict(int)
                for cls, series in self.stats.time_series_counts.items():
                    for timestamp, count in series:
                         all_times.add(timestamp)
                # Create a cumulative timeline (this is approximate, assumes simultaneous updates)
                cumulative_total = 0
                total_plot_points = []
                last_counts = defaultdict(int)
                # Iterate through sorted timestamps across all classes
                for t in sorted(list(all_times)):
                    current_total = 0
                    for cls, series in self.stats.time_series_counts.items():
                         # Find the count for class 'cls' at or before time 't'
                         count_at_t = last_counts[cls] # Start with previous count
                         for ts, cnt in series:
                              if ts <= t:
                                   count_at_t = cnt
                              else: break # series is sorted by time
                         current_total += count_at_t
                         last_counts[cls] = count_at_t # Update last known count for this class

                    total_plot_points.append((t, current_total))

                if total_plot_points:
                     data_to_plot = total_plot_points
                color = 'lime'


            elif class_to_plot in self.stats.time_series_counts:
                # Get data for the specific class, ensuring it's not empty
                series = self.stats.time_series_counts[class_to_plot]
                if series: # Check if the deque is not empty
                    data_to_plot = list(series) # Convert deque to list for plotting
                color = 'orange' # Different color for specific class

            # Plot if data exists
            if data_to_plot:
                # Limit data points shown for performance if necessary (e.g., last 100 points)
                if len(data_to_plot) > 150:
                    data_to_plot = data_to_plot[-150:]

                times, counts = zip(*data_to_plot)
                self.ax.plot(times, counts, marker='.', linestyle='-', markersize=3, color=color)
                # Dynamic Y-axis limit
                max_count = max(counts) if counts else 1
                self.ax.set_ylim(bottom=0, top=max(1, max_count * 1.2))
            else:
                # Set a default Y range if no data or empty series
                self.ax.set_ylim(bottom=0, top=5)

            # Reapply plot styling
            self._style_plot_axes()
            self.fig.tight_layout(pad=0.5)
            self.canvas.draw()

        except Exception as e:
            self.log_to_console(f"Erro ao atualizar o plot: {e}", is_error=True)
            # traceback.print_exc() # Uncomment for detailed plot errors

    # --- Closing ---
    def on_closing(self):
        """Handles application cleanup on window close."""
        self.log_to_console("Fechando a aplicação...")
        self.is_processing_active = False # Stop loops first
        self.stop_processing(source_ended_naturally=False) # Release camera/video

        # Clean up plot resources
        if hasattr(self,'canvas') and self.canvas:
            try:
                self.canvas.get_tk_widget().destroy()
                if self.fig: plt.close(self.fig)
                self.log_to_console("Recursos do plot liberados.")
            except Exception as e:
                self.log_to_console(f"Erro ao limpar plot: {e}", is_error=True)

        # Wait a moment before destroying window
        self.window.after(150, self.window.destroy)
        print("Aplicação encerrada.")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Use a modern theme if available
    try:
        style = ttk.Style(root)
        available_themes = style.theme_names()
        print(f"Available themes: {available_themes}")
        # Prefer 'clam', 'alt', 'vista' if available for better looks
        for theme in ['clam', 'alt', 'vista', 'default']:
             if theme in available_themes:
                  style.theme_use(theme)
                  print(f"Using ttk theme: '{theme}'")
                  break
    except tk.TclError:
        print("Aviso: Não foi possível configurar o estilo ttk.")

    app = ObjectCounterApp(root, "Contador de Objetos em Linha de Montagem")
    root.mainloop()