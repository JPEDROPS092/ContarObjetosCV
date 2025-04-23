#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerenciador de modelos para o sistema IndustriaCount
Responsável por carregar e executar modelos de detecção de objetos
"""

import os
import time
import traceback
import numpy as np
from collections import deque
import random
import gc

# Verificação de disponibilidade do PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    import torchvision
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
    print("PyTorch, Torchvision, e Ultralytics carregados com sucesso.")
except ImportError as e:
    print(f"Aviso: PyTorch/Torchvision/Ultralytics não encontrados. Recursos de detecção de objetos serão desabilitados. Erro: {e}")
    print("Para instalar: pip install torch torchvision torchaudio ultralytics")
    
    # Define dummy classes for graceful fallback
    class DummyModel:
        def __init__(self): 
            self.names = {}
        def __call__(self, *args, **kwargs): 
            return []  # Return empty list
        def to(self, device):
            return self
    YOLO = lambda *args, **kwargs: DummyModel()

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
        """
        Performs object detection on a frame with performance tracking.
        
        Args:
            frame: The input frame (BGR format)
            confidence_threshold: Minimum confidence threshold for detections
            classes_to_detect: List of class IDs to detect, or None for all classes
            
        Returns:
            List of detections in format (box, class_id, confidence)
        """
        if self.current_model is None or not TORCH_AVAILABLE:
            return []  # No model or no torch

        detections = []  # List to store (box, class_id, confidence)

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
                            
                            # Add to results - format: (box, class_id, confidence)
                            detections.append(((x1, y1, x2, y2), int(cls_id), float(conf)))

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
            if not isinstance(e, (RuntimeError,)):
                traceback.print_exc()
            return []

        return detections
    
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
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.log("Recursos do modelo liberados com sucesso.")
            except Exception as e:
                self.log(f"Erro ao liberar recursos do modelo: {e}", is_error=True)
