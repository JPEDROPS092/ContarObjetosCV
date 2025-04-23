#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de rastreamento de objetos para o IndustriaCount
Implementa algoritmos para rastrear objetos entre frames
"""

import numpy as np
import math

# Configurações de rastreamento
MAX_TRACKING_DISTANCE = 75  # Distância máxima em pixels para considerar o mesmo objeto
MAX_FRAMES_DISAPPEARED = 10  # Quantos frames um objeto pode estar ausente antes de ser removido

class ObjectTracker:
    """
    Rastreador de objetos que mantém o estado dos objetos detectados entre frames
    """
    def __init__(self, log_func=print):
        self.log = log_func
        self.tracked_objects = {}  # id -> {'centroid':(x,y), 'last_frame': frame_count, 'counted': False, 'prev_centroid': (x,y), 'class': name, 'box': box}
        self.next_object_id = 0
        self.frame_count = 0
    
    def reset(self):
        """Reseta o rastreador, limpando todos os objetos rastreados"""
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
    
    def increment_frame(self):
        """Incrementa o contador de frames"""
        self.frame_count += 1
        return self.frame_count
    
    def update(self, current_centroids, current_details, current_boxes):
        """
        Atualiza o rastreador com as detecções do frame atual
        
        Args:
            current_centroids: Dicionário de IDs temporários para centroides
            current_details: Dicionário de IDs temporários para detalhes do objeto
            current_boxes: Dicionário de IDs temporários para bounding boxes
        """
        # No detections - update disappearance counters
        if not current_centroids:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['last_frame'] -= 1
                if self.tracked_objects[obj_id]['last_frame'] < -MAX_FRAMES_DISAPPEARED:
                    del self.tracked_objects[obj_id]
            return

        # If no existing tracks, register all new detections
        if not self.tracked_objects:
            for temp_id, centroid in current_centroids.items():
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'prev_centroid': None,  # No previous yet
                    'last_frame': self.frame_count,
                    'counted': False,
                    'class': current_details[temp_id]['class'],
                    'box': current_boxes[temp_id]
                }
                self.next_object_id += 1
            return

        # --- Matching existing tracks with current detections ---
        # Prepare data for matching
        existing_ids = list(self.tracked_objects.keys())
        existing_centroids = np.array([self.tracked_objects[id]['centroid'] for id in existing_ids])
        current_temp_ids = list(current_centroids.keys())
        current_cents = np.array(list(current_centroids.values()))

        # Calculate distance matrix efficiently using numpy
        # For each existing centroid, calculate distance to all current centroids
        dist_matrix = np.zeros((len(existing_ids), len(current_temp_ids)))
        
        # Use numpy broadcasting for faster distance calculation
        for i, ec in enumerate(existing_centroids):
            # Calculate squared distances (avoid sqrt for performance)
            dist_matrix[i] = np.sum((current_cents - ec) ** 2, axis=1)
        
        # Take square root only for values we'll compare against threshold
        dist_matrix = np.sqrt(dist_matrix)
        
        # --- Optimized Matching Algorithm ---
        # Track which objects have been matched
        matched_indices = set()
        matched_existing_ids = set()
        
        # Sort rows by minimum distance for greedy matching
        row_indices = np.argsort(np.min(dist_matrix, axis=1))
        
        # Match existing tracks to current detections
        for row_idx in row_indices:
            if row_idx in matched_existing_ids:
                continue
                
            # Find the closest unmatched detection
            valid_cols = [j for j in range(dist_matrix.shape[1]) if j not in matched_indices]
            if not valid_cols:
                break
                
            # Get distances to unmatched detections
            distances = dist_matrix[row_idx, valid_cols]
            
            # Find minimum distance
            if len(distances) > 0:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # Check if within threshold
                if min_dist <= MAX_TRACKING_DISTANCE:
                    # Get the actual column index
                    col_idx = valid_cols[min_dist_idx]
                    
                    # Update the track
                    existing_id = existing_ids[row_idx]
                    current_temp_id = current_temp_ids[col_idx]
                    
                    # Store previous position before updating
                    self.tracked_objects[existing_id]['prev_centroid'] = self.tracked_objects[existing_id]['centroid']
                    self.tracked_objects[existing_id]['centroid'] = current_centroids[current_temp_id]
                    self.tracked_objects[existing_id]['last_frame'] = self.frame_count
                    self.tracked_objects[existing_id]['box'] = current_boxes[current_temp_id]
                    
                    # Mark as matched
                    matched_indices.add(col_idx)
                    matched_existing_ids.add(row_idx)
        
        # --- Register New Tracks for Unmatched Detections ---
        unmatched_current_indices = set(range(len(current_temp_ids))) - matched_indices
        for idx in unmatched_current_indices:
            temp_id = current_temp_ids[idx]
            self.tracked_objects[self.next_object_id] = {
                'centroid': current_centroids[temp_id],
                'prev_centroid': None,
                'last_frame': self.frame_count,
                'counted': False,
                'class': current_details[temp_id]['class'],
                'box': current_boxes[temp_id]
            }
            self.next_object_id += 1
        
        # --- Update Unmatched Existing Tracks ---
        unmatched_existing_indices = set(range(len(existing_ids))) - matched_existing_ids
        for idx in unmatched_existing_indices:
            existing_id = existing_ids[idx]
            self.tracked_objects[existing_id]['last_frame'] -= 1
            self.tracked_objects[existing_id]['box'] = None  # No current box
            
            # Remove if disappeared for too long
            if self.tracked_objects[existing_id]['last_frame'] < -MAX_FRAMES_DISAPPEARED:
                del self.tracked_objects[existing_id]
    
    def get_objects(self):
        """Retorna todos os objetos rastreados atualmente"""
        return self.tracked_objects
    
    def mark_as_counted(self, object_id):
        """Marca um objeto como contado"""
        if object_id in self.tracked_objects:
            self.tracked_objects[object_id]['counted'] = True
            return True
        return False
