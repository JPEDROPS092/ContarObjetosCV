#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções de processamento de imagem para o sistema IndustriaCount
"""

import cv2
import numpy as np

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
    "Desconhecida": ([0, 0, 0], [0, 0, 0])  # Fallback
}

# --- Drawing Constants ---
LINE_COLOR = (0, 255, 255)  # Yellow for the counting line
LINE_THICKNESS = 2
TRACKING_DOT_COLOR = (255, 0, 0)  # Blue dots for tracked centroids
COUNTING_TEXT_COLOR = (0, 255, 0)  # Green for count text
BOUNDING_BOX_COLOR = (0, 200, 0)  # Green for bounding boxes
COUNTED_BOX_COLOR = (255, 100, 0)  # Orange for counted objects

def get_object_color(frame, box):
    """
    Identifies the dominant color within the bounding box using predefined HSV ranges.
    Uses efficient numpy operations for better performance.
    
    Args:
        frame: The original BGR frame
        box: Tuple of (x1, y1, x2, y2) coordinates
        
    Returns:
        str: Name of the dominant color
    """
    x1, y1, x2, y2 = box
    
    # Ensure coordinates are within frame bounds
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

    if x1 >= x2 or y1 >= y2:
        return "Desconhecida"  # Invalid box

    # Extract region of interest
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Desconhecida"
        
    # Convert to HSV once for all color checks
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # For small objects, use average HSV
    if roi.shape[0] * roi.shape[1] < 1000:
        # Calculate average HSV values
        avg_h = np.mean(hsv_roi[:, :, 0])
        avg_s = np.mean(hsv_roi[:, :, 1])
        avg_v = np.mean(hsv_roi[:, :, 2])
        
        # Find closest match to average HSV
        best_match = "Desconhecida"
        min_distance = float('inf')
        
        for color_name, (lower, upper) in PREDEFINED_COLORS.items():
            if color_name == "Desconhecida":
                continue
                
            # Calculate center of the color range
            h_center = (lower[0] + upper[0]) / 2
            if lower[0] > upper[0]:  # Handle hue wrap-around
                h_center = (lower[0] + upper[0] + 180) / 2
                if h_center > 179:
                    h_center -= 180
                    
            s_center = (lower[1] + upper[1]) / 2
            v_center = (lower[2] + upper[2]) / 2
            
            # Calculate distance to color center (weighted for hue importance)
            h_diff = min(abs(avg_h - h_center), 180 - abs(avg_h - h_center))
            distance = (h_diff * 2) + abs(avg_s - s_center) + abs(avg_v - v_center)
            
            if distance < min_distance:
                min_distance = distance
                best_match = color_name
                
        return best_match
    
    # For larger objects, use pixel counting for more accurate results
    dominant_color_name = "Desconhecida"
    max_pixels = 0
    total_pixels = roi.shape[0] * roi.shape[1]
    
    # Create a mask for each color and count pixels
    for color_name, (lower, upper) in PREDEFINED_COLORS.items():
        if color_name == "Desconhecida":
            continue
            
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        
        # Handle hue wrap-around for mask creation
        if lower_np[0] > upper_np[0]:
            mask1 = cv2.inRange(hsv_roi, np.array([lower_np[0], lower_np[1], lower_np[2]]), 
                               np.array([179, upper_np[1], upper_np[2]]))
            mask2 = cv2.inRange(hsv_roi, np.array([0, lower_np[1], lower_np[2]]), 
                               np.array([upper_np[0], upper_np[1], upper_np[2]]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_roi, lower_np, upper_np)
        
        # Count non-zero pixels in the mask
        pixel_count = cv2.countNonZero(mask)
        
        # Simple dominance: color with most pixels wins
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color_name = color_name
    
    # Add a threshold - only identify if a significant portion is that color
    if max_pixels / total_pixels < 0.15:  # Require at least 15% match
        return "Desconhecida"
        
    return dominant_color_name

def check_line_intersection(p1, p2, p3, p4):
    """
    Checks if line segment p1p2 intersects line segment p3p4 using efficient vector math.
    
    Args:
        p1, p2: Points defining the first line segment (movement path)
        p3, p4: Points defining the second line segment (counting line)
        
    Returns:
        bool: True if the line segments intersect, False otherwise
    """
    # Using vector cross product method for line intersection
    def ccw(a, b, c):
        # Counterclockwise test: positive if points a,b,c form a counterclockwise turn
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
    # Check if lines intersect using CCW tests
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def check_line_crossing(prev_point, current_point, line_start, line_end):
    """
    Checks if the movement from prev_point to current_point crosses the line segment.
    Returns (crossed, direction) where direction is 'LR', 'RL', 'TB', 'BT', or None.
    
    Args:
        prev_point: Previous position of the object
        current_point: Current position of the object
        line_start, line_end: Points defining the counting line
        
    Returns:
        tuple: (bool, str) - Whether crossing occurred and in which direction
    """
    if not prev_point or not current_point or not line_start or not line_end:
        return False, None

    # Check for intersection between the object's movement vector and the counting line
    intersects = check_line_intersection(prev_point, current_point, line_start, line_end)

    if not intersects:
        return False, None

    # Determine crossing direction based on the line orientation
    # First, determine if the line is more horizontal or vertical
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]
    
    # Movement vector
    move_dx = current_point[0] - prev_point[0]
    move_dy = current_point[1] - prev_point[1]

    # Determine direction based on line orientation
    if abs(line_dx) > abs(line_dy):
        # More horizontal line - check if crossing left-to-right or right-to-left
        # Use the normal vector to the line to determine which side the movement is from
        # For a horizontal line, the normal points up/down
        normal_y = line_dx  # Normal vector y component (perpendicular to line)
        
        # Dot product of movement vector with normal determines side
        side = move_dx * normal_y
        
        if side > 0:
            return True, "LR"  # Left to Right
        else:
            return True, "RL"  # Right to Left
    else:
        # More vertical line - check if crossing top-to-bottom or bottom-to-top
        # For a vertical line, the normal points left/right
        normal_x = -line_dy  # Normal vector x component
        
        # Dot product of movement vector with normal determines side
        side = move_dy * normal_x
        
        if side > 0:
            return True, "TB"  # Top to Bottom
        else:
            return True, "BT"  # Bottom to Top

def draw_bounding_box(frame, box, label, confidence, color_name, is_counted=False):
    """
    Draw a bounding box with label on the frame
    
    Args:
        frame: The frame to draw on
        box: Tuple of (x1, y1, x2, y2) coordinates
        label: Class label
        confidence: Detection confidence
        color_name: Detected color name
        is_counted: Whether the object has been counted
        
    Returns:
        None (modifies frame in-place)
    """
    x1, y1, x2, y2 = box
    
    # Choose color based on counted status
    box_color = COUNTED_BOX_COLOR if is_counted else BOUNDING_BOX_COLOR
    
    # Draw thicker bounding box (increased thickness for better visibility)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Draw filled background for text
    text = f"{label} ({confidence:.2f}) {color_name}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), box_color, -1)
    
    # Draw text with contrasting color
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw centroid
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, TRACKING_DOT_COLOR, -1)
    
    return cx, cy  # Return centroid coordinates
