#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gerenciador de estatísticas para o sistema IndustriaCount
Armazena e gerencia estatísticas sobre objetos detectados e contados
"""

import datetime
import time
import json
from collections import defaultdict

class ObjectStatistics:
    """Stores and manages statistics about detected and counted objects with efficient data structures."""
    def __init__(self):
        self.counts_per_class = defaultdict(int)
        self.counts_per_color_per_class = defaultdict(lambda: defaultdict(int))
        self.time_series_counts = defaultdict(list)  # class -> [(timestamp, cumulative_count)]
        self.total_counted = 0
        self.start_time = None
        self.last_update_time = None
        self.count_rate = 0.0  # Objects per minute
        self.max_time_series_points = 1000  # Limit points to prevent memory issues
        self._last_counts = {}  # For rate calculation
        self.last_counted_class = None  # Armazenar a última classe contada
        self.last_counted_color = None  # Armazenar a última cor contada

    def start_monitoring(self):
        """Start the statistics monitoring period."""
        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time
        self._last_counts = {'total': 0, 'timestamp': self.start_time}
        print(f"Monitoring started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_monitoring(self):
        """Stop the statistics monitoring period."""
        self.last_update_time = datetime.datetime.now()
        # Calculate final rate
        self._update_count_rate()
        print(f"Monitoring stopped at: {self.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _update_count_rate(self):
        """Calculate the current count rate (objects per minute)."""
        if not self.start_time or self.total_counted == 0:
            self.count_rate = 0.0
            return
            
        # Calculate time difference in minutes
        now = datetime.datetime.now()
        time_diff = (now - self.start_time).total_seconds() / 60.0
        
        if time_diff > 0:
            self.count_rate = self.total_counted / time_diff
        else:
            self.count_rate = 0.0
            
        # Update last counts for next calculation
        self._last_counts = {'total': self.total_counted, 'timestamp': now}

    def update(self, class_name, color_name, timestamp):
        """Records a counted object with timestamp."""
        if self.start_time is None:  # Ensure monitoring has started
            self.start_monitoring()

        self.total_counted += 1
        self.counts_per_class[class_name] += 1
        self.counts_per_color_per_class[class_name][color_name] += 1
        self.last_update_time = timestamp
        
        # Armazenar a última classe e cor contadas
        self.last_counted_class = class_name
        self.last_counted_color = color_name

        # Add to time series for plotting cumulative count of this class
        cumulative_count = self.counts_per_class[class_name]
        self.time_series_counts[class_name].append((timestamp, cumulative_count))
        
        # Limit time series points to prevent memory issues
        if len(self.time_series_counts[class_name]) > self.max_time_series_points:
            # Keep first point, last N-1 points
            self.time_series_counts[class_name] = [
                self.time_series_counts[class_name][0]
            ] + self.time_series_counts[class_name][-(self.max_time_series_points-1):]
        
        # Update count rate every 10 counts or if it's been more than 5 seconds
        if (self.total_counted % 10 == 0 or 
            (timestamp - self._last_counts.get('timestamp', timestamp)).total_seconds() > 5):
            self._update_count_rate()

    def get_summary(self):
        """Returns a formatted string summary of the statistics."""
        if self.start_time is None:
            return "Monitoramento não iniciado."

        duration_seconds = (self.last_update_time - self.start_time).total_seconds()
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))

        summary = f"--- Resumo Estatístico ({duration_str}) ---\n"
        summary += f"Total Contado: {self.total_counted}\n"
        
        # Add count rate
        if duration_seconds > 0:
            count_per_min = self.count_rate
            count_per_hour = count_per_min * 60
            summary += f"Taxa: {count_per_min:.1f} obj/min ({count_per_hour:.1f} obj/hora)\n"

        if self.counts_per_class:
            summary += "\nContagem por Classe:\n"
            for cls, count in sorted(self.counts_per_class.items(), key=lambda x: x[1], reverse=True):
                percentage = 100.0 * count / self.total_counted if self.total_counted > 0 else 0
                summary += f"  - {cls}: {count} ({percentage:.1f}%)\n"

        if self.counts_per_color_per_class:
            summary += "\nContagem por Cor (dentro da Classe):\n"
            for cls, color_counts in sorted(self.counts_per_color_per_class.items()):
                if color_counts:  # Only show classes with color info
                    summary += f"  - {cls}:\n"
                    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = 100.0 * count / self.counts_per_class[cls]
                        summary += f"    - {color}: {count} ({percentage:.1f}%)\n"
        summary += "--------------------------------------"
        return summary

    def get_data_for_export(self):
        """Returns a dictionary of statistics data suitable for export to JSON/CSV."""
        if not self.start_time:
            return {"error": "No monitoring data available"}
            
        duration_seconds = (self.last_update_time - self.start_time).total_seconds()
        
        export_data = {
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.last_update_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "total_counted": self.total_counted,
            "count_rate_per_minute": self.count_rate,
            "counts_per_class": dict(self.counts_per_class),
            "counts_per_color_per_class": {k: dict(v) for k, v in self.counts_per_color_per_class.items()},
            # Convert timestamps to strings for JSON compatibility
            "time_series": {
                cls: [(ts.strftime("%Y-%m-%d %H:%M:%S.%f"), count) 
                      for ts, count in points]
                for cls, points in self.time_series_counts.items()
            }
        }
        return export_data

    def export_to_json(self, filepath):
        """Export statistics to a JSON file."""
        try:
            data = self.get_data_for_export()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, f"Estatísticas exportadas para {filepath}"
        except Exception as e:
            return False, f"Erro ao exportar estatísticas: {e}"

    def reset(self):
        """Resets all statistics."""
        self.counts_per_class.clear()
        self.counts_per_color_per_class.clear()
        self.time_series_counts.clear()
        self.total_counted = 0
        self.start_time = None
        self.last_update_time = None
        self.count_rate = 0.0
        self._last_counts = {}
        self.last_counted_class = None
        self.last_counted_color = None
        print("Statistics reset.")  # Debug
