"""
Tracking result visualization module.
Supports differentiated colors for blueberries at different maturity stages,
and draws detection boxes, tracks, counting lines, counting areas, and more.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class BlueberryVisualizer:
    def __init__(self):
        """Initialize the visualizer and define color mappings."""
        # Blueberry maturity color mapping in BGR format.
        self.stage_colors = {
            0: (255,165,0),    # Flower - orange
            1: (0, 255, 0),        # Green - green
            2: (255, 0, 255),      # Light Purple - magenta
            3: (255, 0, 0),        # Blue - blue
        }
        
        # Maturity stage name mapping.
        self.stage_names = {
            0: 'Flower',
            1: 'Green', 
            2: 'Light Purple',
            3: 'Blue'
        }
        
        # Default line thickness and font settings.
        self.line_thickness = 3
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.text_thickness = 3
    
    def get_stage_color(self, class_id):
        """Get the color for the corresponding maturity stage."""
        return self.stage_colors.get(class_id, (128, 128, 128))  # Default gray.
    
    def get_stage_name(self, class_id):
        """Get the maturity stage name."""
        return self.stage_names.get(class_id, f'Class_{class_id}')
    
    def draw_detection_box(self, frame, x1, y1, x2, y2, track_id, class_id, confidence = None):
        """
        Draw a single detection box.
        
        Args:
            frame: Input frame.
            x1, y1, x2, y2: Detection box coordinates.
            track_id: Track ID.
            class_id: Class ID (maturity stage).
            confidence: Confidence score (optional).
            
        Returns:
            frame: Frame after drawing.
        """
        color = self.get_stage_color(class_id)
        stage_name = self.get_stage_name(class_id)
        
        # Draw the detection box.
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, self.line_thickness)
        
        # Prepare label text.
        if confidence is not None:
            label = f'{stage_name} {track_id} {confidence:.2f}'
        else:
            label = f'{stage_name} {track_id}'
        
        # Calculate text size.
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.text_thickness
        )
        
        # Draw the text background.
        text_x = int(x1)
        text_y = int(y1) - 10 if y1 > 30 else int(y2) + 25
        
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            color, -1
        )
        
        # Draw the text.
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)  # Choose text color based on the background.
        cv2.putText(
            frame, label, (text_x, text_y),
            self.font, self.font_scale, text_color, self.text_thickness
        )
        
        return frame
    
    def draw_tracks(self, frame, tracks):
        """
        Draw all tracking results.
        
        Args:
            frame: Input frame.
            tracks: Tracking results (N, 6) [x1, y1, x2, y2, track_id, class_id].
            
        Returns:
            frame: Frame after drawing.
        """
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            self.draw_detection_box(
                frame, x1, y1, x2, y2, int(track_id), int(class_id)
            )
        
        return frame
    
    def draw_counting_line(self, frame, line_x, color = (255, 0, 0), thickness = 3):
        """
        Draw the counting line.
        
        Args:
            frame: Input frame.
            line_x: X coordinate of the line.
            color: Line color (BGR).
            thickness: Line thickness.
            
        Returns:
            frame: Frame after drawing.
        """
        height = frame.shape[0]
        cv2.line(frame, (line_x, 0), (line_x, height), color, thickness)
        
        # Add a label.
        # cv2.putText(
        #     frame, 'Count Line', (line_x + 5, 30),
        #     self.font, self.font_scale, color, self.text_thickness
        # )
        
        return frame
    
    def draw_counting_area(self, frame, area_x1, area_x2, color = (0, 0, 255), thickness = 3):
        """
        Draw the counting area.
        
        Args:
            frame: Input frame.
            area_x1, area_x2: Left and right boundaries of the area.
            color: Area boundary color (BGR).
            thickness: Boundary thickness.
            
        Returns:
            frame: Frame after drawing.
        """
        height = frame.shape[0]
        
        # Draw the semi-transparent area.
        overlay = frame.copy()
        cv2.rectangle(overlay, (area_x1, 0), (area_x2, height), color, -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Draw the boundary.
        cv2.rectangle(frame, (area_x1, 0), (area_x2, height), color, thickness)
        
        # Add a label.
        # cv2.putText(
        #     frame, 'Count Area', (area_x1 + 5, 30),
        #     self.font, self.font_scale, color, self.text_thickness
        # )
        
        return frame
    
    def draw_statistics(self, frame, id_count, line_count, area_count, position = 'top_left'):
        """
        Draw statistics.
        
        Args:
            frame: Input frame.
            id_count: ID counting results.
            line_count: Line counting results.
            area_count: Area counting results.
            position: Statistics position ('top_left', 'top_right', 'bottom_left', 'bottom_right').
            
        Returns:
            frame: Frame after drawing.
        """
        height, width = frame.shape[:2]
        
        # Determine the starting position.
        if position == 'top_left':
            start_x, start_y = 10, 30
        elif position == 'top_right':
            start_x, start_y = width - 300, 30
        elif position == 'bottom_left':
            start_x, start_y = 10, height - 150
        else:  # bottom_right
            start_x, start_y = width - 300, height - 150
        
        # Draw the background.
        bg_width, bg_height = 280, 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x - 5, start_y - 25), 
                     (start_x + bg_width, start_y + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw the title.
        cv2.putText(frame, 'Counting Results:', (start_x, start_y), 
                   self.font, 0.7, (255, 255, 255), 2)
        
        # Draw the table header.
        y_offset = start_y + 25
        cv2.putText(frame, 'Stage      ID   Line  Area', (start_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics for each maturity stage.
        y_offset += 20
        for stage_id in [0, 1, 2, 3]:
            stage_name = self.get_stage_name(stage_id)
            color = self.get_stage_color(stage_id)
            
            id_cnt = len(id_count.get(stage_name, set()))
            line_cnt = len(line_count.get(stage_name, set()))
            area_cnt = len(area_count.get(stage_name, set()))
            
            text = f'{stage_name[:6]:6} {id_cnt:3d}  {line_cnt:3d}   {area_cnt:3d}'
            cv2.putText(frame, text, (start_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        return frame
    
    def create_legend(self, frame, position = 'bottom_right'):
        """
        Create a color legend.
        
        Args:
            frame: Input frame.
            position: Legend position.
            
        Returns:
            frame: Frame after drawing.
        """
        height, width = frame.shape[:2]
        
        # Determine the starting position.
        if position == 'bottom_right':
            start_x = width - 150
            start_y = height - 120
        elif position == 'bottom_left':
            start_x = 10
            start_y = height - 120
        elif position == 'top_right':
            start_x = width - 150
            start_y = 30
        else:  # top_left
            start_x = 10
            start_y = 30
        
        # Draw the background.
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x - 5, start_y - 25),
                     (start_x + 140, start_y + 90), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw the title.
        cv2.putText(frame, 'Legend:', (start_x, start_y),
                   self.font, 0.6, (255, 255, 255), 2)
        
        # Draw the legend for each maturity stage.
        y_offset = start_y + 20
        for stage_id in [0, 1, 2, 3]:
            stage_name = self.get_stage_name(stage_id)
            color = self.get_stage_color(stage_id)
            
            # Draw the color block.
            cv2.rectangle(frame, (start_x, y_offset - 8),
                         (start_x + 15, y_offset + 2), color, -1)
            
            # Draw the text.
            cv2.putText(frame, stage_name, (start_x + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        return frame

def visualize_frame(frame, tracks, id_count, line_count, area_count, line_x = None, area_x1 = None, area_x2 = None, show_legend = False, show_statistics = False):
    """
    Convenience function for visualizing single-frame results.
    
    Args:
        frame: Input frame.
        tracks: Tracking results.
        id_count, line_count, area_count: Counting results.
        line_x: Counting line position.
        area_x1, area_x2: Counting area boundaries.
        show_legend: Whether to show the legend.
        show_statistics: Whether to show statistics.
        
    Returns:
        frame: Visualized frame.
    """
    visualizer = BlueberryVisualizer()
    
    # Draw tracking results.
    if len(tracks) > 0:
        frame = visualizer.draw_tracks(frame, tracks)
    
    # Draw the counting line.
    if line_x is not None:
        frame = visualizer.draw_counting_line(frame, line_x)
    
    # Draw the counting area.
    if area_x1 is not None and area_x2 is not None:
        frame = visualizer.draw_counting_area(frame, area_x1, area_x2)
    
    # Draw statistics.
    if show_statistics:
        frame = visualizer.draw_statistics(frame, id_count, line_count, area_count)
    
    # Draw the legend.
    if show_legend:
        frame = visualizer.create_legend(frame)
    
    return frame

def visualize_bbox(image, results):
    """
    Visualize individual detection boxes.
    
    Args:
        image: Input image.
        results: Detection results (x1,y1,x2,y2,conf,cls_id).
    """
    visualizer = BlueberryVisualizer()
    for result in results:
        color = visualizer.get_stage_color(result[5])
        stage_name = visualizer.get_stage_name(result[5])

        cv2.rectangle(image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), color, 2)
        cv2.putText(image, f'{stage_name} {result[4]:.2f}', (int(result[0]), int(result[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
