"""
Video frame slicing module.
Processes large video frames, such as 2160x3840 frames, by splitting them into
small 640x640 tiles suitable for YOLO detection. This avoids losing small
objects during resizing and improves detection accuracy.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict


class FrameProcessor:
    def __init__(self, slice_size = 640, overlap = 50):
        """
        Initialize the frame processor.
        
        Args:
            slice_size: Tile size. Defaults to 640x640 to match the YOLO training size.
            overlap: Overlap in pixels between tiles to avoid splitting boundary objects.
        """
        self.slice_size = slice_size
        self.overlap = overlap
    
    def slice_frame(self, frame):
        """
        Split a large frame into multiple small tiles.
        
        Args:
            frame: Input frame (H, W, C).
            
        Returns:
            slices: List of tiles. Each element is a 640x640 image.
            slice_info: Tile metadata list containing each tile's position in the original image.
        """
        h, w = frame.shape[:2]
        slices = []
        slice_info = []
        
        # Calculate the number of tile rows and columns.
        step = self.slice_size - self.overlap
        rows = (h - self.overlap) // step + (1 if (h - self.overlap) % step > 0 else 0)
        cols = (w - self.overlap) // step + (1 if (w - self.overlap) % step > 0 else 0)
        
        for row in range(rows):
            for col in range(cols):
                # Calculate the tile position in the original image.
                y1 = row * step
                x1 = col * step
                y2 = min(y1 + self.slice_size, h)
                x2 = min(x1 + self.slice_size, w)
                
                # Ensure the tile size is slice_size x slice_size.
                y1 = max(0, y2 - self.slice_size)
                x1 = max(0, x2 - self.slice_size)
                
                # Extract the tile.
                slice_img = frame[y1:y2, x1:x2]
                
                # Pad the tile if it is smaller than the target size.
                if slice_img.shape[:2] != (self.slice_size, self.slice_size):
                    slice_img = self._pad_slice(slice_img, self.slice_size)
                
                slices.append(slice_img)
                slice_info.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'row': row, 'col': col
                })
        
        return slices, slice_info
    
    def _pad_slice(self, slice_img, target_size):
        """
        Pad a tile that is smaller than the target size.
        
        Args:
            slice_img: Input tile.
            target_size: Target size.
            
        Returns:
            padded_slice: Tile after padding.
        """
        h, w = slice_img.shape[:2]
        pad_h = target_size - h
        pad_w = target_size - w
        
        # Use reflection padding.
        padded = cv2.copyMakeBorder(
            slice_img, 0, pad_h, 0, pad_w, 
            cv2.BORDER_REFLECT_101
        )
        return padded
    
    def merge_detections(self, slice_detections, slice_info, original_shape, nms_threshold = 0.5):
        """
        Merge tile detections into the original image coordinate system and apply NMS.
        
        Args:
            slice_detections: List of detection results for each tile.
                Each element is (N, 6) [x1,y1,x2,y2,conf,cls_id].
            slice_info: Tile metadata list.
            original_shape: Original image size (height, width).
            nms_threshold: NMS threshold.
            
        Returns:
            merged_dets: Merged detection results (M, 6).
        """
        all_detections = []
        
        for i, (dets, info) in enumerate(zip(slice_detections, slice_info)):
            if len(dets) == 0:
                continue
                
            # Convert tile detections to the original image coordinate system.
            dets_original = dets.copy()
            dets_original[:, 0] += info['x1']  # x1
            dets_original[:, 1] += info['y1']  # y1
            dets_original[:, 2] += info['x1']  # x2
            dets_original[:, 3] += info['y1']  # y2
            
            # Clip boxes to the original image boundary.
            dets_original[:, 0] = np.clip(dets_original[:, 0], 0, original_shape[1])
            dets_original[:, 1] = np.clip(dets_original[:, 1], 0, original_shape[0])
            dets_original[:, 2] = np.clip(dets_original[:, 2], 0, original_shape[1])
            dets_original[:, 3] = np.clip(dets_original[:, 3], 0, original_shape[0])
            
            all_detections.append(dets_original)
        
        if len(all_detections) == 0:
            return np.empty((0, 6))
        
        # Merge all detection results.
        merged_dets = np.vstack(all_detections)
        
        # Apply NMS separately for each class.
        final_dets = []
        for cls_id in np.unique(merged_dets[:, 5]):
            cls_mask = merged_dets[:, 5] == cls_id
            cls_dets = merged_dets[cls_mask]
            
            # NMS
            nms_dets = self._apply_nms(cls_dets, nms_threshold)
            if len(nms_dets) > 0:
                final_dets.append(nms_dets)
        
        if len(final_dets) == 0:
            return np.empty((0, 6))
        
        return np.vstack(final_dets)
    
    def _apply_nms(self, dets, threshold):
        """
        Apply NMS to detection results for a single class.
        
        Args:
            dets: Detection results (N, 6) [x1,y1,x2,y2,conf,cls_id].
            threshold: NMS threshold.
            
        Returns:
            nms_dets: Detection results after NMS.
        """
        if len(dets) == 0:
            return dets
        
        # Calculate areas.
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence.
        scores = dets[:, 4]  # conf
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU.
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than or equal to the threshold.
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return dets[keep]


def process_frame_with_slicing(frame, model, conf_thres = 0.25, slice_size = 640, overlap = 50, nms_threshold = 0.5):
    """
    Convenience function for slice-based detection on large frames.
    
    Args:
        frame: Input frame.
        model: YOLO model.
        conf_thres: Confidence threshold.
        slice_size: Tile size.
        overlap: Overlap in pixels.
        nms_threshold: NMS threshold.
        
    Returns:
        detections: Detection results (N, 6) [x1,y1,x2,y2,conf,cls_id].
    """
    processor = FrameProcessor(slice_size, overlap)
    
    # Slice the frame.
    slices, slice_info = processor.slice_frame(frame)
    
    # Run detection on each tile.
    slice_detections = []
    for slice_img in slices:
        results = model.predict(slice_img, conf=conf_thres, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None or len(boxes) == 0:
            slice_detections.append(np.empty((0, 6)))
            continue
        
        # Convert to the required format.
        dets = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            dets.append([x1, y1, x2, y2, conf, cls_id])
        
        if len(dets) > 0:
            slice_detections.append(np.array(dets))
        else:
            slice_detections.append(np.empty((0, 6)))
    
    # Merge detection results.
    merged_detections = processor.merge_detections(
        slice_detections, slice_info, frame.shape[:2], nms_threshold
    )
    
    return merged_detections
