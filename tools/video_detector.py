"""
Frame-by-frame video detection module.
Uses YOLO and SAHI to detect objects in input videos frame by frame, and stores
the detection results for all frames in JSON format.
Supports sliced detection for large video frames to avoid losing small objects
during resizing.
"""

import cv2
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


class VideoDetector:
    """Video detector class that uses YOLO and SAHI for sliced detection."""
    
    def __init__(
        self,
        yolo_weights_path: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        conf_threshold: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize the video detector.
        
        Args:
            yolo_weights_path: Path to the YOLO model weights file.
            slice_height: Slice height.
            slice_width: Slice width.
            overlap_height_ratio: Overlap ratio in the height direction.
            overlap_width_ratio: Overlap ratio in the width direction.
            conf_threshold: Confidence threshold.
            device: Device type ('cuda' or 'cpu').
        """
        self.yolo_weights_path = yolo_weights_path
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Initialize the YOLO model.
        self.model = YOLO(yolo_weights_path)
        self.model.to(device)
        
        # Initialize the SAHI detection model.
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=yolo_weights_path,
            confidence_threshold=conf_threshold,
            device=device
        )
    
    def detect_frame_with_sahi(self, frame: np.ndarray) -> List[List[float]]:
        """
        Use SAHI to perform sliced detection on a single frame.
        
        Args:
            frame: Input frame (H, W, C).
            
        Returns:
            detections: List of detection results. Each element is
                [x1, y1, x2, y2, conf, cls_id].
        """
        # Convert the numpy array to a PIL image because SAHI requires PIL format.
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Use SAHI for sliced prediction.
        result = get_sliced_prediction(
            pil_image,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0
        )
        
        # Convert the detection result format.
        detections = []
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            conf = prediction.score.value
            cls_id = prediction.category.id
            
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_id)])
        
        return detections
    
    def process_video(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process the whole video, detect frame by frame, and save the results.
        
        Args:
            video_path: Input video path.
            output_path: Output JSON file path.
            
        Returns:
            results: Detection results dictionary.
        """
        # Open the video.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video information.
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps}fps, total frames: {total_frames}")
        
        # Initialize the results dictionary.
        results = {
            "info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": os.path.abspath(video_path),
                "video_info": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames
                }
            },
            "results": {}
        }
        
        frame_idx = 0
        total_detections = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Use SAHI for detection.
            detections = self.detect_frame_with_sahi(frame)
            
            # Add frame results.
            results["results"][frame_idx] = detections
            total_detections += len(detections)
            
            # Print progress.
            # if frame_idx % 100 == 0:
            #     print(f"Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        
        # Save results to a JSON file.
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Total detection boxes: {total_detections}")
        print(f"Detection complete. Results saved to: {output_path}") 
        
        return results
    
    def process_video_batch(self, video_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple videos in batch.
        
        Args:
            video_paths: List of video paths.
            output_dir: Output directory.
            
        Returns:
            all_results: List of detection results for all videos.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\nProcessing video {i+1}/{len(video_paths)}: {video_path}")
            
            # Generate the output file name.
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}.json")
            
            try:
                results = self.process_video(video_path, output_path)
                all_results.append(results)
            except Exception as e:
                print(f"Error while processing video {video_path}: {e}")
                continue
        
        return all_results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Frame-by-frame video detection tool')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--yolo-weights', type=str, required=True, help='Path to the YOLO weights file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--slice-height', type=int, default=640, help='Slice height')
    parser.add_argument('--slice-width', type=int, default=640, help='Slice width')
    parser.add_argument('--overlap-height', type=float, default=0.2, help='Height overlap ratio')
    parser.add_argument('--overlap-width', type=float, default=0.2, help='Width overlap ratio')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device type')
    
    args = parser.parse_args()
    
    # Create the detector.
    detector = VideoDetector(
        yolo_weights_path=args.yolo_weights,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height,
        overlap_width_ratio=args.overlap_width,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Process the video.
    detector.process_video(args.video, args.output)


if __name__ == '__main__':
    main()
    
