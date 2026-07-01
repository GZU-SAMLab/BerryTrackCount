"""
Single frame detection and visualization tool
Detects blueberries using multiple detectors (RT-DETR-l, DINO-4scale, YOLOX-s, YOLO11-n, YOLO11-l)
and visualizes results including ground truth annotations.
RT-DETR and YOLO11 use SAHI sliced inference for better small object detection.
"""

import os
import sys
import cv2
import json
import torch
import logging
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

colors = {0: (245, 222, 179), 1: (0, 255, 0), 2: (255, 0, 255), 3: (255, 0, 0)}
names = {0: 'Flower', 1: 'Green', 2: 'Light Purple', 3: 'Blue'}


class GroundTruthVisualizer:
    """Visualize ground truth annotations from COCO format"""
    
    def __init__(self, annotation_path):
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def visualize(self, image_path, output_path):
        """Visualize ground truth boxes on image"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Find image_id by filename
        img_name = os.path.basename(image_path)
        img_id = None
        for img_info in self.coco_data['images']:
            if img_info['file_name'] == img_name:
                img_id = img_info['id']
                break
        
        if img_id is None or img_id not in self.img_to_anns:
            logger.warning(f"No annotations found for {img_name}")
            cv2.imwrite(output_path, image)
            return
        
        # Draw boxes (no text, only boxes)
        for ann in self.img_to_anns[img_id]:
            bbox = ann['bbox']  # [x, y, w, h]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            category_id = ann['category_id']
            color = colors.get(category_id, (128, 128, 128))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        cv2.imwrite(output_path, image)
        logger.info(f"GT saved: {os.path.basename(output_path)}")


def detect_with_yolo_sahi(weights_path, image_path, output_path):
    """Detect using YOLO-based models (RT-DETR, YOLO11) with SAHI sliced inference"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from PIL import Image as PILImage
    
    # Initialize SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=weights_path,
        confidence_threshold=0.25,
        device='cuda'
    )
    
    # Load image
    image = cv2.imread(image_path)
    pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Perform sliced prediction
    result = get_sliced_prediction(
        pil_image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )
    
    # Draw boxes
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        cls_id = prediction.category.id
        color = colors.get(cls_id, (128, 128, 128))
        name = names.get(cls_id, f'Class_{cls_id}')
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(image, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(output_path, image)
    logger.info(f"Saved: {os.path.basename(output_path)}")


def detect_with_yolox(weights_path, image_path, output_path):
    """Detect using YOLOX model"""
    sys.path.insert(0, '/home/wh1234_/code/Counting/detector/YOLOX')
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from yolox.data.data_augment import ValTransform
    
    # Load model
    exp = get_exp('configs/detector/yolox/yolox_s_exp.py', None)
    exp.num_classes = 4
    model = exp.get_model()
    model.eval()
    
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    
    # Preprocess
    image = cv2.imread(image_path)
    test_size = (640, 640)
    preproc = ValTransform(legacy=False)
    img_tensor, _ = preproc(image, None, test_size)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float().cuda()
    
    # Detect
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(outputs, exp.num_classes, 0.25, exp.nmsthre)
    
    # Draw boxes
    if outputs[0] is not None:
        output = outputs[0].cpu().numpy()
        bboxes = output[:, 0:4]
        img_h, img_w = image.shape[:2]
        scale = min(test_size[0] / img_h, test_size[1] / img_w)
        bboxes /= scale
        cls_ids = output[:, 6].astype(int)
        
        for bbox, cls_id in zip(bboxes, cls_ids):
            x1, y1, x2, y2 = bbox
            color = colors.get(cls_id, (128, 128, 128))
            name = names.get(cls_id, f'Class_{cls_id}')
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(image, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(output_path, image)
    logger.info(f"Saved: {os.path.basename(output_path)}")


def detect_with_dino(weights_path, config_path, image_path, output_path):
    """Detect using DINO model"""
    sys.path.insert(0, '/home/wh1234_/code/Counting/detector/DINO')
    from main import build_model_main
    from util.slconfig import SLConfig
    import datasets.transforms as T
    
    # Load model
    args = SLConfig.fromfile(config_path)
    args.device = 'cuda'
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    # Transform
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and detect
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor, _ = transform(pil_image, None)
    
    with torch.no_grad():
        output = model(image_tensor[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    
    # Draw boxes
    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']
    select_mask = scores > 0.25
    
    num_detections = select_mask.sum().item()
    logger.info(f"Detected {num_detections} objects")
    
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    
    for box, label in zip(boxes[select_mask], labels[select_mask]):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box.cpu().numpy()
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
        
        cls_id = int(label.cpu().numpy())
        color = colors.get(cls_id, (128, 128, 128))
        name = names.get(cls_id, f'Class_{cls_id}')
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(image, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(output_path, image)
    logger.info(f"Saved: {os.path.basename(output_path)}")


def run_detector(detector_name, weights_path, image_files, output_dir, config_path=None):
    """Run detector on all images"""
    logger.info(f"Running {detector_name}...")
    
    for img_path in image_files:
        img_name = img_path.stem
        output_path = output_dir / f"{img_name}_{detector_name}.jpg"
        
        try:
            if detector_name in ('rtdetr-l', 'yolo11n', 'yolo11l'):
                detect_with_yolo_sahi(str(weights_path), str(img_path), str(output_path))
            elif detector_name == 'yolox-s':
                detect_with_yolox(str(weights_path), str(img_path), str(output_path))
            elif detector_name == 'dino':
                detect_with_dino(str(weights_path), str(config_path), str(img_path), str(output_path))
        except Exception as e:
            logger.error(f"Error processing {img_name} with {detector_name}: {e}")
        logger.info(f"Saved: {os.path.basename(output_path)}")


def main():
    # Configuration
    base_dir = Path('/home/wh1234_/code/Counting')
    images_dir = base_dir / 'images'
    output_dir = base_dir / 'output' / 'vis_single_frame_det'
    weights_dir = base_dir / 'weights' / 'detectors'
    annotation_path = '/home/wh1234_/data/Blueberry_coco_data/annotations.json'
    dino_config = '/home/wh1234_/code/Counting/detector/DINO/config/DINO/DINO_4scale_custom.py'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_dir.glob('*.jpg'))
    
    logger.info(f"Processing {len(image_files)} images")
    logger.info("=" * 60)
    
    # Ground truth
    # logger.info("Visualizing ground truth...")
    # gt_viz = GroundTruthVisualizer(annotation_path)
    # for img_path in image_files:
    #     gt_viz.visualize(str(img_path), str(output_dir / f"{img_path.stem}_gt.jpg"))
    
    # Detectors
    logger.info("=" * 60)
    
    run_detector('rtdetr-l', weights_dir / 'rtdetr-l.pt', image_files, output_dir)
    run_detector('yolo11s', weights_dir / 'yolo11s.pt', image_files, output_dir)
    run_detector('berrydet_s', weights_dir / 'berrydet_s.pt', image_files, output_dir)
    run_detector('yolox-s', weights_dir / 'yolox-s.pth', image_files, output_dir)
    # run_detector('dino', weights_dir / 'dino.pth', image_files, output_dir, 
    #              config_path=dino_config)
    
    logger.info("=" * 60)
    logger.info(f"Completed! Results in {output_dir}")


if __name__ == '__main__':
    main()
