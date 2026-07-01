import os
import json
import shutil
from PIL import Image

# filepath: /path/to/yolo_to_coco.py,label:start from 0
def yolo_to_coco(yolo_dir, coco_dir, categories, split="train"):
    """
    Convert YOLO format annotations to COCO format.

    Args:
        yolo_dir (str): Path to the YOLO dataset directory.
        coco_dir (str): Path to the COCO dataset directory.
        categories (list): List of category dictionaries, e.g., [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}].
        split (str): Dataset split, either "train" or "val".
    """
    images = []
    annotations = []
    annotation_id = 1

    # Paths
    images_dir = os.path.join(coco_dir, "images", split)
    annotations_dir = os.path.join(coco_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Iterate over YOLO annotation files
    yolo_images_dir = os.path.join(yolo_dir, "images", split)
    yolo_labels_dir = os.path.join(yolo_dir, "labels", split)

    for image_file in os.listdir(yolo_images_dir):
        if not image_file.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(yolo_images_dir, image_file)
        label_path = os.path.join(yolo_labels_dir, os.path.splitext(image_file)[0] + ".txt")

        # Skip if no corresponding label file
        if not os.path.exists(label_path):
            continue

        # Get image info
        with Image.open(image_path) as img:
            width, height = img.size

        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # Read YOLO annotations
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                # Convert YOLO bbox to COCO bbox
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width *= width
                bbox_height *= height

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

        # Copy image to COCO directory
        shutil.copy(image_path, os.path.join(images_dir, image_file))

    # Save COCO annotations
    coco_annotations = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(os.path.join(annotations_dir, f"instances_{split}.json"), "w") as f:
        json.dump(coco_annotations, f, indent=4)


# Example usage
if __name__ == "__main__":
    yolo_dataset_dir = "/home/wh1234_/data/20251027_yolo_811_640"
    coco_dataset_dir = "/home/wh1234_/data/20251027_coco_811_640_start0"
    category_list = [
        {"id": 0, "name": "Flower"},
        {"id": 1, "name": "Green"},
        {"id": 2, "name": "Light Purple"},
        {"id": 3, "name": "Blue"},
        # Add more categories as needed
    ]

    # Convert train, val, and test splits
    yolo_to_coco(yolo_dataset_dir, coco_dataset_dir, category_list, split="train")
    yolo_to_coco(yolo_dataset_dir, coco_dataset_dir, category_list, split="val")
    yolo_to_coco(yolo_dataset_dir, coco_dataset_dir, category_list, split="test")