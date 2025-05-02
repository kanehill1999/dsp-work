# train_visdrone.py

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
DATASET_PATH = Path("Data/VisDrone")
YAML_FILE = DATASET_PATH / "VisDrone.yaml"
MODEL_WEIGHTS = "yolov8m.pt"  
SAVE_DIR = "runs/train/visdrone_model4"  

# Function to Convert Annotations to YOLO Format
def visdrone_to_yolo(images_dir, annotations_dir, labels_dir):
    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    labels_dir.mkdir(parents=True, exist_ok=True)  
    pbar = tqdm(list(annotations_dir.glob("*.txt")), desc=f"Converting annotations in {annotations_dir.name}")

    for annotation_file in pbar:
        img_file = images_dir / annotation_file.with_suffix(".jpg").name
        img_size = Image.open(img_file).size 

        with open(annotation_file, "r") as file:
            lines = []
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] == "0":  # Skip ignored regions (class 0)
                    continue
                cls = int(row[5]) - 1  # Class index adjustment
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
            
            # Save to YOLO format
            label_file = labels_dir / annotation_file.name
            with open(label_file, "w") as label_output:
                label_output.writelines(lines)

# Prepare Dataset Directories
def prepare_dataset():
    print("Preparing dataset...")
    subsets = [
        ("Train", "images", "annotations"),
        ("Validation", "images", "annotations"),
        ("Test", "images", "annotations"),
    ]

    for subset, img_dir, ann_dir in subsets:
        visdrone_to_yolo(
            images_dir=DATASET_PATH / subset / img_dir,
            annotations_dir=DATASET_PATH / subset / ann_dir,
            labels_dir=DATASET_PATH / subset / "labels",
        )

# Create YAML File for YOLO Training
def create_yaml():
    print("Creating YAML configuration...")
    yaml_content = f"""
# VisDrone.yaml
path: C:/Users/Kaneh/OneDrive/Documents/Year 3/new/Data/Visdrone
train: Train/images
val: Validation/images
test: Test/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

"""
    YAML_FILE.write_text(yaml_content.strip())

# Train YOLO Model
def train_yolo():
    print("Training YOLO model...")
    model = YOLO(MODEL_WEIGHTS)  
    model.train(data=str(YAML_FILE), epochs=100, imgsz=640, project="runs/train", name="visdrone_model")

# Evaluate YOLO Model
def evaluate_yolo():
    print("Evaluating YOLO model...")
    model = YOLO(f"{SAVE_DIR}/weights/best.pt")  
    results = model.val(data=str(YAML_FILE))  
    print(results)

# Main Function
if __name__ == "__main__":
    
    prepare_dataset()
    create_yaml()

    
    train_yolo()

    print(f"\nTraining complete! The best model is saved at: {SAVE_DIR}/weights/best.pt")
    evaluate_yolo()
