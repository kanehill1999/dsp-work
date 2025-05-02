# train.py
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from code.constant import CLASS_NAMES  # Import class names from your constants file

# Configuration
DATASET_PATH = Path("Data/VisDrone")  # Adjust the path if needed
NUM_CLASSES = len(CLASS_NAMES) + 1  # Add 1 for background
BATCH_SIZE = 8  # Adjust based on your GPU's memory
NUM_EPOCHS = 50  # Number of epochs for training
MODEL_SAVE_PATH = Path("models/faster_rcnn_visdrone.pt")  # Save path for the model

# Function to Convert Annotations to a Format Suitable for Faster R-CNN
def visdrone_to_faster_rcnn(images_dir, annotations_dir, class_names):
    """
    Converts VisDrone annotations to a list of dictionaries, where each dictionary
    represents an image and its annotations in the format expected by Faster R-CNN.
    """
    all_data = []
    pbar = tqdm(list(annotations_dir.glob("*.txt")), desc=f"Converting annotations in {annotations_dir.name}")
    for annotation_file in pbar:
        img_file = images_dir / annotation_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        with open(annotation_file, "r") as file:
            boxes = []
            labels = []
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                xmin, ymin, w, h, original_class_id = map(int, row[:5])  # Get the original class ID
                xmax, ymax = xmin + w, ymin + h

                # Skip invalid boxes or ignored regions (original_class_id == 0)
                if xmax <= xmin or ymax <= ymin or original_class_id == 0:
                    continue

                # Map the original class ID to your desired index
                if original_class_id in class_names.values():
                    label_index = list(class_names.keys())[list(class_names.values()).index(original_class_id)]
                    labels.append(label_index + 1) # Shift to 1-based index
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    print(f"Warning: Skipping object with unknown class ID: {original_class_id} in {annotation_file}")

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            data = {
                "image_path": str(img_file),
                "boxes": boxes,
                "labels": labels,
            }
            all_data.append(data)

    return all_data

# Custom Dataset Class
class VisDroneDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        boxes = self.data[idx]["boxes"]
        labels = self.data[idx]["labels"]

        # Apply transformations if provided
        if self.transforms is not None:
            img = self.transforms(img)

        target = {
            "boxes": boxes,
            "labels": labels,
        }
        return img, target

# Prepare Dataset
def prepare_dataset():
    print("Preparing dataset...")
    subsets = [
        ("Train", "images", "annotations"),
        ("Validation", "images", "annotations"),
    ]
    datasets = {}
    for subset, img_dir, ann_dir in subsets:
        datasets[subset.lower()] = visdrone_to_faster_rcnn(
            images_dir=DATASET_PATH / subset / img_dir,
            annotations_dir=DATASET_PATH / subset / ann_dir,
            class_names=CLASS_NAMES  # Pass the CLASS_NAMES dictionary
        )
    return datasets

# Create Faster R-CNN Model
def create_model(num_classes):
    print("Creating Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Train Faster R-CNN Model
def train_model(model, dataloader, optimizer, device, scaler):
    print("Training Faster R-CNN model...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved at {MODEL_SAVE_PATH}")

# Main Function
if __name__ == "__main__":
    # Prepare dataset
    datasets = prepare_dataset()

    # Data Augmentation
    train_dataset = VisDroneDataset(
        datasets["train"],
        transforms=T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5),
        ])
    )
    val_dataset = VisDroneDataset(datasets["validation"], transforms=T.ToTensor())

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(NUM_CLASSES).to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Train model
    train_model(model, train_dataloader, optimizer, device, scaler)

    print("\nTraining complete!")