
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

def preprocess_image_for_faster_rcnn(image_path):
    """Load and preprocess an image for Faster R-CNN."""
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
    ])
    img_tensor = transform(img)
    # No need to add batch dimension - the model does this internally
    return img_tensor

def predict_fn_faster_rcnn(images, model, selected_classes=None, confidence_threshold=0.5):
    
    all_predictions = []
    
    for img in images:
        with torch.no_grad():
            pred = model([img])
            
        # Extract prediction components
        boxes = pred[0]['boxes'].cpu().numpy()
        labels = pred[0]['labels'].cpu().numpy()
        scores = pred[0]['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            # No detections above threshold
            if selected_classes is not None:
                all_predictions.append(np.zeros(len(selected_classes)))
            else:
                all_predictions.append(np.array([]))
            continue
            
        if selected_classes is not None:
            # Initialize predictions array for selected classes
            class_predictions = np.zeros(len(selected_classes))
            
            # Map each detection to the correct class index in our output
            for i, class_idx in enumerate(selected_classes):
                # Find detections for this class
                class_mask = (labels == class_idx)
                if class_mask.any():
                    # Use the highest confidence score for this class
                    class_predictions[i] = np.max(scores[class_mask])
            
            all_predictions.append(class_predictions)
        else:
            # For each detection, create a sparse prediction vector with
            # confidence score at the position of the detected class
            num_classes = model.roi_heads.box_predictor.cls_score.out_features
            # Subtract 1 because class 0 is background in most models
            predictions = np.zeros(num_classes - 1)
            
            # Place each detection's confidence score at the appropriate class index
            for label, score in zip(labels, scores):
                # Adjust for background class (index 0)
                class_idx = label - 1  # Subtract 1 since class 0 is background
                if 0 <= class_idx < len(predictions):
                    # Update with max confidence if we have multiple detections for same class
                    predictions[class_idx] = max(predictions[class_idx], score)
            
            all_predictions.append(predictions)
    
    return np.array(all_predictions)