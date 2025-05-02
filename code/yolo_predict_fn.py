from PIL import Image
import numpy as np

def preprocess_image_for_yolo(image_path, img_size=(640, 640)):
    """Load and preprocess an image for YOLO."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size, Image.Resampling.LANCZOS)
    return np.array(img)

def predict_fn_yolo(images, model):
    """Predict function compatible with LIME for YOLO models."""
    predictions = []
    
    # Ensure we always have a consistent number of samples
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    
    for img in images:
        results = model(img)
        # Extract confidence scores for predictions
        if results[0].boxes:
            confs = results[0].boxes.conf.cpu().numpy()
        else:
            confs = np.array([0.0])  # Default to 0 if no predictions
        
        # Use the highest confidence score as the prediction
        top_conf = np.max(confs) if confs.size > 0 else 0.0
        predictions.append([top_conf])  # Append as a list to form a 2D array
    
    return np.array(predictions)  # Convert to a consistent 2D array