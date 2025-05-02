# model_config.py
import os
import json
from typing import Dict, List, Optional, Any


class ModelConfig:
    """Class to handle model configuration from JSON files."""
    
    def __init__(self, config_dir: str = "configs"):

        self.config_dir = config_dir
        self.models_config = {}
        self.classes_config = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Load configurations
        self.load_models_config()
        self.load_classes_config()
        
        # Create default configurations if none exist
        if not self.models_config.get("models", []) or not self.classes_config:
            self.create_default_configs()
    
    def load_models_config(self, file_path: Optional[str] = None) -> None:
        
        if file_path is None:
            file_path = os.path.join(self.config_dir, "models.json")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    self.models_config = config
            else:
                print(f"Models config file not found: {file_path}")
                # Create empty models config
                self.models_config = {"models": []}
        except Exception as e:
            print(f"Error loading models config: {str(e)}")
            self.models_config = {"models": []}
    
    def load_classes_config(self, file_path: Optional[str] = None) -> None:
        
        if file_path is None:
            file_path = os.path.join(self.config_dir, "classes.json")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    self.classes_config = config
            else:
                print(f"Classes config file not found: {file_path}")
                # Create empty classes config
                self.classes_config = {}
        except Exception as e:
            print(f"Error loading classes config: {str(e)}")
            self.classes_config = {}
    
    def save_models_config(self, file_path: Optional[str] = None) -> None:
        
        if file_path is None:
            file_path = os.path.join(self.config_dir, "models.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.models_config, f, indent=2)
        except Exception as e:
            print(f"Error saving models config: {str(e)}")
    
    def save_classes_config(self, file_path: Optional[str] = None) -> None:
        
        if file_path is None:
            file_path = os.path.join(self.config_dir, "classes.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.classes_config, f, indent=2)
        except Exception as e:
            print(f"Error saving classes config: {str(e)}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        
        return self.models_config.get("models", [])
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        
        for model in self.get_models():
            if model.get("id") == model_id:
                return model
        return None
    
    def get_model_classes(self, model_id: str) -> Dict[str, str]:
        
        model = self.get_model_by_id(model_id)
        if not model:
            print(f"Model not found with ID: {model_id}")
            return {}
        
        # Get class set ID from model
        class_set_id = model.get("output_classes")
        if not class_set_id or class_set_id not in self.classes_config:
            print(f"Class set not found: {class_set_id} for model {model_id}")
            return {}
        
        # Return classes dictionary
        class_set = self.classes_config.get(class_set_id, {})
        return class_set.get("classes", {})
    
    def add_model(self, model_config: Dict[str, Any]) -> bool:
       
        if "id" not in model_config:
            print("Model configuration must include an 'id' field")
            return False
        
        # Check if model with this ID already exists
        for i, model in enumerate(self.get_models()):
            if model.get("id") == model_config["id"]:
                # Update existing model
                self.models_config["models"][i] = model_config
                self.save_models_config()
                return True
        
        # Add new model
        self.models_config.setdefault("models", []).append(model_config)
        self.save_models_config()
        return True
    
    def remove_model(self, model_id: str) -> bool:
        
        initial_count = len(self.get_models())
        self.models_config["models"] = [
            model for model in self.get_models() 
            if model.get("id") != model_id
        ]
        
        if len(self.get_models()) < initial_count:
            self.save_models_config()
            return True
        return False
    
    def add_class_set(self, class_set_id: str, class_set_config: Dict[str, Any]) -> bool:
       
        if "classes" not in class_set_config:
            print("Class set configuration must include a 'classes' field")
            return False
        
        self.classes_config[class_set_id] = class_set_config
        self.save_classes_config()
        return True
    
    def remove_class_set(self, class_set_id: str) -> bool:
       
        if class_set_id in self.classes_config:
            del self.classes_config[class_set_id]
            self.save_classes_config()
            return True
        return False
    
    def create_default_configs(self) -> None:
        """Create default configuration with COCO classes if none exists."""
        # Add default COCO classes set if no classes exist
        if not self.classes_config:
            from constant import CLASS_NAMES
            coco_classes = {
                "name": "COCO Classes",
                "description": "Default COCO dataset classes",
                "classes": {str(k): v for k, v in CLASS_NAMES.items()}
            }
            
            self.classes_config["coco_classes"] = coco_classes
            self.save_classes_config()
            print("Created default COCO classes configuration")
        
        # Add default models if none exist
        if not self.get_models():
            default_models = [
                {
                    "id": "yolov8m",
                    "name": "YOLOv8 Medium",
                    "description": "Default YOLOv8 medium model",
                    "architecture_type": "yolo",
                    "model_path": "./models",
                    "weight_file": "yolov8m.pt",
                    "output_classes": "coco_classes"
                },
                {
                    "id": "yolov8n",
                    "name": "YOLOv8 Nano",
                    "description": "Default YOLOv8 nano model",
                    "architecture_type": "yolo",
                    "model_path": "./models",
                    "weight_file": "yolov8n.pt",
                    "output_classes": "coco_classes"
                }
            ]
            
            self.models_config["models"] = default_models
            self.save_models_config()
            print("Created default model configurations")