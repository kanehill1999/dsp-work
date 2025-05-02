# model_dialogs.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QComboBox, QPushButton, 
    QGroupBox, QScrollArea, QFrame, QMessageBox,
    QFileDialog, QInputDialog, QWidget
)
from PyQt5.QtCore import Qt


class ModelConfigDialog(QDialog):
    """Dialog for adding/editing model configuration."""
    
    def __init__(self, model_handler, edit_model=None, parent=None):
        super().__init__(parent)
        self.model_handler = model_handler
        self.edit_model = edit_model
        
        self.setWindowTitle("Model Configuration")
        self.setMinimumSize(500, 400)
        
        self.initUI()
        
        if edit_model:
            self.loadModelData(edit_model)
    
    def initUI(self):
        layout = QVBoxLayout()
        
        form_layout = QGridLayout()
        
        # Model ID
        form_layout.addWidget(QLabel("Model ID:"), 0, 0)
        self.id_input = QLineEdit()
        form_layout.addWidget(self.id_input, 0, 1)
        
        # Model Name
        form_layout.addWidget(QLabel("Model Name:"), 1, 0)
        self.name_input = QLineEdit()
        form_layout.addWidget(self.name_input, 1, 1)
        
        # Description
        form_layout.addWidget(QLabel("Description:"), 2, 0)
        self.description_input = QLineEdit()
        form_layout.addWidget(self.description_input, 2, 1)
        
        # Architecture Type
        form_layout.addWidget(QLabel("Architecture Type:"), 3, 0)
        self.architecture_input = QComboBox()
        self.architecture_input.addItems(["yolo", "faster_rcnn"])
        form_layout.addWidget(self.architecture_input, 3, 1)
        
        # Model Path
        form_layout.addWidget(QLabel("Model Path:"), 4, 0)
        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        path_layout.addWidget(self.model_path_input)
        browse_path_btn = QPushButton("Browse")
        browse_path_btn.clicked.connect(self.browsePath)
        path_layout.addWidget(browse_path_btn)
        form_layout.addLayout(path_layout, 4, 1)
        
        # Weight File
        form_layout.addWidget(QLabel("Weight File:"), 5, 0)
        weight_layout = QHBoxLayout()
        self.weight_file_input = QLineEdit()
        weight_layout.addWidget(self.weight_file_input)
        browse_weight_btn = QPushButton("Browse")
        browse_weight_btn.clicked.connect(self.browseWeightFile)
        weight_layout.addWidget(browse_weight_btn)
        form_layout.addLayout(weight_layout, 5, 1)
        
        # Output Classes
        form_layout.addWidget(QLabel("Output Classes:"), 6, 0)
        self.output_classes_input = QComboBox()
        class_sets = list(self.model_handler.config_manager.classes_config.keys())
        self.output_classes_input.addItems(class_sets)
        form_layout.addWidget(self.output_classes_input, 6, 1)
        
        layout.addLayout(form_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.saveModel)
        btn_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def loadModelData(self, model):
        """Load model data into form fields."""
        self.id_input.setText(model.get("id", ""))
        self.name_input.setText(model.get("name", ""))
        self.description_input.setText(model.get("description", ""))
        
        arch_index = self.architecture_input.findText(model.get("architecture_type", "yolo"))
        if arch_index >= 0:
            self.architecture_input.setCurrentIndex(arch_index)
        
        self.model_path_input.setText(model.get("model_path", ""))
        self.weight_file_input.setText(model.get("weight_file", ""))
        
        classes_index = self.output_classes_input.findText(model.get("output_classes", ""))
        if classes_index >= 0:
            self.output_classes_input.setCurrentIndex(classes_index)
    
    def browsePath(self):
        """Browse for model path directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if directory:
            self.model_path_input.setText(directory)
    
    def browseWeightFile(self):
        """Browse for weight file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Weight File", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.weight_file_input.setText(file_path)
    
    def saveModel(self):
        """Save model configuration."""
        model_id = self.id_input.text().strip()
        model_name = self.name_input.text().strip()
        description = self.description_input.text().strip()
        architecture_type = self.architecture_input.currentText()
        model_path = self.model_path_input.text().strip()
        weight_file = self.weight_file_input.text().strip()
        output_classes = self.output_classes_input.currentText()
        
        if not model_id or not model_name or not model_path or not weight_file or not output_classes:
            QMessageBox.warning(self, "Error", "All fields are required!")
            return
        
        model_config = {
            "id": model_id,
            "name": model_name,
            "description": description,
            "architecture_type": architecture_type,
            "model_path": model_path,
            "weight_file": weight_file,
            "output_classes": output_classes
        }
        
        success = self.model_handler.config_manager.add_model(model_config)
        
        if success:
            QMessageBox.information(self, "Success", f"Model '{model_name}' has been saved.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to save model configuration.")


class ClassSetConfigDialog(QDialog):
    """Dialog for adding/editing class set configuration."""
    
    def __init__(self, model_handler, edit_class_set_id=None, parent=None):
        super().__init__(parent)
        self.model_handler = model_handler
        self.edit_class_set_id = edit_class_set_id
        
        self.setWindowTitle("Class Set Configuration")
        self.setMinimumSize(500, 400)
        
        self.initUI()
        
        if edit_class_set_id:
            self.loadClassSetData(edit_class_set_id)
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # Class Set Info
        form_layout = QGridLayout()
        
        form_layout.addWidget(QLabel("Class Set ID:"), 0, 0)
        self.id_input = QLineEdit()
        form_layout.addWidget(self.id_input, 0, 1)
        
        form_layout.addWidget(QLabel("Name:"), 1, 0)
        self.name_input = QLineEdit()
        form_layout.addWidget(self.name_input, 1, 1)
        
        form_layout.addWidget(QLabel("Description:"), 2, 0)
        self.description_input = QLineEdit()
        form_layout.addWidget(self.description_input, 2, 1)
        
        layout.addLayout(form_layout)
        
        # Classes Table
        classes_group = QGroupBox("Classes")
        classes_layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.classes_container = QWidget()
        self.classes_layout = QVBoxLayout()
        self.classes_container.setLayout(self.classes_layout)
        
        self.class_rows = []
        for i in range(10):
            self.addClassRow()
        
        scroll.setWidget(self.classes_container)
        classes_layout.addWidget(scroll)
        
        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self.addClassRow)
        classes_layout.addWidget(add_class_btn)
        
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.saveClassSet)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def addClassRow(self):
        """Add a new class row to the form."""
        row_layout = QHBoxLayout()
        
        index_input = QLineEdit()
        index_input.setPlaceholderText("Class Index")
        index_input.setFixedWidth(100)
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("Class Name")
        
        row_layout.addWidget(index_input)
        row_layout.addWidget(name_input)
        
        self.class_rows.append((index_input, name_input))
        self.classes_layout.addLayout(row_layout)
    
    def loadClassSetData(self, class_set_id):
        """Load class set data into form fields."""
        class_set = self.model_handler.config_manager.classes_config.get(class_set_id, {})
        
        self.id_input.setText(class_set_id)
        self.name_input.setText(class_set.get("name", ""))
        self.description_input.setText(class_set.get("description", ""))
        
        for index_input, name_input in self.class_rows:
            index_input.setText("")
            name_input.setText("")
        
        classes = class_set.get("classes", {})
        for i, (class_idx, class_name) in enumerate(classes.items()):
            if i < len(self.class_rows):
                self.class_rows[i][0].setText(class_idx)
                self.class_rows[i][1].setText(class_name)
            else:
                self.addClassRow()
                self.class_rows[-1][0].setText(class_idx)
                self.class_rows[-1][1].setText(class_name)
    
    def saveClassSet(self):
        """Save class set configuration."""
        class_set_id = self.id_input.text().strip()
        name = self.name_input.text().strip()
        description = self.description_input.text().strip()
        
        if not class_set_id or not name:
            QMessageBox.warning(self, "Error", "Class Set ID and Name are required!")
            return
        
        classes = {}
        for index_input, name_input in self.class_rows:
            class_idx = index_input.text().strip()
            class_name = name_input.text().strip()
            
            if class_idx and class_name:
                classes[class_idx] = class_name
        
        if not classes:
            QMessageBox.warning(self, "Error", "At least one class is required!")
            return
        
        class_set_config = {
            "name": name,
            "description": description,
            "classes": classes
        }
        
        success = self.model_handler.config_manager.add_class_set(class_set_id, class_set_config)
        
        if success:
            QMessageBox.information(self, "Success", f"Class set '{name}' has been saved.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to save class set configuration.")