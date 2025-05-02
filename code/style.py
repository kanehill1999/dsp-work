COLORS = {
    'primary': '#2D3250',    # Dark blue-gray
    'secondary': '#424769',  # Medium blue-gray
    'accent': '#7077A1',     # Light blue-gray
    'background': '#F6F6F6', # Light gray
    'text': '#2D3250',       # Dark blue-gray
    'button': '#7077A1',     # Light blue-gray
    'button_hover': '#424769' # Medium blue-gray
}

class StyleSheet:
    @staticmethod
    def get_stylesheet():
        return f"""
        QMainWindow {{
            background-color: {COLORS['background']};
        }}
        
        QLabel {{
            color: {COLORS['text']};
            font-size: 12px;
            font-weight: bold;
            padding: 5px;
        }}
        
        QPushButton {{
            background-color: {COLORS['button']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 100px;
        }}
        
        QPushButton:hover {{
            background-color: {COLORS['button_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {COLORS['primary']};
        }}
        
        QComboBox {{
            background-color: white;
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            padding: 5px;
            min-width: 150px;
        }}
        
        QGroupBox {{
            background-color: white;
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            color: {COLORS['text']};
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }}
        
        QCheckBox {{
            color: {COLORS['text']};
            spacing: 5px;
            padding: 2px;
        }}
        
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
        }}
        
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        
        QFrame {{
            background-color: white;
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
        }}
        
        QProgressBar {{
            background-color: white;
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background-color: {COLORS['button']};
            border-radius: 4px;
        }}
        """