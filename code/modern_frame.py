from PyQt5.QtWidgets import QFrame
from style import COLORS

class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modernFrame")
        self.setStyleSheet(f""" 
            QFrame#modernFrame {{
                background-color: white;
                border-radius: 8px;
                border: 1px solid {COLORS['accent']};
            }}
        """)
        self.setContentsMargins(15, 15, 15, 15)
