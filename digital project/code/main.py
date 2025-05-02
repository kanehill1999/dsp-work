
#not this one
import sys
from PyQt5.QtWidgets import QApplication
from lime_app import LimeApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        lime_app = LimeApp()
        lime_app.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)