from PyQt5.QtWidgets import QProgressDialog

class ProgressDialog(QProgressDialog):
    def __init__(self, label_text, maximum, parent=None):
        super().__init__(label_text, "Cancel", 0, maximum, parent)
        self.setWindowTitle("Progress")
        self.setValue(0)
        self.setCancelButton(None)
        self.setAutoClose(True)
        self.setMinimumDuration(0)

    def update_progress(self, value, maximum=None):
        if maximum is not None:
            self.setMaximum(maximum)
        self.setValue(value)
