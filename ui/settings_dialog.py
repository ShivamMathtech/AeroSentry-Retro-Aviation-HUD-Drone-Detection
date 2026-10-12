from dataclasses import replace
from pathlib import Path
import threading
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QHBoxLayout,QLabel,QLineEdit,
    QComboBox,QDoubleSpinBox,QSpinBox,QCheckBox,QPushButton,QDialogButtonBox,QFileDialog,
    QMessageBox,QPlainTextEdit,QTabWidget,QWidget)
from vision.video_source import LatestBuffer

STYLE = '''QWidget { background: #041210; color: #00dfa1; font-family: Consolas; font-size: 12px; }
QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox,QPlainTextEdit { border: 1px solid #17614f; padding: 5px; background: #071c18; }
QPushButton { border: 1px solid #218269; padding: 6px 12px; }
QPushButton:hover { background: #123c30; } QTabBar::tab { padding: 8px; }
QCheckBox { spacing: 8px; }'''

class SettingsDialog(QDialog):
    def __init__(self,settings,parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle('AeroSentry • optical settings')
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(620)
        self.fields = {}
        self.inspection = LatestBuffer()
        self.inspector = None
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        source = QWidget(); model = QWidget(); display = QWidget()
        forms = [QFormLayout(w) for w in (source,model,display)]
        for page,title in zip((source,model,display),('SOURCE','MODEL / CLASSES','HUD / TRACKING')):
            tabs.addTab(page,title)
        self.add_check(forms[0],'simulation','Simulation (synthetic tracks only)')
        self.add_combo(forms[0],'source_kind','Input',['Camera','Video','RTSP'])
        self.add_text(forms[0],'source','Camera index / video / RTSP')
        browse = QPushButton('Browse video…'); browse.clicked.connect(self.browse_video)
        forms[0].addRow('',browse)
        self.add_spin(forms[0],'processing_fps','Processing FPS cap',1,60)
        self.add_combo(forms[0],'input_resolution','Inference size',[str(n) for n in (320,416,640,960,1280)])
        self.add_combo(forms[0],'device','Device',['cpu','0'])
        forms[0].addRow(QLabel('cpu = CPU only   |   0 = first CUDA GPU\nRTSP credentials are not saved. Enter them each session.'))
        self.add_text(forms[1],'model_path','Trusted .pt weights')
        row = QHBoxLayout()
        browse_model = QPushButton('Browse weights…'); browse_model.clicked.connect(self.browse_model)
        self.inspect_button = QPushButton('Read checkpoint classes'); self.inspect_button.clicked.connect(self.inspect_model)
        row.addWidget(browse_model); row.addWidget(self.inspect_button)
        forms[1].addRow(row)
        self.classes_view = QPlainTextEdit()
        self.classes_view.setReadOnly(True); self.classes_view.setMaximumHeight(165)
        self.classes_view.setPlaceholderText('Read checkpoint classes to see exact IDs and labels.\nBlank class IDs = recognized drone labels only.')
        forms[1].addRow(self.classes_view)
        self.add_text(forms[1],'class_ids','Selected class IDs (e.g. 0,2)')
        self.add_float(forms[1],'confidence','Confidence',.01,1,.05)
        self.add_float(forms[1],'iou','NMS IoU threshold',.01,1,.05)
        label = QLabel('A matching class name validates metadata only. Detection quality\ndepends on training data and your footage. Generic COCO weights\ndo not provide a drone class. Use only trusted checkpoint files.')
        forms[1].addRow(label)
        self.add_float(forms[2],'brightness','Phosphor brightness',.01,1,.05)
        self.add_check(forms[2],'glow','Phosphor glow')
        self.add_check(forms[2],'crt','CRT grain / scan lines')
        self.add_check(forms[2],'optical','LIVE OPTICAL (off = HUD ONLY)')
        self.add_check(forms[2],'green_tint','Monochrome green optical tint')
        self.add_float(forms[2],'smoothing','EMA alpha (1 = no smoothing)',.01,1,.05)
        self.add_float(forms[2],'lost_seconds','Track history retention (seconds)',.1,30,.5)
        self.add_combo(forms[2],'selection','Target selection',['Nearest center','Highest confidence','Manual'])
        self.add_float(forms[2],'reference_course','Configured CRS reference (not measured)',0,359.9,1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll_inspection); self.timer.start(100)

    def add_text(self,form,name,title):
        w = QLineEdit(str(getattr(self.settings,name))); self.fields[name]=w; form.addRow(title,w)
    def add_combo(self,form,name,title,choices):
        w=QComboBox(); w.addItems(choices); w.setCurrentText(str(getattr(self.settings,name))); self.fields[name]=w; form.addRow(title,w)
    def add_check(self,form,name,title):
        w=QCheckBox(title); w.setChecked(getattr(self.settings,name)); self.fields[name]=w; form.addRow(w)
    def add_spin(self,form,name,title,low,high):
        w=QSpinBox(); w.setRange(low,high); w.setValue(getattr(self.settings,name)); self.fields[name]=w; form.addRow(title,w)
    def add_float(self,form,name,title,low,high,step):
        w=QDoubleSpinBox(); w.setRange(low,high); w.setSingleStep(step); w.setValue(getattr(self.settings,name)); self.fields[name]=w; form.addRow(title,w)
    def browse_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Open video','','Video (*.mp4 *.avi *.mkv *.mov *.webm);;All files (*)')
        if path:
            self.fields['source'].setText(path); self.fields['source_kind'].setCurrentText('Video'); self.fields['simulation'].setChecked(False)
    def browse_model(self):
        path,_=QFileDialog.getOpenFileName(self,'Select trusted YOLO checkpoint','','YOLO weights (*.pt)')
        if path:
            self.fields['model_path'].setText(path)
    def inspect_model(self):
        if self.inspector and self.inspector.is_alive(): return
        path=self.fields['model_path'].text().strip()
        if not Path(path).is_file():
            self.classes_view.setPlainText('Select an existing local .pt checkpoint first.'); return
        self.inspect_button.setEnabled(False)
        self.classes_view.setPlainText('Loading checkpoint metadata…')
        mailbox = self.inspection
        def inspect():
            try:
                from ultralytics import YOLO
                model = YOLO(path)
                names=model.names
                mailbox.put(('ok','\n'.join(f'{i}: {label}' for i,label in names.items())))
            except Exception as exc:
                mailbox.put(('error',str(exc)))
        self.inspector=threading.Thread(target=inspect,name='AeroSentry-model-inspect',daemon=True); self.inspector.start()
    def poll_inspection(self):
        value=self.inspection.get()
        if value:
            self.inspection.clear(); self.classes_view.setPlainText(value[1]); self.inspect_button.setEnabled(True)
    def value(self):
        data={}
        for name,w in self.fields.items():
            if isinstance(w,QCheckBox): data[name]=w.isChecked()
            elif isinstance(w,QComboBox): data[name]=w.currentText()
            elif isinstance(w,(QSpinBox,QDoubleSpinBox)): data[name]=w.value()
            else: data[name]=w.text().strip()
        data['input_resolution']=int(data['input_resolution'])
        return replace(self.settings,**data).validate()
    def accept(self):
        try:
            self.value()
        except ValueError as exc:
            QMessageBox.warning(self,'Invalid settings',str(exc)); return
        super().accept()
