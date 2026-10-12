from datetime import datetime
import time
import cv2
import numpy as np
from PySide6.QtCore import QTimer,Qt
from PySide6.QtGui import QImage,QAction,QKeySequence
from PySide6.QtWidgets import (QApplication,QMainWindow,QDialog,QVBoxLayout,QPlainTextEdit,
    QMessageBox,QMenu,QLabel)
from config import DATA_DIR
from ui.hud_widget import HUDWidget
from ui.settings_dialog import SettingsDialog,STYLE
from vision.telemetry import HUDState
from vision.tracker import TrackStore
from vision.target_manager import TargetManager
from vision.frame_processor import Pipeline
from services.event_logger import EventLogger
from services.screenshot_service import ScreenshotService
from services.recorder import Recorder

HELP = '''AEROSENTRY • VISUAL TRACKING ONLY

F2 / right click  Settings and input / model configuration
Space            Pause / resume capture and processing
L                Toggle selected visual track lock
N                Cycle visible tracked objects
R                Reset ByteTrack and visual history
V                Select / reopen video source (settings)
H                Toggle HUD overlay (source labels remain)
O                Toggle LIVE OPTICAL / HUD ONLY
C                Save screenshot of visible HUD
F9               Start / stop annotated HUD recording
F6               View recent SQLite event history
F11              Full screen
Esc              Release selected track / close a modal
F1               This help

SIMULATION uses generated positions, never a real detector.
LIVE OPTICAL always pairs detections with their exact source frame.
Green brackets = raw detector boxes. Circular EMA = smoothed reticle.
MOV = image pixels/second, not physical speed. CRS = configured reference.
DME, range, altitude and physical speed are N/A.

First live run: F2 → disable Simulation → select input → select a trusted
local drone-trained .pt checkpoint → Read checkpoint classes → choose
class IDs or leave blank for recognized drone labels → OK.

A visual lock holds one ByteTrack ID. Reacquisition needs the same ID;
N selects a new identity after a long loss. This is not object re-identification.

Recordings use a 10 FPS presentation timeline. If frames are dropped under
load, duration may differ from wall-clock duration. Simulation labels are
burned into screenshots and recordings.
'''

class MainWindow(QMainWindow):
    def __init__(self,settings,start_pipeline=True,data_dir=None):
        super().__init__()
        self.settings=settings
        self.data_dir=data_dir or DATA_DIR
        self.setWindowTitle('AeroSentry — Optical Tracking')
        self.hud=HUDWidget(settings); self.setCentralWidget(self.hud)
        screen=QApplication.primaryScreen().availableGeometry()
        side=min(1000,screen.height()-80,screen.width()-50)
        self.resize(max(620,side),max(620,side))
        self.logger=EventLogger(self.data_dir/'events.sqlite3')
        self.screenshots=ScreenshotService(self.data_dir/'screenshots')
        self.store=TrackStore(settings.smoothing,settings.lost_seconds)
        self.targets=TargetManager(settings.selection)
        self.pipeline=None; self.pending_settings=None
        self.recorder=None; self.retiring_recorders=[]
        self.last_seq=-1; self.last_ids=set(); self.last_target_status=None
        self.last_arrival=0.; self.last_record=0.; self.message_until=0.; self.last_error=''
        self.closing=False; self.close_started=0
        self.hud.state.simulation=settings.simulation
        self.hud.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.hud.customContextMenuRequested.connect(self.context_menu)
        shortcuts={'Space':self.toggle_pause,'L':self.lock,'N':self.next_target,'R':self.reset,
                   'V':self.settings_dialog,'H':self.toggle_hud,'C':self.capture,'Escape':self.release,
                   'F2':self.settings_dialog,'F1':self.help,'O':self.toggle_optical,
                   'F6':self.events,'F9':self.toggle_record,'F11':self.fullscreen}
        for key,callback in shortcuts.items():
            action=QAction(self); action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            # Attaching to the canvas prevents shortcuts firing inside modal text fields.
            self.hud.addAction(action); action.triggered.connect(callback)
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)
        if start_pipeline: self.begin(settings)

    def begin(self,settings):
        self.settings=settings; self.hud.settings=settings
        self.store=TrackStore(settings.smoothing,settings.lost_seconds)
        self.targets=TargetManager(settings.selection)
        label='SIMULATION' if settings.simulation else f'CAM {settings.source}' if settings.source_kind=='Camera' else settings.source_kind.upper()
        self.hud.state=HUDState(simulation=settings.simulation,source=label,status='STARTING',model='LOADING' if not settings.simulation else 'SIM / NO MODEL')
        self.last_seq=-1; self.last_ids=set(); self.last_error=''; self.last_arrival=0
        self.pipeline=Pipeline(settings)
        self.logger.log('source_started',mode='simulation' if settings.simulation else settings.source_kind)

    def restart(self,settings):
        self.stop_record()
        self.pending_settings=settings
        if self.pipeline: self.pipeline.stop()
        else:
            self.pending_settings=None; self.begin(settings)
        self.hud.state.status='STOPPING WORKERS'
        self.hud.state.tracks=[]; self.hud.state.selected=None; self.hud.state.image=None

    def message(self,text,seconds=5):
        self.hud.state.message=text; self.message_until=time.monotonic()+seconds

    def update_selection(self):
        s=self.hud.state
        s.selected=self.targets.choose(self.store.tracks,s.frame_size)
        s.locked=self.targets.locked
        state=(s.selected.track_id if s.selected else None,bool(s.selected and s.selected.missing),s.locked)
        if state!=self.last_target_status:
            self.logger.log('target_state',track_id=state[0],lost=state[1],locked=state[2],simulation=s.simulation)
            self.last_target_status=state

    def tick(self):
        now=time.monotonic()
        self.retiring_recorders=[r for r in self.retiring_recorders if r.is_alive()]
        if self.closing:
            if (not self.pipeline or not self.pipeline.alive()) and not self.retiring_recorders and not self.logger.is_alive():
                self.timer.stop(); self.close(); return
            if now-self.close_started>8:
                self.hud.state.message='A driver is still stopping. Close again to exit.'
            self.hud.update(); return
        if self.pending_settings:
            if not self.pipeline or not self.pipeline.alive():
                settings=self.pending_settings; self.pending_settings=None; self.begin(settings)
            self.hud.update(); return
        s=self.hud.state
        if now>self.message_until: s.message=''
        if self.pipeline:
            pipe=self.pipeline
            pipe.poll_start_capture()
            s.paused=pipe.pause_event.is_set()
            packet=pipe.results.get()
            if packet and packet.sequence!=self.last_seq and not s.paused and not pipe.reset_event.is_set():
                self.last_seq=packet.sequence; self.last_arrival=now
                s.frame_size=(packet.frame.shape[1],packet.frame.shape[0])
                s.tracks=self.store.update(packet.detections,packet.timestamp)
                self.update_selection()
                s.latency_ms=packet.latency_ms; s.inference_fps=packet.inference_fps
                s.model=packet.model_name; s.warning=packet.warning
                frame=packet.frame
                if self.settings.green_tint:
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    rgb=np.empty((*gray.shape,3),dtype=np.uint8)
                    rgb[:,:,0]=(gray*.18).astype(np.uint8); rgb[:,:,1]=gray; rgb[:,:,2]=(gray*.68).astype(np.uint8)
                else: rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                s.image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
                ids={t.track_id for t in s.tracks}
                for ident in ids-self.last_ids: self.logger.log('track_acquired',track_id=ident,simulation=s.simulation)
                for ident in self.last_ids-ids: self.logger.log('track_lost',track_id=ident,simulation=s.simulation)
                self.last_ids=ids
            if pipe.capture: s.capture_fps=pipe.capture.fps
            s.status='SIMULATION' if self.settings.simulation else pipe.processor.status
            if pipe.capture and pipe.capture.ended:
                s.status='END OF VIDEO'; s.capture_fps=s.inference_fps=0.
            elif self.last_arrival and not s.paused:
                s.status='TARGET LOST' if s.selected and s.selected.missing else 'TRACKING' if s.tracks else 'SEARCHING'
            if pipe.error:
                if pipe.error!=self.last_error:
                    self.last_error=pipe.error; self.logger.log('pipeline_error',message=pipe.error)
                s.warning=pipe.error; s.status='ERROR'; s.tracks=[]
                if s.selected: s.selected.missing=True
                s.capture_fps=s.inference_fps=0
                pipe.stop()
            elif self.last_arrival and now-self.last_arrival>max(2,2/self.settings.processing_fps) and not s.paused and not (pipe.capture and pipe.capture.ended):
                s.status='WAITING FOR FRAME'; s.tracks=[]
                if s.selected: s.selected.missing=True
                s.inference_fps=0
        if self.logger.error: s.warning='Event journal unavailable: '+self.logger.error
        if self.recorder:
            if self.recorder.error:
                error=self.recorder.error; self.stop_record(); self.message('Recording error: '+error)
            elif now-self.last_record>=.1:
                self.last_record=now
                img=self.hud.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
                array=np.frombuffer(img.constBits(),np.uint8).reshape(img.height(),img.bytesPerLine())[:,:img.width()*3].reshape(img.height(),img.width(),3)
                self.recorder.put(cv2.cvtColor(array,cv2.COLOR_RGB2BGR))
        self.hud.update()

    def settings_dialog(self):
        dialog=SettingsDialog(self.settings,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            settings=dialog.value()
            try: settings.save(self.data_dir/'settings.json')
            except OSError as exc: self.message('Settings could not be saved: '+str(exc))
            self.restart(settings)
        self.hud.setFocus()
    def toggle_pause(self):
        if self.pipeline:
            if self.pipeline.pause_event.is_set(): self.pipeline.pause_event.clear()
            else: self.pipeline.pause_event.set()
    def lock(self): self.targets.toggle_lock(self.store.tracks); self.update_selection()
    def next_target(self): self.targets.cycle(self.store.tracks); self.update_selection()
    def release(self): self.targets.release(); self.update_selection()
    def reset(self):
        self.store.reset(); self.targets.reset(); self.last_seq=-1; self.last_ids=set()
        self.hud.state.tracks=[]; self.hud.state.selected=None
        if self.pipeline: self.pipeline.reset()
        self.logger.log('tracking_reset'); self.message('Tracking history reset')
    def toggle_hud(self): self.hud.overlays=not self.hud.overlays
    def toggle_optical(self): self.settings.optical=not self.settings.optical
    def capture(self):
        try:
            path=self.screenshots.capture(self.hud); self.message('Screenshot saved: '+path.name)
            self.logger.log('screenshot',file=path.name)
        except OSError as exc: self.message(str(exc))
    def toggle_record(self):
        if self.recorder: self.stop_record()
        else:
            try:
                path=self.data_dir/'recordings'/f'AeroSentry_{datetime.now():%Y%m%d_%H%M%S_%f}.avi'
                self.recorder=Recorder(path); self.hud.state.recording=True
                self.logger.log('recording_started',file=path.name)
            except OSError as exc: self.message(str(exc))
    def stop_record(self):
        if self.recorder:
            self.recorder.close(); self.retiring_recorders.append(self.recorder)
            self.logger.log('recording_stopped',file=self.recorder.path.name,dropped=self.recorder.dropped)
            self.recorder=None
        self.hud.state.recording=False
    def text_dialog(self,title,text):
        dlg=QDialog(self); dlg.setWindowTitle(title); dlg.setStyleSheet(STYLE); dlg.resize(740,640)
        layout=QVBoxLayout(dlg); content=QPlainTextEdit(); content.setReadOnly(True); content.setPlainText(text); layout.addWidget(content)
        dlg.exec(); self.hud.setFocus()
    def help(self): self.text_dialog('AeroSentry — controls and model setup',HELP)
    def events(self): self.text_dialog('AeroSentry — recent events','\n\n'.join(' | '.join(row) for row in self.logger.recent()) or 'No recorded events yet.')
    def fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def context_menu(self,pos):
        menu=QMenu(self)
        for text,callback in [('Settings / source   F2',self.settings_dialog),('Help   F1',self.help),('Screenshot   C',self.capture),('Recording   F9',self.toggle_record),('Events   F6',self.events),('Exit',self.close)]:
            menu.addAction(text,callback)
        menu.exec(self.hud.mapToGlobal(pos))
    def closeEvent(self,event):
        if not self.closing:
            self.closing=True; self.close_started=time.monotonic()
            if self.pipeline: self.pipeline.stop()
            self.stop_record(); self.logger.log('session_closed'); self.logger.close()
            self.hud.state.status='STOPPING'; event.ignore()
        elif ((not self.pipeline or not self.pipeline.alive()) and not self.retiring_recorders and not self.logger.is_alive()) or time.monotonic()-self.close_started>8:
            self.timer.stop(); event.accept()
        else: event.ignore()
