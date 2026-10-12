import time
import sqlite3
import cv2
import numpy as np
from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from config import Settings
from services.event_logger import EventLogger
from services.screenshot_service import ScreenshotService
from services.recorder import Recorder
from ui.hud_widget import HUDWidget
from ui.settings_dialog import SettingsDialog
from ui.main_window import MainWindow

def pump(app,predicate,timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate(): return
        time.sleep(.01)
    assert predicate()

def test_screenshot(app,tmp_path):
    hud=HUDWidget(Settings()); hud.show(); app.processEvents()
    path=ScreenshotService(tmp_path).capture(hud)
    assert path.exists() and cv2.imread(str(path)) is not None
    hud.close()

def test_sqlite_journal(tmp_path):
    logger=EventLogger(tmp_path/'events.sqlite3')
    logger.log('track_acquired',track_id=12,simulation=True)
    logger.close(); logger.join(3)
    assert not logger.error
    with sqlite3.connect(logger.path) as db:
        row=db.execute('SELECT kind,payload FROM events').fetchone()
        assert row[0]=='track_acquired' and '12' in row[1]

def test_recording_can_be_decoded(tmp_path):
    recorder=Recorder(tmp_path/'hud.avi',size=(80,60))
    recorder.put(np.full((60,80,3),120,np.uint8))
    recorder.close(); recorder.join(3)
    assert not recorder.error and recorder.written==1
    cap=cv2.VideoCapture(str(recorder.path)); ok,frame=cap.read(); cap.release()
    assert ok and frame.shape[:2]==(60,80)

def test_rtsp_credentials_not_saved(tmp_path):
    s=Settings(source_kind='RTSP',source='rtsp://name:secret@camera/live')
    path=tmp_path/'settings.json'; s.save(path)
    assert 'secret' not in path.read_text()
    assert Settings.load(path).source==''

@pytest.mark.parametrize('kwargs',[{'confidence':0},{'input_resolution':13},{'processing_fps':0},{'class_ids':'x'},{'device':'oops'}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError): Settings(**kwargs).validate()

def test_settings_roundtrip(app):
    d=SettingsDialog(Settings()); value=d.value()
    assert value==Settings()
    d.close()

def test_main_simulation_controls_shutdown(app,tmp_path):
    window=MainWindow(Settings(),data_dir=tmp_path)
    window.show()
    pump(app,lambda:window.last_seq>=1)
    assert window.hud.state.simulation and window.hud.state.tracks
    window.lock(); assert window.targets.locked
    window.release(); assert window.hud.state.selected is None
    window.next_target(); assert window.hud.state.selected is not None
    window.reset(); assert not window.store.tracks
    window.toggle_hud(); assert not window.hud.overlays
    window.toggle_pause(); assert window.pipeline.pause_event.is_set()
    window.capture(); assert list((tmp_path/'screenshots').glob('*.png'))
    window.close()
    pump(app,lambda:not window.isVisible())
    assert not window.pipeline.alive()

def test_settings_text_does_not_trigger_hud_shortcuts(app,tmp_path):
    window=MainWindow(Settings(),start_pipeline=False,data_dir=tmp_path); window.show()
    dialog=SettingsDialog(window.settings,window); dialog.setModal(True); dialog.show()
    field=dialog.fields['source']; field.setFocus(); field.clear(); app.processEvents()
    QTest.keyClicks(field,'camera H L N R C')
    assert field.text()=='camera H L N R C'
    assert window.hud.overlays and not list(tmp_path.glob('screenshots/*.png'))
    dialog.close(); window.close(); pump(app,lambda:not window.isVisible())

def test_actual_cpu_video_to_hud(app,tmp_path):
    import os
    import ultralytics
    model=os.environ.get('AEROSENTRY_TEST_MODEL')
    if not model: pytest.skip('Local yolo11n.pt required for optional end-to-end CPU test')
    source=cv2.imread(str(Path(ultralytics.__file__).parent/'assets'/'bus.jpg'))
    source=cv2.resize(source,(480,640))
    video=tmp_path/'bus.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(480,640))
    assert writer.isOpened()
    for _ in range(20): writer.write(source)
    writer.release()
    window=MainWindow(Settings(simulation=False,source_kind='Video',source=str(video),model_path=model,class_ids='0,5',input_resolution=320),data_dir=tmp_path/'state')
    window.show()
    try:
        pump(app,lambda:bool(window.hud.state.tracks),timeout=15)
        assert not window.hud.state.simulation
        assert window.hud.state.image.width()==480
        assert window.hud.state.model=='yolo11n.pt'
        assert window.hud.state.latency_ms>0
        assert 'NO VERIFIED DRONE LABEL' in window.hud.state.warning
        assert all(t.label in ('person','bus') for t in window.hud.state.tracks)
        assert window.hud.grab().save(str(tmp_path/'live-pipeline.png'))
    finally:
        window.close(); pump(app,lambda:not window.isVisible(),timeout=10)
    assert not window.pipeline.alive()
