import threading
import time
import cv2
import numpy as np
from config import Settings
from vision.video_source import LatestBuffer,VideoSource
from vision.frame_processor import FrameProcessor,Pipeline
from vision.telemetry import FramePacket,Detection

def wait_for(predicate,timeout=4):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.01)
    assert predicate(),'Timed out waiting for worker'

class Camera:
    def __init__(self): self.reads=0; self.released=False
    def isOpened(self): return True
    def get(self,prop): return 30.
    def read(self):
        self.reads+=1
        return (True,np.zeros((60,80,3),np.uint8)) if self.reads==1 else (False,None)
    def release(self): self.released=True

def test_camera_disconnect_releases_resource():
    cap=Camera(); buffer=LatestBuffer()
    worker=VideoSource(Settings(simulation=False),buffer,threading.Event(),threading.Event(),lambda _:cap)
    worker.start(); worker.join(2)
    assert not worker.is_alive() and cap.released
    assert 'disconnected' in worker.error and buffer.get().sequence==0

def test_unavailable_camera():
    cap=Camera(); cap.isOpened=lambda:False
    worker=VideoSource(Settings(),LatestBuffer(),threading.Event(),threading.Event(),lambda _:cap)
    worker.start(); worker.join(2)
    assert 'Cannot open' in worker.error and cap.released

def test_latest_buffer_is_bounded():
    buffer=LatestBuffer()
    for i in range(10000): buffer.put(i)
    assert buffer.get()==9999
    buffer.clear(); assert buffer.get() is None

def test_real_video_decode_and_eof(tmp_path):
    path=tmp_path/'video.avi'
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(80,60))
    assert out.isOpened()
    for i in range(5): out.write(np.full((60,80,3),i*20,np.uint8))
    out.release()
    buffer=LatestBuffer(); settings=Settings(simulation=False,source_kind='Video',source=str(path))
    worker=VideoSource(settings,buffer,threading.Event(),threading.Event())
    worker.start(); worker.join(3)
    assert worker.ended and not worker.error
    assert buffer.get().sequence==4 and abs(buffer.get().timestamp-.2)<.001

def test_missing_model_never_opens_camera():
    pipeline=Pipeline(Settings(simulation=False,model_path='missing.pt'))
    wait_for(lambda:bool(pipeline.error))
    pipeline.poll_start_capture()
    assert pipeline.capture is None
    pipeline.stop(); pipeline.processor.join(2)

def test_simulation_pause_resume_stop():
    pipeline=Pipeline(Settings(simulation=True))
    wait_for(lambda:pipeline.results.get() is not None)
    assert pipeline.results.get().simulated
    assert pipeline.results.get().model_name=='SIM / NO MODEL'
    pipeline.pause_event.set(); time.sleep(.1)
    seq=pipeline.results.get().sequence
    time.sleep(.12); assert pipeline.results.get().sequence==seq
    pipeline.pause_event.clear()
    wait_for(lambda:pipeline.results.get().sequence>seq)
    pipeline.stop(); pipeline.processor.join(2)
    assert not pipeline.alive() and pipeline.capture is None

class Detector:
    name='test-only'; warning=''
    def __init__(self,settings): pass
    def detect(self,frame): return [Detection(1,(1,2,20,30),.9,0,'drone')],1.0
    def reset(self): pass

def test_inference_packet_uses_exact_input_frame():
    frames,results=LatestBuffer(),LatestBuffer()
    frame=np.full((60,80,3),17,np.uint8)
    frames.put(FramePacket(53,8.3,frame))
    stop=threading.Event()
    worker=FrameProcessor(Settings(),frames,results,stop,threading.Event(),threading.Event(),Detector)
    worker.start(); wait_for(lambda:results.get() is not None)
    packet=results.get()
    assert packet.sequence==53 and packet.timestamp==8.3
    assert packet.frame is frame
    stop.set(); worker.join(2)

def test_inference_exception_surfaces():
    class Broken(Detector):
        def detect(self,frame): raise RuntimeError('test device failure')
    frames,results=LatestBuffer(),LatestBuffer()
    frames.put(FramePacket(0,0,np.zeros((2,2,3),np.uint8)))
    worker=FrameProcessor(Settings(),frames,results,threading.Event(),threading.Event(),threading.Event(),Broken)
    worker.start(); worker.join(2)
    assert worker.error=='test device failure' and results.get() is None
