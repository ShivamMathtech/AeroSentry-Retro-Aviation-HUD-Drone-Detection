"""Inference uses its own worker and result mailbox (not an unbounded Qt signal queue)."""
import threading
import time
import math
import cv2
import numpy as np
from vision.telemetry import Detection, ResultPacket
from vision.yolo_detector import YoloDetector
from vision.video_source import LatestBuffer, VideoSource

class FrameProcessor(threading.Thread):
    def __init__(self,settings,frames,results,stop,pause,reset,detector_factory=YoloDetector):
        super().__init__(name='AeroSentry-inference',daemon=True)
        self.settings,self.frames,self.results = settings,frames,results
        self.stop_event,self.pause_event,self.reset_event = stop,pause,reset
        self.detector_factory = detector_factory
        self.error = ''
        self.status = 'LOADING MODEL'
        self.model_name = ''
        self.warning = ''
        self.ready = threading.Event()

    def run(self):
        try:
            detector = self.detector_factory(self.settings)
            self.model_name,self.warning = detector.name,detector.warning
            self.status = 'MODEL READY'
            self.ready.set()
            last_seq,last_result = -1,None
            fps = 0.
            while not self.stop_event.is_set():
                cycle = time.monotonic()
                if self.reset_event.is_set():
                    detector.reset()
                    self.results.clear()
                    self.reset_event.clear()
                    last_seq = -1
                if self.pause_event.is_set():
                    self.stop_event.wait(.04)
                    last_result = None
                    continue
                packet = self.frames.get()
                if packet is None or packet.sequence == last_seq:
                    self.stop_event.wait(.005)
                    continue
                last_seq = packet.sequence
                detections,latency = detector.detect(packet.frame)
                if self.reset_event.is_set():
                    continue
                now = time.monotonic()
                if last_result is not None:
                    sample = 1/max(now-last_result,1e-6)
                    fps = sample if not fps else .2*sample+.8*fps
                last_result = now
                self.results.put(ResultPacket(packet.sequence,packet.timestamp,packet.frame,detections,
                                               latency,fps,detector.name,detector.warning))
                self.status = 'PROCESSING'
                self.stop_event.wait(max(0,1/self.settings.processing_fps-(time.monotonic()-cycle)))
        except Exception as exc:
            self.error = str(exc)
            self.status = 'MODEL ERROR'

class SimulationWorker(threading.Thread):
    def __init__(self,settings,results,stop,pause):
        super().__init__(name='AeroSentry-simulation',daemon=True)
        self.settings,self.results,self.stop_event,self.pause_event = settings,results,stop,pause
        self.error = ''
        self.model_name = 'SIM / NO MODEL'
        self.warning = 'Synthetic visual tracks. No camera input or neural inference.'
        self.status = 'SIMULATION'

    def run(self):
        sequence = 0
        phase = 0.
        try:
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.stop_event.wait(.04)
                    continue
                frame = np.full((900,1000,3),(10,19,17),dtype=np.uint8)
                # A plainly synthetic vector drone used only in explicitly labeled simulation.
                cx,cy = 485+65*math.sin(phase*.35),470+24*math.sin(phase*.6)
                ix,iy = int(cx),int(cy)
                cv2.ellipse(frame,(ix,iy),(24,14),0,0,360,(21,43,32),-1)
                for dx,dy in [(-57,-20),(57,-20),(-48,18),(48,18)]:
                    cv2.line(frame,(ix,iy),(ix+dx,iy+dy),(25,52,39),5)
                    cv2.ellipse(frame,(ix+dx,iy+dy),(26,5),0,0,360,(27,66,49),2)
                detections = [Detection(2,(cx-84,cy-56,cx+84,cy+64),.96,0,'SIM-UAS')]
                # A repeatable three-second occlusion exercises lost/reacquired state.
                if 10 <= phase%18 < 13:
                    detections = []
                    frame[:] = (10,19,17)
                self.results.put(ResultPacket(sequence,phase,frame,detections,0.,0.,
                                              self.model_name,self.warning,True))
                sequence += 1
                phase += 1/self.settings.processing_fps
                self.stop_event.wait(1/self.settings.processing_fps)
        except Exception as exc:
            self.error = str(exc)

class Pipeline:
    def __init__(self,settings,detector_factory=YoloDetector):
        self.settings = settings
        self.stop_event,self.pause_event,self.reset_event = threading.Event(),threading.Event(),threading.Event()
        self.frames,self.results = LatestBuffer(),LatestBuffer()
        self.capture = None
        self.processor = SimulationWorker(settings,self.results,self.stop_event,self.pause_event) if settings.simulation else FrameProcessor(settings,self.frames,self.results,self.stop_event,self.pause_event,self.reset_event,detector_factory)
        self.processor.start()
        self.capture_started = False

    def poll_start_capture(self):
        # Camera opens only once model validation succeeds. Failed weights never flash the camera.
        if not self.settings.simulation and not self.capture_started and self.processor.ready.is_set() and not self.stop_event.is_set():
            self.capture = VideoSource(self.settings,self.frames,self.stop_event,self.pause_event)
            self.capture_started = True
            self.capture.start()

    def stop(self):
        self.stop_event.set()

    def alive(self):
        return self.processor.is_alive() or bool(self.capture and self.capture.is_alive())

    def reset(self):
        if not self.settings.simulation:
            self.reset_event.set()
        self.results.clear()

    @property
    def error(self):
        return self.processor.error or (self.capture.error if self.capture else '')
