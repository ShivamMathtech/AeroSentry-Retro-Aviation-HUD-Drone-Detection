"""Single-slot capture buffering. Camera/decoder resources belong to this worker."""
import threading
import time
from pathlib import Path
import cv2
from vision.telemetry import FramePacket

class LatestBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def put(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value

    def clear(self):
        self.put(None)

class VideoSource(threading.Thread):
    def __init__(self, settings, buffer, stop_event, pause_event, capture_factory=None):
        super().__init__(name='AeroSentry-capture',daemon=True)
        self.settings,self.buffer = settings,buffer
        self.stop_event,self.pause_event = stop_event,pause_event
        self.capture_factory = capture_factory
        self.fps = 0.
        self.status = 'OPENING SOURCE'
        self.error = ''
        self.ended = False

    def _open(self):
        s = self.settings
        if s.source_kind == 'Video' and not Path(s.source).is_file():
            raise ValueError('Video file does not exist. Select it in Settings.')
        if s.source_kind == 'RTSP' and not s.source.lower().startswith(('rtsp://','rtsps://')):
            raise ValueError('RTSP source must begin with rtsp:// or rtsps://')
        source = int(s.source) if s.source_kind == 'Camera' else s.source
        if self.capture_factory:
            return self.capture_factory(source)
        if s.source_kind == 'RTSP':
            return cv2.VideoCapture(source,cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
        cap = cv2.VideoCapture(source)
        if s.source_kind == 'Camera':
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return cap

    def run(self):
        cap = None
        try:
            cap = self._open()
            if not cap.isOpened():
                raise RuntimeError('Cannot open source. Check camera permissions, index, file codec or stream availability.')
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            file_fps = file_fps if 1 <= file_fps <= 240 else 30.
            index = 0
            last = None
            self.status = 'CAPTURING'
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.stop_event.wait(.04)
                    last = None
                    continue
                cycle = time.monotonic()
                ok,frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    if self.settings.source_kind == 'Video':
                        self.status = 'END OF VIDEO'
                        self.ended = True
                        break
                    raise RuntimeError('Camera/stream disconnected or returned an empty frame. Reopen source with V.')
                now = time.monotonic()
                # Source timeline makes pixel/second velocity meaningful for file playback.
                timestamp = index/file_fps if self.settings.source_kind == 'Video' else now
                self.buffer.put(FramePacket(index,timestamp,frame))
                index += 1
                if last is not None:
                    sample = 1/max(now-last,1e-6)
                    self.fps = sample if not self.fps else .15*sample+.85*self.fps
                last = now
                if self.settings.source_kind == 'Video':
                    self.stop_event.wait(max(0,1/file_fps-(time.monotonic()-cycle)))
        except Exception as exc:
            # OpenCV exceptions may contain stream credentials: keep those out of HUD/logs.
            self.error = str(exc) if not isinstance(exc,cv2.error) else 'OpenCV source error. Check source/codec and camera access.'
            self.status = 'SOURCE ERROR'
        finally:
            if cap is not None:
                cap.release()
