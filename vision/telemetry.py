from dataclasses import dataclass, field
from typing import Any
import time

@dataclass(frozen=True)
class Detection:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

@dataclass
class FramePacket:
    sequence: int
    timestamp: float
    frame: Any

@dataclass
class ResultPacket:
    sequence: int
    timestamp: float
    frame: Any
    detections: list[Detection]
    latency_ms: float
    inference_fps: float
    model_name: str
    warning: str = ''
    simulated: bool = False

@dataclass
class HUDState:
    image: Any = None
    frame_size: tuple[int, int] = (1280, 720)
    tracks: list = field(default_factory=list)
    selected: Any = None
    locked: bool = False
    status: str = 'READY'
    source: str = 'SIMULATION'
    model: str = 'SIM / NO MODEL'
    warning: str = ''
    capture_fps: float = 0
    inference_fps: float = 0
    latency_ms: float = 0
    recording: bool = False
    paused: bool = False
    simulation: bool = True
    message: str = ''
    now: float = field(default_factory=time.monotonic)
