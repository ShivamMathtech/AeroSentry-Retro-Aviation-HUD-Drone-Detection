"""Track display state layered over ByteTrack IDs; no invented detections."""
from collections import deque
from dataclasses import dataclass, field
import math
from vision.telemetry import Detection

@dataclass
class Track:
    track_id: int
    box: tuple
    confidence: float
    class_id: int
    label: str
    first_seen: float
    last_seen: float
    smooth: tuple
    velocity: tuple = (0., 0.)
    hits: int = 1
    missing: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=48))

    @property
    def center(self):
        return ((self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2)

    @property
    def age(self):
        return max(0., self.last_seen-self.first_seen)

class TrackStore:
    def __init__(self, alpha=.35, retention=2.):
        self.alpha = alpha
        self.retention = retention
        self.tracks = {}

    def update(self, detections, timestamp):
        for t in self.tracks.values():
            t.missing = True
        for d in detections:
            # None / negative IDs cannot be visually locked.
            if d.track_id is None or d.track_id < 0:
                continue
            t = self.tracks.get(d.track_id)
            if t is None or timestamp-t.last_seen > self.retention:
                t = Track(d.track_id,d.box,d.confidence,d.class_id,d.label,
                          timestamp,timestamp,d.center)
                self.tracks[d.track_id] = t
            else:
                old,dt = t.center,timestamp-t.last_seen
                if dt>0:
                    t.velocity = tuple((n-o)/dt for n,o in zip(d.center,old))
                t.smooth = tuple(self.alpha*n+(1-self.alpha)*o for n,o in zip(d.center,t.smooth))
                t.box,t.confidence,t.label = d.box,d.confidence,d.label
                t.last_seen = timestamp
                t.hits += 1
            t.missing = False
            t.history.append(d.center)
        self.tracks = {i:t for i,t in self.tracks.items() if timestamp-t.last_seen <= self.retention}
        return [t for t in self.tracks.values() if not t.missing]

    def reset(self):
        self.tracks.clear()
