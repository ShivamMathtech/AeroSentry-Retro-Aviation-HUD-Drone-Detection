"""Validated settings; user state lives outside the source tree."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import os

DATA_DIR = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share')) / 'AeroSentry'

@dataclass
class Settings:
    simulation: bool = True
    source_kind: str = 'Camera'
    source: str = '0'
    model_path: str = ''
    class_ids: str = ''
    confidence: float = .30
    iou: float = .50
    input_resolution: int = 640
    processing_fps: int = 15
    device: str = 'cpu'
    brightness: float = .85
    glow: bool = True
    crt: bool = True
    smoothing: float = .35
    selection: str = 'Nearest center'
    optical: bool = True
    green_tint: bool = True
    reference_course: float = 150.0
    lost_seconds: float = 2.0

    def validate(self):
        for name in ('confidence', 'iou', 'brightness', 'smoothing'):
            if not .01 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be between 0.01 and 1.00')
        if self.input_resolution not in (320, 416, 640, 960, 1280):
            raise ValueError('Unsupported inference resolution')
        if not 1 <= self.processing_fps <= 60:
            raise ValueError('Processing FPS must be 1–60')
        if self.source_kind not in ('Camera', 'Video', 'RTSP'):
            raise ValueError('Invalid source type')
        if self.source_kind == 'Camera' and (not self.source.isdigit()):
            raise ValueError('Camera must be a nonnegative index, e.g. 0')
        if self.device not in ('cpu', '0'):
            raise ValueError('Device must be cpu or 0 (CUDA)')
        if not 0 <= self.reference_course < 360:
            raise ValueError('Reference course must be 0–359.9 degrees')
        if not .1 <= self.lost_seconds <= 30:
            raise ValueError('Lost timeout must be 0.1–30 seconds')
        if self.class_ids:
            try:
                ids = [int(x.strip()) for x in self.class_ids.split(',')]
                if any(i < 0 for i in ids):
                    raise ValueError()
            except ValueError:
                raise ValueError('Class IDs must be comma-separated nonnegative integers') from None
        return self

    def save(self, path=None):
        path = Path(path or DATA_DIR / 'settings.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Do not persist embedded RTSP credentials.
        if data['source_kind'] == 'RTSP':
            data['source'] = ''
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temp.replace(path)

    @classmethod
    def load(cls, path=None):
        path = Path(path or DATA_DIR / 'settings.json')
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}).validate()
