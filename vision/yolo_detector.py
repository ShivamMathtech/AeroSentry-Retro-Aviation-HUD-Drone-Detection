"""Ultralytics/ByteTrack adapter; only explicitly chosen model classes are returned."""
from pathlib import Path
import re
import time
from vision.telemetry import Detection

DRONE_LABELS = {'drone','drones','uav','uavs','uas','quadcopter','quadcopter drone','unmanned aerial vehicle'}

def normalized(label):
    return re.sub(r'[_-]+',' ',str(label).strip().lower())

def validate_classes(names, selected=''):
    names = dict(enumerate(names)) if isinstance(names,(tuple,list)) else {int(k):str(v) for k,v in names.items()}
    compatible = [i for i,name in names.items() if normalized(name) in DRONE_LABELS]
    ids = [int(x.strip()) for x in selected.split(',')] if selected.strip() else compatible
    if any(i not in names for i in ids):
        raise ValueError('Selected class ID is absent from the checkpoint. Inspect its class list.')
    warning = ''
    if not compatible:
        warning = 'NO VERIFIED DRONE LABEL: selected classes are not evidence of drone detection.'
    elif ids and not any(i in compatible for i in ids):
        warning = 'NO DRONE CLASS SELECTED: tracking selected non-drone classes only.'
    if not ids:
        raise ValueError('Model has no recognized drone class. Select explicit class IDs in Settings for other-object tracking. Labels: '+str(names))
    return names,sorted(set(ids)),warning

class YoloDetector:
    def __init__(self, settings, model_factory=None):
        path = Path(settings.model_path).expanduser()
        if not path.is_file() or path.suffix.lower() != '.pt':
            raise FileNotFoundError('YOLO weights missing. Select a local, trusted drone-trained .pt checkpoint in F2 Settings.')
        if model_factory is None:
            import torch
            if settings.device != 'cpu' and not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. Select CPU or install a matching CUDA PyTorch build.')
            from ultralytics import YOLO
            model_factory = YOLO
        self.model = model_factory(str(path))
        if getattr(self.model,'task','detect') != 'detect':
            raise ValueError('This application requires an object-detection checkpoint (task=detect).')
        self.names,self.classes,self.warning = validate_classes(self.model.names,settings.class_ids)
        self.settings = settings
        self.name = path.name

    def detect(self, frame):
        start = time.perf_counter()
        results = self.model.track(frame,persist=True,tracker='bytetrack.yaml',
                                   conf=self.settings.confidence,iou=self.settings.iou,
                                   imgsz=self.settings.input_resolution,device=self.settings.device,
                                   classes=self.classes,verbose=False)
        elapsed = (time.perf_counter()-start)*1000
        found = []
        height,width = frame.shape[:2]
        if not results:
            return found,elapsed
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return found,elapsed
        for box,conf,cls,ident in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),
                                    boxes.cls.cpu().tolist(),boxes.id.cpu().tolist()):
            cls = int(cls)
            if cls not in self.classes:
                continue
            x1,y1,x2,y2 = box
            bounded = (max(0,min(width,x1)),max(0,min(height,y1)),
                       max(0,min(width,x2)),max(0,min(height,y2)))
            if bounded[2]>bounded[0] and bounded[3]>bounded[1]:
                found.append(Detection(int(ident),bounded,float(conf),cls,self.names[cls]))
        return found,elapsed

    def reset(self):
        predictor = getattr(self.model,'predictor',None)
        for tracker in getattr(predictor,'trackers',[]) or []:
            tracker.reset()
