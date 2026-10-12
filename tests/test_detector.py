from types import SimpleNamespace
import os
from pathlib import Path
import numpy as np
import pytest
from config import Settings
from vision.yolo_detector import YoloDetector,validate_classes

@pytest.mark.parametrize('names,expected',[({0:'drone'},[0]),({1:'UAV'},[1]),(['bird','quadcopter'],[1]),({0:'unmanned_aerial_vehicle'},[0])])
def test_drone_labels(names,expected):
    assert validate_classes(names)[1]==expected

def test_coco_is_not_drone():
    with pytest.raises(ValueError,match='no recognized drone class'):
        validate_classes({0:'person',1:'airplane',2:'bird'})

def test_explicit_other_class_is_labeled():
    names,ids,warning=validate_classes({0:'person',1:'bird'},'1')
    assert ids==[1] and 'NO VERIFIED DRONE LABEL' in warning

def test_bad_class_id():
    with pytest.raises(ValueError,match='absent'): validate_classes({0:'drone'},'5')

def test_wrong_selection_warning():
    assert 'NO DRONE CLASS SELECTED' in validate_classes({0:'drone',1:'bird'},'1')[2]

def test_missing_model():
    with pytest.raises(FileNotFoundError): YoloDetector(Settings(model_path='missing.pt'))

class Tensor:
    def __init__(self,data): self.data=data
    def cpu(self): return self
    def tolist(self): return self.data

class FakeModel:
    names={0:'drone',1:'bird'}
    task='detect'
    def __init__(self,path): self.kwargs=None
    def track(self,frame,**kwargs):
        self.kwargs=kwargs
        boxes=SimpleNamespace(xyxy=Tensor([[-5,10,110,80],[20,30,40,40]]),conf=Tensor([.91,.9]),cls=Tensor([0,1]),id=Tensor([7,8]))
        return [SimpleNamespace(boxes=boxes)]

def test_adapter_passes_cpu_and_real_boxes(tmp_path):
    path=tmp_path/'fake.pt'; path.touch()
    det=YoloDetector(Settings(model_path=str(path)),FakeModel)
    found,latency=det.detect(np.zeros((90,100,3),np.uint8))
    assert found[0].box==(0,10,100,80) and found[0].track_id==7
    assert len(found)==1 and latency>=0
    assert det.model.kwargs['device']=='cpu'
    assert det.model.kwargs['persist'] is True
    assert det.model.kwargs['tracker']=='bytetrack.yaml'

def test_unconfirmed_boxes_are_not_given_fake_ids(tmp_path):
    path=tmp_path/'fake.pt'; path.touch()
    det=YoloDetector(Settings(model_path=str(path)),FakeModel)
    det.model.track=lambda *a,**k:[SimpleNamespace(boxes=SimpleNamespace(id=None))]
    assert det.detect(np.zeros((90,100,3),np.uint8))[0]==[]

def test_actual_cpu_yolo_bytetrack():
    path=os.environ.get('AEROSENTRY_TEST_MODEL')
    if not path: pytest.skip('Set AEROSENTRY_TEST_MODEL to a local generic yolo11n.pt for optional real CPU smoke test')
    import ultralytics
    import cv2
    image=cv2.imread(str(Path(ultralytics.__file__).parent/'assets'/'bus.jpg'))
    assert image is not None
    det=YoloDetector(Settings(model_path=path,class_ids='0,5',input_resolution=320,device='cpu'))
    first,ms=det.detect(image)
    second,_=det.detect(image)
    assert ms>0 and first and second
    assert {d.track_id for d in first}&{d.track_id for d in second}
    assert 'NO VERIFIED DRONE LABEL' in det.warning
    det.reset()
