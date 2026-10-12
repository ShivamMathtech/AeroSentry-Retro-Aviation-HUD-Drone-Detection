import pytest
from PySide6.QtGui import QImage
from config import Settings
from ui.coordinates import Viewport
from ui.hud_widget import HUDWidget
from vision.tracker import TrackStore
from vision.telemetry import Detection

@pytest.mark.parametrize('source',[(1280,720),(640,480),(900,1600),(1000,1000)])
def test_letterbox_and_center_alignment(source):
    v=Viewport.fit((43,48,914,835),source)
    x,y=v.point((source[0]/2,source[1]/2),source)
    assert x==pytest.approx(500) and y==pytest.approx(465.5)
    assert v.width/v.height==pytest.approx(source[0]/source[1])
    x,y,w,h=v.box((0,0,*source),source)
    assert (x,y,w,h)==pytest.approx((v.x,v.y,v.width,v.height))

def test_bad_dimensions():
    with pytest.raises(ValueError): Viewport.fit((0,0,100,100),(0,720))

@pytest.mark.parametrize('size',[(1000,1000),(1280,720),(800,1000),(1920,1080)])
def test_hud_resize_capture_and_reticle(app,tmp_path,size):
    hud=HUDWidget(Settings())
    hud.resize(*size)
    store=TrackStore(alpha=1)
    hud.state.tracks=store.update([Detection(2,(400,300,600,500),.96,0,'drone')],0)
    hud.state.selected=hud.state.tracks[0]
    hud.show(); app.processEvents()
    image=hud.grab().toImage()
    assert not image.isNull() and image.width()==size[0]
    mapped=hud.viewport.point(hud.state.selected.smooth,hud.state.frame_size)
    box=hud.viewport.box(hud.state.selected.box,hud.state.frame_size)
    assert mapped==(box[0]+box[2]/2,box[1]+box[3]/2)
    assert image.save(str(tmp_path/'hud.png'))
    hud.close()
