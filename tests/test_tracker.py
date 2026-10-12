import pytest
from vision.telemetry import Detection
from vision.tracker import TrackStore
from vision.target_manager import TargetManager

def det(i=1,x=100,conf=.9):
    return Detection(i,(x,100,x+20,120),conf,0,'drone')

def test_ema_does_not_move_raw_box():
    store=TrackStore(alpha=.25)
    store.update([det()],0)
    track=store.update([det(x=140)],1)[0]
    assert track.box==(140,100,160,120)
    assert track.smooth==(120,110)
    assert track.center==(150,110)
    assert track.velocity==(40,0)
    assert track.hits==2 and track.age==1

def test_loss_and_reacquisition_same_id():
    store=TrackStore(retention=2)
    targets=TargetManager()
    store.update([det()],0); targets.choose(store.tracks,(640,480)); targets.toggle_lock(store.tracks)
    store.update([],1)
    assert targets.choose(store.tracks,(640,480)).missing
    store.update([det(x=102)],1.5)
    assert targets.choose(store.tracks,(640,480)).track_id==1
    assert not targets.choose(store.tracks,(640,480)).missing

def test_locked_id_never_jumps_to_other_target():
    store=TrackStore(retention=.5); targets=TargetManager()
    store.update([det()],0); targets.choose(store.tracks,(640,480)); targets.toggle_lock(store.tracks)
    store.update([det(i=9)],2)
    selected=targets.choose(store.tracks,(640,480))
    assert selected.track_id==1 and selected.missing
    targets.cycle(store.tracks)
    assert targets.choose(store.tracks,(640,480)).track_id==9

def test_cycle_and_manual_release():
    store=TrackStore(); targets=TargetManager()
    store.update([det(1),det(2)],0)
    targets.cycle(store.tracks); assert targets.selected_id==1
    targets.cycle(store.tracks); assert targets.selected_id==2
    targets.release(); assert targets.choose(store.tracks,(640,480)) is None

def test_manual_selection():
    store=TrackStore(); targets=TargetManager('Manual')
    store.update([det()],0)
    assert targets.choose(store.tracks,(640,480)) is None
    targets.cycle(store.tracks)
    assert targets.choose(store.tracks,(640,480)).track_id==1

def test_confidence_selection():
    store=TrackStore(); targets=TargetManager('Highest confidence')
    store.update([det(1,conf=.5),det(2,conf=.99)],0)
    assert targets.choose(store.tracks,(640,480)).track_id==2

def test_history_bound_and_reset():
    store=TrackStore()
    for i in range(100): store.update([det(x=i)],i*.1)
    assert len(store.tracks[1].history)==48
    store.reset(); assert not store.tracks

def test_expired_track_removed():
    store=TrackStore(retention=1)
    store.update([det()],0); store.update([],2)
    assert not store.tracks

def test_none_and_negative_ids_ignored():
    store=TrackStore()
    assert store.update([det(None),det(-1)],0)==[]
