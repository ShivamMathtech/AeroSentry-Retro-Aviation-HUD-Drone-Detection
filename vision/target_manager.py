class TargetManager:
    """Selection is stable while locked. Reacquisition requires the same tracker ID."""
    def __init__(self, behavior='Nearest center'):
        self.selected_id = None
        self.locked = False
        self.behavior = behavior
        self.manual_release = False
        self.last_selected = None

    def choose(self, tracks, frame_size):
        visible = [t for t in tracks.values() if not t.missing]
        current = tracks.get(self.selected_id)
        if self.locked:
            if current:
                self.last_selected = current
            elif self.last_selected:
                self.last_selected.missing = True
            return current or self.last_selected
        if self.manual_release or self.behavior == 'Manual':
            return current if current and not current.missing else None
        if current and not current.missing:
            self.last_selected = current
            return current
        if not visible:
            self.selected_id = None
            return None
        if self.behavior == 'Highest confidence':
            target = max(visible,key=lambda t:t.confidence)
        else:
            w,h = frame_size
            target = min(visible,key=lambda t:(t.center[0]-w/2)**2+(t.center[1]-h/2)**2)
        self.selected_id = target.track_id
        self.last_selected = target
        return target

    def cycle(self, tracks):
        ids = sorted(i for i,t in tracks.items() if not t.missing)
        if ids:
            self.selected_id = ids[(ids.index(self.selected_id)+1)%len(ids)] if self.selected_id in ids else ids[0]
            self.last_selected = tracks[self.selected_id]
            self.manual_release = False

    def toggle_lock(self, tracks):
        if self.locked:
            self.locked = False
        else:
            if self.selected_id not in tracks:
                self.cycle(tracks)
            if self.selected_id in tracks:
                self.locked = True
                self.last_selected = tracks[self.selected_id]

    def release(self):
        self.selected_id = None
        self.last_selected = None
        self.locked = False
        self.manual_release = True

    def reset(self):
        self.release()
        self.manual_release = False
