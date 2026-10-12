"""Single custom-painted instrument. No dashboard panels or synthetic live targets."""
from datetime import datetime
import math
import random
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget
from ui.coordinates import Viewport
from vision.telemetry import HUDState

class HUDWidget(QWidget):
    OPTICAL_BOUNDS = (43, 48, 914, 835)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.state = HUDState()
        self.overlays = True
        self.setMinimumSize(620, 620)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.rng = random.Random(47)
        self.grain = [(self.rng.randrange(1000), self.rng.randrange(1000),
                       self.rng.randrange(3, 16)) for _ in range(7500)]
        self.viewport = Viewport.fit(self.OPTICAL_BOUNDS, self.state.frame_size)

    def text(self, p, x, y, text, size=19, bold=True):
        font = QFont('Consolas', 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(size)
        font.setBold(bold)
        p.setFont(font)
        p.drawText(QPointF(x, y), str(text))

    def line(self, p, x1, y1, x2, y2):
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def draw_rect(self, p, x, y, w, h):
        p.drawRect(QRectF(x, y, w, h))

    def boxed(self, p, x, y, value, size=22):
        width = len(str(value)) * size * .62 + 12
        self.draw_rect(p, x-5, y-size, width, size+7)
        self.text(p, x, y, value, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#020b0e'))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        p.translate((self.width()-side)/2, (self.height()-side)/2)
        p.scale(side/1000, side/1000)
        self.viewport = Viewport.fit(self.OPTICAL_BOUNDS, self.state.frame_size)
        v = self.viewport
        if self.settings.optical and self.state.image is not None:
            p.drawImage(QRectF(v.x, v.y, v.width, v.height), self.state.image)
            p.fillRect(QRectF(*self.OPTICAL_BOUNDS), QColor(0, 9, 11, 40))
        self.draw_crt_effects(p)
        color = QColor(0, int(245*self.settings.brightness), int(164*self.settings.brightness))
        # Two restrained strokes give phosphor spread without blurring the optical image.
        if self.overlays:
            if self.settings.glow:
                p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 25), 4.5))
                self.draw_graphics(p)
            p.setPen(QPen(color, 1.35))
            self.draw_graphics(p)
        # Safety/source state is always visible, even when H hides instrument graphics.
        p.setPen(QPen(color, 1.35))
        self.draw_status_indicators(p)
        p.end()

    def draw_graphics(self, p):
        self.draw_outer_frame(p)
        self.draw_top_readouts(p)
        self.draw_left_telemetry(p)
        self.draw_reference_lines(p)
        self.draw_target_brackets(p)
        self.draw_tracking_reticle(p)
        self.draw_right_readouts(p)
        self.draw_bottom_scale(p)

    def draw_outer_frame(self, p):
        self.draw_rect(p, 42, 47, 916, 837)
        self.line(p, 42, 913, 958, 913)
        self.text(p, 814, 37, datetime.now().strftime('%H:%M:%S'), 27)

    def draw_top_readouts(self, p):
        s = self.state
        self.text(p, 64, 81, 'AEROSENTRY', 25)
        self.text(p, 64, 101, 'DRONE DETECTION', 18)
        self.text(p, 64, 134, f'CRS: {self.settings.reference_course:03.0f}  REF', 20)
        self.text(p, 64, 163, 'DME: N/A', 21)
        self.text(p, 64, 188, f'DET: {len(s.tracks)}', 21)
        model = s.model if len(s.model) <= 26 else s.model[:23]+'...'
        self.text(p, 64, 213, f'MODEL: {model}', 16)
        for x, major in [(373, True), (435, False), (500, True), (563, False), (625, True)]:
            self.line(p, x, 103, x, 117 if major else 111)
        self.line(p, 500, 61, 500, 84)
        self.boxed(p, 478, 136, f'{self.settings.reference_course:03.0f}', 24)
        self.text(p, 355, 143, f'{(self.settings.reference_course-10)%360:03.0f}', 20)
        self.text(p, 611, 143, f'{(self.settings.reference_course+10)%360:03.0f}', 20)
        self.text(p, 454, 159, 'REF SCALE', 11, False)
        sel = s.selected
        self.text(p, 813, 109, f'TRK: {sel.track_id:02d}' if sel else 'TRK: --', 24)
        self.text(p, 821, 139, 'VIS LOCK' if s.locked else 'VISUAL', 17)

    def draw_left_telemetry(self, p):
        s, t = self.state, self.state.selected
        rows = [('CONF', f'{t.confidence*100:.1f}%' if t else '--'),
                ('CX PX', f'{t.center[0]:.1f}' if t else '--'),
                ('CY PX', f'{t.center[1]:.1f}' if t else '--'),
                ('PERSIST', str(t.hits) if t else '--'),
                ('INF MS', f'{s.latency_ms:.1f}'),
                ('CAM FPS', f'{s.capture_fps:.1f}'),
                ('INF FPS', f'{s.inference_fps:.1f}')]
        for i, (label, value) in enumerate(rows):
            y = 273+i*54
            self.text(p, 88, y-18, label, 11, False)
            self.text(p, 99, y, value, 22)
        for y in range(263, 660, 17):
            self.line(p, 197, y, 205 if y%51 else 210, y)
        self.text(p, 125, 618, 'VISUAL', 23)
        self.text(p, 126, 647, 'RNG N/A', 17)
        self.text(p, 126, 674, 'ALT N/A', 17)
        self.text(p, 126, 698, 'SPD N/A', 17)
        self.text(p, 64, 790, f'FPS: {s.capture_fps:.1f}', 18)
        self.text(p, 64, 816, f'INF: {s.latency_ms:.1f} ms', 18)
        self.text(p, 64, 841, f'SRC: {s.source}', 16)
        self.text(p, 64, 866, f'RES: {s.frame_size[0]}x{s.frame_size[1]}', 16)

    def draw_reference_lines(self, p):
        self.line(p, 44, 569, 781, 467)
        for x in (300, 697):
            y = 569-(x-44)*102/737
            self.line(p, x, y-8, x, y+8)
        self.text(p, 348, 548, 'OPTICAL REFERENCE', 10, False)
        self.line(p, 373, 223, 373, 205)
        self.line(p, 373, 205, 432, 190)
        self.text(p, 352, 218, '5', 24)
        # Fixed boresight, explicitly separate from the moving EMA reticle.
        for x1, y1, x2, y2 in [(481,261,494,261),(506,261,519,261),
                                (500,241,500,255),(500,267,500,282)]:
            self.line(p, x1, y1, x2, y2)

    def draw_target_brackets(self, p):
        s = self.state
        for t in s.tracks:
            x, y, w, h = self.viewport.box(t.box, s.frame_size)
            selected = s.selected is not None and t.track_id == s.selected.track_id
            p.save()
            if not selected:
                p.setOpacity(.48)
            arm = min(24, w*.22, h*.22)
            for sx, sy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                self.line(p,sx,sy,sx+dx*arm,sy)
                self.line(p,sx,sy,sx,sy+dy*arm)
            # Brackets use raw detector coordinates, never the smoothed reticle.
            self.text(p, x, max(65,y-12), f'{t.label.upper()[:14]}-{t.track_id:02d}', 15)
            if selected:
                points = [QPointF(*self.viewport.point(pt, s.frame_size)) for pt in t.history]
                if len(points)>1:
                    p.drawPolyline(QPolygonF(points))
            p.restore()

    def draw_tracking_reticle(self, p):
        s, t = self.state, self.state.selected
        if t is None or t.missing:
            self.text(p, 415, 467, 'TARGET LOST' if t else 'SEARCHING', 18)
            return
        x, y = self.viewport.point(t.smooth, s.frame_size)
        p.drawEllipse(QPointF(x,y),17,17)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            self.line(p,x+dx*19,y+dy*19,x+dx*33,y+dy*33)
        p.drawEllipse(QPointF(x,y),2,2)
        self.text(p,x+25,y+33,'EMA',10,False)
        vx, vy = t.velocity
        speed = math.hypot(vx,vy)
        if speed > 1:
            # Direction only, magnitude shown separately in pixels/second.
            length = min(55, 16+speed*.15)
            ex, ey = x+vx/speed*length, y+vy/speed*length
            self.line(p,x,y,ex,ey)
            theta = math.atan2(vy,vx)
            for a in (-.5,.5):
                self.line(p,ex,ey,ex-8*math.cos(theta+a),ey-8*math.sin(theta+a))

    def draw_right_readouts(self, p):
        s,t = self.state,self.state.selected
        cx,cy = t.center if t else (0,0)
        values = [f'CLASS: {t.label.upper()[:8]}' if t else 'CLASS: --',
                  f'CONF: {t.confidence:.0%}' if t else 'CONF: --',
                  f'ID:   {t.track_id:02d}' if t else 'ID:   --',
                  f'CX: {cx/s.frame_size[0]:.3f}' if t else 'CX: --',
                  f'CY: {cy/s.frame_size[1]:.3f}' if t else 'CY: --',
                  f'MOV: {math.hypot(*t.velocity):.1f}' if t else 'MOV: --',
                  f'AGE: {t.age:.1f}s' if t else 'AGE: --',
                  f'TRK: {"LOST" if t.missing else "ACTIVE"}' if t else 'TRK: SEARCH']
        ys = [226,252,278,326,353,392,420,446]
        for val,y in zip(values,ys):
            self.text(p,814,y,val,16)
        for y in (206,286,369):
            self.line(p,814,y,942,y)
        for y in range(269,661,17):
            self.line(p,788,y,797,y)
        self.text(p,815,469,'IMAGE PX/SEC',11,False)
        self.boxed(p,818,501,f'{math.hypot(*t.velocity):.1f}' if t else '--',23)
        self.text(p,813,573,'RNG N/A',20)
        self.boxed(p,818,668,'N/A',22)
        self.text(p,787,699,'ALT N/A',22)
        self.text(p,824,813,'CAMERA',18)
        self.text(p,824,839,'TRACKING',18)
        self.text(p,824,864,'2D ONLY',18)

    def draw_bottom_scale(self, p):
        for i in range(-5,6):
            angle = math.radians(i*11)
            x,y = 500+185*math.sin(angle),965-185*math.cos(angle)
            self.line(p,x,y,x+math.sin(angle)*(-15 if i%2==0 else -8),y+math.cos(angle)*(15 if i%2==0 else 8))
        p.drawPolygon(QPolygonF([QPointF(482,867),QPointF(500,831),QPointF(518,867)]))
        self.line(p,493,852,500,842)
        self.line(p,500,842,507,852)
        self.text(p,389,784,'5',24)
        self.text(p,596,784,'5',24)
        self.text(p,460,891,'REF ONLY',10,False)

    def draw_status_indicators(self, p):
        s = self.state
        mode = 'SIMULATION' if s.simulation else 'LIVE OPTICAL' if self.settings.optical else 'HUD ONLY'
        if s.simulation:
            self.text(p,403,184,'SIMULATION',22)
            self.text(p,393,203,'SYNTHETIC / NO YOLO',11)
        self.text(p,55,940,f'MODE: {mode}',14)
        self.text(p,305,940,f'DEV: {self.settings.device.upper()}',14)
        self.text(p,444,940,f'REC: {"ON" if s.recording else "OFF"}',14)
        self.text(p,565,940,f'STATUS: {"PAUSED" if s.paused else s.status}',14)
        self.text(p,55,968,'F2 SETTINGS   SPACE PAUSE   L LOCK   N NEXT   C CAPTURE   F1 HELP',11,False)
        if s.warning or s.message:
            text = s.message or s.warning
            for i in range(0,min(len(text),174),87):
                self.text(p,64,724+(i//87)*17,text[i:i+87],12)

    def draw_crt_effects(self, p):
        if not self.settings.crt:
            return
        p.save()
        p.setClipRect(QRectF(0,0,1000,1000))
        for x,y,alpha in self.grain:
            p.setPen(QColor(0,210,159,alpha))
            p.drawPoint(QPointF(x,y))
        p.setPen(QPen(QColor(0,0,0,22),.6))
        for y in range(0,1000,4):
            self.line(p,0,y,1000,y)
        p.restore()
