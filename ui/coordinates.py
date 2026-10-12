"""Pure aspect-preserving transforms used by video and every target graphic."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Viewport:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def fit(cls, bounds, source_size):
        x, y, w, h = bounds
        sw, sh = source_size
        if min(sw, sh, w, h) <= 0:
            raise ValueError('All dimensions must be positive')
        scale = min(w / sw, h / sh)
        fw, fh = sw * scale, sh * scale
        return cls(x + (w - fw) / 2, y + (h - fh) / 2, fw, fh)

    def point(self, point, source_size):
        return (self.x + point[0] / source_size[0] * self.width,
                self.y + point[1] / source_size[1] * self.height)

    def box(self, box, source_size):
        x1, y1 = self.point(box[:2], source_size)
        x2, y2 = self.point(box[2:], source_size)
        return x1, y1, x2-x1, y2-y1
