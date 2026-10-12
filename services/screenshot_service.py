from pathlib import Path
from datetime import datetime

class ScreenshotService:
    def __init__(self,directory): self.directory=Path(directory)
    def capture(self,widget):
        self.directory.mkdir(parents=True,exist_ok=True)
        path=self.directory / f'AeroSentry_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
        if not widget.grab().save(str(path),'PNG'):
            raise OSError('Unable to save screenshot')
        return path
