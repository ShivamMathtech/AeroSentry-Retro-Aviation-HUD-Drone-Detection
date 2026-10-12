import queue
import threading
from pathlib import Path
import cv2
import numpy as np

class Recorder(threading.Thread):
    """Annotated HUD recording at a declared 10 FPS; bounded queue drops under overload."""
    def __init__(self,path,size=(1000,1000),fps=10):
        super().__init__(name='AeroSentry-recorder',daemon=True)
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.size,self.fps=size,fps
        self.queue=queue.Queue(maxsize=3)
        self.stop_event=threading.Event(); self.error=''; self.dropped=0; self.written=0
        self.start()
    def put(self,frame):
        try: self.queue.put_nowait(frame)
        except queue.Full: self.dropped+=1
    def close(self): self.stop_event.set()
    def run(self):
        writer=None
        try:
            writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
            if not writer.isOpened(): raise OSError('MJPG writer unavailable or output directory not writable')
            while not self.stop_event.is_set() or not self.queue.empty():
                try: frame=self.queue.get(timeout=.1)
                except queue.Empty: continue
                h,w=frame.shape[:2]
                scale=min(self.size[0]/w,self.size[1]/h)
                fw,fh=max(1,round(w*scale)),max(1,round(h*scale))
                canvas=np.zeros((self.size[1],self.size[0],3),np.uint8)
                x,y=(self.size[0]-fw)//2,(self.size[1]-fh)//2
                canvas[y:y+fh,x:x+fw]=cv2.resize(frame,(fw,fh))
                writer.write(canvas); self.written+=1
        except Exception as exc: self.error=str(exc)
        finally:
            if writer is not None: writer.release()
