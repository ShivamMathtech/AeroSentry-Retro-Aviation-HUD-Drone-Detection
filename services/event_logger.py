"""A bounded, asynchronous SQLite event journal; never stores stream URLs."""
import json
import queue
import sqlite3
import threading
from datetime import datetime,timezone
from pathlib import Path

class EventLogger(threading.Thread):
    def __init__(self,path):
        super().__init__(name='AeroSentry-journal',daemon=True)
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=512)
        self.stop_event=threading.Event()
        self.error=''; self.dropped=0
        self.start()
    def log(self,kind,**payload):
        try: self.queue.put_nowait((datetime.now(timezone.utc).isoformat(),kind,json.dumps(payload)))
        except queue.Full: self.dropped+=1
    def run(self):
        try:
            with sqlite3.connect(self.path,timeout=2) as conn:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, utc TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)')
                conn.commit()
                while not self.stop_event.is_set() or not self.queue.empty():
                    try: row=self.queue.get(timeout=.1)
                    except queue.Empty: continue
                    conn.execute('INSERT INTO events (utc,kind,payload) VALUES (?,?,?)',row)
                    conn.commit()
        except Exception as exc: self.error=str(exc)
    def close(self): self.stop_event.set()
    def recent(self,limit=200):
        try:
            with sqlite3.connect(self.path,timeout=.1) as conn:
                return conn.execute('SELECT utc,kind,payload FROM events ORDER BY id DESC LIMIT ?',(limit,)).fetchall()
        except sqlite3.Error: return []
