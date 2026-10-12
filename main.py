"""Run from this directory with Python 3.12. No camera is opened in simulation."""
import argparse
import os
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='AeroSentry visual drone tracker')
    parser.add_argument('--simulation',action='store_true',help='force visibly labeled synthetic HUD mode')
    parser.add_argument('--live',action='store_true',help='force real input (requires weights)')
    parser.add_argument('--model',help='trusted local .pt checkpoint')
    parser.add_argument('--source',help='camera index, video path or RTSP URL')
    parser.add_argument('--classes',help='comma-separated checkpoint class IDs')
    parser.add_argument('--device',choices=['cpu','0'])
    parser.add_argument('--screenshot',help='render simulation screenshot to a local PNG and exit')
    parser.add_argument('--data-dir',type=Path,help='override user settings/output directory')
    args=parser.parse_args()
    if args.screenshot:
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication,QMessageBox
    from config import Settings,DATA_DIR
    from ui.main_window import MainWindow
    app=QApplication(sys.argv[:1]); app.setApplicationName('AeroSentry')
    data=args.data_dir or DATA_DIR
    try: settings=Settings.load(data/'settings.json')
    except (ValueError,TypeError,OSError) as exc:
        settings=Settings()
        QMessageBox.warning(None,'Settings reset','Could not load settings. Using simulation defaults.\n'+str(exc))
    if args.model: settings.model_path=args.model
    if args.classes is not None: settings.class_ids=args.classes
    if args.device: settings.device=args.device
    if args.source:
        settings.source=args.source
        settings.source_kind='Camera' if args.source.isdigit() else 'RTSP' if args.source.lower().startswith(('rtsp://','rtsps://')) else 'Video'
    if args.live or args.model or args.source: settings.simulation=False
    if args.simulation or args.screenshot: settings.simulation=True
    try: settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    window=MainWindow(settings,data_dir=data); window.show()
    if args.screenshot:
        window.resize(1000,1000)
        def save():
            path=Path(args.screenshot); path.parent.mkdir(parents=True,exist_ok=True)
            if not window.hud.grab().save(str(path),'PNG'):
                print('Screenshot failed',file=sys.stderr)
            window.close()
        QTimer.singleShot(1500,save)
    return app.exec()

if __name__=='__main__':
    raise SystemExit(main())
