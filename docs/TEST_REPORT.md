# AeroSentry test report

Tested on 2026-10-12 using Python 3.12.14, Linux x86-64, CPU-only PyTorch 2.6.0+cpu,
Ultralytics 8.3.221, OpenCV 4.11.0, NumPy 2.2.6 and PySide6 6.8.3.
Qt rendering tests used its offscreen platform.

**Result: 51 passed in 3.49 seconds.** See `test-results.txt`.

## Executed verification

- Drone class metadata normalization, explicit class IDs, generic-model incompatibility,
  wrong IDs, missing checkpoints, CPU and ByteTrack argument handling, unassigned IDs.
- **Actual pretrained YOLO11n CPU inference plus ByteTrack**, using the upstream bundled
  bus image. Tracked IDs persisted across successive frames. This is a general-object
  integration test, not an assessment of drone detection.
- **Actual video → OpenCV capture → YOLO/ByteTrack → main window → QPainter HUD** with
  an MJPG video generated from that image. Verified matching image dimensions, live
  mode, real latency, actual class labels, generic-model warning and worker shutdown.
- Raw-box coordinate preservation, EMA interpolation, image-space velocity, bounded
  trail history, ID loss, retention, same-ID reacquisition, lock stability, manual
  release, cycling and confidence-based selection.
- Aspect-preserving transforms for landscape, portrait and square source inputs.
  HUD render/capture at 1000×1000, 1280×720, 800×1000 and 1920×1080.
- Mocked camera disconnection/open failures and capture resource release; actual local
  video decoding, file timestamps and EOF handling; latest-frame mailbox boundedness.
- Model failure preventing camera startup; inference exception reporting; matching
  inference/source frame identity; simulation pause/resume/stop and simulation labeling.
- PNG capture, SQLite journal persistence, AVI encoding/decoding, settings validation,
  RTSP credential non-persistence, settings UI roundtrip, canvas shortcuts isolated
  from modal text entry, simulation window controls and clean normal shutdown.

## Visual review

The generated `hud-preview.png` was inspected against the supplied reference.
It preserves the square instrument composition, fine green outer frame, clock,
left/right readouts, top course reference, slanted reference line, raw acquisition
brackets, EMA circle, short trail and lower orientation scale. The primary surface
is painted by QPainter, not a collection of dashboard panels. Extra source/measurement
labels prevent decorative reference values being mistaken for sensor readings.

## Windows dependency check

PyPI metadata was checked for Windows x64 Python 3.12-compatible wheels for all
pinned top-level dependencies. See `windows-wheel-check.txt`. This checks distribution
availability, not execution on Windows. The installer selects the PyTorch CPU index
before installing the remaining requirements.

## Limits of this validation

- No physical webcam, live RTSP endpoint, Windows desktop session or CUDA GPU was
  available. Those hardware/OS paths require deployment-machine acceptance testing.
- No drone-trained checkpoint or annotated drone test footage was supplied. No drone
  precision/recall, detection range, low-light robustness or guaranteed FPS is claimed.
- General-purpose YOLO11n weights used for QA are excluded from the ZIP. They have no
  drone class. The app requires explicit class selection to use them for other objects.
- Default test runs skip the two optional real-model checks until a local `yolo11n.pt`
  path is provided via `AEROSENTRY_TEST_MODEL`; the other 49 tests require no weights.
- The simulation drone is deliberately synthetic; its appearance is not a detection
  result. Simulation has a persistent label even when the HUD overlay is hidden.
- A hung native camera driver may ignore cancellation. Normal cleanup was tested;
  the documented second-close process-exit fallback is not a hardware timeout guarantee.
