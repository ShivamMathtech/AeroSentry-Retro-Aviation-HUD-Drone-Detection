Put a trusted drone-trained Ultralytics object-detection checkpoint here, e.g. `drone.pt`.

Weights are **not included**. The example filename is not a download or a claim that a
particular checkpoint exists. This application does not auto-download weights.

In F2 Settings, select the actual file and click **Read checkpoint classes**. A checkpoint
with `drone`, `uav`, `uas`, `quadcopter`, or `unmanned aerial vehicle` class names is
recognized automatically (case-insensitive; spaces, underscores and hyphens normalized).
For other custom labels, explicitly select their numeric class IDs. A compatibility
warning remains when labels do not match this vocabulary; inspect your model's training
record before interpreting them as drones. A class name alone does not validate accuracy.

Generic COCO checkpoints have no drone class. Selecting `bird` or `airplane` does not
turn them into drone detectors. Explicit non-drone classes are supported for pipeline
experiments and clearly warned on the HUD.
