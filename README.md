# NETRA-SHIELD

## Meaningful name

**NETRA-SHIELD** = **N**etworked **E**vent **T**racking & **R**isk **A**wareness — **S**mart **H**azard **I**dentification, **E**arly **L**ocalized **D**etection.

The name reflects the project's core idea: use a camera as an AI "eye" that identifies safety risks and turns them into localized, time-stamped alerts.

## What the prototype does

1. Opens the laptop webcam.
2. Uses Ultralytics YOLOE open-vocabulary detection.
3. Detects workers plus helmet, high-visibility vest, safety boot, glove, smoke, and fire prompts.
4. Associates PPE detections with nearby workers using their bounding boxes.
5. Checks the site's PPE policy.
6. Splits the camera view into LEFT/CENTER/RIGHT zones.
7. Generates persistent alerts only after the event survives several frames.
8. Saves alert evidence images and JSON records.
9. Shows a command-center style live dashboard.
10. Allows a clearly labelled jury simulation with the D key.

## Important prototype note

The zero-shot model is a fast-start demonstration path. It is not a production safety certification system. For stronger accuracy, fine-tune custom PPE and fire/smoke models on images from the actual workplace conditions, camera angle, lighting, PPE designs, and hazard scenarios.

## Project structure

```
netra_shield/
├── main.py
├── config.py
├── alerts.py
├── vision.py
├── train_ppe.py
├── train_hazard.py
├── requirements.txt
├── datasets_example/
│   ├── ppe/data.yaml
│   └── hazard/data.yaml
├── weights/
└── data/
    └── alerts/
```

## Windows setup

Recommended for an easy hackathon setup: Python 3.11.

1. Install Python 3.11.
2. Open PowerShell in this folder.
3. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

4. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

5. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

6. Install dependencies:

```powershell
pip install -r requirements.txt
```

7. Start the live system:

```powershell
python main.py
```

The first run can download the YOLOE model and the text encoder. Internet access is therefore useful for the first startup.

## Camera troubleshooting

If your laptop camera does not open:
- Change `CAMERA_INDEX = 0` to `CAMERA_INDEX = 1` in `config.py`.
- Close Teams/Zoom/Meet/Camera if one of them is using the webcam.
- Check Windows Settings -> Privacy & security -> Camera and allow desktop apps to access the camera.

## What to show the jury

Before the demo:
- Put yourself in front of the laptop camera.
- Wear a visible helmet and a bright safety vest if available.
- Keep the upper body and shoes visible enough for the camera.
- Start `python main.py`.

During the live demo:
- First show the normal dashboard and the worker status.
- Toggle gloves with `G` to show that the safety policy is configurable.
- Move out of the required PPE state to trigger a real PPE warning when the model recognizes the missing equipment.
- Point to the LEFT/CENTER/RIGHT zone label to explain location-specific alerts.
- Press `D` to demonstrate the alert escalation interface even if the live camera does not produce a smoke/fire event. The banner explicitly says "SIMULATED" so the demo remains honest.
- Press `S` to save a screenshot.
- Open `data/alerts/alerts.csv` after the demo to show the audit trail.

## Stronger hackathon version: custom training

For better PPE accuracy, create or export a YOLO-format dataset.

PPE classes:
- person
- helmet
- high visibility safety vest
- safety boot
- glove

Hazard classes:
- smoke
- fire

Example YOLO dataset layout:

```
ppe_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Then:

```powershell
python train_ppe.py --data C:\path\to\ppe_dataset\data.yaml
```

And for smoke/fire:

```powershell
python train_hazard.py --data C:\path\to\hazard_dataset\data.yaml
```

The training scripts start from a pretrained YOLO26 nano checkpoint and fine-tune on your dataset.

## How to use your own trained models

For a production-style architecture, use two separate custom models:
- `weights/ppe_best.pt`
- `weights/hazard_best.pt`

Then replace the single YOLOE call in `main.py` with two YOLO model objects and process the two result sets independently. The current code intentionally uses one zero-shot model so the team can demo without waiting for custom training.

## Architecture

```
Laptop Camera
      |
      v
Frame Capture (OpenCV)
      |
      v
YOLOE Vision Layer
      |
      +--> Worker detections
      |       |
      |       v
      |   PPE association
      |       |
      |       v
      |   Policy engine
      |
      +--> Smoke / Fire detections
      |
      v
Zone Engine: LEFT / CENTER / RIGHT
      |
      v
Alert Debouncer
      |
      +--> Live screen banner
      +--> Evidence JPG
      +--> JSON incident record
      +--> CSV audit log
```

## Why the alert is not fired on one frame

A webcam detector can briefly miss an object. This prototype therefore waits for an event to persist for several frames before making an actual alert. That reduces noisy frame-to-frame alarms.

## Future extensions

- Polygon-based factory zones instead of LEFT/CENTER/RIGHT.
- Person tracking IDs for stable worker identity across frames.
- Separate models for PPE and hazards.
- Email/SMS/WhatsApp/IoT siren integration.
- Multi-camera support.
- FastAPI backend + React dashboard.
- ONNX/TensorRT deployment.
- Edge deployment on Jetson/Raspberry Pi + accelerator.
- Role-based dashboards for supervisor/security/control room.

## Suggested pitch

"NETRA-SHIELD turns an ordinary CCTV camera into an AI safety monitor. It detects workers and required PPE, identifies smoke and fire as early hazards, converts camera coordinates into meaningful factory zones, and creates evidence-backed alerts instead of relying only on manual observation."

## Ethics / safety

This is a hackathon prototype. Do not use it as the sole safety control for a real factory. Real deployments need site-specific validation, human verification, fail-safe design, documented thresholds, privacy controls, and compliance review.

## License note

Ultralytics documentation currently lists AGPL-3.0 and Enterprise licensing options. Review the applicable Ultralytics license before using the system in a commercial deployment.
