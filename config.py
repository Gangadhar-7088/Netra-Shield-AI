# NETRA-SHIELD configuration
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ALERT_DIR = PROJECT_ROOT / "data" / "alerts"
LOG_DIR = PROJECT_ROOT / "data"

# Camera: 0 is normally your laptop webcam. Try 1 if another camera is selected.
CAMERA_INDEX = 0

# Live inference settings
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.30
IOU = 0.45

# Faster laptop demo: 512. Increase to 640 on a faster machine.
IMG_SIZE = 512

# Zero-shot model: no custom training is required for the first demo.
# Ultralytics downloads the checkpoint the first time.
MODEL_NAME = "yoloe-11s-seg.pt"

# Prompt labels used by YOLOE.
PROMPTS = [
    "person",
    "helmet",
    "high visibility safety vest",
    "safety boot",
    "glove",
    "smoke",
    "fire",
]

# PPE policy for this factory zone.
# Set gloves to True when gloves are mandatory in your demo zone.
REQUIRED_PPE = {
    "helmet": True,
    "vest": True,
    "boots": True,
    "gloves": False,
}

# A PPE item must be spatially near the worker to count as worn.
# These values are intentionally broad for a webcam prototype.
PPE_REGION_PADDING = 0.08

# Alert logic
ALERT_CONFIRM_FRAMES = 6
ALERT_COOLDOWN_SECONDS = 8
FIRE_CONFIDENCE_BOOST = 0.40

# Zone model: based on the center of a person's bounding box.
ZONES = {
    "LEFT":  (0.00, 0.36),
    "CENTER": (0.36, 0.66),
    "RIGHT": (0.66, 1.00),
}

CAMERA_ID = "CAM-01"
SITE_NAME = "Factory Demo Floor"
