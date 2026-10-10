from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


def xywh_to_xyxy(xywh):
    x, y, w, h = [float(v) for v in xywh]
    return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)


def parse_result(result) -> List[Detection]:
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box, conf, cls_id in zip(
        result.boxes.xywh.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
    ):
        label = str(names[int(cls_id)])
        detections.append(
            Detection(label=label, confidence=float(conf), box=xywh_to_xyxy(box))
        )
    return detections


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def zone_for_box(box, frame_width, zones):
    cx, _ = center(box)
    ratio = max(0.0, min(1.0, cx / max(frame_width, 1)))
    for name, (lo, hi) in zones.items():
        if lo <= ratio <= hi:
            return name
    return "CENTER"


def near_worker(item_box, worker_box, padding=0.08):
    """
    Prototype spatial association:
    PPE center must lie inside an expanded worker box.
    This is intentionally simple for a webcam demo.
    """
    px, py = center(item_box)
    x1, y1, x2, y2 = worker_box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    ex1 = x1 - padding * w
    ey1 = y1 - padding * h
    ex2 = x2 + padding * w
    ey2 = y2 + padding * h
    return ex1 <= px <= ex2 and ey1 <= py <= ey2


def ppe_status_for_worker(worker_box, detections, required):
    """
    Match detected PPE to a worker and return item presence.
    For boots we look for either boot to be inside the lower half;
    for helmets/gloves/vests, containment is enough for the demo.
    """
    x1, y1, x2, y2 = worker_box
    _, ph = y2 - y1, y2 - y1

    found = {"helmet": False, "vest": False, "boots": False, "gloves": False}

    for det in detections:
        if det.label not in {"helmet", "high visibility safety vest", "safety boot", "glove"}:
            continue
        if not near_worker(det.box, worker_box):
            continue

        cx, cy = center(det.box)
        rel_y = (cy - y1) / max(ph, 1)

        if det.label == "helmet" and rel_y < 0.45:
            found["helmet"] = True
        elif det.label == "high visibility safety vest" and 0.15 < rel_y < 0.80:
            found["vest"] = True
        elif det.label == "safety boot" and rel_y > 0.55:
            found["boots"] = True
        elif det.label == "glove" and 0.20 < rel_y < 0.85:
            found["gloves"] = True

    missing = [
        item for item, mandatory in required.items()
        if mandatory and not found[item]
    ]
    compliant = len(missing) == 0
    return found, missing, compliant


def fire_or_smoke(detections, fire_boost=0.40):
    fire = [d for d in detections if d.label == "fire" and d.confidence >= fire_boost]
    smoke = [d for d in detections if d.label == "smoke"]
    return fire, smoke
