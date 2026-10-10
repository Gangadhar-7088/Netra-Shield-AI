import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

from config import (
    ALERT_CONFIRM_FRAMES,
    ALERT_COOLDOWN_SECONDS,
    ALERT_DIR,
    CAMERA_ID,
    CAMERA_INDEX,
    CONFIDENCE,
    FIRE_CONFIDENCE_BOOST,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IOU,
    IMG_SIZE,
    MODEL_NAME,
    PROMPTS,
    REQUIRED_PPE,
    SITE_NAME,
    ZONES,
)
from alerts import AlertManager
from vision import parse_result, ppe_status_for_worker, fire_or_smoke, zone_for_box


WINDOW = "NETRA-SHIELD | AI Safety Command Center"

# Fixed dashboard size so the text is easy to read.
DISPLAY_W = 1280
DISPLAY_H = 720
SIDEBAR_W = 360
HEADER_H = 88
FEED_W = DISPLAY_W - SIDEBAR_W
FEED_H = DISPLAY_H - HEADER_H

# UI colors
BG = (28, 28, 28)
PANEL = (42, 42, 42)
CARD = (53, 53, 53)
WHITE = (245, 245, 245)
MUTED = (180, 180, 180)
GOOD = (70, 190, 95)
BAD = (70, 70, 230)
DEMO = (65, 55, 25)


def txt(img, text, x, y, size=0.60, color=WHITE, thickness=1):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fill(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)


def rect(img, x1, y1, x2, y2, color, thickness=1):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def fit_frame(frame, target_w, target_h):
    """Resize and letterbox the webcam image into a fixed dashboard area."""
    h, w = frame.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def wrapped_lines(text, max_chars=30):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_detection(img, box_xyxy, label, confidence, alert=False):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    color = BAD if alert else WHITE
    rect(img, x1, y1, x2, y2, color, 2)

    tag = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(
        tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )

    fill(img, x1, max(0, y1 - th - 8), x1 + tw + 8, y1, (35, 35, 35))
    txt(img, tag, x1 + 4, y1 - 5, 0.45, WHITE, 1)


def draw_header(canvas):
    fill(canvas, 0, 0, DISPLAY_W, HEADER_H, (33, 33, 33))

    txt(canvas, "NETRA-SHIELD", 24, 35, 0.95, WHITE, 2)
    txt(canvas, "AI SAFETY COMMAND CENTER", 26, 61, 0.48, MUTED, 1)

    # Right-side live status
    fill(canvas, DISPLAY_W - 196, 18, DISPLAY_W - 24, 55, (48, 48, 48))
    txt(canvas, "AI LIVE", DISPLAY_W - 165, 42, 0.55, GOOD, 2)

    now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    txt(canvas, now, DISPLAY_W - 330, 73, 0.42, MUTED, 1)


def metric_card(canvas, x, y, width, label, value, value_color):
    fill(canvas, x, y, x + width, y + 70, CARD)
    txt(canvas, label, x + 14, y + 24, 0.43, MUTED, 1)
    txt(canvas, value, x + 14, y + 56, 0.78, value_color, 2)


def make_event(incident_type, severity, zone, details):
    return {
        "severity": severity,
        "incident_type": incident_type,
        "camera": CAMERA_ID,
        "zone": zone,
        "site": SITE_NAME,
        "details": details,
    }


def build_dashboard(
    frame,
    workers_count,
    compliant_count,
    violation_count,
    smoke_count,
    fire_count,
    fps,
    infer_ms,
    gloves_required,
    alerts,
    demo_banner=False,
):
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    canvas[:] = BG

    draw_header(canvas)

    # ---------------- LIVE CAMERA ----------------
    live = fit_frame(frame, FEED_W - 18, FEED_H - 62)

    x0 = 9
    y0 = HEADER_H + 9

    canvas[y0:y0 + live.shape[0], x0:x0 + live.shape[1]] = live

    fill(
        canvas,
        20,
        DISPLAY_H - 42,
        FEED_W - 18,
        DISPLAY_H - 18,
        (35, 35, 35),
    )

    txt(
        canvas,
        f"{SITE_NAME}   |   {CAMERA_ID}",
        30,
        DISPLAY_H - 25,
        0.42,
        WHITE,
        1,
    )

    # ---------------- RIGHT SIDEBAR ----------------
    sx = FEED_W
    fill(canvas, sx, HEADER_H, DISPLAY_W, DISPLAY_H, (37, 37, 37))

    txt(canvas, "LIVE MONITOR", sx + 20, HEADER_H + 34, 0.58, WHITE, 2)

    card_x = sx + 20
    card_w = SIDEBAR_W - 40
    y = HEADER_H + 50

    metric_card(canvas, card_x, y, card_w, "WORKERS", workers_count, WHITE)
    y += 82

    metric_card(
        canvas,
        card_x,
        y,
        card_w,
        "PPE COMPLIANT",
        compliant_count,
        GOOD,
    )
    y += 82

    metric_card(
        canvas,
        card_x,
        y,
        card_w,
        "PPE VIOLATIONS",
        violation_count,
        BAD if violation_count else GOOD,
    )
    y += 82

    metric_card(
        canvas,
        card_x,
        y,
        card_w,
        "SMOKE / FIRE",
        f"{smoke_count} / {fire_count}",
        BAD if (smoke_count or fire_count) else GOOD,
    )
    y += 82

    # System information
    fill(canvas, card_x, y, DISPLAY_W - 20, y + 111, PANEL)

    txt(canvas, "SYSTEM", card_x + 14, y + 27, 0.46, WHITE, 1)
    txt(canvas, f"FPS: {fps:.1f}", card_x + 14, y + 52, 0.43, MUTED, 1)
    txt(canvas, f"AI inference: {infer_ms:.0f} ms",
        card_x + 14, y + 74, 0.43, MUTED, 1)
    txt(
        canvas,
        f"Gloves required: {'YES' if gloves_required else 'NO'}",
        card_x + 14,
        y + 96,
        0.43,
        MUTED,
        1,
    )

    y += 123

    # Latest alert
    fill(
        canvas,
        card_x,
        y,
        DISPLAY_W - 20,
        y + 145,
        (57, 35, 35) if alerts else PANEL,
    )

    txt(canvas, "LATEST ALERT", card_x + 14, y + 27, 0.46, WHITE, 1)

    if alerts:
        latest = alerts[-1]

        txt(
            canvas,
            latest.get("incident_type", "UNKNOWN"),
            card_x + 14,
            y + 56,
            0.50,
            WHITE,
            2,
        )

        txt(
            canvas,
            f"Zone: {latest.get('zone', '-')}",
            card_x + 14,
            y + 80,
            0.44,
            MUTED,
            1,
        )

        lines = wrapped_lines(latest.get("details", ""), 29)

        for i, line in enumerate(lines[:2]):
            txt(
                canvas,
                line,
                card_x + 14,
                y + 104 + i * 18,
                0.39,
                MUTED,
                1,
            )
    else:
        txt(
            canvas,
            "No active alerts",
            card_x + 14,
            y + 60,
            0.52,
            GOOD,
            2,
        )

        txt(
            canvas,
            "System is monitoring the camera.",
            card_x + 14,
            y + 88,
            0.39,
            MUTED,
            1,
        )

    # Bottom controls
    fill(
        canvas,
        card_x,
        DISPLAY_H - 103,
        DISPLAY_W - 20,
        DISPLAY_H - 20,
        PANEL,
    )

    txt(canvas, "CONTROLS", card_x + 14, DISPLAY_H - 77, 0.45, WHITE, 1)
    txt(
        canvas,
        "Q Quit   S Snapshot   G Gloves",
        card_x + 14,
        DISPLAY_H - 53,
        0.37,
        MUTED,
        1,
    )
    txt(
        canvas,
        "D Demo Alert   R Clear",
        card_x + 14,
        DISPLAY_H - 31,
        0.37,
        MUTED,
        1,
    )

    if demo_banner:
        fill(canvas, 24, 104, FEED_W - 24, 145, DEMO)
        txt(
            canvas,
            "DEMO MODE - SIMULATED EVENT",
            42,
            132,
            0.52,
            WHITE,
            2,
        )

    return canvas


def main():
    print("Loading NETRA-SHIELD...")
    print(f"Model: {MODEL_NAME}")
    print("First run may download the model/text encoder.")

    model = YOLOE(MODEL_NAME)
    model.set_classes(PROMPTS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Try CAMERA_INDEX = 1 in config.py "
            "or close Camera/Teams/Zoom/Meet first."
        )

    alert_manager = AlertManager(
        ALERT_DIR,
        confirm_frames=ALERT_CONFIRM_FRAMES,
        cooldown_seconds=ALERT_COOLDOWN_SECONDS,
    )

    alerts = []

    fps_clock = time.time()
    frame_counter = 0
    fps = 0.0

    gloves_required = REQUIRED_PPE["gloves"]
    demo_banner = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISPLAY_W, DISPLAY_H)

    print("\nControls:")
    print("  Q = quit")
    print("  S = save dashboard screenshot")
    print("  G = toggle glove requirement")
    print("  D = clearly-labelled demo alert")
    print("  R = clear visible alert\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        t0 = time.time()

        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMG_SIZE,
            verbose=False,
        )[0]

        infer_ms = (time.time() - t0) * 1000

        detections = parse_result(result)

        workers = [d for d in detections if d.label == "person"]

        fire_dets, smoke_dets = fire_or_smoke(
            detections,
            FIRE_CONFIDENCE_BOOST,
        )

        # Draw normal detections.
        for det in detections:
            draw_detection(
                frame,
                det.box,
                det.label,
                det.confidence,
                alert=(det.label in {"fire", "smoke"}),
            )

        active_event_keys = set()

        compliant_count = 0
        violation_count = 0

        current_required = dict(REQUIRED_PPE)
        current_required["gloves"] = gloves_required

        # ---------------- PPE ----------------
        for worker_index, worker in enumerate(workers, start=1):
            zone = zone_for_box(worker.box, frame_w, ZONES)

            found, missing, compliant = ppe_status_for_worker(
                worker.box,
                detections,
                current_required,
            )

            x1, y1, x2, y2 = worker.box

            if compliant:
                compliant_count += 1
                status = "COMPLIANT"
                status_color = GOOD
            else:
                violation_count += 1
                status = "PPE VIOLATION"
                status_color = BAD

                event_key = (
                    f"ppe:{zone}:{worker_index}:{','.join(missing)}"
                )

                active_event_keys.add(event_key)

                event = make_event(
                    "PPE NON-COMPLIANCE",
                    "HIGH",
                    zone,
                    f"Worker {worker_index} missing: {', '.join(missing)}",
                )

                created = alert_manager.update(
                    event_key,
                    event,
                    frame,
                )

                if created:
                    alerts.append(created)

            # Worker box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                status_color,
                2,
            )

            txt(
                frame,
                f"W{worker_index} | {zone} | {status}",
                x1,
                max(22, y1 - 12),
                0.50,
                status_color,
                2,
            )

            item_text = (
                f"H:{'Y' if found['helmet'] else 'N'}  "
                f"V:{'Y' if found['vest'] else 'N'}  "
                f"B:{'Y' if found['boots'] else 'N'}  "
                f"G:{'Y' if found['gloves'] else 'N'}"
            )

            txt(
                frame,
                item_text,
                x1,
                min(frame_h - 12, y2 + 20),
                0.44,
                WHITE,
                1,
            )

        # ---------------- FIRE ----------------
        for det in fire_dets:
            zone = zone_for_box(det.box, frame_w, ZONES)

            event_key = f"fire:{zone}"
            active_event_keys.add(event_key)

            event = make_event(
                "FIRE DETECTED",
                "CRITICAL",
                zone,
                f"AI confidence {det.confidence:.2f}; immediate verification required.",
            )

            created = alert_manager.update(event_key, event, frame)

            if created:
                alerts.append(created)

        # ---------------- SMOKE ----------------
        for det in smoke_dets:
            zone = zone_for_box(det.box, frame_w, ZONES)

            event_key = f"smoke:{zone}"
            active_event_keys.add(event_key)

            event = make_event(
                "SMOKE DETECTED",
                "CRITICAL",
                zone,
                f"AI confidence {det.confidence:.2f}; investigate early hazard sign.",
            )

            created = alert_manager.update(event_key, event, frame)

            if created:
                alerts.append(created)

        alert_manager.decay_missing(active_event_keys)

        # ---------------- DEMO ALERT ----------------
        if demo_banner:
            event = make_event(
                "SIMULATED FIRE ALERT",
                "DEMO",
                "CENTER",
                "Simulation only - demonstrates the alert escalation interface.",
            )

            alerts.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event_id": "DEMO-001",
                **event,
                "evidence_file": "not_saved_simulation",
            })

            # Only show it for this dashboard cycle.
            demo_banner = False

        # FPS
        frame_counter += 1

        elapsed = time.time() - fps_clock

        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            fps_clock = time.time()

        # Build clear dashboard.
        dashboard = build_dashboard(
            frame=frame,
            workers_count=len(workers),
            compliant_count=compliant_count,
            violation_count=violation_count,
            smoke_count=len(smoke_dets),
            fire_count=len(fire_dets),
            fps=fps,
            infer_ms=infer_ms,
            gloves_required=gloves_required,
            alerts=alerts,
            demo_banner=False,
        )

        cv2.imshow(WINDOW, dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("s"), ord("S")):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(ALERT_DIR) / f"manual_snapshot_{stamp}.jpg"
            cv2.imwrite(str(path), dashboard)
            print(f"Snapshot saved: {path}")

        elif key in (ord("g"), ord("G")):
            gloves_required = not gloves_required
            print(f"Gloves mandatory: {gloves_required}")

        elif key in (ord("d"), ord("D")):
            demo_banner = True

        elif key in (ord("r"), ord("R")):
            alerts.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
