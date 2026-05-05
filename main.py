"""
Smart Warehouse — Helmet Detection System
- Class 0 = person
- Class 1 = helmet
- أخضر : helmet ✅
- أحمر  : person بدون helmet ❌
+ حفظ النتائج في CSV
"""

import cv2
import time
import csv
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# ══════════════════════════════════════════
#  ⚙️ CONFIG
# ══════════════════════════════════════════
MODEL_PATH  = "best.pt"
VIDEO_PATH  = "test1.mp4"
OUTPUT_PATH = "output.mp4"
CSV_PATH    = "helmet_log.csv"
CONF        = 0.4
SCALE       = 0.5

# ══════════════════════════════════════════
#  📌 CLASS MAPPING (مصحح)
# ══════════════════════════════════════════
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "head",
}

HELMET_IDS  = {1}
PERSON_IDS  = {0}
NO_HELM_IDS = {2, 3}

# ══════════════════════════════════════════
#  🎨 COLORS (BGR)
# ══════════════════════════════════════════
GREEN  = (0, 200, 80)
RED    = (0, 60, 220)
BLUE   = (220, 150, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 215, 255)
DARK   = (20, 20, 20)


# ══════════════════════════════════════════
#  📋 CSV SETUP
# ══════════════════════════════════════════
def init_csv(path):
    file_exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["date", "time", "persons_count",
                         "helmets_count", "violations_count", "status"])
    return f, writer


def log_row(csv_writer, persons, helmets, violations):
    now    = datetime.now()
    status = "SAFE" if violations == 0 else "VIOLATION"
    csv_writer.writerow([
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        persons,
        helmets,
        violations,
        status
    ])


# ══════════════════════════════════════════
#  🔲 DRAWING
# ══════════════════════════════════════════
def draw_box(frame, box, cls_id, conf, is_violation=False):
    x1, y1, x2, y2 = map(int, box)

    if is_violation:
        color = RED
    elif cls_id in HELMET_IDS:
        color = GREEN
    else:
        color = BLUE

    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))}  {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    if is_violation:
        cv2.putText(frame, "NO HELMET!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)


def draw_stats(frame, persons, helmets, violations, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Persons: {persons}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)
    cv2.putText(frame, f"Helmets: {helmets}", (170, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2)
    cv2.putText(frame, f"No Helmet: {violations}", (320, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)

    # شريط الحالة في الأسفل
    bar_color = RED if violations > 0 else GREEN
    status    = "  VIOLATION DETECTED" if violations > 0 else "  ALL SAFE"
    cv2.rectangle(frame, (0, h - 35), (w, h), bar_color, -1)
    cv2.putText(frame, status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


# ══════════════════════════════════════════
#  🚀 MAIN
# ══════════════════════════════════════════
def run():
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Classes: {model.names}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Can't open video!")
        return

    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    OUT_W   = int(fw * SCALE)
    OUT_H   = int(fh * SCALE)

    print(f"[INFO] {fw}x{fh}  →  {OUT_W}x{OUT_H}")

    writer = cv2.VideoWriter(OUTPUT_PATH,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_src, (OUT_W, OUT_H))

    csv_file, csv_writer = init_csv(CSV_PATH)
    print(f"[INFO] Logging to: {CSV_PATH}")

    prev         = time.time()
    log_interval = 3          # سجّل في CSV كل 3 ثواني
    last_log     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF, verbose=False)[0]

        persons_boxes = []
        helmets_boxes = []
        nohelm_boxes  = []

        for det in results.boxes:
            cls_id   = int(det.cls[0])
            conf_val = float(det.conf[0])
            box      = det.xyxy[0].tolist()

            if cls_id in PERSON_IDS:
                persons_boxes.append((box, conf_val))
            elif cls_id in HELMET_IDS:
                helmets_boxes.append((box, conf_val))
            elif cls_id in NO_HELM_IDS:
                nohelm_boxes.append((box, conf_val))

        # لو عدد الناس أكبر من عدد الطواقي = في violations
        violations = max(0, len(persons_boxes) - len(helmets_boxes))

        # ── رسم الطواقي (أخضر) ──
        for box, conf_val in helmets_boxes:
            draw_box(frame, box, 1, conf_val, is_violation=False)

        # ── رسم الناس ──
        # الأولين بعدد الطواقي = آمنين (أزرق) | الباقيين = violation (أحمر)
        for i, (box, conf_val) in enumerate(persons_boxes):
            is_viol = (i >= len(helmets_boxes))
            draw_box(frame, box, 0, conf_val, is_violation=is_viol)

        # ── no_helmet boxes لو موجودة ──
        for box, conf_val in nohelm_boxes:
            draw_box(frame, box, 2, conf_val, is_violation=True)

        # ── FPS ──
        now = time.time()
        fps = 1 / (now - prev + 1e-6)
        prev = now

        draw_stats(frame, len(persons_boxes), len(helmets_boxes), violations, fps)

        # ── CSV Logging كل 3 ثواني ──
        if now - last_log >= log_interval:
            log_row(csv_writer, len(persons_boxes), len(helmets_boxes), violations)
            csv_file.flush()
            last_log = now

        # ── Resize & Show ──
        frame = cv2.resize(frame, (OUT_W, OUT_H))
        cv2.imshow("Warehouse — Helmet Detection", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print("✅ Done!")
    print(f"📹 Video saved : {OUTPUT_PATH}")
    print(f"📋 CSV saved   : {CSV_PATH}")


if __name__ == "__main__":
    run()
