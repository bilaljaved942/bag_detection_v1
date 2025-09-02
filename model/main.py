import os
import cv2
import torch
import time
from ultralytics import YOLO
from groundingdino.util.inference import load_model, predict, annotate, Model

# --- Paths ---
VIDEO_PATH = "../evaluation_data/video1_2.mp4"
OUTPUT_DIR = "../output"
FRAMES_DIR = "../output_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "hybrid_detection.mp4")

# --- Load YOLO (for persons) ---
print("🔹 Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO loaded")

# --- Load GroundingDINO (for bags) ---
CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
device = "cpu"  # change to "cuda" if GPU available

print("🔹 Loading GroundingDINO model...")
start_time = time.time()
gdino_model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)
print(f"✅ GroundingDINO loaded in {time.time() - start_time:.2f}s")

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"❌ Cannot open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# --- Detection ---
TEXT_PROMPT = "bag"
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_start = time.time()
    print(f"▶ Processing frame {frame_count}...")

    # 1. --- YOLO person detection ---
    results = yolo_model(frame, classes=[0])  # class 0 = person
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Person {score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 2. --- GroundingDINO bag detection ---
    image_tensor = Model.preprocess_image(frame)  # ✅ use BGR frame
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino_model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )

    # Annotate DINO results on the same frame
    annotated_frame = annotate(frame, boxes, logits, phrases)

    # Save annotated frame as image
    frame_filename = os.path.join(FRAMES_DIR, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, annotated_frame)

    # Write annotated frame to output video
    out.write(annotated_frame)

    print(f"✔ Frame {frame_count} processed in {time.time() - frame_start:.2f}s")

cap.release()
out.release()
print(f"✅ Processed {frame_count} frames, saved video at {OUTPUT_VIDEO_PATH} and frames in {FRAMES_DIR}")
