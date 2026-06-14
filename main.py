
import sys
sys.path.append('/content/project')

import cv2
import numpy as np
from tqdm import tqdm
from IPython.display import display, Image as IPImage

from tracker      import Tracker
from associator   import Associator
from fusion       import Fusion
from trajectory   import TrajectoryStore
from visualization import (draw_frame,
                           save_trajectory_plot)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
VIDEO_CAM1   = "/content/project/videos/campus4-c1.avi"
VIDEO_CAM2   = "/content/project/videos/campus4-c2.avi"
OUTPUT_VIDEO = "/content/project/output/fused_output.avi"
OUTPUT_PLOT  = "/content/project/output/trajectory_final.png"

FPS              = 15
START_SEC        = 0      # which second to start from
DURATION_SEC     = 25     # how many seconds to process
ASSOC_EVERY      = 10     # run CLIP association every N frames
CONF_THRESHOLD   = 0.35   # YOLO confidence threshold
SIM_THRESHOLD    = 0.72   # CLIP similarity threshold for matching

START_FRAME  = START_SEC  * FPS
END_FRAME    = START_FRAME + DURATION_SEC * FPS

# ─────────────────────────────────────────
# INIT MODULES
# ─────────────────────────────────────────
print("Initialising modules...")
# ByteTrack handles detection internally
# Two separate instances = two independent track ID spaces
tracker1    = Tracker(confidence_threshold=CONF_THRESHOLD)
tracker2    = Tracker(confidence_threshold=CONF_THRESHOLD)
associator  = Associator(similarity_threshold=SIM_THRESHOLD)
fusion      = Fusion()
traj_store  = TrajectoryStore()
print("All modules ready.")

# ─────────────────────────────────────────
# OPEN VIDEOS
# ─────────────────────────────────────────
cap1 = cv2.VideoCapture(VIDEO_CAM1)
cap2 = cv2.VideoCapture(VIDEO_CAM2)

# Seek to start frame
cap1.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
cap2.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

# Get frame dimensions
ret1, sample1 = cap1.read()
ret2, sample2 = cap2.read()

if not ret1 or not ret2:
    raise RuntimeError("Could not read from video files. Check paths.")

H1, W1 = sample1.shape[:2]
H2, W2 = sample2.shape[:2]
print(f"Cam1 resolution: {W1}x{H1}")
print(f"Cam2 resolution: {W2}x{H2}")

# Reset to start
cap1.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
cap2.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

# ─────────────────────────────────────────
# OUTPUT VIDEO SETUP
# Side-by-side: cam1 | cam2 | top-view
# All resized to same height for clean layout
# ─────────────────────────────────────────
PANEL_W    = 900
PANEL_H  = 540
TRAJ_H   = 300          # width of trajectory panel
OUT_W    = PANEL_W * 2   # total width
OUT_H    = PANEL_H

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (OUT_W, OUT_H))
print(f"Output video: {OUT_W}x{OUT_H} @ {FPS}fps")

# ─────────────────────────────────────────
# CROP STORE — accumulates crops per track
# cleared every ASSOC_EVERY frames after
# association runs
# ─────────────────────────────────────────
crops_cam1 = {}   # {track_id: [BGR crop, ...]}
crops_cam2 = {}

def extract_crop(frame, bbox, padding=10):
    """Crop person from frame with small padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

# ─────────────────────────────────────────
# LAST KNOWN ASSOCIATION — reused between
# CLIP runs to avoid flickering global IDs
# ─────────────────────────────────────────
last_matched    = []
last_unmatched1 = []
last_unmatched2 = []

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
frame_id      = 0
total_frames  = END_FRAME - START_FRAME

print(f"\nProcessing {total_frames} frames ({DURATION_SEC}s)...")

with tqdm(total=total_frames, desc="Processing") as pbar:

    while frame_id < total_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print(f"Video ended at frame {frame_id}")
            break

        # ── DETECTION + TRACKING (ByteTrack) ──
        tracks1 = tracker1.update(frame1, frame_id)
        tracks2 = tracker2.update(frame2, frame_id)

        # DEBUG
        ids1 = [t.track_id for t in tracks1]
        ids2 = [t.track_id for t in tracks2]

        if len(ids1) != len(set(ids1)):
          print("DUPLICATE IDS CAM1:", ids1)

        if len(ids2) != len(set(ids2)):
          print("DUPLICATE IDS CAM2:", ids2)

        # ── ACCUMULATE CROPS ───────────────
        for track in tracks1:
            crop = extract_crop(frame1, track.last_bbox)
            if crop is not None:
                if track.track_id not in crops_cam1:
                    crops_cam1[track.track_id] = []
                crops_cam1[track.track_id].append(crop)
                if len(crops_cam1[track.track_id]) > 20:
                    crops_cam1[track.track_id].pop(0)

        for track in tracks2:
            crop = extract_crop(frame2, track.last_bbox)
            if crop is not None:
                if track.track_id not in crops_cam2:
                    crops_cam2[track.track_id] = []
                crops_cam2[track.track_id].append(crop)
                if len(crops_cam2[track.track_id]) > 20:
                    crops_cam2[track.track_id].pop(0)

        # ── ASSOCIATION (every N frames) ────
        if frame_id % ASSOC_EVERY == 0:
            if len(tracks1) > 0 and len(tracks2) > 0:

              last_matched, last_unmatched1, last_unmatched2 =               associator.associate(
              tracks1,
              tracks2,
              crops_cam1,
              crops_cam2
              )
            else:
                last_matched = []
                last_unmatched1 = [t.track_id for t in tracks1]
                last_unmatched2 = [t.track_id for t in tracks2]

        # ── FUSION ─────────────────────────
        fused = fusion.fuse(
            tracks1, tracks2,
            last_matched,
            last_unmatched1,
            last_unmatched2
        )

        # ── TRAJECTORY ─────────────────────
        traj_store.update(fused, frame_id)

        # ── VISUALIZATION ──────────────────
        ann1 = draw_frame(frame1, tracks1, fused, traj_store, "CAM1")
        ann2 = draw_frame(frame2, tracks2, fused, traj_store, "CAM2")

        # Force both camera panels to occupy equal space


        ann1 = cv2.resize(ann1, (PANEL_W, PANEL_H))
        ann2 = cv2.resize(ann2, (PANEL_W, PANEL_H))


        top_row = np.hstack([ann1, ann2])

        traj_panel = cv2.resize(
                     traj_panel,
                     (top_row.shape[1], 300)
                     )

        combined = np.vstack([
                   top_row,
                   ])


        writer.write(combined)

        frame_id += 1
        pbar.update(1)

# ── CLEANUP ────────────────────────────────
cap1.release()
cap2.release()
writer.release()

print(f"\nVideo saved: {OUTPUT_VIDEO}")

# Save final trajectory plot
save_trajectory_plot(traj_store, OUTPUT_PLOT)

# Show final trajectory in Colab
display(IPImage(OUTPUT_PLOT))
print("\nDone.")
