# Multi-Camera Object Tracking and Trajectory Fusion

> A complete proof-of-concept multi-camera tracking and trajectory fusion pipeline built during a 2-day hackathon using only pretrained models, no custom training, and no GPU requirement.

---

## Demo Output




https://github.com/user-attachments/assets/b8653f5e-68ef-458b-93ad-278cc5a41f4b

<img width="1200" height="1200" alt="trajec" src="https://github.com/user-attachments/assets/944e75f4-1707-4c26-a94b-48a5520036e7" />

> Annotated dual-camera output with global IDs and fused trajectory plot.

---

## Problem Statement

Given two synchronized video feeds from cameras at different angles, the system must:

1. Detect people in each camera independently
2. Track them over time within each camera
3. Associate the same person across both cameras
4. Assign a single unified global identity per person
5. Generate fused 2D trajectory visualizations

This addresses a core limitation of single-camera systems — occlusions, identity switches, and incomplete scene coverage — by fusing information from two viewpoints.

---

## System Architecture
Video Cam1 ──→ YOLO Detection ──→ ByteTrack ──→ Crops ──┐
├──→ CLIP Embeddings
Video Cam2 ──→ YOLO Detection ──→ ByteTrack ──→ Crops ──┘        │
▼
Cosine Similarity Matrix
│
▼
Hungarian Optimal Assignment
│
▼
Global ID Creation (Fusion)
│
▼
Trajectory Storage & Visualization

---

## Module Breakdown

| File | Responsibility |
|---|---|
| `detector.py` | YOLOv8n person detection with confidence filtering |
| `tracker.py` | Per-camera ByteTrack with Kalman filter for stable IDs |
| `associator.py` | CLIP embeddings + cosine similarity + Hungarian matching |
| `fusion.py` | Global ID assignment and cross-camera identity mapping |
| `trajectory.py` | Position history storage per global ID |
| `visualization.py` | Annotated video output and matplotlib trajectory plot |
| `main.py` | End-to-end pipeline orchestration |

---

## Dataset

**EPFL Multi-Camera Pedestrian Dataset**

- Source: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
- Two camera views at approximately 90 degree separation
- Multiple people walking through the scene
- Moderate occlusion
- Same people visible across both cameras
- No custom annotations created — used as-is

---

## Tech Stack

| Component | Tool |
|---|---|
| Detection | YOLOv8n (Ultralytics) |
| Tracking | ByteTrack (built into Ultralytics) |
| Appearance Embeddings | CLIP ViT-B/32 (OpenAI) |
| Assignment | Hungarian Algorithm (scipy) |
| Language | Python |
| Environment | Google Colab (CPU) |

---

## Installation

```bash
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
pip install lap scipy
```

---

## Usage

1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-camera-tracking.git
cd multi-camera-tracking
```

2. Place your two video files inside a `videos/` folder

3. Update these lines in `main.py` to point to your videos

```python
VIDEO_CAM1 = "videos/your_cam1_video.avi"
VIDEO_CAM2 = "videos/your_cam2_video.avi"
```

4. Run the pipeline

```bash
python main.py
```

5. Find outputs in the `output/` folder
output/
├── fused_output.avi        # annotated dual-camera video
└── trajectory_final.png    # 2D trajectory plot

---

## Methodology

### 1. Detection
YOLOv8n pretrained on COCO dataset, filtered to person class only. Confidence threshold set to 0.50 to minimize false positives.

### 2. Single-Camera Tracking
ByteTrack runs independently on each camera stream. It uses a Kalman filter for motion prediction and Hungarian algorithm for frame-to-frame assignment. This gives each person a stable per-camera ID even through brief occlusions.

### 3. Cross-Camera Association
This is the core of the system. For each active track:
- Up to 20 person crops are collected over time
- Each crop is passed through CLIP ViT-B/32 to get a 512-dimensional appearance embedding
- Embeddings are averaged into one representative vector per track
- A cosine similarity matrix is built between all Camera 1 tracks and all Camera 2 tracks
- The Hungarian algorithm finds the globally optimal matching
- Pairs above the similarity threshold (0.68) are accepted as the same physical person

### 4. Fusion
Matched pairs across cameras are assigned a single global ID. Unmatched tracks remain camera-specific but still receive a global ID. All mappings persist across frames to prevent ID flickering.

### 5. Trajectory
For every global ID, the system records (frame, x, y) each frame. These are visualized as trailing lines on the output video and as a clean 2D path plot at the end.

---

## Project Structure
multi-camera-tracking/
├── detector.py
├── tracker.py
├── associator.py
├── fusion.py
├── trajectory.py
├── visualization.py
├── main.py
├── videos/
│   ├── cam1.avi
│   └── cam2.avi
├── output/
│   ├── fused_output.avi
│   └── trajectory_final.png
└── README.md

---

## Results

- End-to-end pipeline runs successfully on dual camera input
- Multi-person detection and tracking operational across both cameras
- Cross-camera association assigns consistent global IDs using appearance
- Fused 2D trajectory generated from start to end for each person
- No camera calibration required

---

## Known Limitations

1. Identity switches can occur when two people walk very close together
2. Re-identification may fail after long occlusions when appearance changes
3. Similar-looking people wearing similar clothing can be confused
4. No geometric calibration — association is appearance-only
5. Trajectory fragmentation can occur if a track is lost and restarted

These are expected limitations of a calibration-free appearance-only system and are well understood in the multi-camera tracking literature.

---

## Future Work

1. Camera calibration and homography estimation for geometric consistency
2. Dedicated ReID models such as OSNet, FastReID, or TransReID
3. Motion-based association as a complementary cue alongside appearance
4. Multi-cue fusion combining appearance, geometry, and motion
5. Long-term identity memory across extended occlusions
6. Edge deployment on NVIDIA Jetson or Raspberry Pi

---


The objective was a fully functional proof-of-concept demonstrating the complete pipeline from raw video input to fused trajectory output, prioritizing correctness of architecture over state-of-the-art accuracy.

---


---

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [EPFL Multi-Camera Pedestrian Dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
- [SciPy Hungarian Algorithm]
