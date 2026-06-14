
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Fixed color per global_id so the same person
# always gets the same color across both cameras
COLORS = [
    (255,  80,  80),   # red
    ( 80, 255,  80),   # green
    ( 80,  80, 255),   # blue
    (255, 255,  80),   # yellow
    (255,  80, 255),   # magenta
    ( 80, 255, 255),   # cyan
    (255, 165,  80),   # orange
    (160,  80, 255),   # purple
]

def get_color(global_id):
    """Return a consistent BGR color for a given global_id."""
    return COLORS[(global_id - 1) % len(COLORS)]


def draw_frame(frame, tracks, fused_objects, trajectory_store, camera_label="CAM"):
    """
    Draw bounding boxes, IDs, confidence, and trajectory trail
    on a single camera frame.

    Args:
        frame            : BGR numpy array (will be modified in place)
        tracks           : list of Track objects for this camera
        fused_objects    : list of FusedObject from fusion.py
        trajectory_store : TrajectoryStore instance
        camera_label     : "CAM1" or "CAM2" shown in top-left corner

    Returns:
        annotated frame (numpy array)
    """
    output = frame.copy()

    # Build lookup: cam1_id -> global_id and cam2_id -> global_id
    cam1_global = {}
    cam2_global = {}
    for obj in fused_objects:
        if obj.cam1_id is not None:
            cam1_global[obj.cam1_id] = obj
        if obj.cam2_id is not None:
            cam2_global[obj.cam2_id] = obj

    for track in tracks:
        tid = track.track_id

        # Find which fused object this track belongs to
        fused_obj = cam1_global.get(tid) or cam2_global.get(tid)

        if fused_obj is None:
            # Track exists but not yet fused — draw grey
            color     = (180, 180, 180)
            label     = f"ID:?"
            conf_text = ""
        else:
            global_id = fused_obj.global_id
            color     = get_color(global_id)
            label     = f"ID:{global_id:02d}"
            conf_text = f"{fused_obj.confidence:.2f}"

        x1, y1, x2, y2 = [int(v) for v in track.last_bbox]

        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Label background for readability
        label_text = f"{label} {conf_text}"
        (tw, th), _ = cv2.getTextSize(label_text,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output,
                      (x1, y1 - th - 8),
                      (x1 + tw + 4, y1),
                      color, -1)  # filled rectangle
        cv2.putText(output, label_text,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2)  # black text on colored background

        # Trajectory trail (last 60 frames)
        if fused_obj is not None:
            recent = trajectory_store.get_recent_path(fused_obj.global_id,
                                                       n_frames=60)
            if len(recent) >= 2:
                for i in range(1, len(recent)):
                    pt1 = (int(recent[i-1][1]), int(recent[i-1][2]))
                    pt2 = (int(recent[i][1]),   int(recent[i][2]))
                    # Trail fades: older points more transparent
                    alpha = i / len(recent)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(output, pt1, pt2, color, thickness)

    # Camera label in top-left
    cv2.putText(output, camera_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 3)

    return output


def draw_fused_topview(trajectory_store, frame_size=(600, 600)):
    """
    Draw a top-down 2D trajectory plot for all fused objects.
    This is the "Ground-plane X vs Y" plot the problem statement asks for.

    Args:
        trajectory_store : TrajectoryStore instance
        frame_size       : (width, height) of output image

    Returns:
        top_view : BGR numpy array
    """
    W, H    = frame_size
    canvas  = np.ones((H, W, 3), dtype=np.uint8) * 20  # dark background

    # Title
    cv2.putText(canvas, "FUSED TRAJECTORY",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (220, 220, 220), 2)

    all_ids = trajectory_store.get_all_ids()
    if len(all_ids) == 0:
        return canvas

    # Collect all positions to normalize into canvas space
    all_xs, all_ys = [], []
    for gid in all_ids:
        xs, ys = trajectory_store.get_xy_arrays(gid)
        if len(xs) > 0:
            all_xs.extend(xs.tolist())
            all_ys.extend(ys.tolist())

    if len(all_xs) == 0:
        return canvas

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)

    # Avoid division by zero if all positions are identical
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)

    def to_canvas(cx, cy):
        """Map world position to canvas pixel."""
        px = int(40 + (cx - min_x) / range_x * (W - 80))
        py = int(40 + (cy - min_y) / range_y * (H - 80))
        return px, py

    for gid in all_ids:
        xs, ys = trajectory_store.get_xy_arrays(gid)
        if len(xs) < 50:
            continue

        color = get_color(gid)

        # Draw path
        for i in range(1, len(xs)):
            pt1 = to_canvas(xs[i-1], ys[i-1])
            pt2 = to_canvas(xs[i],   ys[i])
            cv2.line(canvas, pt1, pt2, color, 2)

        # Start marker
        start = to_canvas(xs[0], ys[0])
        cv2.circle(canvas, start, 6, color, -1)
        cv2.putText(canvas, "S",
                    (start[0]+6, start[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255), 1)

        # End marker
        end = to_canvas(xs[-1], ys[-1])
        cv2.circle(canvas, end, 6, (255, 255, 255), -1)
        cv2.putText(canvas, f"G{gid}",
                    (end[0]+6, end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1)

    # Grid lines for reference
    for i in range(1, 5):
        x = int(40 + (W - 80) * i / 4)
        y = int(40 + (H - 80) * i / 4)
        cv2.line(canvas, (x, 40), (x, H-40), (50, 50, 50), 1)
        cv2.line(canvas, (40, y), (W-40, y), (50, 50, 50), 1)

    return canvas


def save_trajectory_plot(trajectory_store, output_path):
    """
    Save a clean matplotlib trajectory plot as PNG.
    Higher quality than the OpenCV version — use this for submission.
    """
    plt.figure(figsize=(8, 8))
    plt.style.use("dark_background")

    all_ids = trajectory_store.get_all_ids()

    for gid in all_ids:
        xs, ys = trajectory_store.get_xy_arrays(gid)
        if len(xs) < 50:
            continue

        color_bgr = get_color(gid)
        color_rgb = (color_bgr[2]/255,
                     color_bgr[1]/255,
                     color_bgr[0]/255)

        plt.plot(xs, ys, color=color_rgb, linewidth=2, label=f"Object {gid:02d}")
        plt.scatter(xs[0],  ys[0],  color=color_rgb, marker="o", s=80, zorder=5)
        plt.scatter(xs[-1], ys[-1], color="white",   marker="x", s=80, zorder=5)

    plt.title("Fused 2D Trajectory", fontsize=14)
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to {output_path}")
