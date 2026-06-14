
import numpy as np

class FusedObject:
    """
    Represents a single real-world object after fusion.

    A fused object may come from:
      - Both cameras (matched)   -> position averaged from both views
      - Camera 1 only (unmatched)
      - Camera 2 only (unmatched)
    """

    def __init__(self, global_id, cx, cy, source, cam1_id=None, cam2_id=None, confidence=1.0):
        """
        global_id  : unique ID across the entire fused system
        cx, cy     : fused center position (pixels — relative to each camera
                     or averaged if both cameras visible)
        source     : "both", "cam1_only", "cam2_only"
        cam1_id    : original track ID in camera 1 (None if not seen)
        cam2_id    : original track ID in camera 2 (None if not seen)
        confidence : higher when seen by both cameras
        """
        self.global_id  = global_id
        self.cx         = cx
        self.cy         = cy
        self.source     = source
        self.cam1_id    = cam1_id
        self.cam2_id    = cam2_id
        self.confidence = confidence

    def __repr__(self):
        return (f"FusedObject(global_id={self.global_id}, "
                f"source={self.source}, "
                f"pos=({self.cx:.1f}, {self.cy:.1f}), "
                f"conf={self.confidence:.2f})")


class Fusion:
    """
    Combines Camera 1 and Camera 2 tracks into a unified object list.

    Fusion strategy:
      - Matched pair (same person seen by both cameras):
          position = average of both camera centers
          confidence = 1.0  (two cameras agree)

      - Unmatched cam1 track:
          position = camera 1 position
          confidence = 0.6  (single camera, less certain)

      - Unmatched cam2 track:
          position = camera 2 position
          confidence = 0.6

    Global IDs are assigned once and kept stable across frames
    by remembering which cam1_id/cam2_id maps to which global_id.
    """

    def __init__(self):
        self.next_global_id = 1

        # persistent mappings
        self.cam1_to_global = {}
        self.cam2_to_global = {}

    def _new_global_id(self):
        gid = self.next_global_id
        self.next_global_id += 1
        return gid

    def _get_center(self, track):
        x1, y1, x2, y2 = track.last_bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def fuse(
        self,
        tracks_cam1,
        tracks_cam2,
        matched_pairs,
        unmatched_cam1,
        unmatched_cam2
    ):

        cam1_lookup = {t.track_id: t for t in tracks_cam1}
        cam2_lookup = {t.track_id: t for t in tracks_cam2}

        fused_objects = []

        # ==================================================
        # MATCHED PAIRS
        # ==================================================

        for cam1_id, cam2_id, similarity in matched_pairs:

            if cam1_id not in cam1_lookup:
                continue

            if cam2_id not in cam2_lookup:
                continue

            # ------------------------------
            # Sticky global ID assignment
            # ------------------------------

            gid1 = self.cam1_to_global.get(cam1_id)
            gid2 = self.cam2_to_global.get(cam2_id)

            if gid1 is not None:
                global_id = gid1

            elif gid2 is not None:
                global_id = gid2

            else:
                global_id = self._new_global_id()

            self.cam1_to_global[cam1_id] = global_id
            self.cam2_to_global[cam2_id] = global_id

            cx1, cy1 = self._get_center(cam1_lookup[cam1_id])

            # Use cam1 coordinates only.
            # Do NOT average coordinates from different cameras.
            fused_objects.append(
                FusedObject(
                    global_id=global_id,
                    cx=cx1,
                    cy=cy1,
                    source="both",
                    cam1_id=cam1_id,
                    cam2_id=cam2_id,
                    confidence=round(float(similarity), 3)
                )
            )

        # ==================================================
        # CAM1 ONLY
        # ==================================================

        for cam1_id in unmatched_cam1:

            if cam1_id not in cam1_lookup:
                continue

            if cam1_id in self.cam1_to_global:
                global_id = self.cam1_to_global[cam1_id]
            else:
                global_id = self._new_global_id()
                self.cam1_to_global[cam1_id] = global_id

            cx, cy = self._get_center(cam1_lookup[cam1_id])

            fused_objects.append(
                FusedObject(
                    global_id=global_id,
                    cx=cx,
                    cy=cy,
                    source="cam1_only",
                    cam1_id=cam1_id,
                    cam2_id=None,
                    confidence=0.6
                )
            )

        # ==================================================
        # CAM2 ONLY
        # ==================================================

        for cam2_id in unmatched_cam2:

            if cam2_id not in cam2_lookup:
                continue

            if cam2_id in self.cam2_to_global:
                global_id = self.cam2_to_global[cam2_id]
            else:
                global_id = self._new_global_id()
                self.cam2_to_global[cam2_id] = global_id

            cx, cy = self._get_center(cam2_lookup[cam2_id])

            fused_objects.append(
                FusedObject(
                    global_id=global_id,
                    cx=cx,
                    cy=cy,
                    source="cam2_only",
                    cam1_id=None,
                    cam2_id=cam2_id,
                    confidence=0.6
                )
            )

        return fused_objects
