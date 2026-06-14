
import numpy as np
from collections import defaultdict

class TrajectoryStore:
    """
    Stores position history for every fused global_id over time.

    For each fused object, we record (frame_id, cx, cy) every frame.
    This gives us the full path from start to end — exactly what the
    problem statement asks for in the 2D trajectory plot.
    """

    def __init__(self, max_history=500):
        """
        max_history: maximum frames to remember per object.
                     500 frames at 10fps = 50 seconds — more than enough.
        """
        # key: global_id
        # value: list of (frame_id, cx, cy)
        self.trajectories = defaultdict(list)
        self.max_history  = max_history

    def update(self, fused_objects, frame_id):
        """
        Record current positions of all fused objects.

        Args:
            fused_objects : list of FusedObject from fusion.py
            frame_id      : current frame number
        """
        for obj in fused_objects:
            self.trajectories[obj.global_id].append(
                (frame_id, obj.cx, obj.cy)
            )

            if len(self.trajectories[obj.global_id]) > self.max_history:
                self.trajectories[obj.global_id].pop(0)

    def get_path(self, global_id):
        """
        Get full position history for one object.

        Returns:
            list of (frame_id, cx, cy) — chronological order
        """
        return self.trajectories.get(global_id, [])

    def get_recent_path(self, global_id, n_frames=60):
        """
        Get only the last n_frames positions.
        Used for drawing trailing trajectory lines on video.
        60 frames = ~6 seconds of trail at 10fps.
        """
        path = self.trajectories.get(global_id, [])
        return path[-n_frames:]

    def get_all_ids(self):
        """Return all global_ids that have any trajectory data."""
        return list(self.trajectories.keys())

    def get_xy_arrays(self, global_id):
        """
        Get x and y as separate numpy arrays.
        Convenient for matplotlib plotting.

        Returns:
            xs: numpy array of x positions
            ys: numpy array of y positions
        """
        path = self.trajectories.get(global_id, [])
        if len(path) == 0:
            return np.array([]), np.array([])
        xs = np.array([p[1] for p in path])
        ys = np.array([p[2] for p in path])
        return xs, ys
