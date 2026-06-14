
import numpy as np
import clip
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

class Associator:
    """
    Associates tracks across two cameras using CLIP appearance embeddings.

    Core idea:
      - Every track has a set of crop images (person snapshots over time)
      - CLIP converts each crop into a 512-dimensional vector
      - Average all crop embeddings into one representative vector per track
      - Cosine similarity between Camera1 and Camera2 vectors = match score
      - Hungarian algorithm finds globally optimal assignment
      - Threshold rejects weak/wrong matches
    """

    def __init__(self, similarity_threshold=0.75):
        """
        similarity_threshold: cosine similarity minimum to accept a match
                              0.75 works well for persons with CLIP
                              raise to 0.80 if too many false matches
                              lower to 0.70 if missing correct matches
        """
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.threshold = similarity_threshold
        print(f"CLIP loaded on {self.device}")

    def _crop_to_embedding(self, crop_bgr):
        """
        Convert a single BGR crop (numpy array) to a 512-dim CLIP embedding.

        Steps:
          1. BGR -> RGB  (CLIP expects RGB)
          2. numpy -> PIL Image
          3. PIL -> CLIP preprocessed tensor
          4. CLIP encoder -> 512-dim vector
          5. L2 normalize (required for cosine similarity to work correctly)
        """
        # BGR to RGB
        crop_rgb = crop_bgr[:, :, ::-1].copy()

        # To PIL
        pil_image = Image.fromarray(crop_rgb)

        # CLIP preprocess: resize, center crop, normalize
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(tensor)

        # L2 normalize so cosine similarity = dot product
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()  # shape: (512,)

    def get_track_embedding(self, crops):
        """
        Get a single representative embedding for a track.

        A track has multiple crops over time (person at different moments).
        We embed each crop and average them.
        Averaging makes the representation more robust than any single frame.

        Args:
            crops: list of BGR numpy arrays (person crops from different frames)

        Returns:
            embedding: numpy array of shape (512,)
                       or None if no valid crops
        """
        if len(crops) == 0:
            return None

        # Sample at most 8 crops evenly spaced — no need for all of them
        if len(crops) > 20:
          crops = crops[-20:]
        if len(crops) > 8:
            indices = np.linspace(0, len(crops)-1, 8, dtype=int)
            crops = [crops[i] for i in indices]

        embeddings = []
        for crop in crops:
            # Skip crops that are too small — CLIP struggles with tiny images
            h, w = crop.shape[:2]
            if h < 20 or w < 10:
                continue
            emb = self._crop_to_embedding(crop)
            embeddings.append(emb)

        if len(embeddings) == 0:
            return None

        # Average all embeddings, then re-normalize
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm == 0:
            return None
        return mean_emb / norm

    def _cosine_similarity(self, emb1, emb2):
        """
        Cosine similarity between two L2-normalized vectors.
        Since both are already normalized: similarity = dot product.
        Range: -1 to 1. Higher = more similar.
        """
        return float(np.dot(emb1, emb2))

    def associate(self, tracks_cam1, tracks_cam2, crops_cam1, crops_cam2):
        """
        Main association function.

        Args:
            tracks_cam1 : list of Track objects from camera 1
            tracks_cam2 : list of Track objects from camera 2
            crops_cam1  : dict {track_id: [list of BGR crop arrays]}
            crops_cam2  : dict {track_id: [list of BGR crop arrays]}

        Returns:
            matched_pairs  : list of (cam1_track_id, cam2_track_id, similarity)
            unmatched_cam1 : list of cam1 track_ids with no match
            unmatched_cam2 : list of cam2 track_ids with no match
        """
        if len(tracks_cam1) == 0 or len(tracks_cam2) == 0:
            cam1_ids = [t.track_id for t in tracks_cam1]
            cam2_ids = [t.track_id for t in tracks_cam2]
            return [], cam1_ids, cam2_ids

        # Step 1: Get embeddings for all tracks
        embeddings_cam1 = {}
        for track in tracks_cam1:
            tid = track.track_id
            crops = crops_cam1.get(tid, [])
            emb = self.get_track_embedding(crops)
            if emb is not None:
                embeddings_cam1[tid] = emb

        embeddings_cam2 = {}
        for track in tracks_cam2:
            tid = track.track_id
            crops = crops_cam2.get(tid, [])
            emb = self.get_track_embedding(crops)
            if emb is not None:
                embeddings_cam2[tid] = emb

        # Only work with tracks that have valid embeddings
        ids1 = list(embeddings_cam1.keys())
        ids2 = list(embeddings_cam2.keys())

        if len(ids1) == 0 or len(ids2) == 0:
            return [], ids1, ids2

        # Step 2: Build similarity matrix
        # sim_matrix[i][j] = similarity between cam1 track ids1[i]
        #                     and cam2 track ids2[j]
        sim_matrix = np.zeros((len(ids1), len(ids2)), dtype=np.float32)

        for i, id1 in enumerate(ids1):
            for j, id2 in enumerate(ids2):
                sim_matrix[i][j] = self._cosine_similarity(
                    embeddings_cam1[id1],
                    embeddings_cam2[id2]
                )

        # Step 3: Hungarian algorithm — find optimal global assignment
        # linear_sum_assignment minimizes cost, so we negate to maximize similarity
        row_indices, col_indices = linear_sum_assignment(-sim_matrix)

        # Step 4: Apply threshold — reject weak matches
        matched_pairs  = []
        matched_rows   = set()
        matched_cols   = set()

        for row, col in zip(row_indices, col_indices):
            sim = sim_matrix[row][col]
            if sim >= self.threshold:
                cam1_id = ids1[row]
                cam2_id = ids2[col]
                matched_pairs.append((cam1_id, cam2_id, round(float(sim), 3)))
                matched_rows.add(row)
                matched_cols.add(col)

        # Unmatched tracks
        unmatched_cam1 = [ids1[i] for i in range(len(ids1)) if i not in matched_rows]
        unmatched_cam2 = [ids2[j] for j in range(len(ids2)) if j not in matched_cols]

        return matched_pairs, unmatched_cam1, unmatched_cam2
