import numpy as np
import meshio
from skimage.segmentation import clear_border
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from collections import defaultdict
from typing import Dict, Tuple, List

class LabelMeshAccumulator(object):
    """
    Accumulates meshes from labeled image chunks.
    Labels touching the borders are discarded to avoid duplicates.
    Meshes can be saved as a single PLY file if required.

    Args:
        spacing_zyx: Tuple of voxel spacing in Z, Y, X order.
        background: Label value to ignore (default: 0).
    """
    def __init__(self, spacing_zyx=(1.0, 1.0, 1.0), background=0):
        self.spacing_zyx = spacing_zyx
        self.background = int(background)
        self._per_label_meshes: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        self.last_lbl = 0

    def shift_labels(self, input_img):
        z = input_img != 0
        input_img[z] += self.last_lbl

    def add_chunk(self, labels_zyx: np.ndarray, origin_zyx=(0, 0, 0), overlap_zyx=(0, 0, 0)):
        """
        Records meshes for a new labeled chunk.
        Labels touching the borders defined by the overlap are discarded.

        Args:
            labels_zyx: 3D numpy array of shape (Z, Y, X) containing integer labels.
            origin_zyx: Tuple of the physical origin of the chunk in Z, Y, X order.
            overlap_zyx: Tuple of the overlap size in voxels in Z, Y, X order. 
                         Labels touching the borders defined by this overlap will be discarded.
        """
        if labels_zyx.ndim != 3:
            raise ValueError(f"labels_zyx must be 3D (Z,Y,X); got shape {labels_zyx.shape}")
        print(f"Converting labels to meshes for block: {origin_zyx}")

        labels_zyx = clear_border(labels_zyx)
        self.shift_labels(labels_zyx)

        Z, Y, X = labels_zyx.shape
        hx, hy, hz = map(int, overlap_zyx)
        z0 = max(0, hz)
        z1 = max(z0, min(Z - hz, Z))
        y0 = max(0, hy)
        y1 = max(y0, min(Y - hy, Y))
        x0 = max(0, hx)
        x1 = max(x0, min(X - hx, X))
        if (z1 - z0) < 2 or (y1 - y0) < 2 or (x1 - x0) < 2:
            print("Chunk too small after applying overlap borders, skipping.")
            return

        sub = labels_zyx[z0:z1, y0:y1, x0:x1]
        unique_vals = np.unique(sub)
        unique_vals = unique_vals[unique_vals != self.background]
        if unique_vals.size == 0:
            return
        self.last_lbl = max(self.last_lbl, int(unique_vals.max()))

        oz, oy, ox = origin_zyx
        dz, dy, dx = self.spacing_zyx
        off_x = ox + x0 * dx
        off_y = oy + y0 * dy
        off_z = oz + z0 * dz

        buffer_eq = np.zeros_like(sub, dtype=bool)
        buffer_mk = np.zeros_like(sub, dtype=np.float32)
        buffer_gs = np.zeros_like(sub, dtype=np.float32)
        sigma_yx  = 0.8
        sigma_z   = sigma_yx * (dy / dz)

        for lbl in unique_vals.astype(int):
            np.equal(sub, lbl, out=buffer_eq)
            np.copyto(buffer_mk, buffer_eq)
            gaussian_filter(buffer_mk, sigma=(sigma_z, sigma_yx, sigma_yx), output=buffer_gs)
            try:
                verts_zyx, faces, _, _ = marching_cubes(
                    volume=buffer_gs,
                    level=0.8,
                    spacing=self.spacing_zyx,
                    allow_degenerate=False,
                    method='lorensen'
                )
            except Exception:
                continue

            if verts_zyx.size == 0 or faces.size == 0:
                continue

            vx = verts_zyx[:, 2] + off_x
            vy = verts_zyx[:, 1] + off_y
            vz = verts_zyx[:, 0] + off_z
            verts_xyz = np.stack([vx, vy, vz], axis=1).astype(np.float32)
            faces = faces.astype(np.int32)

            self._per_label_meshes[int(lbl)].append((verts_xyz, faces))

    def finalize(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for lbl, parts in self._per_label_meshes.items():
            if not parts:
                continue
            all_verts = []
            all_faces = []
            offset = 0
            for v, f in parts:
                all_verts.append(v)
                all_faces.append(f + offset)
                offset += v.shape[0]
            verts = np.concatenate(all_verts, axis=0) if len(all_verts) > 1 else all_verts[0]
            faces = np.concatenate(all_faces, axis=0) if len(all_faces) > 1 else all_faces[0]
            out[lbl] = (verts, faces)
        return out
    
    def save_meshes_as_ply(self, out_path, binary=True):
        meshes = self.finalize()
        if not meshes:
            print("No meshes to save.")
            return
        all_verts = []
        all_faces = []
        all_labels = []
        offset = 0
        for lbl, (verts, faces) in meshes.items():
            all_verts.append(verts)
            all_labels.extend([lbl] * faces.shape[0])
            all_faces.append(faces + offset)
            offset += verts.shape[0]
        if not all_verts or not all_faces:
            print("No meshes to save.")
            return
        verts = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        mesh = meshio.Mesh(points=verts, cells=[("triangle", faces)])
        meshio.write(out_path, mesh, file_format="ply", binary=binary)
        print(f"Saved concatenated mesh to {out_path}")