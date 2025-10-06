# import zarr
from pathlib import Path
import tifffile as tiff
# from tifffile import TiffFile, CHUNKMODE
# from cellpose import models
# from cellpose.io import logger_setup
import numpy as np
import re
import meshio
from skimage.segmentation import clear_border
from scipy.ndimage import gaussian_filter

# class CellPoseRunner3D(object):
#     def __init__(self, model, gpu, obj_size, ani_factor):
#         self.model_name = model
#         self.gpu = gpu
#         self.obj_size = obj_size
#         self.ani_factor = ani_factor
#         self.model = None
#         logger_setup()

#     def create_model(self):
#         if self.model is not None:
#             return
#         self.model = models.CellposeModel(
#             gpu=self.gpu,
#             model_type=self.model_name
#         )

#     def run(self, image):
#         self.create_model()
#         if self.model is None:
#             return None
#         img = np.asarray(image)
#         labels, _, _ = self.model.eval(
#             img,
#             channels=[0, 0],
#             diameter=self.obj_size,
#             anisotropy=self.ani_factor,
#             do_3D=True,
#             z_axis=0
#         )
#         return labels

# def get_shape(dsc):
#     s = dsc.series[0]
#     axes = s.axes
#     if axes != "ZYX":
#         raise ValueError(f"Unexpected axes: {axes}. Expected single-channel 3D 'ZYX'.")
#     return s.shape

# def load_by_chunks(path, chunk_size, overlap, callback=None):
#     path = str(path)
#     with TiffFile(path) as tf:
#         img_shape = get_shape(tf)
#         zobj = tf.aszarr(series=0, level=0, chunkmode=CHUNKMODE.PAGE)
#     arr = zarr.open(zobj, mode="r")  # lazy reading
#     Z, Y, X = img_shape
#     step_z = Z if chunk_size[0] >= Z else max(chunk_size[0] - overlap[0], 1)
#     step_y = Y if chunk_size[1] >= Y else max(chunk_size[1] - overlap[1], 1)
#     step_x = X if chunk_size[2] >= X else max(chunk_size[2] - overlap[2], 1)
#     for z0 in range(0, Z, step_z):
#         for y0 in range(0, Y, step_y):
#             for x0 in range(0, X, step_x):
#                 z1 = min(z0 + chunk_size[0], Z)
#                 y1 = min(y0 + chunk_size[1], Y)
#                 x1 = min(x0 + chunk_size[2], X)
#                 chunk = arr[z0:z1, y0:y1, x0:x1]
#                 if callback:
#                     callback(chunk, (z0, y0, x0), (z1, y1, x1))

# def save_chunks(chunk, start, end):
#     z0, y0, x0 = start
#     z1, y1, x1 = end
#     filename = "/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/"
#     filename += f"chunk_Z{z0}-{z1}_Y{y0}-{y1}_X{x0}-{x1}.tif"
#     tiff.imwrite(filename, chunk)
#     print(f"Saved chunk to {filename}")

# def test_on_heart():
#     path = "/home/clement/Downloads/2025-09-18-vbernard-ims/c2-Resolution Level 1-1.tif"
#     chunk_size = (135, 600, 600)  # z, y, x
#     anisotropy_factor = 3.310710147
#     object_size_yx = 20  # in pixels
#     object_size_z = int(object_size_yx / anisotropy_factor)
#     overlap = (object_size_z, object_size_yx, object_size_yx)
#     load_by_chunks(path, chunk_size, overlap, callback=save_chunks)

# def test_on_c_elegans():
#     path = "/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/eft3RW10035L1_0125071.tif"
#     chunk_size = (140, 140, 300)  # z, y, x
#     anisotropy_factor = 0.122 / 0.116 # z/x
#     object_size_yx = 15  # in pixels
#     object_size_z = int(object_size_yx / anisotropy_factor)
#     overlap = (object_size_z, object_size_yx, object_size_yx)
#     load_by_chunks(path, chunk_size, overlap, callback=save_chunks)

# import numpy as np
# from skimage.measure import marching_cubes
# from collections import defaultdict
# from typing import Dict, Tuple, List

# class LabelMeshAccumulator:

#     def __init__(self, spacing_zyx=(1, 1, 1), background=0):
#         self.spacing_zyx = spacing_zyx
#         self.background = int(background)
#         self._per_label_meshes: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
#         self.last_lbl = 0

#     def shift_labels(self, input_img):
#         z = input_img != 0
#         input_img[z] += self.last_lbl

#     def add_chunk(self, labels_zyx: np.ndarray, origin_zyx=(0, 0, 0), halo_zyx=(0, 0, 0)):
#         if labels_zyx.ndim != 3:
#             raise ValueError(f"labels_zyx must be 3D (Z,Y,X); got shape {labels_zyx.shape}")
#         labels_zyx = clear_border(labels_zyx)
#         self.shift_labels(labels_zyx)

#         Z, Y, X = labels_zyx.shape
#         hx, hy, hz = map(int, halo_zyx)
        
#         z0 = max(0, hz); z1 = max(z0, min(Z - hz, Z))
#         y0 = max(0, hy); y1 = max(y0, min(Y - hy, Y))
#         x0 = max(0, hx); x1 = max(x0, min(X - hx, X))
#         if (z1 - z0) < 2 or (y1 - y0) < 2 or (x1 - x0) < 2:
#             return

#         sub = labels_zyx[z0:z1, y0:y1, x0:x1]
#         unique_vals = np.unique(sub)
#         unique_vals = unique_vals[unique_vals != self.background]
#         if unique_vals.size == 0:
#             return
#         self.last_lbl = max(self.last_lbl, int(unique_vals.max()))

#         oz, oy, ox = map(float, origin_zyx)
#         dz, dy, dx = self.spacing_zyx
#         off_x = ox + x0 * dx
#         off_y = oy + y0 * dy
#         off_z = oz + z0 * dz

#         buffer_eq = np.zeros_like(sub, dtype=bool)
#         buffer_mk = np.zeros_like(sub, dtype=np.float32)
#         buffer_gs = np.zeros_like(sub, dtype=np.float32)
#         sigma_yx  = 0.75
#         sigma_z   = sigma_yx * (dy / dz)

#         for lbl in unique_vals.astype(int):
#             np.equal(sub, lbl, out=buffer_eq)
#             np.copyto(buffer_mk, buffer_eq)
#             gaussian_filter(buffer_mk, sigma=(sigma_z, sigma_yx, sigma_yx), output=buffer_gs)
#             try:
#                 verts_zyx, faces, _, _ = marching_cubes(
#                     volume=buffer_gs,
#                     level=0.8,
#                     spacing=self.spacing_zyx,
#                     allow_degenerate=False,
#                     method='lorensen'
#                 )
#             except Exception:
#                 continue

#             if verts_zyx.size == 0 or faces.size == 0:
#                 continue

#             vx = verts_zyx[:, 2] + off_x
#             vy = verts_zyx[:, 1] + off_y
#             vz = verts_zyx[:, 0] + off_z
#             verts_xyz = np.stack([vx, vy, vz], axis=1).astype(np.float32)
#             faces = faces.astype(np.int32)

#             self._per_label_meshes[int(lbl)].append((verts_xyz, faces))

#     def finalize(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
#         out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
#         for lbl, parts in self._per_label_meshes.items():
#             if not parts:
#                 continue
#             all_verts = []
#             all_faces = []
#             offset = 0
#             for v, f in parts:
#                 all_verts.append(v)
#                 all_faces.append(f + offset)
#                 offset += v.shape[0]
#             verts = np.concatenate(all_verts, axis=0) if len(all_verts) > 1 else all_verts[0]
#             faces = np.concatenate(all_faces, axis=0) if len(all_faces) > 1 else all_faces[0]
#             out[lbl] = (verts, faces)
#         return out

# def save_meshes_as_ply(meshes: Dict[int, Tuple[np.ndarray, np.ndarray]], out_path, binary=True):
#     all_verts = []
#     all_faces = []
#     all_labels = []
#     offset = 0
#     for lbl, (verts, faces) in meshes.items():
#         all_verts.append(verts)
#         # Store label for each face (optional, for segmentation)
#         all_labels.extend([lbl] * faces.shape[0])
#         all_faces.append(faces + offset)
#         offset += verts.shape[0]
#     if not all_verts or not all_faces:
#         print("No meshes to save.")
#         return
#     verts = np.concatenate(all_verts, axis=0)
#     faces = np.concatenate(all_faces, axis=0)
#     mesh = meshio.Mesh(points=verts, cells=[("triangle", faces)])
#     meshio.write(out_path, mesh, file_format="ply", binary=binary)
#     print(f"Saved concatenated mesh to {out_path}")

# def test_run_cellpose():
#     img_path = Path("/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/chunk_Z0-140_Y0-140_X855-994.tif")
#     data = tiff.imread(img_path)
#     cp3d = CellPoseRunner3D(
#         "cyto3",
#         gpu=True,
#         obj_size=15,
#         ani_factor=0.122/0.116
#     )
#     lbls = cp3d.run(data)
#     if lbls is not None:
#         out_path = f"/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/{img_path.name.replace('chunk_', 'labeled_')}"
#         tiff.imwrite(out_path, lbls)
#         print(f"Saved labels to {out_path}")

# def extract_zyx_start(fname):
#     name = Path(fname).name
#     m = re.search(r'_Z(-?\d+)-\d+_Y(-?\d+)-\d+_X(-?\d+)-\d+', name)
#     if not m:
#         return None
#     z, y, x = map(int, m.groups())
#     return (z, y, x)

# def test_make_meshes():
#     folder_path = Path("/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/")
#     label_files = sorted(folder_path.glob("labeled_*.tif"))
#     calib = (0.122, 0.116, 0.116)
#     mesher = LabelMeshAccumulator(spacing_zyx=calib, background=0)

#     for lbls_path in label_files:
#         pxl_origin = extract_zyx_start(lbls_path)
#         if pxl_origin is None:
#             print(f"Could not extract origin from filename: {lbls_path}")
#             continue
#         calib_origin = tuple(p * s for p, s in zip(pxl_origin, calib))
#         labeled = tiff.imread(lbls_path)

#         anisotropy_factor = 0.122 / 0.116 # z/x
#         object_size_yx = 15  # in pixels
#         object_size_z = int(object_size_yx / anisotropy_factor)
#         overlap = (object_size_z, object_size_yx, object_size_yx)

#         mesher.add_chunk(labeled, origin_zyx=calib_origin, halo_zyx=(0, 0, 0))
#     mesh = mesher.finalize()
#     if mesh is None:
#         print("No mesh was created.")
#         return
#     output_path = "/tmp/lorensen.ply"
#     save_meshes_as_ply(mesh, output_path, binary=True)

if __name__ == "__main__":
    test_make_meshes()