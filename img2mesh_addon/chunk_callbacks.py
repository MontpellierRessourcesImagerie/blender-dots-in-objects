from pathlib import Path
import tifffile
import numpy as np

from .cellpose_runner import CellPoseRunner3D
from .spots_finder import SpotsFinder3D

def get_save_chunk_as(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
    def save_chunk_as(chunk, u, v, calib, index):
        z0, y0, x0 = u
        z1, y1, x1 = v
        img_path = folder_path / f"chunk_Z{z0}-{z1}_Y{y0}-{y1}_X{x0}-{x1}.tif"
        tifffile.imwrite(img_path, chunk)
        print(f"Saved chunk {index} at {img_path.name} -> {chunk.shape}")
    return save_chunk_as

def get_cellpose_run(folder_path, obj_size, ani_factor):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
    cp3d = CellPoseRunner3D("cyto3", True, obj_size, ani_factor)
    def cellpose_run(chunk, u, v, calib, index):
        z0, y0, x0 = u
        z1, y1, x1 = v
        img_path = folder_path / f"cp_mask_Z{z0}-{z1}_Y{y0}-{y1}_X{x0}-{x1}.tif"
        lbls = cp3d.run(chunk)
        tifffile.imwrite(img_path, lbls)
        print(f"Saved labels {img_path.name} ({np.max(lbls)} nuclei found)")
    return cellpose_run

def get_segment_as_meshes(overlap, ani_factor, lma):
    obj_size_yx = overlap[-1]
    cp3d = CellPoseRunner3D("cyto3", True, obj_size_yx, ani_factor)
    def segment_as_meshes(chunk, u, v, calib, index):
        calib_origin = tuple([u[i] * calib[i] for i in range(len(calib))])
        labeled = cp3d.run(chunk)
        lma.add_chunk(labeled, origin_zyx=calib_origin, halo_zyx=(0, 0, 0))
    return segment_as_meshes

def get_spots_detector(sd3d):
    def spots_detector(chunk, u, v, calib, index):
        calib_origin = tuple([u[i] * calib[i] for i in range(len(calib))])
        sd3d.find_spots(chunk, origin=calib_origin)
    return spots_detector