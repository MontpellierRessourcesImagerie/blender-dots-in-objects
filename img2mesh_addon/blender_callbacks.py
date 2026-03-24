import bpy

import json
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Iterable

from .chunks_generator import ChunksGenerator
from .make_meshes import LabelMeshAccumulator
from .spots_finder import SpotsFinder3D

from .chunk_callbacks import get_segment_as_meshes, get_spots_detector

def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    elif col.name not in {c.name for c in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(col)
    return col

def get_cp_models():
    json_path = Path(__file__).parent / "cellpose_models.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"CellPose models JSON not found at {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    models = []
    for model in data:
        n = model.get("name", None)
        p = model.get("model_path", None)
        d = model.get("description", "")
        b = model.get("built-in", None)
        if any(x is None for x in (n, p, d, b)):
            print(f"Warning: skipping invalid model entry in JSON: {model}")
            continue
        if not b and not Path(p).is_file():
            print(f"Warning: model '{n}' is not built-in and model file not found at {p}, skipping")
            continue
        models.append((p, n, d))
    return models

def import_label_meshes_np(collection_name: str,
                           meshes_by_label: Dict[int, Tuple[np.ndarray, np.ndarray]]
                           ) -> List[bpy.types.Object]:
    """
    meshes_by_label[label] = (vertices, faces)
      - vertices: np.ndarray shape (N, 3), float
      - faces:    np.ndarray shape (M, 3 or 4), int (triangles or quads)
    """
    col = _ensure_collection("nuclei-"+collection_name)
    created: List[bpy.types.Object] = []

    for label in sorted(meshes_by_label.keys()):
        verts_np, faces_np = meshes_by_label[label]

        if verts_np.ndim != 2 or verts_np.shape[1] != 3:
            raise ValueError(f"Label {label}: vertices must be (N,3), got {verts_np.shape}")
        if faces_np.ndim != 2 or faces_np.shape[1] not in (3, 4):
            raise ValueError(f"Label {label}: faces must be (M,3) or (M,4), got {faces_np.shape}")

        # Ensure correct dtypes without extra copies when possible
        verts = np.asarray(verts_np, dtype=np.float32)
        faces = np.asarray(faces_np, dtype=np.int32)

        # Build mesh (from_pydata expects Python sequences)
        mesh = bpy.data.meshes.new(name=f"mesh-{label}")
        mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
        mesh.validate(clean_customdata=True)
        mesh.update()

        obj = bpy.data.objects.new(name=f"obj-{label}", object_data=mesh)
        col.objects.link(obj)
        created.append(obj)

    # Set origin to geometry for al these objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in created:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = created[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

    return created

def import_points_as_empties(
    collection_name: str,
    points: Iterable[Tuple[float, float, float]],
    prefix: str = "pt",
    empty_type: str = "PLAIN_AXES",
    size: float = 0.1,
):
    col = _ensure_collection("spots-"+collection_name)
    created = []
    for i, p in enumerate(points, 1):
        x, y, z = map(float, p)
        obj = bpy.data.objects.new(name=f"{prefix}-{i:04d}", object_data=None)
        obj.empty_display_type = empty_type
        obj.empty_display_size = float(size)
        obj.location = (x, y, z)
        col.objects.link(obj)
        created.append(obj)
    return created

def segment_and_import(img_path, secondary, calib, obj_size_yx, chunk_size, model, min_obj_size):
    cg = ChunksGenerator(
        [img_path] if not secondary else [img_path, secondary],
        calib, 
        obj_size_yx
    )
    lma = LabelMeshAccumulator(cg.get_calibration(), 0)

    sam_fx = get_segment_as_meshes(cg.get_overlap(), cg.get_anisotropy(), lma)

    cg.load_by_chunks(chunk_size, callback=sam_fx) # (140, 140, 300)
    meshes = lma.finalize()
    import_label_meshes_np(img_path.name, meshes)

def detect_and_import(img_path, calib, thr, chunk_size):
    cg = ChunksGenerator(img_path, calib, 0)
    sd3d = SpotsFinder3D(cg.get_calibration(), thr)
    sd_fx = get_spots_detector(sd3d)
    cg.load_by_chunks(chunk_size, callback=sd_fx) # (140, 140, 300)
    points = sd3d.get_all_spots()
    points = [(x, y, z) for (z, y, x) in points]
    import_points_as_empties(img_path.name, points, prefix="spot", empty_type="SPHERE", size=0.2)