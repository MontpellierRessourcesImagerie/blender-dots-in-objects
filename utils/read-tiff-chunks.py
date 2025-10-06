import zarr
from pathlib import Path
import tifffile as tiff
from tifffile import TiffFile, CHUNKMODE
from cellpose import models
from cellpose.io import logger_setup
import numpy as np
import re

import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter, vtkCleanPolyData
from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkFiltersGeneral import vtkDiscreteFlyingEdges3D
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkFiltersCore import vtkTriangleFilter, vtkWindowedSincPolyDataFilter

class CellPoseRunner3D(object):
    def __init__(self, model, gpu, obj_size, ani_factor):
        self.model_name = model
        self.gpu = gpu
        self.obj_size = obj_size
        self.ani_factor = ani_factor
        self.model = None
        logger_setup()

    def create_model(self):
        if self.model is not None:
            return
        self.model = models.CellposeModel(
            gpu=self.gpu,
            model_type=self.model_name
        )

    def run(self, image):
        self.create_model()
        if self.model is None:
            return None
        img = np.asarray(image)
        labels, _, _ = self.model.eval(
            img,
            channels=[0, 0],
            diameter=self.obj_size,
            anisotropy=self.ani_factor,
            do_3D=True,
            z_axis=0
        )
        return labels

def get_shape(dsc):
    s = dsc.series[0]
    axes = s.axes
    if axes != "ZYX":
        raise ValueError(f"Unexpected axes: {axes}. Expected single-channel 3D 'ZYX'.")
    return s.shape

def load_by_chunks(path, chunk_size, overlap, callback=None):
    path = str(path)
    with TiffFile(path) as tf:
        img_shape = get_shape(tf)
        zobj = tf.aszarr(series=0, level=0, chunkmode=CHUNKMODE.PAGE)
    arr = zarr.open(zobj, mode="r")  # lazy reading
    Z, Y, X = img_shape
    step_z = Z if chunk_size[0] >= Z else max(chunk_size[0] - overlap[0], 1)
    step_y = Y if chunk_size[1] >= Y else max(chunk_size[1] - overlap[1], 1)
    step_x = X if chunk_size[2] >= X else max(chunk_size[2] - overlap[2], 1)
    for z0 in range(0, Z, step_z):
        for y0 in range(0, Y, step_y):
            for x0 in range(0, X, step_x):
                z1 = min(z0 + chunk_size[0], Z)
                y1 = min(y0 + chunk_size[1], Y)
                x1 = min(x0 + chunk_size[2], X)
                chunk = arr[z0:z1, y0:y1, x0:x1]
                if callback:
                    callback(chunk, (z0, y0, x0), (z1, y1, x1))

def save_chunks(chunk, start, end):
    z0, y0, x0 = start
    z1, y1, x1 = end
    filename = "/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/"
    filename += f"chunk_Z{z0}-{z1}_Y{y0}-{y1}_X{x0}-{x1}.tif"
    tiff.imwrite(filename, chunk)
    del chunk
    print(f"Saved chunk to {filename}")

def test_on_heart():
    path = "/home/clement/Downloads/2025-09-18-vbernard-ims/c2-Resolution Level 1-1.tif"
    chunk_size = (135, 600, 600)  # z, y, x
    anisotropy_factor = 3.310710147
    object_size_yx = 20  # in pixels
    object_size_z = int(object_size_yx / anisotropy_factor)
    overlap = (object_size_z, object_size_yx, object_size_yx)
    load_by_chunks(path, chunk_size, overlap, callback=save_chunks)

def test_on_c_elegans():
    path = "/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/eft3RW10035L1_0125071.tif"
    chunk_size = (140, 140, 300)  # z, y, x
    anisotropy_factor = 0.122 / 0.116 # z/x
    object_size_yx = 15  # in pixels
    object_size_z = int(object_size_yx / anisotropy_factor)
    overlap = (object_size_z, object_size_yx, object_size_yx)
    load_by_chunks(path, chunk_size, overlap, callback=save_chunks)

def labeled_volume_to_surface(labels, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), background=0, add_region_ids=True):
    labels_xyz = np.transpose(np.asarray(labels), (2, 1, 0))  # (X, Y, Z)
    X, Y, Z = labels_xyz.shape

    # 2) Wrap into vtkImageData with correct geometry
    img = vtkImageData()
    img.SetOrigin(*origin)
    img.SetSpacing(*spacing)
    img.SetDimensions(X, Y, Z)  # number of POINTS per axis

    vtk_scalars = numpy_to_vtk(labels_xyz.ravel(order="F"), deep=True)
    vtk_scalars.SetName("label")
    img.GetPointData().SetScalars(vtk_scalars)

    # 3) One-pass discrete isosurface for all labels (except background)
    uniq = np.unique(labels_xyz)
    uniq = uniq[uniq != background].astype(int)

    dfe = vtkDiscreteFlyingEdges3D()     # newer, faster than vtkDiscreteMarchingCubes
    dfe.SetInputData(img)
    dfe.ComputeScalarsOn()               # keep label value on output
    for i, val in enumerate(uniq):
        dfe.SetValue(i, int(val))
    dfe.Update()

    poly = dfe.GetOutput()

    # 4) Optional: tag connected components (kept in one mesh)
    if add_region_ids:
        conn = vtkPolyDataConnectivityFilter()
        conn.SetInputData(poly)
        conn.SetExtractionModeToAllRegions()
        conn.ColorRegionsOn()            # adds "RegionId" cell scalars
        conn.Update()
        poly = conn.GetOutput()

    # Clean coincident points (small tidy-up)
    clean = vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()
    poly = clean.GetOutput()
    return poly

def smooth_polydata(poly, iterations=5, pass_band=0.12, feature_angle=120.0,
                    boundary_smoothing=False, feature_edge_smoothing=False):
    # ensure triangles (some VTK filters prefer this)
    tri = vtkTriangleFilter()
    tri.SetInputData(poly)

    ws = vtkWindowedSincPolyDataFilter()
    ws.SetInputConnection(tri.GetOutputPort())
    ws.SetNumberOfIterations(int(iterations))
    ws.SetPassBand(float(pass_band))              # ~0.08–0.2 is typical
    ws.SetFeatureAngle(float(feature_angle))
    if boundary_smoothing: ws.BoundarySmoothingOn()
    else: ws.BoundarySmoothingOff()
    if feature_edge_smoothing: ws.FeatureEdgeSmoothingOn()
    else: ws.FeatureEdgeSmoothingOff()
    ws.NonManifoldSmoothingOn()
    ws.NormalizeCoordinatesOn()
    ws.Update()
    return ws.GetOutput()

import numpy as np
from pathlib import Path
# Headless VTK imports
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkBox
from vtkmodules.vtkFiltersGeneral import vtkDiscreteFlyingEdges3D
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData
from vtkmodules.vtkFiltersCore import vtkClipPolyData, vtkAppendPolyData, vtkCleanPolyData
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter  # if you still use it elsewhere
from vtkmodules.vtkCommonDataModel import vtkBox

from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.util.numpy_support import numpy_to_vtk

class LabelMeshAccumulator:
    def __init__(self, spacing_xyz=(1,1,1), background=0):
        self.spacing = tuple(map(float, spacing_xyz))
        self.background = int(background)
        self._append = vtkAppendPolyData()
        # self._append.SetMergePoints(False)
        self._added = 0
        self.last_lbl = 0

    def shift_labels(self, input_img):
        z = input_img != 0
        input_img[z] += self.last_lbl
        return input_img

    def add_chunk(self, labels_zyx: np.ndarray, origin_xyz=(0,0,0), halo_xyz=(0,0,0)):
        # labels_zyx: (Z,Y,X) ints for this chunk
        # origin_xyz: world coords of voxel (0,0,0) of this chunk
        # halo_xyz: voxels to crop on each side (X,Y,Z) of overlap to avoid duplicates
        labels_zyx = self.shift_labels(labels_zyx)
        lab_xyz = np.transpose(np.asarray(labels_zyx), (2,1,0))  # -> (X,Y,Z)
        X, Y, Z = lab_xyz.shape
        ox, oy, oz = map(float, origin_xyz)
        sx, sy, sz = self.spacing
        hx, hy, hz = map(int, halo_xyz)

        img = vtkImageData()
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(ox, oy, oz)
        img.SetDimensions(X, Y, Z)

        vtk_scal = numpy_to_vtk(lab_xyz.ravel(order="F").astype(np.int32), deep=True)
        vtk_scal.SetName("label")
        img.GetPointData().SetScalars(vtk_scal)

        # One pass discrete isosurface for labels present in this chunk
        vals = [int(v) for v in np.unique(lab_xyz) if v != self.background]
        self.last_lbl = max(vals)+1
        if not vals:
            return
        dfe = vtkDiscreteFlyingEdges3D()
        dfe.SetInputData(img)
        dfe.ComputeScalarsOn()
        for i, v in enumerate(vals):
            dfe.SetValue(i, v)
        dfe.Update()
        poly = dfe.GetOutput()

        # Keep only the interior (non-overlap) region in WORLD coordinates
        xmin = ox + hx * sx
        xmax = ox + (X - hx) * sx
        ymin = oy + hy * sy
        ymax = oy + (Y - hy) * sy
        zmin = oz + hz * sz
        zmax = oz + (Z - hz) * sz

        box = vtkBox()
        box.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)

        clip = vtkClipPolyData()
        clip.SetInputData(poly)
        clip.SetClipFunction(box)
        clip.InsideOutOn()         # keep inside the box
        clip.Update()

        self._append.AddInputData(clip.GetOutput())
        self._added += 1

    def finalize(self):
        if self._added == 0:
            return None
        self._append.Update()
        return self._append.GetOutput()

def save_polydata_as_ply(polydata, path, binary=True):
    w = vtkPLYWriter()
    w.SetFileName(str(path))
    w.SetInputData(polydata)
    if binary:
        w.SetFileTypeToBinary()
    else:
        w.SetFileTypeToASCII()
    w.Write()

def test_run_cellpose():
    img_path = Path("/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/chunk_Z0-140_Y0-140_X855-994.tif")
    data = tiff.imread(img_path)
    cp3d = CellPoseRunner3D(
        "cyto3",
        gpu=True,
        obj_size=15,
        ani_factor=0.122/0.116
    )
    lbls = cp3d.run(data)
    if lbls is not None:
        out_path = f"/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/{img_path.name.replace('chunk_', 'labeled_')}"
        tiff.imwrite(out_path, lbls)
        print(f"Saved labels to {out_path}")

def extract_zyx_start(fname):
    name = Path(fname).name
    m = re.search(r'_Z(-?\d+)-\d+_Y(-?\d+)-\d+_X(-?\d+)-\d+', name)
    if not m:
        return None
    z, y, x = map(int, m.groups())
    return (z, y, x)

def test_make_meshes():
    folder_path = Path("/home/clement/Downloads/2025-10-05-tests-flash-tuto/c_elegans_nuclei/c_elegans_nuclei/test/images/out/")
    label_files = sorted(folder_path.glob("labeled_*.tif"))
    mesher = LabelMeshAccumulator(spacing_xyz=(0.116, 0.116, 0.122), background=0)
    calib = (0.122, 0.116, 0.116)

    for lbls_path in label_files:
        pxl_origin = extract_zyx_start(lbls_path)
        print("Pxl origin:", pxl_origin)
        if pxl_origin is None:
            print(f"Could not extract origin from filename: {lbls_path}")
            continue
        calib_origin = tuple(p * s for p, s in zip(pxl_origin, calib))
        print("Calib origin:", calib_origin)
        xyz_calib = [calib_origin[len(calib)-i-1] for i, p in enumerate(calib_origin)]  # to (x,y,z)
        labeled = tiff.imread(lbls_path)

        anisotropy_factor = 0.122 / 0.116 # z/x
        object_size_yx = 15  # in pixels
        object_size_z = int(object_size_yx / anisotropy_factor)
        overlap = (object_size_z, object_size_yx, object_size_yx)

        mesher.add_chunk(labeled, origin_xyz=xyz_calib, halo_xyz=overlap)
    mesh = mesher.finalize()
    if mesh is None:
        print("No mesh was created.")
        return
    output_path = "/tmp/test_merge.ply"
    save_polydata_as_ply(mesh, output_path, binary=True)
    print(f"Saved merged mesh to {output_path}")

if __name__ == "__main__":
    test_make_meshes()