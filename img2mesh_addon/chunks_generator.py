import zarr
from pathlib import Path
import tifffile as tiff
from tifffile import TiffFile, CHUNKMODE

class ChunksGenerator(object):
    """
    From a list of paths to 3D single-channel TIFF images, generates chunks of the images with specified overlap.
    The chunks are loaded as numpy arrays and passed to a callback function for processing.
    The shape of all images must be the same, and the calibration is assumed to be the same for all images.
    The overlap is set to the object size (in voxels) adapted to the anisotropy of the images.
    """
    def __init__(self, img_paths, calib=(1.0, 1.0, 1.0), obj_diam_yx=30):
        if img_paths is None or len(img_paths) == 0:
            raise ValueError("No image paths provided.")
        
        self.tiff_paths = []
        for img_path in img_paths:
            candidate = Path(img_path)
            if not candidate.exists() or not candidate.is_file():
                print(f"Warning: Image path {candidate} does not exist or is not a file, skipping.")
                continue
            self.tiff_paths.append(candidate)
        if len(self.tiff_paths) == 0:
            raise ValueError("No valid image paths provided.")
        
        self.calibration = calib # Z, Y, X
        self.anisotropy_factor = calib[0] / calib[1]
        self.shape = None
        object_size_z = int(obj_diam_yx / self.anisotropy_factor)
        self.overlap = (object_size_z, obj_diam_yx, obj_diam_yx)
    
    def _get_shape(self):
        """
        Extracts the shape of the images by reading all of the TIFF files.
        Also verifies that all TIFF files have the same shape.
        """
        candidate = None
        for tiff_path in self.tiff_paths:
            with TiffFile(tiff_path) as dsc:
                s = dsc.series[0]
                axes = s.axes
                if axes != "ZYX":
                    raise ValueError(f"Unexpected axes: {axes}. Expected single-channel 3D 'ZYX'.")
                if candidate is None:
                    candidate = s.shape
                elif s.shape != candidate:
                    raise ValueError(f"Inconsistent image shapes: {s.shape} vs {candidate}. All images must have the same shape.")
        if candidate is None:
            raise ValueError("Could not determine image shape from the provided TIFF files.")
        return candidate
    
    def get_shape(self):
        if self.shape is None:
            self.shape = self._get_shape()
        return self.shape
    
    def get_calibration(self):
        return self.calibration
    
    def get_obj_diam_yx(self):
        return self.overlap[-1]
    
    def get_overlap(self):
        return self.overlap
    
    def get_anisotropy(self):
        return self.anisotropy_factor
    
    def load_by_chunks(self, chunk_size, callback=None):
        if callback is None:
            return
        
        img_shape = self.get_shape()
        if any([a > b for a, b in zip(chunk_size, img_shape)]):
            raise ValueError(f"Chunk size {chunk_size} should not exceed image shape {img_shape}.")
        
        Z, Y, X = img_shape
        step_z  = Z if chunk_size[0] >= Z else max(chunk_size[0] - self.overlap[0], 1)
        step_y  = Y if chunk_size[1] >= Y else max(chunk_size[1] - self.overlap[1], 1)
        step_x  = X if chunk_size[2] >= X else max(chunk_size[2] - self.overlap[2], 1)
        
        arrays = []
        for tiff_path in self.tiff_paths:
            with TiffFile(tiff_path) as tf:
                zobj = tf.aszarr(series=0, level=0, chunkmode=CHUNKMODE.PAGE)
            arrays.append(zarr.open(zobj, mode="r"))
        
        index = 0
        for z0 in range(0, Z, step_z):
            for y0 in range(0, Y, step_y):
                for x0 in range(0, X, step_x):
                    z1 = min(z0 + chunk_size[0], Z)
                    y1 = min(y0 + chunk_size[1], Y)
                    x1 = min(x0 + chunk_size[2], X)
                    chunks = tuple(arr[z0:z1, y0:y1, x0:x1] for arr in arrays)
                    callback(
                        *chunks, 
                        (z0, y0, x0), 
                        (z1, y1, x1), 
                        self.calibration, 
                        index
                    )
                    index += 1

def test_objects_segmentation():
    from chunk_callbacks import (
        get_save_chunk_as, 
        get_cellpose_run,
        get_segment_as_meshes
    )
    from make_meshes import LabelMeshAccumulator

    output_folder = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/tests-callbacks")
    input_folder  = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans/")
    cg = ChunksGenerator(
        [
            input_folder / "eft3RW10035L1_0125071.tif",
            input_folder / "eft3RW10035L1_0125071.tif"
        ],
        (0.122, 0.116, 0.116),
        15
    )
    print(f"Shape            : {cg.get_shape()}")
    print(f"Overlap          : {cg.overlap}")
    print(f"Calibration      : {cg.calibration}")
    print(f"Anisotropy Factor: {cg.anisotropy_factor}")

    lma = LabelMeshAccumulator(cg.get_calibration(), 0)

    sca_fx = get_save_chunk_as(output_folder / "chunks")
    cpr_fx = get_cellpose_run(
        output_folder / "cp_masks", 
        cg.get_obj_diam_yx(), 
        cg.get_anisotropy()
    )

    mesh_folder = output_folder / "mesh"
    sam_fx = get_segment_as_meshes(
        cg.get_overlap(), 
        cg.get_anisotropy(), 
        lma
    )

    cg.load_by_chunks((140, 140, 300), callback=sam_fx)
    lma.save_meshes_as_ply(mesh_folder / "final.ply")

def test_spots_detection():
    from .chunk_callbacks import get_spots_detector
    from .spots_finder import SpotsFinder3D

    path = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/tests-callbacks")
    cg = ChunksGenerator(
        "/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans_spots/eft3RW10035L1_0125071.tif",
        (0.122, 0.116, 0.116),
        15
    )
    sd3d = SpotsFinder3D(cg.get_calibration(), -1550)
    sf_fx = get_spots_detector(sd3d)
    cg.load_by_chunks((140, 140, 300), callback=sf_fx)
    sd3d.save_as(path / "spots.csv")

if __name__ == "__main__":
    test_spots_detection()
