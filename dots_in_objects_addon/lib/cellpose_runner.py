import numpy as np
from cellpose import models
from cellpose.io import logger_setup
import torch

class CellPoseRunner3D(object):
    """
    Runs CellPose 3D segmentation on 3D image chunks.
    Single or dual channel images can be processed, depending on the model and the provided images.
    The model is created on demand and can be reused for multiple runs.
    """

    def __init__(self, model_name, gpu, obj_size, ani_factor=1.0, min_size=15):
        """
        Args:
            model_name: Name of the pre-trained CellPose model to use (e.g., "cyto3", "tissuenet_cp3", ...).
            gpu: Whether to use GPU for inference (True/False).
            obj_size: Median diameter of objects in voxels on the XY plane (used by CellPose).
            ani_factor: Anisotropy factor for 3D images (Z/XY, very likely > 1.0).
            min_size: Minimum size of objects to keep (in voxels).

        """
        self.model_name = model_name
        self.gpu        = gpu and torch.cuda.is_available()
        self.obj_size   = obj_size
        self.ani_factor = ani_factor
        self.min_size   = min_size
        self.model      = None
        logger_setup()

    def instanciate_model(self):
        if self.model is not None:
            return
        self.model = models.CellposeModel(
            gpu=self.gpu,
            model_type="cyto3",
            pretrained_model=self.model_name
        )

    def run(self, images):
        image_main = images[0]
        image_secondary = images[1] if len(images) > 1 else None
        self.instanciate_model()
        if self.model is None:
            raise ValueError("The model has not been created.")
        if image_secondary is None:
            img = np.asarray(image_main)[np.newaxis, :]
            channels = [0, 0]
        else:
            img = np.concatenate([
                np.asarray(image_main)[np.newaxis, :],
                np.asarray(image_secondary)[np.newaxis, :]
            ], axis=0)
            channels = [1, 2]
        labels, _, _ = self.model.eval(
            img,
            channels=channels,
            diameter=self.obj_size,
            anisotropy=self.ani_factor,
            do_3D=True,
            z_axis=1,
            channel_axis=0,
            min_size=self.min_size,
            # flow3D_smooth=1
        )
        return labels
    
def example_run_cellpose(setup=1):
    import tifffile as tiff
    from pathlib import Path

    output_folder = Path("/tmp/2026-03-24/cellpose-runner")

    if setup == 1: # single channel (nuclei)
        folder   = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans")
        img_main_path = folder / "eft3RW10035L1_0125071-crop.tif"
        img_sec_path = None
        obj_size = 15
        ani_factor = 0.122/0.116
        min_obj = 250
    elif setup == 2: # dual channel (membranes + nuclei)
        folder = Path("/media/clement/3b801c96-393a-4b2e-be1e-9cabfbb10740/2025-04-22-lglepin")
        img_main_path = folder / "membranes.tif"
        img_sec_path  = folder / "nuclei.tif"
        obj_size = 110
        ani_factor = 1.0/0.2167
        min_obj = 50000
    else:
        raise ValueError(f"Unknown setup {setup}")

    data_main = tiff.imread(img_main_path)
    data_sec  = tiff.imread(img_sec_path) if img_sec_path is not None else None

    cp3d = CellPoseRunner3D(
        "cyto3",
        gpu=True,
        obj_size=obj_size,
        ani_factor=ani_factor,
        min_size=min_obj
    )
    lbls = cp3d.run([data_main, data_sec] if data_sec is not None else [data_main])

    if lbls is not None:
        out_path = output_folder / f"cp_result-{setup}.tif"
        tiff.imwrite(out_path, lbls)
        print(f"Saved labels to {out_path}")

if __name__ == "__main__":
    example_run_cellpose(1)
    example_run_cellpose(2)