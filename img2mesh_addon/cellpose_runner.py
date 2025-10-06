import numpy as np
from cellpose import models
from cellpose.io import logger_setup

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
            raise ValueError("The model has not been created.")
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
    
def test_run_cellpose():
    import tifffile as tiff
    from pathlib import Path
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

if __name__ == "__main__":
    test_run_cellpose()