from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import gaussian_laplace

img_folder = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans_spots")
img_name = "eft3RW10035L1_0125071.tif"
img_data = tifffile.imread(img_folder / img_name)
img_data = img_data.astype(np.float32)

log = gaussian_laplace(img_data, sigma=1.5)
spots_mask = log < -1550

res_name = img_name.replace(".tif", "_spots_mask.tif")
# tifffile.imwrite(img_folder / res_name, log)
tifffile.imwrite(img_folder / res_name, spots_mask.astype(np.uint8)*255)