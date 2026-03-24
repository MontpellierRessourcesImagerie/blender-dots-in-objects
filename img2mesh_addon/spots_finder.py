from scipy.ndimage import (
    gaussian_laplace, 
    gaussian_filter,
    prewitt,
    sobel,
    grey_opening
)
from skimage.filters import threshold_otsu
from skimage.feature import (
    peak_local_max, 
    hessian_matrix_det
)
import numpy as np
import csv

def normalize_image(img):
    img = img.astype(np.float32)
    img -= np.min(img)
    if np.max(img) > 0:
        img /= np.max(img)
    return img

def prefilter_none(img, sigmas):
    return normalize_image(img)

def prefilter_laplacian_of_gaussian(img, sigmas):
    filtered = gaussian_laplace(img.astype(np.float32), sigma=sigmas)
    filtered = -1.0 * filtered
    return normalize_image(filtered)

def prefilter_gaussian(img, sigmas):
    filtered = gaussian_filter(img.astype(np.float32), sigma=sigmas)
    return normalize_image(filtered)

def prefilter_opening_by_reconstruction(img, sigmas):
    sigmas = tuple(int(s*2)+1 for s in sigmas)
    opened = grey_opening(img, size=sigmas)
    opened = np.minimum(opened, img)
    return normalize_image(img - opened)

def prefilter_hessian_determinant(img, sigmas):
    hessian_det = -1.0 * hessian_matrix_det(img.astype(np.float32), sigma=sigmas)
    return normalize_image(hessian_det)

def spots_pre_filters():
    return {
        'None'                     : prefilter_none,
        'Laplacian of Gaussian'    : prefilter_laplacian_of_gaussian,
        'Gaussian'                 : prefilter_gaussian,
        'Opening by reconstruction': prefilter_opening_by_reconstruction,
        'Hessian determinant'      : prefilter_hessian_determinant
    }

class SpotsFinder3D(object):
    """
    Class supposed to find spots in 3D images, using a combination of pre-filtering and local maxima detection.
     - The pre-filtering can be used to enhance the contrast of the spots and reduce noise.
     - The local maxima are extracted from the prefiltered image that has been normalized to the [0, 1] range.
     - You can manually provide an intensity threshold or give a negative value to automatically compute it using Otsu's method.
    """
    def __init__(self, sigma=1.0, calib=(1.0, 1.0, 1.0), threshold_abs=-1.0, min_dist=1, prefilter_name='None'):
        """
        Args:
            sigma: Standard deviation for the pre-filtering. Becomes the filter radius for non-linear filters.
            calib: Tuple of (Z, Y, X) voxel sizes in microns. Used to compute the anisotropy factor and to convert spot coordinates to physical units.
            threshold_abs: Absolute intensity threshold for local maxima detection. If negative, it will be computed using Otsu's method on the pre-filtered image.
            min_dist: Minimum distance (in voxels) between detected spots. This is used by the peak_local_max function to avoid detecting multiple nearby peaks.
            prefilter_name: Name of the pre-filter to apply before local maxima detection. Must be one of the keys in the dictionary returned by spots_pre_filters().
        """
        self.sigma         = sigma
        self.calib         = calib
        self.threshold_abs = threshold_abs
        self.spots_list    = np.array([], dtype=np.float32).reshape(0, 3)
        self.min_dist      = min_dist
        self.anisotropy    = calib[0] / calib[1]
        self.prefilter     = spots_pre_filters().get(prefilter_name, prefilter_none)

    def find_spots(self, chunk, origin=(0, 0, 0)):
        filtered = self.prefilter(
            chunk, 
            sigmas=(self.sigma / self.anisotropy, self.sigma, self.sigma)
        )
        
        if self.threshold_abs < 0:
            self.threshold_abs = threshold_otsu(filtered)
        print(f"Using intensity threshold of {self.threshold_abs:.3f} for local maxima detection.")

        coordinates = peak_local_max(
            filtered, 
            min_distance=self.min_dist, 
            threshold_abs=0.9 * self.threshold_abs
        )
        coordinates = coordinates.astype(np.float32) * np.array(self.calib).astype(np.float32) + np.array(origin).astype(np.float32)
        self.spots_list = np.vstack((self.spots_list, coordinates))
    
    def get_all_spots(self):
        return self.spots_list
    
    def save_as(self, csv_path):
        spots = self.get_all_spots()
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Z', 'Y', 'X'])
            for spot in spots:
                writer.writerow(spot)
        print(f"Saved {len(spots)} spots to {csv_path}")

def make_control_image(original_image, spot_coordinates):
    from scipy.ndimage import binary_dilation
    control_image = np.zeros_like(original_image, dtype=np.uint8)
    for z, y, x in spot_coordinates:
        z_idx = int(round(z))
        y_idx = int(round(y))
        x_idx = int(round(x))
        control_image[z_idx, y_idx, x_idx] = 1
    return binary_dilation(control_image)

def example_spots_finder():
    from pathlib import Path
    import tifffile as tiff

    input_folder = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans_spots")
    output_folder = Path("/tmp/2026-03-24/spots-filters")
    img_path = input_folder / "eft3RW10035L1_0125071.tif"
    img = tiff.imread(img_path)

    calibration = (0.122, 0.116, 0.116) # Z, Y, X
    sigma = 2.0

    finder = SpotsFinder3D(
        sigma=sigma,
        calib=calibration,
        threshold_abs=-1.0,
        min_dist=3,
        prefilter_name='Hessian determinant'
    )
    finder.find_spots(img)
    coordinates = finder.get_all_spots()
    print(f"Found {len(coordinates)} spots.")

if __name__ == "__main__":
    example_spots_finder()