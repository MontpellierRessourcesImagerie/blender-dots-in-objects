from scipy.ndimage import gaussian_laplace
from skimage.measure import label, regionprops
import numpy as np
import csv

class SpotsFinder3D(object):
    def __init__(self, calib=(1.0, 1.0, 1.0), threshold_abs=-1550):
        self.calib = calib
        self.threshold_abs = threshold_abs
        self.spots_list = []
        self.min_dist = 0.0001

    def find_spots(self, chunk, origin=(0, 0, 0)):
        img_data = chunk.astype(np.float32)
        log = gaussian_laplace(img_data, sigma=1.5)
        spots_mask = log < -1550
        labeled = label(spots_mask)
        props = regionprops(labeled)
        local_spots = []
        for p in props:
            zc, yc, xc = p.centroid
            zc = zc * self.calib[0] + origin[0]
            yc = yc * self.calib[1] + origin[1]
            xc = xc * self.calib[2] + origin[2]
            local_spots.append((zc, yc, xc))
        self.spots_list.extend(local_spots)

    def filter_by_distance(self):
        if self.spots_list is None or len(self.spots_list) == 0:
            return []
        spots = np.array(self.spots_list)
        filtered = []
        for i, spot in enumerate(spots):
            dists = np.linalg.norm(spots - spot, axis=1)
            if np.sum(dists < self.min_dist) == 1:
                filtered.append(spot)
        self.spots_list = filtered
        return filtered
    
    def get_all_spots(self):
        return self.filter_by_distance()
    
    def save_as(self, csv_path):
        spots = self.get_all_spots()
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Z', 'Y', 'X'])
            for spot in spots:
                writer.writerow(spot)
        print(f"Saved {len(spots)} spots to {csv_path}")