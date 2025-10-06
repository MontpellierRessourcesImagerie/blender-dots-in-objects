import numpy as np
import tifffile
from scipy.signal import fftconvolve
from pathlib import Path

# Chargement de la PSF
psf = tifffile.imread("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/psf.tif").astype(np.float32)
psf /= psf.sum()  # Normalisation de l'énergie

masks_folder = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans_masks")
spots_folder = Path("/home/clement/Documents/projects/mifobio-2025/flash-tuto/data/c_elegans_spots")
image_name = "eft3RW10035L1_0125071.tif"

# Dimensions du volume de destination
image_path = masks_folder / image_name
spots_path = spots_folder / image_name
labels_data = tifffile.imread(image_path)
shape = labels_data.shape

# Création du volume
volume = np.zeros(shape, dtype=np.uint16)
# Transform labels into masks
binary_mask = (labels_data > 0).astype(np.uint16)

# Placement de 16 points aléatoires
np.random.seed(42)  # pour reproductibilité
for _ in range(4000):
    z = np.random.randint(0, shape[0])
    y = np.random.randint(0, shape[1])
    x = np.random.randint(0, shape[2])
    value = np.random.randint(32000, 35001)
    volume[z, y, x] = value

volume *= binary_mask  # Appliquer le masque
tifffile.imwrite(spots_folder / f"test-spots.tif", volume)

# Convolution 3D memory-effective avec fftconvolve (utilise le moins de RAM possible)
convolved = fftconvolve(volume.astype(np.float32), psf, mode='same')

target_max = 20000.0
min_val = convolved.min()
max_val = convolved.max()
convolved = (convolved - min_val) / (max_val - min_val) * target_max

# Clip et cast en uint16
convolved = np.clip(convolved, 0, 65535).astype(np.uint16)

# Slice du milieu
# middle_slice = convolved[shape[0] // 2]

# Sauvegarde en TIFF
tifffile.imwrite(spots_path, convolved)

print(f"Fichier {spots_path.name} sauvegardé avec la slice du milieu convoluée.")
