# Dots in objects

This Blender addon realizes several tasks:
    - Segment 3D bloboids (nuclei, cells, ...) using CellPose and Python.
    - Import them as mesh objects in Blender, without storing the labeled image.
    - Detect dots (RNA dots, vesicles, lipid droplets, ...) and import them as empty objects.
    - Assign each dot to an object using the hierarchy.
    - Count all combinations of co-occurrences between N channels of dots.

## Install the addon

### A. Download the extra package

- `blender-dots-in-objects` depends on another package currently developed by MRI-CIA. It is still being developed and is not available on PyPI yet.
- You can download it as a ZIP archive from this repo: [github.com/MontpellierRessourcesImagerie/nd_co_occurrences](https://github.com/MontpellierRessourcesImagerie/nd_co_occurrences)
- Simply unzip it and go to the next step.

### B. Install dependencies

- In your Blender install folder, go to `4.5/python/bin`. Note that here, Blender 4.5 is used but if you are using another version of Blender, the "4.5" folder won't exist and a folder named after your version will exist instead.
- Open a terminal in this folder (available in the right-click menu).
- Install the dependencies using the command: `./pip install -U numpy tifffile scikit-image cellpose==3.1.1 zarr meshio scipy`
- Install the package that you just downloaded with the command `./pip install "path/to/the/folder/nd_co_occurrences"`.

### C. Install the add-on

- Start by downloading the ".zip" archive in the [releases list](https://github.com/MontpellierRessourcesImagerie/blender-dots-in-objects/releases) of this repository.
- Do not unzip it and open Blender.
- In the top-bar menu of Blender, go to "Edit" > "Preferences" > "Add-ons".
- In the upper right corner, you should see a "V" button. If you click on it, you will be proposed to install an add-on from the disk.
- Use the file browser to indicate where the ".zip" is.
- In the list of add-ons, you should have "Dots in objects". Make sure that it is activated.
- You should now restart Blender.

## Usage

### General

- Images are expected to be represented by folders with one TIFF per channel.
- If you open Blender and go to the properties panel (that you can open by pressing the "N" key), you should see "Object Segmentation" in the vertical tabs.
- In the `Root folder` field, provide the path to the folder containing the different channels of an image.
- Provide the physical pixels size for all axes.
- If the image is too big for the RAM of your computer, you should choose a chunk size which is the size of the block loaded at a time. If the image is quite small, you can just check "Full image (no chunking)".

### Objects

- In the dropdown menu of the "Objects" section, select the channel corresponding to the object to segment.
- If you have seed objects (example: nuclei for cells) you can provide this second channel as well.
- In the "objects size" you should provide the diameter in pixels of an object in your image on the XY plane.
- The "Min size" field allows you to filter small debris on the fly.
- If you click on "Segment objects", the CellPose runner will start. You can follow its progress in the terminal.
- At the end of the run, the objects should be in a new collection named after the channel.

### Dots

- For each channel of dots that you have (one after the other), select it in the dropdown menu.
- Choose a prefilter to apply and the sigma at which it will be applied.
- Determine a prominence threshold
- Click on "Detect dots"
- Each channel of dots will have its own collection and its own shape.

### Co-occurrences

- Use the picker to select the collection into which the objects are.
- Click the "Assign dots to objects" button to bind every dot of every channel to the object containing it.
- Click on "Remove dots outside" to remove dots that were not assigned to any objects.
- If your objective is just to count dots, you can simply click on "Count dots per object".
- If you activate "Per object?" two dots can be linked only if they belong to the same object. Otherwise, co-occurrences will be processed globally.
