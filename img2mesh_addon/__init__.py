bl_info = {
    "name": "Nuclei Segmenter",
    "author": "Clement H. Benedetti (MRI-CIA)",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "3D Viewport > N panel > Nuclei Seg",
    "description": "Segment nuclei from a TIF stack and show the result as a mesh",
    "category": "Import-Export",
}

import bpy
from bpy.props import (
    StringProperty, IntProperty, FloatVectorProperty, IntVectorProperty, PointerProperty
)
from bpy.types import PropertyGroup, Operator, Panel

from pathlib import Path

from .blender_callbacks import segment_and_import, detect_and_import
from .count_spots import spots_per_nucleus

class NUCSEG_Props(PropertyGroup):
    spots_path: StringProperty(
        name="Spots image",
        description="Path to a 3D spots single-channel TIFF (ZYX)",
        subtype='FILE_PATH',
        default=""
    )
    spots_threshold: IntProperty(
        name="Spots threshold",
        description="Intensity threshold for spot detection",
        min=-3000, max=0, default=-1550
    )
    nuclei_path: StringProperty(
        name="Nuclei image",
        description="Path to a 3D nuclei single-channel TIFF (ZYX)",
        subtype='FILE_PATH',
        default=""
    )
    obj_size: IntProperty(
        name="Object size",
        description="Typical nucleus size in pixels (diameter; used for prefiltering/scaling, optional)",
        min=1, default=15
    )
    calib: FloatVectorProperty(
        name="Calibration (Z,Y,X)",
        description="Voxel size / spacing (X, Y, Z)",
        size=3, default=(0.116, 0.116, 0.122), subtype='XYZ',
        precision=4
    )
    patch: IntVectorProperty(
        name="Patch size (X, Y, Z)",
        description="Read/processing patch size; not required for simple Otsu, reserved for large data",
        size=3, default=(300, 140, 140), min=1, subtype='XYZ'
    )

# ---- The worker operator ----

class NUCSEG_OT_Launch(Operator):
    bl_idname = "nucseg.launch"
    bl_label = "Launch Nuclei Segmentation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.nucseg_props
        path = Path(bpy.path.abspath(props.nuclei_path) if props.nuclei_path else "")
        print(path)
        if not path.is_file():
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        print("Launching the process...")
        calib = (props.calib[2], props.calib[1], props.calib[0])  # Z, Y, X
        obj_size_yx = props.obj_size
        chunk_size = (props.patch[2], props.patch[1], props.patch[0])  # Z, Y, X
        print(f"Image path: {path.name}")
        print(f"Calib: {calib}")
        print(f"Obj size: {obj_size_yx}")
        print(f"Chunks: {chunk_size}")
        segment_and_import(
            img_path=path,
            calib=calib,
            obj_size_yx=obj_size_yx,
            chunk_size=chunk_size
        )
        return {'FINISHED'}
    
class SPOTSDEC_OT_Launch(Operator):
    bl_idname = "spotsdec.launch"
    bl_label = "Launch Spots Detection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.nucseg_props
        path = Path(bpy.path.abspath(props.spots_path) if props.spots_path else "")
        print(path)
        if not path.is_file():
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        print("Launching the process...")
        calib = (props.calib[2], props.calib[1], props.calib[0])  # Z, Y, X
        chunk_size = (props.patch[2], props.patch[1], props.patch[0])  # Z, Y, X
        spots_threshold = props.spots_threshold
        print(f"Image path: {path.name}")
        print(f"Calib: {calib}")
        print(f"Chunks: {chunk_size}")
        detect_and_import(
            img_path=path,
            calib=calib,
            thr=spots_threshold,
            chunk_size=chunk_size
        )
        return {'FINISHED'}
    
class COUNTSPOTS_OT_Launch(Operator):
    bl_idname = "count_spots.launch"
    bl_label = "Spots per nucleus"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        spots_per_nucleus()
        return {'FINISHED'}

# ---- UI Panel (N-panel, 3D View) ----

class NUCSEG_PT_Panel(Panel):
    bl_idname = "NUCSEG_PT_panel"
    bl_label = "Nuclei Seg"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Nuclei Seg"

    def draw(self, context):
        layout = self.layout
        props = context.scene.nucseg_props

        col = layout.column(align=True)
        col.label(text="Data")
        col.prop(props, "nuclei_path", text="Nuclei")
        col.prop(props, "spots_path", text="Spots")

        col.separator()
        col.prop(props, "spots_threshold", text="Spots Thr.")

        col.separator()
        col.label(text="Parameters")
        col.prop(props, "obj_size", text="Object size")
        row = col.row(align=True)
        row.prop(props, "calib", text="Calibration")
        row = col.row(align=True)
        row.prop(props, "patch", text="Patches")

        col.separator()
        col.operator("spotsdec.launch", text="Detect spots", icon='PLAY')
        col.operator("nucseg.launch", text="Segment nuclei", icon='PLAY')
        col.operator("count_spots.launch", text="Count spots/nucleus", icon='SORTSIZE')

# ---- register ----

classes = (
    NUCSEG_Props,
    NUCSEG_OT_Launch,
    SPOTSDEC_OT_Launch,
    COUNTSPOTS_OT_Launch,
    NUCSEG_PT_Panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.nucseg_props = PointerProperty(type=NUCSEG_Props)

def unregister():
    del bpy.types.Scene.nucseg_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
